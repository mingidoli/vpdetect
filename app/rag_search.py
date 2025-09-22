# app/rag_search.py
# SentenceTransformer + FAISS 기반 RAG 검색 유틸
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer

from .services.loader import load_corpus
from . import config

# ------------------------------
# 전역 상태
# ------------------------------
_MODEL: Optional[SentenceTransformer] = None
_INDEX: Optional[faiss.Index] = None
_DOCS: Optional[List[Dict[str, Any]]] = None

_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
_MODEL_NAME = os.getenv("RAG_MODEL_NAME", "BAAI/bge-m3")

# 항상 Path 객체로 관리 (문자열 금지)
_INDEX_PATH: Path = Path(os.getenv("RAG_INDEX_PATH", str(config.INDEX_PATH))).resolve()
_CORPUS_PATH: Path = Path(os.getenv("RAG_CORPUS_PATH", str(config.CORPUS_PATH))).resolve()

# ------------------------------
# 내부 유틸
# ------------------------------
def _force_cpu_reload():
    """CUDA 오류 시 CPU로 전환 후 임베딩 모델 재로드."""
    global _DEVICE, _MODEL
    _DEVICE = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _MODEL = SentenceTransformer(_MODEL_NAME, device=_DEVICE)
    # 워밍업
    _MODEL.encode(["warmup"], normalize_embeddings=True, convert_to_numpy=True,
                  batch_size=1, show_progress_bar=False, device=_DEVICE)
    print("[RAG] switched to CPU and reloaded embed model")

def _ensure_model():
    """임베딩 모델 로드 (CUDA → 실패 시 CPU 폴백)."""
    global _MODEL
    if _MODEL is not None:
        return
    try:
        _MODEL = SentenceTransformer(_MODEL_NAME, device=_DEVICE)
        print(f"[RAG] embed model loaded: {_MODEL_NAME} on {_DEVICE}")
        try:
            _MODEL.encode(["warmup"], normalize_embeddings=True, convert_to_numpy=True,
                          batch_size=1, show_progress_bar=False, device=_DEVICE)
            print("[RAG] embed warmup done")
        except Exception as e:
            print("[RAG] embed warmup skipped:", repr(e))
    except RuntimeError as e:
        msg = str(e).lower()
        if "cuda" in msg:
            print("[RAG] CUDA load failed, falling back to CPU:", repr(e))
            _force_cpu_reload()
        else:
            raise

def _ensure_index():
    """
    FAISS 인덱스와 코퍼스를 메모리에 로드.
    - 인덱스는 config.INDEX_PATH (models/faiss.index)
    - 코퍼스는 config.CORPUS_PATH (models/corpus.jsonl)
    """
    global _INDEX, _DOCS

    # 인덱스
    if _INDEX is None:
        if not _INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found: {_INDEX_PATH}")
        _INDEX = faiss.read_index(str(_INDEX_PATH))
        print(f"[RAG] faiss index loaded: {_INDEX_PATH}")

    # 코퍼스
    if _DOCS is None:
        docs = load_corpus(str(_CORPUS_PATH))  # 안전 로더(utf-8/utf-8-sig/cp949)
        # 필드 보정
        norm_docs: List[Dict[str, Any]] = []
        for i, d in enumerate(docs):
            if not isinstance(d, dict):
                continue
            text = d.get("text") or d.get("content") or ""
            norm_docs.append({"id": d.get("id", i), "text": text})
        _DOCS = norm_docs
        print(f"[RAG] corpus loaded via loader (N={len(_DOCS)})")

def load_index() -> bool:
    """앱 시작 시 1회 호출."""
    _ensure_model()
    _ensure_index()
    return True

# ------------------------------
# 임베딩
# ------------------------------
def _embed(texts):
    _ensure_model()

    if isinstance(texts, str):
        texts = [texts]

    def _do(device: str):
        vecs = _MODEL.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=64 if device == "cuda" else 16,
            device=device,
        )
        # (D,) → (1,D), float32 & contiguous
        vecs = np.asarray(vecs, dtype="float32")
        if vecs.ndim == 1:
            vecs = vecs[None, :]
        return np.ascontiguousarray(vecs, dtype="float32")

    try:
        return _do(_DEVICE)
    except RuntimeError as e:
        msg = str(e).lower()
        if "cuda" in msg:
            print("[RAG] CUDA encode failed, fallback to CPU:", repr(e))
            _force_cpu_reload()
            return _do("cpu")
        raise

# ------------------------------
# 검색
# ------------------------------
def search(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    질의문으로 상위 k개 문서 검색.
    - 인덱스: L2/Inner Product 기반 (코사인 유사도 가정)
    - 반환 score는 -1..1 (1에 가까울수록 유사)
    """
    if not query or not query.strip():
        return []

    _ensure_index()  # 인덱스/코퍼스 보장

    if _INDEX is None or not len(_DOCS):
        print("[RAG][WARN] index/docs not ready; returning empty hits")
        return []

    qv = _embed(query)                  # (1, D)
    D, I = _INDEX.search(qv, k)         # D: (1,k), I: (1,k)

    hits: List[Dict[str, Any]] = []
    k_found = min(k, len(I[0]))
    for rank in range(k_found):
        idx = int(I[0][rank])
        if idx < 0 or idx >= len(_DOCS):
            continue
        doc = _DOCS[idx]
        hits.append({
            "score": float(D[0][rank]),
            "id": doc.get("id", idx),
            "text": doc.get("text", ""),
        })
    if not hits:
        print("[RAG][DEBUG] search produced empty hits. "
              f"k={k}, topI={I.tolist()}, docsN={len(_DOCS)}")
    return hits
