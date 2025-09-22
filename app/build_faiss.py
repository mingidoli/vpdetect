# app/build_faiss.py
# corpus.jsonl -> FAISS 인덱스 재생성 (임베딩은 CPU에서 수행: 안정/호환성 우선)

import os, json, numpy as np, faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("RAG_MODEL_NAME", "BAAI/bge-m3")
CORPUS_PATH = os.getenv("RAG_CORPUS_PATH", "models/corpus.jsonl")
INDEX_PATH  = os.getenv("RAG_INDEX_PATH", "models/faiss.index")

def load_corpus(path: str):
    """JSON Lines 파일을 BOM 포함/미포함 모두 안전하게 읽기"""
    docs, bad, total = [], 0, 0
    with open(path, "r", encoding="utf-8-sig") as f:  # utf-8-sig: BOM 자동 제거
        for line in f:
            total += 1
            s = line.strip()
            if not s:
                bad += 1
                continue
            try:
                j = json.loads(s)
                if "text" in j:
                    docs.append(j)
                else:
                    bad += 1
            except Exception:
                bad += 1
                continue
    print(f"[BUILD] corpus lines={total}, loaded={len(docs)}, skipped={bad}")
    if not docs:
        raise ValueError("corpus.jsonl is empty or invalid.")
    return docs

def main():
    # 임베딩은 CPU 고정 (GPU 커널/아키텍처 이슈 회피)
    device = "cpu"
    print(f"[BUILD] device={device}, model={MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    docs  = load_corpus(CORPUS_PATH)
    texts = [d["text"] for d in docs]

    # 문장 임베딩 (코사인 유사도용 정규화 ON)
    embs = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,           # CPU 배치
        show_progress_bar=True,
        device=device,           # 명시
    ).astype("float32")

    # FAISS (Inner Product) 인덱스 생성
    dim   = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    faiss.write_index(index, INDEX_PATH)
    print(f"[BUILD] faiss index saved -> {INDEX_PATH} (N={embs.shape[0]})")

if __name__ == "__main__":
    main()
