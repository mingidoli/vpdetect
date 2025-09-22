# app/config.py
import os
from pathlib import Path

# 프로젝트 루트 기준 경로
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

# ✅ RAG/유사도용 코퍼스: models/ 폴더 고정
CORPUS_PATH = os.getenv(
    "CORPUS_PATH",
    str(BASE_DIR / "corpus.jsonl")    # ✅ app/corpus.jsonl 가리키도록
)

# ✅ FAISS 인덱스: models/ 폴더 고정
INDEX_PATH = os.getenv(
    "INDEX_PATH",
    str(ROOT_DIR / "models" / "faiss.index")
)

# ✅ 정규식/키워드 레파토리: app/repertoire/ 폴더
REPERTOIRE_PATH = os.getenv(
    "REPERTOIRE_PATH",
    str(BASE_DIR / "repertoire" / "repertoire.jsonl")
)

# ✅ 가중치 파일: app/ 폴더 (기존 유지)
WEIGHTS_PATH = os.getenv(
    "WEIGHTS_PATH",
    str(BASE_DIR / "weights.yaml")
)
