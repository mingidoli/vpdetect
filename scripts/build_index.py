# scripts/build_index.py
import os, glob, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

KB_GLOB = "data/kb/*.md"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

# 1) 문서 읽기 (파일 하나 = 하나의 청크로 간단 처리)
docs = []
for i, path in enumerate(sorted(glob.glob(KB_GLOB))):
    txt = open(path, "r", encoding="utf-8").read().strip()
    if txt:
        docs.append({"id": i, "text": txt})

with open(os.path.join(OUT_DIR, "corpus.jsonl"), "w", encoding="utf-8") as w:
    for d in docs:
        w.write(json.dumps(d, ensure_ascii=False) + "\n")

# 2) 임베딩
model = SentenceTransformer("BAAI/bge-m3")
emb = model.encode([d["text"] for d in docs], normalize_embeddings=True)
emb = np.array(emb, dtype="float32")

# 3) FAISS IP(코사인 유사도와 동일: 벡터 정규화했기 때문)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, os.path.join(OUT_DIR, "faiss.index"))

print(f"built index with {len(docs)} docs → models/faiss.index")
