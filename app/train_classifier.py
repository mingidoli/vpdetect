# app/train_classifier.py
import os
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DATA  = "data/train.csv"  # 이미 준비된 train.csv 사용
DEFAULT_OUT   = "models/text_clf.joblib"

def resolve_device(arg_device: str) -> str:
    """
    arg_device: 'auto' | 'cpu' | 'cuda'
    - auto: cuda 가능하면 cuda, 아니면 cpu
    """
    if arg_device in ("cpu", "cuda"):
        return arg_device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA, help="CSV path with columns: text,label")
    ap.add_argument("--out",  default=DEFAULT_OUT,  help="Output classifier path (*.joblib)")
    ap.add_argument("--embed", default=DEFAULT_MODEL, help="SentenceTransformer model name")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"], help="Device for embeddings")
    ap.add_argument("--batch", type=int, default=64, help="Embedding batch size")
    args = ap.parse_args()

    device = resolve_device(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    assert os.path.exists(args.data), f"data not found: {args.data}"

    # CSV 로드 (UTF-8/UTF-8-SIG 자동 인식)
    df = pd.read_csv(args.data)
    if not {"text","label"}.issubset(df.columns):
        raise ValueError("CSV must contain columns: text,label")

    df = df.dropna(subset=["text","label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    X = df["text"].tolist()
    y = df["label"].tolist()

    # 임베딩 (GPU/CPU 선택)
    print(f"[embed] model: {args.embed} | device: {device} | batch: {args.batch}")
    embed = SentenceTransformer(args.embed, device=device)
    embeddings = embed.encode(X, show_progress_bar=True, batch_size=args.batch)

    # 학습/검증 분리
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, y, test_size=0.15, random_state=42, stratify=y
    )

    # 분류기 (CPU 기반)
    clf = LogisticRegression(
        max_iter=2000,
        multi_class="multinomial",
        solver="saga",
        class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    # 평가
    preds = clf.predict(X_val)
    print("\n[report]\n", classification_report(y_val, preds, digits=4))
    print("\n[confusion]\n", confusion_matrix(y_val, preds))

    # 저장
    joblib.dump({"embed_model_name": args.embed, "clf": clf}, args.out)
    print(f"\n[saved] {args.out}")

if __name__ == "__main__":
    main()
