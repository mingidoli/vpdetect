# app/classifier_infer.py
import argparse
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_FILE = "models/text_clf.joblib"

def resolve_device(arg_device: str) -> str:
    if arg_device in ("cpu", "cuda"):
        return arg_device
    try:
        import torch  # noqa: F401
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

class TextClassifier:
    def __init__(self, model_file: str = MODEL_FILE, device: str = "auto"):
        meta = joblib.load(model_file)
        self.embed_name = meta["embed_model_name"]
        self.clf = meta["clf"]
        dev = resolve_device(device)
        self.embed = SentenceTransformer(self.embed_name, device=dev)

    def predict(self, text: str):
        emb = self.embed.encode([text])
        proba = self.clf.predict_proba(emb)[0]
        classes = [str(c) for c in self.clf.classes_]
        best_idx = int(np.argmax(proba))
        prob_all = {classes[i]: float(proba[i]) for i in range(len(classes))}
        return {
            "label": classes[best_idx],
            "prob": float(proba[best_idx]),
            "prob_all": prob_all
        }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    c = TextClassifier(device=args.device)
    demo = "검찰청 사이버수사대입니다. 귀하의 계좌가 범죄와 연루되었습니다. 안전계좌로 이체하십시오."
    print(c.predict(demo))
