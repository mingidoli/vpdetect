# app/scoring.py
import re, yaml

conf = yaml.safe_load(open("configs/rules.yaml", "r", encoding="utf-8"))

def compute_risk(text: str) -> float:
    score = 0.0
    for kw in conf["keywords"]:
        if re.search(kw["pattern"], text):
            score += kw["weight"]
    return min(1.0, score)
