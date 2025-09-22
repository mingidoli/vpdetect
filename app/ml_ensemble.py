# app/ml_ensemble.py
from typing import Dict, Any

def label_to_riskprob(lbl: str) -> float:
    # ML 라벨을 "최소 위험 확률"로 매핑(보수적)
    table = {
        "SAFE": 0.05,
        "WARN": 0.45,
        "HIGH": 0.75,
        "CRITICAL": 0.95
    }
    return table.get(lbl, 0.45)

def aggregate_final(parts: Dict[str, float], weights: Dict[str, Any]) -> float:
    w = (weights or {}).get("weights", {})
    s = 0.0
    for k, v in parts.items():
        s += float(w.get(k, 0.0)) * float(v)
    return max(0.0, min(1.0, s))

def level_from_score(score: float, weights: Dict[str, Any]) -> str:
    th = (weights or {}).get("thresholds", {})
    if score >= th.get("CRITICAL", 0.90): return "CRITICAL"
    if score >= th.get("HIGH", 0.70):     return "HIGH"
    if score >= th.get("WARN", 0.45):     return "WARN"
    return "SAFE"
