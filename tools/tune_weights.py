# tools/tune_weights.py
# - app/rag_search: load_index(), search() 함수 사용
# - app/risk_scorer: RiskScorer(score(FeatureScores)) API에 맞춰 호출
# - 탐색: 가중치(weights) + 판정 임계(decision_threshold) 그리드
# - 리콜 우선(Recall>=0.9) → F1 최대

import os, sys, re, json, argparse, itertools
# ❌ GPU 막는 코드 제거: 필요 시만 환경변수 FORCE_CPU=1 로 강제하세요.
# if os.environ.get("FORCE_CPU") == "1":
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""

from pathlib import Path

BASE = Path(__file__).resolve().parents[1]   # C:\vpdetect
sys.path.append(str(BASE))

import pandas as pd
import yaml

# 앱 모듈
from app.rag_search import load_index, search
from app.risk_scorer import RiskScorer, FeatureScores

# ------------------------------
# 패턴(간단 규칙)
# ------------------------------
KW_PATTERNS = [
    r"인증번호", r"OTP", r"안전계좌", r"이체", r"송금", r"수수료", r"대출", r"검찰청", r"수사관",
    r"CVC", r"보안카드", r"본인\s*인증", r"명의도용", r"벌금", r"압류",
]
BANK_PATTERNS = [r"농협", r"국민", r"우리", r"신한", r"하나", r"카카오", r"토스", r"기업", r"수협"]

# 금액 추출(만원 단위로 정규화: 5만원 -> 5, 300000원 -> 30)
RE_MANWON = re.compile(r"(\d+(?:[.,]\d+)?)\s*만원")
RE_WON    = re.compile(r"(\d{3,})\s*원")  # 3자리 이상 원 금액

def extract_amount_in_manwon(text: str) -> float | None:
    t = str(text)
    m = RE_MANWON.search(t)
    if m:
        return float(m.group(1).replace(",", ""))
    m = RE_WON.search(t)
    if m:
        return float(m.group(1).replace(",", "")) / 10000.0
    return None

def count_keywords(text: str) -> int:
    return sum(1 for p in KW_PATTERNS if re.search(p, text))

def has_bank_entity(text: str) -> bool:
    return any(re.search(p, text) for p in BANK_PATTERNS)

# ------------------------------
# RAG 어댑터(유사도만 사용)
# ------------------------------
class SimpleRag:
    def __init__(self):
        load_index()  # 인덱스/코퍼스 로드

    @staticmethod
    def _score_to_01(cos_sim: float) -> float:
        # rag_search.search() 가 코사인(-1..1)이라면 0..1로 보정
        try:
            x = float(cos_sim)
        except Exception:
            return 0.0
        return max(0.0, min(1.0, (x + 1.0) / 2.0))

    def sim01(self, text: str, k: int = 3) -> float:
        hits = search(text, k=k)  # [{score, id, text}, ...]
        if not hits:
            return 0.0
        sims = [self._score_to_01(h["score"]) for h in hits]
        return sum(sims) / len(sims)

# ------------------------------
# 유틸
# ------------------------------
def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

def evaluate(df: pd.DataFrame, scorer: RiskScorer, rag: SimpleRag, decision_thr: float) -> dict:
    y_true, y_pred = [], []
    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        label = int(row["label"])
        channel = (str(row.get("channel", "")).strip() or "other").lower()

        # parts → FeatureScores 로 매핑
        sim = rag.sim01(text, k=3)
        kw_hits = count_keywords(text)
        ent = has_bank_entity(text)
        amt = extract_amount_in_manwon(text)

        fs = FeatureScores(
            sim=sim,
            keyword_hits=kw_hits,
            entity_present=ent,
            amount_value=amt,
            channel=channel
        )
        out = scorer.score(fs)                 # {'risk', 'level', 'parts'} 반환
        risk = float(out["risk"])
        pred = 1 if risk >= decision_thr else 0

        y_true.append(label)
        y_pred.append(pred)

    tp = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==1)
    fp = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==1)
    tn = sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==0)
    fn = sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==0)
    precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall    = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    return dict(tp=tp, fp=fp, tn=tn, fn=fn, precision=precision, recall=recall, f1=f1)

# ------------------------------
# 메인: 가중치/판정임계 그리드 탐색
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(BASE/"data/labeled_samples.csv"))
    ap.add_argument("--weights", default=str(BASE/"app/weights.yaml"))
    ap.add_argument("--save-best", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    weights_path = Path(args.weights)

    # RiskScorer는 cfg_path에서 현재 설정을 읽음
    scorer = RiskScorer(cfg_path=str(weights_path))
    rag = SimpleRag()

    # 그리드(필요시 확장)
    W_GRID = {
        "sim":     [0.4, 0.5, 0.6],
        "keyword": [0.1, 0.2, 0.3],
        "entity":  [0.05, 0.1, 0.2],
        "amount":  [0.05, 0.1, 0.2],
        "channel": [0.0, 0.1, 0.2],
    }
    DECISION_THR = [0.45, 0.50, 0.55, 0.60]  # risk ≥ thr → Positive(피싱)

    best = None
    for wsim in W_GRID["sim"]:
        for wkw in W_GRID["keyword"]:
            for went in W_GRID["entity"]:
                for wamt in W_GRID["amount"]:
                    for wch in W_GRID["channel"]:
                        s = wsim + wkw + went + wamt + wch
                        w = {
                            "sim": wsim/s, "keyword": wkw/s,
                            "entity": went/s, "amount": wamt/s, "channel": wch/s
                        }
                        # 현재 scorer에 가중치 주입(메모리상 override)
                        if hasattr(scorer, "w"):
                            scorer.w = w

                        for thr in DECISION_THR:
                            m = evaluate(df, scorer, rag, thr)
                            # 우선순위: Recall>=0.9 → 그 안에서 F1 최고
                            key = (1 if m["recall"] >= 0.9 else 0, m["f1"])
                            rec = {
                                "key": key, "metrics": m, "weights": w, "decision_threshold": thr
                            }
                            if (best is None) or (rec["key"] > best["key"]):
                                best = rec

    print("=== BEST CONFIG ===")
    out = {
        "f1": best["metrics"]["f1"],
        "recall": best["metrics"]["recall"],
        "precision": best["metrics"]["precision"],
        "conf": {k:int(best["metrics"][k]) for k in ("tp","fp","tn","fn")},
        "weights": best["weights"],
        "decision_threshold": best["decision_threshold"]
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.save_best:
        # app/weights.yaml 의 risk.weights 만 갱신 (features/level_bins는 유지)
        cfg = load_yaml(weights_path)
        cfg.setdefault("risk", {})
        cfg["risk"]["weights"] = best["weights"]
        save_yaml(weights_path, cfg)
        print(f"\nSaved best weights into: {weights_path}")

if __name__ == "__main__":
    main()
