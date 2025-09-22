# app/main.py
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────────────
# 외부/내부 모듈 로딩 (없어도 서버가 최소 구동되도록 안전장치 포함)
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# config (경로 참조)
try:
    from . import config  # type: ignore
except Exception:
    class _Cfg:
        CORPUS_PATH = BASE_DIR / "corpus.jsonl"
        INDEX_PATH = BASE_DIR / "models" / "faiss.index"
        REPERTOIRE_PATH = BASE_DIR / "repertoire.jsonl"
        WEIGHTS_PATH = BASE_DIR / "weights.yaml"
    config = _Cfg()  # type: ignore

# RAG
try:
    from . import rag_search  # type: ignore
except Exception:
    class _DummyRag:
        def load_index(self) -> None:
            print("[RAG] dummy: load_index skipped")
        def search(self, text: str, k: int = 3) -> List[Dict[str, Any]]:
            return []
    rag_search = _DummyRag()  # type: ignore

# ASR
try:
    from .asr_service import load_asr, transcribe_file  # type: ignore
except Exception:
    def load_asr(name: str = "small") -> None:  # type: ignore
        print("[ASR] dummy: load_asr skipped")
    def transcribe_file(path: str) -> str:  # type: ignore
        return ""

# Risk scorer
try:
    from .risk_scorer import RiskScorer, FeatureScores  # type: ignore
except Exception:
    # 최소 동작용 더미
    class FeatureScores(BaseModel):  # type: ignore
        text: str = ""
        sim: float = 0.0
        keyword_hits: int = 0
        entity_present: bool = False
        amount_value: Optional[float] = None
        channel: str = "sms"

    class RiskScorer:  # type: ignore
        def __init__(self, path: str) -> None:
            pass
        def score(self, fs: FeatureScores) -> Dict[str, Any]:
            # 규칙 점수 0, 구조만 반환
            return {"parts": {"sim": fs.sim, "keyword": min(1.0, fs.keyword_hits / 10.0),
                              "entity": 1.0 if fs.entity_present else 0.0,
                              "amount": (fs.amount_value or 0.0) / 100.0,
                              "channel": 0.05 if fs.channel == "sms" else 0.0},
                    "evidence": []}

# ML classifier
try:
    from .classifier_infer import TextClassifier  # type: ignore
except Exception:
    class TextClassifier:  # type: ignore
        def __init__(self, device: str = "auto") -> None:
            pass
        def predict(self, text: str) -> Dict[str, Any]:
            # 간단 더미: 길이에 따라 확률 차등
            p = max(0.05, min(0.95, len(text) / 120.0))
            return {"label": "WARN" if p < 0.6 else "HIGH", "prob": p}

# ──────────────────────────────────────────────────────────────────────
# FastAPI 앱
# ──────────────────────────────────────────────────────────────────────
app = FastAPI(default_response_class=ORJSONResponse)

# ──────────────────────────────────────────────────────────────────────
# 유틸 / 초기화
# ──────────────────────────────────────────────────────────────────────
def ensure_utf8_bytes(b: bytes) -> bytes:
    """요청 바이트를 UTF-8로 강제. UTF-16/CP949 등 폴백 지원."""
    try:
        b.decode("utf-8")
        return b
    except UnicodeDecodeError:
        pass
    for enc in ("utf-16", "utf-16le", "utf-16be", "cp949", "euc-kr"):
        try:
            return b.decode(enc).encode("utf-8")
        except UnicodeDecodeError:
            continue
    return b

def load_weights_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# ✨ 추가: 유니코드 정규화(한글 호환형/제로폭 문자 제거)
import unicodedata
def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    return s.replace("\u200b", "").replace("\ufeff", "")

WEIGHTS_PATH: Path = getattr(config, "WEIGHTS_PATH", BASE_DIR / "weights.yaml")
weights_yaml: Dict[str, Any] = load_weights_yaml(Path(WEIGHTS_PATH))

# 스코어러/분류기
scorer = RiskScorer(str(WEIGHTS_PATH))
clf = TextClassifier(device="auto")

# ──────────────────────────────────────────────────────────────────────
# 정규식 레퍼토리(패턴) 프리컴파일 캐시
# ──────────────────────────────────────────────────────────────────────
def _load_regex_catalog() -> List[Dict[str, Any]]:
    """
    repertoire.jsonl에서 pattern/regex/re 또는 patterns(list) 를 모두 지원해
    IGNORECASE|DOTALL로 컴파일하여 캐싱.
    """
    import codecs
    rep_path = getattr(config, "REPERTOIRE_PATH", BASE_DIR / "repertoire.jsonl")
    compiled: List[Dict[str, Any]] = []
    try:
        with codecs.open(rep_path, "r", "utf-8-sig") as f:
            for i, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                except Exception:
                    continue

                # 1) patterns(list) 우선
                if isinstance(rec.get("patterns"), list):
                    for pat in rec["patterns"]:
                        if not isinstance(pat, str) or not pat:
                            continue
                        try:
                            cre = re.compile(pat, re.IGNORECASE | re.DOTALL)
                            compiled.append({
                                "re": cre,
                                "regex": pat,
                                "label": rec.get("label"),
                                "tags": rec.get("tags", []),
                                "severity": rec.get("severity"),
                                "line": i,
                                "type": rec.get("type"),
                            })
                        except Exception:
                            pass
                    continue

                # 2) 단일 키 지원: pattern / regex / re
                pat = rec.get("pattern") or rec.get("regex") or rec.get("re")
                if isinstance(pat, str) and pat:
                    try:
                        cre = re.compile(pat, re.IGNORECASE | re.DOTALL)
                        compiled.append({
                            "re": cre,
                            "regex": pat,
                            "label": rec.get("label"),
                            "tags": rec.get("tags", []),
                            "severity": rec.get("severity"),
                            "line": i,
                            "type": rec.get("type"),
                        })
                    except Exception:
                        pass
    except FileNotFoundError:
        print("[PATTERN] repertoire.jsonl not found")
    except Exception as e:
        print("[PATTERN] load failed:", repr(e))
    return compiled

# RAG 인덱스/ASR/패턴 로드
@app.on_event("startup")
def _startup() -> None:
    try:
        if hasattr(rag_search, "load_index"):
            rag_search.load_index()
            print("RAG index loaded.")
    except Exception as e:
        print("RAG load skipped:", repr(e))
    try:
        load_asr("small")
        print("ASR model loaded.")
    except Exception as e:
        print("ASR load skipped:", repr(e))

    # 정규식 프리컴파일 캐시
    app.state.compiled_patterns = _load_regex_catalog()
    print(f"[PATTERN] compiled: {len(app.state.compiled_patterns or [])}")

# ──────────────────────────────────────────────────────────────────────
# 입력 모델
# ──────────────────────────────────────────────────────────────────────
class InText(BaseModel):
    text: str
    topk: int = 3
    channel: Optional[str] = "sms"

# ──────────────────────────────────────────────────────────────────────
# 간단 피처 추출 유틸
# ──────────────────────────────────────────────────────────────────────
_money_pat = re.compile(r"([0-9][0-9,\.]{3,})\s*(만원|원|KRW)?")
_acct_pat = re.compile(r"(계좌|농협|국민|신한|우리|하나|카카오|기업|수협|새마을|우체국|IBK|KEB|신협|증권)\s*\d")
_kw_list = [
    "인증번호", "계좌", "이체", "대출", "환급", "세금", "검찰", "수사", "압수", "공문",
    "체납", "보이스피싱", "수수료", "전액", "차단", "법원", "검사", "금지", "링크",
    "URL", "홈페이지", "본인인증", "피싱", "범죄", "송금", "OTP",
]
_kw_pat = re.compile("|".join(map(re.escape, _kw_list)))

def _amount_to_manhwon(text: str) -> Optional[float]:
    if not text:
        return None
    m = _money_pat.search(text.replace(",", ""))
    if not m:
        return None
    num = float(m.group(1))
    unit = (m.group(2) or "").strip()
    if unit == "만원":
        return num
    return num / 10000.0

def _count_keyword_hits(text: str) -> int:
    return len(_kw_pat.findall(text or "")) if text else 0

def _run_rag(text: str, k: int = 3) -> Dict[str, Any]:
    try:
        if hasattr(rag_search, "search"):
            hits = rag_search.search(text, k=k)  # type: ignore[attr-defined]
        else:
            hits = []
        sim = max((float(h.get("score", 0.0)) for h in hits), default=0.0)
        return {"sim": max(0.0, min(1.0, sim)), "hits": hits}
    except Exception as e:
        print("[RAG] search failed:", repr(e))
        return {"sim": 0.0, "hits": []}

def _build_segment(text: str, channel: str, sim: float) -> Dict[str, Any]:
    caller_like = bool(re.search(r"(수사|조사|안전.?계좌|앱\s*설치|OTP|인증번호)", text or "", re.I))
    return {
        "text": text or "",
        "channel": (channel or "other").lower(),
        "is_caller": caller_like,
        "sim": float(sim),
        "keyword_hits": _count_keyword_hits(text or ""),
        "entity_present": bool(_acct_pat.search(text or "")),
        "amount_value": _amount_to_manhwon(text or ""),
    }

# ──────────────────────────────────────────────────────────────────────
# 가중합/레벨 결정 (weights.yaml의 다양한 포맷 허용)
# ──────────────────────────────────────────────────────────────────────
PART_KEYS = {"pattern", "tags", "sim", "keyword", "entity", "amount", "channel", "ml"}

def _extract_weights(root: Dict[str, Any]) -> Dict[str, float]:
    if not isinstance(root, dict):
        return {}
    if isinstance(root.get("risk"), dict) and isinstance(root["risk"].get("weights"), dict):
        return root["risk"]["weights"]  # type: ignore[return-value]
    if isinstance(root.get("weights"), dict):
        return root["weights"]  # type: ignore[return-value]
    return {k: float(root.get(k, 0.0)) for k in PART_KEYS if k in root}

def _extract_thresholds(root: Dict[str, Any]) -> Dict[str, float]:
    if not isinstance(root, dict):
        return {}
    if isinstance(root.get("risk"), dict) and isinstance(root["risk"].get("thresholds"), dict):
        return root["risk"]["thresholds"]  # type: ignore[return-value]
    if isinstance(root.get("thresholds"), dict):
        return root["thresholds"]  # type: ignore[return-value]
    # flat keys
    out = {}
    for k in ("SAFE", "WARN", "HIGH", "CRITICAL"):
        if k in root:
            out[k] = float(root[k])
    return out

def label_to_riskprob(lbl: str) -> float:
    table = {"SAFE": 0.05, "WARN": 0.45, "HIGH": 0.75, "CRITICAL": 0.95}
    return table.get(lbl, 0.45)

def aggregate_final(parts: Dict[str, float], root: Dict[str, Any]) -> float:
    w = _extract_weights(root)
    s = 0.0
    for k, v in parts.items():
        s += float(w.get(k, 0.0)) * float(v)
    return max(0.0, min(1.0, s))

def level_from_score(score: float, root: Dict[str, Any]) -> str:
    th = _extract_thresholds(root) or {"SAFE": 0.25, "WARN": 0.45, "HIGH": 0.70, "CRITICAL": 0.88}
    if score >= th.get("CRITICAL", 0.90):
        return "CRITICAL"
    if score >= th.get("HIGH", 0.70):
        return "HIGH"
    if score >= th.get("WARN", 0.45):
        return "WARN"
    return "SAFE"

def _apply_ensemble(risk_out: Dict[str, Any], text: str) -> Dict[str, Any]:
    ml_res = clf.predict(text or "")
    ml_prob_for_score = max(float(ml_res.get("prob", 0.0)), label_to_riskprob(str(ml_res.get("label", "WARN"))))
    parts = dict(risk_out.get("parts", {}))
    parts["ml"] = float(ml_prob_for_score)
    final_score = aggregate_final(parts, weights_yaml)
    level = level_from_score(final_score, weights_yaml)
    evidence = list(risk_out.get("evidence", [])) + [{"type": "ml", "label": ml_res.get("label"), "prob": ml_res.get("prob")}]
    return {"score": final_score, "level": level, "parts": parts, "ml": ml_res, "evidence": evidence}

# ──────────────────────────────────────────────────────────────────────
# 엔드포인트
# ──────────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/score")
async def score_text(request: Request) -> Dict[str, Any]:
    raw = ensure_utf8_bytes(await request.body())
    data = json.loads(raw.decode("utf-8"))

    text = data.get("text", "") or ""
    topk = int(data.get("topk", 3))
    channel = (data.get("channel") or "sms").lower()

    rag_result = _run_rag(text, k=topk)
    fs = FeatureScores(
        text=text,
        sim=float(rag_result.get("sim", 0.0)),
        keyword_hits=_count_keyword_hits(text),
        entity_present=bool(_acct_pat.search(text)),
        amount_value=_amount_to_manhwon(text),
        channel=channel,
    )
    risk_out = scorer.score(fs)
    ens = _apply_ensemble(risk_out, text)

    seg = _build_segment(text, channel=channel, sim=float(rag_result.get("sim", 0.0)))
    seg.update({"risk": ens["score"], "level": ens["level"], "parts": ens["parts"], "evidence": ens["evidence"]})

    return {
        "score": ens["score"],
        "risk": ens["score"],
        "level": ens["level"],
        "parts": ens["parts"],
        "ml": ens["ml"],
        "hits": rag_result.get("hits", []),
        "segments": [seg],
        "sim": float(rag_result.get("sim", 0.0)),
        "message": "ok",
    }

@app.post("/score/sms")
async def score_sms(request: Request) -> Dict[str, Any]:
    raw = ensure_utf8_bytes(await request.body())
    data = json.loads(raw.decode("utf-8"))

    text = data.get("text", "") or ""
    topk = int(data.get("topk", 3))

    rag_result = _run_rag(text, k=topk)
    fs = FeatureScores(
        text=text,
        sim=float(rag_result.get("sim", 0.0)),
        keyword_hits=_count_keyword_hits(text),
        entity_present=bool(_acct_pat.search(text)),
        amount_value=_amount_to_manhwon(text),
        channel="sms",
    )
    risk_out = scorer.score(fs)
    ens = _apply_ensemble(risk_out, text)

    seg = _build_segment(text, channel="sms", sim=float(rag_result.get("sim", 0.0)))
    seg.update({"risk": ens["score"], "level": ens["level"], "parts": ens["parts"], "evidence": ens["evidence"]})

    return {
        "score": ens["score"],
        "risk": ens["score"],
        "level": ens["level"],
        "parts": ens["parts"],
        "ml": ens["ml"],
        "hits": rag_result.get("hits", []),
        "segments": [seg],
        "sim": float(rag_result.get("sim", 0.0)),
        "message": "ok",
    }

@app.post("/asr/file")
async def asr_and_score(file: UploadFile = File(...), topk: int = 3, sms_mode: bool = False) -> Dict[str, Any]:
    """
    m4a/wav 업로드 → STT → RAG → RiskScorer → ML 앙상블 → 응답
    sms_mode=True이면 channel='sms', 아니면 'call'
    """
    import shutil
    import tempfile
    import traceback

    fd, tmp_path = tempfile.mkstemp(suffix=f"_{file.filename or 'audio.m4a'}")
    os.close(fd)
    try:
        # 업로드 저장
        with open(tmp_path, "wb") as w:
            shutil.copyfileobj(file.file, w)

        # STT
        text = transcribe_file(tmp_path) or ""
        print("[DEBUG] STT text (head):", text[:120])

        # RAG
        try:
            rag_result = _run_rag(text, k=topk)
            print("[DEBUG] sim:", rag_result.get("sim", 0.0), "| hits:", len(rag_result.get("hits", [])))
        except Exception:
            print("[ERROR] rag_search.search failed")
            traceback.print_exc()
            rag_result = {"sim": 0.0, "hits": []}

        channel = "sms" if sms_mode else "call"
        fs = FeatureScores(
            text=text,
            sim=float(rag_result.get("sim", 0.0)),
            keyword_hits=_count_keyword_hits(text),
            entity_present=bool(_acct_pat.search(text)),
            amount_value=_amount_to_manhwon(text),
            channel=channel,
        )
        risk_out = scorer.score(fs)
        ens = _apply_ensemble(risk_out, text)

        seg = _build_segment(text, channel=channel, sim=float(rag_result.get("sim", 0.0)))
        seg["risk"] = ens["score"]
        seg["level"] = ens["level"]
        seg["parts"] = ens["parts"]
        seg["evidence"] = ens["evidence"]

        flat = {
            "text": text,
            "score": ens["score"],
            "risk": ens["score"],
            "level": ens["level"],
            "parts": ens["parts"],
            "ml": ens["ml"],
            "hits": rag_result.get("hits", []),
            "segments": [seg],
            "sim": float(rag_result.get("sim", 0.0)),
            "message": "ok",
        }
        return {
            **flat,
            "result": {"score": flat["score"], "level": flat["level"], "hits": flat["hits"], "message": flat["message"]},
        }
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@app.post("/stt")
async def stt_only(file: UploadFile = File(...)) -> Dict[str, Any]:
    """진단용: 음성 → STT 텍스트만 반환"""
    import shutil
    import tempfile

    fd, tmp_path = tempfile.mkstemp(suffix=f"_{file.filename or 'audio.m4a'}")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as w:
            shutil.copyfileobj(file.file, w)
        text = transcribe_file(tmp_path) or ""
        return {"text": text}
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@app.get("/debug/paths")
def debug_paths() -> Dict[str, Any]:
    return {
        "corpus_path": str(getattr(config, "CORPUS_PATH", BASE_DIR / "corpus.jsonl")),
        "index_path": str(getattr(config, "INDEX_PATH", BASE_DIR / "models" / "faiss.index")),
        "repertoire_path": str(getattr(config, "REPERTOIRE_PATH", BASE_DIR / "repertoire.jsonl")),
        "weights_path": str(getattr(config, "WEIGHTS_PATH", BASE_DIR / "weights.yaml")),
    }

@app.post("/debug/score_explain")
async def debug_score(inp: InText) -> Dict[str, Any]:
    rag_result = _run_rag(inp.text, k=inp.topk)
    fs = FeatureScores(
        text=inp.text or "",
        sim=float(rag_result.get("sim", 0.0)),
        keyword_hits=_count_keyword_hits(inp.text or ""),
        entity_present=bool(_acct_pat.search(inp.text or "")),
        amount_value=_amount_to_manhwon(inp.text or ""),
        channel=(inp.channel or "other"),
    )
    risk_out = scorer.score(fs)
    ml_res = clf.predict(inp.text or "")
    ml_prob = max(float(ml_res.get("prob", 0.0)), label_to_riskprob(str(ml_res.get("label", "WARN"))))
    parts = dict(risk_out.get("parts", {}))
    parts["ml"] = float(ml_prob)

    w = _extract_weights(weights_yaml or {})
    termwise: List[Dict[str, Any]] = []
    total = 0.0
    for k, v in parts.items():
        wt = float(w.get(k, 0.0)) if k in w else 0.0
        contrib = wt * float(v)
        termwise.append({"part": k, "value": float(v), "weight": wt, "contrib": contrib})
        total += contrib

    return {
        "parts": parts,
        "weights_used": w,
        "termwise": termwise,
        "score_sum": max(0.0, min(1.0, total)),
        "thresholds": _extract_thresholds(weights_yaml or {}),
        "ml": ml_res,
    }

@app.post("/debug/regex_probe")
def regex_probe(inp: InText) -> Dict[str, Any]:
    """
    repertoire.jsonl의 정규식이 입력에 어떻게 매칭되는지 확인
    - 캐시된 컴파일(app.state.compiled_patterns) 우선 사용 → 속도/일관성 향상
    - 입력 텍스트는 NFKC 정규화 & 제로폭 제거로 안정화
    - tags는 list/str/dict 모두 안전 정규화
    """
    text = normalize_text(inp.text or "")
    matches: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    # 1) 캐시 사용
    comps = getattr(app.state, "compiled_patterns", None)
    if isinstance(comps, list) and comps:
        for rec in comps:
            try:
                cp = rec.get("re")
                if not cp:
                    continue
                m = cp.search(text)
                if not m:
                    continue
                raw_tags = rec.get("tags")
                if isinstance(raw_tags, dict):
                    tag_info = [k for k in raw_tags.keys()]
                elif isinstance(raw_tags, list):
                    tag_info = [t for t in raw_tags if isinstance(t, str)]
                elif isinstance(raw_tags, str):
                    tag_info = [raw_tags]
                else:
                    tag_info = []

                s, e = m.span()
                snippet = text[max(0, s-15):min(len(text), e+15)]
                matches.append({
                    "line": rec.get("line"),
                    "id": rec.get("id"),
                    "label": rec.get("label"),
                    "pattern": rec.get("regex"),
                    "tags": tag_info,
                    "severity": rec.get("severity"),
                    "type": rec.get("type"),
                    "match": m.group(0),
                    "span": [s, e],
                    "snippet": snippet,
                })
            except Exception as ex:
                errors.append({"where": "cache", "error": repr(ex), "pattern": rec.get("regex")})
        return {"matches": matches, "errors": errors}

    # 2) 캐시가 없으면 파일 직접 파싱(기존 동작 유지, 단 입력 정규화 적용)
    import codecs
    rep_path = getattr(config, "REPERTOIRE_PATH", BASE_DIR / "repertoire.jsonl")
    try:
        with codecs.open(rep_path, "r", "utf-8-sig") as f:
            for i, line in enumerate(f, 1):
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)

                    # 패턴들 꺼내기
                    pats: List[str] = []
                    for key in ("pattern", "regex", "re"):
                        if isinstance(rec.get(key), str):
                            pats.append(rec[key])
                    if "patterns" in rec:
                        if isinstance(rec["patterns"], str):
                            pats.append(rec["patterns"])
                        elif isinstance(rec["patterns"], list):
                            pats.extend([p for p in rec["patterns"] if isinstance(p, str)])

                    if not pats:
                        continue

                    # 태그 처리
                    raw_tags = rec.get("tags")
                    if isinstance(raw_tags, dict):
                        tag_info = list(raw_tags.keys())
                    elif isinstance(raw_tags, list):
                        tag_info = [t for t in raw_tags if isinstance(t, str)]
                    elif isinstance(raw_tags, str):
                        tag_info = [raw_tags]
                    else:
                        tag_info = []

                    # 매칭
                    for pat in pats:
                        try:
                            rx = re.compile(pat, flags=re.IGNORECASE | re.DOTALL | re.UNICODE)
                            m = rx.search(text)
                            if m:
                                s0, e0 = m.span()
                                snippet = text[max(0, s0-15):min(len(text), e0+15)]
                                matches.append({
                                    "line": i,
                                    "id": rec.get("id"),
                                    "label": rec.get("label"),
                                    "pattern": pat,
                                    "tags": tag_info,
                                    "severity": rec.get("severity"),
                                    "type": rec.get("type"),
                                    "match": m.group(0),
                                    "span": [s0, e0],
                                    "snippet": snippet,
                                })
                        except re.error as rxerr:
                            errors.append({"line": i, "error": f"re.error: {rxerr}", "pattern": pat})
                except Exception as e:
                    errors.append({"line": i, "error": repr(e)})
    except FileNotFoundError:
        return {"message": "repertoire.jsonl not found", "matches": [], "errors": []}

    return {"matches": matches, "errors": errors}
