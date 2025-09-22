# app/risk_scorer.py
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
import math
import yaml
import re
import unicodedata

from .services.loader import load_repertoire, normalize_text
from .config import WEIGHTS_PATH

# ----------------------------
# 유틸
# ----------------------------
def _to_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _to_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _get(d: dict, key: str, default):
    if not isinstance(d, dict):
        return default
    v = d.get(key, default)
    return v if v is not None else default

# 정규식 플래그 문자열 → re flags
FLAG_MAP = {
    "i": re.IGNORECASE,
    "s": re.DOTALL,
    "m": re.MULTILINE,
    "x": re.VERBOSE,
    "a": re.ASCII,
    "u": re.UNICODE,
}

def _flags_from_string(flag_str: str) -> int:
    val = 0
    for ch in str(flag_str or ""):
        if ch in FLAG_MAP:
            val |= FLAG_MAP[ch]
    return val

def _extract_inline_flags(pat: str) -> Tuple[str, int]:
    """
    패턴 맨 앞의 인라인 플래그 (?is), (?m) ... 파싱
    여러 개 연속도 허용: (?is)(?m)
    """
    flags_val = 0
    m = re.match(r'^(?:\(\?[ismxau]+\))+', pat)
    if m:
        blob = m.group(0)
        for grp in re.findall(r'\(\?\s*([ismxau]+)\s*\)', blob):
            for ch in grp:
                if ch in FLAG_MAP:
                    flags_val |= FLAG_MAP[ch]
        pat = pat[len(blob):]
    return pat, flags_val

# ----------------------------
# 입력 피처
# ----------------------------
@dataclass
class FeatureScores:
    sim: float                 # 0~1 (임베딩/RAG 유사도)
    keyword_hits: int
    entity_present: bool
    amount_value: Optional[float]  # 금액(만원 단위) or None
    channel: str               # 'sms' | 'call' | 'mms' | 'other'
    text: Optional[str] = None

# ----------------------------
# 스케일러
# ----------------------------
def _linear(x: float, lo, hi) -> float:
    lo = _to_float(lo, 0.0)
    hi = _to_float(hi, 1.0)
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    if x <= lo: return 0.0
    if x >= hi: return 1.0
    return (x - lo) / (hi - lo)

def _logistic(x: float, k=10, x0=0.5):
    k = _to_float(k, 10.0)
    x0 = _to_float(x0, 0.5)
    try:
        return 1 / (1 + math.exp(-k * (x - x0)))
    except OverflowError:
        return 0.0 if (k * (x - x0)) < 0 else 1.0

# ----------------------------
# 위험도 스코어러
# ----------------------------
class RiskScorer:
    def __init__(self, cfg_path: str = None):
        cfg_path = cfg_path or WEIGHTS_PATH
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f) or {}

        raw_w = _get(self.cfg.get("risk", {}), "weights", {})
        self.w: Dict[str, float] = {
            "sim":     _to_float(raw_w.get("sim", 0.55)),
            "keyword": _to_float(raw_w.get("keyword", 0.20)),
            "entity":  _to_float(raw_w.get("entity", 0.10)),
            "amount":  _to_float(raw_w.get("amount", 0.10)),
            "channel": _to_float(raw_w.get("channel", 0.05)),
            "pattern": _to_float(raw_w.get("pattern", 0.35)),
            "tags":    _to_float(raw_w.get("tags", 0.10)),
        }

        self.fcfg = _get(self.cfg.get("risk", {}), "features", {})

        # 태그 테이블/키워드 (선택)
        tcfg = _get(self.fcfg, "tags", {})
        self.tag_table: Dict[str, float] = {}
        for k, v in (tcfg.get("table", {}) or {}).items():
            self.tag_table[str(k)] = _to_float(v, 0.0)

        self.tag_keywords: Dict[str, List[str]] = {}
        for k, arr in (tcfg.get("keywords", {}) or {}).items():
            if isinstance(arr, list):
                self.tag_keywords[str(k)] = [str(x) for x in arr if str(x).strip()]

        # 레벨 경계
        self.level_bins = []
        for lo, hi, lab in _get(self.cfg.get("risk", {}), "level_bins", []):
            self.level_bins.append([_to_float(lo), _to_float(hi), str(lab)])

        # 레파토리 패턴 로드 (+ flags/tags/severity 포함)
        self.repertoire = load_repertoire() or []

        # 정규식 컴파일 (인라인 플래그/flags 필드 모두 반영)
        self._compiled_patterns: List[Tuple[re.Pattern, float, List[str], str]] = []
        for item in self.repertoire:
            raw_rx = item.get("regex") or item.get("pattern")
            if not raw_rx:
                continue
            sev = _to_float(item.get("severity", 0.7))
            tags = item.get("tags") or []
            flags = _flags_from_string(item.get("flags", "")) | re.IGNORECASE
            body, inl = _extract_inline_flags(raw_rx)
            try:
                cp = re.compile(body, flags | inl)
                self._compiled_patterns.append((cp, sev, tags, raw_rx))
            except re.error as e:
                print(f"[pattern-skip] invalid regex: {raw_rx!r} -> {e}")

        # 긴급 핵심 패턴 (태그 부여)
        self._critical_patterns: List[Tuple[str, float, List[str]]] = [
            (r"https?://open\.kakao\.com/o/[A-Za-z0-9]+", 0.95, ["오픈채팅"]),
            (r"\bUSDT\b", 0.90, ["코인송금"]),
            (r"\b(TRC20|ERC20)\b", 0.90, ["코인송금"]),
            (r"0x[a-fA-F0-9]{40}", 0.90, ["코인송금"]),
            (r"안전\s*계좌.*(즉시|바로).{0,4}(이체|송금)", 0.95, ["안전계좌"]),
        ]

        print("[tags] table:", self.tag_table, "keywords:", self.tag_keywords)

    # ---- 개별 피처 ----
    def _score_keyword(self, hits: int) -> float:
        kcfg = _get(self.fcfg, "keyword", {})
        per_hit = _to_float(kcfg.get("per_hit", 0.15))
        cap = _to_float(kcfg.get("cap", 1.0))
        return min(_to_int(hits, 0) * per_hit, cap)

    def _score_entity(self, present: bool) -> float:
        return 1.0 if bool(present) else 0.0

    def _score_amount(self, value) -> float:
        if value is None:
            return 0.0
        val = _to_float(value, 0.0)
        acfg = _get(self.fcfg, "amount", {})
        buckets = acfg.get("buckets", [])
        for b in buckets:
            if not isinstance(b, (list, tuple)) or len(b) < 3:
                continue
            lo, hi, s = b
            lo = _to_float(lo)
            hi = _to_float(hi, float("inf"))
            s = _to_float(s)
            if lo <= val < hi:
                return s
        return 0.0

    def _score_channel(self, ch: str) -> float:
        ccfg = _get(self.fcfg, "channel", {})
        table = ccfg.get("table", {})
        key = (ch or "other").strip().lower()
        default = _to_float(table.get("other", 0.0))
        return _to_float(table.get(key, default))

    def _score_sim(self, sim: float) -> float:
        scfg = _get(self.fcfg, "sim", {})
        kind = str(scfg.get("kind", "linear")).lower()
        if kind == "logistic":
            return _logistic(sim, k=scfg.get("k", 10), x0=scfg.get("x0", 0.5))
        lo = scfg.get("thresh_low", 0.35)
        hi = scfg.get("thresh_high", 0.80)
        return _linear(sim, lo, hi)

    # ---- 패턴/태그 매칭 ----
    def _match_patterns(self, text: Optional[str]) -> Tuple[float, List[str], List[Dict[str, Any]]]:
        """
        텍스트에서 등록된 패턴을 찾고, (max severity, tags, 상세 hits) 반환.
        hits: [{pattern, severity, tags, span, snippet}]
        """
        if not text:
            return 0.0, [], []
        t = normalize_text(text)
        max_sev = 0.0
        found_tags: List[str] = []
        hits: List[Dict[str, Any]] = []

        # 레파토리 패턴
        for cp, sev, tags, raw_rx in self._compiled_patterns:
            m = cp.search(t)
            if m:
                s, e = m.span()
                if sev > max_sev:
                    max_sev = sev
                if tags:
                    found_tags.extend(tags)
                hits.append({
                    "pattern": raw_rx, "severity": sev, "tags": tags,
                    "span": [s, e], "snippet": t[max(0, s-15):min(len(t), e+15)]
                })

        # 긴급 핵심 패턴
        for pat, sev, tags in self._critical_patterns:
            m = re.search(pat, t, flags=re.IGNORECASE)
            if m:
                s, e = m.span()
                if sev > max_sev:
                    max_sev = sev
                if tags:
                    found_tags.extend(tags)
                hits.append({
                    "pattern": pat, "severity": sev, "tags": tags,
                    "span": [s, e], "snippet": t[max(0, s-15):min(len(t), e+15)]
                })

        # 중복 제거
        found_tags = list({tt for tt in found_tags if tt})
        return max_sev, found_tags, hits

    def _score_tags(self, text: Optional[str], tags_from_patterns: List[str]) -> Tuple[float, List[str]]:
        """
        태그 점수:
        - 패턴 매칭 시 딸려온 태그들에 대해 table 점수 중 최댓값
        - (옵션) tag_keywords 테이블 기반 키워드 매칭으로 보강
        """
        t = normalize_text(text or "")
        candidates: List[str] = list(tags_from_patterns)

        # 키워드 기반 태그 추론 (선택)
        for tag, kw_list in self.tag_keywords.items():
            for kw in kw_list:
                try:
                    if re.search(kw, t, flags=re.IGNORECASE):
                        candidates.append(tag)
                        break
                except re.error:
                    continue

        candidates = list({tt for tt in candidates if tt})
        if not candidates:
            return 0.0, []

        score = 0.0
        for tag in candidates:
            val = _to_float(self.tag_table.get(tag, 0.0))
            if val > score:
                score = val
        return score, candidates

    # ---- 최종 ----
    def score(self, fs: FeatureScores) -> Dict[str, Any]:
        # 패턴/태그 매칭
        pat_score, pat_tags, hits = self._match_patterns(fs.text)
        tag_score, tag_names = self._score_tags(fs.text, pat_tags)

        parts = {
            "sim":     self._score_sim(_to_float(fs.sim)),
            "keyword": self._score_keyword(_to_int(fs.keyword_hits)),
            "entity":  self._score_entity(bool(fs.entity_present)),
            "amount":  self._score_amount(fs.amount_value),
            "channel": self._score_channel(fs.channel),
            "pattern": pat_score,
            "tags":    tag_score,
        }

        risk = (
            self.w["sim"]     * parts["sim"]     +
            self.w["keyword"] * parts["keyword"] +
            self.w["entity"]  * parts["entity"]  +
            self.w["amount"]  * parts["amount"]  +
            self.w["channel"] * parts["channel"] +
            self.w["pattern"] * parts["pattern"] +
            self.w["tags"]    * parts["tags"]
        )

        # --- 무증거 억제 로직 ---
        if (
            parts["pattern"] == 0.0 and
            parts["keyword"] == 0.0 and
            parts["entity"]  == 0.0 and
            parts["amount"]  == 0.0 and
            parts["tags"]    == 0.0
        ):
            if parts["sim"] < 0.75:
                risk = min(risk, 0.25)

        # 레벨 산정
        level = "SAFE"
        for lo, hi, lab in self.level_bins:
            if lo <= risk < hi:
                level = lab
                break

        evidence = []
        if parts["pattern"] > 0:
            evidence.append(f"pattern:{parts['pattern']:.2f}")
        if parts["tags"] > 0 and tag_names:
            evidence.append("tags:" + "|".join(tag_names))
        if parts["sim"] > 0:
            evidence.append(f"similarity:{parts['sim']:.2f}")

        return {
            "score": round(risk, 4),
            "risk":  round(risk, 4),
            "level": level,
            "parts": parts,
            "evidence": evidence,
            "hits": hits,  # ← 프로브/디버깅에 활용
        }

    # ---- 디버그용: 텍스트에서 히트만 추출 ----
    def probe(self, text: str) -> Dict[str, Any]:
        _, _, hits = self._match_patterns(text)
        return {
            "compiled_total": len(self._compiled_patterns) + len(self._critical_patterns),
            "hits": hits
        }

# ----------------------------
# 세그먼트 합산
# ----------------------------
def aggregate_segment_scores(
    scorer: "RiskScorer",
    segments: List[dict],
    caller_boost: float = 1.3
) -> dict:
    if not segments:
        return {
            "score": 0.0, "risk": 0.0, "level": "SAFE",
            "parts": {}, "evidence": [], "per_segment": []
        }

    per = []
    num = 0.0
    denom = 0.0
    for seg in segments:
        fs = FeatureScores(
            sim=float(seg.get("sim", 0.0)),
            keyword_hits=int(seg.get("keyword_hits", 0)),
            entity_present=bool(seg.get("entity_present", False)),
            amount_value=seg.get("amount_value", None),
            channel=seg.get("channel", "call"),
            text=seg.get("text", "")
        )
        out = scorer.score(fs)
        w = caller_boost if seg.get("is_caller", False) else 1.0
        num += w * out["risk"]
        denom += w
        per.append({
            "speaker": "caller" if seg.get("is_caller", False) else "victim",
            "risk": out["risk"],
            "level": out["level"],
            "evidence": out.get("evidence", []),
            "parts": out.get("parts", {})
        })
    final_risk = num / max(denom, 1.0)

    level = "SAFE"
    for lo, hi, lab in scorer.level_bins:
        if lo <= final_risk < hi:
            level = lab
            break

    return {
        "score": round(final_risk, 4),
        "risk":  round(final_risk, 4),
        "level": level,
        "per_segment": per
    }
