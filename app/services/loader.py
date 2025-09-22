# app/services/loader.py
from __future__ import annotations

from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
import re
import unicodedata

from .. import config  # CORPUS_PATH, REPERTOIRE_PATH

# ----------------------------
# 공통 유틸
# ----------------------------
def _as_path(p: Optional[str | Path]) -> Path:
    return Path(p) if p else Path("")

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    JSONL 파일을 안전하게 읽어 dict 목록을 반환.
    - BOM(utf-8-sig) 허용
    - //, # 로 시작하는 주석 라인 무시
    """
    items: List[Dict[str, Any]] = []
    if not path or not path.exists():
        print(f"[loader] skip (not found): {path}")
        return items
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                if line.startswith("//") or line.startswith("#"):
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        items.append(obj)
                except Exception as e:
                    print(f"[loader] jsonl parse error @ {path.name}:{ln} -> {e}")
    except Exception as e:
        print(f"[loader] read error: {path} -> {e}")
    return items

def normalize_text(s: str) -> str:
    """유니코드 NFKC 정규화 + 제로폭 제거"""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    return s.replace("\u200b", "").replace("\ufeff", "")

# ----------------------------
# severity 파서 (문자열/숫자 모두 안전하게 변환)
# ----------------------------
SEVERITY_MAP = {
    "safe": 0.0,
    "low": 0.3,
    "warn": 0.7,
    "medium": 0.7,
    "high": 0.9,
    "critical": 0.98,
}

def parse_severity(v) -> float:
    try:
        x = float(v)
        if x < 0.0: return 0.0
        if x > 1.0: return 1.0
        return x
    except Exception:
        s = str(v or "").strip().lower()
        return SEVERITY_MAP.get(s, 0.7)

# ----------------------------
# RAG 코퍼스 로더 (텍스트/콘텐츠만)
# ----------------------------
def load_corpus(path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    """
    JSONL 각 라인에서 text/content/title/source 등을 모아
    RAG 검색용 코퍼스를 만든다. (패턴/룰 항목은 제외)
    """
    corpus_path = _as_path(path) if path else Path(config.CORPUS_PATH)
    raw = _read_jsonl(corpus_path)
    out: List[Dict[str, Any]] = []

    for i, d in enumerate(raw):
        if not isinstance(d, dict):
            continue

        # 명시적으로 pattern/rule/regex 타입은 제외
        t = str(d.get("type", "")).lower()
        if t in {"pattern", "rule", "regex"}:
            continue

        text = d.get("text") or d.get("content") or ""
        title = d.get("title") or d.get("label") or ""
        source = d.get("source") or d.get("url") or str(corpus_path.name)

        text = str(text or "").strip()
        title = str(title or "").strip()
        if not text:
            continue

        out.append({
            "id": d.get("id", f"corpus_{i}"),
            "title": title,
            "text": normalize_text(text),
            "source": source,
            "meta": {k: v for k, v in d.items() if k not in {"text", "content"}}
        })

    print(f"[loader] load_corpus: {len(out)} items from {corpus_path}")
    return out

# ----------------------------
# 패턴/태그 추출 유틸
# ----------------------------
def _extract_patterns(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    주어진 JSONL 레코드 목록에서 패턴 항목만 추출.
    허용 키:
      • 'regex' (단수 문자열)
      • 'pattern' (단수 문자열)
      • 'patterns' (문자열 배열)
      • 'flags'   (예: "is")
    severity 기본값: 0.7
    tags: list[str] 또는 dict -> list로 정규화
    """
    res: List[Dict[str, Any]] = []

    for idx, d in enumerate(items):
        if not isinstance(d, dict):
            continue

        # type 이 명시되었고 패턴류가 아니라면 스킵
        t = str(d.get("type", "")).lower()
        if t and t not in {"pattern", "regex", "rule"}:
            pass

        # 단수/복수 모든 후보를 수집
        candidates: List[str] = []
        for k in ("regex", "pattern"):
            rx = d.get(k)
            if isinstance(rx, str) and rx.strip():
                candidates.append(rx.strip())
        if isinstance(d.get("patterns"), list):
            for rx in d["patterns"]:
                if isinstance(rx, str) and rx.strip():
                    candidates.append(rx.strip())

        if not candidates:
            continue

        # 태그 정규화
        tags = d.get("tags") or []
        if isinstance(tags, dict):
            tags = list(tags.keys())
        if isinstance(tags, str):
            tags = [tags]
        tags = [str(tg).strip() for tg in tags if str(tg).strip()]

        base_label = d.get("label", "")
        flags_str = str(d.get("flags", "") or "")

        # 각 후보 패턴을 개별 항목으로 확장
        for j, rx in enumerate(candidates):
            try:
                re.compile(rx)
            except re.error as e:
                print(f"[loader] skip invalid regex: {rx!r} -> {e}")
                continue

            res.append({
                "id": d.get("id", f"pat_{idx}_{j}"),
                "label": base_label,
                "regex": rx,
                "pattern": rx,   # 호환 키
                "flags": flags_str,
                "severity": parse_severity(d.get("severity", 0.7)),
                "tags": list(tags),
            })

    return res

def _dedup_patterns(pats: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    동일한 정규식은 하나로 합치고 severity는 최댓값 사용.
    tags는 합집합으로 병합. flags는 가장 긴(혹은 최초) 값 채택.
    """
    merged_by_rx: Dict[str, Dict[str, Any]] = {}
    for it in pats:
        rx = str(it.get("regex") or it.get("pattern") or "").strip()
        if not rx:
            continue
        prev = merged_by_rx.get(rx)
        if not prev:
            merged_by_rx[rx] = {
                "id": it.get("id", f"rx_{len(merged_by_rx)}"),
                "label": it.get("label", ""),
                "regex": rx,
                "pattern": rx,
                "flags": it.get("flags", ""),
                "severity": parse_severity(it.get("severity", 0.7)),
                "tags": list(set(it.get("tags", []) or [])),
            }
        else:
            prev["severity"] = max(prev["severity"], parse_severity(it.get("severity", 0.7)))
            prev["tags"] = list(set((prev.get("tags") or []) + (it.get("tags") or [])))
            if not prev.get("flags"):
                prev["flags"] = it.get("flags", "")

    return list(merged_by_rx.values())

# ----------------------------
# 레파토리 로더
# ----------------------------
def load_repertoire(
    repertoire_path: Optional[str | Path] = None,
    corpus_path: Optional[str | Path] = None,
    merge_corpus_patterns: bool = True,
) -> List[Dict[str, Any]]:
    """
    - repertoire.jsonl의 패턴 + (옵션) corpus.jsonl의 `"type":"pattern"`를 병합
    - 각 항목에 tags/flags/severity를 유지
    - 결과: {'regex','pattern','flags','severity','tags',...} 리스트
    """
    rep_path = _as_path(repertoire_path) if repertoire_path else Path(config.REPERTOIRE_PATH)
    cor_path = _as_path(corpus_path) if corpus_path else Path(config.CORPUS_PATH)

    rep_raw = _read_jsonl(rep_path)
    rep_pats = _extract_patterns(rep_raw)

    cor_pats: List[Dict[str, Any]] = []
    if merge_corpus_patterns:
        cor_raw = _read_jsonl(cor_path)
        cor_pats = _extract_patterns(cor_raw)

    all_pats = rep_pats + cor_pats
    merged = _dedup_patterns(all_pats)

    print(
        "[loader] load_repertoire: "
        f"rep={len(rep_pats)} from {rep_path.name}, "
        f"cor={len(cor_pats)} from {cor_path.name if merge_corpus_patterns else 'skip'}, "
        f"merged={len(merged)}"
    )
    return merged

# ----------------------------
# 선택: 진단용 summary
# ----------------------------
def debug_summary() -> Dict[str, Any]:
    cor = load_corpus()
    rep = load_repertoire()
    return {
        "corpus_count": len(cor),
        "repertoire_patterns": len(rep),
        "corpus_path": str(config.CORPUS_PATH),
        "repertoire_path": str(config.REPERTOIRE_PATH),
    }

if __name__ == "__main__":
    s = debug_summary()
    print("[loader] summary:", s)
