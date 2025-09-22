# app/feature_extractor.py
import re, yaml, tldextract, phonenumbers

conf = yaml.safe_load(open("configs/rules.yaml", "r", encoding="utf-8"))

def keyword_score(text: str) -> float:
    s = 0.0
    for kw in conf["keywords"]:
        if re.search(kw["pattern"], text, flags=re.I):
            s += kw["weight"]
    return min(1.0, s)

def entity_score(text: str) -> float:
    s = 0.0
    # URL
    for m in re.findall(r"https?://[^\s]+", text):
        if any(bad in m for bad in conf.get("suspicious_domains", [])):
            s += conf["entities"]["url_weight"]
    # 전화번호 (KR 추정)
    for _m in phonenumbers.PhoneNumberMatcher(text, "KR"):
        s += conf["entities"]["phone_weight"]
        break
    # 금액
    if re.search(r"([0-9]{1,3}(,[0-9]{3})*|[0-9]+)\s*(원|만원|억원)", text):
        s += conf["entities"]["amount_weight"]
    # 계좌 (러프한 국문 은행명 + 번호 패턴)
    if re.search(r"(국민|신한|우리|하나|농협|기업|카카오|케이뱅크|토스).{0,12}\d{2,3}-\d{2,6}-\d{2,6}", text):
        s += conf["entities"]["account_weight"]
    return min(1.0, s)
