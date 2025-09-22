from risk_scorer import RiskScorer, FeatureScores

scorer = RiskScorer("weights.yaml")
fs = FeatureScores(sim=0.7, keyword_hits=2, entity_present=True, amount_value=150, channel="sms")

print(scorer.score(fs))
