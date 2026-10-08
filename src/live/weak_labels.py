"""Weak labels: keyword rules + similarity to KB articles.

These are only SUGGESTIONS to speed up human labeling (and an optional low-weight training signal).
All reported accuracy must come from human-confirmed labels, never from these.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import LABELS
from ..model import clean

_STATES = ("alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|georgia|hawaii|idaho|"
           "illinois|indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|"
           "mississippi|missouri|montana|nebraska|nevada|new hampshire|new jersey|new mexico|new york|"
           "north carolina|north dakota|ohio|oklahoma|oregon|pennsylvania|rhode island|south carolina|"
           "south dakota|tennessee|texas|utah|vermont|virginia|washington state|west virginia|wisconsin|wyoming")

RULES = {
    "refund_status": [r"\brefunds?\b", r"direct deposit", r"balance due", r"\bowe[sd]?\b", r"amount owed",
                      r"where'?s my", r"installment", r"pay (?:my |the )?(?:tax(?:es)?|balance)"],
    "efile_pin": [r"e-?fil", r"reject", r"\bpin\b", r"prior[- ]year agi", r"electronic(?:ally)? (?:sign|fil)",
                  r"identity protection", r"transmit"],
    "w2_1099_import": [r"\bw-?2\b", r"upload", r"\bimport", r"\bscan", r"\bbox \d+", r"1099-(?:int|div|r|g|misc)", r"\bocr\b"],
    "state_return": [r"\bstate (?:tax|return|refund|income|filing)", rf"\b(?:{_STATES})\b", r"part[- ]year",
                     r"non-?resident", r"multi-?state", r"reciprocity"],
    "deductions_credits": [r"deduct", r"\bcredits?\b", r"itemiz", r"standard deduction", r"child tax", r"earned income",
                           r"write[- ]?offs?", r"schedule a\b", r"dependents?", r"\bhsa\b", r"\bira contribution"],
    "self_employed": [r"self[- ]employ", r"freelanc", r"schedule c\b", r"1099-(?:nec|k)", r"\bgig\b", r"sole proprietor",
                      r"\bllc\b", r"independent contractor", r"\b(?:uber|lyft|doordash|etsy|upwork|fiverr)\b",
                      r"estimated tax", r"quarterly"],
    "investments_rental": [r"crypto|bitcoin|ethereum", r"capital gains?|capital loss", r"1099-(?:b|da)\b",
                           r"\bstocks?\b|\bshares?\b|\betfs?\b|mutual funds?", r"rental|landlord|tenant|depreciat",
                           r"cost basis", r"schedule (?:d|e)\b", r"dividends?", r"\brsus?\b|\bespp\b|stock options?"],
    "amended_return": [r"amend", r"1040-?x", r"correct(?:ed|ing)? (?:a |my |the )?(?:return|mistake)",
                       r"mistake (?:on|in) my (?:return|taxes)", r"after (?:i )?filed"],
    "extension_deadline": [r"extension", r"\b4868\b", r"deadline", r"due date", r"april 15|apr 15", r"late fil",
                           r"penalt", r"missed the"],
    "account_payment": [r"password", r"log ?in\b|sign ?in\b", r"\baccount\b", r"charged", r"promo|coupon",
                        r"\bprice\b|\bfee\b|how much (?:does|is) .* cost", r"premium", r"cancel", r"subscription"],
    "other_human": [r"\baudit", r"\bcp-?\d{2,4}\b", r"\bnotices?\b", r"identity theft", r"lawyer|attorney", r"complain",
                    r"speak (?:to|with)|talk to (?:a )?(?:real |live )?(?:person|human|agent)", r"\blevy\b|garnish"],
}
_COMPILED = {k: [re.compile(p) for p in v] for k, v in RULES.items()}
assert set(RULES) == set(LABELS)


class WeakLabeler:
    def __init__(self, kb, min_kb_sim: float = 0.15):
        self.kb = kb
        self.min_kb_sim = min_kb_sim
        self._vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self._matrix = self._vec.fit_transform([f'{a["title"]} {a["text"]}' for a in kb])

    def rule_hits(self, text: str) -> dict:
        t = clean(text)
        hits = {k: sum(1 for p in pats if p.search(t)) for k, pats in _COMPILED.items()}
        return {k: v for k, v in hits.items() if v}

    def kb_guess(self, text: str):
        sims = (self._matrix @ self._vec.transform([clean(text)]).T).toarray().ravel()
        i = int(np.argmax(sims)) if sims.size else 0
        return (self.kb[i]["category"], float(sims[i])) if sims.size else (None, 0.0)

    def label(self, text: str) -> dict:
        hits = self.rule_hits(text)
        kb_cat, kb_sim = self.kb_guess(text)
        if hits:
            best = max(hits.values())
            tied = [k for k, v in hits.items() if v == best]
            winner = kb_cat if kb_cat in tied else sorted(tied, key=LABELS.index)[0]
            agree = kb_cat == winner and kb_sim >= self.min_kb_sim
            score = 0.9 if agree else (0.7 if best >= 2 else 0.5)
            return {"label": winner, "score": score, "reason": f"rules={hits} kb={kb_cat}:{kb_sim:.2f}"}
        if kb_cat and kb_sim >= self.min_kb_sim:
            return {"label": kb_cat, "score": 0.4, "reason": f"kb_only={kb_cat}:{kb_sim:.2f}"}
        return {"label": "other_human", "score": 0.3, "reason": "no_signal"}
