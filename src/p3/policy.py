"""When to show an article vs hand off to a human.

A small logistic regression learns P(top result is correct) from cheap signals (retriever agreement, term coverage,
semantic similarity, ...). Training data is automatic: leave-field-out pseudo-queries (label = hit/miss) plus
out-of-scope queries (label = miss). The threshold is the lowest probability that reaches the target precision on
cross-validated predictions.
"""
import math
import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..live.weak_labels import RULES
from ..model import clean
from .text_utils import tokens

# topics that should reach a person even if some article matches: Phase 2 rules + common identity-theft / fraud phrasings
EXTRA_AGENT = [r"stole[n]? (?:my )?(?:identity|ssn|social)", r"identity (?:was |got |has been )?stolen",
               r"(?:someone|somebody) (?:else )?(?:already )?(?:filed|used) (?:a |my )?(?:return|taxes|ssn|social)",
               r"\bfraud", r"\bscam", r"\bhacked\b", r"\blawsuit\b|\bsue\b"]
HUMAN_INTENT = re.compile("|".join(RULES["other_human"] + EXTRA_AGENT))
FEATURES = ["agreement", "margin", "bm25_norm", "coverage", "cos_top1", "cos_margin", "known_ratio", "log_len"]


def needs_agent(query: str) -> bool:
    return bool(HUMAN_INTENT.search(clean(query)))


def features(query, run, bm25, sem="lsa"):
    fused = run["fused"]
    order = np.argsort(-fused)
    t = order[0]
    f2 = fused[order[1]] if len(order) > 1 else 0.0
    bm = run["per"]["bm25"][t]
    cs = run["per"][sem]
    cs2 = np.sort(cs)[-2] if len(cs) > 1 else 0.0
    return [fused[t] / run["max_fused"], (fused[t] - f2) / max(fused[t], 1e-9),
            bm / max(bm25.max_possible(query), 1e-9), bm25.coverage(query, t), float(cs[t]), float(cs[t] - cs2),
            bm25.known_ratio(query), math.log1p(len(tokens(query)))]


class AnswerPolicy:
    def __init__(self, target_precision=0.9, min_accepted=10):
        self.target, self.min_accepted = target_precision, min_accepted
        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        self.threshold, self.report = 0.5, {}

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, int)
        self.model.fit(X, y)
        pos, neg = int(y.sum()), int(len(y) - y.sum())
        if min(pos, neg) >= 5:
            cv = StratifiedKFold(n_splits=min(5, pos, neg), shuffle=True, random_state=0)
            p = cross_val_predict(self.model, X, y, cv=cv, method="predict_proba")[:, 1]
        else:
            p = self.model.predict_proba(X)[:, 1]
        best = None
        for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
            acc = p >= t
            if acc.sum() >= self.min_accepted and y[acc].mean() >= self.target:
                best = float(t); break
        self.threshold = best if best is not None else 0.95
        acc = p >= self.threshold
        self.report = {"n_train": int(len(y)), "n_correct_top1": pos, "n_wrong_or_oos": neg,
                       "threshold": self.threshold, "target_precision": self.target,
                       "precision_at_threshold": float(y[acc].mean()) if acc.any() else None,
                       "answered_share": float(acc.mean()), "target_reached": best is not None}
        return self

    def prob(self, feats) -> float:
        return float(self.model.predict_proba(np.asarray([feats], float))[0, 1])
