"""Naive Bayes router: classify a question, then rank KB articles inside that category."""
import json
import re
import threading

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from .config import (CATEGORIES, CONFIDENCE_THRESHOLD, FEEDBACK_PATH, KB_PATH,
                     LABELS, MODEL_PATH, TOP_K_ARTICLES)

# --- cleaning ---------------------------------------------------------------
_PII = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), " ssnredacted "),
    (re.compile(r"\b\d{9}\b"), " numredacted "),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), " emailredacted "),
    (re.compile(r"\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"), " phoneredacted "),
]


def clean(text: str) -> str:
    """Lowercase, mask PII so it never reaches the model or the logs."""
    text = text.strip()
    for pattern, repl in _PII:
        text = pattern.sub(repl, text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def make_vectorizer() -> HashingVectorizer:
    # Hashing = fixed feature space, so partial_fit (online learning) works.
    return HashingVectorizer(n_features=2 ** 16, ngram_range=(1, 2), alternate_sign=False,
                             norm="l2", token_pattern=r"(?u)\b[\w-]+\b")


def make_classifier(alpha: float = 0.05) -> MultinomialNB:
    return MultinomialNB(alpha=alpha)


# --- router -----------------------------------------------------------------
class Router:
    def __init__(self, clf=None, kb=None):
        self.vec = make_vectorizer()
        self.clf = clf
        self.kb = kb or json.loads(KB_PATH.read_text())
        self._lock = threading.Lock()
        self._kb_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        texts = [f'{a["title"]} {a["title"]} {a["text"]}' for a in self.kb]
        self._kb_matrix = self._kb_vec.fit_transform(texts)

    @classmethod
    def load(cls):
        router = cls(clf=joblib.load(MODEL_PATH))
        router.replay_feedback()
        return router

    def save(self):
        joblib.dump(self.clf, MODEL_PATH)

    # prediction
    def predict_proba(self, text: str) -> np.ndarray:
        x = self.vec.transform([clean(text)])
        return self.clf.predict_proba(x)[0]

    def articles_for(self, text: str, category: str, k: int = TOP_K_ARTICLES):
        idx = [i for i, a in enumerate(self.kb) if a["category"] == category]
        if not idx:
            return []
        q = self._kb_vec.transform([clean(text)])
        sims = (self._kb_matrix[idx] @ q.T).toarray().ravel()
        order = np.argsort(-sims)[:k]
        return [{"title": self.kb[idx[i]]["title"], "url": self.kb[idx[i]]["url"],
                 "score": round(float(sims[i]), 3)} for i in order]

    def classify(self, text: str) -> dict:
        proba = self.predict_proba(text)
        order = np.argsort(-proba)
        classes = list(self.clf.classes_)
        top = classes[order[0]]
        conf = float(proba[order[0]])
        auto = conf >= CONFIDENCE_THRESHOLD and top != "other_human"
        category = top if auto else "other_human"
        return {
            "category": top,
            "confidence": round(conf, 3),
            "queue": CATEGORIES[category],
            "decision": "auto_route" if auto else "human_review",
            "alternatives": [{"category": classes[i], "confidence": round(float(proba[i]), 3)}
                             for i in order[1:3]],
            "top_articles": self.articles_for(text, top) if auto else [],
        }

    # online learning
    def learn(self, text: str, label: str, persist: bool = True):
        if label not in LABELS:
            raise ValueError(f"unknown label: {label}")
        x = self.vec.transform([clean(text)])
        with self._lock:
            self.clf.partial_fit(x, [label])
        if persist:
            FEEDBACK_PATH.parent.mkdir(exist_ok=True)
            with open(FEEDBACK_PATH, "a") as f:
                # store the cleaned text only (PII already masked)
                f.write(json.dumps({"text": clean(text), "label": label}) + "\n")

    def replay_feedback(self):
        if not FEEDBACK_PATH.exists():
            return 0
        n = 0
        for line in FEEDBACK_PATH.read_text().splitlines():
            row = json.loads(line)
            self.learn(row["text"], row["label"], persist=False)
            n += 1
        return n
