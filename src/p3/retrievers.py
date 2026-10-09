"""Retrievers over articles. All expose  .scores(query) -> np.ndarray (one score per article)."""
import math
import re
from collections import Counter

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from .articles import FIELD_WEIGHTS, field_texts
from .text_utils import tokens


def weighted_doc_tokens(article, exclude=(), weights=FIELD_WEIGHTS):
    out = []
    for f, text in field_texts(article).items():
        if f in exclude or not text:
            continue
        out.extend(tokens(text) * weights.get(f, 1))
    return out


def _unit(M):
    n = np.linalg.norm(M, axis=-1, keepdims=True)
    return M / np.where(n == 0, 1, n)


class BM25Retriever:
    """BM25 over a field-weighted document (title/slug/tags/headings/FAQ repeated more than body)."""
    name = "bm25"

    def __init__(self, articles, exclude=(), k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(weighted_doc_tokens(a, exclude)) for a in articles]
        self.len = np.array([sum(c.values()) for c in self.tf], float)
        self.avg = max(float(self.len.mean()), 1.0) if len(self.len) else 1.0
        n, df = len(articles), Counter()
        for c in self.tf:
            df.update(c.keys())
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    @staticmethod
    def terms(query):
        return list(dict.fromkeys(tokens(query)))

    def scores(self, query):
        s = np.zeros(len(self.tf))
        for t in self.terms(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, c in enumerate(self.tf):
                f = c.get(t, 0)
                if f:
                    s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s

    def max_possible(self, query):
        return sum(self.idf[t] * (self.k1 + 1) for t in self.terms(query) if t in self.idf)

    def coverage(self, query, i):
        known = [t for t in self.terms(query) if t in self.idf]
        mass = sum(self.idf[t] for t in known)
        return sum(self.idf[t] for t in known if self.tf[i].get(t)) / mass if mass else 0.0

    def known_ratio(self, query):
        ts = self.terms(query)
        return sum(t in self.idf for t in ts) / len(ts) if ts else 0.0


class LSARetriever:
    """Latent semantic analysis (TF-IDF + SVD): an offline 'meaning' retriever that needs no downloads."""
    name = "lsa"

    def __init__(self, articles, exclude=(), dims=120):
        docs = [" ".join(weighted_doc_tokens(a, exclude)) for a in articles]
        self.tfidf = TfidfVectorizer(tokenizer=str.split, token_pattern=None, lowercase=False, sublinear_tf=True)
        X = self.tfidf.fit_transform(docs)
        k = max(1, min(dims, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=0)
        self.Z = _unit(self.svd.fit_transform(X))

    def scores(self, query):
        v = _unit(self.svd.transform(self.tfidf.transform([" ".join(tokens(query))])))
        return self.Z @ v[0]


class DenseRetriever:
    """Sentence-embedding retriever. `encoder(list[str]) -> np.ndarray`. Optional: needs a model download."""
    name = "dense"

    def __init__(self, articles, encoder, exclude=()):
        self.enc = encoder
        docs = []
        for a in articles:
            f = field_texts(a)
            parts = [f[k] for k in ("title", "subtitle", "tags", "headings", "faq") if k not in exclude and f[k]]
            if "body" not in exclude and f["body"]:
                parts.append(f["body"][:1200])
            docs.append(" . ".join(parts) or a.get("title", ""))
        self.D = _unit(np.asarray(encoder(docs), dtype=float))

    def scores(self, query):
        return self.D @ _unit(np.asarray(self.enc([query]), dtype=float))[0]


def load_sentence_encoder(name):
    from sentence_transformers import SentenceTransformer   # optional dependency
    model = SentenceTransformer(name)
    return lambda texts: model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)


class NBRetriever:
    """Naive Bayes baseline: every article is a class; its fields / body windows are the training documents."""
    name = "nb"

    def __init__(self, articles, exclude=(), alpha=0.1, window=40):
        texts, labels = [], []
        for i, a in enumerate(articles):
            f = field_texts(a)
            for k in ("title", "slug", "subtitle", "tags"):
                if k not in exclude and f[k]:
                    texts.append(f[k]); labels.append(i)
            for k in ("headings", "faq"):
                if k not in exclude:
                    for piece in a.get(k, []) or []:
                        texts.append(piece); labels.append(i)
            if "body" not in exclude and f["body"]:
                words = f["body"].split()
                for s in range(0, max(len(words), 1), window):
                    texts.append(" ".join(words[s:s + window])); labels.append(i)
        self.n = len(articles)
        self.cv = CountVectorizer(analyzer=tokens)
        self.clf = MultinomialNB(alpha=alpha).fit(self.cv.fit_transform(texts), labels)

    def scores(self, query):
        lp = self.clf.predict_log_proba(self.cv.transform([query]))[0]
        s = np.full(self.n, lp.min() - 10.0)
        s[self.clf.classes_] = lp
        return s


class Hybrid:
    """Reciprocal-rank fusion. `agreement` = how strongly the retrievers agree on the #1 article."""

    def __init__(self, retrievers: dict, weights=None, k=20):
        self.r, self.k = retrievers, k
        self.w = {n: (weights or {}).get(n, 1.0) for n in retrievers}
        self.max_fused = sum(self.w.values()) / (k + 1)

    def run(self, query):
        per, fused = {}, None
        for name, r in self.r.items():
            s = r.scores(query)
            per[name] = s
            order = np.argsort(-s, kind="stable")
            rank = np.empty(len(s)); rank[order] = np.arange(len(s))
            part = self.w[name] / (self.k + rank + 1)
            fused = part if fused is None else fused + part
        return {"per": per, "fused": fused, "max_fused": self.max_fused}


def build_stack(articles, exclude=(), encoder=None) -> dict:
    """All retrievers for one view of the articles (exclude = fields hidden from the index)."""
    st = {"bm25": BM25Retriever(articles, exclude), "lsa": LSARetriever(articles, exclude),
          "nb": NBRetriever(articles, exclude)}
    members = {"bm25": st["bm25"], "lsa": st["lsa"]}
    if encoder is not None:
        st["dense"] = DenseRetriever(articles, encoder, exclude)
        members["dense"] = st["dense"]
    st["hybrid"] = Hybrid(members)
    return st
