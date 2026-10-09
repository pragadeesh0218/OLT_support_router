"""SearchRouter: query -> one article (with URL) or a hand-off to a human agent."""
import os
from time import perf_counter

import numpy as np

from .articles import load_articles
from .policy import AnswerPolicy, features, needs_agent
from .pseudo_queries import build_eval_sets, casual_items, oos_queries
from .retrievers import build_stack, load_sentence_encoder
from .text_utils import tokens

HUMAN_URL = "https://www.olt.com/main/home/service.php"     # OLT customer service page
DEFAULT_EMBED_MODEL = os.getenv("P3_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class SearchRouter:
    def __init__(self, articles, encoder=None, use_dense=False, target_precision=0.9, seed=0):
        self.articles, self.notes, self.seed = articles, [], seed
        if encoder is None and use_dense:
            try:
                encoder = load_sentence_encoder(DEFAULT_EMBED_MODEL)
            except Exception as e:                      # no package / no model download
                self.notes.append(f"dense retriever disabled ({type(e).__name__}); using BM25 + LSA")
        self.encoder = encoder
        self._stacks = {}
        self.full = self.stack(())
        self.sem = "dense" if encoder is not None else "lsa"
        self.policy = AnswerPolicy(target_precision)
        self._train_policy()

    def stack(self, exclude):
        key = tuple(sorted(exclude))
        if key not in self._stacks:
            self._stacks[key] = build_stack(self.articles, key, self.encoder)
        return self._stacks[key]

    def _train_policy(self):
        X, y = [], []
        for spec in build_eval_sets(self.articles, self.seed).values():
            st = self.stack(spec["exclude"])
            for items in (spec["items"], casual_items(spec["items"], self.seed)):
                for q, gold in items:
                    run = st["hybrid"].run(q)
                    X.append(features(q, run, st["bm25"], self.sem))
                    y.append(int(int(np.argmax(run["fused"])) in gold))
        for q in oos_queries(self.seed):
            run = self.full["hybrid"].run(q)
            X.append(features(q, run, self.full["bm25"], self.sem))
            y.append(0)
        self.policy.fit(X, y)

    def _human(self, reason, conf=0.0, alts=None):
        return {"decision": "human", "reason": reason, "confidence": round(conf, 3), "article": None,
                "alternatives": alts or [], "human": {"url": HUMAN_URL, "label": "Talk to a support agent"}}

    def _card(self, i):
        a = self.articles[i]
        return {"id": a.get("id"), "title": a.get("title"), "url": a.get("url"), "category": a.get("category")}

    def answer(self, query: str) -> dict:
        t0 = perf_counter()
        q = (query or "").strip()
        if not tokens(q):
            out = self._human("unclear_question")
        elif needs_agent(q):
            out = self._human("needs_agent_topic")
        else:
            run = self.full["hybrid"].run(q)
            order = [int(i) for i in np.argsort(-run["fused"])[:4]]
            p = self.policy.prob(features(q, run, self.full["bm25"], self.sem))
            alts = [self._card(i) for i in order[1:4]]
            if p >= self.policy.threshold:
                out = {"decision": "article", "reason": "confident_match", "confidence": round(p, 3),
                       "article": self._card(order[0]), "alternatives": alts, "human": None}
            else:
                out = self._human("low_confidence", p, alts)
        out["latency_ms"] = round((perf_counter() - t0) * 1000, 2)
        return out


def build_router(use_dense=None, **kw) -> SearchRouter:
    if use_dense is None:
        use_dense = os.getenv("P3_DENSE", "0") == "1"
    return SearchRouter(load_articles(), use_dense=use_dense, **kw)
