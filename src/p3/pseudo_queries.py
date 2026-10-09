"""Automatic evaluation / training queries built from the articles themselves (no manual labels).

Each set hides the field the queries come from, so the retriever cannot just look the answer up
('leave-field-out'). Casual variants add typos, filler and dropped punctuation.
"""
import random
import re

from .articles import field_texts
from .text_utils import STOP

HIDE = {  # query source -> fields removed from the index while testing it
    "title": ("title", "slug"), "slug": ("title", "slug"), "subtitle": ("subtitle",), "tags": ("tags",),
    "heading": ("headings",), "body_sentence": ("body",), "faq": ("faq",),
}
PREFIX = ["how do i ", "can i ", "help with ", "question about ", "i need ", "what about ", ""]
SUFFIX = ["", "", " please", " pls", " asap", " ?"]

OOS_SMOKE = [  # tiny out-of-scope sanity list (product config, not training data)
    "hello", "hi there", "good morning", "thanks", "what is the weather today", "tell me a joke", "who are you",
    "best pizza near me", "how to reset my wifi router", "play some music", "what is the capital of france",
    "how tall is mount everest", "write me a poem", "translate this to spanish", "my laptop wont turn on",
    "cheap flights to london", "football score yesterday", "how do i bake bread", "what time is it",
    "recommend a good movie", "asdfgh", "qwerty uiop", "zzzz", "test", "help", "ok", "lorem ipsum dolor",
]


def _typo(w, rng):
    if len(w) < 5:
        return w
    i = rng.randrange(1, len(w) - 2)
    return w[:i] + w[i + 1] + w[i] + w[i + 2:]


def casualize(q: str, rng: random.Random) -> str:
    words = re.sub(r"[^\w\s'-]", " ", q.lower()).split()
    if len(words) > 5 and rng.random() < 0.5:
        words = [w for w in words if w not in STOP] or words
    words = [_typo(w, rng) if rng.random() < 0.12 else w for w in words]
    return (rng.choice(PREFIX) + " ".join(words) + rng.choice(SUFFIX)).strip()


def _sentences(body: str):
    out = []
    for s in re.split(r"(?<=[.!?])\s+", body or ""):
        n = len(s.split())
        if 8 <= n <= 30:
            out.append(s.strip())
    return out


def _spread(items, k):
    if len(items) <= k:
        return items
    step = len(items) / k
    return [items[int(i * step)] for i in range(k)]


def build_eval_sets(articles, seed=0, per_article=2):
    """-> {set_name: {"exclude": tuple, "items": [(query, {gold article indices})]}} (empty sets dropped)."""
    sets = {k: [] for k in HIDE}
    for i, a in enumerate(articles):
        f = field_texts(a)
        if f["title"]:
            sets["title"].append((f["title"], {i}))
        if f["slug"] and len(f["slug"].split()) >= 2:
            sets["slug"].append((f["slug"], {i}))
        if f["subtitle"]:
            sets["subtitle"].append((f["subtitle"], {i}))
        tags = a.get("tags") or []
        if tags:
            sets["tags"].append((" ".join(tags[:3]), {i}))
        heads = [h for h in a.get("headings", []) or [] if 3 <= len(h.split()) <= 14]
        for h in _spread(heads, per_article):
            sets["heading"].append((h, {i}))
        for s in _spread(_sentences(f["body"]), per_article):
            sets["body_sentence"].append((s, {i}))
        for q in a.get("faq", []) or []:
            sets["faq"].append((q, {i}))
    # a FAQ question linked from several articles has several correct answers
    merged = {}
    for q, g in sets["faq"]:
        merged.setdefault(q, set()).update(g)
    sets["faq"] = list(merged.items())
    return {k: {"exclude": HIDE[k], "items": v} for k, v in sets.items() if v}


def casual_items(items, seed=0):
    rng = random.Random(seed)
    return [(casualize(q, rng), g) for q, g in items]


def oos_queries(seed=0, n_gibberish=25):
    rng = random.Random(seed)
    letters = "abcdefghijklmnopqrstuvwxyz"
    gib = [" ".join("".join(rng.choice(letters) for _ in range(rng.randint(3, 8))) for _ in range(rng.randint(1, 4)))
           for _ in range(n_gibberish)]
    return OOS_SMOKE + gib
