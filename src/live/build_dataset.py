"""Raw live data -> clean, de-duplicated, PII-masked CSV ready for human labeling.

    python -m src.live.build_dataset

Reads  data/live/raw/*.jsonl   (collectors + production log)
Writes data/live/unlabeled.csv (skips ids already in data/live/labeled.csv)
"""
import argparse
import json
import re
from itertools import zip_longest

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import KB_PATH, LABELED_PATH, RAW_DIR, UNLABELED_PATH
from ..model import mask_pii
from .weak_labels import WeakLabeler

NEAR_DUP_SIM = 0.85


def load_raw(raw_dir=RAW_DIR) -> pd.DataFrame:
    rows = []
    for f in sorted(raw_dir.glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def build_text(row) -> str:
    """Real support messages are short: use the title, add a body excerpt only if the title is tiny."""
    title = (row.get("title") or "").strip()
    body = (row.get("text") or "").strip()
    if len(title.split()) < 4 and body:
        title = f"{title} {body[:200]}".strip()
    return mask_pii(" ".join(title.split()))


def acceptable(text: str) -> bool:
    words = text.split()
    if not 3 <= len(words) <= 80:
        return False
    ascii_ratio = sum(c.isascii() for c in text) / max(len(text), 1)
    return ascii_ratio >= 0.9 and bool(re.search(r"[A-Za-z]{3}", text))


def drop_duplicates(texts, threshold=NEAR_DUP_SIM, chunk=500):
    """Greedy near-duplicate removal (exact + TF-IDF cosine). Returns the indices to keep."""
    norm = [re.sub(r"\W+", " ", t.lower()).strip() for t in texts]
    seen, first = set(), []
    for i, n in enumerate(norm):
        if n not in seen:
            seen.add(n); first.append(i)
    if len(first) < 2:
        return first
    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform([norm[i] for i in first])
    keep = []
    for start in range(0, len(first), chunk):
        sims = (X[start:start + chunk] @ X.T).tocsr()
        for r in range(sims.shape[0]):
            idx = start + r
            row = sims.getrow(r)
            earlier_dup = any(c < idx and v >= threshold for c, v in zip(row.indices, row.data))
            if not earlier_dup:
                keep.append(first[idx])
    return keep


def build(raw: pd.DataFrame, kb, already_labeled=frozenset(), seed=42) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    df["text"] = [build_text(r) for r in df.to_dict("records")]
    df = df[~df["id"].isin(already_labeled)]
    df = df[[acceptable(t) for t in df["text"]]].reset_index(drop=True)
    df = df.iloc[drop_duplicates(df["text"].tolist())].reset_index(drop=True)

    wl = WeakLabeler(kb)
    weak = [wl.label(t) for t in df["text"]]
    df["suggested_label"] = [w["label"] for w in weak]
    df["suggested_score"] = [w["score"] for w in weak]
    df["reason"] = [w["reason"] for w in weak]

    # interleave suggested classes so the labeler sees every topic early, not 500 refund questions in a row
    rng = np.random.default_rng(seed)
    groups = [g.sample(frac=1, random_state=int(rng.integers(1e6))).to_dict("records")
              for _, g in df.groupby("suggested_label")]
    ordered = [r for tup in zip_longest(*groups) for r in tup if r is not None]
    cols = ["id", "source", "created", "url", "text", "suggested_label", "suggested_score", "reason"]
    return pd.DataFrame(ordered).reindex(columns=cols)


def main():
    argparse.ArgumentParser().parse_args()
    raw = load_raw()
    if raw.empty:
        raise SystemExit(f"No raw data in {RAW_DIR}. Run the collectors first (see README_PHASE2.md).")
    done = set(pd.read_csv(LABELED_PATH)["id"]) if LABELED_PATH.exists() else set()
    kb = json.loads(KB_PATH.read_text())
    out = build(raw, kb, done)
    UNLABELED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(UNLABELED_PATH, index=False)
    print(f"raw={len(raw)}  after cleaning/dedupe={len(out)}  (already labeled skipped: {len(done)})")
    print(out["source"].value_counts().to_string())
    print("\nsuggested label mix:\n" + out["suggested_label"].value_counts().to_string())
    print(f"\nwrote {UNLABELED_PATH}\nnext: python -m src.live.label_cli")


if __name__ == "__main__":
    main()
