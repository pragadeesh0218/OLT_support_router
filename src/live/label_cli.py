"""Terminal labeling tool.  python -m src.live.label_cli [--n 100]

For each question: press Enter to accept the suggestion, type a number to pick a category,
s = skip, u = undo last label, q = quit.  Progress is saved after every label.
"""
import argparse
from datetime import datetime, timezone

import pandas as pd

from ..config import CATEGORIES, LABELED_PATH, LABELS, UNLABELED_PATH

COLS = ["id", "text", "label", "source", "created", "url", "suggested_label", "labeled_at"]


def _load_labeled(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=COLS)


def _menu():
    return "\n".join(f"  {i + 1:>2}  {k:<20} {CATEGORIES[k]}" for i, k in enumerate(LABELS))


def run(unlabeled_path=UNLABELED_PATH, labeled_path=LABELED_PATH, limit=None, input_fn=input, out=print):
    todo = pd.read_csv(unlabeled_path)
    labeled = _load_labeled(labeled_path)
    done = set(labeled["id"])
    todo = todo[~todo["id"].isin(done)].reset_index(drop=True)
    if limit:
        todo = todo.head(limit)
    out(_menu())
    out(f"\n{len(todo)} to label, {len(labeled)} already labeled. Enter=accept suggestion, s=skip, u=undo, q=quit\n")

    n_new = 0
    for i, row in todo.iterrows():
        sug = row["suggested_label"]
        out(f"[{i + 1}/{len(todo)}] ({row['source']})\n  {row['text']}\n  suggestion: {sug} ({LABELS.index(sug) + 1})")
        while True:
            a = input_fn("  label> ").strip().lower()
            if a == "q":
                out(f"saved {n_new} new labels. total {len(labeled)}")
                return labeled
            if a == "s":
                break
            if a == "u":
                if len(labeled):
                    out(f"  undid: {labeled.iloc[-1]['text'][:60]}")
                    labeled = labeled.iloc[:-1]
                    labeled.to_csv(labeled_path, index=False)
                    n_new = max(0, n_new - 1)
                continue
            label = sug if a == "" else (LABELS[int(a) - 1] if a.isdigit() and 1 <= int(a) <= len(LABELS)
                                         else (a if a in LABELS else None))
            if label is None:
                out("  not valid: press Enter, a number 1-%d, a category name, s, u or q" % len(LABELS))
                continue
            rec = {"id": row["id"], "text": row["text"], "label": label, "source": row["source"],
                   "created": row["created"], "url": row["url"], "suggested_label": sug,
                   "labeled_at": datetime.now(timezone.utc).isoformat()}
            labeled = pd.concat([labeled, pd.DataFrame([rec])], ignore_index=True)
            labeled_path.parent.mkdir(parents=True, exist_ok=True)
            labeled.to_csv(labeled_path, index=False)
            n_new += 1
            break
    out(f"done. saved {n_new} new labels. total {len(labeled)}")
    return labeled


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="label at most N questions this session")
    run(limit=ap.parse_args().n)
