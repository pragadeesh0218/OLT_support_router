"""Automatic evaluation.  python -m src.p3.evaluate

Every set hides the field its queries came from (see pseudo_queries.py). 'casual' adds typos/filler.
These are PROXY numbers: they show which retriever is better, not the accuracy real customers will see.
"""
import json

import numpy as np

from ..config import MODEL_DIR
from .articles import load_articles
from .pseudo_queries import build_eval_sets, casual_items
from .retrievers import build_stack


def rank_metrics(scores_list, golds):
    top1 = top3 = mrr = 0.0
    for s, gold in zip(scores_list, golds):
        order = np.argsort(-s, kind="stable")
        ranks = [int(np.where(order == g)[0][0]) + 1 for g in gold]
        best = min(ranks)
        top1 += best == 1; top3 += best <= 3; mrr += 1.0 / best
    n = max(len(golds), 1)
    return {"top1": round(top1 / n, 3), "top3": round(top3 / n, 3), "mrr": round(mrr / n, 3), "n": len(golds)}


def evaluate(articles, encoder=None, seed=0):
    results = {}
    for name, spec in build_eval_sets(articles, seed).items():
        st = build_stack(articles, spec["exclude"], encoder)
        for variant, items in (("raw", spec["items"]), ("casual", casual_items(spec["items"], seed))):
            golds = [g for _, g in items]
            row = {}
            for rname, r in st.items():
                fn = (lambda q, r=r: r.run(q)["fused"]) if rname == "hybrid" else r.scores
                row[rname] = rank_metrics([fn(q) for q, _ in items], golds)
            results[f"{name}/{variant}"] = row
    return results


def table(results):
    names = sorted({r for row in results.values() for r in row})
    lines = ["query set".ljust(24) + "".join(f"{n:>16}" for n in names), "(top1 / top3)".ljust(24)]
    for k, row in results.items():
        lines.append(k.ljust(24) + "".join(
            f"{row[n]['top1']:>8.2f}/{row[n]['top3']:<6.2f}" if n in row else " " * 16 for n in names))
    return "\n".join(lines)


def main():
    from .router import SearchRouter
    arts = load_articles()
    src = arts[0].get("source", "live") if arts else "none"
    print(f"{len(arts)} articles (source: {src})")
    router = SearchRouter(arts)
    res = evaluate(arts, router.encoder)
    print(table(res))
    print("\nanswer policy:", json.dumps(router.policy.report, indent=1))
    MODEL_DIR.mkdir(exist_ok=True)
    (MODEL_DIR / "p3_eval.json").write_text(json.dumps({"retrieval": res, "policy": router.policy.report}, indent=1))


if __name__ == "__main__":
    main()
