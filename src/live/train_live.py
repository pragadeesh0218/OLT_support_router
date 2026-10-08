"""Train the router on HUMAN-LABELED live data.   python -m src.live.train_live [--use-weak]

Refuses to report numbers unless there are enough human labels. Evaluation uses:
  * repeated stratified cross-validation over the human labels
  * a time-based split (train on older questions, test on the newest) when dates are available
The confidence threshold is chosen automatically from out-of-fold predictions.
"""
import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from ..config import LABELED_PATH, LABELS, LIVE_MODEL_PATH, MODEL_DIR, UNLABELED_PATH
from ..model import clean, make_vectorizer

SEED = 42
MIN_LABELS = 150
MIN_PER_CLASS = 5
TARGET_AUTO_ACCURACY = 0.95
NB_VARIANTS = {"MultinomialNB": MultinomialNB, "ComplementNB": ComplementNB}
ALPHAS = [0.03, 0.1, 0.3, 1.0]
WEAK_WEIGHT = 0.3


class NotEnoughLabels(Exception):
    pass


def check_labels(df: pd.DataFrame, min_labels=MIN_LABELS, min_per_class=MIN_PER_CLASS):
    counts = df["label"].value_counts()
    if len(df) < min_labels:
        raise NotEnoughLabels(f"{len(df)} human labels, need at least {min_labels}. Keep labeling "
                              f"(python -m src.live.label_cli). Current counts:\n{counts.to_string()}")
    usable = counts[counts >= 3].index.tolist()        # a class needs >=3 examples to be cross-validated
    dropped = sorted(set(counts.index) - set(usable))
    thin = [k for k in usable if counts[k] < min_per_class]
    return usable, dropped, thin, counts


def _X(vec, texts):
    return vec.transform([clean(t) for t in texts])


def _folds(y, n_repeats, seed=SEED):
    k = int(min(5, pd.Series(y).value_counts().min()))
    return RepeatedStratifiedKFold(n_splits=max(k, 2), n_repeats=n_repeats, random_state=seed)


def _fit(make, X, y, w=None):
    m = make()
    m.fit(X, y, sample_weight=w) if w is not None else m.fit(X, y)
    return m


def train(df: pd.DataFrame, weak: pd.DataFrame = None, min_labels=MIN_LABELS, min_per_class=MIN_PER_CLASS,
          target_accuracy=TARGET_AUTO_ACCURACY) -> dict:
    usable, dropped, thin, counts = check_labels(df, min_labels, min_per_class)
    df = df[df["label"].isin(usable)].reset_index(drop=True)
    vec = make_vectorizer()
    X, y = _X(vec, df["text"]), df["label"].values
    Xw = yw = ww = None
    if weak is not None and len(weak):
        weak = weak[weak["suggested_label"].isin(usable)]
        Xw, yw = _X(vec, weak["text"]), weak["suggested_label"].values
        ww = np.full(len(weak), WEAK_WEIGHT)

    def fit_fold(make, tr):
        from scipy.sparse import vstack
        if Xw is None:
            return _fit(make, X[tr], y[tr])
        return _fit(make, vstack([X[tr], Xw]), np.r_[y[tr], yw], np.r_[np.ones(len(tr)), ww])

    # --- model selection (cross-validation on human labels only) ---
    cv = _folds(y, n_repeats=3)
    grid = {}
    for name, cls in NB_VARIANTS.items():
        for a in ALPHAS:
            f1s = [f1_score(y[va], fit_fold(lambda: cls(alpha=a), tr).predict(X[va]), average="macro")
                   for tr, va in cv.split(X, y)]
            grid[(name, a)] = float(np.mean(f1s))
    variant, alpha = max(grid, key=grid.get)
    make_best = lambda: NB_VARIANTS[variant](alpha=alpha)
    lr_f1 = float(np.mean([f1_score(y[va], fit_fold(lambda: LogisticRegression(max_iter=2000, C=10), tr).predict(X[va]),
                                    average="macro") for tr, va in cv.split(X, y)]))

    # --- out-of-fold confidence / threshold ---
    conf, pred, true = [], [], []
    for tr, va in _folds(y, n_repeats=3, seed=7).split(X, y):
        m = fit_fold(make_best, tr)
        p = m.predict_proba(X[va])
        conf += list(p.max(axis=1)); pred += list(m.classes_[p.argmax(axis=1)]); true += list(y[va])
    conf, pred, true = map(np.array, (conf, pred, true))
    table = []
    for th in np.round(np.arange(0.05, 0.95, 0.025), 3):
        keep = (conf >= th) & (pred != "other_human")
        if keep.sum() >= 10:
            table.append({"threshold": float(th), "auto_routed": float(keep.mean()),
                          "accuracy_when_auto": float((pred[keep] == true[keep]).mean())})
    ok = [r for r in table if r["accuracy_when_auto"] >= target_accuracy]
    threshold = min(ok, key=lambda r: r["threshold"])["threshold"] if ok else (table[-1]["threshold"] if table else 0.5)

    # --- time-based split ---
    temporal = None
    dates = pd.to_datetime(df["created"], errors="coerce", utc=True)
    if dates.notna().mean() >= 0.6:
        order = np.argsort(dates.fillna(dates.min()).values)
        cut = int(len(order) * 0.7)
        tr, te = order[:cut], order[cut:]
        m = fit_fold(make_best, tr)
        p = m.predict(X[te])
        temporal = {"train_n": int(len(tr)), "test_n": int(len(te)), "accuracy": float(accuracy_score(y[te], p)),
                    "macro_f1": float(f1_score(y[te], p, average="macro"))}

    # --- final model on all human labels (+ weak, if asked) ---
    from scipy.sparse import vstack
    final = make_best()
    if Xw is None:
        final.partial_fit(X, y, classes=sorted(usable))
    else:
        final.partial_fit(vstack([X, Xw]), np.r_[y, yw], classes=sorted(usable),
                          sample_weight=np.r_[np.ones(len(y)), ww])
    return {
        "bundle": {"clf": final, "threshold": threshold, "variant": variant, "alpha": alpha,
                   "classes": sorted(usable), "n_labeled": int(len(df)),
                   "trained_at": datetime.now(timezone.utc).isoformat()},
        "report": {"n_labeled": int(len(df)), "class_counts": counts.to_dict(), "dropped_classes": dropped,
                   "thin_classes": thin, "selected": [variant, alpha],
                   "cv_macro_f1": {f"{k[0]} a={k[1]}": round(v, 3) for k, v in grid.items()},
                   "logreg_cv_macro_f1": round(lr_f1, 3), "temporal_split": temporal,
                   "threshold": threshold, "target_auto_accuracy": target_accuracy,
                   "threshold_table": table, "used_weak_labels": Xw is not None},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-weak", action="store_true", help="add rule-labeled rows at low weight (training only)")
    a = ap.parse_args()
    if not LABELED_PATH.exists():
        raise SystemExit("No labeled data yet: run build_dataset, then label_cli.")
    df = pd.read_csv(LABELED_PATH)
    weak = pd.read_csv(UNLABELED_PATH) if a.use_weak and UNLABELED_PATH.exists() else None
    if weak is not None:
        weak = weak[(weak["suggested_score"] >= 0.9) & ~weak["id"].isin(df["id"])]
    try:
        res = train(df, weak)
    except NotEnoughLabels as e:
        raise SystemExit(str(e))
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(res["bundle"], LIVE_MODEL_PATH)
    (MODEL_DIR / "live_metrics.json").write_text(json.dumps(res["report"], indent=2))
    r = res["report"]
    print(f"labels={r['n_labeled']}  selected={r['selected']}  LR baseline macro-F1={r['logreg_cv_macro_f1']}")
    print("best CV macro-F1:", max(r["cv_macro_f1"].values()))
    print("temporal split:", r["temporal_split"])
    print(f"threshold={r['threshold']} (target {r['target_auto_accuracy']} accuracy on auto-routed)")
    if r["dropped_classes"]:
        print("WARNING classes with <3 labels were dropped:", r["dropped_classes"])
    if r["thin_classes"]:
        print("WARNING thin classes (label more):", r["thin_classes"])
    print("saved", LIVE_MODEL_PATH)
    print("serve it: ROUTER_MODEL=models/nb_live.joblib uvicorn src.app:app")


if __name__ == "__main__":
    main()
