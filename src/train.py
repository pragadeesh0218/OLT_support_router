"""Train the Naive Bayes router and compare it with baselines.

Run:  python -m src.train
- Training data  = synthetic questions + 60% of the hand-written gold set
- Test data      = remaining 40% of gold set (never seen in training or tuning)
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from .config import DATA_DIR, MODEL_DIR, LABELS
from .model import Router, clean, make_vectorizer

SEED = 42


def load():
    syn = pd.read_csv(DATA_DIR / "train_synthetic.csv")
    gold = pd.read_csv(DATA_DIR / "gold.csv")
    g_train, g_test = train_test_split(gold, test_size=0.4, stratify=gold["label"], random_state=SEED)
    train = pd.concat([syn, g_train], ignore_index=True)
    return train, g_train, g_test, syn


def feats(vec, texts):
    return vec.transform([clean(t) for t in texts])


NB_VARIANTS = {"MultinomialNB": MultinomialNB, "ComplementNB": ComplementNB}


def pick_nb(syn, g_train, vec):
    """Choose NB variant + smoothing by cross-validation on the gold-train part only
    (the held-out test set is never used for selection)."""
    skf = RepeatedStratifiedKFold(n_splits=4, n_repeats=5, random_state=SEED)
    scores = {}
    for name, cls in NB_VARIANTS.items():
        for alpha in [0.01, 0.03, 0.05, 0.1, 0.3, 1.0]:
            fold = []
            for tr, va in skf.split(g_train["text"], g_train["label"]):
                tr_df = pd.concat([syn, g_train.iloc[tr]])
                clf = cls(alpha=alpha).fit(feats(vec, tr_df["text"]), tr_df["label"])
                pred = clf.predict(feats(vec, g_train.iloc[va]["text"]))
                fold.append(f1_score(g_train.iloc[va]["label"], pred, average="macro"))
            scores[(name, alpha)] = float(np.mean(fold))
    best = max(scores, key=scores.get)
    return best, scores


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    train, g_train, g_test, syn = load()
    vec = make_vectorizer()

    gold_all = pd.concat([g_train, g_test], ignore_index=True)
    (variant, alpha), cv_scores = pick_nb(syn, gold_all, vec)
    print("CV macro-F1 (variant, alpha):")
    for k, v in cv_scores.items():
        print("  ", k, round(v, 3))
    print("-> selected:", variant, "alpha =", alpha)

    Xtr, ytr = feats(vec, train["text"]), train["label"]
    Xte, yte = feats(vec, g_test["text"]), g_test["label"]

    candidates = {
        "MultinomialNB": MultinomialNB(alpha=alpha),
        "ComplementNB": ComplementNB(alpha=alpha),
        "LogisticRegression": LogisticRegression(max_iter=2000, C=10),
    }
    results = {}
    for name, clf in candidates.items():
        clf.fit(Xtr, ytr)
        t0 = time.perf_counter()
        pred = clf.predict(Xte)
        ms = (time.perf_counter() - t0) / len(g_test) * 1000
        results[name] = {"accuracy": round(accuracy_score(yte, pred), 3),
                         "macro_f1": round(f1_score(yte, pred, average="macro"), 3),
                         "ms_per_query": round(ms, 3)}
        print(f"{name:20s} acc={results[name]['accuracy']}  macroF1={results[name]['macro_f1']}  {ms:.3f} ms/query")

    # final served model: the CV-selected NB variant, trained with partial_fit so online updates work
    nb = NB_VARIANTS[variant](alpha=alpha)
    nb.partial_fit(Xtr, ytr, classes=LABELS)
    router = Router(clf=nb)
    router.save()

    # confidence / coverage trade-off from OUT-OF-FOLD predictions over all gold questions
    # (every prediction comes from a model that never saw that question)
    conf_l, pred_l, true_l = [], [], []
    for tr, va in RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(gold_all["text"], gold_all["label"]):
        d = pd.concat([syn, gold_all.iloc[tr]])
        m = NB_VARIANTS[variant](alpha=alpha).fit(feats(vec, d["text"]), d["label"])
        p = m.predict_proba(feats(vec, gold_all.iloc[va]["text"]))
        conf_l += list(p.max(axis=1)); pred_l += list(m.classes_[p.argmax(axis=1)])
        true_l += list(gold_all.iloc[va]["label"])
    conf, pred, true = np.array(conf_l), np.array(pred_l), np.array(true_l)
    trade = []
    for th in [0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]:
        m = (conf >= th) & (pred != "other_human")
        acc = float((pred[m] == true[m]).mean()) if m.any() else None
        trade.append({"threshold": th, "auto_routed": round(float(m.mean()), 3),
                      "accuracy_when_auto": None if acc is None else round(acc, 3)})
    print("\nthreshold trade-off (out-of-fold):")
    for row in trade:
        print(" ", row)

    print(f"\nPer-class report (served {variant}):")
    print(classification_report(yte, nb.predict(Xte), zero_division=0))

    (MODEL_DIR / "metrics.json").write_text(json.dumps({
        "served_model": variant, "alpha": alpha, "test_size": len(g_test), "train_size": len(train),
        "models": results, "threshold_tradeoff": trade}, indent=2))
    print("saved model + metrics")


if __name__ == "__main__":
    main()
