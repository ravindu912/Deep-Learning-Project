"""
test_baseline_tfidf.py — final TEST-set evaluation of the TF-IDF + LogReg baseline.

This reproduces notebooks/02_baseline.ipynb EXACTLY (same vectoriser settings,
same classifier settings, same seed) and differs in one respect only: the
fitted model is scored on the held-out test split instead of validation.

Nothing is tuned here. Every hyperparameter is copied from the notebook that
selected them on validation:
    TfidfVectorizer(max_features=10000, ngram_range=(1,2),
                    stop_words='english', sublinear_tf=True)
    LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000,
                       solver='lbfgs', random_state=42)

Run once:
    python scripts/test_baseline_tfidf.py

Writes results/baseline_tfidf_test_seed42.json (split="test").
Takes about two minutes on a laptop CPU; no GPU required.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data import load_splits                                       # noqa: E402
from src.evaluate import evaluate_model, save_results, time_inference  # noqa: E402

SEED = 42


def main():
    print("=" * 62)
    print("  TEST SET RUN — only do this once, for the final result.")
    print("=" * 62)

    np.random.seed(SEED)
    train, val, test, meta = load_splits("data")
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")

    # Fit on TRAINING text only, exactly as the notebook did.
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    t0 = time.time()
    X_train = tfidf.fit_transform(train.text)
    X_test = tfidf.transform(test.text)
    print(f"TF-IDF fit in {time.time() - t0:.2f}s | shape {X_train.shape}")

    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=SEED,
    )
    t0 = time.time()
    clf.fit(X_train, train.y)
    train_time_s = time.time() - t0
    print(f"trained in {train_time_s:.2f}s")

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    res = evaluate_model(test.y, y_pred, y_proba, meta["classes"], split="test")
    # Same parameter accounting as the notebook: coefficients + intercepts.
    res["params"] = int(X_train.shape[1] * meta["n_classes"] + meta["n_classes"])
    res["train_time_s"] = float(train_time_s)
    res["peak_gpu_mb"] = None

    def predict_fn(batch_text):
        return clf.predict_proba(tfidf.transform(batch_text))

    res["inference_ms_per_1k"] = time_inference(predict_fn, test.text[:1000])

    save_results(
        "baseline_tfidf_test",
        res,
        config={
            "C": 1.0,
            "max_features": 10000,
            "ngram_range": [1, 2],
            "stop_words": "english",
            "sublinear_tf": True,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
        seed=SEED,
    )

    print(f"\nTEST macro F1 {res['f1_macro']:.4f} | accuracy {res['accuracy']:.4f} "
          f"| {res['inference_ms_per_1k']:.0f} ms / 1k")
    print("Every deep model must beat this to justify its cost.")


if __name__ == "__main__":
    main()
