"""
evaluate.py — the shared evaluation module. EVERY model reports through this.

Why it is shared: the rubric marks the comparison on fairness. If each member
computes metrics their own way, the numbers are not comparable and we lose marks.

Usage in a model script:

    from src.evaluate import evaluate_model, count_params, time_inference, save_results

    res = evaluate_model(y_true, y_pred, y_proba, classes, split="test")
    res["params"] = count_params(model)
    res["inference_ms_per_1k"] = time_inference(predict_fn, X_test)
    save_results("distilbert", res, history=history, out_dir="results")

Then plot_all("results") builds the comparison table and figures.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def evaluate_model(y_true, y_pred, y_proba=None, classes=None, split="test") -> dict:
    """All the metrics the rubric asks for, in one dict.

    y_proba: (n, n_classes) probabilities. Needed for ROC-AUC; pass None to skip.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    res = {
        "split": split,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # Macro F1 is our headline number: the classes are imbalanced, so accuracy
    # alone would reward a model that ignores the small categories.
    if y_proba is not None:
        try:
            res["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
        except ValueError as e:
            res["roc_auc_ovr_macro"] = None
            print(f"ROC-AUC skipped: {e}")

    res["per_class"] = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True, zero_division=0
    )
    res["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    res["classes"] = list(classes) if classes is not None else None
    return res


def count_params(model) -> dict:
    """Trainable and total parameters. Works for a torch module."""
    try:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"total": int(total), "trainable": int(trainable)}
    except AttributeError:
        return {"total": None, "trainable": None}


def time_inference(predict_fn, X, n: int = 1000, warmup: int = 2) -> float:
    """Milliseconds to predict `n` examples. Same n for every model, or it is unfair."""
    sample = X[:n]
    for _ in range(warmup):
        predict_fn(sample)
    start = time.perf_counter()
    predict_fn(sample)
    return (time.perf_counter() - start) * 1000.0


def peak_gpu_mb() -> float | None:
    """Peak GPU memory in MB since the last reset, or None on CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e6
    except ImportError:
        pass
    return None


# --------------------------------------------------------------------------
# saving and loading
# --------------------------------------------------------------------------

def save_results(model_name: str, results: dict, history=None, config=None,
                 seed: int = 42, out_dir: str = "results") -> str:
    """Write one JSON per model per seed. The comparison reads these."""
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "model": model_name,
        "seed": seed,
        "config": config or {},
        "history": history or {},
        **results,
    }
    path = os.path.join(out_dir, f"{model_name}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved -> {path}   f1_macro={results.get('f1_macro'):.4f}")
    return path


# Directories that hold intermediate runs, not final results. A tuning trial
# is not a result: averaging them into a model's headline number would drag it
# down and misrepresent the model.
SKIP_DIRS = {"tuning", "sweep", "sweeps", "trials", "analysis", "checkpoints",
             "figures", "runs", "wandb"}


def load_all(out_dir: str = "results", include_intermediate: bool = False) -> list:
    """Load every final result file, searching subdirectories.

    Members store results differently — some as flat files, some as
    results/<model>/final/seed_N/metrics.json — so this walks the tree.

    Two things it protects against:
      * tuning and sweep trials being counted as results (SKIP_DIRS)
      * the same (model, seed, split) appearing twice, e.g. an early run at
        results/transformer/seed_42/ and the final one at
        results/transformer/final/seed_42/. A path under final/ wins;
        otherwise the most recently modified file wins.
    """
    found = []
    for root, dirs, files in os.walk(out_dir):
        if not include_intermediate:
            dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS]
        for name in sorted(files):
            if not name.endswith(".json"):
                continue
            path = os.path.join(root, name)
            try:
                with open(path) as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError):
                continue
            # Not every JSON under results/ is a result (configs, summaries,
            # split metadata). A result has a macro F1.
            if not isinstance(data, dict) or "f1_macro" not in data:
                continue
            data["_path"] = path
            found.append(data)

    # Deduplicate on (model, seed, split).
    best: dict = {}
    for d in found:
        key = (d.get("model"), d.get("seed"), d.get("split"))
        current = best.get(key)
        if current is None:
            best[key] = d
            continue
        is_final = "final" in d["_path"].replace(os.sep, "/").split("/")
        was_final = "final" in current["_path"].replace(os.sep, "/").split("/")
        if is_final and not was_final:
            best[key] = d
        elif is_final == was_final and \
                os.path.getmtime(d["_path"]) > os.path.getmtime(current["_path"]):
            best[key] = d

    dropped = len(found) - len(best)
    if dropped:
        print(f"({dropped} duplicate result file(s) ignored — kept final/newest)")

    return sorted(best.values(), key=lambda d: (str(d.get("split")),
                                                str(d.get("model")),
                                                str(d.get("seed"))))


# --------------------------------------------------------------------------
# comparison outputs (run this at the end, once all models are done)
# --------------------------------------------------------------------------

def _params_millions(params):
    """Parameter count in millions, tolerating either result format.

    Most models store {"total": n, "trainable": n}; the TF-IDF baseline
    stores a bare int. Both are valid records of what was run, so the
    reader adapts rather than the result file being rewritten.
    """
    if isinstance(params, dict):
        n = params.get("trainable") or params.get("total")
    elif isinstance(params, (int, float)):
        n = params
    else:
        return None
    return n / 1e6 if n else None


def comparison_table(out_dir: str = "results"):
    """Mean +- std across seeds, one row per model. Returns a DataFrame."""
    import pandas as pd

    rows = []
    for r in load_all(out_dir):
        rows.append({
            "split": r.get("split"),
            "model": r["model"],
            "seed": r.get("seed"),
            "accuracy": r.get("accuracy"),
            "f1_macro": r.get("f1_macro"),
            "f1_weighted": r.get("f1_weighted"),
            "precision_macro": r.get("precision_macro"),
            "recall_macro": r.get("recall_macro"),
            "roc_auc": r.get("roc_auc_ovr_macro"),
            "params_m": _params_millions(r.get("params")),
            "train_time_s": r.get("train_time_s"),
            "inference_ms_per_1k": r.get("inference_ms_per_1k"),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No result files found.")
        return df

    # Group by split as well as model. Averaging a test score together with
    # validation scores would be meaningless.
    num = df.select_dtypes("number").columns.drop("seed", errors="ignore")
    agg = df.groupby(["split", "model"])[list(num)].agg(["mean", "std"]).round(4)

    # Report how many runs each mean is built from — one seed is not a mean.
    agg[("runs", "n")] = df.groupby(["split", "model"]).size()

    missing = [c for c in ("inference_ms_per_1k", "params_m")
               if c in df.columns and df[c].isna().any()]
    if missing:
        models = sorted(df.loc[df[missing[0]].isna(), "model"].unique())
        print(f"note: {', '.join(missing)} missing for {', '.join(models)} — "
              "measure all models on one machine before reporting efficiency")

    return agg


def plot_confusion(results: dict, save_to: str | None = None, normalise: bool = True):
    """Confusion matrix heatmap for one model."""
    import matplotlib.pyplot as plt

    cm = np.array(results["confusion_matrix"], dtype=float)
    if normalise:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

    classes = results.get("classes") or [str(i) for i in range(len(cm))]
    short = [c[:28] for c in classes]

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
    ax.set_xticks(range(len(short)), short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(short)), short, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{results.get('model','model')} — confusion matrix"
                 f"{' (row-normalised)' if normalise else ''}")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()

    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
        print(f"saved -> {save_to}")
    return fig


def plot_curves(history: dict, title: str = "", save_to: str | None = None):
    """Learning curves. history: {'train_loss':[], 'val_loss':[], 'val_f1':[]}"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    axes[0].plot(epochs, history.get("train_loss", []), label="train")
    axes[0].plot(epochs, history.get("val_loss", []), label="validation")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history.get("val_f1", []), color="tab:green")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("macro F1")
    axes[1].set_title("Validation macro F1"); axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    if save_to:
        fig.savefig(save_to, dpi=150, bbox_inches="tight")
        print(f"saved -> {save_to}")
    return fig


if __name__ == "__main__":
    print(comparison_table())
