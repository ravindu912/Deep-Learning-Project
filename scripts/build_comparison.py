"""
build_comparison.py — final comparison table and figures.

Reads the saved result files and produces everything the report's Results
section needs. Runs no training and modifies no result file.

    python scripts/build_comparison.py

Outputs into results/figures/:
    comparison_test.csv        the table, for the record
    comparison_test.md         the same table, ready to paste into the report
    f1_by_model.png            test macro F1 per model
    f1_vs_params.png           accuracy against model size (the cost figure)
    confusion_<model>.png      one row-normalised confusion matrix per model
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluate import load_all, plot_confusion   # noqa: E402

FIG_DIR = os.path.join("results", "figures")

# Display names, so the table doesn't show internal run names.
PRETTY = {
    "distilbert": "DistilBERT (fine-tuned)",
    "baseline_tfidf_test": "TF-IDF + Logistic Regression",
    "baseline_tfidf": "TF-IDF + Logistic Regression",
    "transformer_test": "Transformer (from scratch)",
    "transformer": "Transformer (from scratch)",
    "bilstm_attn_test": "BiLSTM + attention",
    "bilstm_attn": "BiLSTM + attention",
    "textcnn_test": "TextCNN",
    "textcnn": "TextCNN",
}

# Which models are deep learning; the baseline is reported but not counted.
BASELINE_KEYS = {"baseline_tfidf", "baseline_tfidf_test"}


def params_millions(params):
    if isinstance(params, dict):
        n = params.get("trainable") or params.get("total")
    elif isinstance(params, (int, float)):
        n = params
    else:
        return None
    return n / 1e6 if n else None


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    os.makedirs(FIG_DIR, exist_ok=True)

    results = [r for r in load_all("results") if r.get("split") == "test"]
    if not results:
        sys.exit("No test-split results found. Run the --test evaluations first.")

    rows = []
    for r in results:
        key = r.get("model", "?")
        rows.append({
            "model": PRETTY.get(key, key),
            "_key": key,
            "macro F1": r.get("f1_macro"),
            "accuracy": r.get("accuracy"),
            "precision": r.get("precision_macro"),
            "recall": r.get("recall_macro"),
            "ROC-AUC": r.get("roc_auc_ovr_macro"),
            "params (M)": params_millions(r.get("params")),
            "train (s)": r.get("train_time_s"),
            "inference (ms/1k)": r.get("inference_ms_per_1k"),
        })

    df = (pd.DataFrame(rows)
            .sort_values("macro F1", ascending=False)
            .reset_index(drop=True))

    # ---- table -----------------------------------------------------------
    out = df.drop(columns=["_key"]).round(4)
    csv_path = os.path.join(FIG_DIR, "comparison_test.csv")
    out.to_csv(csv_path, index=False)

    # to_markdown needs the optional `tabulate` package; build it by hand
    # so the script runs on a plain install.
    def _md_table(frame):
        cols = list(frame.columns)
        def cell(v):
            if v is None or (isinstance(v, float) and v != v):
                return ""
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        lines = ["| " + " | ".join(cols) + " |",
                 "| " + " | ".join("---" for _ in cols) + " |"]
        for _, r in frame.iterrows():
            lines.append("| " + " | ".join(cell(r[c]) for c in cols) + " |")
        return "\n".join(lines)

    md = _md_table(out)
    md_path = os.path.join(FIG_DIR, "comparison_test.md")
    with open(md_path, "w") as f:
        f.write("# Test-set comparison\n\n")
        f.write("All models evaluated once on the held-out test split "
                "(12,851 complaints), using the configuration selected on "
                "validation.\n\n")
        f.write(md + "\n\n")
        f.write("Blank cells: the measurement was not recorded by that "
                "model's script. Timings come from the hardware that ran "
                "each model and are indicative, not strictly comparable; "
                "parameter count is measured identically for all.\n")

    print(md)
    print(f"\nsaved -> {csv_path}\n         {md_path}")

    # ---- bar chart -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    colours = ["#6c8ebf" if k in BASELINE_KEYS else "#4269d0" for k in df["_key"]]
    ax.barh(df["model"][::-1], df["macro F1"][::-1], color=colours[::-1])
    ax.set_xlabel("test macro F1")
    ax.set_xlim(0.80, 0.90)
    ax.set_title("Test macro F1 by model (lighter bar = classical baseline)")
    for i, v in enumerate(df["macro F1"][::-1]):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "f1_by_model.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    # ---- accuracy vs cost ------------------------------------------------
    sub = df.dropna(subset=["params (M)"])
    if len(sub) > 1:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.scatter(sub["params (M)"], sub["macro F1"], s=90, color="#4269d0",
                   zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row["model"], (row["params (M)"], row["macro F1"]),
                        textcoords="offset points", xytext=(8, 4), fontsize=9)
        ax.set_xscale("log")
        ax.set_xlabel("trainable parameters, millions (log scale)")
        ax.set_ylabel("test macro F1")
        ax.set_title("Accuracy against model size")
        ax.grid(alpha=0.3, zorder=0)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "f1_vs_params.png"), dpi=150,
                    bbox_inches="tight")
        plt.close(fig)

    # ---- confusion matrices ---------------------------------------------
    for r in results:
        if "confusion_matrix" not in r:
            continue
        name = PRETTY.get(r.get("model", "x"), r.get("model", "x"))
        safe = "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()
        r = {**r, "model": name}
        fig = plot_confusion(r, os.path.join(FIG_DIR, f"confusion_{safe}.png"))
        plt.close(fig)

    # ---- console summary -------------------------------------------------
    best = df.iloc[0]
    base = df[df["_key"].isin(BASELINE_KEYS)]
    print("\n" + "=" * 62)
    print(f"Best model: {best['model']} at {best['macro F1']:.4f} macro F1")

    if not base.empty:
        b = base.iloc[0]
        gap = best["macro F1"] - b["macro F1"]
        print(f"Baseline:   {b['macro F1']:.4f}  (gap to best: {gap:+.4f})")
        beaten = df[(df["macro F1"] < b["macro F1"]) &
                    (~df["_key"].isin(BASELINE_KEYS))]
        if len(beaten):
            print(f"\nDeep models that do NOT beat the classical baseline: "
                  f"{len(beaten)}")
            for _, row in beaten.iterrows():
                print(f"   {row['model']:34} {row['macro F1']:.4f} "
                      f"({row['macro F1'] - b['macro F1']:+.4f})")
        if best["params (M)"] and b["params (M)"]:
            print(f"\nBest model uses {best['params (M)'] / b['params (M)']:.0f}x "
                  f"the parameters of the baseline for {gap:+.4f} macro F1.")
    print("=" * 62)
    print(f"\nFigures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
