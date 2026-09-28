"""
plot_transformer.py — figures and the 3-seed summary for Member 4's Transformer.

Reads only saved results (validation split); never loads a model or the test set.

    python scripts/plot_transformer.py

Writes to results/transformer/:
    final_summary.json                         mean +- std over seeds 42, 1, 2
    figures/transformer_seeds_curves.png       loss + val macro F1, all 3 seeds
    figures/transformer_tuning_f1.png          8 tuning runs vs seed noise band
    figures/transformer_baseline_overfit.png   baseline train vs val loss
    figures/transformer_curves_seed42.png      shared src.evaluate.plot_curves
    figures/transformer_confusion_seed42.png   shared src.evaluate.plot_confusion
"""

from __future__ import annotations

import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.evaluate import plot_confusion, plot_curves               # noqa: E402

RES = os.path.join(ROOT, "results", "transformer")
FIG = os.path.join(RES, "figures")
SEEDS = [42, 1, 2]

# Reference categorical palette, fixed order (blue, orange, aqua), plus chrome.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]            # second channel, so identity is not color-only
SURFACE, INK, INK2, MUTED, GRID, AXIS = ("#fcfcfb", "#0b0b0b", "#52514e",
                                        "#898781", "#e1e0d9", "#c3c2b7")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
    "lines.linewidth": 2,
})


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def run_files(folder: str) -> tuple[dict, dict]:
    return load(os.path.join(folder, "metrics.json")), load(os.path.join(folder, "training_history.json"))


def mean_std(xs) -> dict:
    xs = np.asarray(xs, dtype=float)
    return {"mean": round(float(xs.mean()), 4),
            "std": round(float(xs.std(ddof=1)), 4) if len(xs) > 1 else 0.0,
            "min": round(float(xs.min()), 4), "max": round(float(xs.max()), 4)}


def summary(finals: dict) -> dict:
    per_seed, cols = {}, {k: [] for k in ("f1_macro", "accuracy", "precision_macro",
                                          "recall_macro", "roc_auc_ovr_macro",
                                          "best_epoch", "train_time_s")}
    for s, (m, h) in finals.items():
        vl = h["val_loss"]
        per_seed[s] = {
            "val_f1_macro": round(m["f1_macro"], 4), "val_accuracy": round(m["accuracy"], 4),
            "val_precision_macro": round(m["precision_macro"], 4),
            "val_recall_macro": round(m["recall_macro"], 4),
            "val_roc_auc_ovr_macro": round(m["roc_auc_ovr_macro"], 4),
            "best_epoch": m["best_epoch"], "epochs_run": m["epochs_run"],
            "train_time_min": round(m["train_time_s"] / 60, 1),
            "min_val_loss_epoch": int(np.argmin(vl)) + 1,
            "final_train_minus_val_acc": round(h["epochs"][-1]["train_accuracy"]
                                               - h["epochs"][-1]["val_accuracy"], 4),
        }
        for k in cols:
            cols[k].append(m[k])
    m0 = next(iter(finals.values()))[0]
    per_class = {c: mean_std([finals[s][0]["per_class"][c]["f1-score"] for s in finals])
                 for c in m0["classes"]}
    return {
        "model": "transformer (from scratch)",
        "split": "val",
        "selected_config": {"learning_rate": 5e-5, "dropout": 0.1, "batch_size": 32,
                            "weight_decay": 1e-4, "max_length": 256, "layers": 4,
                            "heads": 4, "d_model": 256, "d_ff": 1024},
        "seeds": list(finals),
        "params_trainable": m0["params"]["trainable"],
        "device": m0["device"],
        "val_f1_macro": mean_std(cols["f1_macro"]),
        "val_accuracy": mean_std(cols["accuracy"]),
        "val_precision_macro": mean_std(cols["precision_macro"]),
        "val_recall_macro": mean_std(cols["recall_macro"]),
        "val_roc_auc_ovr_macro": mean_std(cols["roc_auc_ovr_macro"]),
        "best_epoch": mean_std(cols["best_epoch"]),
        "train_time_min": mean_std(np.array(cols["train_time_s"]) / 60),
        "per_class_val_f1": per_class,
        "per_seed": per_seed,
        "note": "Test set not used. Selection and all numbers here are validation only.",
    }


def fig_seed_curves(finals: dict, path: str) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, (s, (m, h)) in enumerate(finals.items()):
        ep = np.arange(1, len(h["val_loss"]) + 1)
        c, mk = SERIES[i], MARKERS[i]
        a1.plot(ep, h["train_loss"], color=c, ls="--", lw=1.5, alpha=0.8)
        a1.plot(ep, h["val_loss"], color=c, marker=mk, ms=6, label=f"seed {s}")
        a2.plot(ep, h["val_f1"], color=c, marker=mk, ms=6, label=f"seed {s}")
        b = m["best_epoch"]
        a2.plot([b], [h["val_f1"][b - 1]], marker=mk, ms=11, mfc="none", mec=INK, mew=1.2)
    a1.plot([], [], color=MUTED, ls="--", lw=1.5, label="train (dashed)")
    a1.plot([], [], color=MUTED, lw=2, label="validation (solid)")
    a1.set(title="Loss per epoch", xlabel="epoch", ylabel="class-weighted cross-entropy")
    a2.set(title="Validation macro F1 per epoch (ringed = selected epoch)",
           xlabel="epoch", ylabel="macro F1")
    a1.legend(loc="upper right", fontsize=9)
    a2.legend(loc="lower right", fontsize=9)
    for a in (a1, a2):
        a.set_xticks(range(1, 11))
    fig.suptitle("Transformer (from scratch), lr 5e-5 — three seeds", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_tuning(rows: list[dict], seed_stats: dict, path: str) -> None:
    rows = sorted(rows, key=lambda r: float(r["best_val_f1_macro"]))
    labels = [f"run {r['run']}: {r['varied']} (lr {float(r['lr']):g}, do {float(r['dropout']):g}, "
              f"bs {r['batch_size']}, wd {float(r['weight_decay']):g})" for r in rows]
    f1 = [float(r["best_val_f1_macro"]) for r in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    mu, sd = seed_stats["mean"], seed_stats["std"]
    ax.axvspan(mu - sd, mu + sd, color="#cde2fb", zorder=0,
               label=f"3-seed mean ± 1 std of chosen config ({mu:.4f} ± {sd:.4f})")
    ax.axvline(mu, color="#86b6ef", lw=1, zorder=1)
    for yi, v in zip(y, f1):
        ax.plot([v], [yi], "o", ms=8, color=SERIES[0], mec=SURFACE, mew=2, zorder=3)
        ax.text(v + 0.00012, yi, f"{v:.4f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, labels, fontsize=8.5)
    ax.set_xlabel("best validation macro F1 (seed 42)")
    ax.set_xlim(min(f1) - 0.0015, max(f1) + 0.0015)
    ax.grid(axis="y", visible=False)
    ax.set_title("Hyperparameter tuning: every run lies within ~0.003 F1", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_overfit(h: dict, path: str) -> None:
    ep = np.arange(1, len(h["val_loss"]) + 1)
    tr = [e["train_accuracy"] for e in h["epochs"]]
    va = [e["val_accuracy"] for e in h["epochs"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, h["train_loss"], color=SERIES[0], marker="o", ms=6, label="train")
    a1.plot(ep, h["val_loss"], color=SERIES[1], marker="s", ms=6, label="validation")
    k = int(np.argmin(h["val_loss"]))
    a1.annotate(f"lowest val loss\n(epoch {k + 1}, {h['val_loss'][k]:.3f})",
                (k + 1, h["val_loss"][k]), xytext=(k + 2.2, h["val_loss"][k] + 0.12),
                fontsize=8.5, color=INK2, arrowprops=dict(arrowstyle="-", color=MUTED))
    a1.set(title="Loss: validation rises after epoch 3", xlabel="epoch", ylabel="loss")
    a2.plot(ep, tr, color=SERIES[0], marker="o", ms=6, label="train")
    a2.plot(ep, va, color=SERIES[1], marker="s", ms=6, label="validation")
    a2.annotate("", (ep[-1], tr[-1]), (ep[-1], va[-1]),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
    a2.text(ep[-1] - 0.15, (tr[-1] + va[-1]) / 2, f"gap {tr[-1] - va[-1]:.3f}",
            ha="right", va="center", fontsize=8.5, color=INK2)
    a2.set(title="Accuracy: train keeps climbing, validation plateaus", xlabel="epoch",
           ylabel="accuracy")
    for a in (a1, a2):
        a.legend(fontsize=9)
        a.set_xticks(range(1, 11))
    fig.suptitle("Baseline (lr 1e-4, seed 42): mild overfitting", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    finals = {s: run_files(os.path.join(RES, "final", f"seed_{s}")) for s in SEEDS}
    for s, (m, _) in finals.items():
        assert m["split"] == "val" and not m.get("smoke_test_limit"), f"seed {s}: unexpected run"

    summ = summary(finals)
    with open(os.path.join(RES, "final_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)

    with open(os.path.join(RES, "tuning_results.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "done"]
    assert len(rows) == 8, f"expected 8 finished tuning runs, found {len(rows)}"

    fig_seed_curves(finals, os.path.join(FIG, "transformer_seeds_curves.png"))
    fig_tuning(rows, summ["val_f1_macro"], os.path.join(FIG, "transformer_tuning_f1.png"))
    fig_overfit(load(os.path.join(RES, "seed_42", "training_history.json")),
                os.path.join(FIG, "transformer_baseline_overfit.png"))

    # The team's shared plotting functions, so these match the other models.
    m42, h42 = finals[42]
    fig = plot_curves(h42, title="Transformer (from scratch), seed 42 — validation")
    fig.savefig(os.path.join(FIG, "transformer_curves_seed42.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    with plt.rc_context({"axes.grid": False}):
        fig = plot_confusion({**m42, "model": "Transformer (seed 42, validation)"})
    # Write each row-normalised share into its cell, in ink that contrasts.
    cm = np.array(m42["confusion_matrix"], dtype=float)
    cm = cm / cm.sum(axis=1, keepdims=True)
    ax = fig.axes[0]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color=SURFACE if cm[i, j] > 0.5 else INK2)
    fig.savefig(os.path.join(FIG, "transformer_confusion_seed42.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    v = summ["val_f1_macro"]; a = summ["val_accuracy"]
    print(f"val macro F1 {v['mean']:.4f} ± {v['std']:.4f} | val accuracy {a['mean']:.4f} ± {a['std']:.4f}")
    print(f"saved -> {os.path.relpath(os.path.join(RES, 'final_summary.json'), ROOT)}")
    for f in sorted(os.listdir(FIG)):
        print(f"saved -> {os.path.relpath(os.path.join(FIG, f), ROOT)}")


if __name__ == "__main__":
    main()
