"""Summarize three finished Member 3 validation runs; never loads complaint data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.evaluate import comparison_table, plot_curves


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs=3, type=Path, help="Finished result JSONs for seeds 42, 1, 2")
    args = parser.parse_args()
    records = [json.loads(path.read_text(encoding="utf-8")) for path in args.runs]
    if {r["seed"] for r in records} != {42, 1, 2}:
        raise ValueError("Exactly seeds 42, 1 and 2 are required")
    # Paths and the seed may differ; the actual training settings must match.
    ignored = {"seed", "data_dir", "config_source", "run_id", "checkpoint_path", "device"}
    settings = [{k: v for k, v in r["config"].items() if k not in ignored} for r in records]
    if any(cfg != settings[0] for cfg in settings[1:]):
        raise ValueError("Runs have different training settings or runtime environments")
    for r in records:
        cfg, history = r["config"], r["history"]
        epochs = len(history["val_f1"])
        stopped = epochs - r["best_epoch"] >= cfg["patience"]
        if r["split"] != "val" or cfg["experiment_kind"] != "full_training":
            raise ValueError("Only full validation runs are allowed")
        if not (stopped or epochs == cfg["max_epochs"]):
            raise ValueError("Run has not reached early stopping or the epoch limit")
        if abs(max(history["val_f1"]) - r["f1_macro"]) > 1e-12:
            raise ValueError("Saved score does not match the best validation epoch")

    out = ROOT / "results/bilstm_attn/repeats"
    seeds_dir = out / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, r in zip(args.runs, records):
        shutil.copyfile(path, seeds_dir / f"bilstm_attn_seed{r['seed']}.json")
        figure = plot_curves(r["history"], title=f"BiLSTM + Attention: seed {r['seed']}",
                             save_to=str(out / f"learning_curves_seed{r['seed']}.png"))
        plt.close(figure)
        rows.append({"seed": r["seed"], "run_id": r["config"]["run_id"],
                     "best_epoch": r["best_epoch"], "epochs": len(r["history"]["val_f1"]),
                     "f1_macro": r["f1_macro"], "accuracy": r["accuracy"],
                     "source": path.resolve().relative_to(ROOT).as_posix()})
    # Reuse the team's comparison function (pandas sample std, ddof=1).
    comparison_table(str(seeds_dir)).to_csv(out / "comparison.csv")
    scores = [r["f1_macro"] for r in records]
    summary = {"split": "val", "configuration": "05_smaller_lstm", "runs": rows,
               "f1_macro_mean": statistics.mean(scores),
               "f1_macro_sample_std": statistics.stdev(scores), "std_ddof": 1,
               "note": "Validation-selected checkpoints; not independent test performance."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    lines = ["# Three-seed validation results", "",
             "The configuration was fixed after the six-configuration seed-42 search. "
             "Each repeat selected its checkpoint using validation macro F1. All three used the RTX 4050.", "",
             "| Seed | Best epoch | Epochs run | Validation macro F1 | Validation accuracy |",
             "|---|---|---|---|---|"]
    for row in sorted(rows, key=lambda row: row["seed"]):
        lines.append(f"| {row['seed']} | {row['best_epoch']} | {row['epochs']} | "
                     f"{row['f1_macro']:.6f} | {row['accuracy']:.6f} |")
    lines += ["", f"**Macro F1: {statistics.mean(scores):.6f} ± {statistics.stdev(scores):.6f}** "
              "(mean ± sample standard deviation, n=3, ddof=1).", "",
              "These are saved GPU training-evaluation metrics. The separate CPU attention "
              "analysis is not substituted into this comparison. Seed 42 remains the documented "
              "attention-analysis checkpoint; no seed was selected to inflate the aggregate.", "",
              "Validation informed configuration and checkpoint selection. These results measure "
              "variation between seeds on this fixed split, not final test generalization. "
              "Checkpoints remain local and ignored by Git."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
