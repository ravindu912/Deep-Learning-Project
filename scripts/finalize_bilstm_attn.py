"""Finish Member 3 reporting after all six real experiments complete.

Run with --wait alongside the existing sweep. This never starts another training
run, opens test data, or substitutes partial experiment scores.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluate import plot_curves
from scripts.train_bilstm_attn import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    sweep_dir = ROOT / "results/bilstm_attn/sweep"
    status_path = sweep_dir / "completion_status.json"
    best_path = sweep_dir / "best.json"
    while not best_path.exists():
        ledger = json.loads((sweep_dir / "experiments.json").read_text(encoding="utf-8"))
        completed = sum(row["status"] == "completed" for row in ledger.values())
        failed = [row["name"] for row in ledger.values() if row["status"] in ("failed", "interrupted")]
        write_json(status_path, {
            "status": "waiting_for_training" if not failed else "training_needs_attention",
            "completed_configurations": completed, "total_configurations": 6,
            "failed_configurations": failed, "updated_utc": datetime.now(timezone.utc).isoformat(),
        })
        if failed:
            raise RuntimeError(f"Sweep did not complete: {failed}")
        if not args.wait:
            print(f"Training incomplete: {completed}/6. Use --wait to finish reporting automatically.")
            return
        time.sleep(15)

    ledger = json.loads((sweep_dir / "experiments.json").read_text(encoding="utf-8"))
    if len(ledger) != 6 or any(row["status"] != "completed" for row in ledger.values()):
        raise RuntimeError("Cannot finalize without six completed experiments")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    actual_best = max(ledger.values(), key=lambda row: row["validation_macro_f1"])
    if best["name"] != actual_best["name"]:
        raise RuntimeError("Saved winner does not match the validation score table")
    result_dir = Path(best["result_dir"])
    result = json.loads((result_dir / f"bilstm_attn_seed{best['seed']}.json").read_text(encoding="utf-8"))
    checkpoint = result["config"]["checkpoint_path"]
    write_json(status_path, {"status": "generating_selected_model_analysis", "selected_configuration": best["name"]})
    subprocess.run([sys.executable, "-B", "-u", str(ROOT / "scripts/analyze_bilstm_attn.py"),
                    "--checkpoint", checkpoint], cwd=ROOT, check=True)
    analysis_root = ROOT / "results/bilstm_attn/analysis"
    matches = []
    for path in analysis_root.glob("*/analysis.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("is_selected_checkpoint") and payload["config"]["run_id"] == result["config"]["run_id"]:
            matches.append((path, payload))
    if not matches:
        raise RuntimeError("Selected checkpoint analysis was not saved")
    path, analysis = max(matches, key=lambda item: item[0].stat().st_mtime)
    if len(analysis["figures"]) != 4 or any(
        not (path.parent / item["figure"]).is_file() for item in analysis["figures"]
    ):
        raise RuntimeError("Four saved attention figures are required before reporting completion")
    if abs(analysis["metrics"]["f1_macro"] - best["validation_macro_f1"]) > 1e-3:
        raise RuntimeError("Recomputed validation F1 differs from the selected checkpoint's score")
    figure = plot_curves(result["history"], title=f"BiLSTM + Attention: {best['name']}",
                         save_to=str(path.parent / "learning_curves.png"))
    plt.close(figure)
    metrics, groups = analysis["metrics"], analysis["error_groups"]
    summary = ["# Section 8: BiLSTM + Attention", "",
               f"Using seed {best['seed']}, validation-only tuning selected **{best['name']}** "
               f"(learning rate {best['learning_rate']}, dropout {best['dropout']}, hidden size "
               f"{best['hidden_size']} per direction, batch size {best['batch_size']}). Its best checkpoint "
               f"was epoch {best['best_epoch']}, with validation macro F1 **{metrics['f1_macro']:.4f}** "
               f"and accuracy **{metrics['accuracy']:.4f}**. It misclassified "
               f"**{analysis['misclassified_count']:,} of {analysis['config']['val_rows']:,}** validation complaints.", "",
               f"Keyword-based review flagged {groups['possible_category_overlap']['count']:,} errors with "
               f"multiple financial-topic cues; {groups['short_text_20_words_or_fewer']['count']:,} errors "
               f"had at most 20 cleaned words, and {groups['truncated_after_224_words']['count']:,} errors "
               "exceeded the 224-word input limit. These groups overlap and describe possible review priorities, "
               "not proven causes. Example validation cases:", ""]
    for item in analysis["figures"]:
        if item["true_category"] != item["predicted_category"]:
            excerpt = " ".join(item["words"][:18])
            if len(item["words"]) > 18:
                excerpt += " …"
            summary.append(f"- Row {item['val_row_id']}: **{item['true_category']} → {item['predicted_category']}**. "
                           f"Excerpt: “{excerpt}” [Attention figure]({item['figure']})")
    summary += ["", "Attention weights show how contextual word representations were combined. They are not "
                "a complete or causal explanation of a prediction. Manual review and controlled changes to inputs "
                "would be needed to investigate causes. [Learning curves](learning_curves.png) provide the evidence "
                "for assessing overfitting; errors alone cannot establish it.", "",
                "These validation scores were used to choose the model and are not final held-out test performance. "
                "No test data was used. Cross-model error comparisons remain unavailable because the other members' "
                "per-complaint predictions have not been supplied. Exact required files and further real examples are "
                "listed in the [detailed analysis](section8_analysis.md).", ""]
    (path.parent / "section8_summary.md").write_text("\n".join(summary), encoding="utf-8")
    lines = ["# Member 3: completed validation experiments", "",
             "All six configurations ran on the full shared train/validation splits. No test data was used.", "",
             "| Configuration | LR | Dropout | Hidden size | Batch size | Best epoch | Validation macro F1 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in ledger.values():
        lines.append(f"| {row['name']} | {row['learning_rate']} | {row['dropout']} | {row['hidden_size']} | "
                     f"{row['batch_size']} | {row['best_epoch']} | {row['validation_macro_f1']:.6f} |")
    lines += ["", f"Selected: **{best['name']}**, validation macro F1 **{best['validation_macro_f1']:.6f}**.",
              "These validation scores guided tuning and are not final test results.", "",
              f"[Section 8 analysis](../analysis/{path.parent.name}/section8_analysis.md)",
              f"[Concise Section 8 summary](../analysis/{path.parent.name}/section8_summary.md)",
              f"[Learning curves](../analysis/{path.parent.name}/learning_curves.png)", "",
              "Cross-model error comparisons remain unavailable until other members supply the prediction files "
              "specified in the Section 8 analysis. No comparisons were invented.", ""]
    (sweep_dir / "completed_summary.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(status_path, {
        "status": "complete", "completed_configurations": 6, "selected_configuration": best["name"],
        "validation_macro_f1": best["validation_macro_f1"], "analysis_directory": str(path.parent.relative_to(ROOT)),
        "summary": str((sweep_dir / "completed_summary.md").relative_to(ROOT)),
        "cross_model_analysis": "missing_other_members_predictions",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    })
    print(f"Completed Member 3 experiments and selected-model report: {sweep_dir / 'completed_summary.md'}")


if __name__ == "__main__":
    main()
