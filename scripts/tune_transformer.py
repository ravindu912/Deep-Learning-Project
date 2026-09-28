"""
tune_transformer.py — Member 4's hyperparameter search for the Transformer.

Runs the 8 configurations below one after another with seed 42, using the
VALIDATION set only (each run is scripts/train_transformer.py). The test set is
never loaded into a model or used to choose anything.

Resumable: a run whose metrics.json already exists is skipped, so after a
Colab disconnect just start the same command again.

    python scripts/tune_transformer.py                  # all pending runs
    python scripts/tune_transformer.py --runs 2 3       # only these
    python scripts/tune_transformer.py --collect-only   # just rebuild the CSV

Writes results/transformer/tuning_results.csv. A best configuration is named
only once every run has finished.

The architecture is fixed throughout: 4 layers, 4 heads, d_model 256,
d_ff 1024, max length 256, 5 classes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import train_transformer as tt                                     # noqa: E402

SEED = 42

# Run 1 is the configs/transformer.yaml baseline; each other run changes one
# hyperparameter from it.
RUNS = {
    1: dict(lr=1e-4, dropout=0.1, batch_size=32, weight_decay=1e-4),
    2: dict(lr=5e-5, dropout=0.1, batch_size=32, weight_decay=1e-4),
    3: dict(lr=2e-4, dropout=0.1, batch_size=32, weight_decay=1e-4),
    4: dict(lr=1e-4, dropout=0.2, batch_size=32, weight_decay=1e-4),
    5: dict(lr=1e-4, dropout=0.3, batch_size=32, weight_decay=1e-4),
    6: dict(lr=1e-4, dropout=0.1, batch_size=16, weight_decay=1e-4),
    7: dict(lr=1e-4, dropout=0.1, batch_size=64, weight_decay=1e-4),
    8: dict(lr=1e-4, dropout=0.1, batch_size=32, weight_decay=1e-3),
}
CHANGED = {1: "baseline", 2: "lr", 3: "lr", 4: "dropout", 5: "dropout",
           6: "batch_size", 7: "batch_size", 8: "weight_decay"}

FIELDS = ["run", "varied", "lr", "dropout", "batch_size", "weight_decay",
          "status", "best_val_f1_macro", "val_accuracy", "val_precision_macro",
          "val_recall_macro", "best_epoch", "epochs_run", "stopped_early",
          "train_time_min", "mean_epoch_time_s", "device", "output_dir"]


def run_name(i: int, hp: dict) -> str:
    return (f"run{i}_lr{hp['lr']:g}_do{hp['dropout']:g}"
            f"_bs{hp['batch_size']}_wd{hp['weight_decay']:g}")


def run_dir(i: int, hp: dict, root: str) -> str:
    """Run 1 is the baseline, trained into results/transformer/seed_42."""
    if i == 1 and root == DEFAULT_ROOT:
        return os.path.join(root, f"seed_{SEED}")
    return os.path.join(root, "tuning", run_name(i, hp), f"seed_{SEED}")


DEFAULT_ROOT = os.path.join(ROOT, "results", "transformer")


def train_one(i: int, hp: dict, out: str, args) -> None:
    ns = argparse.Namespace(
        config=args.config, seed=SEED, lr=hp["lr"], batch_size=hp["batch_size"],
        dropout=hp["dropout"], weight_decay=hp["weight_decay"],
        epochs=args.epochs, label_smoothing=None, warmup_ratio=None, max_len=None,
        patience=None, run_name=None, out_dir=os.path.dirname(out),
        device=args.device, limit=args.limit, overwrite=False,
    )
    tt.train(ns)


def collect(root: str) -> list[dict]:
    rows = []
    for i, hp in RUNS.items():
        out = run_dir(i, hp, root)
        row = {"run": i, "varied": CHANGED[i], **hp,
               "output_dir": os.path.relpath(out, ROOT)}
        path = os.path.join(out, "metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                m = json.load(f)
            row.update(
                status="done" if not m.get("smoke_test_limit") else "smoke",
                best_val_f1_macro=round(m["f1_macro"], 4),
                val_accuracy=round(m["accuracy"], 4),
                val_precision_macro=round(m["precision_macro"], 4),
                val_recall_macro=round(m["recall_macro"], 4),
                best_epoch=m["best_epoch"],
                epochs_run=m["epochs_run"],
                stopped_early=m["stopped_early"],
                train_time_min=round(m["train_time_s"] / 60, 1),
                mean_epoch_time_s=m.get("mean_epoch_time_s"),
                device=m.get("device"),
            )
        else:
            partial = os.path.exists(os.path.join(out, "training_history.json"))
            row["status"] = "incomplete" if partial else "pending"
        rows.append(row)
    return rows


def write_csv(rows: list[dict], root: str) -> str:
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "tuning_results.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    return path


def summarise(rows: list[dict], path: str) -> None:
    print("\n" + "=" * 78)
    print(f"{'run':>3} {'varied':<12} {'lr':>7} {'drop':>5} {'bs':>3} {'wd':>7} "
          f"{'status':<10} {'val F1':>7} {'val acc':>7} {'epoch':>5} {'min':>6}")
    for r in rows:
        print(f"{r['run']:>3} {r['varied']:<12} {r['lr']:>7g} {r['dropout']:>5g} "
              f"{r['batch_size']:>3} {r['weight_decay']:>7g} {r['status']:<10} "
              f"{r.get('best_val_f1_macro', ''):>7} {r.get('val_accuracy', ''):>7} "
              f"{r.get('best_epoch', ''):>5} {r.get('train_time_min', ''):>6}")
    print(f"\nsaved -> {os.path.relpath(path, ROOT)}")

    done = [r for r in rows if r["status"] == "done"]
    if len(done) < len(rows):
        waiting = [r["run"] for r in rows if r["status"] != "done"]
        print(f"{len(done)}/{len(rows)} runs complete; still waiting on runs {waiting}. "
              "No best configuration is selected until all runs finish.")
        return
    devices = {r["device"] for r in done}
    best = max(done, key=lambda r: r["best_val_f1_macro"])
    print(f"BEST (validation macro F1): run {best['run']} -> "
          f"lr {best['lr']:g}, dropout {best['dropout']:g}, batch {best['batch_size']}, "
          f"weight decay {best['weight_decay']:g}: val F1 {best['best_val_f1_macro']}, "
          f"acc {best['val_accuracy']}, epoch {best['best_epoch']}")
    if len(devices) > 1:
        print(f"note: runs used different hardware {sorted(devices)}; "
              "training times are not comparable across them")
    with open(os.path.join(os.path.dirname(path), "best_config.json"), "w") as f:
        json.dump({"selected_on": "validation macro F1, seed 42",
                   "run": best["run"], "hyperparameters": RUNS[best["run"]],
                   "best_val_f1_macro": best["best_val_f1_macro"],
                   "val_accuracy": best["val_accuracy"],
                   "output_dir": best["output_dir"]}, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Member 4 Transformer tuning (validation only).")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "transformer.yaml"))
    ap.add_argument("--runs", type=int, nargs="*", help="run numbers (default: all)")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--device")
    ap.add_argument("--epochs", type=int, help="override epochs (default: config)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test only")
    ap.add_argument("--out-root", default=DEFAULT_ROOT,
                    help="smoke tests: write somewhere other than results/transformer")
    args = ap.parse_args()
    root = os.path.abspath(args.out_root)
    if args.limit and root == DEFAULT_ROOT:
        sys.exit("--limit is a smoke test: pass --out-root so it cannot mix with real results")

    todo = args.runs or list(RUNS)
    if not args.collect_only:
        for i in todo:
            hp = RUNS[i]
            out = run_dir(i, hp, root)
            if os.path.exists(os.path.join(out, "metrics.json")):
                print(f"run {i}: already done, skipping ({os.path.relpath(out, ROOT)})")
                continue
            if os.path.exists(os.path.join(out, "training_history.json")):
                # Training rewrites every file, so no deletion is needed. Make
                # sure no other process is still training into this folder.
                print(f"run {i}: found an unfinished run, restarting it from scratch")
            print(f"\n{'#' * 78}\nrun {i}/{len(RUNS)} ({CHANGED[i]}): {hp}\n{'#' * 78}")
            t0 = time.perf_counter()
            train_one(i, hp, out, args)
            print(f"run {i} finished in {(time.perf_counter() - t0) / 60:.1f} min")
            write_csv(collect(root), root)       # update after every run

    rows = collect(root)
    summarise(rows, write_csv(rows, root))


if __name__ == "__main__":
    main()
