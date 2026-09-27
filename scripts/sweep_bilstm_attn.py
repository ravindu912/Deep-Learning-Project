"""Member 3: six full-data experiments, selected only by validation macro F1.

Completed runs are reused when the command is restarted. An interrupted
configuration restarts from its seed, not from a partial best checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from scripts.train_bilstm_attn import load_train_val, project_path, read_config, train_model, write_json


CANDIDATES = [
    ("01_baseline", {}),
    ("02_lower_lr", {"learning_rate": 0.0003}),
    ("03_higher_lr", {"learning_rate": 0.003}),
    ("04_more_dropout", {"dropout": 0.5}),
    ("05_smaller_lstm", {"hidden_size": 64}),
    ("06_larger_batch", {"batch_size": 128}),
]


def save_table(path: Path, rows: list[dict]) -> None:
    fields = ["name", "status", "learning_rate", "dropout", "hidden_size", "batch_size",
              "seed", "max_epochs", "validation_macro_f1", "best_epoch", "result_dir", "error"]
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def prepare(config_dir: Path) -> None:
    base = read_config(ROOT / "configs/bilstm_attn.yaml", seed=42)
    for name, changes in CANDIDATES:
        path = config_dir / f"{name}.yaml"
        if path.exists():
            continue
        cfg = {**base, **changes, "experiment_kind": "full_training", "experiment_name": name}
        cfg.pop("max_train_batches", None)
        cfg.pop("max_val_batches", None)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Write all configs and pending rows, without training")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    config_dir = ROOT / "configs/bilstm_attn_sweep"
    result_dir = ROOT / "results/bilstm_attn/sweep"
    config_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    prepare(config_dir)
    ledger_path = result_dir / "experiments.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {}
    configs = {}
    for name, _ in CANDIDATES:
        path = config_dir / f"{name}.yaml"
        cfg = read_config(path)
        if cfg.get("experiment_kind") != "full_training" or cfg.get("max_train_batches") or cfg.get("max_val_batches"):
            raise ValueError("Sweep candidates must use complete train and validation splits")
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in ledger and ledger[name]["config_sha256"] != fingerprint:
            raise ValueError(f"{name} changed after recording; use a separate sweep directory")
        configs[name] = cfg
        ledger.setdefault(name, {
            "name": name, "status": "pending", "config_sha256": fingerprint,
            **{key: cfg[key] for key in ("learning_rate", "dropout", "hidden_size", "batch_size", "seed", "max_epochs")},
            "validation_macro_f1": None, "best_epoch": None, "result_dir": None, "error": None,
        })

    def persist():
        write_json(ledger_path, ledger)
        save_table(result_dir / "experiments.csv", list(ledger.values()))

    persist()
    if args.prepare_only:
        print(f"Prepared six configurations: {config_dir}")
        print(f"Status table: {result_dir / 'experiments.csv'}")
        return
    train, val, meta = load_train_val(project_path(next(iter(configs.values()))["data_dir"]))
    for name, cfg in configs.items():
        if ledger[name]["status"] == "completed":
            print(f"Reusing completed experiment: {name}", flush=True)
            continue
        print(f"\nStarting full-data experiment: {name}", flush=True)
        ledger[name].update(status="running", error=None)
        persist()
        try:
            cfg = {**cfg, "device": args.device, "config_source": str(config_dir / f"{name}.yaml")}
            output = train_model(train, val, meta, cfg)
            result = json.loads((output / f"bilstm_attn_seed{cfg['seed']}.json").read_text(encoding="utf-8"))
            ledger[name].update(
                status="completed", validation_macro_f1=result["f1_macro"],
                best_epoch=result["best_epoch"], result_dir=str(output),
            )
        except BaseException as error:
            ledger[name].update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                error=f"{type(error).__name__}: {error}")
            persist()
            raise
        persist()

    # Trials and unfinished runs can never supply the selected configuration.
    completed = [row for row in ledger.values() if row["status"] == "completed"]
    if len(completed) != len(CANDIDATES):
        raise RuntimeError("All six configurations must complete before choosing the best")
    best = max(completed, key=lambda row: row["validation_macro_f1"])
    best_cfg = configs[best["name"]]
    (ROOT / "configs/bilstm_attn.yaml").write_text(
        "# Selected using full validation macro F1, seed 42; test data was never used.\n"
        + yaml.safe_dump(best_cfg, sort_keys=False), encoding="utf-8",
    )
    write_json(result_dir / "best.json", {**best, "selection_split": "val", "completed_configurations": len(completed)})
    print(f"Best configuration: {best['name']} | validation macro F1={best['validation_macro_f1']:.6f}")


if __name__ == "__main__":
    main()
