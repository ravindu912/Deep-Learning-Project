"""Member 3 training helpers: train and validate the BiLSTM with attention.

Only validation macro F1 selects the best checkpoint. No test evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import time
from datetime import datetime, timezone

# Allow running `python scripts/train_bilstm_attn.py` from the project root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

from src.data import build_vocab, class_weights, encode, load_glove
from src.evaluate import count_params, evaluate_model, save_results
from src.models.bilstm_attn import BiLSTMAttention


def project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_train_val(data_dir: Path):
    """Read unchanged shared files, without ever opening the reserved test split.

    Member 3 exception: src.data.load_splits always loads test.csv too.
    All vocabulary, encoding and class-weight logic still uses src.data.
    """
    with (data_dir / "split_meta.json").open(encoding="utf-8") as handle:
        meta = json.load(handle)
    if len(meta["classes"]) != meta["n_classes"]:
        raise ValueError("Metadata class count does not match its class names")
    expected_ids = set(range(meta["n_classes"]))
    parts = []
    for name in ("train", "val"):
        frame = pd.read_csv(data_dir / f"{name}.csv")
        if not {"text", "y"}.issubset(frame.columns):
            raise ValueError(f"{name}.csv must contain text and y columns")
        if frame.empty or frame[["text", "y"]].isna().any().any():
            raise ValueError(f"{name}.csv is empty or has missing text/labels; data was not changed")
        if len(frame) != meta["sizes"][name]:
            raise ValueError(f"{name}.csv row count differs from shared metadata; data was not changed")
        if not pd.api.types.is_integer_dtype(frame.y) or set(frame.y.unique()) != expected_ids:
            raise ValueError(f"{name}.csv must contain integer labels for every metadata class")
        parts.append(frame)
    return parts[0], parts[1], meta


def read_config(path: Path, seed: int | None = None) -> dict:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if seed is not None:
        cfg["seed"] = seed
    # Keep the team's common settings and the implemented architecture fixed.
    required = {"max_len": 224, "patience": 3, "num_layers": 2, "bidirectional": True}
    for key, value in required.items():
        if cfg.get(key) != value:
            raise ValueError(f"{key} must be {value}")
    for key in ("batch_size", "max_epochs", "embedding_dim", "hidden_size", "vocab_min_freq"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["vocab_max_size"] < 2 or cfg["num_workers"] < 0:
        raise ValueError("Vocabulary needs at least two IDs; num_workers cannot be negative")
    if not 0 <= cfg["dropout"] < 1 or cfg["min_delta"] < 0:
        raise ValueError("Invalid dropout or min_delta")
    if cfg["learning_rate"] <= 0 or cfg["gradient_clip_norm"] <= 0:
        raise ValueError("learning_rate and gradient_clip_norm must be positive")
    return cfg


def write_json(path: Path, payload: dict) -> None:
    # Replace a finished temporary file so interruption cannot leave half a JSON.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def seed_everything(seed: int, deterministic: bool) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


def run_epoch(model, loader, criterion, device, gradient_clip_norm, optimizer=None, max_batches=None,
              progress_path=None, epoch=None):
    """One pass; an optimizer is supplied only for the training pass."""
    training = optimizer is not None
    model.train(training)
    loss_sum, weight_sum = 0.0, 0.0
    probabilities, labels = [], []
    with torch.set_grad_enabled(training):
        for batch_index, (words, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            words, targets = words.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, _ = model(words)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; stopping without saving invalid results")
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm, error_if_nonfinite=True)
                optimizer.step()
            # Weighted mean loss is divided by the sum of target-class weights.
            # This also gives the right epoch average for a short final batch.
            batch_weight = criterion.weight[targets].sum().item()
            loss_sum += loss.item() * batch_weight
            weight_sum += batch_weight
            if not training:
                probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
                labels.append(targets.cpu().numpy())
            total_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
            if progress_path is not None and ((batch_index + 1) % 50 == 0 or batch_index + 1 == total_batches):
                write_json(progress_path, {
                    "epoch": epoch, "stage": "train" if training else "val",
                    "batch": batch_index + 1, "total_batches": total_batches,
                    "updated_utc": datetime.now(timezone.utc).isoformat(),
                })
    if training:
        return loss_sum / weight_sum, None, None
    return loss_sum / weight_sum, np.concatenate(labels), np.concatenate(probabilities)


def train_model(train, val, meta: dict, cfg: dict, output_root: Path = ROOT) -> Path:
    """Train on supplied shared splits; this function never loads split files."""
    cfg = dict(cfg)
    torch.set_num_threads(cfg.get("torch_num_threads", 4))
    cfg["torch_num_threads"] = torch.get_num_threads()
    cfg.setdefault("experiment_kind", "full_training")
    if cfg["experiment_kind"] != "short_trial" and (
        cfg.get("max_train_batches") is not None or cfg.get("max_val_batches") is not None
    ):
        raise ValueError("Batch limits are allowed only for a short_trial, never a tuning run")
    seed_everything(cfg["seed"], cfg["deterministic"])
    requested_device = cfg["device"]
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if requested_device == "auto" else requested_device
    )

    # Build words and loss weights using training data only.
    vocab = build_vocab(train.text, max_size=cfg["vocab_max_size"], min_freq=cfg["vocab_min_freq"])
    weights = class_weights(train.y, meta["n_classes"])
    datasets = []
    for frame in (train, val):
        inputs = encode(frame.text, vocab, max_len=cfg["max_len"])
        datasets.append(TensorDataset(
            torch.from_numpy(inputs), torch.as_tensor(frame.y.to_numpy(copy=True), dtype=torch.long)
        ))
    generator = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(
        datasets[0], batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], generator=generator,
    )
    val_loader = DataLoader(
        datasets[1], batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"],
    )

    matrix = None
    if cfg["glove_path"]:
        matrix = load_glove(str(project_path(cfg["glove_path"])), vocab, dim=cfg["embedding_dim"])
    model = BiLSTMAttention(
        len(vocab), meta["n_classes"], embedding_dim=cfg["embedding_dim"],
        hidden_size=cfg["hidden_size"], dropout=cfg["dropout"],
        embedding_matrix=matrix, freeze_embeddings=cfg["freeze_embeddings"],
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
        betas=tuple(cfg["adam_betas"]), eps=cfg["adam_eps"],
        amsgrad=False, foreach=False, fused=False,
    )

    # A unique directory keeps every configuration and seed from overwriting others.
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"_seed{cfg['seed']}"
    phase = "trials" if cfg["experiment_kind"] == "short_trial" else "validation"
    result_dir = output_root / "results" / "bilstm_attn" / phase / run_id
    checkpoint_dir = output_root / "checkpoints" / "bilstm_attn" / run_id
    result_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = checkpoint_dir / "best.pt"
    cfg.update({
        "run_id": run_id, "resolved_device": str(device), "n_classes": meta["n_classes"],
        "classes": meta["classes"], "vocab_size": len(vocab), "padding_idx": 0,
        "class_weights": weights.tolist(), "train_rows": len(train), "val_rows": len(val),
        "optimizer": "AdamW", "amsgrad": False, "foreach": False, "fused": False,
        "loss": "class_weighted_cross_entropy", "loss_reduction": "mean",
        "label_smoothing": 0.0, "selection_metric": "val_f1_macro",
        "shuffle_train": True, "shuffle_val": False, "drop_last": False,
        "pin_memory": False, "scheduler": None, "mixed_precision": False,
        "glove_initialization_seed": 42 if matrix is not None else None,
        "checkpoint_path": str(checkpoint_path), "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "numpy_version": np.__version__, "python_version": sys.version,
        "cudnn_benchmark": False, "cublas_workspace_config": ":4096:8",
    })
    write_json(result_dir / "config.json", cfg)
    print(f"Device: {device} | train: {len(train)} | val: {len(val)} | classes: {meta['n_classes']}")
    print(f"Run directory: {result_dir}", flush=True)
    if cfg["experiment_kind"] == "short_trial":
        print(f"SHORT TRIAL ONLY: train batches={cfg['max_train_batches']}, "
              f"validation batches={cfg['max_val_batches']}; excluded from tuning selection.", flush=True)

    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1, best_epoch, stale_epochs = -float("inf"), 0, 0
    best_metrics = None
    started = time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        train_loss, _, _ = run_epoch(
            model, train_loader, criterion, device, cfg["gradient_clip_norm"], optimizer,
            max_batches=cfg.get("max_train_batches"),
            progress_path=result_dir / "progress.json", epoch=epoch,
        )
        val_loss, y_true, y_proba = run_epoch(
            model, val_loader, criterion, device, cfg["gradient_clip_norm"],
            max_batches=cfg.get("max_val_batches"),
            progress_path=result_dir / "progress.json", epoch=epoch,
        )
        metrics = evaluate_model(
            y_true, y_proba.argmax(axis=1), y_proba, meta["classes"], split="val"
        )
        f1 = metrics["f1_macro"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(f1)
        write_json(result_dir / "history.json", history)
        improved = f1 > best_f1 + cfg["min_delta"]
        if improved:
            best_f1, best_epoch, stale_epochs = f1, epoch, 0
            best_metrics = metrics
            # Save vocabulary with weights so later predictions use identical IDs.
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save({
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": cfg, "vocab": vocab, "meta": meta,
                "epoch": epoch, "val_f1_macro": f1,
            }, temporary)
            temporary.replace(checkpoint_path)
        else:
            stale_epochs += 1
        with (result_dir / "epochs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_f1": f1, "best_epoch": best_epoch, "improved": improved,
            }) + "\n")
        print(
            f"Epoch {epoch:02d}/{cfg['max_epochs']} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_macro_f1={f1:.4f} | "
            + ("saved best checkpoint" if improved else f"patience {stale_epochs}/{cfg['patience']}"),
            flush=True,
        )
        # Store best-epoch metrics alongside the complete learning history so far.
        save_results(
            "bilstm_attn", {**best_metrics, "best_epoch": best_epoch,
                            "params": count_params(model),
                            "train_time_s": time.perf_counter() - started},
            history=history, config=cfg, seed=cfg["seed"], out_dir=str(result_dir),
        )
        if stale_epochs >= cfg["patience"]:
            print(f"Early stopping: no validation macro F1 improvement for {cfg['patience']} epochs.")
            break
    print(f"Best validation macro F1: {best_f1:.4f} at epoch {best_epoch}")
    print(f"Best checkpoint: {checkpoint_path}")
    return result_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bilstm_attn.yaml")
    parser.add_argument("--seed", type=int, help="Override the YAML seed; saved in the run settings")
    parser.add_argument("--trial", action="store_true", help="One short execution trial, excluded from tuning")
    args = parser.parse_args()
    config_path = project_path(args.config)
    cfg = read_config(config_path, args.seed)
    if args.trial:
        cfg.update(experiment_kind="short_trial", max_epochs=1, max_train_batches=2, max_val_batches=4)
    cfg["config_source"] = str(config_path.resolve())
    cfg["data_dir"] = str(project_path(cfg["data_dir"]).resolve())
    if cfg["glove_path"]:
        cfg["glove_path"] = str(project_path(cfg["glove_path"]).resolve())
    train, val, meta = load_train_val(Path(cfg["data_dir"]))
    train_model(train, val, meta, cfg)


if __name__ == "__main__":
    main()
