"""
train_transformer.py — train Member 4's from-scratch Transformer encoder.

Reads configs/transformer.yaml, trains on the shared train split, selects the
best epoch on validation macro F1, and writes everything to

    results/transformer/seed_<seed>/                 (config as in the YAML)
    results/transformer/<run-name>/seed_<seed>/      (tuning runs, overrides)

        best_model.pt           weights + vocab + model config (gitignored)
        training_history.json   per-epoch losses, metrics, lr, timing
        config_used.yaml        the fully resolved configuration of this run
        metrics.json            validation metrics of the best epoch

The TEST SET IS NEVER USED here: only train and val are encoded, and model
selection uses validation macro F1 only.

Usage (from the project root):
    python scripts/train_transformer.py                         # config as is
    python scripts/train_transformer.py --seed 1
    python scripts/train_transformer.py --lr 3e-4 --dropout 0.2 # tuning run
    python scripts/train_transformer.py --limit 256 --epochs 2 --out-dir <tmp>
                                                                 # smoke test
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import random
import subprocess
import sys
import time

# Deterministic cuBLAS; must be set before CUDA initialises.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data import class_weights                                   # noqa: E402
from src.evaluate import count_params, evaluate_model, peak_gpu_mb   # noqa: E402
from src.models.transformer import (                                 # noqa: E402
    TransformerClassifier,
    make_loader,
    padding_mask,
    prepare_data,
)

# Training settings the YAML may leave out. They are written to
# config_used.yaml, so every run records exactly what it used.
TRAINING_DEFAULTS = {
    "warmup_ratio": 0.1,      # linear warmup over the first 10% of steps
    "scheduler": "cosine",    # then cosine decay to 0; "constant" disables
    "grad_clip": 1.0,
    "label_smoothing": 0.0,
    "class_weighted_loss": True,
    "mixed_precision": True,  # CUDA only
    "betas": [0.9, 0.98],
    "log_every": 100,         # steps between progress lines
}


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Seed every RNG and make cuDNN/cuBLAS deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str, args) -> tuple[dict, dict]:
    """YAML config, then defaults for missing training keys, then CLI overrides."""
    with open(path, encoding="utf8") as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    tr = cfg.setdefault("training", {})
    for k, v in TRAINING_DEFAULTS.items():
        tr.setdefault(k, v)
    tr.setdefault("early_stopping", {}).setdefault("patience", 3)

    overrides = {
        ("training", "seed"): args.seed,
        ("training", "learning_rate"): args.lr,
        ("training", "batch_size"): args.batch_size,
        ("training", "epochs"): args.epochs,
        ("training", "weight_decay"): args.weight_decay,
        ("training", "label_smoothing"): args.label_smoothing,
        ("training", "warmup_ratio"): args.warmup_ratio,
        ("model", "dropout"): args.dropout,
        ("model", "max_length"): args.max_len,
    }
    changed = {}
    for (sec, key), val in overrides.items():
        if val is not None:
            cfg[sec][key] = val
            if key != "seed":
                changed[key] = val
    if args.patience is not None:
        tr["early_stopping"]["patience"] = args.patience
        changed["patience"] = args.patience
    if args.dropout is not None and "attn_dropout" in cfg["model"]:
        cfg["model"]["attn_dropout"] = args.dropout      # keep the two in step
    return cfg, changed


def run_dir(cfg: dict, changed: dict, args) -> str:
    """results/transformer/[<run-name>/]seed_<seed>. Tuning runs get their own
    folder named after the overridden hyperparameters."""
    seed = cfg["training"]["seed"]
    if args.out_dir:
        base = args.out_dir
    else:
        base = os.path.join(ROOT, "results", "transformer")
        name = args.run_name
        if name is None and changed:
            name = "_".join(f"{k}{v:g}" if isinstance(v, (int, float)) else f"{k}{v}"
                            for k, v in changed.items())
        if name:
            base = os.path.join(base, name)
    return os.path.join(base, f"seed_{seed}")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:                                    # noqa: BLE001
        return None


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, betas) -> torch.optim.AdamW:
    """AdamW. No weight decay on biases, LayerNorm weights or the embedding:
    decaying those only shrinks scales and rare-word vectors."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() < 2 or name.startswith("embedding."):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=tuple(betas),
    )


def build_scheduler(opt, total_steps: int, warmup_ratio: float, kind: str):
    """Linear warmup, then cosine decay to 0 (or constant)."""
    warmup = int(total_steps * warmup_ratio)

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        if kind == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(opt, factor)


def make_scaler(enabled: bool):
    # torch.amp.GradScaler on new versions, torch.cuda.amp on older ones.
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def run_eval(model, loader, loss_fn, device, use_amp: bool):
    """Mean loss, probabilities and labels over a loader."""
    model.eval()
    total_loss, n = 0.0, 0
    probs, trues = [], []
    for ids, y in loader:
        ids, y = ids.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(ids, padding_mask(ids))
        logits = logits.float()
        total_loss += loss_fn(logits, y).item() * len(y)
        n += len(y)
        probs.append(torch.softmax(logits, -1).cpu().numpy())
        trues.append(y.cpu().numpy())
    return total_loss / max(n, 1), np.vstack(probs), np.concatenate(trues)


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train(args) -> dict:
    cfg, changed = load_config(args.config, args)
    m_cfg, t_cfg, d_cfg = cfg["model"], cfg["training"], cfg.get("data", {})
    seed = int(t_cfg["seed"])

    out_dir = run_dir(cfg, changed, args)
    if os.path.exists(os.path.join(out_dir, "metrics.json")) and not args.overwrite:
        sys.exit(f"{out_dir} already holds a finished run. Use --overwrite, "
                 f"--run-name or --out-dir to keep both.")
    os.makedirs(out_dir, exist_ok=True)

    set_seed(seed)
    device = pick_device(args.device)
    use_amp = device.type == "cuda" and bool(t_cfg["mixed_precision"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "cpu"
    print(f"device: {device} ({dev_name})   seed: {seed}   mixed precision: {use_amp}")
    print(f"output: {out_dir}")
    if changed:
        print(f"overrides: {changed}")

    # ---- data: shared pipeline, train + val only ------------------------
    max_len = int(m_cfg["max_length"])
    data_dir = d_cfg.get("data_dir", os.path.join(ROOT, "data"))
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(ROOT, data_dir)
    if args.limit:
        print(f"*** SMOKE TEST: {args.limit} rows per split - numbers are meaningless ***")
    data = prepare_data(data_dir, max_len=max_len, max_vocab=int(m_cfg["vocab_size"]),
                        min_freq=int(d_cfg.get("min_freq", 2)),
                        splits=("train", "val"), limit=args.limit)
    assert "test" not in data["datasets"], "the test split must not be encoded here"
    vocab, meta = data["vocab"], data["meta"]
    n_classes = int(meta["n_classes"])
    if n_classes != int(m_cfg["num_classes"]):
        sys.exit(f"config num_classes {m_cfg['num_classes']} != data {n_classes}")
    train_ds, val_ds = data["datasets"]["train"], data["datasets"]["val"]
    for split, s in data["stats"].items():
        print(f"{split}: {s['rows']:,} rows | truncated {s['truncated_pct']}% | "
              f"<unk> {s['unk_pct']}% | mean tokens {s['mean_real_tokens']}")

    bs = int(t_cfg["batch_size"])
    train_dl = make_loader(train_ds, bs, shuffle=True, seed=seed)
    val_dl = make_loader(val_ds, bs * 2, shuffle=False)

    # ---- model ------------------------------------------------------------
    model = TransformerClassifier(
        vocab_size=len(vocab),
        n_classes=n_classes,
        d_model=int(m_cfg["d_model"]),
        n_heads=int(m_cfg["n_heads"]),
        n_layers=int(m_cfg["num_layers"]),
        d_ff=int(m_cfg["dim_feedforward"]),
        dropout=float(m_cfg["dropout"]),
        attn_dropout=m_cfg.get("attn_dropout"),
        max_len=max_len,
        activation=m_cfg.get("activation", "gelu"),
        norm_first=bool(m_cfg.get("norm_first", True)),
    ).to(device)
    params = count_params(model)
    print(f"model: {params['trainable']:,} trainable parameters "
          f"(vocab {len(vocab):,}, {len(model.layers)} layers, "
          f"{model.layers[0].attn.n_heads} heads, d_model {model.d_model})")

    # Class-weighted loss from TRAIN labels only; the same loss is used for
    # validation so the two loss curves are directly comparable.
    y_train = train_ds.tensors[1].numpy()
    weight = (torch.tensor(class_weights(y_train, n_classes)).to(device)
              if t_cfg["class_weighted_loss"] else None)
    loss_fn = nn.CrossEntropyLoss(weight=weight,
                                  label_smoothing=float(t_cfg["label_smoothing"]))

    epochs = int(t_cfg["epochs"])
    optimizer = build_optimizer(model, float(t_cfg["learning_rate"]),
                                float(t_cfg["weight_decay"]), t_cfg["betas"])
    scheduler = build_scheduler(optimizer, len(train_dl) * epochs,
                                float(t_cfg["warmup_ratio"]), t_cfg["scheduler"])
    scaler = make_scaler(use_amp)
    patience = int(t_cfg["early_stopping"]["patience"])

    # ---- record the resolved config before training ----------------------
    cfg_used = copy.deepcopy(cfg)
    cfg_used["run"] = {
        "output_dir": os.path.relpath(out_dir, ROOT),
        "overrides": changed,
        "smoke_test_limit": args.limit or None,
        "vocab_size_actual": len(vocab),
        "params_trainable": params["trainable"],
        "device": f"{device} ({dev_name})",
        "torch": str(torch.__version__),
        "python": platform.python_version(),
        "git_commit": git_commit(),
        "class_weights": None if weight is None else [round(float(w), 4) for w in weight],
    }
    with open(os.path.join(out_dir, "config_used.yaml"), "w", encoding="utf8") as f:
        yaml.safe_dump(cfg_used, f, sort_keys=False)

    # ---- training loop --------------------------------------------------
    history = {"epochs": [], "train_loss": [], "val_loss": [], "val_f1": [],
               "val_accuracy": [], "epoch_time_s": []}
    best = {"f1": -1.0, "epoch": 0, "metrics": None}
    ckpt_path = os.path.join(out_dir, "best_model.pt")
    patience_left = patience
    log_every = int(t_cfg["log_every"])
    stopped_early = False
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        run_loss, run_correct, run_n = 0.0, 0, 0

        for step, (ids, y) in enumerate(train_dl, 1):
            ids, y = ids.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(ids, padding_mask(ids))
                loss = loss_fn(logits.float(), y)
            if not torch.isfinite(loss):
                sys.exit(f"non-finite loss at epoch {epoch} step {step}: {loss.item()}")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(t_cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            run_loss += loss.item() * len(y)
            run_correct += (logits.argmax(-1) == y).sum().item()
            run_n += len(y)
            if step % log_every == 0 or step == len(train_dl):
                elapsed = time.perf_counter() - epoch_start
                eta = elapsed / step * (len(train_dl) - step)
                print(f"\r  epoch {epoch}/{epochs} step {step}/{len(train_dl)} "
                      f"loss {run_loss / run_n:.4f} acc {run_correct / run_n:.4f} "
                      f"lr {scheduler.get_last_lr()[0]:.2e} | {elapsed:.0f}s, "
                      f"eta {eta:.0f}s   ", end="", flush=True)
        print()
        train_time_epoch = time.perf_counter() - epoch_start

        # ---- validation (never test) -------------------------------------
        val_start = time.perf_counter()
        val_loss, probs, trues = run_eval(model, val_dl, loss_fn, device, use_amp)
        res = evaluate_model(trues, probs.argmax(1), probs, meta["classes"], split="val")
        val_time = time.perf_counter() - val_start
        epoch_time = time.perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": run_loss / run_n,
            "train_accuracy": run_correct / run_n,
            "val_loss": val_loss,
            "val_accuracy": res["accuracy"],
            "val_precision_macro": res["precision_macro"],
            "val_recall_macro": res["recall_macro"],
            "val_f1_macro": res["f1_macro"],
            "val_roc_auc_ovr_macro": res.get("roc_auc_ovr_macro"),
            "lr_end": scheduler.get_last_lr()[0],
            "train_time_s": round(train_time_epoch, 2),
            "val_time_s": round(val_time, 2),
            "epoch_time_s": round(epoch_time, 2),
        }
        history["epochs"].append(row)
        # flat lists in the shape src.evaluate.plot_curves expects
        history["train_loss"].append(row["train_loss"])
        history["val_loss"].append(val_loss)
        history["val_f1"].append(res["f1_macro"])
        history["val_accuracy"].append(res["accuracy"])
        history["epoch_time_s"].append(row["epoch_time_s"])

        improved = res["f1_macro"] > best["f1"]
        print(f"  epoch {epoch}: train loss {row['train_loss']:.4f} acc {row['train_accuracy']:.4f} | "
              f"val loss {val_loss:.4f} acc {res['accuracy']:.4f} "
              f"P {res['precision_macro']:.4f} R {res['recall_macro']:.4f} "
              f"F1 {res['f1_macro']:.4f} | {epoch_time:.0f}s"
              f"{'  * best' if improved else ''}")

        # Model selection on validation macro F1 only.
        if improved:
            best = {"f1": res["f1_macro"], "epoch": epoch, "metrics": res}
            torch.save({
                "model_state": model.state_dict(),
                "model_config": model.config,
                "vocab": vocab,
                "max_len": max_len,
                "classes": meta["classes"],
                "epoch": epoch,
                "val_f1_macro": res["f1_macro"],
                "val_accuracy": res["accuracy"],
                "seed": seed,
                "config": cfg_used,
            }, ckpt_path)
            patience_left = patience
        elif patience:
            patience_left -= 1
            if patience_left == 0:
                print(f"  early stop: no val macro F1 improvement for {patience} epochs")
                stopped_early = True

        with open(os.path.join(out_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)       # rewritten each epoch
        if stopped_early:
            break

    total_time = time.perf_counter() - train_start

    # ---- metrics of the selected (best-val) epoch -------------------------
    metrics = {
        "model": "transformer",
        "seed": seed,
        "selection": "best validation macro F1",
        "best_epoch": best["epoch"],
        "epochs_run": len(history["epochs"]),
        "epochs_max": epochs,
        "stopped_early": stopped_early,
        "train_time_s": round(total_time, 1),
        "mean_epoch_time_s": round(float(np.mean(history["epoch_time_s"])), 1),
        "params": params,
        "peak_gpu_mb": peak_gpu_mb(),
        "device": f"{device} ({dev_name})",
        "smoke_test_limit": args.limit or None,
        **best["metrics"],
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 62)
    print(f"best validation macro F1 : {best['f1']:.4f}  (epoch {best['epoch']})")
    print(f"best validation accuracy : {best['metrics']['accuracy']:.4f}")
    print(f"total training time      : {total_time / 60:.1f} min ({total_time:.0f}s)")
    print(f"trainable parameters     : {params['trainable']:,}")
    shown = os.path.relpath(ckpt_path, ROOT) if ckpt_path.startswith(ROOT) else ckpt_path
    print(f"checkpoint               : {shown}")
    print("=" * 62)
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Train the from-scratch Transformer (Member 4).")
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "transformer.yaml"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--dropout", type=float)
    ap.add_argument("--weight-decay", type=float)
    ap.add_argument("--label-smoothing", type=float)
    ap.add_argument("--warmup-ratio", type=float)
    ap.add_argument("--max-len", type=int)
    ap.add_argument("--patience", type=int, help="0 disables early stopping")
    ap.add_argument("--run-name", help="subfolder under results/transformer/")
    ap.add_argument("--out-dir", help="write here instead of results/transformer/")
    ap.add_argument("--device", help="force a device, e.g. cpu")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test on N rows per split; never report these numbers")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace a finished run in the same folder")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
