"""
distilbert.py — fine-tune DistilBERT for complaint product classification.

Model 4 of 4. Owner: Dulanja.

Why DistilBERT: it arrives already knowing English from pretraining, so unlike
the other three models it does not have to learn word meaning from our 60k
complaints. It is a 6-layer distillation of BERT-base — about 40% smaller and
60% faster, retaining ~97% of BERT's performance — which makes it the only
pretrained transformer that fine-tunes in reasonable time on a free GPU.

Usage:
    python src/models/distilbert.py --seed 42
    python src/models/distilbert.py --seed 42 --test     # final run only
    python src/models/distilbert.py --lr 3e-5 --epochs 2 --batch-size 32

Tune with the validation set. Only pass --test once, at the very end.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# make `src` importable when run as a script from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data import load_splits, class_weights                      # noqa: E402
from src.evaluate import (                                            # noqa: E402
    count_params,
    evaluate_model,
    peak_gpu_mb,
    save_results,
)

MODEL_NAME = "distilbert-base-uncased"


# --------------------------------------------------------------------------
# reproducibility
# --------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():          # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

class ComplaintDataset(Dataset):
    """Tokenised complaints. Tokenisation happens up front, once."""

    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][i],
            "attention_mask": self.encodings["attention_mask"][i],
            "labels": self.labels[i],
        }


# --------------------------------------------------------------------------
# train / evaluate
# --------------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader, device):
    """Return (probabilities, true labels, mean loss) over a loader."""
    model.eval()
    probs, trues, losses = [], [], []
    loss_fn = nn.CrossEntropyLoss()

    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)

        logits = model(input_ids=ids, attention_mask=mask).logits
        losses.append(loss_fn(logits, y).item())
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        trues.append(y.cpu().numpy())

    return np.vstack(probs), np.concatenate(trues), float(np.mean(losses))


def train(args):
    from sklearn.metrics import f1_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    set_seed(args.seed)
    device = pick_device()
    print(f"device: {device}   seed: {args.seed}")

    # ---- data -----------------------------------------------------------
    tr, va, te, meta = load_splits(args.data)
    n_classes = meta["n_classes"]

    if args.limit:
        # Smoke test: run the whole pipeline on a few hundred rows to catch
        # errors in seconds instead of 40 minutes in. Never report these numbers.
        print(f"*** SMOKE TEST: {args.limit} rows per split — results are meaningless ***")
        tr, va, te = (d.head(args.limit) for d in (tr, va, te))

    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,} | "
          f"classes {n_classes}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    t0 = time.perf_counter()
    train_ds = ComplaintDataset(tr.text, tr.y, tokenizer, args.max_len)
    val_ds = ComplaintDataset(va.text, va.y, tokenizer, args.max_len)
    print(f"tokenised in {time.perf_counter() - t0:.1f}s")

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size * 2)

    # ---- model ----------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=n_classes
    ).to(device)

    # Class-weighted loss: the classes are imbalanced (1.6:1), so without
    # weighting the model under-predicts retail_banking and macro F1 suffers.
    weights = torch.tensor(class_weights(tr.y, n_classes)).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = len(train_dl) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimiser,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    use_amp = device.type == "cuda" and not args.no_amp
    # torch.amp.GradScaler on new versions, torch.cuda.amp on older ones.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("mixed precision: on")

    # ---- training loop --------------------------------------------------
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_f1, best_state, patience_left = -1.0, None, args.patience
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_start = time.perf_counter()

        for step, batch in enumerate(train_dl, 1):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)

            optimiser.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(input_ids=ids, attention_mask=mask).logits
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()

            epoch_losses.append(loss.item())
            if step % 100 == 0 or step == len(train_dl):
                print(f"\r  epoch {epoch} step {step}/{len(train_dl)} "
                      f"loss {np.mean(epoch_losses[-100:]):.4f}", end="", flush=True)
        print()

        val_probs, val_true, val_loss = predict(model, val_dl, device)
        val_f1 = f1_score(val_true, val_probs.argmax(1), average="macro",
                          zero_division=0)

        history["train_loss"].append(float(np.mean(epoch_losses)))
        history["val_loss"].append(val_loss)
        history["val_f1"].append(float(val_f1))

        print(f"  epoch {epoch}: train loss {np.mean(epoch_losses):.4f} | "
              f"val loss {val_loss:.4f} | val macro F1 {val_f1:.4f} | "
              f"{time.perf_counter() - epoch_start:.0f}s")

        # Early stopping on validation macro F1, never on the test set.
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  early stop: no improvement for {args.patience} epochs")
                break

    train_time = time.perf_counter() - train_start
    print(f"training finished in {train_time/60:.1f} min, "
          f"best val macro F1 {best_f1:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- final evaluation ----------------------------------------------
    # The test set is opened ONCE, only with --test, only for the final run.
    if args.test:
        print("\nEvaluating on the TEST set (final, once only)")
        eval_df, split_name = te, "test"
    else:
        print("\nEvaluating on the VALIDATION set")
        eval_df, split_name = va, "val"

    eval_ds = ComplaintDataset(eval_df.text, eval_df.y, tokenizer, args.max_len)
    eval_dl = DataLoader(eval_ds, batch_size=args.batch_size * 2)

    probs, trues, _ = predict(model, eval_dl, device)
    preds = probs.argmax(1)

    res = evaluate_model(trues, preds, probs, meta["classes"], split=split_name)
    res["params"] = count_params(model)
    res["train_time_s"] = round(train_time, 1)
    res["best_val_f1"] = round(float(best_f1), 4)
    res["epochs_run"] = len(history["train_loss"])
    res["peak_gpu_mb"] = peak_gpu_mb()

    # Inference timing: same 1,000 examples for every model, so it is comparable.
    timing_ds = ComplaintDataset(eval_df.text[:1000], eval_df.y[:1000],
                                 tokenizer, args.max_len)
    timing_dl = DataLoader(timing_ds, batch_size=args.batch_size * 2)
    predict(model, timing_dl, device)                     # warm up
    t0 = time.perf_counter()
    predict(model, timing_dl, device)
    res["inference_ms_per_1k"] = round((time.perf_counter() - t0) * 1000, 1)

    config = {
        "model_name": MODEL_NAME,
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "patience": args.patience,
        "class_weighted_loss": True,
        "mixed_precision": use_amp,
        "device": str(device),
    }

    name = "distilbert" if args.test else "distilbert_val"
    save_results(name, res, history=history, config=config, seed=args.seed)

    print(f"\naccuracy {res['accuracy']:.4f} | macro F1 {res['f1_macro']:.4f} | "
          f"params {res['params']['trainable']/1e6:.1f}M | "
          f"{res['inference_ms_per_1k']:.0f} ms / 1k")

    if args.save_model:
        os.makedirs("checkpoints", exist_ok=True)
        path = f"checkpoints/distilbert_seed{args.seed}"
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        print(f"model saved -> {path}")

    return res


def main():
    ap = argparse.ArgumentParser(description="Fine-tune DistilBERT.")
    ap.add_argument("--data", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-len", type=int, default=224,
                    help="p90 of the training set is 194")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true",
                    help="disable mixed precision")
    ap.add_argument("--save-model", action="store_true",
                    help="write the fine-tuned weights (~260 MB)")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test on N rows per split (e.g. 200)")
    ap.add_argument("--test", action="store_true",
                    help="FINAL RUN ONLY: evaluate on the test set")
    args = ap.parse_args()

    if args.test:
        print("=" * 62)
        print("  TEST SET RUN — only do this once, for the final result.")
        print("=" * 62)

    train(args)


if __name__ == "__main__":
    main()
