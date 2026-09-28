"""
benchmark_transformer.py — inference cost of Member 4's trained Transformer.

For each final checkpoint (results/transformer/final/seed_<N>/best_model.pt):

  1. reload check: predict the full VALIDATION set and confirm macro F1
     matches the value recorded at training time
  2. inference time per 1,000 complaints, with the shared
     src.evaluate.time_inference (same 1,000 validation complaints, 2 warm-up
     passes), repeated and reported as the median; fp32 is the headline,
     fp16 autocast is recorded as an extra on CUDA
  3. peak GPU memory during inference

Only data/val.csv is read. The test set is never opened.

    python scripts/benchmark_transformer.py              # on the Colab T4
    python scripts/benchmark_transformer.py --device cpu --repeats 1 --val-limit 500

Writes results/transformer/final/seed_<N>/benchmark.json and the combined
results/transformer/efficiency.json (training cost + inference cost).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.data import encode                                          # noqa: E402
from src.evaluate import evaluate_model, time_inference              # noqa: E402
from src.models.transformer import TransformerClassifier, padding_mask   # noqa: E402

RES = os.path.join(ROOT, "results", "transformer")
SEEDS = [42, 1, 2]


def load_model(path: str, device: torch.device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = TransformerClassifier(**ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


def make_predict_fn(model, device: torch.device, batch_size: int, fp16: bool):
    """numpy ids (n, T) -> numpy probabilities (n, C). Includes the host-to-
    device copy and returns to the CPU, which also synchronises CUDA, so the
    timer measures the whole prediction."""
    use_amp = fp16 and device.type == "cuda"

    @torch.no_grad()
    def predict(x: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(x), batch_size):
            ids = torch.from_numpy(x[i:i + batch_size]).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(ids, padding_mask(ids))
            out.append(torch.softmax(logits.float(), -1).cpu().numpy())
        return np.vstack(out)

    return predict


def timed(predict, x: np.ndarray, repeats: int) -> dict:
    runs = [time_inference(predict, x, n=1000, warmup=2) for _ in range(repeats)]
    return {"median": round(statistics.median(runs), 1),
            "min": round(min(runs), 1), "max": round(max(runs), 1),
            "runs": [round(r, 1) for r in runs]}


def main():
    ap = argparse.ArgumentParser(description="Benchmark Member 4's Transformer (validation only).")
    ap.add_argument("--device", help="default: cuda if available")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--val-limit", type=int, default=0,
                    help="reload check on the first N val rows only (quick local test)")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else (platform.processor() or "cpu")
    print(f"device: {device} ({dev_name}) | batch {args.batch_size} | repeats {args.repeats}")

    val = pd.read_csv(os.path.join(ROOT, "data", "val.csv"))
    if args.val_limit:
        val = val.head(args.val_limit)
        print(f"*** quick run: reload check on {len(val)} val rows only ***")

    per_seed = {}
    for s in SEEDS:
        folder = os.path.join(RES, "final", f"seed_{s}")
        model, ckpt = load_model(os.path.join(folder, "best_model.pt"), device)
        with open(os.path.join(folder, "metrics.json")) as f:
            trained = json.load(f)
        x = encode(val["text"], ckpt["vocab"], max_len=ckpt["max_len"])
        y = val["y"].to_numpy()

        # 1. reload check
        predict32 = make_predict_fn(model, device, args.batch_size, fp16=False)
        probs = predict32(x)
        res = evaluate_model(y, probs.argmax(1), probs, ckpt["classes"], split="val")
        diff = abs(res["f1_macro"] - trained["f1_macro"])
        reload_ok = diff < 1e-3 if not args.val_limit else None

        # 2 + 3. inference time and memory, same 1,000 validation complaints
        x1k = x[:1000]
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        t32 = timed(predict32, x1k, args.repeats)
        peak = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
        t16 = (timed(make_predict_fn(model, device, args.batch_size, fp16=True), x1k, args.repeats)
               if device.type == "cuda" else None)

        out = {
            "seed": s,
            "device": f"{device} ({dev_name})",
            "batch_size": args.batch_size,
            "n_examples_timed": len(x1k),
            "inference_ms_per_1k": t32["median"],
            "inference_ms_per_1k_fp32": t32,
            "inference_ms_per_1k_fp16": t16,
            "peak_gpu_mb_inference": round(peak, 1) if peak is not None else None,
            "reload_check": {
                "val_rows": len(val),
                "val_f1_macro_now": round(res["f1_macro"], 4),
                "val_f1_macro_at_training": round(trained["f1_macro"], 4),
                "abs_diff": round(diff, 5),
                "passed": reload_ok,
            },
            "timing_note": "shared src.evaluate.time_inference; median of repeats; "
                           "includes host-to-device copy and softmax; ids pre-encoded",
        }
        if not args.val_limit:
            with open(os.path.join(folder, "benchmark.json"), "w") as f:
                json.dump(out, f, indent=2)
        per_seed[s] = (out, trained)
        f16 = f" | fp16 {t16['median']:.1f} ms" if t16 else ""
        mem = f" | peak GPU {peak:.0f} MB" if peak is not None else ""
        chk = ("PASS" if reload_ok else "FAIL") if reload_ok is not None else "partial"
        print(f"seed {s}: reload F1 {res['f1_macro']:.4f} vs trained {trained['f1_macro']:.4f} [{chk}] | "
              f"fp32 {t32['median']:.1f} ms / 1k{f16}{mem}")

    if args.val_limit:
        print("quick run: nothing written")
        return

    def ms(values):
        v = np.asarray(values, dtype=float)
        return {"mean": round(float(v.mean()), 1), "std": round(float(v.std(ddof=1)), 1)}

    outs = [o for o, _ in per_seed.values()]
    trains = [t for _, t in per_seed.values()]
    eff = {
        "model": "transformer",
        "device": outs[0]["device"],
        "seeds": SEEDS,
        "params_trainable": trains[0]["params"]["trainable"],
        "val_f1_macro": {"mean": round(float(np.mean([t["f1_macro"] for t in trains])), 4),
                         "std": round(float(np.std([t["f1_macro"] for t in trains], ddof=1)), 4)},
        "train_time_per_epoch_s": ms([t["mean_epoch_time_s"] for t in trains]),
        "train_time_total_s": ms([t["train_time_s"] for t in trains]),
        "peak_gpu_mb_training": round(max(t["peak_gpu_mb"] or 0 for t in trains), 1) or None,
        "inference_ms_per_1k": ms([o["inference_ms_per_1k"] for o in outs]),
        "inference_ms_per_1k_fp16": (ms([o["inference_ms_per_1k_fp16"]["median"] for o in outs])
                                     if outs[0]["inference_ms_per_1k_fp16"] else None),
        "peak_gpu_mb_inference": (round(max(o["peak_gpu_mb_inference"] for o in outs), 1)
                                  if outs[0]["peak_gpu_mb_inference"] is not None else None),
        "reload_check_passed": all(o["reload_check"]["passed"] for o in outs),
        "note": "Validation data only. Training cost from training runs; inference cost "
                "from this benchmark.",
    }
    with open(os.path.join(RES, "efficiency.json"), "w") as f:
        json.dump(eff, f, indent=2)

    print("\n" + "=" * 62)
    print(f"trainable parameters     : {eff['params_trainable']:,}")
    print(f"time per epoch           : {eff['train_time_per_epoch_s']['mean']} s")
    print(f"total training time      : {eff['train_time_total_s']['mean']} s")
    print(f"inference per 1,000      : {eff['inference_ms_per_1k']['mean']} ms (fp32)"
          + (f", {eff['inference_ms_per_1k_fp16']['mean']} ms (fp16)" if eff["inference_ms_per_1k_fp16"] else ""))
    print(f"peak GPU memory          : training {eff['peak_gpu_mb_training']} MB, "
          f"inference {eff['peak_gpu_mb_inference']} MB")
    print(f"reload check (all seeds) : {'PASS' if eff['reload_check_passed'] else 'FAIL'}")
    print(f"saved -> {os.path.relpath(os.path.join(RES, 'efficiency.json'), ROOT)}")
    print("=" * 62)


if __name__ == "__main__":
    main()
