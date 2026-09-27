"""Member 3 validation-only predictions, attention figures, and error evidence.

Requires a checkpoint from full-data training, not the two-batch smoke trial.
Example: python scripts/analyze_bilstm_attn.py --checkpoint checkpoints/bilstm_attn/RUN/best.pt
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data import encode
from src.evaluate import evaluate_model
from src.models.bilstm_attn import BiLSTMAttention


CUES = {
    "credit_card": r"credit card|card issuer",
    "credit_reporting": r"credit report|credit bureau|credit score|equifax|experian|transunion",
    "debt_collection": r"debt collect|collection agency|collector|collections",
    "mortgages_and_loans": r"mortgage|student loan|loan servicer|auto loan|car loan",
    "retail_banking": r"checking account|savings account|overdraft|bank account|wire transfer",
}


def error_tags(text: str, max_len: int) -> tuple[list[str], list[str]]:
    cues = [name for name, pattern in CUES.items() if re.search(pattern, text, re.IGNORECASE)]
    tags = []
    if len(cues) >= 2:
        tags.append("possible_category_overlap")
    if len(text.split()) <= 20:
        tags.append("short_text_20_words_or_fewer")
    if len(text.split()) > max_len:
        tags.append("truncated_after_224_words")
    return tags or ["other_unassigned"], cues


def draw_attention(words, weights, row, destination: Path, status: str):
    # Each word has its actual attention weight beneath it. Color is per figure.
    positions, x, line = [], 0, 0
    for word in words:
        width = max(len(word), 6) + 2
        if x + width > 110 and x:
            x, line = 0, line + 1
        positions.append((x, line, width))
        x += width
    fig, ax = plt.subplots(figsize=(15, max(4, (line + 1) * 0.55 + 2.5)))
    ax.set_xlim(-1, 112)
    ax.set_ylim((line + 1) * 2 + 1, -1)
    ax.axis("off")
    scale = max(float(max(weights)), 1e-12)
    for word, weight, (x, line_index, width) in zip(words, weights, positions):
        color = plt.cm.YlOrRd(float(weight) / scale)
        ax.text(x + width / 2, line_index * 2, word, ha="center", va="center",
                fontsize=9, family="monospace", parse_math=False,
                color="white" if weight / scale > 0.7 else "black",
                bbox={"boxstyle": "round,pad=0.22", "facecolor": color, "edgecolor": "none"})
        ax.text(x + width / 2, line_index * 2 + 0.8, f"{weight:.4f}",
                ha="center", va="center", fontsize=8, family="monospace")
    fig.suptitle(
        f"Validation row {row.val_row_id} | {status}\n"
        f"True: {row.true_category} | Predicted: {row.predicted_category}\n"
        f"Shown: {len(words)} of {row.word_count} cleaned words; predicted probability: {row.confidence:.3f}",
        fontsize=11,
    )
    fig.text(0.02, 0.02,
             "Darker = larger attention within this complaint; numbers are actual weights. "
             "Padding is omitted.\nAttention describes model weighting, not a complete or causal explanation.", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.88))
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(2)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    # Snapshot the exact bytes, even if the training process later replaces best.pt.
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    cfg, vocab, meta = checkpoint["config"], checkpoint["vocab"], checkpoint["meta"]
    if cfg.get("experiment_kind") == "short_trial" or cfg.get("max_train_batches") is not None:
        raise ValueError("A smoke-trial checkpoint is not suitable for the requested trained-model report")
    selected = False
    best_path = ROOT / "results/bilstm_attn/sweep/best.json"
    if best_path.exists():
        best = json.loads(best_path.read_text(encoding="utf-8"))
        selected = (
            best.get("completed_configurations") == 6
            and Path(best["result_dir"]).name == cfg["run_id"]
            and best["best_epoch"] == checkpoint["epoch"]
            and np.isclose(best["validation_macro_f1"], checkpoint["val_f1_macro"], atol=1e-8)
        )
    is_selected_checkpoint = bool(selected)

    # This is the only split read. Keep its original row order for cross-model joins.
    val_path = ROOT / "data/val.csv"
    val_bytes = val_path.read_bytes()
    val_hash = hashlib.sha256(val_bytes).hexdigest()
    val = pd.read_csv(io.BytesIO(val_bytes))
    if len(val) != meta["sizes"]["val"] or set(val.y.unique()) != set(range(meta["n_classes"])):
        raise ValueError("Validation data does not match checkpoint metadata")
    classes = meta["classes"]
    model = BiLSTMAttention(
        len(vocab), meta["n_classes"], embedding_dim=cfg["embedding_dim"],
        hidden_size=cfg["hidden_size"], dropout=cfg["dropout"],
        freeze_embeddings=cfg["freeze_embeddings"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    inputs = encode(val.text, vocab, max_len=cfg["max_len"])
    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(val), args.batch_size):
            logits, _ = model(torch.from_numpy(inputs[start:start + args.batch_size]))
            probabilities.append(logits.softmax(dim=1).numpy())
            if start % (20 * args.batch_size) == 0:
                print(f"Validation predictions: {min(start + args.batch_size, len(val))}/{len(val)}", flush=True)
    probabilities = np.concatenate(probabilities)
    predictions = probabilities.argmax(axis=1)
    metrics = evaluate_model(val.y.to_numpy(), predictions, probabilities, classes, split="val")

    analysis_id = f"{checkpoint_path.parent.name}_epoch{checkpoint['epoch']}_{checkpoint_hash[:8]}"
    output = ROOT / "results/bilstm_attn/analysis" / analysis_id
    # Full complaint exports remain in a directory already excluded from Git.
    evidence = ROOT / "runs/bilstm_attn/analysis" / analysis_id
    output.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    # Preserve this exact model even if training replaces the original best.pt.
    snapshot = evidence / "checkpoint_snapshot.pt"
    snapshot.write_bytes(checkpoint_bytes)
    prediction_meta = {
        "model": "bilstm_attn", "seed": cfg["seed"], "epoch": checkpoint["epoch"],
        "label2id": meta["label2id"], "validation_sha256": val_hash,
        "checkpoint_sha256": checkpoint_hash, "checkpoint_snapshot": str(snapshot.relative_to(ROOT)),
        "split": "val", "row_id_convention": "zero-based row in unchanged val.csv",
    }
    (evidence / "val_predictions_meta.json").write_text(json.dumps(prediction_meta, indent=2), encoding="utf-8")
    result = pd.DataFrame({
        "val_row_id": np.arange(len(val)),
        "text_sha256": [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in val.text],
        "y_true": val.y, "y_pred": predictions,
        "true_category": [classes[y] for y in val.y],
        "predicted_category": [classes[y] for y in predictions],
        "confidence": probabilities.max(axis=1), "word_count": val.text.str.split().str.len(),
        "text": val.text,
    })
    result["correct"] = result.y_true == result.y_pred
    result["possible_error_types"] = [";".join(error_tags(t, cfg["max_len"])[0]) for t in val.text]
    result["category_cues"] = [";".join(error_tags(t, cfg["max_len"])[1]) for t in val.text]
    result.to_csv(evidence / "bilstm_attn_val_predictions.csv", index=False)
    errors = result[~result.correct]
    errors.to_csv(evidence / "bilstm_attn_val_errors.csv", index=False)

    # Select reproducibly by row order, not by unusually impressive attention.
    groups = {}
    for tag in ("possible_category_overlap", "short_text_20_words_or_fewer", "truncated_after_224_words", "other_unassigned"):
        matches = errors[errors.possible_error_types.str.split(";").map(lambda tags: tag in tags)]
        groups[tag] = {"count": len(matches), "examples": matches.head(2).val_row_id.tolist()}
    selected = []
    for tag in ("possible_category_overlap", "short_text_20_words_or_fewer", "truncated_after_224_words"):
        selected.extend(i for i in groups[tag]["examples"] if i not in selected and len(selected) < 3)
    for i in result[result.correct].val_row_id:
        if i not in selected and len(selected) < 4:
            selected.append(int(i))
    for i in result.val_row_id:
        if i not in selected and len(selected) < 4:
            selected.append(int(i))

    status = ("VALIDATION-SELECTED" if is_selected_checkpoint else "PROVISIONAL")
    status += f": {cfg.get('experiment_name', 'baseline')}, epoch {checkpoint['epoch']}"
    figure_records = []
    with torch.inference_mode():
        for row_id in selected:
            row = result.iloc[row_id]
            logits, attention = model(torch.from_numpy(inputs[row_id:row_id + 1]))
            assert int(logits.argmax(dim=1)[0]) == int(row.y_pred)
            words = row.text.split()[:cfg["max_len"]]
            weights = attention[0, :len(words)].numpy()
            assert np.isclose(weights.sum(), 1.0, atol=1e-6)
            assert torch.count_nonzero(attention[0, len(words):]).item() == 0
            filename = f"attention_val_{row_id}.png"
            draw_attention(words, weights, row, output / filename, status)
            figure_records.append({
                "val_row_id": row_id, "figure": filename, "true_category": row.true_category,
                "predicted_category": row.predicted_category, "words": words,
                "attention_weights": weights.tolist(), "full_word_count": int(row.word_count),
            })
    payload = {"status": status, "checkpoint": str(checkpoint_path.relative_to(ROOT)),
               "is_selected_checkpoint": is_selected_checkpoint,
               "checkpoint_snapshot": str(snapshot.relative_to(ROOT)),
               "checkpoint_sha256": checkpoint_hash, "validation_sha256": val_hash,
               "epoch": checkpoint["epoch"], "config": cfg, "metrics": metrics,
               "misclassified_count": len(errors), "error_groups": groups, "figures": figure_records,
               "predictions_directory": str(evidence.relative_to(ROOT))}
    (output / "analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Section 8: BiLSTM validation error analysis" + ("" if is_selected_checkpoint else " (provisional)"), "",
        f"Checkpoint: `{payload['checkpoint']}`, epoch {checkpoint['epoch']}, seed {cfg['seed']}.",
        f"Exact saved snapshot: `{snapshot.relative_to(ROOT).as_posix()}` (gitignored).",
        ("This checkpoint won the completed six-configuration search using validation macro F1 only."
         if is_selected_checkpoint else
         "This is an interim full-data-training checkpoint, not the winner of the unfinished six-configuration search."),
        f"All {len(val):,} validation complaints were evaluated; test data was not loaded.",
        f"Validation macro F1: **{metrics['f1_macro']:.4f}**; accuracy: **{metrics['accuracy']:.4f}**; "
        f"misclassified: **{len(errors):,}/{len(val):,}**.", "",
        "## Attention figures", "",
        "Figures show the cleaned words actually supplied to the model (up to 224), with an attention weight "
        "under each word, the true category, and the predicted category. Padding has zero attention. "
        "Colors are scaled separately within each complaint; numeric weights allow direct inspection. "
        "Attention shows how the model combines its contextual LSTM outputs. It is not a complete or causal "
        "explanation, and a large weight does not prove that changing that word would change the prediction.", "",
    ]
    for record in figure_records:
        lines.append(f"- Validation row {record['val_row_id']}: {record['true_category']} → "
                     f"{record['predicted_category']}. [Figure]({record['figure']})")
    lines += ["", "## Possible error types", "",
              "These are overlapping, rule-based review groups, not proven causes. Counts must not be added together. "
              "Examples below are verbatim excerpts of the existing cleaned validation text; row IDs are zero-based.", ""]
    explanations = {
        "possible_category_overlap": "At least two financial-topic keyword groups occur. This suggests possible overlapping subject matter, not a label error.",
        "short_text_20_words_or_fewer": "At most 20 words remain after shared preprocessing. Short length may limit context; it does not by itself establish ambiguity.",
        "truncated_after_224_words": "The complaint exceeds the 224-word input limit. The unseen tail could contain context, but this has not been tested as a cause.",
        "other_unassigned": "No preceding heuristic matched. These errors need manual review; no cause is assigned.",
    }
    for tag, group in groups.items():
        lines += [f"### {tag}: {group['count']:,} errors", "", explanations[tag], ""]
        for row_id in group["examples"]:
            row = result.iloc[row_id]
            excerpt = " ".join(row.text.split()[:60])
            if row.word_count > 60:
                excerpt += " […]"
            lines += [f"- **Validation row {row_id}**: true `{row.true_category}`; predicted `{row.predicted_category}`; "
                      f"{row.word_count} words. Topic cues: {row.category_cues or 'none from the review list'}.",
                      f"  > {excerpt}", ""]
    lines += ["## Limits and next steps", "",
              ("These are validation results used during model selection, not an unbiased final test estimate. "
               if is_selected_checkpoint else
               "The checkpoint is early in training. Repeat this analysis after validation-only model selection. ")
              + "Errors alone do not establish overfitting. Compare the saved train/validation curves before making "
              "that claim. Review overlap cases manually and check whether meaningful context was truncated before "
              "considering changes to the team's shared sequence-length setting.", "",
              "## Cross-model comparison: unavailable", "",
              "No other members' per-complaint validation prediction files were available during this analysis. "
              "Aggregate accuracy/F1 JSONs are insufficient to identify shared or unique errors. Request these files:", "",
              "- `results/textcnn/val_predictions.csv`", "- `results/transformer/val_predictions.csv`",
              "- `results/distilbert/val_predictions.csv`",
              "- Optional baseline: `results/baseline/val_predictions.csv`", "",
              "Each CSV must contain `val_row_id` (zero-based original val.csv row), `text_sha256` "
              "(SHA-256 of the exact UTF-8 cleaned text), `y_true`, and `y_pred` using split_meta.json label IDs. "
              "Include a companion `val_predictions_meta.json` with model name, checkpoint/config ID, seed, label2id, "
              "and the SHA-256 of val.csv. Include every validation row once and use validation-selected checkpoints. "
              "Prediction probabilities are optional. These are requested output paths, not files that currently exist.", "",
              f"This analysis used val.csv SHA-256 `{val_hash}`. Its complete predictions and error exports are in "
              f"`{evidence.relative_to(ROOT).as_posix()}` (gitignored).", ""]
    (output / "section8_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {len(figure_records)} attention figures and Section 8 draft: {output}")
    print(f"Validation macro F1={metrics['f1_macro']:.6f}; misclassified={len(errors)}/{len(val)}")


if __name__ == "__main__":
    main()
