"""
data.py — the shared data pipeline. EVERY model uses this. Do not write your own.

What it does:
  1. loads the CFPB subset produced by get_data.py
  2. cleans the narratives and merges renamed product labels
  3. removes duplicates BEFORE splitting (this is the leakage guard)
  4. makes one stratified 70/15/15 split with a fixed seed and saves it
  5. builds the word vocabulary from the TRAINING SET ONLY
  6. encodes text to padded integer sequences for TextCNN / BiLSTM / Transformer

DistilBERT ignores the vocabulary and uses its own tokenizer, but it MUST use
the same split files.

Run once:
    python src/data.py --input data/complaints_subset.csv --out data/

Then in every model script:
    from src.data import load_splits, build_vocab, encode
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
NARRATIVE_COL = "Consumer complaint narrative"
PRODUCT_COL = "Product"

PAD, UNK = "<pad>", "<unk>"

# CFPB renamed several product categories over the years. Without this merge
# the same complaint type appears under two labels and the model is punished
# for a distinction that does not exist.
LABEL_MAP = {
    "Credit card": "Credit card or prepaid card",
    "Prepaid card": "Credit card or prepaid card",
    "Credit reporting": "Credit reporting or other personal consumer reports",
    "Credit reporting, credit repair services, or other personal consumer reports":
        "Credit reporting or other personal consumer reports",
    "Bank account or service": "Checking or savings account",
    "Consumer Loan": "Vehicle loan or lease",
    "Payday loan": "Payday loan, title loan, or personal loan",
    "Payday loan, title loan, personal loan, or advance loan":
        "Payday loan, title loan, or personal loan",
    "Money transfers": "Money transfer, virtual currency, or money service",
    "Virtual currency": "Money transfer, virtual currency, or money service",
}


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------

# CFPB masks personal details as runs of X. They carry no signal and would
# otherwise dominate the vocabulary.
_MASK = re.compile(r"\bX{2,}\b", re.IGNORECASE)
_URL = re.compile(r"https?://\S+|www\.\S+")
_NUM = re.compile(r"\b\d[\d,.]*\b")
_NONWORD = re.compile(r"[^a-z0-9\s'$%]")
_SPACE = re.compile(r"\s+")


def clean_text(text: str, lower: bool = True) -> str:
    """Normalise one narrative. lower=False for DistilBERT-cased variants."""
    t = str(text)
    if lower:
        t = t.lower()
    t = _MASK.sub(" ", t)
    t = _URL.sub(" url ", t)
    t = _NUM.sub(" num ", t)          # amounts matter as a concept, not as values
    if lower:
        t = _NONWORD.sub(" ", t)
    return _SPACE.sub(" ", t).strip()


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------

def make_splits(
    input_csv: str,
    out_dir: str,
    min_words: int = 10,
    min_class: int = 1000,
    seed: int = SEED,
) -> dict:
    """Clean, dedupe, split 70/15/15 and write train/val/test CSVs."""
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(input_csv)

    if NARRATIVE_COL not in df.columns or PRODUCT_COL not in df.columns:
        raise ValueError(
            f"Expected columns '{NARRATIVE_COL}' and '{PRODUCT_COL}'. Got: {list(df.columns)}"
        )

    df = df[[NARRATIVE_COL, PRODUCT_COL]].dropna()
    df.columns = ["text", "label"]
    n_start = len(df)

    df["label"] = df["label"].replace(LABEL_MAP)

    # LEAKAGE GUARD: the same complaint sometimes appears more than once.
    # Dedupe here, before the split, or a test complaint can also be in train.
    # Dedupe on the RAW text: cleaning masks numbers and dates, which would
    # make genuinely different complaints look identical and delete them.
    df = df.drop_duplicates(subset=["text"])

    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.split().str.len() >= min_words]

    counts = df["label"].value_counts()
    keep = counts[counts >= min_class].index
    df = df[df["label"].isin(keep)].reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No rows left after filtering. Check --min-words and --min-class, "
            "and that the input CSV really holds narratives."
        )

    classes = sorted(df["label"].unique())
    label2id = {c: i for i, c in enumerate(classes)}
    df["y"] = df["label"].map(label2id)

    # 70 / 15 / 15, stratified so every split has the same class proportions.
    train, temp = train_test_split(
        df, test_size=0.30, stratify=df["y"], random_state=seed
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["y"], random_state=seed
    )

    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.reset_index(drop=True).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)

    meta = {
        "seed": seed,
        "rows_in": int(n_start),
        "rows_kept": int(len(df)),
        "n_classes": len(classes),
        "classes": classes,
        "label2id": label2id,
        "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "min_words": min_words,
        "min_class": min_class,
    }
    with open(os.path.join(out_dir, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"rows in: {n_start:,} -> kept: {len(df):,}")
    print(f"classes: {len(classes)}")
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")
    print(f"written to {out_dir}/")
    return meta


def load_splits(data_dir: str = "data"):
    """Load the three splits and the metadata. Use this in every model script."""
    parts = {}
    for name in ("train", "val", "test"):
        path = os.path.join(data_dir, f"{name}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing — run `python src/data.py` first.")
        parts[name] = pd.read_csv(path)
    with open(os.path.join(data_dir, "split_meta.json")) as f:
        meta = json.load(f)
    return parts["train"], parts["val"], parts["test"], meta


# --------------------------------------------------------------------------
# vocabulary and encoding (TextCNN / BiLSTM / Transformer)
# --------------------------------------------------------------------------

def build_vocab(train_texts, max_size: int = 30000, min_freq: int = 2) -> dict:
    """Build word -> id from the TRAINING SET ONLY. Never pass val or test here."""
    counter = Counter()
    for t in train_texts:
        counter.update(str(t).split())

    vocab = {PAD: 0, UNK: 1}
    for word, freq in counter.most_common():
        if freq < min_freq or len(vocab) >= max_size:
            break
        vocab[word] = len(vocab)
    print(f"vocab: {len(vocab):,} words (from {len(counter):,} unique)")
    return vocab


def encode(texts, vocab: dict, max_len: int = 256) -> np.ndarray:
    """Text -> padded integer array of shape (n, max_len). Post-padded, truncated."""
    unk = vocab[UNK]
    out = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, unk) for w in str(t).split()[:max_len]]
        out[i, : len(ids)] = ids
    return out


def class_weights(y, n_classes: int) -> np.ndarray:
    """Inverse-frequency weights for the loss. Compute on TRAIN labels only."""
    counts = np.bincount(np.asarray(y), minlength=n_classes).astype(float)
    counts[counts == 0] = 1.0
    w = counts.sum() / (n_classes * counts)
    return w.astype(np.float32)


def suggested_max_len(train_texts, percentile: int = 90) -> int:
    """Pick the sequence length from the training-set length distribution."""
    lengths = np.array([len(str(t).split()) for t in train_texts])
    val = int(np.percentile(lengths, percentile))
    print(f"length p50={int(np.percentile(lengths,50))} "
          f"p{percentile}={val} max={lengths.max()}")
    return val


def load_glove(path: str, vocab: dict, dim: int = 300) -> np.ndarray:
    """Build an embedding matrix from a GloVe txt file. Missing words stay random."""
    rng = np.random.default_rng(SEED)
    matrix = rng.normal(0, 0.1, (len(vocab), dim)).astype(np.float32)
    matrix[vocab[PAD]] = 0.0

    found = 0
    with open(path, encoding="utf8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab:
                matrix[vocab[word]] = np.asarray(parts[1:], dtype=np.float32)
                found += 1
    print(f"GloVe: {found:,}/{len(vocab):,} words found ({found/len(vocab):.1%})")
    return matrix


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/complaints_subset.csv")
    ap.add_argument("--out", default="data")
    ap.add_argument("--min-words", type=int, default=10)
    ap.add_argument("--min-class", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    make_splits(args.input, args.out, args.min_words, args.min_class, args.seed)
