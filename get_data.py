"""
get_data.py — build the working subset from a CFPB complaints snapshot.

IMPORTANT — why this reads a local file instead of downloading:

The CFPB stopped publishing consumer complaint narratives on 14 August 2026
(https://www.consumerfinance.gov/about-us/newsroom/the-cfpb-to-cease-discretionary-publication-of-complaint-narratives-and-visualizations/).
The live download at files.consumerfinance.gov now has 15 columns and no
narrative text, so it is useless for this project. We therefore use an
archived pre-August-2026 snapshot, which is public-domain (CC0) data.

Where to get one (any of these works):
  * Kaggle: https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp
  * Kaggle: https://www.kaggle.com/datasets/iuriivoloshyn/cfpb-consumer-complaint-database
  * Kaggle: https://www.kaggle.com/datasets/selener/consumer-complaint-database

Download it, put the .csv (or .zip) in data/, then:

    python get_data.py --input data/<file>.csv --out data/ --per-class 15000

The script finds the narrative and product columns itself, so it does not
matter which snapshot you use or exactly how the columns are named.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile

import pandas as pd

# Column names vary between snapshots, so we look for any of these.
NARRATIVE_CANDIDATES = [
    "consumer complaint narrative",
    "complaint narrative",
    "consumer_complaint_narrative",
    "narrative",
    "complaint_what_happened",
    "complaint what happened",
    "text",
    "consumer_message",
]
PRODUCT_CANDIDATES = [
    "product",
    "product_name",
    "category",
    "label",
    "class",
]

# CFPB renamed several categories over the years. Merging them stops the model
# being punished for a distinction that does not exist.
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


def find_column(columns, candidates, what: str) -> str:
    """Match a column name case-insensitively against a candidate list."""
    lookup = {c.strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in lookup:
            return lookup[cand]
    # fall back to a partial match
    for key, original in lookup.items():
        if any(cand in key for cand in candidates):
            return original
    sys.exit(
        f"Could not find the {what} column.\n"
        f"Columns present: {list(columns)}\n"
        f"Pass it explicitly with --{what}-col."
    )


def open_snapshot(path: str):
    """Return a file-like handle to the CSV, unzipping on the fly if needed."""
    if path.lower().endswith(".zip"):
        zf = zipfile.ZipFile(path)
        inner = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not inner:
            sys.exit(f"No CSV inside {path}")
        print(f"reading {inner[0]} from {os.path.basename(path)}")
        return zf.open(inner[0])
    return open(path, "rb")


def load_snapshot(path: str, narrative_col: str | None, product_col: str | None):
    """Stream the snapshot, keeping only rows that have narrative text."""
    # peek at the header first
    with open_snapshot(path) as fh:
        header = pd.read_csv(fh, nrows=0)
    cols = list(header.columns)
    print(f"{len(cols)} columns found")

    narrative_col = narrative_col or find_column(cols, NARRATIVE_CANDIDATES, "narrative")
    product_col = product_col or find_column(cols, PRODUCT_CANDIDATES, "product")
    print(f"  narrative -> '{narrative_col}'")
    print(f"  product   -> '{product_col}'")

    keep, total = [], 0
    with open_snapshot(path) as fh:
        for chunk in pd.read_csv(
            fh,
            usecols=[narrative_col, product_col],
            chunksize=200_000,
            low_memory=False,
        ):
            total += len(chunk)
            keep.append(chunk.dropna(subset=[narrative_col]))
            print(f"\r  scanned {total:,} rows", end="", flush=True)
    print()

    df = pd.concat(keep, ignore_index=True)
    df = df.rename(columns={
        narrative_col: "Consumer complaint narrative",
        product_col: "Product",
    })[["Consumer complaint narrative", "Product"]]

    print(f"{total:,} rows total, {len(df):,} with narrative text "
          f"({len(df)/max(total,1):.1%})")
    if df.empty:
        sys.exit(
            "No narratives found. This looks like a post-August-2026 snapshot, "
            "which no longer contains complaint text. Use an archived snapshot "
            "— see the links at the top of this file."
        )
    return df


def build_subset(df: pd.DataFrame, per_class: int, min_words: int, seed: int):
    df = df.dropna()
    df["Product"] = df["Product"].replace(LABEL_MAP)

    df = df.drop_duplicates(subset=["Consumer complaint narrative"])
    long_enough = df["Consumer complaint narrative"].str.split().str.len() >= min_words
    df = df[long_enough]
    print(f"after dedupe and length filter: {len(df):,}")

    counts = df["Product"].value_counts()
    small = counts[counts < 1000]
    if len(small):
        print(f"dropping {len(small)} rare classes: {list(small.index)}")
    df = df[df["Product"].isin(counts[counts >= 1000].index)]

    # Cap each class at per_class rows. Built with an explicit loop rather than
    # groupby().apply(), which drops the grouping column in pandas >= 2.2.
    parts = [
        group.sample(min(len(group), per_class), random_state=seed)
        for _, group in df.groupby("Product", sort=False)
    ]
    df = (pd.concat(parts)
            .sample(frac=1, random_state=seed)
            .reset_index(drop=True))

    print("\nclass distribution:")
    print(df["Product"].value_counts().to_string())
    return df


def main():
    ap = argparse.ArgumentParser(
        description="Build the working subset from a CFPB snapshot (csv or zip)."
    )
    ap.add_argument("--input", required=True, help="path to the snapshot .csv or .zip")
    ap.add_argument("--out", default="data")
    ap.add_argument("--per-class", type=int, default=15000)
    ap.add_argument("--min-words", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--narrative-col", default=None, help="override auto-detection")
    ap.add_argument("--product-col", default=None, help="override auto-detection")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Not found: {args.input}")
    os.makedirs(args.out, exist_ok=True)

    df = load_snapshot(args.input, args.narrative_col, args.product_col)
    subset = build_subset(df, args.per_class, args.min_words, args.seed)

    path = os.path.join(args.out, "complaints_subset.csv")
    subset.to_csv(path, index=False)
    print(f"\nsaved {len(subset):,} rows -> {path}")
    print("next: python src/data.py --input "
          f"{path} --out {args.out}/")


if __name__ == "__main__":
    main()
