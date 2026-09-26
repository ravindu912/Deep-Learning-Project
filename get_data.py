"""
get_data.py — download the CFPB Consumer Complaint Database and build the
working subset used by every model in this project.

Usage:
    python get_data.py --out data/ --per-class 15000

Two download routes:
  1. Bulk CSV (fast, ~2-3 GB zipped). Whole database, one file.
  2. Public API fallback (slower, paged) if the bulk file is unavailable.

Output:
    data/raw_complaints.csv      full download (route 1 only)
    data/complaints_subset.csv   cleaned, label-merged, stratified sample
"""

import argparse
import io
import os
import sys
import time
import zipfile

import pandas as pd
import requests

BULK_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"

NARRATIVE_COL = "Consumer complaint narrative"
PRODUCT_COL = "Product"

# Older product names -> current categories. CFPB renamed several
# categories in 2017 and again later; without this merge the same
# complaint type appears under two different labels.
LABEL_MAP = {
    "Credit card": "Credit card or prepaid card",
    "Prepaid card": "Credit card or prepaid card",
    "Credit reporting": "Credit reporting, credit repair services, or other personal consumer reports",
    "Credit reporting or other personal consumer reports": "Credit reporting, credit repair services, or other personal consumer reports",
    "Bank account or service": "Checking or savings account",
    "Consumer Loan": "Vehicle loan or lease",
    "Payday loan": "Payday loan, title loan, or personal loan",
    "Payday loan, title loan, personal loan, or advance loan": "Payday loan, title loan, or personal loan",
    "Money transfers": "Money transfer, virtual currency, or money service",
    "Virtual currency": "Money transfer, virtual currency, or money service",
    "Other financial service": "Other financial service",
}


def download_bulk(out_dir: str) -> pd.DataFrame:
    """Route 1: download and unzip the full database."""
    print(f"Downloading bulk file from {BULK_URL} ...")
    resp = requests.get(BULK_URL, stream=True, timeout=120)
    resp.raise_for_status()

    buf = io.BytesIO()
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=1 << 20):
        buf.write(chunk)
        downloaded += len(chunk)
        print(f"\r  {downloaded / 1e6:.0f} MB", end="", flush=True)
    print()

    with zipfile.ZipFile(buf) as zf:
        name = zf.namelist()[0]
        print(f"Extracting {name} ...")
        with zf.open(name) as f:
            df = pd.read_csv(f, low_memory=False)

    raw_path = os.path.join(out_dir, "raw_complaints.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved full download to {raw_path}  ({len(df):,} rows)")
    return df


def download_api(target_rows: int, out_dir: str) -> pd.DataFrame:
    """Route 2: page through the public search API, narratives only."""
    print(f"Falling back to the API; fetching about {target_rows:,} rows ...")
    page_size = 1000
    frames, fetched = [], 0

    while fetched < target_rows:
        params = {
            "size": page_size,
            "frm": fetched,
            "format": "csv",
            "no_aggs": "true",
            "has_narrative": "true",
            "field": "all",
        }
        r = requests.get(API_URL, params=params, timeout=120)
        r.raise_for_status()
        chunk = pd.read_csv(io.StringIO(r.text))
        if chunk.empty:
            break
        frames.append(chunk)
        fetched += len(chunk)
        print(f"\r  {fetched:,} rows", end="", flush=True)
        time.sleep(1)  # be polite to a public API

    print()
    if not frames:
        sys.exit("API returned no rows. Download the CSV manually instead.")
    return pd.concat(frames, ignore_index=True)


def build_subset(df: pd.DataFrame, per_class: int, min_words: int, seed: int) -> pd.DataFrame:
    """Clean, merge labels and take a stratified sample."""
    before = len(df)

    df = df[[NARRATIVE_COL, PRODUCT_COL]].dropna()
    print(f"Rows with a narrative: {len(df):,} of {before:,}")

    # Merge renamed categories.
    df[PRODUCT_COL] = df[PRODUCT_COL].replace(LABEL_MAP)

    # Drop very short narratives - they carry almost no signal.
    df = df[df[NARRATIVE_COL].str.split().str.len() >= min_words]

    # Drop exact duplicates BEFORE splitting, otherwise the same complaint
    # can land in both train and test, which is data leakage.
    df = df.drop_duplicates(subset=[NARRATIVE_COL])
    print(f"After dedupe and length filter: {len(df):,}")

    # Drop classes too small to learn or evaluate.
    counts = df[PRODUCT_COL].value_counts()
    keep = counts[counts >= 1000].index
    dropped = counts[counts < 1000]
    if len(dropped):
        print(f"Dropping {len(dropped)} rare classes: {list(dropped.index)}")
    df = df[df[PRODUCT_COL].isin(keep)]

    # Stratified cap: at most `per_class` rows per category.
    df = (
        df.groupby(PRODUCT_COL, group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_class), random_state=seed))
        .sample(frac=1, random_state=seed)  # shuffle
        .reset_index(drop=True)
    )

    print("\nFinal class distribution:")
    print(df[PRODUCT_COL].value_counts().to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="output directory")
    ap.add_argument("--per-class", type=int, default=15000, help="max rows per class")
    ap.add_argument("--min-words", type=int, default=10, help="minimum narrative length")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--api", action="store_true", help="skip the bulk file, use the API")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.api:
        df = download_api(args.per_class * 12, args.out)
    else:
        try:
            df = download_bulk(args.out)
        except Exception as e:  # noqa: BLE001
            print(f"Bulk download failed ({e}).")
            df = download_api(args.per_class * 12, args.out)

    subset = build_subset(df, args.per_class, args.min_words, args.seed)

    path = os.path.join(args.out, "complaints_subset.csv")
    subset.to_csv(path, index=False)
    print(f"\nSaved {len(subset):,} rows to {path}")
    print("Next: run src/data.py to create the 70/15/15 split.")


if __name__ == "__main__":
    main()
