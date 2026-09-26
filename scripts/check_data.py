"""
check_data.py — prove every member is training on identical data.

The group leader runs this once after making the split and commits
data/CHECKSUMS.txt. Everyone else runs it after copying the CSVs from the
shared Drive folder; if a hash differs, their data is wrong and their results
are not comparable.

    python scripts/check_data.py --write    # leader, once
    python scripts/check_data.py            # everyone else, to verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

FILES = ["train.csv", "val.csv", "test.csv", "split_meta.json"]


def md5(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def collect(data_dir: str) -> dict:
    out = {}
    for name in FILES:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            sys.exit(f"MISSING: {path}\nCopy the split files from the shared Drive folder.")
        out[name] = {"md5": md5(path), "bytes": os.path.getsize(path)}
    return out


def summarise(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "split_meta.json")) as f:
        meta = json.load(f)
    return {
        "seed": meta["seed"],
        "n_classes": meta["n_classes"],
        "sizes": meta["sizes"],
        "classes": meta["classes"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--write", action="store_true",
                    help="leader only: write data/CHECKSUMS.txt")
    args = ap.parse_args()

    current = collect(args.data)
    meta = summarise(args.data)
    ref_path = os.path.join(args.data, "CHECKSUMS.txt")

    if args.write:
        payload = {"files": current, "split": meta}
        with open(ref_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Written {ref_path}. Commit it, and share the CSVs via Drive.\n")
        for name, info in current.items():
            print(f"  {name:18s} {info['md5']}  {info['bytes']/1e6:8.2f} MB")
        return

    if not os.path.exists(ref_path):
        sys.exit(f"{ref_path} not found — pull the latest main branch.")

    with open(ref_path) as f:
        ref = json.load(f)

    ok = True
    for name, info in ref["files"].items():
        mine = current[name]["md5"]
        match = mine == info["md5"]
        ok &= match
        print(f"  {'OK  ' if match else 'WRONG'}  {name:18s} {mine}")

    print()
    if ok:
        print("All files match. You are training on the same data as everyone else.")
        print(f"  classes: {meta['n_classes']}   sizes: {meta['sizes']}")
    else:
        print("MISMATCH. Do not train on this data — your results will not be")
        print("comparable. Re-copy the CSVs from the shared Drive folder.")
        sys.exit(1)


if __name__ == "__main__":
    main()
