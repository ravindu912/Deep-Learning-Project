"""Package current Member 3 code and train/val for a manual Colab upload.

No Git actions, checkpoints, or test split are included. Output is gitignored.
"""

from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = [
        "src/__init__.py", "src/data.py", "src/evaluate.py",
        "src/models/__init__.py", "src/models/bilstm_attn.py",
        "scripts/train_bilstm_attn.py", "scripts/sweep_bilstm_attn.py",
        "scripts/analyze_bilstm_attn.py", "scripts/finalize_bilstm_attn.py",
        "configs/bilstm_attn.yaml", "data/train.csv", "data/val.csv", "data/split_meta.json",
    ]
    paths.extend(str(path.relative_to(ROOT)) for path in sorted((ROOT / "configs/bilstm_attn_sweep").glob("*.yaml")))
    destination = ROOT / "data/member3_colab.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in paths:
            path = ROOT / relative
            archive.write(path, arcname=path.relative_to(ROOT).as_posix())
    print(f"Created {destination}; train/val only, no test split or checkpoints")


if __name__ == "__main__":
    main()
