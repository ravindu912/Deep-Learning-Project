# Financial Complaint Classification with Deep Learning

SE4050 – Deep Learning, 2026 Semester 2. Group assignment.

We classify consumer financial complaints into product categories using only the
complaint narrative, and compare four distinct deep learning architectures under
identical experimental conditions.

## Dataset

CFPB Consumer Complaint Database — <https://www.consumerfinance.gov/data-research/consumer-complaints/>

The CFPB stopped publishing complaint narratives on 14 August 2026, so the live
export no longer contains text. This project uses an archived pre-August-2026
snapshot (CC0), obtained from Kaggle:
<https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp>

Split files for the group (do not re-split):
https://drive.google.com/drive/folders/15HSUSIsoxDiv5TYDdNGu87w5lE_ikv-i?usp=sharing

Raw data is not committed.

| Column | Use |
| --- | --- |
| `Consumer complaint narrative` | input text |
| `Product` | target label — 5 classes |

Split: 59,970 train / 12,851 val / 12,851 test, stratified, seed 42, no overlap.
`MAX_LEN` = 224 for all models.

## Models

| Model | Owner | File |
| --- | --- | --- |
| TF-IDF + Logistic Regression (baseline) | Member 2 | `notebooks/02_baseline.ipynb` |
| TextCNN | Member 2 | `src/models/textcnn.py` |
| BiLSTM + attention | Member 3 | `src/models/bilstm_attn.py` |
| Transformer encoder (from scratch) | Member 4 | `src/models/transformer.py` |
| DistilBERT (fine-tuned) | Member 1 | `src/models/distilbert.py` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Running it

```bash
# 1. download and build the working subset  (once, by the group leader)
python get_data.py --out data/ --per-class 15000

# 2. build the shared 70/15/15 split         (once, by the group leader)
python src/data.py --input data/complaints_subset.csv --out data/

# 3. train your model                        (each member)
python src/models/textcnn.py --seed 42

# 4. build the comparison                    (at the end, group leader)
python src/evaluate.py
```

## Rules for the group

- **Use the shared split.** Never re-split, re-clean or re-sample. `src/data.py`
  runs once and everyone loads `train.csv`, `val.csv`, `test.csv` from `data/`.
- **Use `src/evaluate.py` for all metrics.** Do not compute your own accuracy.
- **The test set is opened once**, at the end, for the final numbers. All tuning
  uses the validation set.
- **Branch per model**: `model/textcnn`, `model/bilstm`, and so on. Merge via pull
  request with one reviewer.
- **Commit at least twice a week**, with messages that say what changed.
- **Log every run**: config and seed go in `configs/`, results in `results/`.

## Reproducibility

Seed 42 everywhere. Each model's final configuration is a YAML file in `configs/`.
Results are JSON files in `results/`, one per model per seed.

## Structure

```
.
├── get_data.py          # download + build the subset
├── requirements.txt
├── configs/             # one YAML per model
├── data/                # splits (gitignored); raw data never committed
├── notebooks/           # 01_eda, 02_baseline, ... 07_comparison
├── src/
│   ├── data.py          # SHARED: cleaning, split, vocab, encoding
│   ├── evaluate.py      # SHARED: metrics, plots, comparison table
│   └── models/          # one file per model
└── results/             # metrics JSON + figures
```

## Team

| Member | Student ID | Model | Report sections |
| --- | --- | --- | --- |
| | | DistilBERT + pipeline | 3, 4, 7 |
| | | TextCNN + baseline | 2 |
| | | BiLSTM + attention | 5, 8 |
| | | Transformer from scratch | 1, 9, 10 |

## Acknowledgements

Dataset: U.S. Consumer Financial Protection Bureau. Pretrained model:
`distilbert-base-uncased` (Hugging Face). Embeddings: GloVe 6B 300d (Stanford).
AI assistance is acknowledged in the report.
