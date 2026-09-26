# Team guide — read this before you write any code

SE4050 Deep Learning group assignment. Deadline **30 September**.

We are comparing four deep learning models on the same task. The comparison is
only valid if every model sees identical data and is measured the same way.
That is what this guide is for.

---

## 1. What we are building

Given a consumer's written complaint about a financial product, predict which
**product category** it is about (credit card, mortgage, debt collection, and
so on). Input is the raw complaint text. Output is one of about 9 classes.

Four models, one per person, plus a classical baseline:

| Model | Owner | File to create |
| --- | --- | --- |
| TextCNN | Member 2 | `src/models/textcnn.py` |
| BiLSTM + attention | Member 3 | `src/models/bilstm_attn.py` |
| Transformer encoder (from scratch) | Member 4 | `src/models/transformer.py` |
| DistilBERT (fine-tuned) | Member 1 | `src/models/distilbert.py` |
| TF-IDF + Logistic Regression (baseline) | Member 2 | `notebooks/02_baseline.ipynb` |

The marks are **not** for the highest accuracy. 30% is for the critical
analysis — explaining *why* the models differ.

---

## 2. Setup (10 minutes, do this now)

```bash
git clone https://github.com/ravindu912/Deep-Learning-Project.git
cd Deep-Learning-Project

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. Getting the data — do NOT download it yourself

**Only the group leader downloads and splits the data.** The CFPB database
updates daily, so if you download it on a different day you get different
complaints, a different split, and results nobody can compare to.

1. Download `train.csv`, `val.csv`, `test.csv` and `split_meta.json` from the
   shared Drive folder:
   https://drive.google.com/drive/folders/15HSUSIsoxDiv5TYDdNGu87w5lE_ikv-i?usp=sharing
2. Put all four into the `data/` folder of your clone.
3. Verify you have the right files:

```bash
python scripts/check_data.py
```

It must print `All files match`. If it does not, re-copy the files. Do not
train until it passes.

The split is **59,970 train / 12,851 val / 12,851 test**, 5 classes, seed 42,
with zero overlap between any two splits.

The CSVs are gitignored on purpose — they are too large for GitHub.

---

## 4. The two shared modules — use them, don't reinvent them

### `src/data.py` — loading the data

```python
from src.data import load_splits, build_vocab, encode, class_weights

train, val, test, meta = load_splits('data')
# train.text  -> cleaned complaint text
# train.y     -> integer label
# meta['classes'], meta['n_classes'], meta['label2id']
```

For TextCNN / BiLSTM / Transformer:

```python
vocab = build_vocab(train.text)            # training set ONLY
X_train = encode(train.text, vocab, max_len=MAX_LEN)
X_val   = encode(val.text,   vocab, max_len=MAX_LEN)
w = class_weights(train.y, meta['n_classes'])   # for the loss
```

DistilBERT skips the vocabulary and uses its own tokenizer, but loads the same
split files.

**Never** re-clean the text, re-split the data, or build the vocabulary from
validation or test data. That is data leakage and it costs us marks.

### `src/evaluate.py` — reporting results

```python
from src.evaluate import evaluate_model, count_params, time_inference, save_results

res = evaluate_model(y_true, y_pred, y_proba, meta['classes'], split='test')
res['params']              = count_params(model)
res['train_time_s']        = elapsed_seconds
res['inference_ms_per_1k'] = time_inference(predict_fn, X_test)
save_results('textcnn', res, history=history, config=cfg, seed=42)
```

Do not compute your own accuracy or F1. One shared function, or the numbers
are not comparable.

---

## 5. Shared settings — everyone uses these

| Setting | Value |
| --- | --- |
| `MAX_LEN` | **224** (p90 of the training set is 194) |
| Seeds | 42, then 1 and 2 for the repeat runs |
| Loss | class-weighted cross-entropy |
| Optimiser | AdamW |
| Early stopping | on validation macro F1, patience 3 |
| Headline metric | **macro F1** (classes are imbalanced) |

---

## 6. Rules that protect our marks

1. **The test set is opened once**, at the very end, by the group leader. All
   your tuning uses `val` only. If you evaluate on test while tuning, the
   results are invalid.
2. **Tune 6–10 configurations** (learning rate, dropout, hidden size, batch
   size) and log every one. Save your best config as `configs/<model>.yaml`.
3. **Run your best config with 3 seeds** (42, 1, 2) so we can report mean and
   standard deviation. Two seeds if time runs out.
4. **Save learning curves** — training and validation loss per epoch, and
   validation macro F1 per epoch. `save_results(..., history=...)` handles it.
5. **Never commit data or checkpoints.** They are gitignored already.

---

## 7. Git workflow

```bash
git checkout -b model/textcnn          # your own branch
# ... work ...
git add src/models/textcnn.py configs/textcnn.yaml results/textcnn_seed42.json
git commit -m "Add TextCNN with 3/4/5 kernels, val macro F1 0.81"
git push -u origin model/textcnn
```

Then open a pull request and tag one other member to review.

- **Commit at least twice a week**, and the marking scheme says so explicitly.
- Commit messages must describe the change. "update" and "fix" score nothing.
- A burst of commits on the 29th is treated as evidence of *not* participating.

---

## 8. What you personally deliver

- [ ] Your model file in `src/models/`
- [ ] Your best config in `configs/`
- [ ] Result JSONs for 3 seeds in `results/`
- [ ] Learning curve figures
- [ ] Your model's part of report Section 6 (architecture, layers, activations,
      loss, optimiser, hyperparameters — and *why* each choice)
- [ ] Your model's part of report Section 8 (where it succeeds, where it fails,
      does it overfit, what would you change)
- [ ] Your assigned report sections (see the work breakdown document)
- [ ] Be able to explain **every layer of your model** in the viva. Each member
      is examined individually and 20% of the grade rides on it.

---

## 9. If you get stuck

Post in the group chat with what you tried and the error. Do not stay silent
for a day — with four days left, a blocked member is a failed section.
