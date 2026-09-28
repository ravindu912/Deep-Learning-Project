# Section 8: BiLSTM validation error analysis

Checkpoint: `checkpoints\bilstm_attn\20260927T171701242213Z_seed42\best.pt`, epoch 3, seed 42.
Exact saved snapshot: `runs/bilstm_attn/analysis/20260927T171701242213Z_seed42_epoch3_8a61417e/checkpoint_snapshot.pt` (gitignored).
This checkpoint won the completed six-configuration search using validation macro F1 only.
All 12,851 validation complaints were evaluated; test data was not loaded.
Validation macro F1: **0.8532**; accuracy: **0.8525**; misclassified: **1,896/12,851**.

## Attention figures

Figures show the cleaned words actually supplied to the model (up to 224), with an attention weight under each word, the true category, and the predicted category. Padding has zero attention. Colors are scaled separately within each complaint; numeric weights allow direct inspection. Attention shows how the model combines its contextual LSTM outputs. It is not a complete or causal explanation, and a large weight does not prove that changing that word would change the prediction.

- Validation row 3: credit_card → credit_reporting. [Figure](attention_val_3.png)
- Validation row 105: credit_reporting → mortgages_and_loans. [Figure](attention_val_105.png)
- Validation row 47: credit_reporting → debt_collection. [Figure](attention_val_47.png)
- Validation row 0: credit_card → credit_card. [Figure](attention_val_0.png)

## Possible error types

These are overlapping, rule-based review groups, not proven causes. Counts must not be added together. Examples below are verbatim excerpts of the existing cleaned validation text; row IDs are zero-based.

### possible_category_overlap: 395 errors

At least two financial-topic keyword groups occur. This suggests possible overlapping subject matter, not a label error.

- **Validation row 3**: true `credit_card`; predicted `credit_reporting`; 41 words. Topic cues: credit_card;credit_reporting.
  > credit card opened name without knowledge authorization disputed card experian dispute closed experian stating card wont removed credit though stated never opened account contacted well told account created social security still reported experian biggest concern someone use card charge affect livelihood

- **Validation row 105**: true `credit_reporting`; predicted `mortgages_and_loans`; 215 words. Topic cues: credit_reporting;mortgages_and_loans.
  > submitted forbearance application became unemployed period time due covid received package back sun trust document sign return return document impression signed document loan would placed forbearance decided would necessary u place loan forbearance attempted call loss mitigation department suntrust mortgage several time sent letter name contact information home preservation specialist within suntrust placed hold hour day several day row finally […]

### short_text_20_words_or_fewer: 238 errors

At most 20 words remain after shared preprocessing. Short length may limit context; it does not by itself establish ambiguity.

- **Validation row 47**: true `credit_reporting`; predicted `debt_collection`; 12 words. Topic cues: none from the review list.
  > account fraud knowledge account report one consent open account without authorization fraud

- **Validation row 57**: true `credit_reporting`; predicted `debt_collection`; 19 words. Topic cues: none from the review list.
  > reporting debt credit file owe debt paid need removed know debt checked credit immediately paid receive day right dispute

### truncated_after_224_words: 113 errors

The complaint exceeds the 224-word input limit. The unseen tail could contain context, but this has not been tested as a cause.

- **Validation row 275**: true `credit_card`; predicted `mortgages_and_loans`; 634 words. Topic cues: credit_reporting;mortgages_and_loans.
  > victim fraud tenant renting property filed forged deed claiming owner property county tenant order avoid eviction fell month behind rent filed suit individual filed forged deed committed identity theft claiming ownership property texas district court county multiple retained attorney advise notify mortgage company pending lawsuit forged deed additionally advised make payment litigation resolved stood chance lose monies property additionally monies […]

- **Validation row 569**: true `mortgages_and_loans`; predicted `retail_banking`; 249 words. Topic cues: retail_banking.
  > provided essential worker pandemic logged bank account morning find account totally wiped drawn contacted bank informed provisional credit issued request credit informed giving credit asked said credit given regard stimulus check account negative time issued credit mandated sure could full benefit stimulus check well problem account negative day two deposit one deposit another deposit day account negative opinion bunch lie […]

### other_unassigned: 1,211 errors

No preceding heuristic matched. These errors need manual review; no cause is assigned.

- **Validation row 13**: true `credit_reporting`; predicted `mortgages_and_loans`; 49 words. Topic cues: credit_reporting.
  > first decided try get loan looked online loan company one time gave email address sudden several company emailing loan immediately unsubscribed every one could typed message clear longer market unsubscribed call email six company keep hitting credit report never spoken equity il de ia wi celebrity home loan co

- **Validation row 19**: true `retail_banking`; predicted `credit_card`; 40 words. Topic cues: none from the review list.
  > booked vacation using delayed payment first payment followed payment used debit card make payment received provisional credit payment made filed complaint bank good service rendered trip booked bank tell past day denying claim understanding day time limit time service date

## Limits and next steps

These are validation results used during model selection, not an unbiased final test estimate. Errors alone do not establish overfitting. Compare the saved train/validation curves before making that claim. Review overlap cases manually and check whether meaningful context was truncated before considering changes to the team's shared sequence-length setting.

## Cross-model comparison: unavailable

No other members' per-complaint validation prediction files were available during this analysis. Aggregate accuracy/F1 JSONs are insufficient to identify shared or unique errors. Request these files:

- `results/textcnn/val_predictions.csv`
- `results/transformer/val_predictions.csv`
- `results/distilbert/val_predictions.csv`
- Optional baseline: `results/baseline/val_predictions.csv`

Each CSV must contain `val_row_id` (zero-based original val.csv row), `text_sha256` (SHA-256 of the exact UTF-8 cleaned text), `y_true`, and `y_pred` using split_meta.json label IDs. Include a companion `val_predictions_meta.json` with model name, checkpoint/config ID, seed, label2id, and the SHA-256 of val.csv. Include every validation row once and use validation-selected checkpoints. Prediction probabilities are optional. These are requested output paths, not files that currently exist.

This analysis used val.csv SHA-256 `db03f918b8c559b5142da3f4c50ba1a803608a27c2d105d246b96e659303248f`. Its complete predictions and error exports are in `runs/bilstm_attn/analysis/20260927T171701242213Z_seed42_epoch3_8a61417e` (gitignored).
