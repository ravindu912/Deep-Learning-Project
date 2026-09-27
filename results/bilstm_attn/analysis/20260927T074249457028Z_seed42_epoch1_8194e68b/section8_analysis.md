# Section 8: BiLSTM validation error analysis (provisional)

Checkpoint: `checkpoints\bilstm_attn\20260927T074249457028Z_seed42\best.pt`, epoch 1, seed 42.
Exact saved snapshot: `runs/bilstm_attn/analysis/20260927T074249457028Z_seed42_epoch1_8194e68b/checkpoint_snapshot.pt` (gitignored).
This is an interim full-data-training checkpoint, not the winner of the unfinished six-configuration search.
All 12,851 validation complaints were evaluated; test data was not loaded.
Validation macro F1: **0.8377**; accuracy: **0.8372**; misclassified: **2,092/12,851**.

## Attention figures

Figures show the cleaned words actually supplied to the model (up to 224), with an attention weight under each word, the true category, and the predicted category. Padding has zero attention. Colors are scaled separately within each complaint; numeric weights allow direct inspection. Attention shows how the model combines its contextual LSTM outputs. It is not a complete or causal explanation, and a large weight does not prove that changing that word would change the prediction.

- Validation row 3: credit_card → credit_reporting. [Figure](attention_val_3.png)
- Validation row 11: retail_banking → credit_card. [Figure](attention_val_11.png)
- Validation row 20: credit_card → retail_banking. [Figure](attention_val_20.png)
- Validation row 0: credit_card → credit_card. [Figure](attention_val_0.png)

## Possible error types

These are overlapping, rule-based review groups, not proven causes. Counts must not be added together. Examples below are verbatim excerpts of the existing cleaned validation text; row IDs are zero-based.

### possible_category_overlap: 438 errors

At least two financial-topic keyword groups occur. This suggests possible overlapping subject matter, not a label error.

- **Validation row 3**: true `credit_card`; predicted `credit_reporting`; 41 words. Topic cues: credit_card;credit_reporting.
  > credit card opened name without knowledge authorization disputed card experian dispute closed experian stating card wont removed credit though stated never opened account contacted well told account created social security still reported experian biggest concern someone use card charge affect livelihood

- **Validation row 11**: true `retail_banking`; predicted `credit_card`; 47 words. Topic cues: credit_card;retail_banking.
  > continuously blocking debit card pnc visa credit card allow bank account connect tried making several purchase different site called least time problem resolved tell want purchase go say reset everything work keep happening account card proven identity multuple time point account refuse allow use money block purchase

### short_text_20_words_or_fewer: 276 errors

At most 20 words remain after shared preprocessing. Short length may limit context; it does not by itself establish ambiguity.

- **Validation row 20**: true `credit_card`; predicted `retail_banking`; 20 words. Topic cues: credit_reporting.
  > received debit card u bank unemployment ohio unemployed live indiana called deactivate account also made police report contacted credit bureau

- **Validation row 57**: true `credit_reporting`; predicted `debt_collection`; 19 words. Topic cues: none from the review list.
  > reporting debt credit file owe debt paid need removed know debt checked credit immediately paid receive day right dispute

### truncated_after_224_words: 131 errors

The complaint exceeds the 224-word input limit. The unseen tail could contain context, but this has not been tested as a cause.

- **Validation row 155**: true `debt_collection`; predicted `mortgages_and_loans`; 663 words. Topic cues: debt_collection;mortgages_and_loans.
  > way package forwarded mail new jersey address true copy envelope enclosed hereto exhibit phh mortgage service phh continued practice falsely addressing estate displayed page one communication dated enclosed hereto exhibit b relevant time husband name never husband estate right subject property operation upon death page two said communication dated phh falsely responded complaint regarding trial modification subject account enclosing alleged […]

- **Validation row 275**: true `credit_card`; predicted `mortgages_and_loans`; 634 words. Topic cues: credit_reporting;mortgages_and_loans.
  > victim fraud tenant renting property filed forged deed claiming owner property county tenant order avoid eviction fell month behind rent filed suit individual filed forged deed committed identity theft claiming ownership property texas district court county multiple retained attorney advise notify mortgage company pending lawsuit forged deed additionally advised make payment litigation resolved stood chance lose monies property additionally monies […]

### other_unassigned: 1,314 errors

No preceding heuristic matched. These errors need manual review; no cause is assigned.

- **Validation row 13**: true `credit_reporting`; predicted `mortgages_and_loans`; 49 words. Topic cues: credit_reporting.
  > first decided try get loan looked online loan company one time gave email address sudden several company emailing loan immediately unsubscribed every one could typed message clear longer market unsubscribed call email six company keep hitting credit report never spoken equity il de ia wi celebrity home loan co

- **Validation row 25**: true `credit_reporting`; predicted `mortgages_and_loans`; 73 words. Topic cues: credit_reporting.
  > journey arrears think around tow truck come pick car however paid full amount received car back car never repossessed ally financial initiated process never went paid arrears car reported credit report every month showing made absolutely payment year tried fix company avail still car making time payment since received car back argued everyone one well help balance reported credit report […]

## Limits and next steps

The checkpoint is early in training. These errors cannot establish the final architecture's weaknesses or overfitting. Repeat this analysis after validation-only model selection; compare the saved train/validation curves before making claims about overfitting. Review overlap cases manually and check whether meaningful context was truncated before considering changes to the team's shared sequence-length setting.

## Cross-model comparison: unavailable

No other members' per-complaint validation prediction files were available during this analysis. Aggregate accuracy/F1 JSONs are insufficient to identify shared or unique errors. Request these files:

- `results/textcnn/val_predictions.csv`
- `results/transformer/val_predictions.csv`
- `results/distilbert/val_predictions.csv`
- Optional baseline: `results/baseline/val_predictions.csv`

Each CSV must contain `val_row_id` (zero-based original val.csv row), `text_sha256` (SHA-256 of the exact UTF-8 cleaned text), `y_true`, and `y_pred` using split_meta.json label IDs. Include a companion `val_predictions_meta.json` with model name, checkpoint/config ID, seed, label2id, and the SHA-256 of val.csv. Include every validation row once and use validation-selected checkpoints. Prediction probabilities are optional. These are requested output paths, not files that currently exist.

This analysis used val.csv SHA-256 `db03f918b8c559b5142da3f4c50ba1a803608a27c2d105d246b96e659303248f`. Its complete predictions and error exports are in `runs/bilstm_attn/analysis/20260927T074249457028Z_seed42_epoch1_8194e68b` (gitignored).
