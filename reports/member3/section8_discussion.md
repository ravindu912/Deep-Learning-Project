# Section 8 contribution: validation results and error analysis

The six-configuration validation search selected the two-layer BiLSTM with 64 hidden units per direction, dropout 0.3, learning rate 0.001 and batch size 64. The selected seed-42 checkpoint achieved GPU validation macro F1 **0.853275** at epoch 3. The [three-seed summary](../../results/bilstm_attn/repeats/summary.md) reports the fixed configuration's results with seeds 42, 1 and 2; all three repeat runs use the local RTX 4050. Validation informed model selection, so these scores must not be presented as final test performance.

The three seed scores are 0.853275, 0.847495 and 0.852118, respectively: **mean macro F1 0.850963 ± 0.003058** (sample standard deviation, n=3). Seeds 1 and 2 completed their early-stopping histories and saved verified checkpoints, but their processes reported abnormal exit statuses after the final output. This local runtime limitation is recorded in the [handoff notes](README.md); the summary was generated successfully from the saved results.

For seed 42, training loss continued to decrease after epoch 3, while validation loss increased and validation macro F1 declined. This pattern is consistent with overfitting. Early stopping ended training at epoch 6 and retained the epoch-3 checkpoint. [Learning curves](../../results/bilstm_attn/analysis/20260927T171701242213Z_seed42_epoch3_8a61417e/learning_curves.png) show this behavior.

The attention-analysis script re-evaluated the selected checkpoint on CPU. That run produced macro F1 **0.853207**, accuracy approximately **0.8525**, and **1,896 errors among 12,851 validation complaints**. The small CPU/GPU score difference is retained explicitly rather than substituting one evaluation's numbers for the other. Error counts and examples below belong to the CPU analysis; configuration selection and the repeat summary use the saved GPU evaluations.

Rule-based inspection flagged 395 errors with multiple financial-topic keywords, 238 errors with at most 20 cleaned words, and 113 errors with more than 224 words. These flags can overlap. They suggest cases for inspection; they neither prove the cause of an error nor establish higher error rates for those groups without their full-group denominators.

| Validation row (zero-based) | True category | Predicted category | Real excerpt and possible issue |
|---|---|---|---|
| 3 | credit_card | credit_reporting | “credit card opened name without knowledge authorization disputed card experian…” describes both a disputed card and credit reporting. Overlapping financial topics may make the boundary difficult. |
| 105 | credit_reporting | mortgages_and_loans | “submitted forbearance application became unemployed period time due covid received package back sun trust…” contains loan/forbearance cues despite the credit-reporting label. This is a possible topic/label mismatch, not proof the label is wrong. |
| 47 | credit_reporting | debt_collection | “account fraud knowledge account report one consent open account without authorization fraud” is short and gives limited detail about the underlying product. |

Four figures display complaint words and attention weights with their true and predicted categories:

- [Row 0: correctly classified comparison](../../results/bilstm_attn/analysis/20260927T171701242213Z_seed42_epoch3_8a61417e/attention_val_0.png)
- [Row 3: overlapping topics](../../results/bilstm_attn/analysis/20260927T171701242213Z_seed42_epoch3_8a61417e/attention_val_3.png)
- [Row 105: forbearance wording](../../results/bilstm_attn/analysis/20260927T171701242213Z_seed42_epoch3_8a61417e/attention_val_105.png)
- [Row 47: short complaint](../../results/bilstm_attn/analysis/20260927T171701242213Z_seed42_epoch3_8a61417e/attention_val_47.png)

Attention weights describe the model's combination of contextual word representations. They do not completely or causally explain its decision. Controlled input changes and additional manual review would be needed to investigate the suggested error causes. For long complaints, the model sees only the first 224 words; the presence of truncation alone does not prove that truncation caused an error.

Cross-model agreement and shared-error analysis cannot yet be reported: other members' validation predictions were unavailable. Needed files are `results/textcnn/val_predictions.csv`, `results/transformer/val_predictions.csv`, and `results/distilbert/val_predictions.csv` (optionally `results/baseline/val_predictions.csv`). Each needs `val_row_id` in original zero-based validation order, `text_sha256` of the exact UTF-8 validation text, `y_true`, and `y_pred` using the group's label IDs. Each model directory also needs `val_predictions_meta.json` identifying model, checkpoint/configuration, seed, label2id, and the validation-file SHA-256. These permit matching the same complaints and checking label alignment without using the test split.

The [detailed analysis](../../results/bilstm_attn/analysis/20260927T171701242213Z_seed42_epoch3_8a61417e/section8_analysis.md) records additional real examples and export locations. Full complaint prediction/error CSVs and model checkpoints remain local and ignored by Git.
