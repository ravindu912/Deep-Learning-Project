# Member 3: completed validation experiments

All six configurations ran on the full shared train/validation splits. No test data was used.

| Configuration | LR | Dropout | Hidden size | Batch size | Best epoch | Validation macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| 01_baseline | 0.001 | 0.3 | 128 | 64 | 4 | 0.850772 |
| 02_lower_lr | 0.0003 | 0.3 | 128 | 64 | 6 | 0.847575 |
| 03_higher_lr | 0.003 | 0.3 | 128 | 64 | 4 | 0.851619 |
| 04_more_dropout | 0.001 | 0.5 | 128 | 64 | 3 | 0.849580 |
| 05_smaller_lstm | 0.001 | 0.3 | 64 | 64 | 3 | 0.853275 |
| 06_larger_batch | 0.001 | 0.3 | 128 | 128 | 4 | 0.848930 |

Selected: **05_smaller_lstm**, validation macro F1 **0.853275**.
These validation scores guided tuning and are not final test results.

[Section 8 analysis](../analysis/20260927T171701242213Z_seed42_epoch3_8a61417e/section8_analysis.md)
[Concise Section 8 summary](../analysis/20260927T171701242213Z_seed42_epoch3_8a61417e/section8_summary.md)
[Learning curves](../analysis/20260927T171701242213Z_seed42_epoch3_8a61417e/learning_curves.png)

Cross-model error comparisons remain unavailable until other members supply the prediction files specified in the Section 8 analysis. No comparisons were invented.
