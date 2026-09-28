# Three-seed validation results

The configuration was fixed after the six-configuration seed-42 search. Each repeat selected its checkpoint using validation macro F1. All three used the RTX 4050.

| Seed | Best epoch | Epochs run | Validation macro F1 | Validation accuracy |
|---|---|---|---|---|
| 1 | 3 | 6 | 0.847495 | 0.846860 |
| 2 | 5 | 8 | 0.852118 | 0.850984 |
| 42 | 3 | 6 | 0.853275 | 0.852541 |

**Macro F1: 0.850963 ± 0.003058** (mean ± sample standard deviation, n=3, ddof=1).

These are saved GPU training-evaluation metrics. The separate CPU attention analysis is not substituted into this comparison. Seed 42 remains the documented attention-analysis checkpoint; no seed was selected to inflate the aggregate.

Validation informed configuration and checkpoint selection. These results measure variation between seeds on this fixed split, not final test generalization. Checkpoints remain local and ignored by Git.
