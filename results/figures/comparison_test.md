# Test-set comparison

All models evaluated once on the held-out test split (12,851 complaints), using the configuration selected on validation.

| model | macro F1 | accuracy | precision | recall | ROC-AUC | params (M) | train (s) | inference (ms/1k) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DistilBERT (fine-tuned) | 0.8701 | 0.8686 | 0.8694 | 0.8709 | 0.9783 | 66.9573 | 1323.2000 | 5675.0000 |
| TF-IDF + Logistic Regression | 0.8597 | 0.8585 | 0.8575 | 0.8623 | 0.9749 | 0.0500 | 3.9417 | 58.8197 |
| Transformer (from scratch) | 0.8572 | 0.8562 | 0.8556 | 0.8592 | 0.9761 | 8.4513 | 1000.3000 |  |
| BiLSTM + attention | 0.8498 | 0.8489 | 0.8491 | 0.8513 | 0.9739 | 6.4955 | 6411.7020 |  |
| TextCNN | 0.8438 | 0.8423 | 0.8424 | 0.8460 | 0.9690 | 2.8007 | 2763.9209 | 3248.4385 |

Blank cells: the measurement was not recorded by that model's script. Timings come from the hardware that ran each model and are indicative, not strictly comparable; parameter count is measured identically for all.
