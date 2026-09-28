# Section 8: BiLSTM + Attention

Using seed 42, validation-only tuning selected **05_smaller_lstm** (learning rate 0.001, dropout 0.3, hidden size 64 per direction, batch size 64). Its best checkpoint was epoch 3, with validation macro F1 **0.8532** and accuracy **0.8525**. It misclassified **1,896 of 12,851** validation complaints.

Keyword-based review flagged 395 errors with multiple financial-topic cues; 238 errors had at most 20 cleaned words, and 113 errors exceeded the 224-word input limit. These groups overlap and describe possible review priorities, not proven causes. Example validation cases:

- Row 3: **credit_card → credit_reporting**. Excerpt: “credit card opened name without knowledge authorization disputed card experian dispute closed experian stating card wont removed credit …” [Attention figure](attention_val_3.png)
- Row 105: **credit_reporting → mortgages_and_loans**. Excerpt: “submitted forbearance application became unemployed period time due covid received package back sun trust document sign return return …” [Attention figure](attention_val_105.png)
- Row 47: **credit_reporting → debt_collection**. Excerpt: “account fraud knowledge account report one consent open account without authorization fraud” [Attention figure](attention_val_47.png)

Attention weights show how contextual word representations were combined. They are not a complete or causal explanation of a prediction. Manual review and controlled changes to inputs would be needed to investigate causes. [Learning curves](learning_curves.png) provide the evidence for assessing overfitting; errors alone cannot establish it.

These validation scores were used to choose the model and are not final held-out test performance. No test data was used. Cross-model error comparisons remain unavailable because the other members' per-complaint predictions have not been supplied. Exact required files and further real examples are listed in the [detailed analysis](section8_analysis.md).
