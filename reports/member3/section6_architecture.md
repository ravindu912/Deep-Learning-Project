# Section 6 contribution: BiLSTM with word attention

The Member 3 classifier converts a complaint into word embeddings, processes them with a two-layer bidirectional LSTM, combines the contextual word representations using masked attention, and predicts one of five financial categories. The implementation is `src/models/bilstm_attn.py`; the selected settings are recorded in `configs/bilstm_attn.yaml`.

| Stage | Output shape for a batch of B complaints | Selected setting |
|---|---|---|
| Shared word encoding | B × 224 | Right padding 0; unknown word 1 |
| Word embeddings | B × 224 × 300 | 20,666 vocabulary entries; learned embeddings |
| Two-layer bidirectional LSTM | B × 224 × 128 | 64 hidden units in each direction |
| Masked word attention | B × 224 | Scores through Linear(128,64), tanh, Linear(64,1) |
| Weighted contextual summary | B × 128 | Sum of contextual outputs weighted by attention |
| Dropout and classifier | B × 5 | Dropout 0.3; linear category logits |

The original model default is 128 hidden units per direction. Validation tuning selected 64; the training configuration overrides the default. The selected model contains 6,495,485 trainable parameters. No GloVe vectors were used in these experiments. The model supports a supplied embedding matrix for future experiments, but the reported embeddings were initialized randomly and learned during training.

The two LSTM directions allow each word representation to include preceding and following context. Packed sequences exclude right padding from both directions. For contextual output h_t, attention computes s_t = vᵀ tanh(W h_t + b), normalizes the scores over real words with softmax, and forms c = Σ_t α_t h_t. Padding scores are excluded before normalization and their final weights are zero. The weights for a nonempty complaint sum to one. An all-padding input is handled without an undefined softmax and returns zero attention. The category layer consumes the summary c and returns raw logits; weighted cross-entropy consumes those logits directly.

Word attention gives the classifier a learned way to combine contextual outputs and provides weights that can be visualized. This is a flat word-attention model, not the sentence-and-word hierarchy of Yang et al. (2016). A high attention weight does not establish that the displayed word caused the prediction: each output already contains surrounding context, and attention alone is not a complete explanation (Jain and Wallace, 2019).

Training uses the group's 59,970 training complaints and 12,851 validation complaints. Vocabulary construction and inverse-frequency class weights use training data only. Encoding uses the shared `src.data.encode` function with `max_len=224`. A Member 3 loader reads only the two permitted CSVs and metadata, because the shared three-split loader also opens the reserved test split. The supplied data is neither regenerated nor resplit.

The optimizer is AdamW with learning rate 0.001 and weight decay 0.01. The batch size is 64, gradient norm is clipped at 1.0, and training runs for at most 20 epochs. Dropout is applied between LSTM layers and before classification. Early stopping monitors validation macro F1 with patience 3 and saves the best validation checkpoint. Metrics come from the shared evaluation module. Complete settings and software versions accompany every saved run.

Six configurations varied learning rate, dropout, hidden size, or batch size. The seed-42 search selected `05_smaller_lstm` with validation macro F1 0.853275 at epoch 3. This is an empirical selection on this split, not evidence that a smaller hidden state is universally superior. The baseline search run used CPU and the other five used GPU; timing comparisons between those runs are not controlled. The selected configuration's three-seed results are recorded separately in [the repeat summary](../../results/bilstm_attn/repeats/summary.md).

References:

- Yang et al. (2016), [Hierarchical Attention Networks for Document Classification](https://aclanthology.org/N16-1174/).
- Jain and Wallace (2019), [Attention is not Explanation](https://aclanthology.org/N19-1357/).
- Loshchilov and Hutter, [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
