# Suggested PR title

Complete Member 3 BiLSTM attention model and validation analysis

# Suggested PR body

Adds the Member 3 complaint classifier: learned embeddings, a two-layer bidirectional LSTM, masked word attention and a five-category classifier. The training entry point uses the shared vocabulary, encoding, class weights and evaluation functions, while an approved train/validation-only loader keeps the reserved test split unopened.

Six logged configurations select hidden size 64 per direction, dropout 0.3, learning rate 0.001 and batch size 64. The fixed configuration is evaluated with seeds 42, 1 and 2; see `results/bilstm_attn/repeats/summary.md` for the actual validation mean and sample standard deviation. Checkpoints and complaint-level exports remain local and ignored. The search includes one CPU baseline and five GPU runs; the three selected-configuration seed runs use the RTX 4050.

Four validation attention figures, real error examples, learning curves and report contributions are included. Attention is presented as a view of aggregation weights, not a complete causal explanation. Cross-model error comparison is explicitly pending other members' validation predictions.

Validation: `python -B -m src.models.check_bilstm_attn` checks shapes, masking, normalized attention, padding invariance, and finite backward gradients. Training histories and saved settings document the completed experiments. Shared modules and other members' models are unchanged by this completion work. No test evaluation is included.

Reviewer: please check the architecture, padding behavior, validation-only selection, report figures and three-seed aggregation. The Section 5 input-preparation draft needs alignment with the actual group report heading before integration.

Known runtime issue: seeds 1 and 2 saved their complete early-stopping results but returned abnormal exit statuses after final output. Independent checkpoint-load/forward checks passed, and the summary command exited successfully. See `reports/member3/README.md` for exact statuses; native process shutdown remains unresolved.

# Publication status

This is a local PR-description draft, not an opened or reviewed PR. A read-only GitHub API check found no pull request for `ravindu912:model/bilstm-attn` on 28 September 2026 (Asia/Colombo). Opening/publishing a PR and obtaining another member's review remain separate steps. No dataset or checkpoint should be staged.
