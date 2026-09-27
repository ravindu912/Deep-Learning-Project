# Member 3 input-preparation contribution (provisional placement)

The repository assigns Section 5 to Member 3 but does not specify its title or rubric. This text documents the actual input pipeline; confirm the group report's Section 5 requirements before placing it there. It does not claim to complete an unknown section brief.

Member 3 used the group's existing split files without regenerating or modifying them. The training set contains 59,970 complaints and the validation set contains 12,851 complaints. Metadata defines five categories: credit_card, credit_reporting, debt_collection, mortgages_and_loans, and retail_banking. The test split was reserved for the group leader and was not used in Member 3 training, tuning, or attention analysis.

The shared vocabulary builder counts whitespace-separated words in training complaints only, keeps words appearing at least twice, and caps the vocabulary at 30,000 entries. The resulting vocabulary contains 20,666 entries, including padding ID 0 and unknown-word ID 1. Validation words absent from this vocabulary receive ID 1. The existing split text is used as supplied; the Member 3 training loader does not clean it again.

Shared encoding retains the first 224 words of each complaint and pads shorter complaints on the right. Training inputs therefore have shape [59970, 224], validation inputs [12851, 224], and labels [59970] and [12851]. Model batches use integer word IDs [B, 224] and integer class IDs [B]. Long complaints can lose information after word 224; this limitation is considered in the error analysis.

To account for unequal class frequencies, the shared class-weight function computes w_c = N / (C × n_c) using training labels only. In metadata class order, the weights are approximately [1.162209, 0.863499, 0.857449, 0.927826, 1.356020]. They are supplied to weighted cross-entropy. Validation labels are used for checkpoint/configuration selection and evaluation, not to build vocabulary or class weights.

The seed is saved alongside all settings and runtime details. After configuration selection with seed 42, the same configuration is repeated with seeds 1 and 2. The three-seed summary reports validation mean and sample standard deviation; these validation-selected scores are not final held-out test scores.
