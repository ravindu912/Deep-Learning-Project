# Member 3 handoff

Ready-to-integrate report material:

- [Section 6: architecture and training rationale](section6_architecture.md)
- [Section 8: results, attention and error discussion](section8_discussion.md)
- [Input-preparation draft](section5_input_preparation_draft.md): provisional Section 5 placement; its official heading/requirements were not available.
- [PR title and description](PR_DESCRIPTION.md): local draft, not an opened/reviewed PR.
- [Three-seed validation summary](../../results/bilstm_attn/repeats/summary.md), with individual settings/results and learning curves alongside it.
- [Six-configuration experiment ledger](../../results/bilstm_attn/sweep/experiments.csv).

Run from the project root in PowerShell using the GPU environment:

```powershell
# Tiny synthetic model check; does not train or read complaint data.
& .\.venv-bilstm-gpu\Scripts\python.exe -B -m src.models.check_bilstm_attn

# Reproduction commands; these start NEW training runs if repeated.
& .\.venv-bilstm-gpu\Scripts\python.exe -B -u scripts/train_bilstm_attn.py --config configs/bilstm_attn.yaml --seed 1
& .\.venv-bilstm-gpu\Scripts\python.exe -B -u scripts/train_bilstm_attn.py --config configs/bilstm_attn.yaml --seed 2
```

The existing seed-42 winner is `20260927T171701242213Z_seed42`; its checkpoint is used for the four documented attention figures. Do not replace it with whichever repeat obtains the highest score. The repeat experiment estimates seed variation with fixed settings, and reports all three scores.

The seed-1 trainer printed all six epoch results, its early-stopping message and final best-checkpoint message, but its command session reported exit status 1 without a traceback. Follow-up verification successfully loaded the saved checkpoint, matched its epoch and F1 to the completed history, executed a finite synthetic forward pass, and confirmed identical effective settings/runtime versions to seed 42. The unexpected command status is retained here rather than reported as a clean process exit.

Seed 2 likewise printed all eight epochs and its final early-stopping/checkpoint messages, then Python returned native Windows exit code -1073740791 (0xC0000409), without a Python traceback. Its checkpoint independently loads and produces finite outputs, and its epoch-5 score matches the saved history. The failure appears during process shutdown after training output, but its underlying native-library cause has not been established. Neither command is claimed to have exited cleanly. The aggregation command and both separate checkpoint verification commands exited with code 0.

Recorded validation macro F1: seed 42 = 0.8532751930, seed 1 = 0.8474949076, seed 2 = 0.8521175845. Mean ± sample standard deviation = **0.8509625617 ± 0.0030583465**. Seed 1 stopped after 6 epochs (best epoch 3); seed 2 stopped after 8 epochs (best epoch 5). All three used the same effective configuration and runtime versions. Results are available despite the documented shutdown issue; that issue remains a limitation of this local training environment.

To regenerate the aggregate without training or opening data:

```powershell
& .\.venv-bilstm-gpu\Scripts\python.exe -B scripts/summarize_bilstm_seeds.py results/bilstm_attn/validation/20260927T171701242213Z_seed42/bilstm_attn_seed42.json results/bilstm_attn/validation/20260927T191723748502Z_seed1/bilstm_attn_seed1.json results/bilstm_attn/validation/20260927T192624405313Z_seed2/bilstm_attn_seed2.json
```

Remaining group work: confirm the Section 5 brief and integrate these drafts into the actual report; supply other members' validation predictions for cross-model error analysis; publish a PR and obtain another member's review. Only the group leader should perform final test evaluation. No test data is required by these Member 3 completion commands.

No commits or pushes were made during this completion work. Before any later commit, review the changed-file list and stage only Member 3 code/configuration, aggregate results and report material. Keep datasets, checkpoints and full complaint exports out of Git.
