# Member 3: local RTX 4050 training

The laptop has an NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB VRAM. The earlier
temporary Python environment installed a CPU-only PyTorch wheel, which is why
that environment reported no usable GPU. The GPU sweep uses the separate,
gitignored `.venv-bilstm-gpu` environment.

From the project root in PowerShell, installation is:

```powershell
python -m venv .venv-bilstm-gpu
& .\.venv-bilstm-gpu\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
& .\.venv-bilstm-gpu\Scripts\python.exe -m pip install numpy pandas scikit-learn pyyaml matplotlib
& .\.venv-bilstm-gpu\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Do not start duplicate jobs if a sweep is already running. To start or resume an
interrupted sweep when no sweep process is active:

```powershell
& .\.venv-bilstm-gpu\Scripts\python.exe -B -u scripts/sweep_bilstm_attn.py --device cuda
```

Completed configurations are skipped; an interrupted configuration starts again
from its seed. An epoch-level resume is not implemented.

In a second terminal, generate final reports automatically once all six real
experiments have finished:

```powershell
& .\.venv-bilstm-gpu\Scripts\python.exe -B -u scripts/finalize_bilstm_attn.py --wait
```

Inspect `results/bilstm_attn/sweep/completion_status.json` for the reporting state,
and `experiments.csv` in that folder for actual completed scores. Only a status
of `complete` means that all six configurations and selected-checkpoint reporting
have finished. `completed_summary.md` is written only after those checks pass.

The finalization step creates four attention figures, learning curves, a Section
8 analysis with real validation errors, and reproducible prediction exports.
It verifies that the reloaded winning checkpoint reproduces its saved validation
macro F1. Training and analysis do not use test.csv. Other members' prediction
files are still needed for cross-model error comparisons.

Source for CUDA installation: [PyTorch installation instructions](https://docs.pytorch.org/get-started/locally/).
