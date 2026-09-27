# Member 3: run the existing project in Google Colab

Colab is practical for this model because its GPU runtime can accelerate PyTorch.
GPU availability and session duration are not guaranteed. See Google's
[Colab FAQ](https://research.google.com/colaboratory/faq.html).
These are instructions, not a claim that any Colab experiment has run.

## 1. Package the current local code

From the `Deep-Learning-Project` directory in PowerShell:

```powershell
python scripts/package_bilstm_colab.py
```

This creates `data/member3_colab.zip`, which Git ignores. It includes your current
uncommitted model, training/sweep code, configs, and only `train.csv`, `val.csv`,
and `split_meta.json`. It does not read or include `test.csv`, `.git`, or checkpoints.
Do not use a repository clone alone: your new files may not be committed/pushed.

## 2. Create a GPU notebook and upload

Open https://colab.research.google.com/, create a notebook, and choose
**Runtime > Change runtime type > a GPU hardware accelerator** (for example T4
if offered). Run these cells in order.

```python
from google.colab import files
files.upload()  # Select data/member3_colab.zip from your computer.
```

```python
from pathlib import Path
import zipfile
with zipfile.ZipFile('/content/member3_colab.zip') as archive:
    archive.extractall('/content/Deep-Learning-Project')
```

```python
%cd /content/Deep-Learning-Project
%pip install numpy pandas scikit-learn pyyaml matplotlib
import torch
assert torch.cuda.is_available(), 'Select a GPU runtime before starting the sweep.'
print(torch.__version__, torch.cuda.get_device_name(0))
```

Use Colab's installed CUDA-enabled PyTorch. Do not install the local CPU-only
wheel. The packages above cover training and the final figures/report.

## 3. Keep results and checkpoints in your Drive

```python
from google.colab import drive
from datetime import datetime, timezone
drive.mount('/content/drive')
run_name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup = Path('/content/drive/MyDrive/Member3_BiLSTM') / run_name
backup.mkdir(parents=True, exist_ok=False)
for name in ('results', 'checkpoints', 'runs'):
    (backup / name).mkdir()
    Path(name).symlink_to(backup / name, target_is_directory=True)
import shutil
shutil.copytree('configs', backup / 'configs')
print('Save this path:', backup)
```

The input data stays on the Colab disk. Per-epoch logs and best checkpoints go
directly to this Drive folder, so completed work is retained if the runtime ends.

## 4. Run the short trial, then the six configurations

```python
!python -u scripts/train_bilstm_attn.py --config configs/bilstm_attn.yaml --seed 42 --trial
```

Check that it finishes without errors. Its score is only a short execution check
and is excluded from choosing a model. Then run:

```python
!python -u scripts/sweep_bilstm_attn.py --device cuda
```

Each of the six configs uses full training and validation splits, seed 42,
MAX_LEN 224, at most 20 epochs, and early stopping with patience 3. No test file
is needed. The results table is `results/bilstm_attn/sweep/experiments.csv`.
Blank scores belong to experiments that have not completed.

Once all six finish, the script writes `results/bilstm_attn/sweep/best.json` and
updates `configs/bilstm_attn.yaml` using validation macro F1 only. Generate the
selected model's figures and report with:

```python
!python -u scripts/finalize_bilstm_attn.py
```

Save the selected config back to Drive:

```python
shutil.copy2('configs/bilstm_attn.yaml', backup / 'configs/bilstm_attn.yaml')
```

To resume after a disconnected session, upload/extract the same archive, install
dependencies, mount Drive, and set `backup` to the saved existing folder instead
of making a new folder. Recreate the three symlinks to its `results`, `checkpoints`, and `runs`
folders, then rerun the sweep command. Completed configs are skipped; the
interrupted config restarts from seed 42. Epoch-level resume is not implemented.

Keep CPU and Colab sweeps in separate output folders; do not combine their
ledgers. Download only your code/config/results for review. Do not commit datasets,
the upload archive, or checkpoints.
