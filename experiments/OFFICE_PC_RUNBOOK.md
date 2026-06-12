# Office GPU PC Runbook — Phase 4 Benchmark Matrix

> Copy-paste guide for running the 8-model × 3-seed matrix on the office PC.
> Total: 24 runs, expected ~12–48 GPU-hours depending on the card.
> The queue is resume-safe: kill it, reboot, rerun the same command — it
> skips finished runs and resumes the interrupted one from checkpoint.

## 1. One-time setup (~20 min)

```powershell
# Clone (or pull) the repo
git clone https://github.com/abbas-ahmad-cowlar/LSTM_PFD.git C:\work\lstm-pfd
cd C:\work\lstm-pfd
git checkout main   # or the active phase branch Claude tells you

# Python env (3.12+ recommended; 3.14 is what the laptop runs)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu121   # GPU build
pip install -r requirements.txt -r requirements-test.txt
pip freeze > requirements.lock.gpu.txt   # commit this later (P4.1 DoD)

# Verify GPU is visible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Get the dataset (1.9 GB) — either:
dvc pull data/generated/dataset_v2.h5.dvc     # if DVC remote is configured
# ...or copy data\generated\dataset_v2.h5 from the laptop via USB/network
# to data\generated\dataset_v2.h5 (path must match exactly).

# Sanity: suite + 2-epoch smoke queue (~10 min)
pytest -q
python scripts\run_benchmark.py --smoke --models cnn1d hybrid_pinn --seeds 0
```

## 2. Launch the real matrix (DETACHED — survives logout/app closure)

```powershell
cd C:\work\lstm-pfd
Start-Process -FilePath "C:\work\lstm-pfd\venv\Scripts\python.exe" `
  -ArgumentList "scripts/run_benchmark.py" `
  -WorkingDirectory "C:\work\lstm-pfd" -WindowStyle Hidden
```

Keep the machine awake:

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

## 3. Watch progress

```powershell
Get-Content C:\work\lstm-pfd\logs\benchmark_queue.log -Tail 10 -Wait
```

Count completed runs at any time:

```powershell
(Get-ChildItem C:\work\lstm-pfd\results\benchmark\deep -Recurse -Filter metrics.json).Count
# Done when it reaches 24
```

## 4. If something dies (power cut, reboot, anything)

Just rerun the same Start-Process command from §2. Completed runs are
skipped (metrics.json present); the interrupted run resumes from its own
checkpoint. Nothing is lost except the in-flight epoch.

## 5. Ship results back

The small artifacts (metrics.json files) are what matters; checkpoints stay
on the office PC for now.

```powershell
cd C:\work\lstm-pfd
git checkout -b p4/benchmark-results
git add results/benchmark
git commit -m "P4.4: benchmark matrix results from office GPU"
git push -u origin p4/benchmark-results
```

(Or zip `results\benchmark\` and transfer it to the laptop — Claude merges it.)

## 6. Optional: Tier-2 extension (only if the matrix finished early)

```powershell
Start-Process -FilePath "C:\work\lstm-pfd\venv\Scripts\python.exe" `
  -ArgumentList "scripts/run_benchmark.py","--models","multi_scale_cnn","se_resnet18","signal_transformer" `
  -WorkingDirectory "C:\work\lstm-pfd" -WindowStyle Hidden
```

---

## Appendix: Google Colab lane (Linux — different commands!)

> **HISTORICAL — Phase 4 only, already complete** (results live in
> `results/benchmark/`). Do NOT run these cells again — Cell 2 checks out
> the old `p4/benchmark` branch and Cell 6 reruns the finished matrix.
> **For Phase 5 use `experiments/COLAB_PHASE5_RUNBOOK.md`.**

The main runbook is for the Windows office PC. On Colab, paths use **forward
slashes** (`scripts/run_benchmark.py`, never `scripts\...` — bash eats
backslashes), and sessions are **ephemeral** (~12 h max, idle disconnects),
so results must live on Google Drive to survive. The queue's resume-safety
makes disconnects cheap: rerun the launch cell and it continues.

Paste these as notebook cells:

```python
# Cell 1 — GPU check + mount Drive (approve the popup)
!nvidia-smi -L
from google.colab import drive
drive.mount('/content/drive')
```

```bash
# Cell 2 — clone + env (use %%bash or ! prefixes)
%%bash
cd /content
git clone https://github.com/abbas-ahmad-cowlar/LSTM_PFD.git lstm-pfd
cd lstm-pfd
git checkout p4/benchmark          # active phase branch
pip -q install -r requirements.txt -r requirements-test.txt   # torch+CUDA is preinstalled on Colab
```

```bash
# Cell 3 — dataset: keep the master copy on Drive, copy to fast local disk
%%bash
mkdir -p /content/lstm-pfd/data/generated
cp "/content/drive/MyDrive/lstm-pfd/dataset_v2.h5" /content/lstm-pfd/data/generated/dataset_v2.h5
ls -lh /content/lstm-pfd/data/generated/
```

```bash
# Cell 4 — persist results on Drive (symlink BEFORE any run)
%%bash
mkdir -p "/content/drive/MyDrive/lstm-pfd/results_benchmark"
mkdir -p /content/lstm-pfd/results
ln -sfn "/content/drive/MyDrive/lstm-pfd/results_benchmark" /content/lstm-pfd/results/benchmark
```

```bash
# Cell 5 — sanity (suite + 2-run smoke queue, ~5 min on T4)
%%bash
cd /content/lstm-pfd
pytest -q
python scripts/run_benchmark.py --smoke --models cnn1d hybrid_pinn --seeds 0
```

```bash
# Cell 6 — the matrix (rerun this exact cell after any disconnect; it resumes)
%%bash
cd /content/lstm-pfd
python scripts/run_benchmark.py
```

```bash
# Cell 7 — progress check (run anytime in a separate cell)
%%bash
tail -5 /content/lstm-pfd/logs/benchmark_queue.log
find /content/lstm-pfd/results/benchmark/deep -name metrics.json | wc -l   # done at 24
```

**After a disconnect**: rerun Cells 2–4 and 6 (clone+env ~3 min; results and
checkpoints are already on Drive through the symlink, so completed runs are
skipped and the interrupted one resumes from its checkpoint).

**Shipping results back**: zip the Drive folder or commit from Colab:
```bash
%%bash
cd /content/lstm-pfd
git checkout -b p4/benchmark-results
cp -r /content/drive/MyDrive/lstm-pfd/results_benchmark/* results/benchmark/ 2>/dev/null || true
git add results/benchmark && git commit -m "P4.4: matrix results (Colab T4)"
# push needs a GitHub token: git push https://<TOKEN>@github.com/abbas-ahmad-cowlar/LSTM_PFD.git p4/benchmark-results
```

---

## Appendix: Phase-5 GPU experiments on Colab (§8.2–8.5)

**Moved to its own clean, Colab-only runbook:
`experiments/COLAB_PHASE5_RUNBOOK.md`** — that file contains nothing but
the 7 cells for the Phase-5 session (no office-PC/Windows/DVC content).
