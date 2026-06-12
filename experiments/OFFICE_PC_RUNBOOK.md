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

Drive layout this appendix expects (owner's layout, confirmed 2026-06-13):

```
MyDrive/lstm-pfd/
├── data/dataset_v2.h5    <- you uploaded this (~1.8 GB; exact name matters)
├── results_benchmark/    <- Phase 4 results — nothing below ever writes here
└── results_phase5/       <- created automatically by Cell 4; Phase-5 results land here
```

One-time notebook setup: colab.research.google.com → New notebook →
Runtime → Change runtime type → select **T4 GPU** → Save.

Run the cells below **in order, one at a time**. Each cell says what you
should see. If a cell does not show the expected output, STOP — do not
run the next cell.

```python
# Cell 1 — GPU check + Drive mount (mount FIRST, before anything touches /content/drive)
!nvidia-smi -L
from google.colab import drive
drive.mount('/content/drive')
```

Expect: a line like `GPU 0: Tesla T4 ...`, then a popup asking you to
authorize Drive, then `Mounted at /content/drive`.
If `nvidia-smi` fails: Runtime → Change runtime type → T4 GPU, rerun the cell.

```bash
# Cell 2 — get the code (~3 min)
%%bash
cd /content
git clone https://github.com/abbas-ahmad-cowlar/LSTM_PFD.git lstm-pfd
cd lstm-pfd
git checkout p5/physics-exp
pip -q install -r requirements.txt -r requirements-test.txt
```

Expect: `Switched to a new branch 'p5/physics-exp'` near the top.
pip warnings are fine; red ERROR lines are not.

```bash
# Cell 3 — copy the dataset from Drive to Colab's fast local disk (~1 min)
%%bash
ls -lh /content/drive/MyDrive/lstm-pfd/data/dataset_v2.h5
mkdir -p /content/lstm-pfd/data/generated
cp /content/drive/MyDrive/lstm-pfd/data/dataset_v2.h5 /content/lstm-pfd/data/generated/
ls -lh /content/lstm-pfd/data/generated/dataset_v2.h5
```

Expect: TWO lines, both showing **1.8G**. If the first `ls` says
"No such file or directory", the Drive upload hasn't finished yet —
wait for it to complete in the Drive web UI, then rerun this cell.

```bash
# Cell 4 — connect results to Drive (MUST happen before the first run)
%%bash
mkdir -p /content/drive/MyDrive/lstm-pfd/results_phase5
ln -sfn /content/drive/MyDrive/lstm-pfd/results_phase5 /content/lstm-pfd/results/phase5
ls -la /content/lstm-pfd/results/ | grep phase5
```

Expect EXACTLY one line ending in:
`phase5 -> /content/drive/MyDrive/lstm-pfd/results_phase5`
If the `->` arrow is missing, STOP and report — otherwise results die
with the session. (`results_benchmark` is a separate folder; untouched.)

```bash
# Cell 5 — smoke test: tiny 2-epoch version of every experiment (~10 min)
%%bash
cd /content/lstm-pfd
python scripts/run_phase5_gpu.py --smoke --seeds 0
```

Expect final line: `Phase-5 GPU queue finished: 15/15 complete.`
(Smoke output goes to a local throwaway folder, not to Drive.)

```bash
# Cell 6 — THE REAL QUEUE: 45 runs, ~4-6 h on T4. Start it and leave it running.
%%bash
cd /content/lstm-pfd
python scripts/run_phase5_gpu.py
```

Expect: lines like `[1/45] data_efficiency/... — starting`, then per-epoch
progress. Final line: `Phase-5 GPU queue finished: 45/45 complete.`

```bash
# Cell 7 — progress check (run anytime in a SEPARATE cell while Cell 6 works)
%%bash
tail -5 /content/lstm-pfd/logs/phase5_gpu.log
echo "runs finished, out of 45:"
find /content/drive/MyDrive/lstm-pfd/results_phase5 -name metrics.json | wc -l
```

Counting on the Drive side doubles as proof the results are being
persisted (the count can lag a minute behind the log).

**If Colab disconnects** (it eventually will): you get a fresh machine.
Re-run Cells 1, 2, 3, 4 — then Cell 6. Already-finished runs are skipped
and the interrupted run resumes from its checkpoint; both live on Drive
through the Cell-4 link, so nothing is lost.

Optional, if you prefer several short sessions over one long one — append
one of these to the Cell-6 command:
`--only data_efficiency` (21 runs) · `--only severity_ood` (12) ·
`--only pinn_ablation` (9) · `--only true_metadata` (3).

**When the count reaches 45/45**: everything is already in
`MyDrive/lstm-pfd/results_phase5/`. Back on the laptop: download that
folder from the Drive web UI, extract into `results/phase5/`, and Claude
verifies provenance + aggregates.
