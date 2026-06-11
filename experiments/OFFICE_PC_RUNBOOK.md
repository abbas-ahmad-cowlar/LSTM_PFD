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
