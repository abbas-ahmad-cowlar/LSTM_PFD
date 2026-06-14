# Colab Runbook — Phase-5 §8.4 rerun with the FIXED (differentiable) physics loss

> Run this AFTER the inert 45/45 run is complete. It re-runs only §8.4
> (`pinn_ablation`, 9 runs, ~2–3 h on a T4) on the fix branch, writing to a
> SEPARATE Drive folder `results_phase5_fixed/` so the inert "before" results
> in `results_phase5/` are never touched. Same frozen budget, same seeds.
>
> Two things differ from the main runbook, both critical:
>   - Cell 2 checks out **`p5/physics-loss-fix`** (the differentiable loss).
>   - Cell 4 symlinks to **`results_phase5_fixed`** (a NEW empty folder — if it
>     pointed at the existing `results_phase5`, the queue would see the 9
>     inert metrics.json and skip everything).

Drive layout after this run:

```
MyDrive/lstm-pfd/
├── data/dataset_v2.h5
├── results_benchmark/        <- Phase 4, untouched
├── results_phase5/           <- inert "before" (45 runs), untouched
└── results_phase5_fixed/     <- created by Cell 4; the fixed §8.4 "after" lands here
```

Use a T4 runtime (Runtime → Change runtime type → T4 GPU). Cells in order.

```python
# Cell 1 — GPU check + Drive mount
!nvidia-smi -L
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 2 — get the code on the FIX branch (~3 min)
%%bash
cd /content
rm -rf lstm-pfd                 # drop any clone left from the inert run
git clone https://github.com/abbas-ahmad-cowlar/LSTM_PFD.git lstm-pfd
cd lstm-pfd
git checkout p5/physics-loss-fix
git log --oneline -1            # expect e401681 (or later) -- the fixed loss
pip -q install -r requirements.txt -r requirements-test.txt
```

Expect: `Switched to ... 'p5/physics-loss-fix'` and the `git log` line showing
the fixed-loss commit. THIS is the only run that uses this branch.

```python
# Cell 3 — copy the dataset to fast local disk (~1 min)
%%bash
ls -lh /content/drive/MyDrive/lstm-pfd/data/dataset_v2.h5
mkdir -p /content/lstm-pfd/data/generated
cp /content/drive/MyDrive/lstm-pfd/data/dataset_v2.h5 /content/lstm-pfd/data/generated/
ls -lh /content/lstm-pfd/data/generated/dataset_v2.h5
```

Expect: two lines, both **1.8G**.

```python
# Cell 4 — results to a NEW Drive folder (results_phase5_fixed) -- MUST be empty
%%bash
mkdir -p /content/drive/MyDrive/lstm-pfd/results_phase5_fixed
ln -sfn /content/drive/MyDrive/lstm-pfd/results_phase5_fixed /content/lstm-pfd/results/phase5
ls -la /content/lstm-pfd/results/ | grep phase5
echo "existing fixed runs (should be 0 on a fresh run):"
find /content/drive/MyDrive/lstm-pfd/results_phase5_fixed -name metrics.json | wc -l
```

Expect: `phase5 -> /content/drive/MyDrive/lstm-pfd/results_phase5_fixed`
and an existing-count of **0**. (The `results_phase5` inert folder is a
different name — left alone.)

```python
# Cell 5 — smoke the FIXED loss on GPU (2-epoch, 3 runs, ~5 min). Recommended:
# this is new code on the GPU/AMP path; the smoke catches any issue cheaply.
%cd /content/lstm-pfd
!python scripts/run_phase5_gpu.py --smoke --only pinn_ablation --seeds 0
```

Expect: live per-epoch lines with a NON-zero, VARYING `phys` value, and final
`Phase-5 GPU queue finished: 3/3 complete.` (Smoke uses a throwaway local
folder, not Drive.)

```python
# Cell 6 — the fixed §8.4 rerun: 9 runs (w in {0.1,0.3,1.0} x 3 seeds), ~2-3 h
%cd /content/lstm-pfd
!python scripts/run_phase5_gpu.py --only pinn_ablation
```

Expect: `[1/9] pinn_ablation/w0.1/seed0 — starting` ... →
`Phase-5 GPU queue finished: 9/9 complete.`

```python
# Cell 7 — progress (separate cell after a reconnect, or the Terminal anytime)
%%bash
tail -5 /content/lstm-pfd/logs/phase5_gpu.log
echo "fixed runs finished, out of 9:"
find /content/drive/MyDrive/lstm-pfd/results_phase5_fixed -name metrics.json | wc -l
```

Disconnect/resume: same as the main runbook — rerun Cells 1,2,3,4 then 6; the
fixed folder on Drive means finished runs skip and the interrupted one resumes.

When it reads 9/9: download `results_phase5_fixed` from the Drive web UI,
extract into `results/phase5_fixed/` on the laptop, and Claude builds the
before-vs-after §8.4 table.

---

## §8.2 data-efficiency rerun with the fixed loss (the low-data physics test)

The one principled "fair shot" for physics: low-data is the regime where
physics priors should help most, and §8.4 only tested full data. Same fixed
branch, a NEW Drive folder, and the data-efficiency subset. Identical to the
cells above with **two changes** (Cells 1–3 unchanged: mount, fix-branch
clone, dataset):

```python
# Cell 4 — NEW folder results_phase5_dataeff_fixed (must be empty -> nothing skipped)
%%bash
mkdir -p /content/drive/MyDrive/lstm-pfd/results_phase5_dataeff_fixed
ln -sfn /content/drive/MyDrive/lstm-pfd/results_phase5_dataeff_fixed /content/lstm-pfd/results/phase5
ls -la /content/lstm-pfd/results/ | grep phase5
find /content/drive/MyDrive/lstm-pfd/results_phase5_dataeff_fixed -name metrics.json | wc -l   # expect 0
```

```python
# Cell 6 — data-efficiency only: 21 runs (pc_cnn fixed + resnet18 x {10,25,50,100}% x 3), ~1-1.5 h
%cd /content/lstm-pfd
!python scripts/run_phase5_gpu.py --only data_efficiency
```

Expect `21/21 complete`. (resnet18 + pc_cnn-100% reproduce earlier numbers —
a free cross-check; the new science is pc_cnn-fixed at 10/25/50%.) Then
download `results_phase5_dataeff_fixed` → `results/phase5_dataeff_fixed/` and
Claude builds the fixed data-efficiency curve vs the inert one and vs vanilla.
