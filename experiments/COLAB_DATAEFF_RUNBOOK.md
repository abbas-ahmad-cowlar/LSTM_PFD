# Colab Runbook — RUN THIS NOW (§8.2 data-efficiency, fixed physics loss)

> **This is the only thing to run right now.** It is fully self-contained and
> starts FRESH — no previous results, no resume, no folder-merging. 21 training
> runs, ~1–1.5 h on a T4.
>
> (The older `COLAB_PHASE5_RUNBOOK.md` and `COLAB_PHASE5_FIXED_RUNBOOK.md` are
> finished/historical — ignore them.)

## What you need on Drive (nothing new)

Only the dataset you already have:

```
MyDrive/lstm-pfd/
└── data/dataset_v2.h5      <- already there (~1.8 GB). That's all you need.
```

You do NOT need to upload any results. This run creates its own fresh output
folder `results_phase5_dataeff_fixed/` automatically. Leave your other Drive
folders alone.

## Notebook to create

1. colab.research.google.com → **New notebook**.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Paste the cells below in order, one at a time. Each says what you should see;
   if a cell's output doesn't match, STOP and report it.

---

```python
# Cell 1 — GPU + mount Drive
!nvidia-smi -L
from google.colab import drive
drive.mount('/content/drive')
```
Expect: a `Tesla T4` line, a Drive auth popup, then `Mounted at /content/drive`.

```python
# Cell 2 — get the FIXED-loss code (~3 min)
%%bash
cd /content
rm -rf lstm-pfd
git clone https://github.com/abbas-ahmad-cowlar/LSTM_PFD.git lstm-pfd
cd lstm-pfd
git checkout p5/physics-loss-fix
git log --oneline -1
pip -q install -r requirements.txt -r requirements-test.txt
```
Expect: `Switched to ... 'p5/physics-loss-fix'` and a `git log` line. The branch
name **must** be `p5/physics-loss-fix` (this is the differentiable-loss code).

```python
# Cell 3 — copy the dataset to fast local disk (~1 min)
%%bash
mkdir -p /content/lstm-pfd/data/generated
cp /content/drive/MyDrive/lstm-pfd/data/dataset_v2.h5 /content/lstm-pfd/data/generated/
ls -lh /content/lstm-pfd/data/generated/dataset_v2.h5
```
Expect: one line showing **1.8G**.

```python
# Cell 4 — point results at a NEW empty Drive folder
%%bash
mkdir -p /content/drive/MyDrive/lstm-pfd/results_phase5_dataeff_fixed
ln -sfn /content/drive/MyDrive/lstm-pfd/results_phase5_dataeff_fixed /content/lstm-pfd/results/phase5
ls -la /content/lstm-pfd/results/ | grep phase5
find /content/drive/MyDrive/lstm-pfd/results_phase5_dataeff_fixed -name metrics.json | wc -l
```
Expect: a line ending `phase5 -> /content/drive/MyDrive/lstm-pfd/results_phase5_dataeff_fixed`,
and the count line printing **0** (fresh start). If the `->` arrow is missing, STOP.

```python
# Cell 5 — THE RUN: 21 data-efficiency runs (~1–1.5 h). Start it and leave the tab open.
%cd /content/lstm-pfd
!python scripts/run_phase5_gpu.py --only data_efficiency
```
Expect: live `[1/21] data_efficiency/... — starting` lines with per-epoch
progress, ending at `Phase-5 GPU queue finished: 21/21 complete.`

```python
# Cell 6 — progress check (only needed after a disconnect, before re-running Cell 5)
%%bash
tail -5 /content/lstm-pfd/logs/phase5_gpu.log
echo "runs finished, out of 21:"
find /content/drive/MyDrive/lstm-pfd/results_phase5_dataeff_fixed -name metrics.json | wc -l
```

## If it disconnects

Re-run **Cells 1, 2, 3, 4, then 5**. Finished runs are already on Drive
(through the Cell-4 link), so they're skipped and only the rest run. Nothing is
lost. (Cell 4's count will show how many are already done.)

## When it reaches 21/21

Download `results_phase5_dataeff_fixed` from the Drive web UI, drop it in
`results/` on the laptop, and tell Claude — he builds the fixed-vs-inert-vs-
vanilla data-efficiency curves (the low-data physics test).
