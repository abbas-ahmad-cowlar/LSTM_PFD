# Colab Runbook — Step 5 physics reruns (band-energy loss, record-level)

> **Run this to redo the physics-loss experiments with the CORRECTED loss.**
> P6 remediation Step 5. The model-side physics was rebuilt in Steps 1–4:
> journal-bearing signature DB, a **differentiable band-energy `compute_physics_loss`**
> (tonal + broadband + mixed), and **per-sample rpm** wired into the experiments.
> This rerun replaces the contaminated §8.2/§8.3/§8.4 results.
>
> **Scope:** §8.4 (ablation), §8.2 (data-efficiency), §8.3 (severity-OOD) — **42
> runs, ~3–5 h on a T4**, resume-safe. It does **NOT** run §8.5 (`true_metadata`):
> HybridPINN's physics branch is still rolling-element and must be rebuilt first
> (separate task). §8.6a XAI is recomputed on the laptop afterward, not here.
>
> Older runbooks (`COLAB_DATAEFF_RUNBOOK.md`, `COLAB_PHASE5_*`) are historical —
> ignore them; their results are the contaminated ones we are replacing.

## What you need on Drive (nothing new)

Only the dataset you already have:

```
MyDrive/lstm-pfd/
└── data/dataset_v2.h5      <- already there (~1.8 GB). That's all you need.
```

This run creates its own fresh output folder `results_phase5_bandenergy/`
automatically. Leave your other Drive folders alone.

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
# Cell 2 — get the BAND-ENERGY code (branch p6/docs) (~3 min)
%%bash
cd /content
rm -rf lstm-pfd
git clone https://github.com/abbas-ahmad-cowlar/LSTM_PFD.git lstm-pfd
cd lstm-pfd
git checkout p6/docs
git log --oneline -1
pip -q install -r requirements.txt -r requirements-test.txt
```
Expect: `Switched to ... 'p6/docs'` and a recent `git log` line mentioning
**P6 Step 4 / band-energy**. The branch name **must** be `p6/docs`.

```python
# Cell 3 — copy the dataset to fast local disk (~1 min)
%%bash
mkdir -p /content/lstm-pfd/data/generated
cp /content/drive/MyDrive/lstm-pfd/data/dataset_v2.h5 /content/lstm-pfd/data/generated/
ls -lh /content/lstm-pfd/data/generated/dataset_v2.h5
```
Expect: one line showing **1.8G**.

```python
# Cell 4 — VERIFY this runtime has the corrected loss + healthy reference (~30 s). Do not skip.
%cd /content/lstm-pfd
!PYTHONIOENCODING=utf-8 python -m pytest tests/test_physics_band_energy_loss.py tests/test_signature_db_consistency.py -q
```
Expect: `... passed` (no failures). This proves the band-energy loss, the
journal-bearing signature DB, and the **frozen healthy-class reference**
(`packages/core/models/physics/healthy_reference.json`, committed) are the code
about to train — the CI test checks each fault's signature against the real
healthy baseline on the dataset you just copied. If anything fails, STOP and
report it — do not run the experiments on wrong code.

```python
# Cell 5 — point results at a NEW empty Drive folder (BEFORE the first run)
%%bash
mkdir -p /content/drive/MyDrive/lstm-pfd/results_phase5_bandenergy
ln -sfn /content/drive/MyDrive/lstm-pfd/results_phase5_bandenergy /content/lstm-pfd/results/phase5
ls -la /content/lstm-pfd/results/ | grep phase5
find /content/drive/MyDrive/lstm-pfd/results_phase5_bandenergy -name metrics.json | wc -l
```
Expect: a line ending `phase5 -> /content/drive/MyDrive/lstm-pfd/results_phase5_bandenergy`,
and the count line printing **0** (fresh start). If the `->` arrow is missing, STOP.
(This is the `ln -sfn`-before-first-run step — it must run BEFORE Cell 7, or the
queue writes to local disk and the results vanish on disconnect.)

```python
# Cell 6 — GPU smoke sanity (~3 min). Confirms the loss runs on the T4 before the real thing.
%cd /content/lstm-pfd
!python scripts/run_phase5_gpu.py --only pinn_ablation --smoke --seeds 0
!rm -rf results/phase5_smoke
```
Expect: 3 short runs with per-epoch lines that include a non-zero, generally
**decreasing `phys`** value (e.g. `phys 0.14` → `phys 0.11`). A decreasing phys
loss = the band-energy term is differentiable and actually training. Then the
smoke folder is deleted so it doesn't mix with the real output.

```python
# Cell 7 — THE RUN: 42 physics reruns (~3–5 h). Start it and leave the tab open.
%cd /content/lstm-pfd
!python scripts/run_phase5_gpu.py --only pinn_ablation      # §8.4  (9 runs)
!python scripts/run_phase5_gpu.py --only data_efficiency    # §8.2  (21 runs)
!python scripts/run_phase5_gpu.py --only severity_ood       # §8.3  (12 runs)
```
Each command is independent and resume-safe (a finished run is skipped). Expect
live `[k/N] ... — starting` lines with per-epoch progress, each ending
`Phase-5 GPU queue finished: N/N complete.` Do NOT run the bare
`run_phase5_gpu.py` with no `--only` — that would also run §8.5 (`true_metadata`),
which is still on the rolling-element HybridPINN and must not be rerun yet.

```python
# Cell 8 — progress check (run anytime, and after a disconnect before re-running Cell 7)
%%bash
tail -5 /content/lstm-pfd/logs/phase5_gpu.log
echo "runs finished so far (target 42):"
find /content/drive/MyDrive/lstm-pfd/results_phase5_bandenergy -name metrics.json | wc -l
```

## If it disconnects

Re-run **Cells 1, 2, 3, 4, 5, then 7** (you can skip Cell 6 on a resume).
Finished runs are already on Drive (through the Cell-5 link), so they're skipped
and only the rest run. Nothing is lost. Cell 8 shows how many of 42 are done.

## When it reaches 42/42

1. In the Drive web UI, download the folder `results_phase5_bandenergy`.
2. On the laptop, extract it into `results/` (it arrives nested as
   `results_phase5_bandenergy-<timestamp>/results_phase5_bandenergy/…` — move the
   inner `results_phase5_bandenergy` so it sits directly under `results/`).
3. Tell Claude. He will: verify counts + provenance, recompute the §8.2/§8.3/§8.4
   numbers **at record level** (the Step-2 method), recompute §8.6a XAI alignment
   on the laptop against the corrected bands, and then rewrite + re-ratify
   FINDINGS. **No physics claim is made until those record-level reruns support it.**

## What this does and does not establish

- **Does:** tests whether the *corrected* journal-bearing band-energy physics loss
  (with per-sample rpm) helps accuracy / data-efficiency / severity-OOD, judged at
  the record level — the experiment that had not actually been run.
- **Does not:** rehabilitate §8.5 (HybridPINN, still rolling-element) or make any
  physics-benefit claim on its own. Results are reported honestly either way.
