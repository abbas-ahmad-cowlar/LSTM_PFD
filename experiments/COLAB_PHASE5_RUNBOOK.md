# Colab Runbook — Phase-5 GPU experiments (§8.2–8.5)

> The ONLY file you need for the Phase-5 Colab session. 45 training runs,
> ~4–6 h on a T4. Everything lands on your Drive as it finishes, so
> disconnects never lose work.
>
> Run cells **in order, one at a time**. Each cell says what you should
> see. If a cell does not show the expected output, STOP and report.
> Run everything in **notebook cells** — the Terminal is a paid-plan
> feature and gives no advantage here (see "Disconnects" at the bottom).

Drive layout this runbook expects (owner's layout, confirmed 2026-06-13):

```
MyDrive/lstm-pfd/
├── data/dataset_v2.h5    <- you uploaded this (~1.8 GB; exact name matters)
├── results_benchmark/    <- Phase 4 results — nothing here ever writes to it
└── results_phase5/       <- created by Cell 4; all Phase-5 results land here
```

One-time notebook setup: colab.research.google.com → New notebook →
Runtime → Change runtime type → select **T4 GPU** → Save.

---

```python
# Cell 1 — GPU check + Drive mount (mount FIRST, before anything touches /content/drive)
!nvidia-smi -L
from google.colab import drive
drive.mount('/content/drive')
```

Expect: a line like `GPU 0: Tesla T4 ...`, then a popup asking you to
authorize Drive, then `Mounted at /content/drive`.
If `nvidia-smi` fails: Runtime → Change runtime type → T4 GPU, rerun the cell.

```python
# Cell 2 — get the code (~3 min; output appears only when the cell finishes)
%%bash
cd /content
git clone https://github.com/abbas-ahmad-cowlar/LSTM_PFD.git lstm-pfd
cd lstm-pfd
git checkout p5/physics-exp
pip -q install -r requirements.txt -r requirements-test.txt
```

Expect: `Switched to a new branch 'p5/physics-exp'`.
pip warnings are fine; red ERROR lines are not.
(Why clone to `/content` and not Drive: Drive is a slow network mount —
git and training through it are 10–100× slower. Code and data live on the
fast local disk; only results go to Drive.)

```python
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

```python
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

```python
# Cell 5 — smoke test: tiny 2-epoch version of every experiment (~10 min, live output)
%cd /content/lstm-pfd
!python scripts/run_phase5_gpu.py --smoke --seeds 0
```

Expect: log lines appear LIVE as it works (this cell uses `!`, which
streams; `%%bash` cells stay silent until done — that is normal).
Final line: `Phase-5 GPU queue finished: 15/15 complete.`
(Smoke output goes to a local throwaway folder, not to Drive.)

```python
# Cell 6 — THE REAL QUEUE: 45 runs, ~4–6 h on T4. Start it and leave the tab open.
%cd /content/lstm-pfd
!python scripts/run_phase5_gpu.py
```

Expect: live lines like `[1/45] data_efficiency/... — starting`, then
per-epoch progress. Final line: `Phase-5 GPU queue finished: 45/45 complete.`

```python
# Cell 7 — progress check (run anytime in a SEPARATE cell while Cell 6 works)
%%bash
tail -5 /content/lstm-pfd/logs/phase5_gpu.log
echo "runs finished, out of 45:"
find /content/drive/MyDrive/lstm-pfd/results_phase5 -name metrics.json | wc -l
```

Counting on the Drive side doubles as proof the results are being
persisted (the count can lag a minute behind the log).

---

## Disconnects & resuming (even days later)

A disconnect or a 2-day gap always means a **fresh machine** — nothing on
the old VM survives, and this is true for Terminal and cells alike. What
DOES survive is everything on Drive: every finished run's `metrics.json`
and the interrupted run's checkpoint (through the Cell-4 link).

To resume — today, tomorrow, or next week:
**re-run Cells 1, 2, 3, 4, then Cell 6.**
Finished runs are skipped automatically; the interrupted run continues
from its checkpoint. Nothing is ever lost or recomputed.

Optional, if you prefer several short sessions over one long one — append
one of these to the Cell-6 command:
`--only data_efficiency` (21 runs) · `--only severity_ood` (12) ·
`--only pinn_ablation` (9) · `--only true_metadata` (3).

## When the count reaches 45/45

Everything is already in `MyDrive/lstm-pfd/results_phase5/`. Back on the
laptop: download that folder from the Drive web UI, extract into
`results/phase5/`, and Claude verifies provenance + aggregates.

## Do NOT run these here

- `scripts/run_benchmark.py` — that is the finished Phase-4 matrix
  (results already in `results_benchmark/`); rerunning adds nothing.
- `dvc pull` — DVC is only for syncing the dataset between the owner's
  own machines; on Colab the dataset comes from your Drive (Cell 3).
