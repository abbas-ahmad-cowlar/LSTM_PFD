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
# Cell 7 — progress check
%%bash
tail -5 /content/lstm-pfd/logs/phase5_gpu.log
echo "runs finished, out of 45:"
find /content/drive/MyDrive/lstm-pfd/results_phase5 -name metrics.json | wc -l
```

Counting on the Drive side doubles as proof the results are being
persisted (the count can lag a minute behind the log).

WHEN to use Cell 7: a notebook runs one cell at a time, so while Cell 6
is busy this cell would just queue behind it. While Cell 6 runs, either
read its own streaming output (the `[N/45]` and `DONE:` lines), or paste
the three commands above into the Terminal if your plan has one. Use
Cell 7 as a cell after a reconnect, before restarting the queue.

Terminal-only self-refreshing monitor (updates every 30 s; Ctrl+C to
exit the display — training is unaffected):

```bash
watch -n 30 'echo "== latest log =="; tail -3 /content/lstm-pfd/logs/phase5_gpu.log; echo; echo "== finished runs on Drive (of 45) =="; find /content/drive/MyDrive/lstm-pfd/results_phase5 -name metrics.json | wc -l; echo; echo "== GPU compute % =="; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
```

Note: the Colab resources panel shows GPU MEMORY, not compute usage.
~1.2/15 GB is correct for these small models at the frozen batch size
(64); do not try to fill the GPU — that would change the protocol.

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

### Resuming on a NEW account/Drive (from a downloaded results folder)

The skip-logic only sees runs whose `metrics.json` sits at the exact path
Cell 4 symlinks to: `MyDrive/lstm-pfd/results_phase5/<experiment>/...`. If you
upload a Drive **download** of prior results, its folder is named like
`results_phase5-<timestamp>-NNN` and contains a nested `results_phase5/`
inside — so the runs are NOT where the symlink looks, and the queue restarts
from `[1/45]`. After Cells 1–4 (which create an empty `results_phase5`),
relocate the runs into it BEFORE Cell 6:

```bash
%%bash
# point SRC at the inner results_phase5 of your uploaded download folder
SRC="/content/drive/MyDrive/lstm-pfd/results_phase5-<TIMESTAMP>/results_phase5"
DST="/content/drive/MyDrive/lstm-pfd/results_phase5"
n=$(find "$SRC" -name metrics.json 2>/dev/null | wc -l)
echo "found $n metrics.json under SRC"
if [ "$n" -lt 1 ]; then
  echo "SRC path wrong -- dirs under the upload:"
  find "$(dirname "$SRC")" -maxdepth 2 -type d
else
  cp -r "$SRC"/* "$DST"/
  echo "results_phase5 now has $(find "$DST" -name metrics.json | wc -l) metrics.json"
fi
```

Verify the count matches what you finished (e.g. 39), THEN run Cell 6 — it will
log `complete, skip` for each and only train the remainder. (Simpler next time:
use the small `logs/phase5_resume_bundle.zip` Claude can build — extract it so
its `results_phase5/` lands directly at `MyDrive/lstm-pfd/results_phase5/`.)

## When the count reaches 45/45

Everything is already in `MyDrive/lstm-pfd/results_phase5/`. Back on the
laptop: download that folder from the Drive web UI, extract into
`results/phase5/`, and Claude verifies provenance + aggregates.

## Do NOT run these here

- `scripts/run_benchmark.py` — that is the finished Phase-4 matrix
  (results already in `results_benchmark/`); rerunning adds nothing.
- `dvc pull` — DVC is only for syncing the dataset between the owner's
  own machines; on Colab the dataset comes from your Drive (Cell 3).
