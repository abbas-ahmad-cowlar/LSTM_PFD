# Colab runbook — F9 scrambled-reference control (PROTOCOL §8.7)

**Owner-operated GPU run.** This runs the ONE decisive control that tells us whether
the surviving §8.4 noise-robustness benefit is **physics-specific** or just
**generic high-weight regularization**:

> `physics_constrained_cnn`, **w=1.0**, with the per-class band targets **scrambled**
> (each fault judged against a *different* fault's bands + healthy reference — same
> band-energy loss strength/structure, **wrong physics**), × seeds {0,1,2}, eval
> clean + 5 dB.

The decision rule is pre-registered (`experiments/PROTOCOL.md` §8.7). After the 3
runs come home, **Claude runs the record-level comparison on the laptop** (vs the
existing §8.4 w=0 CE-only and w=1.0 correct-physics arms) — that produces the verdict.

- **Compute:** Colab **T4** (Runtime → Change runtime type → T4 GPU). ~15–45 min.
- **Why GPU here:** training; the laptop is CPU-only.
- This control is **additive** — it does not retrain or change §8.2–§8.6.

---

## Cell 1 — GPU check + clone `p6/docs`
```bash
!nvidia-smi -L
!git clone -b p6/docs https://github.com/abbas-ahmad-cowlar/LSTM_PFD.git /content/lstm-pfd
%cd /content/lstm-pfd
!git rev-parse --short HEAD
!pip -q install -r requirements.txt
```
**Expect:** a `Tesla T4` line and a short HEAD sha. **STOP** if no GPU (you'll get a
CPU and it will be slow) — switch the runtime to T4 first.

## Cell 2 — mount Drive + copy the dataset
```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p data/generated
!cp /content/drive/MyDrive/lstm-pfd/data/dataset_v2.h5 data/generated/dataset_v2.h5
!python -c "import h5py; print('OK groups:', list(h5py.File('data/generated/dataset_v2.h5').keys()))"
```
**Expect:** `OK groups: ['test', 'test_snr10', 'test_snr20', 'test_snr5', 'train', 'val']`.
**STOP** if the file is missing — fix the Drive path (it should match where
`dataset_v2.h5` lives in your Drive) before continuing.

## Cell 3 — sanity-check the control is wired (no training)
```bash
!python -c "
from packages.core.models.pinn.physics_constrained_cnn import PhysicsConstrainedCNN
from scripts.run_phase5_gpu import scrambled_class_permutation
m = PhysicsConstrainedCNN(backbone='cnn1d')
print('default reference_permutation:', m.reference_permutation)   # must be None
p = scrambled_class_permutation()
print('scramble permutation:', p)
assert p[0] == 0 and all(p[i] != i for i in range(1, 11)), 'NOT a derangement'
print('derangement OK (no fault maps to its own bands)')
"
```
**Expect:** `default reference_permutation: None`, an 11-element permutation, and
`derangement OK`. **STOP** if it errors.

## Cell 4 — persist outputs to Drive (resume-safe), then GPU smoke
```bash
# the scramble output dir does NOT exist in the repo, so this symlink is clean
# (no I2 nesting trap — unlike results/phase5 which ships populated):
!mkdir -p /content/drive/MyDrive/lstm-pfd/f9_scramble
!ln -s /content/drive/MyDrive/lstm-pfd/f9_scramble results/phase5_bandenergy/pinn_ablation_scramble
!ls -la results/phase5_bandenergy/pinn_ablation_scramble   # must show a  ->  arrow to Drive
# 2-epoch smoke (writes to results/phase5_bandenergy_smoke/, NOT the symlinked dir):
!python scripts/run_phase5_gpu.py --control f9_scramble --smoke --seeds 0
```
**Expect:** an `F9 SCRAMBLE reference_permutation=[...]` line, 2 epochs, a `DONE` line.
The smoke writes to a separate `_smoke` folder, so it does **not** pollute the real
run. **STOP** if you see a `FAILED`/traceback.

## Cell 5 — the real control run (3 seeds)
```bash
!python scripts/run_phase5_gpu.py --control f9_scramble
```
**Expect:** `Queue: 3 runs`, and for each seed an `F9 SCRAMBLE ...` line, ~5–15 min of
epochs, and a `DONE: test XX.XX% | test_snr5 YY.YY%` line. Re-running is **resume-safe**
(finished seeds are skipped; an interrupted seed resumes from its checkpoint via the
Drive symlink). Clean test should land ~96%; the printed 5 dB number is *window-level*
— the real verdict is the laptop record-level analysis (next).

## Cell 6 — verify + confirm outputs are on Drive
```bash
!python -c "
import json, glob
for f in sorted(glob.glob('results/phase5_bandenergy/pinn_ablation_scramble/w1.0/seed*/metrics.json')):
    m = json.load(open(f))
    print(f.split('pinn_ablation_scramble/')[1], '|',
          'perm', m.get('reference_permutation'), '|',
          {k: round(v['accuracy'],2) for k,v in m['evals'].items()})
"
!ls -la /content/drive/MyDrive/lstm-pfd/f9_scramble/w1.0/seed*/best_model.pth
```
**Expect:** 3 lines (seed0/1/2), each showing the recorded `reference_permutation`
(same for all) and `test`/`test_snr5` accuracies; and 3 `best_model.pth` files on
Drive (~46 MB each). **STOP here.**

---

## Then — bring it home (laptop)
The 3 runs are on your Drive at `MyDrive/lstm-pfd/f9_scramble/`. Download that folder
(metrics.json + best_model.pth per seed) and drop it into the repo at
`results/phase5_bandenergy/pinn_ablation_scramble/` on the laptop (the checkpoints are
gitignored; the 3 `metrics.json` get committed). Then tell Claude — it will run the
**record-level** comparison (scramble-w1.0 vs CE-only vs correct-w1.0 @5 dB, McNemar)
and report the §8.7 verdict per the pre-registered decision rule.

## Two-line reminder of what the result means (PROTOCOL §8.7)
- Scramble degrades at 5 dB **like CE-only** (≫ correct w=1.0) → the benefit is
  **physics-specific** → the paper may say "physics."
- Scramble **stays robust** (like correct w=1.0) → the benefit is **generic
  regularization** → narrow the claim to "the band-energy term helped."
