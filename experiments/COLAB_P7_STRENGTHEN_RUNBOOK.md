# Colab runbook — §8.8 strengthen grid (random-band control + n=12 seeds)

**Owner-operated GPU run.** This strengthens the one surviving result (the 5 dB
noise-robustness benefit) so it can be written up. It runs the **pre-registered
§8.8 grid** (`experiments/PROTOCOL.md` §8.8): four arms of `physics_constrained_cnn`
(same ResNet1D backbone, frozen budget, per-sample rpm), **seeds 0–11**:

| arm | what | command flag |
|---|---|---|
| **CE-only (w=0)** | baseline, **same code path** (not the borrowed Phase-4 one) | `--only pinn_ablation --weights 0.0` |
| **correct (w=1.0)** | the validated band-energy physics loss | `--only pinn_ablation --weights 1.0` |
| **scramble (w=1.0)** | §8.7 control: wrong *real* bands | `--control f9_scramble` |
| **random-band (w=1.0)** | §8.8 control: random *non-fault* bands | `--control random_bands` |

**48 runs total** (4 arms × 12 seeds), all fresh at one commit, **resume-safe** (rerun
the cell after any disconnect; finished seeds skip). Budget ~5–6 h on a T4 — **it may
take two sessions; that's fine.** After they come home, **Claude runs the record-level
seed-level analysis on the laptop** and reports the §8.8 verdict per the pre-registered
decision rule.

- **Compute:** Colab **T4** (Runtime → Change runtime type → T4 GPU).
- **Why GPU:** training; the laptop is CPU-only.
- **Additive:** does not change the validated loss (controls are opt-in attributes).

---

## Cell 1 — GPU check + clone `p7/strengthen`
```bash
!nvidia-smi -L
!git clone -b p7/strengthen https://github.com/abbas-ahmad-cowlar/LSTM_PFD.git /content/lstm-pfd
%cd /content/lstm-pfd
!git rev-parse --short HEAD
!pip -q install -r requirements.txt
```
**Expect:** a `Tesla T4` line and a short HEAD sha. **STOP** if no GPU — switch the
runtime to T4 first.

## Cell 2 — mount Drive + copy the dataset
```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p data/generated
!cp /content/drive/MyDrive/lstm-pfd/data/dataset_v2.h5 data/generated/dataset_v2.h5
!python -c "import h5py; print('OK groups:', list(h5py.File('data/generated/dataset_v2.h5').keys()))"
```
**Expect:** `OK groups: ['test', 'test_snr10', 'test_snr20', 'test_snr5', 'train', 'val']`.
**STOP** if the file is missing — fix the Drive path first.

## Cell 3 — sanity-check both controls are wired (no training)
**Paste as a normal Python cell** (NOT `!python -c "..."` — the `!` shell escape
mangles a multi-line quoted string):
```python
from packages.core.models.pinn.physics_constrained_cnn import PhysicsConstrainedCNN
from packages.core.models.physics.fault_signatures import load_random_reference
from scripts.run_phase5_gpu import scrambled_class_permutation
m = PhysicsConstrainedCNN(backbone='cnn1d')
print('defaults (must be None):', m.reference_permutation, m.random_signature)
sigs, ref = load_random_reference()
assert sigs is not None, 'random_reference.json missing — should be committed'
print('random bands loaded for', len(sigs), 'classes; sain bands:',
      sigs['sain'].tonal, sigs['sain'].bands_hz)
p = scrambled_class_permutation()
assert p[0] == 0 and all(p[i] != i for i in range(1, 11)), 'not a derangement'
print('scramble derangement OK; random + scramble both wired')
```
**Expect:** `defaults (must be None): None None`, random bands for 11 classes
(`sain` empty), and `... both wired`. **STOP** if it errors.

## Cell 4 — one fresh Drive-backed output folder (resume-safe)
The grid writes everything under `results_p7_strengthen/` (a NEW dir — **not** the
repo-shipped `results/phase5`, so there is **no I2 nesting trap** and nothing to
`rm`). One symlink to Drive:
```bash
!mkdir -p /content/drive/MyDrive/lstm-pfd/p7_strengthen
!ln -s /content/drive/MyDrive/lstm-pfd/p7_strengthen results_p7_strengthen
!ls -la results_p7_strengthen   # MUST show a  ->  arrow pointing into Drive
!find results_p7_strengthen -name metrics.json | wc -l   # MUST print 0 (fresh)
```
**Expect:** the `->` arrow into Drive and a metrics count of **0**. **STOP** if either
is wrong (no arrow = the link landed inside a real dir; non-zero = not a fresh folder).

## Cell 5 — GPU smoke (2 epochs each; writes to a separate `_smoke` dir)
```bash
!python scripts/run_phase5_gpu.py --only pinn_ablation --weights 0.0 1.0 --smoke --seeds 0
!python scripts/run_phase5_gpu.py --control random_bands --out-root results_p7_strengthen_smoke --smoke --seeds 0
```
**Expect:** a few `epoch 1/2 ... 2/2` blocks, a `§8.8 RANDOM-BAND control active` line,
and `DONE` lines; no traceback. **STOP** on any `FAILED`. (The smoke uses throwaway
dirs, so it does not touch the real run.)

## Cell 6 — the real grid, 48 runs (resume-safe; rerun after any disconnect)
```bash
S="0 1 2 3 4 5 6 7 8 9 10 11"
# 24 runs: CE-only (w=0) + correct (w=1.0), same code path
!python scripts/run_phase5_gpu.py --only pinn_ablation --weights 0.0 1.0 --out-root results_p7_strengthen --seeds $S
# 12 runs: §8.7 scramble control
!python scripts/run_phase5_gpu.py --control f9_scramble --out-root results_p7_strengthen --seeds $S
# 12 runs: §8.8 random-band control
!python scripts/run_phase5_gpu.py --control random_bands --out-root results_p7_strengthen --seeds $S
```
**Expect:** `Queue: 24 runs` then `12` then `12`; each run logs epochs + a
`DONE: test XX% | test_snr5 YY%`. If the VM disconnects, **just rerun this cell** —
finished seeds are skipped and an interrupted seed resumes from its Drive checkpoint.
The printed 5 dB numbers are *window-level*; the real verdict is the laptop
record-level analysis.

## Cell 7 — verify all 48 are done + on Drive
**Paste as a normal Python cell:**
```python
import glob, json
roots = {'pinn_ablation/w0.0': 'CE-only', 'pinn_ablation/w1.0': 'correct',
         'pinn_ablation_scramble/w1.0': 'scramble', 'pinn_ablation_random/w1.0': 'random'}
for sub, name in roots.items():
    fs = sorted(glob.glob(f'results_p7_strengthen/{sub}/seed*/metrics.json'))
    accs = [round(json.load(open(f))['evals']['test_snr5']['accuracy'], 1) for f in fs]
    print(f"{name:9s} {sub:30s} {len(fs):2d}/12 seeds  5dB(win): {accs}")
```
**Expect:** four lines, each `12/12 seeds`. **STOP here** once all four read 12/12.

---

## Then — bring it home (laptop)
The grid is on your Drive at `MyDrive/lstm-pfd/p7_strengthen/`. Download that whole
folder and tell Claude where it is. Claude will place the four arms under
`results/phase5_bandenergy/` (the `metrics.json` get committed; the `best_model.pth`
checkpoints stay gitignored), then run the **record-level, seed-level** analysis:
clean→5 dB degradation per seed for all 4 arms, Wilcoxon signed-rank vs CE-only
(n=12), and the robust-seed counts — producing the §8.8 verdict.

## What the result will mean (PROTOCOL §8.8, τ = 1.0 pt)
- **random-band stays robust like correct** (≥10/12 seeds, Wilcoxon-significant vs
  CE-only) → **generic spectral regularizer, earned**: even random non-fault bands
  reproduce it. FINDINGS §0 wording stands, now with a matched non-physics control.
- **random-band degrades like CE-only** while correct/scramble stay robust → the
  effect needs *real* fault-frequency bands → narrow to "a spectral-fault-band
  regularizer." FINDINGS §0 updates.
