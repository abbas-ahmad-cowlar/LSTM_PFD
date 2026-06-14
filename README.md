# LSTM-PFD — Physics-grounded benchmark for journal-bearing fault diagnosis

> A physics-based **synthetic-data** research platform for **journal /
> hydrodynamic bearing** fault diagnosis (11 classes), with a frozen,
> pre-registered benchmark and a rigorous, honest study of whether
> physics-informed learning helps.
>
> 🧭 **State & roadmap:** [PROJECT_STATE.md](PROJECT_STATE.md) ·
> step plan [CONVERGENCE_PLAN.md](CONVERGENCE_PLAN.md) ·
> **headline findings** [results/FINDINGS.md](results/FINDINGS.md)
>
> ⚠️ **Scope:** all data is physics-based **synthetic** (no real-world
> validation). Results characterise models *on this synthetic benchmark*.

## What this is

A reproducible pipeline that generates synthetic vibration signals for
hydrodynamic-bearing faults from documented physics ([docs/PHYSICS.md](docs/PHYSICS.md),
enforced by a 34-test spectral CI battery), then benchmarks classical ML, deep
nets, and physics-informed models on them under a **frozen protocol**
([experiments/PROTOCOL.md](experiments/PROTOCOL.md)). The journal-bearing focus is
deliberate — public datasets (CWRU, Paderborn) are almost all *rolling-element*.

## Headline findings (Phase 5 — see [results/FINDINGS.md](results/FINDINGS.md))

- **Physics-informed learning gives no accuracy advantage** on this clean
  synthetic data — tested across noise, data-efficiency, severity-OOD, a
  physics-weight ablation, and operating-condition metadata. At 10% data it even
  *hurts*. A rigorous, pre-registered **negative result**.
- The data-driven models already reach **~96%** — the generator embeds the fault
  signatures cleanly, leaving little for a physics prior to add.
- **But** physics-informed training (same backbone) yields a *modest*
  **interpretability + calibration** gain: attributions align more with
  characteristic fault frequencies, and lower ECE. Physics here is an
  interpretability/trust lever, not an accuracy lever.
- Methodological caution documented: a naive frequency-consistency physics loss
  was silently **non-differentiable** until fixed (see
  [experiments/PHYSICS_LOSS_DIAGNOSIS.md](experiments/PHYSICS_LOSS_DIAGNOSIS.md)).

## Fault taxonomy (11 journal-bearing classes)

| # | Key (FR) | Class |
|---|---|---|
| 0 | sain | Healthy |
| 1 | desalignement | Misalignment |
| 2 | desequilibre | Imbalance |
| 3 | jeu | Bearing clearance |
| 4 | lubrification | Lubrication issue |
| 5 | cavitation | Cavitation |
| 6 | usure | Wear |
| 7 | oilwhirl | Oil whirl |
| 8–10 | mixed_* | Misalign+Imbalance · Wear+Lube · Cavitation+Clearance |

## What's in the repo

- **Signal generator** (`data/signal_generation/`) — physics-grounded, CI-locked
  (`tests/test_physics_signatures.py`); **Dataset v2** (`data/generated/`,
  DVC-tracked): 3,520 records, stratified, leakage-checked, with SNR-20/10/5 test
  variants.
- **Models** (`packages/core/models/`) — classical (RandomForest/SVM/GradientBoosting),
  deep (CNN1D, ResNet1D, CNN-LSTM, PatchTST, AttentionCNN), physics-informed
  (PhysicsConstrainedCNN, HybridPINN, MultitaskPINN), and a voting ensemble.
- **Experiments** (`scripts/`, `results/`) — frozen benchmark, noise robustness,
  data-efficiency, severity-OOD, physics-weight ablation, XAI alignment,
  MC-dropout calibration. Index: [results/README.md](results/README.md).
- **Deployment** — ONNX export + latency appendix (`results/deployment/`).
- **Dashboard** (`packages/dashboard/`) — ⚠️ **experimental, frozen**: it boots
  but is unfinished and excluded from CI (Convergence Plan Phase D). The core
  pipeline does not depend on it.

## Performance

Frozen benchmark ([experiments/PROTOCOL.md](experiments/PROTOCOL.md)), 11-class,
1 s windows, 2,640 test windows, mean ± std over 3 seeds. Full table + stats +
provenance: [results/benchmark/summary.md](results/benchmark/summary.md).

| Model | Test accuracy | Macro-F1 |
| --- | --- | --- |
| Voting ensemble (top-3) | **96.48%** | 0.964 |
| ResNet18-1D | 96.14 ± 0.28% | 0.961 |
| CNN-LSTM | 96.12 ± 0.16% | 0.961 |
| PhysicsConstrainedCNN | 95.98 ± 0.36% | 0.960 |
| RandomForest (36 features) | 94.61 ± 0.05% | 0.946 |
| CNN1D | 91.94 ± 2.84% | 0.917 |

Top-3 deep models are statistically tied (McNemar p > 0.2). Inference (ResNet18,
CPU, 1 s window): **~13 ms** via ONNX FP32 — [results/deployment/appendix.md](results/deployment/appendix.md).

## Quick start

```bash
git clone https://github.com/abbas-ahmad-cowlar/LSTM_PFD.git
cd LSTM_PFD
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# exact maintainer env: pip install -r requirements.lock.txt  (Python 3.14 / torch 2.9.1+cpu)
pytest -q   # expect ~240 passed, 6 deselected
```

Reproduce experiments: see `scripts/` (`run_benchmark.py`, `run_noise_robustness.py`,
`run_phase5_gpu.py`, `run_xai_calibration.py`) and the Colab runbook
[experiments/COLAB_DATAEFF_RUNBOOK.md](experiments/COLAB_DATAEFF_RUNBOOK.md).

## Project structure

```
packages/core/     # models, training, evaluation, features, explainability
data/              # signal_generation/ (generator + physics) · generated/ (Dataset v2, DVC)
experiments/       # PROTOCOL.md (frozen), DATASET_V2.md, runbooks, PHYSICS_LOSS_DIAGNOSIS.md
scripts/           # benchmark / phase-5 / XAI / verification runners
results/           # committed evidence (json/md/png) + FINDINGS.md + README index
docs/              # PHYSICS.md (normative) · tests/ — incl. 34-test physics CI
packages/dashboard/  # experimental, frozen (Phase D)
```

## Documentation

| Resource | Location |
| --- | --- |
| Physics (normative) | [docs/PHYSICS.md](docs/PHYSICS.md) |
| Benchmark protocol (frozen) | [experiments/PROTOCOL.md](experiments/PROTOCOL.md) |
| Dataset v2 design | [experiments/DATASET_V2.md](experiments/DATASET_V2.md) |
| Phase-5 findings | [results/FINDINGS.md](results/FINDINGS.md) |
| Results index | [results/README.md](results/README.md) |
| Project state / plan | [PROJECT_STATE.md](PROJECT_STATE.md) · [CONVERGENCE_PLAN.md](CONVERGENCE_PLAN.md) |

## Limitations

Synthetic data only — no real-bearing validation. Accuracy figures reflect
learnability of the *generator's* signatures, not real-world performance. The
physics-informed negative is bounded to the mechanisms implemented here (see
[results/FINDINGS.md](results/FINDINGS.md) §4b).
