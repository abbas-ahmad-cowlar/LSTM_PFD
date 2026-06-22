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
>
> 🛠️ **Status (2026-06-17): remediation complete; DRAFT verdict pending owner
> re-ratification.** Two independent external audits
> ([2026-06-14](audit_reports/INDEPENDENT_SCIENCE_AUDIT_2026-06-14.md) opened the
> blast radius; [2026-06-16](audit_reports/INDEPENDENT_SCIENCE_AUDIT_2026-06-16.md)
> reviewed the band-energy fix and **independently reproduced every record-level
> number**). Net: the **synthetic dataset + benchmark stand as a classification
> benchmark** (record-level near-ceiling, no row showing a physics accuracy
> advantage), and **one** physics positive survives at record level — a **5 dB
> noise-robustness gain from the corrected band-energy loss in a same-architecture
> ablation**; clean-accuracy / data-efficiency / severity-OOD / interpretability /
> calibration advantages do **not**. Read [results/FINDINGS.md](results/FINDINGS.md)
> §0 (DRAFT) and [PROJECT_STATE.md](PROJECT_STATE.md) for the live state.

## What this is

A reproducible pipeline that generates synthetic vibration signals for
hydrodynamic-bearing faults from documented physics ([docs/PHYSICS.md](docs/PHYSICS.md),
enforced by a 34-test spectral CI battery), then benchmarks classical ML, deep
nets, and physics-informed models on them under a **frozen protocol**
([experiments/PROTOCOL.md](experiments/PROTOCOL.md)). The journal-bearing focus is
deliberate — public datasets (CWRU, Paderborn) are almost all *rolling-element*.

## Status of findings (Phase 5/6 — DRAFT verdict, see [results/FINDINGS.md](results/FINDINGS.md) §0)

> The band-energy physics remediation is complete and judged at the **record
> level** (528 records); two independent external audits reviewed it (the second
> reproduced every number). The synthesis in
> [results/FINDINGS.md](results/FINDINGS.md) §0 is a **DRAFT awaiting owner
> re-ratification.** What can and cannot be said today:

- **Supportable:** a balanced, leakage-checked, group-split **synthetic
  journal-bearing classification benchmark** (record-level near-ceiling; no row
  shows a physics accuracy advantage); and **one** physics positive — a **5 dB
  noise-robustness gain from the correctly-implemented PC-CNN band-energy loss at
  high weight, in a same-architecture ablation** (record level, 528 records:
  McNemar 14–0, p=1.2e-4; mechanism = noisy `lubrification` records rescued from a
  `mixed_wear_lube` mislabel). Frame it as "the *implemented band-energy term*
  helped," not "physics helps."
- **Not supportable:** any physics advantage in **clean accuracy,
  data-efficiency, severity-OOD, interpretability (§8.6a *reverses*), or
  calibration (a wash)** — each was tested at record level and did **not** survive;
  any "physics-informed learning helps" family claim; any real-bearing claim. The
  vs-vanilla noise edge is significant but small/seed-sensitive (secondary to the
  same-arch ablation), and the benefit is **not yet isolated** from generic
  high-weight regularization (controls are future work).
- **Methodological caution (a contribution):** the physics loss had five
  successive defects (non-differentiable argmax → wrong bearing type → tonal-only
  → flat baseline) that all passed a green suite; the corrected
  band-energy-vs-healthy-reference loss is the fix, and the inert generic path is
  now **hard-blocked** ([tests/test_physics_quarantine.py](tests/test_physics_quarantine.py)).

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
1 s windows, mean ± std over 3 seeds. Accuracies below are **window-level and
descriptive**; significance is **recomputed at the record level** (528 records,
not 2,640 correlated windows) in
[results/benchmark/summary_record_level.md](results/benchmark/summary_record_level.md).
Full window-level table + provenance:
[results/benchmark/summary.md](results/benchmark/summary.md).

| Model | Window accuracy | Macro-F1 | Note |
| --- | --- | --- | --- |
| Voting ensemble (top-3) | **96.48%** | 0.964 | single seed |
| ResNet18-1D | 96.14 ± 0.28% | 0.961 | strongest vanilla |
| CNN-LSTM | 96.12 ± 0.16% | 0.961 | |
| PhysicsConstrainedCNN | 95.98 ± 0.36% | 0.960 | **CE-only — architecture row, physics loss OFF** |
| RandomForest (36 features) | 94.61 ± 0.05% | 0.946 | classical bar |
| CNN1D | 91.94 ± 2.84% | 0.917 | |

The top vanilla models cluster near 96%. The "physics-labeled" rows
(hybrid_pinn / multitask_pinn, ~90%) were **not** trained with valid physics
(hybrid uses a rolling-element branch + constant metadata; multitask ran
single-task) — they are **not** physics results. Record-level significance
supersedes any window-level p-value. Inference (ResNet18, CPU, 1 s window):
**~13 ms** via ONNX FP32 — [results/deployment/appendix.md](results/deployment/appendix.md).

## Quick start

```bash
git clone https://github.com/abbas-ahmad-cowlar/LSTM_PFD.git
cd LSTM_PFD
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# exact maintainer env: pip install -r requirements.lock.txt  (Python 3.14 / torch 2.9.1+cpu)
pytest -q   # expect 251 passed, 6 deselected
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
learnability of the *generator's* signatures, not real-world performance. The one
surviving physics positive — 5 dB noise robustness from the band-energy loss in a
same-architecture ablation (§8.4) — is **synthetic-only, n=3 seeds, near-ceiling**,
and **not yet isolated** from generic high-weight regularization (controls are
future work); frame it as "the *implemented band-energy term* helped," not
"physics helps." Clean-accuracy / data-efficiency / severity-OOD / interpretability
/ calibration advantages were tested and did **not** survive; §8.5 HybridPINN stays
excluded (rolling-element branch). The findings memo is a **DRAFT pending owner
re-ratification** — see [results/FINDINGS.md](results/FINDINGS.md) §0 and
[PROJECT_STATE.md](PROJECT_STATE.md).
