# Chunk 6: Research & Science (Domain 5) + Scripts

**Verdict: 🔴 Research Sprawl — Lots Written, Little Wired In**

---

## The Numbers

| Directory            | Files                 | Total Size | Assessment                      |
| -------------------- | --------------------- | ---------- | ------------------------------- |
| `scripts/research/`  | 9 .py                 | ~180 KB    | 🔴 Monster files, dead code     |
| `scripts/` (root)    | 10 .py                | ~140 KB    | 🟡 Duplicates package code      |
| `scripts/utilities/` | 11 files              | ~118 KB    | 🟡 Grab bag, some overlap       |
| `experiments/`       | 6 .py                 | ~60 KB     | 🟡 Duplicates scripts/research/ |
| `visualization/`     | 13 .py                | ~168 KB    | 🔴 Heavy overlap + dead code    |
| `benchmarks/`        | 4 .py                 | ~30 KB     | 🟢 Fine                         |
| `docs/research/`     | 8 .md                 | ~23 KB     | 🟢 Fine                         |
| `docs/paper/`        | 2 files (LaTeX + bib) | ~33 KB     | 🟢 Justified                    |
| `reproducibility/`   | 4 files               | ~16 KB     | 🟢 Fine                         |

---

## Problem 1: 🔴 Monster Research Scripts

| File                                             | Size      | Lines  | What It Does                          |
| ------------------------------------------------ | --------- | ------ | ------------------------------------- |
| `scripts/research/ablation_study.py`             | **43 KB** | ~1,100 | Full ablation framework               |
| `scripts/research/contrastive_physics.py`        | **38 KB** | ~950   | Contrastive learning + physics hybrid |
| `scripts/research/ood_testing.py`                | 17 KB     | ~430   | Out-of-distribution testing           |
| `scripts/research/hyperparameter_sensitivity.py` | 16 KB     | ~400   | Hyperparameter sensitivity analysis   |

> [!CAUTION]
> **`contrastive_physics.py` (38 KB) has zero imports anywhere.** It's a self-contained research prototype that was never integrated. 950 lines of dead code.

`ablation_study.py` at 43 KB is the **single largest Python file in the entire project**. It should either be broken into modules or confirmed as a standalone research script that produces paper results.

**Verdict:**

- **REMOVE** `contrastive_physics.py` — dead code, never integrated
- **REVIEW** `ablation_study.py` — does it produce results you've used? If not, archive it

---

## Problem 2: 🔴 Duplicate Experiment Locations

Two files implementing PINN ablation studies:

| File                                | Size  | Location            |
| ----------------------------------- | ----- | ------------------- |
| `scripts/research/pinn_ablation.py` | 13 KB | Research scripts    |
| `experiments/pinn_ablation.py`      | 15 KB | Experiments package |

Same pattern for experiment management:

- `experiments/experiment_manager.py` (3.6 KB) — lightweight manager
- `experiments/cnn_experiment.py` (18 KB) — CNN-specific experiment runner
- vs. dashboard's `experiment_wizard_callbacks.py` (17 KB) — UI-driven experiment runner

**Three places to run experiments.** None share a common interface.

**Verdict:** **CONSOLIDATE** `experiments/` into `scripts/research/` or vice versa. Pick one home for research scripts.

---

## Problem 3: 🔴 Visualization Sprawl — 13 Files, 168 KB

| File                             | Size  | Overlap?                                               |
| -------------------------------- | ----- | ------------------------------------------------------ |
| `cnn_analysis.py`                | 20 KB | 🔴 Overlaps with `cnn_visualizer.py`                   |
| `cnn_visualizer.py`              | 16 KB | 🔴 Overlaps with `cnn_analysis.py`                     |
| `xai_dashboard.py`               | 19 KB | 🔴 **Zero external imports** — standalone XAI Dash app |
| `counterfactual_explanations.py` | 14 KB | 🟡 Only imported by `xai_dashboard.py`                 |
| `saliency_maps.py`               | 15 KB | 🟡 Standard                                            |
| `attention_viz.py`               | 14 KB | 🟢 Justified                                           |
| `feature_visualization.py`       | 16 KB | 🟢 Justified                                           |
| `latent_space_analysis.py`       | 14 KB | 🟢 Justified                                           |
| `activation_maps_2d.py`          | 14 KB | 🟡 Only self-referencing                               |
| `spectrogram_plots.py`           | 13 KB | 🟢 Justified                                           |
| `signal_plots.py`                | 4 KB  | 🟢 Core                                                |
| `performance_plots.py`           | 4 KB  | 🟢 Core                                                |

**Key issues:**

1. `cnn_analysis.py` + `cnn_visualizer.py` — 36 KB of overlapping CNN visualization code
2. `xai_dashboard.py` — a standalone Dash app for XAI that **duplicates** the dashboard's own XAI page
3. `counterfactual_explanations.py` — only used by the dead `xai_dashboard.py`

**Verdict:**

- **MERGE** `cnn_analysis.py` + `cnn_visualizer.py` → single file
- **REMOVE** `visualization/xai_dashboard.py` (dead standalone app, dashboard already has XAI page)
- **REMOVE** `counterfactual_explanations.py` (only consumer is dead code)

---

## Problem 4: 🟡 Root-Level Scripts Duplicate Package Code

| Script                             | Size  | Package Equivalent                                         |
| ---------------------------------- | ----- | ---------------------------------------------------------- |
| `scripts/export_onnx.py`           | 6 KB  | `packages/deployment/optimization/onnx_export.py` (14 KB)  |
| `scripts/train_cnn.py`             | 18 KB | `packages/core/training/cnn_trainer.py`                    |
| `scripts/inference_cnn.py`         | 13 KB | `packages/deployment/optimization/inference.py` (16 KB)    |
| `scripts/evaluate_cnn.py`          | 14 KB | `packages/core/evaluation/cnn_evaluator.py`                |
| `scripts/quantize_model.py`        | 7 KB  | `packages/deployment/optimization/quantization.py` (13 KB) |
| `scripts/train_spectrogram_cnn.py` | 12 KB | `packages/core/training/spectrogram_trainer.py`            |

These scripts were written as standalone CLI tools before the `packages/` structure existed. The package code is generally more mature.

**Verdict:** **REMOVE root-level scripts** that have package equivalents. If you need CLI entry points, add `__main__.py` blocks to the packages.

---

## Problem 5: 🟡 `scripts/utilities/` — Grab Bag

| File                      | Size  | Assessment                                  |
| ------------------------- | ----- | ------------------------------------------- |
| `cross_validation.py`     | 16 KB | 🟡 Should be in `packages/core/evaluation/` |
| `temporal_cv.py`          | 14 KB | 🟡 Specialized CV — same concern            |
| `statistical_analysis.py` | 17 KB | 🟡 Should be in `packages/core/evaluation/` |
| `mixed_precision_test.py` | 14 KB | 🟡 Should be in `tests/`                    |
| `onnx_export.py`          | 10 KB | 🔴 Third copy of ONNX export                |
| `pdf_report.py`           | 12 KB | 🟢 Standalone utility                       |
| `check_data_leakage.py`   | 13 KB | 🟢 Useful auditing tool                     |
| `check_requirements.py`   | 8 KB  | 🟢 Useful                                   |
| `check_syntax.py`         | 1 KB  | 🟢 Useful                                   |
| `fix_imports.py`          | 5 KB  | 🟢 Useful                                   |
| `disable_gpu_timeout.ps1` | 4 KB  | 🟢 Windows-specific                         |

**Three copies of ONNX export:** `scripts/export_onnx.py` + `scripts/utilities/onnx_export.py` + `packages/deployment/optimization/onnx_export.py`.

**Verdict:**

- **REMOVE** `scripts/utilities/onnx_export.py` (third copy)
- **MOVE** `cross_validation.py`, `temporal_cv.py`, `statistical_analysis.py` → `packages/core/evaluation/`
- **MOVE** `mixed_precision_test.py` → `tests/`

---

## Summary Scorecard

| Action                                                                                 | Impact           |
| -------------------------------------------------------------------------------------- | ---------------- |
| Remove `contrastive_physics.py`                                                        | -38 KB dead code |
| Remove dead visualization files (`xai_dashboard.py`, `counterfactual_explanations.py`) | -33 KB           |
| Merge CNN visualization files                                                          | -1 file          |
| Remove root scripts that duplicate packages                                            | -6 files, ~70 KB |
| Remove third ONNX export copy                                                          | -1 file          |
| Consolidate `experiments/` → `scripts/research/`                                       | -1 directory     |

> [!IMPORTANT]
> **This domain has the worst "code written vs code actually used" ratio.** ~350 KB of Python across scripts/research/ and visualization/, with large chunks never imported by anything.

---

_Next: Chunk 7 — Cross-Cutting Concerns (Integration layer, root-level utils, final summary)_
