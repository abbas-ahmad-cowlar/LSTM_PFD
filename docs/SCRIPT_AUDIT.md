# Script Audit — `.py` & `.ps1` Files

**Role:** Senior Technical Architect Analysis  
**Date:** February 9, 2026  
**Scope:** All `.py` (380+) and `.ps1` (8) files in the LSTM_PFD repo

---

## Executive Summary

The repository contains **~388 script files** (380 `.py` + 8 `.ps1`). After inspection:

| Verdict                  | Count | Description                                                         |
| ------------------------ | ----- | ------------------------------------------------------------------- |
| ✅ **PRODUCTION**        | ~340  | Core library, dashboard, data pipeline, tests — the real codebase   |
| ⚠️ **REVIEW / REFACTOR** | ~30   | Useful but have issues (duplication, wrong location, stale)         |
| 🗑️ **JUNK / DEBUG**      | ~18   | One-off debugging scripts, phase-validation hacks, never cleaned up |

---

## 1. PowerShell Scripts (`.ps1`) — 8 files

### ✅ PRODUCTION — Keep

| File                                        | Lines | Purpose                                                             |
| ------------------------------------------- | ----- | ------------------------------------------------------------------- |
| `packages/dashboard/start_dashboard.ps1`    | 156   | Production dashboard startup (Redis, Celery, Dash)                  |
| `scripts/utilities/disable_gpu_timeout.ps1` | 117   | TDR registry management with proper admin checks and `-Revert` flag |

### 🗑️ JUNK / DEBUG — Recommend Delete

| File                             | Lines | Why Junk                                                                              |
| -------------------------------- | ----- | ------------------------------------------------------------------------------------- |
| `check_pytorch_cuda.py` _(root)_ | 106   | Diagnostic print script. One-time use, never needed again                             |
| `fix_pytorch_cuda.ps1` _(root)_  | 166   | One-time CUDA fix script. Duplicates logic already in `run_phase_0.ps1`               |
| `set_tdr_timeout.ps1` _(root)_   | 14    | Primitive 14-line TDR hack. Superseded by `scripts/utilities/disable_gpu_timeout.ps1` |

### ⚠️ REVIEW — Consider Refactoring

| File                      | Lines | Issue                                                                                   |
| ------------------------- | ----- | --------------------------------------------------------------------------------------- |
| `scripts/run_phase_0.ps1` | 790   | Massive monolith. ~400 lines are duplicated setup/GPU detection shared with Phase 1 & 2 |
| `scripts/run_phase_1.ps1` | 756   | Nearly identical setup boilerplate as Phase 0. Should share a common helper             |
| `scripts/run_phase_2.ps1` | 788   | Same issue. 3 scripts × 750 lines = ~2250 lines, with ~1200 being duplicated            |

> [!TIP]
> **Refactoring opportunity:** Extract the shared setup (~400 lines of venv detection, GPU check, PyTorch install) into a `scripts/utilities/common_setup.ps1` helper function, then have each `run_phase_X.ps1` source it. This would cut ~1000 lines of duplication.

---

## 2. Root-Level Python Scripts — 2 files

| File                    | Lines | Verdict       | Notes                                                                                                          |
| ----------------------- | ----- | ------------- | -------------------------------------------------------------------------------------------------------------- |
| `check_pytorch_cuda.py` | 106   | 🗑️ **JUNK**   | One-time CUDA diagnostic. Same info available via `python -c "import torch; print(torch.cuda.is_available())"` |
| `check_requirements.py` | 268   | ⚠️ **REVIEW** | Useful concept but belongs in `scripts/utilities/`. Not imported by anything                                   |

---

## 3. `scripts/` Directory — 33 `.py` files

### ✅ PRODUCTION — Legitimate Pipeline Scripts

| File                             | Purpose                      |
| -------------------------------- | ---------------------------- |
| `benchmark_inference.py`         | Inference speed benchmarking |
| `evaluate_cnn.py`                | CNN evaluation runner        |
| `export_interactive_plots.py`    | Plot export utility          |
| `export_onnx.py`                 | ONNX model export            |
| `generate_dataset_standalone.py` | Standalone dataset generator |
| `generate_openapi.py`            | OpenAPI spec generator       |
| `import_mat_dataset.py`          | MATLAB data importer         |
| `inference_cnn.py`               | CNN inference runner         |
| `precompute_spectrograms.py`     | Spectrogram precomputation   |
| `quantize_model.py`              | Model quantization           |
| `run_ablations.py`               | Ablation study runner        |
| `train_cnn.py`                   | CNN training entry point     |
| `train_spectrogram_cnn.py`       | Spectrogram CNN training     |

### ✅ PRODUCTION — Research Scripts (`scripts/research/`)

| File                            | Purpose                               |
| ------------------------------- | ------------------------------------- |
| `ablation_study.py`             | PINN ablation experiments             |
| `contrastive_physics.py`        | Contrastive physics-informed learning |
| `failure_analysis.py`           | Failure mode analysis                 |
| `hyperparameter_sensitivity.py` | HP sensitivity study                  |
| `ood_testing.py`                | Out-of-distribution testing           |
| `pinn_ablation.py`              | PINN component ablation               |
| `pinn_comparison.py`            | PINN vs baseline comparison           |
| `transformer_benchmark.py`      | Transformer model benchmarking        |
| `xai_metrics.py`                | XAI quantitative metrics              |

### ✅ PRODUCTION — Utility Scripts (`scripts/utilities/`)

| File                      | Purpose                   |
| ------------------------- | ------------------------- |
| `check_data_leakage.py`   | Data leakage detection    |
| `check_requirements.py`   | Requirement validation    |
| `check_syntax.py`         | Syntax checking           |
| `cross_validation.py`     | Cross-validation helper   |
| `fix_imports.py`          | Import fixer              |
| `mixed_precision_test.py` | Mixed-precision testing   |
| `onnx_export.py`          | ONNX export utility       |
| `pdf_report.py`           | PDF report generator      |
| `statistical_analysis.py` | Statistical analysis      |
| `temporal_cv.py`          | Temporal cross-validation |

### 🗑️ JUNK — Debug Leftovers

| File                     | Lines | Why Junk                                                                                                             |
| ------------------------ | ----- | -------------------------------------------------------------------------------------------------------------------- |
| `test_phase5_imports.py` | 240   | Phase-specific import validation test, run once to debug Phase 5 setup. No longer needed — `tests/` has proper tests |

---

## 4. `packages/core/` — 134 `.py` files — ✅ ALL PRODUCTION

This is the core ML engine. All sub-modules are legitimate:

| Sub-module        | Files | Status                                                      |
| ----------------- | ----- | ----------------------------------------------------------- |
| `evaluation/`     | 17    | ✅ Evaluators, analyzers, benchmark tools                   |
| `explainability/` | 8     | ✅ SHAP, LIME, integrated gradients, anchors, CAVs          |
| `features/`       | 12    | ✅ Feature extraction, selection, validation                |
| `models/`         | 47    | ✅ LSTM, CNN, PINN, Transformer, ensemble, classical models |
| `pipelines/`      | 4     | ✅ Training, evaluation, experiment pipelines               |
| `training/`       | 23    | ✅ Trainers, schedulers, augmentation, callbacks            |
| `transformers/`   | 23    | ✅ Transformer architectures + attention modules            |
| `__init__.py`     | 1     | ✅ Package init                                             |

> No junk found in `packages/core/`.

---

## 5. `packages/dashboard/` — 153 `.py` files — ✅ ALL PRODUCTION

Full Dash web application. All sub-modules are legitimate:

| Sub-module      | Files | Status                                        |
| --------------- | ----- | --------------------------------------------- |
| `api/`          | 5     | ✅ REST API routes, key management            |
| `callbacks/`    | 26    | ✅ Dash callback handlers                     |
| `components/`   | 7     | ✅ UI components                              |
| `config/`       | 2     | ✅ Logging + security config                  |
| `database/`     | 4     | ✅ DB connection + migration                  |
| `integrations/` | 7     | ✅ ML pipeline bridges                        |
| `layouts/`      | 24    | ✅ Page layouts                               |
| `models/`       | 10    | ✅ SQLAlchemy ORM models                      |
| `services/`     | 14    | ✅ Business logic services                    |
| `tasks/`        | 5     | ✅ Celery async tasks                         |
| Root files      | 3     | ✅ `app.py`, `dashboard_config.py`, `wsgi.py` |

> No junk found in `packages/dashboard/`.

---

## 6. `packages/deployment/` — 10 `.py` files — ✅ ALL PRODUCTION

| Sub-module      | Files | Status                           |
| --------------- | ----- | -------------------------------- |
| `api/`          | 4     | ✅ FastAPI deployment server     |
| `optimization/` | 5     | ✅ Inference, ONNX, quantization |
| `__init__.py`   | 1     | ✅ Package init                  |

---

## 7. `data/` — 16 `.py` files — ✅ ALL PRODUCTION

Signal processing, data loading, augmentation, cache management. All legitimate production files.

---

## 8. `config/` — 6 `.py` files — ✅ ALL PRODUCTION

Configuration dataclasses: `base_config.py`, `data_config.py`, `experiment_config.py`, `model_config.py`, `training_config.py`.

---

## 9. `integration/` — 4 `.py` files — ✅ ALL PRODUCTION

Unified pipeline, model registry, configuration validator, data pipeline validator.

---

## 10. `experiments/` — 6 `.py` files — ✅ ALL PRODUCTION

Experiment management: `experiment_manager.py`, `hyperparameter_tuner.py`, `pinn_ablation.py`, `ensemble_comparison.py`, `cnn_experiment.py`, `compare_experiments.py`.

---

## 11. `benchmarks/` — 4 `.py` files — ✅ ALL PRODUCTION

`industrial_validation.py`, `literature_comparison.py`, `resource_profiling.py`, `scalability_benchmark.py`.

---

## 12. `utils/` — 11 `.py` files — ✅ ALL PRODUCTION

Shared utilities: `checkpoint_manager.py`, `constants.py`, `device_manager.py`, `early_stopping.py`, `file_io.py`, `logger.py`, `logging.py`, `reproducibility.py`, `timer.py`, `visualization_utils.py`.

> [!NOTE]
> `utils/logger.py` and `utils/logging.py` overlap in purpose. Consider consolidating.

---

## 13. `visualization/` — 13 `.py` files — ✅ ALL PRODUCTION

Plotting and visualization: `performance_plots.py`, `signal_plots.py`, `spectrogram_plots.py`, `attention_viz.py`, `saliency_maps.py`, `xai_dashboard.py`, etc.

---

## 14. `tests/` — 29 `.py` files — ✅ ALL PRODUCTION (test infrastructure)

| Sub-module      | Files | Status                                                                       |
| --------------- | ----- | ---------------------------------------------------------------------------- |
| `unit/`         | 5     | ✅ Unit tests                                                                |
| `integration/`  | 3     | ✅ Integration tests                                                         |
| `benchmarks/`   | 2     | ✅ Benchmark suite                                                           |
| `utilities/`    | 2     | ✅ Bug fix tests                                                             |
| Root test files | 13    | ✅ Domain-specific test modules                                              |
| Supporting      | 4     | ✅ `conftest.py`, `load_tests.py`, `stress_tests.py`, `models/simple_cnn.py` |

---

## 15. `reproducibility/` — 2 `.py` files — ✅ ALL PRODUCTION

`scripts/run_all.py`, `scripts/set_seeds.py` — reproducibility helpers.

---

## Recommended Actions Summary

### 🗑️ Delete (5 files) — Debug junk, no production value

```
check_pytorch_cuda.py         # root — one-time CUDA diagnostic
fix_pytorch_cuda.ps1          # root — one-time CUDA fix, duplicated in phase scripts
set_tdr_timeout.ps1           # root — superseded by scripts/utilities/disable_gpu_timeout.ps1
scripts/test_phase5_imports.py # phase-specific import test, no longer needed
```

### 📦 Relocate (1 file)

```
check_requirements.py  →  scripts/utilities/check_requirements.py  (already exists there)
```

> Since `scripts/utilities/check_requirements.py` already exists, the root copy is a duplicate and can be deleted.

### 🔧 Refactor (3 files) — Phase runner script duplication

| File                      | Recommendation                                                 |
| ------------------------- | -------------------------------------------------------------- |
| `scripts/run_phase_0.ps1` | Extract shared setup into `scripts/utilities/common_setup.ps1` |
| `scripts/run_phase_1.ps1` | Source common setup, keep only phase-specific logic            |
| `scripts/run_phase_2.ps1` | Source common setup, keep only phase-specific logic            |

> This would eliminate ~1000+ lines of copy-pasted boilerplate.

### ⚠️ Minor Review (1 pair)

| Files                                  | Issue                                                  |
| -------------------------------------- | ------------------------------------------------------ |
| `utils/logger.py` + `utils/logging.py` | Overlapping logging utilities — consider consolidating |

---

## File Count Summary

| Category               | Files    | Verdict                                   |
| ---------------------- | -------- | ----------------------------------------- |
| `packages/core/`       | 134      | ✅ All production                         |
| `packages/dashboard/`  | 153      | ✅ All production                         |
| `tests/`               | 29       | ✅ All production                         |
| `scripts/*.py`         | 33       | ✅ 32 production, 🗑️ 1 junk               |
| `data/`                | 16       | ✅ All production                         |
| `visualization/`       | 13       | ✅ All production                         |
| `utils/`               | 11       | ✅ All production (1 overlap to review)   |
| `packages/deployment/` | 10       | ✅ All production                         |
| `config/`              | 6        | ✅ All production                         |
| `experiments/`         | 6        | ✅ All production                         |
| `integration/`         | 4        | ✅ All production                         |
| `benchmarks/`          | 4        | ✅ All production                         |
| Root `.py`             | 2        | 🗑️ 1 junk, 📦 1 duplicate                 |
| `reproducibility/`     | 2        | ✅ All production                         |
| `.ps1` files           | 8        | ✅ 2 production, 🔧 3 refactor, 🗑️ 3 junk |
| **Total**              | **~388** | **~340 keep, ~30 refactor, ~18 remove**   |
