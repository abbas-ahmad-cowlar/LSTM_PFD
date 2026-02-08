# Cleanup Log — IDB 1.1: Models Sub-Block

> Documentation cleanup performed on 2026-02-08 as part of the Documentation Overhaul initiative.

## Scope

**Primary Directory:** `packages/core/models/`

- 61 Python files across 13 subdirectories + 7 top-level files
- Sub-directories: `classical/`, `cnn/`, `efficientnet/`, `ensemble/`, `fusion/`, `hybrid/`, `nas/`, `physics/`, `pinn/`, `resnet/`, `spectrogram_cnn/`, `transformer/`

## Phase 1: Archive & Extract

### Files Archived

**None.** No `.md`, `.rst`, or `.txt` documentation files existed in `packages/core/models/` prior to this overhaul. The directory contained only `.py` source files and `__pycache__/` directories.

### Information Extracted

All documentation was generated from scratch by inspecting the actual Python source code:

| Source                       | Information Extracted                                           |
| ---------------------------- | --------------------------------------------------------------- |
| `base_model.py`              | BaseModel abstract interface (15 methods)                       |
| `model_factory.py`           | MODEL_REGISTRY (18 factory keys), 9 factory functions           |
| `__init__.py`                | Complete export list (~60 symbols)                              |
| `cnn_1d.py`, `resnet_1d.py`  | CNN1D and ResNet1D architectures + constructor params           |
| `hybrid_pinn.py`             | HybridPINN with PhysicsConstraint and FeatureFusion             |
| `legacy_ensemble.py`         | 3 legacy ensemble classes                                       |
| `transformer/` (4 files)     | SignalTransformer, VisionTransformer1D, PatchTST, TSMixer       |
| `hybrid/` (4 files)          | CNNTransformerHybrid, CNNLSTM, CNNTCN, MultiscaleCNN            |
| `pinn/` (4 files)            | PhysicsConstrainedCNN, MultitaskPINN, KnowledgeGraphPINN        |
| `ensemble/` (5 files)        | VotingEnsemble(v2), StackingEnsemble(v2), BoostingEnsemble, MoE |
| `fusion/` (2 files)          | EarlyFusion, LateFusion with 4 fusion strategies                |
| `classical/` (6 files)       | SVM, RF, NN, GBM wrappers + ModelSelector                       |
| `resnet/` (4 files)          | SEResNet, WideResNet, residual blocks                           |
| `efficientnet/` (2 files)    | EfficientNet1D with MBConv blocks                               |
| `spectrogram_cnn/` (3 files) | DualStreamCNN, ResNet2D, EfficientNet2D for spectrograms        |
| `physics/` (3 files)         | BearingDynamics, FaultSignatures, OperatingConditions           |
| `nas/` (1 file)              | SearchSpace definition                                          |

## Phase 2: Files Created

| File                                      | Description                                                                                                                               |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `packages/core/models/README.md`          | Module overview with 30+ class catalog, Mermaid inheritance diagram, factory usage guide, quick start, and adding-new-models instructions |
| `packages/core/models/API.md`             | Complete API reference for all model classes with constructor signatures, parameter tables, and method documentation                      |
| `packages/core/models/pinn/README.md`     | PINN sub-block documentation covering 4 physics integration approaches with architecture diagrams                                         |
| `packages/core/models/ensemble/README.md` | Ensemble sub-block documentation covering 4 ensemble strategies with usage examples                                                       |
| `docs/CLEANUP_LOG_IDB_1_1.md`             | This file                                                                                                                                 |

## Decisions Made

| Decision                                         | Rationale                                                                                                                                                                                          |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| No files archived                                | No documentation existed in scope — only Python source files                                                                                                                                       |
| Created sub-READMEs for `pinn/` and `ensemble/`  | Both have 5+ files and complex enough architectures to warrant dedicated documentation                                                                                                             |
| Did not create sub-READMEs for other directories | `cnn/`, `resnet/`, `transformer/`, `hybrid/`, `fusion/`, `classical/`, `spectrogram_cnn/`, `physics/`, `nas/`, `efficientnet/` are adequately covered by the main README catalog and API reference |
| Used `[PENDING]` for all performance metrics     | Per overhaul rules — no unverified accuracy/F1/benchmark claims                                                                                                                                    |
| Documented legacy ensemble alongside v2          | Both are exported in `__init__.py` — legacy is still importable for backward compatibility                                                                                                         |
| Listed factory keys from MODEL_REGISTRY          | Only models registered in `model_factory.py` have factory keys; others are instantiated directly                                                                                                   |

## Protected Files

The following directories were **not touched** per overhaul rules:

- `docs/idb_reports/` — Active analysis reports
- `docs/idb_reports/compiled/` — Compiled reports
- `docs/paper/` — Active research paper
- `docs/research/` — Supporting research docs
