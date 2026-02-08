# Cleanup Log — IDB 1.2 (Training — Core ML Engine)

**Date:** 2026-02-09
**Scope:** `packages/core/training/`
**Agent:** IDB 1.2

## Phase 1: Archive & Extract

### Files Scanned

Searched for `*.md`, `*.rst`, `*.txt` documentation files in `packages/core/training/`.

### Result

**No documentation files found.** Phase 1 is trivially complete — nothing to archive or extract.

### Source Files Analyzed (23)

| File                          | Lines | Component                              |
| ----------------------------- | ----- | -------------------------------------- |
| `__init__.py`                 | 5     | Package init                           |
| `trainer.py`                  | 283   | Base Trainer                           |
| `cnn_trainer.py`              | 460   | CNN Trainer                            |
| `pinn_trainer.py`             | 530   | PINN Trainer                           |
| `spectrogram_trainer.py`      | 418   | Spectrogram & MultiTFR Trainer         |
| `mixed_precision.py`          | 141   | Mixed Precision Trainer                |
| `knowledge_distillation.py`   | 424   | Distillation Trainer & Loss            |
| `progressive_resizing.py`     | 345   | Progressive Resizing Trainer           |
| `losses.py`                   | 170   | Focal, LabelSmoothing, PhysicsInformed |
| `cnn_losses.py`               | 341   | CNN losses & factory                   |
| `physics_loss_functions.py`   | 357   | Physics constraint losses              |
| `callbacks.py`                | 218   | Base callbacks                         |
| `cnn_callbacks.py`            | 544   | Extended CNN callbacks                 |
| `optimizers.py`               | 198   | Legacy optimizer/scheduler factory     |
| `cnn_optimizer.py`            | 212   | CNN optimizer factory & configs        |
| `cnn_schedulers.py`           | 493   | CNN LR schedulers                      |
| `transformer_schedulers.py`   | 240   | Transformer LR schedulers              |
| `metrics.py`                  | 248   | Metrics tracker & functions            |
| `grid_search.py`              | 216   | Grid search optimizer                  |
| `random_search.py`            | 193   | Random search optimizer                |
| `bayesian_optimizer.py`       | 246   | Bayesian optimizer (Optuna)            |
| `advanced_augmentation.py`    | 483   | Signal-level augmentation              |
| `transformer_augmentation.py` | 442   | Patch-level augmentation               |

## Phase 2: Create New Documentation

### Files Created

| File                                       | Purpose                                                                    | Lines |
| ------------------------------------------ | -------------------------------------------------------------------------- | ----- |
| `packages/core/training/README.md`         | Module overview with architecture diagram, component catalogs, quick start | ~260  |
| `packages/core/training/TRAINING_GUIDE.md` | Practical guide: 12 topics with decision tables and code examples          | ~310  |
| `packages/core/training/API.md`            | Full API reference for all public classes and functions                    | ~380  |

### Key Decisions

1. **Two callback systems documented side-by-side:** `callbacks.py` (base, used by `Trainer`) and `cnn_callbacks.py` (extended, more callbacks). Both are documented rather than favoring one.
2. **Deprecated `optimizers.create_optimizer()` noted:** The `optimizers.py` factory delegates to `cnn_optimizer.py`; deprecation is documented in the guide.
3. **Duplicate loss classes acknowledged:** Both `losses.py` and `cnn_losses.py` implement `FocalLoss` and `LabelSmoothingCrossEntropy` with slightly different APIs. Both are documented in the API reference.
4. **`DistillationTrainer` and `ProgressiveResizingTrainer` documented as standalone trainers:** They don't inherit from `Trainer` but follow a compatible interface pattern.
5. **All performance claims use `[PENDING — run experiment to fill]`** — no unverified metrics.

## Extracted Information

No information was extracted from archive (no prior docs existed). All documentation was written from codebase inspection.
