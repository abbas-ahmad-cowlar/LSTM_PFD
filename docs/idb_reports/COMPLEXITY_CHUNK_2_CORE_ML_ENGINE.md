# Chunk 2: Core ML Engine (Domain 1)

**Verdict: 🔴 Significantly Over-Engineered**

---

## The Numbers

| Sub-module        | Files           | Lines (approx) | Assessment                              |
| ----------------- | --------------- | -------------- | --------------------------------------- |
| `models/`         | 65 (13 subdirs) | ~5,000+        | 🔴 40+ model files, only ~15 in factory |
| `training/`       | 26              | ~3,000+        | 🔴 Parallel per-type hierarchies        |
| `evaluation/`     | 19              | ~2,500+        | 🟡 Similar duplication pattern          |
| `features/`       | 15              | ~1,000+        | 🟢 Mostly justified                     |
| `explainability/` | 11              | ~1,600+        | 🟡 Over-scoped for this project         |
| `pipelines/`      | 5               | ~500+          | 🟢 Fine                                 |
| `transformers/`   | 9               | ~2,000+        | 🔴 Entirely dead code                   |

---

## Problem 1: 🔴 `models/` — 40+ Model Files, Most Unregistered

### Duplicate Implementations

Two separate `CNN1D` classes exist side-by-side:

| File                   | Lines | Architecture                                           | Status  |
| ---------------------- | ----- | ------------------------------------------------------ | ------- |
| `models/cnn_1d.py`     | 232   | 6 conv layers, hardcoded params                        | Legacy  |
| `models/cnn/cnn_1d.py` | 313   | Configurable, uses `ConvBlock1D` from `conv_blocks.py` | Current |

Same for ResNet:

| File                         | Lines | Status                                     |
| ---------------------------- | ----- | ------------------------------------------ |
| `models/resnet_1d.py`        | 339   | Legacy (defines its own `BasicBlock1D`)    |
| `models/resnet/resnet_1d.py` | 378   | Current (uses shared `residual_blocks.py`) |

**Verdict:** **REMOVE** `models/cnn_1d.py` and `models/resnet_1d.py` (the root-level legacy copies). Update the model factory to point to the subdir versions only.

### Dead Code: `transformers/advanced/`

| File                      | Size  | What It Is                | Imports Found |
| ------------------------- | ----- | ------------------------- | ------------- |
| `bert.py`                 | 20 KB | BERT for NLP              | **None**      |
| `gpt.py`                  | 17 KB | GPT decoder               | **None**      |
| `t5.py`                   | 23 KB | T5 encoder-decoder        | **None**      |
| `swin_transformer.py`     | 21 KB | Swin (2D vision)          | **None**      |
| `vision_transformer.py`   | 15 KB | ViT (2D patches)          | **None**      |
| `attention_mechanisms.py` | 17 KB | Supporting attention code | **None**      |

> [!CAUTION]
> **~113 KB of code that is never imported anywhere in the project.** These are NLP and 2D vision architectures — the project does 1D bearing vibration signal classification. They serve no purpose here.

**Verdict:** **REMOVE the entire `transformers/advanced/` directory.**

### Dead Code: `models/legacy_ensemble.py`

11 KB file, zero imports anywhere.

**Verdict:** **REMOVE.**

### Dead Code: `models/hybrid_pinn.py` (root-level)

13 KB file duplicating `models/pinn/hybrid_pinn.py` (16 KB, the better version).

**Verdict:** **REMOVE** the root-level copy.

### Questionable: Model Zoo Size

The `model_factory.py` registers only **~15 models**, but the repo contains **40+ model files** across 12 subdirectories. Unregistered families include:

- `efficientnet/` (EfficientNet 1D) — 2 files
- `spectrogram_cnn/` (2D spectrogram models) — 3 files
- `fusion/` (early/late fusion) — 2 files
- `nas/` (Neural Architecture Search) — 1 file (just a search space definition)
- `classical/` — 6 files (SVM, RF, GBT) — wired through a separate `classical_ml_pipeline.py`, not the factory

**Question for you:** Have any of these unregistered models ever produced results you care about? If not, they should go.

---

## Problem 2: 🔴 `training/` — Per-Model-Type Duplication

The training directory has evolved a **parallel hierarchy** where every model type got its own trainer, callbacks, losses, and schedulers:

| Generic         | CNN-Specific        | PINN-Specific               | Spectrogram              | Transformer                   |
| --------------- | ------------------- | --------------------------- | ------------------------ | ----------------------------- |
| `trainer.py`    | `cnn_trainer.py`    | `pinn_trainer.py`           | `spectrogram_trainer.py` | —                             |
| `callbacks.py`  | `cnn_callbacks.py`  | —                           | —                        | —                             |
| `losses.py`     | `cnn_losses.py`     | `physics_loss_functions.py` | —                        | —                             |
| `optimizers.py` | `cnn_optimizer.py`  | —                           | —                        | —                             |
| —               | `cnn_schedulers.py` | —                           | —                        | `transformer_schedulers.py`   |
| —               | —                   | —                           | —                        | `transformer_augmentation.py` |

**What should have happened:** One configurable `Trainer` class with strategy/callback hooks, not 4 separate trainer implementations.

**Verdict:**

- **CONSOLIDATE** into a single `Trainer` with configurable loss/optimizer/callback strategies
- The `physics_loss_functions.py` is genuinely domain-specific and should stay as a loss module
- `knowledge_distillation.py`, `progressive_resizing.py`, `mixed_precision.py` are fine as standalone utilities

**Impact:** -6 to -8 files if you merge the CNN/PINN/Spectrogram trainers into one.

---

## Problem 3: 🟡 `evaluation/` — Same Per-Type Pattern

| Generic        | CNN                | PINN                | Ensemble                | Spectrogram                |
| -------------- | ------------------ | ------------------- | ----------------------- | -------------------------- |
| `evaluator.py` | `cnn_evaluator.py` | `pinn_evaluator.py` | `ensemble_evaluator.py` | `spectrogram_evaluator.py` |

Plus standalone analysis tools:

- `architecture_comparison.py` (15 KB)
- `attention_visualization.py` (16 KB)
- `error_analysis.py` (16 KB)
- `physics_interpretability.py` (18 KB)
- `time_vs_frequency_comparison.py` (15 KB)

The per-type evaluators follow the same anti-pattern as training. The standalone analysis tools are more justifiable — they're visualization/analysis scripts.

**Verdict:**

- **CONSOLIDATE** the 4 per-type evaluators into a single `Evaluator` with model-type dispatch
- **KEEP** the standalone analysis scripts (they're tools, not redundant code)

---

## Problem 4: 🟡 `explainability/` — 7 XAI Methods

| File                            | Method               | Size  |
| ------------------------------- | -------------------- | ----- |
| `shap_explainer.py`             | SHAP                 | 14 KB |
| `lime_explainer.py`             | LIME                 | 13 KB |
| `integrated_gradients.py`       | Integrated Gradients | 13 KB |
| `anchors.py`                    | Anchors              | 18 KB |
| `concept_activation_vectors.py` | TCAV                 | 18 KB |
| `partial_dependence.py`         | PDP                  | 19 KB |
| `uncertainty_quantification.py` | MC Dropout           | 13 KB |

**Is this overkill?** For a research project defending a thesis — maybe not, these show breadth. For a production bearing diagnosis tool — absolutely. Nobody runs 7 XAI methods in production.

**Verdict:** **KEEP for now** if this is academic work. Flag for removal if transitioning to production. At minimum, confirm that each method has actually been tested with your models.

---

## Problem 5: 🟢 `features/` — Mostly Justified

11 extractors covering time-domain, frequency-domain, wavelet, bispectrum, envelope analysis. These are standard signal processing features for bearing diagnosis in the literature. The `feature_validator.py`, `feature_normalization.py`, and `feature_selector.py` are reasonable supporting code.

**Verdict:** **KEEP.** This is the most well-justified module.

---

## Summary Scorecard

| Action                                                                                                    | Impact                              |
| --------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| Remove `transformers/advanced/`                                                                           | -7 files, -113 KB of dead code      |
| Remove legacy root-level duplicates (`cnn_1d.py`, `resnet_1d.py`, `hybrid_pinn.py`, `legacy_ensemble.py`) | -4 files                            |
| Consolidate 4 trainers → 1                                                                                | -3 files (merge, keep physics loss) |
| Consolidate 4 evaluators → 1                                                                              | -3 files                            |
| Audit unregistered model families                                                                         | Potentially -5 to -10 more files    |

> [!IMPORTANT]
> **Core question for this chunk:** Is this a research model zoo (keep breadth) or a production system (ruthlessly prune to the 3-4 models that actually work)? The answer determines whether you keep 40+ model files or cut to ~15.

---

_Next: Chunk 3 — Dashboard Platform (Domain 2) — 211 files including 26 callbacks, 24 layouts, 14 services._
