# Dead vs Integratable: Comprehensive Classification

> **Purpose:** This report re-examines every item flagged as "dead code" in Chunks 1-7
> and classifies each as either **🔨 INTEGRATABLE** (can be wired in now) or **💀 TRULY DEAD**
> (project has moved past it). Your call on what to build vs remove.

---

## Category A: 🔨 Built But Unwired — Could Integrate Now

These items are **complete or near-complete implementations** that were never connected
to the rest of the system. They represent work that could add value if wired in.

---

### A1. Unregistered Model Architectures

**What:** 6 model families (~20 files) exist in `packages/core/models/` with proper
implementations but are NOT registered in `model_factory.py`, so nothing can create them.

| Family                                                                                                                                     | Files  | What It Does                                       | Integration Effort                             |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ------ | -------------------------------------------------- | ---------------------------------------------- |
| `transformer/signal_transformer.py`                                                                                                        | 13 KB  | 1D signal transformer for time-series              | **Low** — add to factory registry              |
| `transformer/patchtst.py`                                                                                                                  | 10 KB  | PatchTST — state-of-art time-series transformer    | **Low** — add to factory                       |
| `transformer/tsmixer.py`                                                                                                                   | 9 KB   | TSMixer — lightweight time-series mixing           | **Low** — add to factory                       |
| `transformer/vision_transformer_1d.py`                                                                                                     | 15 KB  | ViT adapted for 1D signals                         | **Low** — add to factory                       |
| `efficientnet/efficientnet_1d.py` + `mbconv_block.py`                                                                                      | 22 KB  | EfficientNet adapted for 1D signals                | **Low** — add to factory                       |
| `fusion/early_fusion.py` + `late_fusion.py`                                                                                                | 25 KB  | Multi-modal fusion architectures                   | **Medium** — needs multi-input data pipeline   |
| `spectrogram_cnn/dual_stream_cnn.py`                                                                                                       | 13 KB  | Dual-stream CNN for spectrograms                   | **Medium** — needs spectrogram data pipeline   |
| `spectrogram_cnn/resnet2d_spectrogram.py`                                                                                                  | 12 KB  | 2D ResNet for spectrogram input                    | **Medium** — needs spectrogram pipeline        |
| `spectrogram_cnn/efficientnet2d_spectrogram.py`                                                                                            | 12 KB  | 2D EfficientNet for spectrograms                   | **Medium** — same                              |
| `classical/random_forest.py`, `svm_classifier.py`, `gradient_boosting.py`, `neural_network.py`, `stacked_ensemble.py`, `model_selector.py` | ~36 KB | Classical ML baselines (RF, SVM, GB, NN, stacking) | **Medium** — needs feature extraction pipeline |
| `nas/search_space.py`                                                                                                                      | 7 KB   | Neural Architecture Search                         | **High** — research feature                    |

> [!NOTE]
> **Correction from Chunk 2:** I previously reported "BERT, GPT, T5, Swin, ViT — NLP/vision models irrelevant to vibration signals." This was **wrong**. There are NO NLP/vision models in the codebase. The actual transformer implementations are all **1D domain-adapted** architectures designed for vibration signals.

**To integrate:** Add `create_*` factory functions and register in `MODEL_REGISTRY` dict in `model_factory.py`. For 1D models (transformers, EfficientNet), this is ~20 lines of code per model. For spectrogram models, the data pipeline also needs connecting.

---

### A2. Contrastive Physics Pretraining (`scripts/research/contrastive_physics.py`)

**What:** A **1068-line, fully implemented** SimCLR-style contrastive learning framework
where physics similarity (eccentricity, clearance, viscosity) defines positive/negative pairs.

**Key components:**

- `PhysicsContrastiveDataset` — builds triplets based on physics parameter similarity
- `SignalEncoder` — CNN backbone with projection head
- `PhysicsInfoNCELoss` — InfoNCE loss adapted for physics pairs
- `ContrastivePretrainer` — full training loop with cosine annealing
- `ContrastiveFineTuner` — downstream classification with frozen/unfrozen encoder
- `run_benchmark()` — supervised vs contrastive comparison with confidence intervals

**Why it's unwired:** It's a standalone CLI script (`python scripts/research/contrastive_physics.py --full-pipeline`). Never imported by anything. But it's a **novel research contribution** — physics-informed contrastive learning for bearing fault diagnosis.

**Integration effort:** **Medium** — needs to be wrapped into the training pipeline and connected to the signal generator's physics parameters.

---

### A3. Contrastive Learning for Spectrograms (`data/contrast_learning_tfr.py`)

**What:** A **403-line** SimCLR implementation specifically for spectrogram data:

- `ContrastiveSpectrogramDataset` — generates augmented view pairs
- `NTXentLoss` + `SimCLRLoss` — two contrastive loss implementations
- `ProjectionHead` — MLP projection head
- `ContrastiveEncoder` — encoder + projection wrapper
- `pretrain_contrastive()` — complete pretraining function

**Why it's unwired:** Built to work with spectrogram augmentation pipeline but never connected.

**Integration effort:** **Medium** — needs spectrogram data pipeline + training script.

---

### A4. Integration Layer (`integration/`)

**What:** A cross-domain orchestration layer with 4 complete files:

| File                         | Size   | Purpose                              | Status                                                 |
| ---------------------------- | ------ | ------------------------------------ | ------------------------------------------------------ |
| `unified_pipeline.py`        | 8.9 KB | End-to-end ML pipeline (Phase 0-9)   | Phase 0-1 implemented, Phases 2-9 are **placeholders** |
| `model_registry.py`          | 8.5 KB | SQLite-based model tracking database | **Fully implemented**                                  |
| `configuration_validator.py` | 8.7 KB | Cross-phase config validation        | **Fully implemented**                                  |
| `data_pipeline_validator.py` | 6 KB   | Data format/quality validation       | **Fully implemented**                                  |

**Why it's unwired:** The dashboard connects to core ML directly, bypassing this layer. The
`UnifiedMLPipeline` has Phase 0 (data gen) and Phase 1 (classical ML) working, but Phases 2-9
are stub methods returning `{'status': 'placeholder'}`.

**Integration effort:**

- `model_registry.py` — **Low** — plug into training loop to auto-register trained models
- `configuration_validator.py` + `data_pipeline_validator.py` — **Low** — add validation calls before pipeline runs
- `unified_pipeline.py` — **High** — needs the placeholder phases filled with real training code

---

### A5. NAS Search Space (`packages/core/models/nas/search_space.py`)

**What:** Neural Architecture Search — defines a search space of architectures.

**Why it's unwired:** NAS requires significant compute and a training loop integration.
The dashboard has NAS callbacks and NAS service files, but the search space itself isn't
connected to the actual training pipeline.

**Integration effort:** **High** — needs Optuna/Ray Tune or custom search loop.

---

### A6. Dashboard Enterprise Features

These features are **fully implemented** in the dashboard but may be "ahead of the project's
current deployment reality." They're not dead code — they work — but you need to decide
if you want them active.

| Feature                                          | Files                                                                      | Status               |
| ------------------------------------------------ | -------------------------------------------------------------------------- | -------------------- |
| **Notification System** (Slack, Teams, Webhooks) | `notification_service.py`, `notification_callbacks.py`, providers/         | ✅ Fully implemented |
| **API Key Management**                           | `api_key_service.py`, `api_key_callbacks.py`, migration 001                | ✅ Fully implemented |
| **Email Digest System**                          | `email_digest_service.py`, `email_digest_callbacks.py`, migrations 003-004 | ✅ Fully implemented |
| **Webhook Integration**                          | `webhook_service.py`, `webhook_callbacks.py`, migrations 005-006           | ✅ Fully implemented |
| **2FA / Login History**                          | migration 010 (11 KB SQL)                                                  | ✅ Fully implemented |
| **Rate Limiting**                                | `rate_limiter.py` middleware                                               | ✅ Fully implemented |
| **NAS Dashboard**                                | `nas_service.py`, `nas_callbacks.py`                                       | ✅ Fully implemented |
| **Deployment Dashboard**                         | `deployment_service.py`, `deployment_callbacks.py`                         | ✅ Fully implemented |

**Your decision:** Keep these if you plan multi-user deployment. Remove if single-researcher use.

---

### A7. Visualization Components

| File                                           | Size  | Status                                                                          |
| ---------------------------------------------- | ----- | ------------------------------------------------------------------------------- |
| `visualization/xai_dashboard.py`               | 19 KB | Standalone Dash app for XAI exploration — works independently of main dashboard |
| `visualization/counterfactual_explanations.py` | 14 KB | Counterfactual explanation generator — imported by `xai_dashboard.py`           |
| `visualization/activation_maps_2d.py`          | 14 KB | 2D activation map visualization — self-contained                                |

**Why unwired:** The main dashboard has its own XAI page. These are standalone research
visualization tools that run separately.

**Integration effort:** **Low** — could be launched as a separate Dash app, or the
visualization functions could be imported into the main dashboard's XAI page.

---

### A8. Utility Scripts with Standalone Value

| Script                                      | Size  | What It Does                           | Where It Should Go                        |
| ------------------------------------------- | ----- | -------------------------------------- | ----------------------------------------- |
| `scripts/utilities/cross_validation.py`     | 16 KB | K-fold CV with stratification          | → `packages/core/evaluation/`             |
| `scripts/utilities/temporal_cv.py`          | 14 KB | Time-series aware CV (no data leakage) | → `packages/core/evaluation/`             |
| `scripts/utilities/statistical_analysis.py` | 17 KB | Statistical significance testing       | → `packages/core/evaluation/`             |
| `scripts/utilities/check_data_leakage.py`   | 13 KB | Data leakage detection tool            | → `tests/` or `packages/core/evaluation/` |
| `scripts/utilities/mixed_precision_test.py` | 14 KB | Mixed precision training benchmark     | → `tests/benchmarks/`                     |

**These are complete, useful tools** that just need to be moved to the right location
and imported by the appropriate modules.

---

## Category B: 💀 Truly Dead — Project Has Moved Past These

These items are genuinely obsolete. The project has evolved in a different direction,
or they've been superseded by better implementations.

---

### B1. Root-Level Model Duplicates

| File                                      | Size  | Why Dead                                                                                                |
| ----------------------------------------- | ----- | ------------------------------------------------------------------------------------------------------- |
| `packages/core/models/cnn_1d.py`          | 7 KB  | **Superseded** by `packages/core/models/cnn/cnn_1d.py` (in subdirectory). Root copy is the old version. |
| `packages/core/models/resnet_1d.py`       | 9 KB  | **Superseded** by `packages/core/models/resnet/resnet_1d.py`. Same situation.                           |
| `packages/core/models/hybrid_pinn.py`     | 13 KB | **Superseded** by `packages/core/models/pinn/` directory with multiple PINN variants.                   |
| `packages/core/models/legacy_ensemble.py` | 11 KB | **Superseded** by `packages/core/models/ensemble/` directory with 5 ensemble types.                     |

> [!CAUTION]
> Check which version `model_factory.py` imports. If it imports the root-level copies,
> you need to update the imports before removing them.

---

### B2. Dashboard Dead Code

| File                                                | Size | Why Dead                                                                                              |
| --------------------------------------------------- | ---- | ----------------------------------------------------------------------------------------------------- |
| `packages/dashboard/utils/auth_utils_improved.py`   | —    | **Zero imports.** An improved auth module that was never swapped in for the original `auth_utils.py`. |
| `packages/dashboard/integrations/phase0_adapter.py` | —    | **Zero imports.** Legacy integration adapter from an older architecture.                              |
| `packages/dashboard/integrations/phase1_adapter.py` | —    | **Zero imports.** Same — old adapter pattern replaced by direct service calls.                        |

---

### B3. Duplicate Augmentation

| File                           | Size   | Why Dead                                                                                                                                                           |
| ------------------------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `data/augmentation.py` (NumPy) | ~15 KB | **Superseded** by `data/signal_augmentation.py` (PyTorch). Same methods (Mixup, TimeWarp, MagnitudeWarp), but PyTorch version integrates with DataLoader pipeline. |

**Keep:** `data/signal_augmentation.py` (PyTorch — integrates with training)
**Remove:** `data/augmentation.py` (NumPy — standalone, not used by training pipeline)

---

### B4. Out-of-Scope Dataset

| File                   | Size   | Why Dead                                                                                                                                          |
| ---------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/cwru_dataset.py` | ~12 KB | Loads CWRU **rolling element** bearing data. This project is about **hydrodynamic** (journal) bearings. Different physics, different fault modes. |

---

### B5. Root-Level Scripts (Superseded by Packages)

| Script                             | Size  | Superseded By                                                               |
| ---------------------------------- | ----- | --------------------------------------------------------------------------- |
| `scripts/train_cnn.py`             | 18 KB | `packages/core/training/cnn_trainer.py`                                     |
| `scripts/evaluate_cnn.py`          | 14 KB | `packages/core/evaluation/cnn_evaluator.py`                                 |
| `scripts/inference_cnn.py`         | 13 KB | `packages/deployment/optimization/inference.py`                             |
| `scripts/export_onnx.py`           | 6 KB  | `packages/deployment/optimization/onnx_export.py`                           |
| `scripts/quantize_model.py`        | 7 KB  | `packages/deployment/optimization/quantization.py`                          |
| `scripts/train_spectrogram_cnn.py` | 12 KB | `packages/core/training/spectrogram_trainer.py`                             |
| `scripts/utilities/onnx_export.py` | 10 KB | Third copy — `packages/deployment/optimization/onnx_export.py` is canonical |

These are older standalone scripts that predated the `packages/` structure. The package
versions are more mature.

---

### B6. Duplicate PINN Ablation

| File                           | Size  | Why Dead                                                                          |
| ------------------------------ | ----- | --------------------------------------------------------------------------------- |
| `experiments/pinn_ablation.py` | 15 KB | Duplicate of `scripts/research/pinn_ablation.py` (13 KB). Keep one, remove other. |

---

### B7. `milestones/` Directory (203 files)

Full frozen snapshots of the project at 4 points in time. Contains copies of `data/`,
`models/`, `scripts/`, `training/`, `utils/` from each milestone. Git history preserves
all of this. **Truly dead weight.**

---

### B8. `docs/archive/` (72 files)

Phase usage guides, milestone READMEs, completed implementation plans, and a 160 KB
master roadmap. Historical records that git already preserves.

---

### B9. Runtime Files in Git

| File                              | Size   | Why Dead                        |
| --------------------------------- | ------ | ------------------------------- |
| `packages/dashboard/app.log`      | 430 KB | Runtime log file tracked in git |
| `packages/dashboard/dashboard.db` | 557 KB | SQLite database tracked in git  |

These should be in `.gitignore`, not in the repository.

---

## Summary Decision Matrix

| #   | Item                                          | Category                         | Your Decision      |
| --- | --------------------------------------------- | -------------------------------- | ------------------ |
| A1  | 6 unregistered model families (~20 files)     | 🔨 Wire into factory             | ☐ Build / ☐ Remove |
| A2  | Contrastive physics pretraining (38 KB)       | 🔨 Novel research contribution   | ☐ Build / ☐ Remove |
| A3  | Contrastive learning for spectrograms (11 KB) | 🔨 Self-supervised pretraining   | ☐ Build / ☐ Remove |
| A4  | Integration layer (33 KB)                     | 🔨 Pipeline orchestration        | ☐ Build / ☐ Remove |
| A5  | NAS search space (7 KB)                       | 🔨 Architecture search           | ☐ Build / ☐ Remove |
| A6  | Dashboard enterprise features (~300 KB)       | 🔨 Multi-user ready              | ☐ Keep / ☐ Remove  |
| A7  | Standalone XAI visualization (47 KB)          | 🔨 Research visualization        | ☐ Build / ☐ Remove |
| A8  | Utility scripts (74 KB)                       | 🔨 Move to proper locations      | ☐ Move / ☐ Remove  |
| B1  | Root-level model duplicates (40 KB)           | 💀 Superseded                    | ☐ Remove           |
| B2  | Dashboard dead code (3 files)                 | 💀 Never used                    | ☐ Remove           |
| B3  | Duplicate augmentation (15 KB)                | 💀 Superseded by PyTorch version | ☐ Remove           |
| B4  | CWRU dataset (12 KB)                          | 💀 Wrong bearing type            | ☐ Remove           |
| B5  | Root-level scripts (80 KB)                    | 💀 Superseded by packages        | ☐ Remove           |
| B6  | Duplicate PINN ablation (15 KB)               | 💀 Duplicate                     | ☐ Remove           |
| B7  | `milestones/` (203 files)                     | 💀 Git preserves history         | ☐ Remove           |
| B8  | `docs/archive/` (72 files)                    | 💀 Git preserves history         | ☐ Remove           |
| B9  | Runtime files in git (987 KB)                 | 💀 Should be gitignored          | ☐ Remove           |

> [!IMPORTANT]
> **Key correction:** My earlier reports overstated dead code. Many items I flagged as "dead"
> are actually **complete implementations awaiting integration.** The transformers are all
> domain-relevant 1D architectures, the contrastive physics module is a novel research
> contribution, and the integration layer is a useful orchestration framework with some
> phases left as placeholders to fill.
