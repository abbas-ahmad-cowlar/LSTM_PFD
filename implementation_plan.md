# LSTM-PFD Master Convergence Plan

> A phased, self-expanding plan to transform LSTM-PFD from code-complete-but-unvalidated into a **publication-ready research platform** and **production-grade SaaS**.

> [!IMPORTANT]
> **Agent Instructions**: Any agent working on this plan **MUST** update this file after completing items. Mark items `[x]` when done, `[/]` when in-progress. Add notes on corrections or discoveries inline. This file is the single source of truth for project progress.
>
> **Branch**: All work is on `fix/master-plan` (do NOT commit directly to `main`).
>
> **Last Updated**: 2026-03-15T10:30 PKT — Phase 1 complete, Phase 2 mostly done (2.1-2.3, 2.5-2.10, 2.13-2.17, 2.21-2.22 all done). Phase 6 security done. Phase 3 smoke-tested (22 commits).

---

## Design Philosophy: IDB Audit Feedback Loop

Each phase includes **IDB Audit Checkpoints** (🔍). When we reach one, the assigned IDB team (a dedicated agent pass) performs a targeted audit of its domain, then **inserts newly-discovered issues as sub-steps** into the plan ahead. This creates a self-expanding feedback loop:

```
Execute steps → Hit audit checkpoint → IDB agents scan their domain
  → New issues discovered → Injected into upcoming phases → Continue
```

The 18 IDB teams are already defined in `config/docs/idb_reports/`:

| Domain | IDBs |
|---|---|
| **Core ML Engine** | 1.1 Models, 1.2 Training, 1.3 Evaluation, 1.4 Features, 1.5 XAI |
| **Dashboard Platform** | 2.1 UI, 2.2 Services, 2.3 Callbacks, 2.4 Tasks |
| **Data Engineering** | 3.1 Signal Gen, 3.2 Data Loading, 3.3 Storage |
| **Infrastructure** | 4.1 Database, 4.2 Deployment, 4.3 Testing, 4.4 Config |
| **Research & Integration** | 5.1 Research, 5.2 Visualization, 6.0 Integration |

---

## Phase 0: Git Hygiene & Branch Convergence ✅
**Goal**: Clean slate on `main` before any code changes.

- [x] **0.1** Merge `feature/hardening-and-docs` → `main` (fast-forward, 11 commits)
- [x] **0.2** Delete 14 stale local branches already merged into `main`
- [x] **0.3** Verified `.gitignore` excludes `*.db`, `*.log`, `.env` — these are NOT tracked in git
- [x] **0.4** Verified `.gitignore` covers dashboard artifacts
- [x] **0.5** Created `fix/master-plan` branch from clean `main` for all subsequent work

---

## Phase 1: Critical P0 Bug Fixes (Security & Stability) ✅
**Goal**: Fix all show-stopping issues that block any meaningful use.

### 1A: Security Hardening
- [x] **1.1** Remove hardcoded secrets from `docker-compose.yml` — replaced with `${VARIABLE}` env refs (4 service blocks)
- [x] **1.2** Remove hardcoded `POSTGRES_PASSWORD=changeme` from Helm `values.yaml:89`
- [x] **1.3** Fix CORS wildcard `["*"]` in `config.py` — now defaults to localhost origins, overridable via `CORS_ORIGINS` env
- [x] **1.4** Redis auth enabled in production Helm values — added `redis.auth.enabled: true` and `password: ""` with instructions to set via `--set`

### 1B: Code Portability
- [x] **1.5** Remove all `sys.path` hacks — Phase 1: removed hardcoded `/home/user/LSTM_PFD` paths. Phase 2 (IDB 1.1 audit): removed 8 remaining `sys.path.insert()` calls in `fusion/`, `ensemble/`, `transformer/`, `hybrid/` — all replaced with relative imports (`from ..base_model import BaseModel`)
- [x] **1.5b** Removed sys.path hacks from `pinn_trainer.py`, `spectrogram_trainer.py`, `physics_loss_functions.py` (relative imports), and `api/main.py` (relative imports)
- [x] **1.5c** Removed sys.path hacks from `streaming_hdf5_dataset.py` and `contrast_learning_tfr.py`
- [x] **1.6** Standardize all imports to use proper package paths — shim files (`hybrid_pinn.py`, `resnet_1d.py`) converted from absolute to relative imports; `cnn/cnn_1d.py` also converted

### 1C: Duplicate Elimination
- [x] **1.7** Consolidated `HybridPINN` — top-level converted to re-export shim, `pinn/hybrid_pinn.py` is canonical (480 lines)
- [x] **1.8** Consolidated `ResNet1D` — top-level converted to re-export shim, `resnet/resnet_1d.py` is canonical (378 lines)
- [x] **1.9** Consolidated `FocalLoss` and `LabelSmoothingCrossEntropy` — `losses.py` now re-exports from `cnn_losses.py`
- [ ] **1.10** Merge dual callback systems — deferred (divergent interfaces: trainer-centric vs logs-centric, needs careful testing). *IDB 1.2 audit confirms `cnn_callbacks.py` is the more complete system (has batch hooks, `CallbackList`); merge `callbacks.py` INTO `cnn_callbacks.py`.*
- [x] **1.10b** Consolidated `CNN1D` — top-level `cnn_1d.py` (232 lines) converted to re-export shim pointing to canonical `cnn/cnn_1d.py` (313 lines); `create_cnn1d` factory added to canonical file

### 1D: Data Integrity
- [x] **1.11** Fixed HDF5 file handle leaks in `OnTheFlyTFRDataset` — added `close()` method with `_h5_closed` flag
- [x] **1.11b** Fixed HDF5 file handle leak in `MultiTFRDataset` — same pattern with `close()` and `_h5_closed`
- [x] **1.11c** Fixed 2 bare `except:` in `streaming_hdf5_dataset.py` → `except Exception:`. `tfr_dataset.py` already clean (no bare excepts found).
- [x] **1.11d** Fixed thread-safety in `StreamingHDF5Dataset` LRU cache — added `threading.Lock` around `_cache` dict and `_cache_order` list
- [x] **1.12** Replaced Redis `KEYS` with `SCAN` iterator in `cache_service.py` — production-safe cursor-based pattern match
- [x] **1.13** Fix LR scheduler bug in `cnn_trainer.py` — now handles `ReduceLROnPlateau` with metric arg

> [!IMPORTANT]
> 🔍 **IDB Audit Checkpoint #1**: After Phase 1 completion, IDB teams **1.1, 1.2, 3.2, 4.2** perform targeted re-audit of their domains to verify all P0 fixes and surface any newly-exposed P0/P1 issues. Discovered items are inserted into Phase 2.

---

## Phase 2: Architectural Consolidation
**Goal**: Unify fragmented systems into coherent, maintainable architectures.

### 2A: Training Infrastructure Unification
- [x] **2.1** Created `BaseTrainer` abstract class in `base_trainer.py` with template method pattern (`_forward_pass`, `_compute_loss`, `_backward_pass`, `_optimizer_step`)
- [x] **2.2** Refactored `CNNTrainer`, `ProgressiveResizingTrainer`, `DistillationTrainer` to inherit `BaseTrainer`
- [x] **2.2b** N/A — PINNTrainer already uses `fit()` from parent `Trainer`, no standalone `train()` method exists
- [x] **2.2c** Fixed ReduceLROnPlateau in `progressive_resizing.py` and `knowledge_distillation.py` — now checks isinstance and passes val_loss as metric
- [x] **2.2d** Fixed batch-level loss averaging in both trainers — now uses `loss.item() * batch_size` and divides by total samples
- [x] **2.3** Implemented mixin architecture in `mixins.py`: `MixedPrecisionMixin`, `PhysicsLossMixin`, `SpecAugmentMixin`
- [ ] **2.4** Standardize checkpoint format with version field, model config, optimizer/scheduler/scaler states
- [x] **2.5** Consolidated loss functions into `training/losses/` subdirectory — `losses.py` is now a re-export shim, real implementations in `losses/classification.py`, `physics_loss_functions.py`, `contrastive/losses.py`
- [x] **2.6** Consolidated schedulers: `schedulers.py` (417 lines) has all implementations + unified factory `create_scheduler()`. `cnn_schedulers.py` and `transformer_schedulers.py` are now backward-compat re-export shims
- [x] **2.7** Deleted deprecated `optimizers.py` — no code imports it (all use `cnn_optimizer.py`)

### 2B: Model Architecture Cleanup
- [x] **2.8** All 35 top-level models now inherit `BaseModel` with `get_config()` — includes PatchTST, TSMixer, AttentionCNN1D, LightweightAttentionCNN, MultiScaleCNN1D, DilatedMultiScaleCNN, SignalEncoder, ContrastiveClassifier, all PINN/hybrid/fusion/ensemble/spectrogram models
- [x] **2.9** Complete model registry in `model_factory.py` — 65+ entries covering all models with factory functions
- [x] **2.10** Added `register_model()` decorator supporting both function and decorator usage
- [ ] **2.11** Add `export_onnx()` and `predict()` to `BaseModel`
- [x] **2.12** N/A — only 1 `ConvBlock1D` in `cnn/conv_blocks.py`, no duplicates to consolidate
- [x] **2.13** Consolidated PositionalEncoding — PatchTST now imports from `signal_transformer.py` (-20 lines)
- [x] **2.14** Converted `legacy_ensemble.py` to re-export shim — imports from `ensemble/` subpackage, provides backward-compat aliases (-360 lines)
- [x] **2.15** Replaced magic numbers: TSMixer (102400→SIGNAL_LENGTH, 11→NUM_CLASSES), `progressive_resizing.py` (102400→SIGNAL_LENGTH), `physics_loss_functions.py` (4×20480→SAMPLING_RATE), `pinn_trainer.py` (20480→SAMPLING_RATE), `feature_extractor.py` (20480→SAMPLING_RATE), `schedulers.py` (deprecated verbose=True removed)
- [ ] **2.8b** Audit and document `training/contrastive/` sub-module (5 files: `dataset.py`, `losses.py`, `physics_similarity.py`, `pretrainer.py`, `__init__.py`) — currently undocumented and not covered by any IDB report *(IDB 1.2 audit)*

### 2C: Data Pipeline Cleanup
- [x] **2.16** Split `signal_generator.py` into `data/signal_generation/` package: `generator.py`, `fault_modeler.py`, `noise_generator.py`, `metadata.py`, `__init__.py`
- [x] **2.16b** Replaced duplicate `Compose` in `cnn_transforms.py` with import from canonical `transforms.py`; added `__repr__` to canonical
- [ ] **2.16c** Merge duplicate dataloader modules `dataloader.py` + `cnn_dataloader.py` into single `dataloader.py` with all features (`DataLoaderConfig`, `InfiniteDataLoader`, `prefetch_to_device`, `compute_class_weights`) *(IDB 3.2 audit)*
- [x] **2.16d** Added 40+ missing exports to `data/__init__.py` — CNN datasets, streaming datasets, TFR datasets, CNN transforms
- [x] **2.16e** Verified: no `print()` calls in `tfr_dataset.py` (already clean)
- [x] **2.16f** Drop CWRU entirely — deleted orphaned `data/__pycache__/cwru_dataset.cpython-314.pyc`, removed all CWRU references from 12 files (`dataset_card.yaml`, `benchmarks/`, `data/` READMEs, `tests/`, `scripts/`, dashboard, docs), renamed `compare_with_cwru_benchmark` → `compare_with_journal_bearing_benchmark`, updated `dataset_card.yaml` class names to match actual hydrodynamic fault types
- [x] **2.17** Externalized physics magic numbers into `utils/physics_constants.py` (`BearingPhysics`, `ViscosityModel`, `FaultAmplitudes`, `NoiseDefaults`)
- [x] **2.18** N/A — noise layer count docstring not found, likely already fixed
- [x] **2.19** Unified random state to `np.random` in `spectrogram_augmentation.py` (5 calls) and `signal_augmentation.py` (2 calls) — removed `import random`
- [ ] **2.20** Add post-generation signal validation (NaN, Inf, std range checks)
- [ ] **2.20b** Integrate `data_validator.py` into dataset `from_hdf5()` / `__init__()` methods — currently the 17KB validator exists but is never called from any dataset class *(IDB 3.2 audit)*

### 2D: Dashboard Cleanup
- [x] **2.21** Split `settings.py` (1,005 lines) into 6 modules under `layouts/settings/`: `main.py`, `profile_tab.py`, `security_tab.py`, `api_keys_tab.py`, `notifications_tab.py`, `webhooks_tab.py`
- [x] **2.22** Split `experiment_comparison.py` — extracted into `layouts/experiment_comparison/` package
- [ ] **2.23** Consolidate sidebar CSS (currently in 3 places) into `theme.css` only
- [ ] **2.24** Remove `custom.css` hardcoded colors — use CSS variables
- [ ] **2.25** Wire up unused `skeleton.py` components into layouts (replace `dcc.Loading` spinners)
- [ ] **2.26** Add ARIA accessibility attributes to all interactive elements

### 2E: Test Infrastructure
- [ ] **2.27** Move all `if __name__ == "__main__": test_*()` code from production files to `tests/`
- [x] **2.27b** Verified: `signal_length` NameError not found in current files — already fixed by IDB agents or different version
- [x] **2.28** Populated `training/__init__.py` with exports for all trainers, callbacks, and loss functions
- [ ] **2.29** Add missing dashboard callback tests (tightly coupled UI/Service layer)

> [!IMPORTANT]
> 🔍 **IDB Audit Checkpoint #2**: All 18 IDB teams perform full re-audit after consolidation. Focus: verify no regressions from refactoring, identify remaining P1/P2 issues, and check cross-domain integration contracts still hold (trainer↔evaluator, model↔factory, dashboard↔services).

---

## Phase 3: End-to-End Experimentation
**Goal**: Generate real data, train real models, produce real benchmarks. **This is the most critical phase.**

### 3A: Data Generation
- [ ] **3.1** Generate production-grade synthetic dataset using `signal_generator.py`:
  - 11 fault types × 100+ samples each × 3 severities
  - Train/Val/Test splits (70/15/15)
  - Export as HDF5 with full metadata
- [ ] **3.2** Validate generated dataset (class balance, signal statistics, no NaN/Inf)
- [ ] **3.3** Version dataset with DVC, push to remote storage

### 3B: Baseline Model Training
- [ ] **3.4** Train Classical ML baselines (SVM, Random Forest, XGBoost) — fill Phase 1 checkpoints
- [ ] **3.5** Train 1D CNN baseline — fill Phase 2 checkpoints
- [ ] **3.6** Train ResNet-1D variant — fill Phase 3 checkpoints
- [ ] **3.7** Train Transformer (SignalTransformer) — fill Phase 4 checkpoints
- [ ] **3.8** Train PINN (HybridPINN with ResNet-18 backbone) — fill Phase 6 checkpoints
- [ ] **3.9** Train Ensemble (soft-voting of top performers) — fill Phase 8 checkpoints
- [ ] **3.10** Each training produces: saved `.pth` checkpoint, training history JSON, confusion matrix plot

### 3C: Benchmark Generation
- [ ] **3.11** Run standardized evaluation on all trained models: accuracy, precision, recall, F1, per-class metrics
- [ ] **3.12** Run inference latency benchmarks (CPU and GPU if available)
- [ ] **3.13** Run ONNX export + quantization on best model
- [ ] **3.14** **Update `README.md` performance table** with real numbers (replace `[PENDING]`)
- [ ] **3.15** **Update `CHANGELOG.md`** accuracy/latency (replace `[PENDING]`)
- [ ] **3.16** **Fix `model_metadata.json`** in handover package with real checksums and real metrics

### 3D: Checkpoint & Artifact Management
- [ ] **3.17** Populate `checkpoints/phase1/` through `checkpoints/phase9/` with actual model files
- [ ] **3.18** Export best model to ONNX format, place in `deliverables/HANDOVER_PACKAGE/models/`
- [ ] **3.19** Generate quantized (INT8) ONNX model for edge deployment

> [!IMPORTANT]
> 🔍 **IDB Audit Checkpoint #3**: IDB teams **1.1, 1.2, 1.3, 1.4, 3.1** audit results. Verify: training converged, no data leakage (via `check_data_leakage.py`), metrics are scientifically valid, checkpoints load correctly across all evaluator types.

---

## Phase 4: Research Validation & Publication
**Goal**: Execute all research scripts, produce publication-quality results, finalize IEEE manuscript.

### 4A: Research Experiment Execution
- [ ] **4.1** Run `ablation_study.py` — full (all components, 5 seeds, 50 epochs)
- [ ] **4.2** Run `contrastive_physics.py` — SimCLR pretraining vs supervised baseline
- [ ] **4.3** Run `pinn_comparison.py` — Raissi/Karniadakis vs Our formulation
- [ ] **4.4** Run `pinn_ablation.py` — 9-configuration ablation with McNemar's test
- [ ] **4.5** Run `transformer_benchmark.py` — CNN1D vs SignalTransformer vs PatchTST vs TSMixer
- [ ] **4.6** Run `hyperparameter_sensitivity.py` — physics loss weight grid search
- [ ] **4.7** Run `ood_testing.py` — severity shift + novel fault robustness
- [ ] **4.8** Run `failure_analysis.py` — misclassification analysis with real data
- [ ] **4.9** Run `xai_metrics.py` — faithfulness, stability, sparsity evaluation

### 4B: Research Infrastructure Fixes
- [ ] **4.10** Extract common `ExperimentRunner` base class to eliminate ~200 lines of duplicated training loops
- [ ] **4.11** Create shared data loading module for research scripts
- [ ] **4.12** Add CLI to `xai_metrics.py`
- [ ] **4.13** Add statistical significance tests (paired t-test/Wilcoxon) to `transformer_benchmark.py`
- [ ] **4.14** Standardize all research outputs: CSV + JSON + LaTeX tables + publication-quality PNGs
- [x] **4.15** Replaced 5 local seed implementations with canonical `utils.reproducibility.set_seed` imports

### 4C: Publication Preparation
- [ ] **4.16** Update IEEE TII manuscript with real experimental results
- [ ] **4.17** Generate publication-quality figures (consistent fonts, colors, DPI=300+)
- [ ] **4.18** Produce LaTeX tables from all benchmark results
- [ ] **4.19** Create Zenodo reproducibility package (code + data + configs + exact seeds)
- [ ] **4.20** Update `reproducibility/zenodo_metadata.yaml` with real DOI
- [ ] **4.21** Write supplementary materials (extended ablation tables, per-class confusion matrices)

> [!IMPORTANT]
> 🔍 **IDB Audit Checkpoint #4**: IDB teams **5.1, 5.2, 1.3, 1.5** audit research outputs. Verify: statistical significance of all claims, figure quality meets IEEE submission standards, XAI metrics are computed correctly, all experiments are reproducible with documented seeds.

---

## Phase 5: Dashboard Verification & Polish
**Goal**: Ensure the dashboard actually works end-to-end with real models and data.

### 5A: Dashboard Functional Testing
- [ ] **5.1** Verify dashboard boots with `python app.py` (using SQLite for dev)
- [ ] **5.2** Test data generation flow through dashboard UI
- [ ] **5.3** Test experiment wizard → training → results flow
- [ ] **5.4** Test XAI dashboard with a real trained model
- [ ] **5.5** Test experiment comparison with 2+ real experiments

### 5B: Dashboard-Model Integration
- [ ] **5.6** Wire real model checkpoints into dashboard model loading
- [ ] **5.7** Verify Celery task execution for training, HPO, XAI computation
- [ ] **5.8** Test notification service (Slack/Teams webhooks) with real events
- [ ] **5.9** Verify search/tag functionality works with real experiment data

### 5C: UI Polish
- [ ] **5.10** Fix all orphaned UI elements (buttons that do nothing, empty tabs)
- [ ] **5.11** Add real-time training progress from Celery tasks to training monitor
- [ ] **5.12** Integrate skeleton loaders for all async content
- [ ] **5.13** Visual QA pass on all 24 layout pages

> [!IMPORTANT]
> 🔍 **IDB Audit Checkpoint #5**: IDB teams **2.1, 2.2, 2.3, 2.4** audit the dashboard after real data integration. Focus: callback errors in browser console, service layer exceptions under real data, UI rendering issues, and performance under load.

---

## Phase 6: Production Deployment Pipeline
**Goal**: Make the Docker/K8s deployment actually deployable.

### 6A: Docker Fix-ups
- [x] **6.0a** Fixed docker-compose.yml — replaced 3 hardcoded `lstm_password` with `${POSTGRES_PASSWORD:?...}` env-var refs
- [x] **6.0b** N/A — `sys.path.insert()` already removed from `api/main.py` in Phase 1
- [x] **6.0c** Updated `.env.example` — replaced hardcoded password with `CHANGE_ME` placeholder, added `POSTGRES_PASSWORD` var
- [ ] **6.1** Create `nginx.conf` (referenced by `docker-compose.yml` but doesn't exist)
- [ ] **6.2** Create `ssl/` directory structure with self-signed cert generation script
- [ ] **6.3** Harden dashboard `Dockerfile` — add non-root user, health check, multi-stage build
- [ ] **6.4** Add CPU/memory resource limits to dashboard `docker-compose.yml`
- [ ] **6.5** Verify `docker-compose up` actually starts all 7 services cleanly
- [ ] **6.5a** Fix ONNX static quantization stub (`onnx_export.py:474-476`) — either implement calibration-based static quantization or raise `NotImplementedError` instead of silently returning FP32 model *(IDB 4.2 audit P1)*
- [ ] **6.5b** Fix `benchmark_inference()` in `inference.py:474-479` — currently skips PyTorch backend; implement `model_class` parameter support *(IDB 4.2 audit P1)*
- [x] **6.5c** Added `weights_only=True` to `torch.load()` in `quantization.py`, `inference.py`, and `attention_viz.py` — security fix
- [x] **6.5d** N/A — `onnx.optimizer` not found in codebase; already clean
- [x] **6.5e** Fixed `optimize_for_deployment()` parameter shadowing — `pruning_amount` now respects caller override instead of being unconditionally overwritten

### 6B: Kubernetes & Helm
- [ ] **6.6** Add GPU resource requests (`nvidia.com/gpu`) to K8s deployment
- [ ] **6.7** Add K8s network policies for pod-to-pod isolation
- [ ] **6.8** Replace placeholder domains (`api.lstm-pfd.example.com`) with configurable Helm values
- [ ] **6.9** Add K8s Secrets for all sensitive configuration
- [ ] **6.10** Test Helm install on a real K8s cluster (or minikube)

### 6C: FastAPI Inference Server
- [ ] **6.11** Verify `packages/deployment/api/main.py` actually loads a model and serves predictions
- [ ] **6.12** Fix deprecated `@app.on_event` decorators → use FastAPI lifespan
- [ ] **6.13** Add rate limiting middleware (SlowAPI or similar)
- [ ] **6.14** Run smoke tests (`deliverables/HANDOVER_PACKAGE/tests/smoke_tests.py`) against running API
- [ ] **6.15** Benchmark API throughput under load

### 6D: CI/CD Verification
- [ ] **6.16** Verify all 6 GitHub Actions workflows pass on current codebase
- [ ] **6.16a** Update deprecated GitHub Actions versions: `actions/checkout@v3`→`v4`, `actions/setup-python@v4`→`v5`, replace archived `actions/create-release@v1` with `softprops/action-gh-release` *(IDB 4.2 audit P2)*
- [ ] **6.17** Fix any broken workflow steps
- [ ] **6.18** Add workflow for automated model testing on PR

> [!IMPORTANT]
> 🔍 **IDB Audit Checkpoint #6**: IDB teams **4.1, 4.2, 4.3, 4.4** audit infrastructure. Verify: all Docker services start, K8s manifests apply cleanly, CI passes, API handles edge cases, monitoring alerts fire correctly.

---

## Phase 7: Test Coverage & Quality Gate
**Goal**: Comprehensive test suite with known pass/fail status.

- [ ] **7.1** Run existing test suite (`pytest -v`), document pass/fail for all 21 test files
- [ ] **7.2** Fix all failing tests
- [ ] **7.3** Add integration tests for: data generation → training → evaluation → XAI pipeline
- [ ] **7.4** Add dashboard callback tests (currently missing — P0 from Jan audit)
- [ ] **7.5** Add Celery task tests with mocked Redis/Postgres
- [ ] **7.6** Run load tests (`tests/load_tests.py`) and document results
- [ ] **7.7** Run stress tests (`tests/stress_tests.py`) and document results
- [ ] **7.8** Achieve ≥80% test coverage (currently unknown)
- [ ] **7.9** Configure `pytest` + `coverage` in CI pipeline
- [ ] **7.10** Populate `audit_reports/` directory with test results and coverage reports

> [!IMPORTANT]
> 🔍 **IDB Audit Checkpoint #7**: All 18 IDB teams perform final audit. This is the convergence checkpoint — every team verifies their domain is production-ready, all P0/P1 issues from previous checkpoints are resolved, and documentation matches actual behavior.

---

## Phase 8: Documentation Finalization
**Goal**: All docs match reality. No stale references, no TODO placeholders.

- [ ] **8.1** Sweep all `*.md` files for stale content, broken links, outdated paths
- [ ] **8.2** Update `README.md` with real performance numbers and verified quickstart
- [ ] **8.3** Update `CHANGELOG.md` [Unreleased] section with all changes from this plan
- [ ] **8.4** Verify MkDocs builds cleanly (`mkdocs build`)
- [ ] **8.5** Update `CONTRIBUTING.md` with current project structure
- [ ] **8.6** Archive or delete stale IDB report content that no longer applies
- [ ] **8.7** Create final `RELEASE_NOTES.md` for v2.0.0

---

## Phase 9: SaaS Productionization
**Goal**: Transform from research tool into deployable SaaS for industrial clients.

- [ ] **9.1** Design multi-tenant architecture (per-client data isolation)
- [ ] **9.2** Implement user management with proper JWT auth flow
- [ ] **9.3** Add billing/usage tracking hooks
- [ ] **9.4** Create onboarding flow for factory data import (MATLAB `.mat` files)
- [ ] **9.5** Build admin dashboard for client management
- [ ] **9.6** Add data retention policies and GDPR compliance hooks
- [ ] **9.7** Create client-facing SLA documentation
- [ ] **9.8** Deploy to cloud (AWS/GCP/Azure) with Terraform or Pulumi IaC
- [ ] **9.9** Set up monitoring, alerting, and on-call rotation
- [ ] **9.10** Load test under realistic multi-tenant workload

---

## Summary: Effort Estimates

| Phase | Description | Estimated Effort |
|---|---|---|
| 0 | Git Hygiene | 1-2 hours |
| 1 | P0 Bug Fixes | 2-3 days |
| 2 | Architectural Consolidation | 1-2 weeks |
| 3 | End-to-End Experimentation | 1-2 weeks |
| 4 | Research & Publication | 1-2 weeks |
| 5 | Dashboard Verification | 3-5 days |
| 6 | Deployment Pipeline | 1 week |
| 7 | Test Coverage | 1 week |
| 8 | Documentation | 2-3 days |
| 9 | SaaS Productionization | 2-4 weeks |

**Total: ~8-12 weeks** (with IDB audit loops adding ~20% overhead for discovered issues)

---

## IDB Audit Schedule Summary

| Checkpoint | After Phase | IDB Teams | Focus |
|---|---|---|---|
| #1 | Phase 1 | 1.1, 1.2, 3.2, 4.2 | P0 fixes verified, new exposure |
| #2 | Phase 2 | All 18 | Refactoring regressions, integration contracts |
| #3 | Phase 3 | 1.1, 1.2, 1.3, 1.4, 3.1 | Training validity, data leakage, metrics |
| #4 | Phase 4 | 5.1, 5.2, 1.3, 1.5 | Research rigor, figure quality, reproducibility |
| #5 | Phase 5 | 2.1, 2.2, 2.3, 2.4 | Dashboard under real data |
| #6 | Phase 6 | 4.1, 4.2, 4.3, 4.4 | Infrastructure readiness |
| #7 | Phase 7 | All 18 | Final convergence verification |
