# LSTM-PFD Master Convergence Plan

> A phased, self-expanding plan to transform LSTM-PFD from code-complete-but-unvalidated into a **publication-ready research platform** and **production-grade SaaS**.

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

## Phase 0: Git Hygiene & Branch Convergence
**Goal**: Clean slate on `main` before any code changes.

- [ ] **0.1** Merge `feature/hardening-and-docs` → `main` (11 commits: doc migrations, script relocations)
- [ ] **0.2** Delete 14 stale local branches already merged into `main`:
  - `Code-Overhaul`, `Feats/Deficiencies`, `doc-overhaul`, `docs/api-and-mkdocs`, `feat/hpo-dashboard-updates`, `feat/phase4-cicd-dr`, `feat/phase4-docker-codequality`, `feat/phase4-helm-kubernetes`, `feat/phase4-observability`, `feat/phase4-security-hardening`, `feature/dashboard-fixes-and-audit`, `feature/data-explorer-advanced-visualizations`, `fix/p1-magic-numbers`, `massive-improvements`
- [ ] **0.3** Clean up tracked artifacts that shouldn't be in git:
  - `packages/dashboard/dashboard.db`, `packages/dashboard/dashboard_data.db`
  - `packages/dashboard/app.log` (430KB)
  - `packages/dashboard/.env` (should be gitignored)
- [ ] **0.4** Verify `.gitignore` covers `*.db`, `*.log`, `.env` inside dashboard
- [ ] **0.5** Create fresh `develop` branch from clean `main` for all subsequent work

---

## Phase 1: Critical P0 Bug Fixes (Security & Stability)
**Goal**: Fix all show-stopping issues that block any meaningful use.

### 1A: Security Hardening
- [ ] **1.1** Remove hardcoded secrets from `docker-compose.yml` (lines 68, 114, 143) — use `.env` file references with `${VARIABLE}` syntax
- [ ] **1.2** Remove hardcoded `POSTGRES_PASSWORD=changeme` from Helm `values.yaml:89`
- [ ] **1.3** Fix CORS wildcard `["*"]` in `packages/deployment/api/config.py:50` — make environment-specific
- [ ] **1.4** Ensure Redis auth is enabled in production Helm values (`values-prod.yaml`)

### 1B: Code Portability
- [ ] **1.5** Remove all hardcoded `sys.path` manipulation (`/home/user/LSTM_PFD`) — 8 files in `fusion/`, `ensemble/` directories
- [ ] **1.6** Standardize all imports to use proper package paths (`packages.core.models.*`)

### 1C: Duplicate Elimination
- [ ] **1.7** Remove duplicate `HybridPINN` — keep `pinn/hybrid_pinn.py`, delete top-level `hybrid_pinn.py`
- [ ] **1.8** Remove duplicate `ResNet1D` — keep `resnet/resnet_1d.py`, delete top-level `resnet_1d.py`
- [ ] **1.9** Consolidate duplicate `FocalLoss` and `LabelSmoothingCrossEntropy` — keep versions in `cnn_losses.py`, update `losses.py` to re-export
- [ ] **1.10** Merge dual callback systems (`callbacks.py` + `cnn_callbacks.py`) into unified `Callback` base class

### 1D: Data Integrity
- [ ] **1.11** Fix HDF5 file handle leaks in `OnTheFlyTFRDataset` — add context managers
- [ ] **1.12** Replace Redis `KEYS` pattern with `SCAN` in cache service
- [ ] **1.13** Fix LR scheduler bug in `cnn_trainer.py:275` — handle `ReduceLROnPlateau` correctly

> [!IMPORTANT]
> 🔍 **IDB Audit Checkpoint #1**: After Phase 1 completion, IDB teams **1.1, 1.2, 3.2, 4.2** perform targeted re-audit of their domains to verify all P0 fixes and surface any newly-exposed P0/P1 issues. Discovered items are inserted into Phase 2.

---

## Phase 2: Architectural Consolidation
**Goal**: Unify fragmented systems into coherent, maintainable architectures.

### 2A: Training Infrastructure Unification
- [ ] **2.1** Create `BaseTrainer` abstract class with template method pattern (`_forward_pass`, `_compute_loss`, `_backward_pass`, `_optimizer_step`)
- [ ] **2.2** Refactor `CNNTrainer`, `PINNTrainer`, `SpectrogramTrainer`, `ProgressiveResizingTrainer`, `DistillationTrainer` to inherit `BaseTrainer`
- [ ] **2.3** Implement mixin architecture: `MixedPrecisionMixin`, `PhysicsLossMixin`, `SpecAugmentMixin`
- [ ] **2.4** Standardize checkpoint format with version field, model config, optimizer/scheduler/scaler states
- [ ] **2.5** Consolidate loss functions into `training/losses/` subdirectory (classification, contrastive, physics, distillation)
- [ ] **2.6** Consolidate schedulers: merge `cnn_schedulers.py` + `transformer_schedulers.py` → `training/schedulers.py`
- [ ] **2.7** Remove deprecated `optimizers.py`, update all references to `cnn_optimizer.py`

### 2B: Model Architecture Cleanup
- [ ] **2.8** Make `TSMixer`, `PatchTST`, `AttentionCNN1D`, `LightweightAttentionCNN`, `MultiScaleCNN1D`, `DilatedMultiScaleCNN` inherit `BaseModel`
- [ ] **2.9** Complete model registry in `model_factory.py` — register all ~55 models
- [ ] **2.10** Add decorator-based auto-registration: `@register_model("name")`
- [ ] **2.11** Add `export_onnx()` and `predict()` to `BaseModel`
- [ ] **2.12** Consolidate duplicate `ConvBlock` implementations
- [ ] **2.13** Consolidate 3 duplicate `PositionalEncoding` implementations in `transformer/`
- [ ] **2.14** Remove `legacy_ensemble.py` (superseded by `ensemble/`)
- [ ] **2.15** Replace all 45+ hardcoded magic numbers (`102400`, `20480`, `5000`) with constants from `utils/constants.py`

### 2C: Data Pipeline Cleanup
- [ ] **2.16** Split monolithic `signal_generator.py` (37KB, 935 lines) into focused modules
- [ ] **2.17** Externalize physics magic numbers into `PhysicsConstants` dataclass or YAML
- [ ] **2.18** Fix noise layer count docstring (7 → 8)
- [ ] **2.19** Unify random state to `np.random` only (remove `random.random()` from augmentation)
- [ ] **2.20** Add post-generation signal validation (NaN, Inf, std range checks)

### 2D: Dashboard Cleanup
- [ ] **2.21** Split `settings.py` (1,005 lines) into 6 modules under `layouts/settings/`
- [ ] **2.22** Split `experiment_comparison.py` (805 lines) — extract visualization helpers
- [ ] **2.23** Consolidate sidebar CSS (currently in 3 places) into `theme.css` only
- [ ] **2.24** Remove `custom.css` hardcoded colors — use CSS variables
- [ ] **2.25** Wire up unused `skeleton.py` components into layouts (replace `dcc.Loading` spinners)
- [ ] **2.26** Add ARIA accessibility attributes to all interactive elements

### 2E: Test Infrastructure
- [ ] **2.27** Move all `if __name__ == "__main__": test_*()` code from production files to `tests/`
- [ ] **2.28** Populate `__init__.py` exports in `packages/core/training/`
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
- [ ] **4.15** Ensure all scripts use `utils.reproducibility.set_seed` (replace 4 local implementations)

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
- [ ] **6.1** Create `nginx.conf` (referenced by `docker-compose.yml` but doesn't exist)
- [ ] **6.2** Create `ssl/` directory structure with self-signed cert generation script
- [ ] **6.3** Harden dashboard `Dockerfile` — add non-root user, health check, multi-stage build
- [ ] **6.4** Add CPU/memory resource limits to dashboard `docker-compose.yml`
- [ ] **6.5** Verify `docker-compose up` actually starts all 7 services cleanly

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
