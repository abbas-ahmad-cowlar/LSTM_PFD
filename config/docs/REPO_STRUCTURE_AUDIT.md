# Repository Structure Audit

**Date:** February 9, 2026  
**Scope:** All directories and sub-directories (excluding `venv/`, `.git/`, `__pycache__/`)

---

## 1. Current Directory Tree (Annotated)

```
LSTM_PFD/
├── .agent/                        # AI agent workflow configs
├── audit_reports/                 # ⚠️ EMPTY — never used
├── benchmarks/                    # Performance benchmark scripts (4 .py files)
├── checkpoints/                   # ⚠️ 9 EMPTY phase sub-dirs (phase1–phase9)
│   ├── phase1/ … phase9/         #    placeholder folders, no actual checkpoints stored here
│
├── config/                        # Configuration dataclasses (5 .py + README)
│
├── data/                          # Signal data layer + processing code (16 .py)
│   ├── generated/                 # Synthetic signal output directory
│   ├── processed/                 # Post-processing output
│   ├── raw/                       # Raw bearing vibration data
│   │   └── bearing_data/          # Fault-type sub-dirs (11 categories)
│   │       ├── ball_fault/
│   │       ├── cavitation/
│   │       ├── combined/
│   │       ├── imbalance/
│   │       ├── inner_race/
│   │       ├── looseness/
│   │       ├── misalignment/
│   │       ├── normal/
│   │       ├── oil_deficiency/
│   │       ├── oil_whirl/
│   │       └── outer_race/
│   └── spectrograms/              # Pre-computed spectrograms
│       ├── cwt/                   # Continuous Wavelet Transform
│       ├── stft/                  # Short-Time Fourier Transform
│       └── wvd/                   # Wigner-Ville Distribution
│
├── deliverables/                  # ⚠️ Legacy handover package
│   └── HANDOVER_PACKAGE/
│       ├── deployment/kubernetes/ # Duplicate K8s manifest (same as deploy/)
│       ├── models/                # model_metadata.json
│       └── tests/                 # Smoke tests
│
├── deploy/                        # Production deployment configs
│   ├── helm/lstm-pfd/             # Helm chart
│   │   ├── templates/             # K8s resource templates (8 files)
│   │   ├── Chart.yaml
│   │   ├── values.yaml
│   │   ├── values-staging.yaml
│   │   └── values-prod.yaml
│   ├── kubernetes/                # Standalone K8s deployment.yaml
│   └── monitoring/                # Prometheus alerts + Grafana dashboard
│
├── docs/                          # Project documentation hub
│   ├── analysis/                  # Technical analysis documents
│   ├── api/                       # OpenAPI spec (openapi.json)
│   ├── archive/                   # Archived legacy documentation
│   │   ├── Figures and livescripts/  # ⚠️ Deep MATLAB figure archive (20+ sub-dirs)
│   │   │   ├── Figures/
│   │   │   │   ├── Comparative/
│   │   │   │   ├── Faults/ (7 sub-dirs)
│   │   │   │   ├── Healthy/
│   │   │   │   └── Mixed/ (3 sub-dirs)
│   │   │   └── LiveScripts/Figures/Healthy/
│   │   ├── implementation_history/
│   │   ├── milestones/
│   │   └── planning/
│   ├── assets/                    # Diagrams, interactive HTML, screenshots
│   │   ├── diagrams/
│   │   ├── interactive/
│   │   └── screenshots/
│   ├── features/                  # Feature descriptions
│   ├── getting-started/           # Getting started guides
│   ├── idb_reports/               # IDB documentation overhaul
│   │   ├── compiled/
│   │   └── docs-cleanup/          # Per-IDB cleanup prompts (20+ files)
│   ├── javascripts/               # MkDocs extra JS (mathjax)
│   ├── operations/                # Operations docs
│   ├── paper/                     # Academic paper drafts
│   ├── reference/                 # Reference docs
│   ├── reports/                   # Generated reports
│   ├── research/                  # Research docs (PINN theory, XAI, etc.)
│   ├── stylesheets/               # MkDocs extra CSS
│   │   └── troubleshooting/
│   └── user-guide/                # User guide
│       └── phases/                # Phase-specific guides
│
├── experiments/                   # Experiment management (6 .py)
│
├── integration/                   # Unified pipeline + validators (4 .py)
│
├── logs/                          # ⚠️ EMPTY — runtime log output dir
│
├── milestones/                    # ⚠️ Legacy milestone snapshots
│   ├── milestone-1/ (82 children) # Full project snapshot (duplicates data/, models/, etc.)
│   │   ├── config/
│   │   ├── data/raw/bearing_data/ # French-named fault categories
│   │   ├── data_generation/
│   │   ├── evaluation/
│   │   ├── models/cnn|efficientnet|resnet/
│   │   ├── results/checkpoints_full/ + final_eval/
│   │   ├── scripts/
│   │   ├── training/
│   │   ├── utils/
│   │   └── visualization/
│   ├── milestone-2/ (41 children) # Smaller snapshot
│   ├── milestone-3/ (65 children) # CNN/transformer snapshot
│   └── milestone-4/ (15 children) # PINN snapshot
│
├── packages/                      # Core application packages
│   ├── core/                      # ML Engine (134 .py)
│   │   ├── evaluation/            # Evaluators, analyzers (17 .py)
│   │   ├── explainability/        # SHAP, LIME, IG, anchors (8 .py)
│   │   ├── features/              # Feature extraction/selection (12 .py)
│   │   ├── models/                # Model architectures (47 .py)
│   │   │   ├── classical/         # SVM, RF, gradient boosting
│   │   │   ├── cnn/               # 1D CNN variants
│   │   │   ├── efficientnet/      # EfficientNet 1D
│   │   │   ├── ensemble/          # Voting, stacking, boosting, MoE
│   │   │   ├── fusion/            # Early/late fusion
│   │   │   ├── hybrid/            # CNN-LSTM, CNN-TCN, CNN-Transformer
│   │   │   ├── nas/               # Neural Architecture Search
│   │   │   ├── physics/           # Physics-constrained CNN
│   │   │   ├── pinn/              # Physics-Informed Neural Networks
│   │   │   ├── resnet/            # ResNet 1D, SE-ResNet, WideResNet
│   │   │   ├── spectrogram_cnn/   # 2D spectrogram CNN
│   │   │   └── transformer/       # Signal transformer variants
│   │   ├── pipelines/             # Training/eval pipelines (4 .py)
│   │   ├── training/              # Trainers, schedulers, losses (23 .py)
│   │   └── transformers/          # Transformer architectures (23 .py)
│   │       └── advanced/          # BERT, GPT, T5, ViT, Swin
│   │
│   ├── dashboard/                 # Dash Web Application (153 .py)
│   │   ├── api/                   # REST API routes
│   │   ├── assets/                # Static CSS/JS
│   │   ├── callbacks/             # Dash callback handlers (26 .py)
│   │   ├── components/            # UI components (7 .py)
│   │   ├── config/                # Logging + security
│   │   ├── database/              # DB connection + migration
│   │   │   └── migrations/        # Alembic migrations
│   │   ├── integrations/          # ML pipeline bridges
│   │   ├── layouts/               # Page layouts (24 .py)
│   │   ├── middleware/            # Request middleware
│   │   ├── models/                # SQLAlchemy ORM (10 .py)
│   │   ├── services/              # Business logic (14 .py)
│   │   │   └── notification_providers/  # Slack, Teams, webhook
│   │   ├── storage/               # ⚠️ File storage (datasets, models, results, uploads)
│   │   ├── tasks/                 # Celery async tasks (5 .py)
│   │   ├── templates/email_templates/  # Email HTML templates
│   │   ├── tests/                 # ⚠️ Dashboard-specific tests (separate from /tests)
│   │   └── utils/                 # Dashboard utilities
│   │
│   ├── deployment/                # Deployment/inference package (10 .py)
│   │   ├── api/                   # FastAPI server
│   │   └── optimization/          # ONNX, quantization, inference
│   │
│   └── storage/                   # ⚠️ Empty storage placeholder
│       ├── datasets/
│       ├── models/
│       ├── results/
│       └── uploads/
│
├── reproducibility/               # Reproducibility configs + scripts (2 .py)
│   ├── config/                    # pinn_optimal.yaml
│   └── scripts/                   # run_all.py, set_seeds.py
│
├── scripts/                       # CLI & utility scripts (32 .py)
│   ├── disaster-recovery/         # DR scripts
│   ├── research/                  # Research experiment scripts (9 .py)
│   └── utilities/                 # Helper scripts (10 .py)
│
├── tests/                         # Test suite (29 .py)
│   ├── benchmarks/                # Performance benchmarks
│   ├── integration/               # Integration tests
│   ├── models/                    # Test model fixtures
│   ├── unit/                      # Unit tests
│   └── utilities/                 # Bug fix tests
│
├── utils/                         # ⚠️ Shared utilities (11 .py) — separate from packages/
│
└── visualization/                 # Visualization library (13 .py)
```

### Root-Level Files (for reference)

```
.coveragerc, .dockerignore, .env.example, .gitignore
CHANGELOG.md, CONTRIBUTING.md, README.md
dataset_card.yaml, docker-compose.yml, mkdocs.yml
pyproject.toml, pytest.ini
Dockerfile, Dockerfile.worker, setup.py
```

---

## 2. Issues Found

### 🔴 Critical Issues

#### Issue 1: `milestones/` — Giant Legacy Snapshots (200+ files)

- **Problem:** 4 milestone directories, each being a **full copy** of the project at that point in time — duplicating data/, models/, scripts/, training/, utils/, visualization/.
- **Size impact:** 200+ files of duplicated logic
- **Recommendation:** **Archive or delete entirely.** Git history already preserves every version. If needed for reference, tag the relevant commits instead.

#### Issue 2: Scattered Utilities — `utils/` vs `packages/dashboard/utils/`

- **Problem:** `utils/` at root has 11 .py files (checkpoint_manager, device_manager, early_stopping, etc.) that are **core ML utilities**, not generic helpers. Meanwhile, `packages/dashboard/utils/` has dashboard-specific utilities.
- **Recommendation:** **Move `utils/` into `packages/core/utils/`** to colocate with the ML engine. Root-level `utils/` is an anti-pattern in monorepo structures.

#### Issue 3: `visualization/` at Root — Wrong Location

- **Problem:** 13 .py files for ML visualization (signal_plots, saliency_maps, xai_dashboard) are at root level but are tightly coupled to `packages/core/`.
- **Recommendation:** **Move to `packages/core/visualization/`.**

#### Issue 4: `experiments/` at Root — Wrong Location

- **Problem:** 6 .py files (experiment_manager, hyperparameter_tuner, pinn_ablation, etc.) at root level but depend entirely on `packages/core/`.
- **Recommendation:** **Move to `packages/core/experiments/`.**

#### Issue 5: `benchmarks/` at Root — Wrong Location

- **Problem:** 4 .py benchmark files at root. Same pattern as above.
- **Recommendation:** **Move to `packages/core/benchmarks/` or `tests/benchmarks/`.**

---

### 🟡 Moderate Issues

#### Issue 6: `deliverables/HANDOVER_PACKAGE/` — Stale Duplicate

- **Problem:** Contains a K8s deployment.yaml that duplicates `deploy/kubernetes/deployment.yaml`, a model metadata JSON, and smoke tests. This was a one-time delivery artifact.
- **Recommendation:** **Archive to `docs/archive/`** or **delete.**

#### Issue 7: Empty Directories

| Directory               | Status                    | Recommendation                                  |
| ----------------------- | ------------------------- | ----------------------------------------------- |
| `audit_reports/`        | Empty                     | 🗑️ Delete                                       |
| `logs/`                 | Empty (gitignored target) | 🗑️ Delete (auto-created at runtime)             |
| `checkpoints/phase1–9/` | 9 empty sub-dirs          | 🗑️ Delete (auto-created at runtime, gitignored) |
| `packages/storage/`     | Empty placeholder         | 🗑️ Delete (never used)                          |

#### Issue 8: `integration/` at Root — Ambiguous

- **Problem:** 4 .py files (unified_pipeline, model_registry, validators). This is a thin orchestration layer over `packages/core/`.
- **Recommendation:** **Move to `packages/core/pipelines/`** or keep at root if it's genuinely cross-cutting.

#### Issue 9: `docs/archive/Figures and livescripts/` — Deep MATLAB Archive

- **Problem:** 20+ nested directories of MATLAB-era figures with French-named fault categories. Takes up archive space.
- **Recommendation:** Consider compressing into a single `.zip` or moving to external storage (Google Drive, OneDrive).

#### Issue 10: Two Test Locations

- **Problem:** `tests/` at root AND `packages/dashboard/tests/` — split test hierarchy.
- **Recommendation:** Either consolidate all tests under `tests/` (with `tests/dashboard/`), or keep dashboard tests co-located. Pick one convention — don't mix.

---

### 🟢 Minor Issues

#### Issue 11: `data/raw/bearing_data/` — English Fault Names

- **Note:** English naming (ball_fault, inner_race, etc.) while `milestones/milestone-1/data/raw/bearing_data/` uses French names (desalignement, desequilibre, jeu, etc.). No action needed since milestones should be removed.

#### Issue 12: `reproducibility/` at Root

- **Note:** Only 2 .py files and 1 YAML. Could merge into `config/` or `scripts/utilities/`.

#### Issue 13: `config/` at Root — Ambiguous Scope

- **Note:** Contains dataclass definitions for the ML engine. Could be moved into `packages/core/config/`.

---

## 3. Recommended Professional Structure

Below is what a well-organized ML research + production repo of this scale should look like:

```
LSTM_PFD/
├── .github/                    # CI/CD workflows, issue templates
├── data/                       # Data layer (same as now, well-organized ✅)
│   ├── raw/
│   ├── processed/
│   ├── generated/
│   └── spectrograms/
├── deploy/                     # Deployment configs (same as now ✅)
│   ├── helm/
│   ├── kubernetes/
│   └── monitoring/
├── docs/                       # Documentation (streamline sub-dirs)
│   ├── api/
│   ├── archive/
│   ├── getting-started/
│   ├── research/
│   └── user-guide/
├── packages/                   # All application code in one place
│   ├── core/                   # ML Engine ✅
│   │   ├── benchmarks/         # ← moved from /benchmarks
│   │   ├── config/             # ← moved from /config
│   │   ├── evaluation/
│   │   ├── experiments/        # ← moved from /experiments
│   │   ├── explainability/
│   │   ├── features/
│   │   ├── models/
│   │   ├── pipelines/          # ← absorb /integration
│   │   ├── training/
│   │   ├── transformers/
│   │   ├── utils/              # ← moved from /utils
│   │   └── visualization/      # ← moved from /visualization
│   ├── dashboard/              # Dashboard ✅ (keep as-is)
│   └── deployment/             # Deployment API ✅
├── reproducibility/            # Keep ✅ (small but important)
├── scripts/                    # CLI scripts ✅ (keep as-is)
├── tests/                      # All tests in one place
│   ├── unit/
│   ├── integration/
│   ├── benchmarks/
│   └── dashboard/              # ← moved from packages/dashboard/tests
└── [root config files]         # .gitignore, pyproject.toml, etc.
```

### Key Differences from Current

| Change                                            | What Moves                                                     | Why                                         |
| ------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------- |
| `utils/` → `packages/core/utils/`                 | 11 files                                                       | These are ML utilities, not generic helpers |
| `visualization/` → `packages/core/visualization/` | 13 files                                                       | Tightly coupled to core models              |
| `experiments/` → `packages/core/experiments/`     | 6 files                                                        | ML experiment code belongs with ML engine   |
| `benchmarks/` → `packages/core/benchmarks/`       | 4 files                                                        | Same reasoning                              |
| `config/` → `packages/core/config/`               | 6 files                                                        | ML config dataclasses belong with ML engine |
| `integration/` → `packages/core/pipelines/`       | 4 files                                                        | Merge with existing pipelines module        |
| Delete `milestones/`                              | 200+ files                                                     | Git tags replace folder snapshots           |
| Delete `deliverables/`                            | 3 files                                                        | One-time artifact, no longer needed         |
| Delete empty dirs                                 | `audit_reports/`, `logs/`, `checkpoints/`, `packages/storage/` | Noise                                       |

---

## 4. Impact Summary

| Metric                               | Before                                                                        | After            |
| ------------------------------------ | ----------------------------------------------------------------------------- | ---------------- |
| Top-level directories                | 18                                                                            | 8                |
| Orphaned root-level code dirs        | 5 (`utils/`, `visualization/`, `experiments/`, `benchmarks/`, `integration/`) | 0                |
| Empty directories                    | 4+ dirs (~15 sub-dirs)                                                        | 0                |
| Legacy snapshots (`milestones/`)     | 200+ files                                                                    | 0 (use git tags) |
| Duplicated configs (`deliverables/`) | ~3 files                                                                      | 0                |

### Before vs After — Top Level

```diff
  LSTM_PFD/
- ├── audit_reports/          # empty
- ├── benchmarks/             # orphaned
- ├── checkpoints/            # empty
- ├── config/                 # orphaned
  ├── data/
- ├── deliverables/           # legacy
  ├── deploy/
  ├── docs/
- ├── experiments/            # orphaned
- ├── integration/            # orphaned
- ├── logs/                   # empty
- ├── milestones/             # legacy snapshots
  ├── packages/
  ├── reproducibility/
  ├── scripts/
  ├── tests/
- ├── utils/                  # orphaned
- └── visualization/          # orphaned
+ └── [8 clean top-level dirs]
```
