# LSTM-PFD Project Audit — 2026-06-11

> **Purpose**: Establish ground truth about what this project actually is, what works, what is broken,
> what is facade, and what is fabricated — as the basis for a convergence plan that strips
> unnecessary complexity and makes the project workable and authentic.
>
> **Method**: Direct verification (file inventories, checkpoint/HDF5 inspection, full test run,
> targeted greps) plus five scoped read-only audit agents (core ML, dashboard, deployment/infra,
> data+research, docs). Where agent claims conflicted with direct evidence, direct evidence wins —
> discrepancies are noted in §9. **Nothing in this report is taken from project documentation
> on trust.**

---

## 1. Executive Summary

**What this project is**: A synthetic-data bearing fault diagnosis research platform
(11 hydrodynamic/journal-bearing fault classes, physics-based signal generator, ~34 PyTorch
model architectures, classical ML baselines, XAI methods) wrapped in an aspirational
"enterprise SaaS" shell (Dash dashboard + PostgreSQL + Celery + Docker/K8s/Helm + FastAPI).

**Verdict**: The project is a **working core pipeline buried inside ~5–10x of unvalidated
scaffolding**. Exactly one end-to-end path has ever been proven: synthetic data generation →
HDF5 → CNN1D training on CPU (interrupted at epoch 30/100, best val acc 88.8%). Everything
else — 33 other architectures, PINN physics claims, the dashboard, deployment, benchmarks,
research scripts, the IEEE paper — is either unvalidated, partially broken, or fabricated.

| Dimension | Reality |
|---|---|
| Code size | 489 Python files, ~130K lines (excl. venv/site) |
| Docs | 136 markdown files + LaTeX paper + 2.6MB PDF report |
| Trained models | **1** (CNN1D, incomplete run) out of 34 architectures |
| `results/` | **Empty** |
| `audit_reports/` (before this file) | **Empty** |
| Test suite | **45 failed, 220 passed, 13 skipped** (+1 file crashes pytest collection) |
| Dashboard | **Cannot boot** (NameError at import) |
| `docker-compose up` | **Cannot start** (missing files, wrong paths) |
| `mkdocs build` | **Cannot build** (docs/ dir doesn't exist) |
| IEEE paper | **Fabricated results** (98.1% accuracy, fake ablations, "domain expert validation") |

---

## 2. Hard Evidence (verified directly, 2026-06-11)

### 2.1 What exists on disk

| Artifact | Status | Evidence |
|---|---|---|
| `data/generated/dataset.h5` | ✅ Real, 1.03 GB | 2,860 signals × 102,400 samples, 11 classes, train/val/test 2002/429/429, seed 42, generated 2026-03-15 |
| `data/generated/fault_testing_v1.h5` | ✅ Real, 2.3 GB | older generation run (2026-01-19) |
| `data/processed/dataset.h5` + `.dvc` | ✅ Real, 0.7 GB | DVC-tracked (2026-01-20) |
| `checkpoints/cnn/best_model.pth` | ✅ Real, 13.5 MB | Loads cleanly. Contains: epoch 17, best_val_acc **88.81%**, full history, model_config, optimizer/scheduler state |
| `logs/train_overnight.log` | ✅ Real | CNN1D on **CPU**, 100 epochs planned, **log ends at epoch 30** (~7.5 min/epoch); only model ever trained |
| `results/`, `experiments/` outputs | ❌ Empty | No evaluation, benchmark, XAI, ablation, or comparison output has ever been produced |
| ONNX / quantized models | ❌ None exist | despite "deployment-ready" claims |
| `deliverables/HANDOVER_PACKAGE/` | ⚠️ Nearly empty | 1 K8s manifest copy + 1 smoke-test file. `model_metadata.json` was deleted in March for being fake |

### 2.2 Test suite (run today, venv Python 3.14.0, torch 2.9.1+cpu)

```
267 tests collected; 45 failed, 220 passed, 13 skipped (5 min)
```

Excluded from the run, and themselves findings:
- `tests/utilities/test_training_imports.py` — calls `sys.exit()` at module level, **crashes pytest collection** (script masquerading as a test).
- `tests/load_tests.py`, `tests/stress_tests.py` — load/stress, not unit tests.

**Failure clusters (root causes verified):**

| Cluster | Count | Root cause |
|---|---|---|
| `test_pinn.py` HybridPINN forward/gradients | 7+ | **Real bug**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x75 and 576x576→256)` — the physics branch of the project's namesake model cannot do a forward pass |
| `test_data_generation.py` | ~20 | **Stale tests**: written against pre-refactor API (`FaultConfig.enable_sain` etc. no longer exist after the `signal_generation/` package split). Generation itself works (proven by real runs) — the tests were never updated |
| `test_dashboard_sanity.py` | 2 | **Real bug**: `app.py:69` uses `REFRESH_INTERVAL_MS` before importing it at `app.py:96` → dashboard cannot import/boot at all |
| `test_models.py` HybridPINN | 2 | Same PINN shape bug |
| ONNX/deployment/quantization tests | ~6 | Runtime/TypeErrors + missing `onnxruntime` in venv |
| integration `test_comprehensive.py` | 3 | Import errors in cross-validation / leakage-check / statistical utilities |
| `test_models.py::test_serialization` | 1 | Windows `PermissionError` (tempfile handling) |

### 2.3 Environment

- venv: Python **3.14.0**, torch **2.9.1+cpu** (no CUDA on this machine). README claims "Python 3.8+ (3.10 recommended)" — untested spread.
- `.coverage` file is from January; no current coverage data.

---

## 3. What Genuinely Works (verified)

1. **Synthetic signal generation** (`data/signal_generation/`): physics-motivated fault models
   for 11 classes (misalignment harmonics, imbalance with speed-dependence, Sommerfeld-scaled
   lubrication stick-slip, cavitation bursts, oil whirl, 3 mixed modes). Coefficients are
   empirical rather than first-principles, which is acceptable and is honestly documented in
   `data/PHYSICS_MODEL_GUIDE.md` ("signal fidelity ... has not been validated"). Validation
   (`validate_signal()`) is wired into generation (warn-only).
2. **HDF5 data pipeline**: `BearingFaultDataset.from_hdf5()` (data/dataset.py) — proven by the
   March run; clean splits, metadata, seeds.
3. **CNN training loop**: `scripts/train_overnight.py` → `CNNTrainer`/`BaseTrainer` — proven;
   checkpointing works, checkpoint loads, history is complete.
4. **Model factory breadth (instantiation level)**: 81 registry entries ≈ 34 distinct
   architectures + ~47 aliases/size-variants. Spot checks and the import-smoke tests show most
   families instantiate. **Exception**: HybridPINN forward pass is broken (above).
5. **Parts of the dashboard backend**: services for API keys (bcrypt), webhooks, 2FA/TOTP,
   notifications, and the SQLAlchemy schema + 10 SQL migrations are substantive code —
   but unreachable end-to-end today because the app cannot boot (§4).
6. **Utilities**: `utils/reproducibility.py`, checkpoint manager, logging, constants — actively
   used by the proven pipeline.

---

## 4. What Is Broken (verified, with locations)

| # | Problem | Location | Severity |
|---|---|---|---|
| 1 | HybridPINN physics-branch shape mismatch → forward pass fails | `packages/core/models/pinn/hybrid_pinn.py` (75-dim feature input vs 576-dim expected layer) | **P0** — flagship model unusable |
| 2 | Dashboard cannot import: `REFRESH_INTERVAL_MS` used at line 69, imported at line 96; also shadow-import risk between root `utils/constants.py` and `packages/dashboard/utils/constants.py` | `packages/dashboard/app.py:69,96` | **P0** — entire dashboard dead |
| 3 | `docker-compose up` fails: missing `nginx.conf` + `ssl/` (referenced at docker-compose.yml:48-49); `MODEL_PATH=/app/checkpoints/best_model.pth` but real path is `checkpoints/cnn/best_model.pth`; dashboard Dockerfile COPYs nonexistent `dash_app/` | `docker-compose.yml`, `packages/dashboard/Dockerfile:13-14`, `packages/deployment/api/config.py:32` | **P0** for deployment story |
| 4 | `mkdocs build` fails: no root `docs/` dir (docs actually live in `config/docs/`); README links to `docs/...` are broken; `site/` is a stale pre-built copy | `mkdocs.yml`, `README.md:98,133-141` | P1 |
| 5 | ~20 data-generation tests stale after `signal_generation/` refactor | `tests/test_data_generation.py` | P1 |
| 6 | `tests/utilities/test_training_imports.py` kills pytest collection (`sys.exit` at module scope) | that file:120 | P1 |
| 7 | `experiments/cnn_experiment.py` imports nonexistent top-level `models/`, `training/`, `evaluation/` packages — broken since the `packages/` restructure | `experiments/cnn_experiment.py:26-31` | P1 |
| 8 | ONNX static quantization silently returns FP32 model instead of failing | `packages/deployment/optimization/onnx_export.py:475-476` | P1 |
| 9 | `benchmark_inference()` skips the PyTorch backend entirely | `packages/deployment/optimization/inference.py:474-478` | P2 |
| 10 | CI workflows: deprecated action versions (`checkout@v3`, `setup-python@v4`, docker actions v2); docs build step guaranteed to fail (no `docs/`); `tests/benchmarks/benchmark_suite.py` referenced but absent | `.github/workflows/ci.yml`, `deploy.yml` | P2 |
| 11 | K8s/Helm: placeholder domains (`api.lstm-pfd.example.com`), empty required secrets, probe port 8050 on an API (8000) deployment, missing `_helpers.tpl`, undefined PVC/Secret objects | `deploy/helm/lstm-pfd/values.yaml:33,89,106,114`, `deploy/kubernetes/deployment.yaml` | P2 (aspirational anyway) |

---

## 5. What Is Facade / Aspirational (code exists, never proven)

- **33 of 34 model architectures**: never trained. The "35-model Colab pipeline"
  (`scripts/colab/`, `notebooks/colab_train_all_models.ipynb`) is internally consistent
  (registry keys match) but has **never been executed** — zero checkpoints from it.
- **The entire evaluation suite** (~20 modules, ~7.2K lines in `packages/core/evaluation/`):
  fully written, **never run against a real trained model** (results/ is empty).
- **XAI** (`packages/core/explainability/`, 8 method families): substantive implementations,
  never exercised; the dashboard XAI page imports a nonexistent service and silently
  swallows the ImportError (`xai_callbacks.py`).
- **Dashboard feature surface**: significant parts are UI-only — feature engineering callbacks
  empty, evaluation metrics not computed, NAS results placeholder, testing dashboard not wired,
  system health gauge hardcoded "100% Online" (`home_callbacks.py:136-145`), **no login page
  exists** despite full auth service code.
- **Benchmarks**: `benchmarks/industrial_validation.py:67-68` contains
  `real_test_accuracy = 0.89  # Placeholder` and a synthetic noise-decay formula — outputs
  look like results but are invented.
- **Research scripts**: `scripts/research/failure_analysis.py:290-305` generates its
  "misclassifications" from `np.random`. `ablation_study.py` runs but only on its own toy
  synthetic data with a local model — conclusions wouldn't transfer. PINN comparison/ablation
  scripts unverified and depend on the broken HybridPINN.
- **SaaS layer** (implementation_plan Phase 9: multi-tenancy, billing, GDPR, Terraform): pure
  aspiration; nothing implemented.

---

## 6. Fabrications (claims of results that never existed)

| Location | Fabrication | Disposition |
|---|---|---|
| `config/docs/paper/main.tex` | "**98.1% accuracy** while providing faithful explanations **validated by domain experts**"; ablation deltas ("removing physics loss decreases accuracy by 1.6%"); per-class results ("89.2% on this class") — **no experiment has ever produced these numbers** | Quarantine as template; strip all numbers |
| `config/docs/reports/Final_Report.pdf` (2.6 MB) | Full "final report" presumably built on the same fake numbers | Archive |
| `reproducibility/README.md:62-67` | "Key Results: CNN Baseline 94.2% / PINN (ours) **98.1%**" presented as "our research results" | Replace with [PENDING] or delete table |
| `config/docs/idb_reports/IDB_4_1_DATABASE_ANALYSIS.md:129` | "database layer is production-ready" — never load-tested, dashboard never booted | Correct |
| `benchmarks/industrial_validation.py` | Hardcoded "real_test_accuracy = 0.89" placeholder masquerading as industrial validation | Label or delete |
| Historical note | Git history shows a prior cleanup wave (e.g., `model_metadata.json` deleted as fake; README/CHANGELOG converted to `[PENDING]`) — i.e., fabrication was broader before; **remnants above are what's left** | — |

**Honest docs worth noting**: current `README.md` and `CHANGELOG.md` use `[PENDING]` correctly;
`data/PHYSICS_MODEL_GUIDE.md` explicitly disclaims validation; research planning docs in
`config/docs/research/` mark all results `[PENDING]`; `DOCUMENTATION_STANDARDS.md` even codifies
"never state results unless verified."

---

## 7. Structural Problems (complexity to remove)

1. **Dataset class sprawl**: 9–10 dataset classes across `data/`; only `BearingFaultDataset`
   (+ transforms) is on the proven path. `RawSignalDataset`, `CachedRawSignalDataset`,
   streaming variants, and 3 TFR datasets are orphaned or research-only.
2. **Shim archaeology**: ~6 backward-compat re-export shims (`cnn_1d.py`, `resnet_1d.py`,
   `hybrid_pinn.py`, `legacy_ensemble.py`, `losses.py`, `cnn_dataloader.py`, `cnn_schedulers.py`,
   `transformer_schedulers.py`) — fine individually, but they exist to preserve import paths
   nobody external uses.
3. **Registry alias inflation**: 81 registry entries ≈ 34 real architectures; docs cite "65+"
   as if they were distinct models.
4. **Three doc roots**: `config/docs/` (the real hub, oddly nested under config), root-level
   `*.md`, and stale prebuilt `site/`. README links assume a `docs/` root that doesn't exist.
5. **Two `utils` packages** (root `utils/` and `packages/dashboard/utils/`) with name-shadowing
   import risk — implicated in the dashboard boot bug.
6. **Process debris**: ~25-30 agent-generated audit/IDB markdown files (incl. `compiled/`),
   `tmp_gitlog.txt` committed at repo root, empty placeholder dirs (`results/`, `audit_reports/`
   until now), 19 remote branches (15+ stale).
7. **Dead orchestration**: `experiments/cnn_experiment.py` (broken imports),
   `integration/unified_pipeline.py` (phase stubs), `data/data_validator.py` (never called).

---

## 8. Planned vs. Implemented (from `implementation_plan.md`, cross-checked)

| Phase | Plan says | Audit verdict |
|---|---|---|
| 0 Git hygiene | ✅ done | Confirmed (but 19 remote branches remain) |
| 1 P0 security/stability fixes | ✅ done | Mostly confirmed in code (secrets→env, sys.path hacks removed, HDF5 leaks fixed) |
| 2 Architectural consolidation | "~85% done" | Largely real (BaseTrainer, losses/schedulers consolidation, shims) — but it introduced the stale data-gen tests and likely the dashboard import bug; 2.4, 2.23-2.27, 2.29 open |
| 3 End-to-end experimentation | unchecked; "smoke-tested" | **1 of ~19 items actually achieved** (one dataset + one partial training). No benchmarks, no ONNX, no checkpoints/phase dirs, README still [PENDING] |
| 4 Research & publication | unchecked | Nothing run. Paper contains fabricated results (§6) |
| 5 Dashboard verification | unchecked | Dashboard currently cannot boot |
| 6 Deployment pipeline | partially checked | Spot fixes done; compose/K8s still non-functional (§4) |
| 7 Test coverage gate | unchecked | 45 failures today; coverage unknown |
| 8 Docs finalization | unchecked | mkdocs broken; doc sprawl |
| 9 SaaS productionization | unchecked | Pure aspiration — recommend cutting entirely |

---

## 9. Reliability Notes on Sub-Audits

Five audit agents were used; their reports were cross-checked and three corrections matter:

- **Docs agent** concluded "documentation is 98% trustworthy" — **wrong**: it only scanned
  `.md` files and missed the LaTeX paper, the PDF report, and `reproducibility/README.md`
  fabrications (§6).
- **Dashboard agent** scored the dashboard "70-75% functional" from code reading — **misleading**:
  the app cannot even import (§4.2). Its service-layer findings remain useful.
- **Deployment agent** claimed `tests/unit/`, `tests/integration/`, and `packages/core/` don't
  exist — **wrong** (all three exist; tests ran from them today). Its docker-compose and mkdocs
  findings were independently verified and are correct.
- **Core-ML agent**'s "zero critical bugs, 89% usable" preceded the test run that shows the
  PINN forward pass is broken. Its inventory (34 architectures, no stub bodies) stands.

Lesson encoded for the next phase: **static code reading systematically over-estimates health;
only execution evidence counts.**

---

## 10. Recommended Convergence Direction (input to the next planning step)

The project's honest, defensible core is: **physics-based synthetic vibration data generator +
a focused set of fault-classification models + rigorous evaluation + XAI, runnable end-to-end
as scripts/CLI**. Everything else should serve that or go.

**Proposed thrusts (to be turned into a phased plan):**

1. **Stabilize ground floor** — fix the 4 real code bugs (PINN forward, dashboard import,
   pytest-collection killer, stale data-gen tests); get the suite green or explicitly skipped
   with reasons; pin the environment story (one Python version).
2. **Shrink to the proven spine** — one dataset class + transforms; curate the model zoo to
   ~6-10 architectures actually worth benchmarking (CNN1D, ResNet variants, 1-2 transformers,
   fixed HybridPINN, 1 ensemble); delete or archive orphaned dataset classes, broken
   experiments/, placeholder benchmarks, fake-output research scripts.
3. **Produce real results** — finish the interrupted training matrix on the curated zoo
   (Colab pipeline already exists and is consistent); run the evaluation suite once for real;
   fill README/CHANGELOG `[PENDING]` with actual numbers; only then revisit the paper, stripped
   of all fabricated claims.
4. **Make the dashboard honest or optional** — either fix boot + cut facade pages (XAI, NAS,
   testing, feature-eng tabs) down to what's wired, or demote the dashboard to "experimental"
   and let the CLI be the product.
5. **Deployment to match reality** — one working `docker-compose` (API + model), delete or
   clearly mark Helm/K8s as templates; fix CI to run the real test suite only.
6. **Docs consolidation** — single `docs/` root; archive IDB/process reports and the fabricated
   paper/PDF into `archive/`; keep the ~15 genuinely valuable references (architecture, physics
   guide, training/XAI/metrics guides, deployment guide, standards).

**Suggested cut entirely**: Phase 9 SaaS (multi-tenancy/billing/GDPR), NAS dashboard,
testing-dashboard, webhooks/2FA surface (until there's a login page), K8s/Helm (keep as
reference templates at most).

---

*Audit performed by Claude Code on branch `audit/project-state-2026-06`. Evidence current as of
2026-06-11. Test run: venv Python 3.14.0, torch 2.9.1+cpu, 267 tests collected.*
