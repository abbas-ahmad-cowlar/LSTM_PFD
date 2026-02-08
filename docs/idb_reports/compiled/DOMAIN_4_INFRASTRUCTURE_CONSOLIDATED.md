# Domain 4: Infrastructure — Consolidated Analysis

> **Compilation Date:** 2026-01-24  
> **Source IDBs:** 4.1 (Database), 4.2 (Deployment), 4.3 (Testing), 4.4 (Configuration)  
> **Total Files Analyzed:** 8 reports (4 Analysis + 4 Best Practices)

---

## 1. Domain Overview

- **Purpose:** Provide foundational infrastructure for the LSTM_PFD project including database persistence, deployment pipelines, testing frameworks, and configuration management.
- **Sub-blocks:** Database (4.1), Deployment (4.2), Testing (4.3), Configuration (4.4)
- **Overall Independence Score:** **(8 + 8 + 7 + 10) / 4 = 8.25/10**
- **Key Interfaces:**
  - Database: SQLAlchemy ORM models, session management, migrations
  - Deployment: Docker/K8s orchestration, ONNX/quantization APIs, health endpoints
  - Testing: pytest fixtures, stress/load test frameworks
  - Configuration: Dataclass configs, YAML serialization, JSON Schema validation

---

## 2. Current State Summary

### What's Implemented

| IDB | Component     | Key Implementations                                                                                                                            |
| --- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| 4.1 | Database      | 28 SQLAlchemy models, 10 migrations, connection pooling (30+30), slow query logging                                                            |
| 4.2 | Deployment    | Multi-stage Dockerfiles, docker-compose (7 services), Helm charts, ONNX export, quantization (Dynamic/Static/FP16/QAT), K8s manifests with HPA |
| 4.3 | Testing       | 14 test files, 12 fixtures, 5 custom markers, ~4,500 LOC, stress/load/benchmark suites                                                         |
| 4.4 | Configuration | 6 config files, 20 dataclass config classes, JSON Schema validation, YAML serialization                                                        |

### What's Working Well

1. **Database Layer (4.1):**
   - Proper connection pooling with `pool_pre_ping` health checks
   - Consistent `BaseModel` inheritance with auto-timestamps
   - PostgreSQL-specific optimizations with SQLite development fallbacks
   - Idempotent migrations with verification scripts
   - Slow query logging (>1s threshold)

2. **Deployment Pipeline (4.2):**
   - Multi-stage Docker builds with non-root user security
   - Comprehensive health checks (Dockerfile + K8s probes)
   - Complete ONNX export + validation pipeline
   - Full quantization suite (Dynamic/Static/FP16/QAT)
   - SLO-based Prometheus alerting (99.5% availability)
   - Clean `BaseInferenceEngine` ABC abstraction

3. **Testing Infrastructure (4.3):**
   - Comprehensive stress/load testing with memory leak detection
   - Physics-aware testing (PINN gradient validation)
   - Fault consistency tests ensuring 11-class alignment
   - Device-agnostic fixtures (CUDA/CPU)
   - Constants usage over magic numbers

4. **Configuration System (4.4):**
   - Pure data module (10/10 independence)
   - Hierarchical dataclass composition
   - Factory pattern for model selection
   - Comprehensive constants file (629 lines)
   - PEP 561 typed package marker

### What's Problematic

| IDB | Issue Category | Description                                                                         |
| --- | -------------- | ----------------------------------------------------------------------------------- |
| 4.1 | Deprecations   | `declarative_base()` and `datetime.utcnow` deprecated                               |
| 4.2 | **Security**   | Hardcoded secrets in docker-compose, CORS allows all origins, missing rate limiting |
| 4.3 | Coverage Gaps  | Zero tests for Dashboard Callbacks (IDB 2.3) and Celery Async Tasks (IDB 2.4)       |
| 4.4 | Env Support    | Zero environment variable support—secrets must be YAML-loaded                       |

---

## 3. Critical Issues Inventory

### P0 Issues (Critical - Production Blockers)

| IDB | Issue                                                                   | Impact                                               | Effort | Dependencies            |
| --- | ----------------------------------------------------------------------- | ---------------------------------------------------- | ------ | ----------------------- |
| 4.2 | Hardcoded secrets in docker-compose (`POSTGRES_PASSWORD`, `SECRET_KEY`) | Security vulnerability in production                 | 2h     | K8s Secrets/Vault setup |
| 4.2 | CORS allows all origins (`cors_origins: ["*"]`)                         | API security risk                                    | 1h     | None                    |
| 4.2 | PostgreSQL password in Helm values (`postgresPassword: "changeme"`)     | Security vulnerability                               | 1h     | Secret management       |
| 4.3 | Missing Dashboard Callback tests                                        | Bugs in tightly-coupled callbacks go undetected      | 4-6h   | Callback refactoring    |
| 4.3 | Missing Celery Async Task tests                                         | `run_hpo_task`, `process_batch_predictions` untested | 4h     | Celery test fixtures    |
| 4.4 | No environment variable support                                         | Secrets/URIs must be hardcoded or YAML-loaded        | 2h     | None                    |

### P1 Issues (High Priority)

| IDB | Issue                                                        | Impact                                                              | Effort | Dependencies        |
| --- | ------------------------------------------------------------ | ------------------------------------------------------------------- | ------ | ------------------- |
| 4.1 | Deprecated `declarative_base()`                              | SQLAlchemy 2.0 deprecation warning                                  | 2h     | None                |
| 4.1 | Missing `User` relationship back_populates                   | Incomplete ORM relationships                                        | 30m    | None                |
| 4.1 | Unused constants imports in models                           | Code smell, slight memory waste                                     | 1h     | None                |
| 4.2 | Dashboard Dockerfile lacks non-root user/health check        | Security gap                                                        | 1h     | None                |
| 4.2 | Static quantization raises NotImplementedError               | ONNX static quantization broken                                     | 4h     | ONNX runtime update |
| 4.2 | Missing GPU resource requests in K8s                         | No `nvidia.com/gpu` requests                                        | 1h     | None                |
| 4.3 | Flaky stress test thresholds                                 | Memory leak tests environment-sensitive                             | 2h     | None                |
| 4.3 | Slow tests not consistently marked                           | Many >1s tests lack `@pytest.mark.slow`                             | 2h     | None                |
| 4.3 | Fixture scope mismatch (`simple_cnn_model` function-scoped)  | Unnecessary model recreation                                        | 1h     | None                |
| 4.4 | Incomplete schema validation (~40% coverage)                 | 60% of config fields unvalidated                                    | 4h     | None                |
| 4.4 | `ConfigValidator` utility never used                         | Dead code                                                           | 30m    | None                |
| 4.4 | No cross-field validation                                    | e.g., `conv_channels` and `kernel_sizes` length mismatch undetected | 2h     | None                |
| 4.4 | Hardcoded paths in defaults                                  | `checkpoint_dir='checkpoints'` not configurable                     | 1h     | None                |
| 4.4 | Deprecated `from_matlab_struct()` raises NotImplementedError | Dead code                                                           | 30m    | None                |

### P2 Issues (Medium Priority)

| IDB | Issue                                         | Impact                          | Effort | Dependencies        |
| --- | --------------------------------------------- | ------------------------------- | ------ | ------------------- |
| 4.1 | `datetime.utcnow` deprecated                  | Python 3.12+ compatibility      | 2h     | None                |
| 4.1 | No migration version tracking                 | Can't detect applied migrations | 4h     | None                |
| 4.1 | CRLF line endings in `permissions.py`         | Inconsistent with LF files      | 15m    | None                |
| 4.2 | Redis auth disabled (`auth.enabled: false`)   | Security gap                    | 30m    | None                |
| 4.2 | No network policies in K8s                    | Missing pod-to-pod restrictions | 2h     | None                |
| 4.2 | Missing API rate limiting                     | No request throttling           | 4h     | SlowAPI integration |
| 4.3 | Duplicate SimpleCNN definitions (3 locations) | Maintenance burden              | 1h     | None                |
| 4.3 | Mixed test frameworks (unittest + pytest)     | Inconsistency                   | 2h     | None                |
| 4.3 | Utilities directory uses script-style tests   | Won't be discovered by pytest   | 2h     | None                |
| 4.4 | Mixed precision disabled by default           | Performance left on table       | 30m    | None                |
| 4.4 | Undocumented schema enum values               | Poor discoverability            | 2h     | None                |
| 4.4 | Tuple type hints incomplete                   | Should be `Tuple[float, float]` | 1h     | None                |

---

## 4. Technical Debt Registry

### Quick Wins (< 1 hour)

| IDB | Task                                                | Benefit             |
| --- | --------------------------------------------------- | ------------------- |
| 4.1 | Remove unused `utils.constants` imports from models | Code hygiene        |
| 4.1 | Add missing `User` relationship back_populates      | ORM consistency     |
| 4.1 | Fix CRLF line endings in `permissions.py`           | File consistency    |
| 4.2 | Add non-root user to dashboard Dockerfile           | Security            |
| 4.3 | Add `@pytest.mark.slow` to unmarked slow tests      | Test categorization |
| 4.4 | Remove dead `ConfigValidator` class OR integrate it | Code hygiene        |
| 4.4 | Remove deprecated `from_matlab_struct()` method     | Code hygiene        |
| 4.4 | Enable mixed precision by default                   | Performance         |

### Medium Tasks (1-4 hours)

| IDB | Task                                                        | Benefit                |
| --- | ----------------------------------------------------------- | ---------------------- |
| 4.1 | Update to SQLAlchemy 2.0 `DeclarativeBase`                  | Future-proofing        |
| 4.1 | Replace `datetime.utcnow` with `datetime.now(timezone.utc)` | Python 3.12+ compat    |
| 4.2 | Move secrets to K8s Secrets/environment variables           | Security               |
| 4.2 | Configure environment-specific CORS origins                 | Security               |
| 4.2 | Add health check to dashboard Dockerfile                    | Reliability            |
| 4.3 | Consolidate SimpleCNN to single `tests/models/` location    | Maintainability        |
| 4.3 | Migrate unittest tests to pytest style                      | Consistency            |
| 4.3 | Convert `utilities/` scripts to pytest format               | Test discovery         |
| 4.3 | Optimize fixture scopes (module scope where safe)           | Test performance       |
| 4.4 | Add environment variable support to config classes          | Deployment flexibility |
| 4.4 | Complete JSON Schema validation for all fields              | Data integrity         |
| 4.4 | Add cross-field validation in `__post_init__`               | Config safety          |

### Large Refactors (1+ days)

| IDB | Task                                       | Benefit                     |
| --- | ------------------------------------------ | --------------------------- |
| 4.1 | Implement Alembic for migration management | Auto-migrations, versioning |
| 4.1 | Add `schema_migrations` tracking table     | Operational safety          |
| 4.2 | Add rate limiting middleware (SlowAPI)     | API protection              |
| 4.2 | Implement K8s network policies             | Security hardening          |
| 4.2 | Add GPU resource requests to K8s           | GPU scheduling              |
| 4.3 | Create comprehensive callback test suite   | Coverage for IDB 2.3        |
| 4.3 | Create Celery async task test suite        | Coverage for IDB 2.4        |
| 4.3 | Reorganize test directory structure        | Logical organization        |

---

## 5. "If We Could Rewrite" Insights

### Cross-Sub-Block Themes

1. **Security Gaps:** Secrets management is weak across Deployment (hardcoded) and Config (no env var support)
2. **Deprecation Debt:** Both Database and Config have Python/SQLAlchemy deprecation warnings
3. **Incomplete Validation:** Database schema is well-normalized, but Config validation covers only ~40% of fields
4. **Test Coverage Gaps:** Critical Dashboard components (Callbacks, Async Tasks) have zero test coverage

### Fundamental Architectural Changes

1. **Unified Secret Management:** Implement K8s Secrets + environment variable loading across Config and Deployment
2. **Alembic Migration System:** Replace manual SQL migrations with Alembic for auto-generation and version tracking
3. **Complete Schema Validation:** Extend JSON Schema coverage to 100% of config fields with cross-field validation
4. **Comprehensive Test Suite:** Add callback and async task tests to eliminate P0 coverage gaps
5. **SQLAlchemy 2.0 Migration:** Update ORM patterns to modern SQLAlchemy 2.0 style

### Patterns to Preserve

| Pattern                    | Location       | Reason                             |
| -------------------------- | -------------- | ---------------------------------- |
| `BaseModel` abstraction    | Database 4.1   | Consistent timestamps, `to_dict()` |
| Connection pooling config  | Database 4.1   | Production-ready (30+30 pool)      |
| Multi-stage Docker builds  | Deployment 4.2 | Optimal image size                 |
| `BaseInferenceEngine` ABC  | Deployment 4.2 | Pluggable backends                 |
| Device-agnostic fixtures   | Testing 4.3    | CUDA/CPU flexibility               |
| Dataclass config hierarchy | Config 4.4     | Type-safe, IDE-friendly            |
| Constants centralization   | Config 4.4     | Single source of truth (629 lines) |
| JSON Schema pattern        | Config 4.4     | Standard validation approach       |
| YAML serialization         | Config 4.4     | Human-readable configs             |
| Factory pattern for models | Config 4.4     | Clean model selection              |

### Patterns to Eliminate

| Anti-Pattern           | Location                  | Replacement                         |
| ---------------------- | ------------------------- | ----------------------------------- |
| Hardcoded secrets      | Deployment docker-compose | K8s Secrets/Vault                   |
| CORS wildcard `["*"]`  | Deployment API config     | Environment-specific origins        |
| `declarative_base()`   | Database base.py          | `DeclarativeBase` class             |
| `datetime.utcnow`      | Database models           | `datetime.now(timezone.utc)`        |
| Duplicate SimpleCNN    | Testing (3 locations)     | Single `tests/models/simple_cnn.py` |
| Script-style tests     | Testing utilities/        | pytest format                       |
| Dead `ConfigValidator` | Config base_config.py     | Use or remove                       |
| No env var support     | Config all files          | `os.getenv()` with fallbacks        |

---

## 6. Best Practices Observed

### Code Conventions

- **Naming:**
  - Models: PascalCase (`APIKey`, `TrainingRun`)
  - Tables: snake_case (`api_keys`, `training_runs`)
  - Configs: `<Component>Config` pattern
- **Imports:**
  - Constants from `utils/constants.py` (NUM_CLASSES, SIGNAL_LENGTH)
  - Relative imports within packages
- **Docstrings:**
  - Class-level with example usage
  - Module-level with purpose and author
- **Type Hints:**
  - Dataclass fields fully typed
  - `py.typed` marker for PEP 561

### Design Patterns Worth Preserving

1. **BaseModel Abstraction** — Consistent `id`, `created_at`, `updated_at`, `to_dict()`
2. **Connection Pooling** — `pool_pre_ping`, sized for 26+ Dash callbacks
3. **Slow Query Logging** — Event listener for >1s queries
4. **Multi-Stage Docker** — Build/runtime separation
5. **BaseInferenceEngine ABC** — Pluggable ONNX/PyTorch/TensorRT backends
6. **Dataclass Configs** — Type hints, immutable-friendly, IDE autocomplete
7. **Factory Pattern** — `ModelConfig.get_active_config()` for model selection
8. **Fixture Cleanup** — `yield` pattern with `shutil.rmtree()` cleanup
9. **Session-Scoped Device** — Single CUDA/CPU detection per test session

### Testing Patterns

| Pattern                    | Implementation                                       | File            |
| -------------------------- | ---------------------------------------------------- | --------------- |
| Device-agnostic            | `@pytest.fixture(scope="session") def device()`      | conftest.py     |
| Temp cleanup               | `yield temp_dir; shutil.rmtree(temp_dir)`            | conftest.py     |
| Constants usage            | `from utils.constants import NUM_CLASSES`            | All test files  |
| Skip conditions            | `@pytest.mark.skipif(not torch.cuda.is_available())` | stress_tests.py |
| Memory leak detection      | `tracemalloc.start()` + threshold assertions         | stress_tests.py |
| Gradient flow verification | `assert param.grad is not None`                      | test_pinn.py    |

### Interface Contracts

**Database Session API:**

```python
with get_db_session() as session:
    session.query(Model).filter_by(...).first()
# Auto-commit on success, rollback on exception
```

**ONNX Export API:**

```python
export_to_onnx(model, dummy_input, 'model.onnx')
validate_onnx_export('model.onnx', model, test_input)
optimize_onnx_model('model.onnx', 'model_opt.onnx', 'extended')
```

**Configuration API:**

```python
config = TrainingConfig.from_yaml(Path('config.yaml'))
config.validate()  # JSON Schema validation
config.to_dict()   # For MLflow logging
```

---

## 7. Cross-Domain Dependencies

### Inbound Dependencies

| From Domain         | Component | What Infrastructure Provides                            |
| ------------------- | --------- | ------------------------------------------------------- |
| **Core ML (1.x)**   | Models    | `ModelConfig`, `TrainingConfig` for architecture params |
| **Core ML (1.x)**   | Training  | `OptimizerConfig`, `SchedulerConfig`, checkpointing     |
| **Dashboard (2.x)** | Services  | Database models, session management                     |
| **Dashboard (2.x)** | Callbacks | Database queries, async task dispatch                   |
| **Data (3.x)**      | Signals   | `DataConfig`, `SignalConfig` for generation             |
| **Research (5.x)**  | Scripts   | `ExperimentConfig`, MLflow integration                  |

### Outbound Dependencies

| To Domain    | What Infrastructure Consumes                                |
| ------------ | ----------------------------------------------------------- |
| **Utils**    | `constants.py` (NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE)  |
| **External** | `jsonschema`, `pyyaml`, `sqlalchemy`, `onnx`, `onnxruntime` |

### Integration Risks

| Risk                                        | Likelihood | Impact | Mitigation                                  |
| ------------------------------------------- | ---------- | ------ | ------------------------------------------- |
| Schema changes break Dashboard queries      | Medium     | High   | Add migration version tracking, tests       |
| Config validation rejects valid experiments | Low        | High   | Complete schema coverage, integration tests |
| Docker image breaks on dependency update    | Medium     | Medium | Pin versions, CI/CD testing                 |
| Test suite flakiness in CI                  | High       | Medium | Fix memory thresholds, add retries          |

---

## 8. Domain-Specific Recommendations

### Top 3 Quick Wins

1. **[4.2] Move secrets to environment variables** (2h)
   - Replace hardcoded `POSTGRES_PASSWORD` and `SECRET_KEY` in docker-compose
   - Add env var loading to Helm values

2. **[4.4] Add environment variable support to configs** (2h)
   - Implement `os.getenv()` with fallbacks in `MLflowConfig`, `ExperimentConfig`
   - Pattern: `tracking_uri: str = field(default_factory=lambda: os.getenv('MLFLOW_TRACKING_URI', './mlruns'))`

3. **[4.3] Mark all slow tests with `@pytest.mark.slow`** (1h)
   - Audit `test_data_generation.py`, `integration/test_comprehensive.py`, `load_tests.py`
   - Enable fast CI with `pytest -m "not slow"`

### Top 3 Strategic Improvements

1. **[4.3] Create Callback + Async Task Test Suites** (1-2 days)
   - Priority: P0 coverage gaps for Dashboard domain
   - Create `tests/unit/dashboard/test_callbacks.py` (15-20 tests)
   - Create `tests/unit/dashboard/test_tasks.py` with Celery fixtures (10-15 tests)

2. **[4.1] Implement Alembic Migration System** (2-3 days)
   - Replace manual SQL migrations with auto-generation
   - Add `schema_migrations` version tracking table
   - Enable multi-branch development safely

3. **[4.4] Complete Configuration Validation** (1 day)
   - Extend JSON Schema coverage from ~40% to 100%
   - Add `__post_init__` cross-field validation
   - Integrate or remove `ConfigValidator` dead code

### Team Coordination Requirements

| Change                      | Affected Teams      | Coordination Need              |
| --------------------------- | ------------------- | ------------------------------ |
| Database schema changes     | Dashboard, Research | Migration review, backup       |
| Docker image updates        | DevOps, CI/CD       | Staging deployment first       |
| Config schema changes       | All domains         | YAML compatibility testing     |
| Test infrastructure changes | All domains         | Ensure no broken fixtures      |
| Secret management migration | DevOps              | K8s Secrets/Vault provisioning |

---

## Appendix: File Inventory

### IDB 4.1 — Database

| Path                                        | Lines    | Purpose                             |
| ------------------------------------------- | -------- | ----------------------------------- |
| `packages/dashboard/database/connection.py` | ~100     | Engine, session management, pooling |
| `packages/dashboard/database/base.py`       | ~50      | `Base`, `BaseModel` classes         |
| `packages/dashboard/models/`                | 26 files | ORM models (28 total)               |
| `packages/dashboard/database/migrations/`   | 10 files | SQL migrations                      |

### IDB 4.2 — Deployment

| Path                                | Lines  | Purpose                              |
| ----------------------------------- | ------ | ------------------------------------ |
| `Dockerfile`                        | 64     | Multi-stage API container            |
| `docker-compose.yml`                | 197    | Full-stack orchestration             |
| `packages/dashboard/Dockerfile`     | 26     | Dashboard container                  |
| `packages/deployment/optimization/` | ~1,500 | ONNX export, quantization, inference |
| `deploy/kubernetes/`                | ~150   | K8s manifests                        |
| `deploy/helm/lstm-pfd/`             | ~300   | Helm chart                           |
| `deploy/monitoring/`                | ~200   | Prometheus/Grafana                   |

### IDB 4.3 — Testing

| Path                    | Lines | Purpose                   |
| ----------------------- | ----- | ------------------------- |
| `tests/conftest.py`     | 164   | Shared fixtures           |
| `tests/unit/`           | ~700  | Unit tests                |
| `tests/integration/`    | ~650  | Pipeline tests            |
| `tests/benchmarks/`     | ~400  | Performance benchmarks    |
| `tests/stress_tests.py` | 494   | Memory/concurrency stress |
| `tests/load_tests.py`   | 980   | Load testing              |

### IDB 4.4 — Configuration

| Path                          | Lines | Purpose                     |
| ----------------------------- | ----- | --------------------------- |
| `config/base_config.py`       | 193   | Abstract base, validation   |
| `config/data_config.py`       | 365   | 9 data config classes       |
| `config/model_config.py`      | 387   | 7 model config classes      |
| `config/training_config.py`   | 364   | 6 training config classes   |
| `config/experiment_config.py` | 208   | 3 experiment config classes |
| `config/__init__.py`          | 72    | Exports                     |

---

_Consolidated from IDB 4.1, 4.2, 4.3, 4.4 Analysis and Best Practices reports._
