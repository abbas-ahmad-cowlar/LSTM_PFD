# Architectural Recommendations — Strategic Insights

**Synthesis Date:** 2026-01-24  
**Source:** 5 Domain Consolidated Reports (IDBs 1.x–6.0)  
**Total Technical Debt Items Analyzed:** 100+ issues across 6 domains

---

## Executive Summary

### Top 5 Architectural Changes Recommended

1. **Unify Inheritance Hierarchies** — Create proper base classes for Trainers (5/8 orphan), Evaluators (3/5 orphan), and Datasets to eliminate 1,500+ lines of duplicated code
2. **Implement Centralized Secret Management** — Replace hardcoded secrets in docker-compose and eliminate zero environment variable support in config system
3. **Split God-Class Files** — Refactor `settings.py` (1,005 lines), `signal_generator.py` (935 lines), and `NotificationService` (713 lines) into focused modules
4. **Complete Test Coverage for Dashboard Layer** — Add missing tests for Callbacks (IDB 2.3) and Celery Async Tasks (IDB 2.4) which have zero coverage
5. **Replace `sys.path.insert()` Anti-Pattern** — Convert to proper Python packaging across Research Scripts and Integration Layer

### Patterns to Preserve

| Pattern                             | Domains                           | Evidence                                                                |
| ----------------------------------- | --------------------------------- | ----------------------------------------------------------------------- |
| **Factory Pattern with Registry**   | Core ML, Dashboard                | `model_factory.py`, `NotificationProviderFactory` — clean extensibility |
| **Context Manager for Resources**   | All                               | `with get_db_session() as session:` — guaranteed cleanup                |
| **Dataclass Configurations**        | Core ML, Infrastructure, Research | Type-safe, serializable, IDE-friendly                                   |
| **Thread-Local State**              | Data Engineering                  | `threading.local()` for HDF5 handles — multi-worker safety              |
| **Graceful Library Fallback**       | Core ML, Research                 | `HAS_TORCH`, optional SHAP — degrades without crashing                  |
| **Progress Tracking via Callbacks** | Dashboard                         | `self.update_state()` for Celery — real-time UI feedback                |
| **Config-Hash Cache Invalidation**  | Data Engineering                  | SHA-256 of config dict — auto-invalidates on changes                    |

### Patterns to Eliminate

| Anti-Pattern                        | Occurrences     | Domains            | Impact                                           |
| ----------------------------------- | --------------- | ------------------ | ------------------------------------------------ |
| `sys.path.append('/home/user/...')` | 12+ files       | 1.1, 1.2, 1.5, 6.0 | Fragile imports, breaks portability              |
| Bare `except Exception` / `except:` | 100+ instances  | 2.x, 3.x, 5.x      | Silent failures, impossible debugging            |
| Hardcoded magic numbers             | 50+ occurrences | All                | `102400`, `20480`, `5000` scattered everywhere   |
| Duplicate class implementations     | 15+ cases       | 1.x, 2.x, 3.x      | `FocalLoss`, `Compose`, `SimpleCNN` defined 2-3x |
| God-class files (>500 lines)        | 4 files         | 2.1, 2.2, 3.1      | Merge conflicts, cognitive overload              |
| Empty `__init__.py`                 | 5+ packages     | 1.2, 1.3           | No public API, awkward imports                   |

---

## Cross-Cutting Architectural Themes

### Theme 1: Inheritance Fragmentation Creates Massive Duplication

**Evidence:**

- **Core ML (Domain 1):** 5/8 trainers and 3/5 evaluators don't inherit from base classes
- **Data Engineering (Domain 3):** 12+ dataset classes with no common `BaseSignalDataset`
- **Dashboard (Domain 2):** 23 Celery tasks without `BaseTask` abstraction

**Root Causes:**

- Organic growth without architectural planning
- Copy-paste development for "quick" specialized implementations
- Missing abstract interface contracts before implementation

**Recommended Solution:**

```python
# Pattern: Plugin Architecture for Trainers
class UnifiedTrainer:
    def __init__(self, model, config, plugins=None):
        self.plugins = PluginManager([
            MixedPrecisionPlugin(),
            GradientClippingPlugin(),
            PhysicsLossPlugin(),  # Optional for PINN
        ])

# Pattern: Abstract Hook Points for Evaluators
class BaseEvaluator:
    def evaluate(self, dataloader) -> EvaluationResult:
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                self._process_batch(batch)  # Abstract
        return self._compute_metrics()      # Abstract
```

**Migration Path:**

1. **Phase 1 (1 week):** Create abstract base classes with minimal required methods
2. **Phase 2 (2 weeks):** Migrate existing implementations to inherit from bases
3. **Phase 3 (1 week):** Add integration tests to verify behavioral consistency
4. **Phase 4 (ongoing):** Enforce inheritance via code review policy

---

### Theme 2: Dashboard Coupling is the Highest Risk Area

**Evidence:**

- **IDB 2.3 (Callbacks):** Independence score of **4/10** — lowest in entire system
- Average Dashboard independence: **(7 + 5 + 4 + 6) / 4 = 5.5/10** ⚠️
- Services layer (2.2) tightly couples to Core ML and Database

**Root Causes:**

- Callbacks directly access DB bypassing service layer (5 modules)
- Fat callbacks with embedded business logic (8 modules)
- No callback → service contract tests

**Recommended Solution:**

```python
# Callback Decorator Pattern for Cross-Cutting Concerns
@authenticated
@with_error_boundary
@app.callback(...)
def protected_callback(data):
    return service.do_something(data)  # Thin callback

# ServiceResult Dataclass for Uniform Returns
@dataclass
class ServiceResult:
    success: bool
    data: Optional[Any] = None
    error: Optional[ServiceError] = None
```

**Migration Path:**

1. **Phase 1 (1 week):** Create callback decorator infrastructure
2. **Phase 2 (2 weeks):** Extract business logic from 8 fat callbacks to services
3. **Phase 3 (1 week):** Add callback → service contract tests
4. **Phase 4 (1 week):** Reorganize callbacks into domain folders

---

### Theme 3: Security Gaps in Deployment and Configuration

**Evidence:**

- **Hardcoded secrets** in docker-compose (`POSTGRES_PASSWORD`, `SECRET_KEY`)
- **CORS allows all origins** (`cors_origins: ["*"]`)
- **Zero environment variable support** in config system (IDB 4.4)
- **In-memory 2FA rate limiting** — bypassed in multi-container deployment
- **Redis auth disabled** (`auth.enabled: false`)

**Root Causes:**

- Development-first approach without production hardening
- No secrets management integration (Vault, K8s Secrets)
- Configuration system designed without 12-factor principles

**Recommended Solution:**

```python
# Config: Environment Variable Support
@dataclass
class MLflowConfig:
    tracking_uri: str = field(
        default_factory=lambda: os.getenv('MLFLOW_TRACKING_URI', './mlruns')
    )

# Deployment: K8s Secret References
# values.yaml
postgresql:
  auth:
    existingSecret: lstm-pfd-secrets
    secretKeys:
      adminPasswordKey: postgres-password
```

**Migration Path:**

1. **Phase 1 (2 hours):** Move secrets to environment variables
2. **Phase 2 (4 hours):** Configure environment-specific CORS origins
3. **Phase 3 (1 day):** Migrate 2FA rate limiting to Redis
4. **Phase 4 (2 days):** Implement K8s Secrets/Vault integration

---

### Theme 4: Testing Coverage Has Critical Gaps

**Evidence:**

- **Dashboard Callbacks (IDB 2.3):** 0 tests for 28 callback modules
- **Celery Async Tasks (IDB 2.4):** 0 tests for 23 tasks
- **Data Storage Layer (IDB 3.3):** 0 tests for `cache_manager.py`, `matlab_importer.py`, `data_validator.py`
- **Model Registry (IDB 6.0):** No integration tests

**Root Causes:**

- Dash callbacks difficult to test without framework support
- Celery tasks require mock infrastructure
- Storage layer tested only via integration with other domains

**Recommended Solution:**

```python
# Callback Testing Pattern
def test_experiment_wizard_callback():
    # Use dash.testing.dash_duo fixture
    dash_duo = DashDuo(app)
    dash_duo.find_element("#experiment-wizard-next").click()
    assert dash_duo.find_element("#wizard-step-2").is_displayed()

# Celery Task Testing Pattern
@pytest.fixture
def celery_app():
    app.conf.update(CELERY_ALWAYS_EAGER=True)
    return app
```

**Migration Path:**

1. **Phase 1 (1 day):** Set up Celery test fixtures with `CELERY_ALWAYS_EAGER`
2. **Phase 2 (2 days):** Create 15-20 callback tests for critical paths
3. **Phase 3 (2 days):** Create storage layer unit tests
4. **Phase 4 (1 day):** Add Model Registry CRUD tests

---

### Theme 5: Monolithic Files Impede Maintainability

**Evidence:**

- `settings.py` (IDB 2.1): **1,005 lines / 42KB** — 10x recommended limit
- `signal_generator.py` (IDB 3.1): **935 lines / 37KB** — 4 classes in 1 file
- `NotificationService` (IDB 2.2): **713 lines** — god-class
- `ablation_study.py` (IDB 5.1): **1,175 lines** — should be modules

**Root Causes:**

- No file size enforcement in code review
- Organic growth without periodic refactoring
- Single-developer ownership patterns

**Recommended Solution:**

```
# settings.py Split
layouts/settings/
├── __init__.py      # Exports combined layout
├── api_keys.py      # API key management
├── profile.py       # User profile settings
├── security.py      # 2FA, session settings
├── notifications.py # Email, webhook configs
└── webhooks.py      # Webhook management

# signal_generator.py Split
data_generation/
├── signal_generator.py  # Main orchestrator only
├── fault_modeler.py     # 11 fault implementations
├── noise_generator.py   # 8-layer noise model
├── signal_metadata.py   # Dataclass
└── io/
    ├── hdf5_writer.py
    └── mat_writer.py
```

**Migration Path:**

1. **Phase 1 (4 hours):** Split `settings.py` — highest conflict frequency
2. **Phase 2 (2-3 days):** Split `signal_generator.py` — most complex
3. **Phase 3 (4 hours):** Split `NotificationService`
4. **Phase 4 (optional):** Split `ablation_study.py`

---

## Domain-Specific Deep Dives

### Core ML Engine (Domain 1)

**Current Architecture Assessment:**

| Strength                                             | Weakness                                             |
| ---------------------------------------------------- | ---------------------------------------------------- |
| `BaseModel` abstract class with consistent interface | 5/8 trainers don't inherit from base                 |
| Factory pattern with registry and aliases            | Only ~15 of ~55 models registered                    |
| Preset configurations (`*_small`, `*_base`)          | Duplicate implementations (`HybridPINN`, `ResNet1D`) |
| Scientific documentation with formulas               | `sys.path` manipulation in 8+ files                  |
| Graceful SHAP library fallback                       | KernelSHAP uses incorrect weighting                  |

**Recommended Changes:**

1. Create `UnifiedTrainer` with plugin architecture
2. Implement decorator-based auto-registration for models
3. Merge dual callback systems into single implementation
4. Add ONNX export method to `BaseModel`

**Migration Path:**

- **Phase 1:** Populate empty `__init__.py` in Training/Evaluation (1 hour)
- **Phase 2:** Unify trainer hierarchy (2-3 days)
- **Phase 3:** Merge callback systems (1-2 days)
- **Phase 4:** Implement explanation cache (1-2 days)

---

### Dashboard Platform (Domain 2)

**Current Architecture Assessment:**

| Strength                                   | Weakness                                     |
| ------------------------------------------ | -------------------------------------------- |
| CSS Custom Properties with dark mode       | Callbacks have 4/10 independence ⚠️          |
| Factory pattern for notification providers | 70+ broad `except Exception` blocks          |
| Context manager for DB sessions            | No clientside callbacks (server round-trips) |
| Bound tasks with progress tracking         | No retry logic in 23 Celery tasks            |
| Registration pattern for callbacks         | Settings.py at 1,005 lines                   |

**Recommended Changes:**

1. Create `BaseTask` class with retry, time limits, monitoring
2. Split `settings.py` into 6 module folder
3. Introduce callback middleware (`@authenticated`, `@with_error_boundary`)
4. Convert 10+ toggle callbacks to clientside

**Migration Path:**

- **Phase 1:** Add Celery reliability config (1 hour)
- **Phase 2:** Fix 2FA rate limiting with Redis (2 hours)
- **Phase 3:** Create `BaseTask` class (1 day)
- **Phase 4:** Split `settings.py` (4 hours)

---

### Data Engineering (Domain 3)

**Current Architecture Assessment:**

| Strength                                       | Weakness                                 |
| ---------------------------------------------- | ---------------------------------------- |
| Physics-based modeling (Sommerfeld, viscosity) | Monolithic 37KB signal_generator.py      |
| Layered noise with ablation support            | 0 unit tests for cache/import/validation |
| Thread-local HDF5 handles                      | HDF5 file handle leaks                   |
| Config-hash cache invalidation                 | Redis `KEYS` blocking in production      |
| Factory methods for data sources               | Duplicate `Compose` class                |

**Recommended Changes:**

1. Split `signal_generator.py` into focused modules
2. Add comprehensive unit tests for storage layer
3. Create unified `BaseSignalDataset` interface
4. Replace bare `except:` with specific exceptions

**Migration Path:**

- **Phase 1:** Quick wins — fix `KEYS` → `SCAN`, delete duplicate `Compose` (1 hour)
- **Phase 2:** Add storage layer tests (2 days)
- **Phase 3:** Split signal_generator.py (2-3 days)
- **Phase 4:** Create `BaseSignalDataset` ABC (1-2 days)

---

### Infrastructure (Domain 4)

**Current Architecture Assessment:**

| Strength                                 | Weakness                                 |
| ---------------------------------------- | ---------------------------------------- |
| Connection pooling with health checks    | Hardcoded secrets in docker-compose      |
| Multi-stage Docker builds                | Zero environment variable support        |
| Complete ONNX + quantization pipeline    | Dashboard Dockerfile lacks non-root user |
| Comprehensive constants file (629 lines) | Missing tests for Dashboard components   |
| Device-agnostic test fixtures            | Deprecated SQLAlchemy patterns           |

**Recommended Changes:**

1. Move secrets to K8s Secrets/environment variables
2. Add environment variable support to configs
3. Create Callback + Async Task test suites
4. Implement Alembic migration system

**Migration Path:**

- **Phase 1:** Move secrets to env vars (2 hours)
- **Phase 2:** Add env var support to configs (2 hours)
- **Phase 3:** Create test suites (1-2 days)
- **Phase 4:** Implement Alembic (2-3 days)

---

### Research & Integration (Domain 5-6)

**Current Architecture Assessment:**

| Strength                            | Weakness                                      |
| ----------------------------------- | --------------------------------------------- |
| Dataclass configs with `.to_dict()` | DPI inconsistency (150 vs 300) across 7 files |
| Multi-seed experiments for validity | No unified visualization style config         |
| McNemar's significance test         | `sys.path.insert()` in all adapters           |
| `--quick` flag for development      | 7/10 unified pipeline phases are placeholders |
| Adapter pattern with callbacks      | ~200 lines of duplicated training loops       |

**Recommended Changes:**

1. Create `visualization/style_config.py` for unified styling
2. Extract common `ExperimentRunner` base class
3. Replace `sys.path.insert()` with proper packaging
4. Implement or remove unified pipeline placeholders

**Migration Path:**

- **Phase 1:** Create style config (2 hours)
- **Phase 2:** Consolidate seed handling (30 min)
- **Phase 3:** Extract experiment infrastructure (5 days)
- **Phase 4:** Replace `sys.path` with packaging (3 days)

---

## Interface Redesign Recommendations

### Interfaces to Refactor

| Interface              | Current State                  | Recommended State                    | Breaking? | Migration            |
| ---------------------- | ------------------------------ | ------------------------------------ | --------- | -------------------- |
| `Trainer.fit()`        | Each trainer has own signature | Unified signature via `BaseTrainer`  | Yes       | Deprecation period   |
| `Evaluator.evaluate()` | Returns various dict shapes    | Returns `EvaluationResult` dataclass | Yes       | Add adapter layer    |
| Service returns        | Mixed dicts/tuples             | `ServiceResult` dataclass            | No        | Gradual adoption     |
| Celery task returns    | Inconsistent keys              | `{success, data, error}` standard    | No        | Update incrementally |
| Explanations           | Per-explainer format           | `ExplanationResult` protocol         | Yes       | Implement wrapper    |

### Backward Compatibility Considerations

1. **Deprecation Warnings:** Add 2-version deprecation cycle for breaking changes
2. **Adapter Pattern:** Wrap old interfaces in new ones during transition
3. **Version Fields:** Add `version` to checkpoint formats for migration
4. **Feature Flags:** Use flags to opt-in to new behaviors before making default

---

## Technology Stack Recommendations

### Libraries to Add

| Library       | Purpose                     | Impact                            |
| ------------- | --------------------------- | --------------------------------- |
| `pydantic`    | Type-safe config validation | Replace current schema validation |
| `slowapi`     | API rate limiting           | Security hardening                |
| `alembic`     | Database migrations         | Replace manual SQL                |
| `structlog`   | Structured logging          | Better observability              |
| `pytest-dash` | Callback testing            | Enable dashboard tests            |

### Libraries to Remove/Replace

| Current                 | Replacement                  | Reason                      |
| ----------------------- | ---------------------------- | --------------------------- |
| Manual SQL migrations   | Alembic                      | Auto-generation, versioning |
| `declarative_base()`    | `DeclarativeBase` class      | SQLAlchemy 2.0 deprecation  |
| `datetime.utcnow`       | `datetime.now(timezone.utc)` | Python 3.12+ deprecation    |
| In-memory rate limiting | Redis-backed                 | Multi-process safety        |

### Framework Upgrades

| Framework    | Current Issue              | Target               |
| ------------ | -------------------------- | -------------------- |
| SQLAlchemy   | Using deprecated patterns  | 2.0 syntax migration |
| Python       | `datetime.utcnow` usage    | 3.12+ compatibility  |
| ONNX Runtime | Static quantization broken | Latest stable        |

---

## Greenfield Guidance: "If Starting From Scratch..."

### Architecture Patterns We Would Use

1. **Clean Architecture with Clear Boundaries**
   - Core domain logic independent of frameworks
   - Adapters for external dependencies (DB, Celery, UI)
   - Use cases as primary organizational unit

2. **Event-Driven for Dashboard ↔ Core ML Communication**
   - Message queue instead of direct function calls
   - Async by default, sync as optimization
   - Progress updates via events, not polling

3. **Repository Pattern for Data Access**
   - Abstract data access behind interfaces
   - Easy to swap storage backends
   - Testable without real database

4. **Protocol Classes for Interfaces**

   ```python
   class ModelProtocol(Protocol):
       def forward(self, x: Tensor) -> Tensor: ...
       def get_config(self) -> Dict[str, Any]: ...
   ```

5. **Dependency Injection**
   - No hardcoded dependencies in constructors
   - Configuration via composition root
   - Easy testing with mock dependencies

### Technology Choices

| Area       | Current                   | Ideal                                    |
| ---------- | ------------------------- | ---------------------------------------- |
| Config     | Dataclasses + JSON Schema | Pydantic with env var support            |
| Migrations | Manual SQL files          | Alembic auto-generation                  |
| Task Queue | Celery (no base task)     | Celery with `BaseTask` + Dramatiq backup |
| Logging    | Dual modules              | Single structlog                         |
| Testing    | Pytest (gaps)             | Pytest + hypothesis + dash.testing       |

### File Organization

```
packages/
├── core/                     # Domain 1: Core ML
│   ├── models/
│   │   ├── __init__.py       # Explicit exports
│   │   ├── base.py           # BaseModel ABC
│   │   ├── registry.py       # Auto-registration decorator
│   │   └── architectures/    # One file per architecture family
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py        # UnifiedTrainer with plugins
│   │   ├── plugins/          # MixedPrecision, PhysicsLoss, etc.
│   │   ├── callbacks/        # Single unified system
│   │   └── losses/           # Single source of truth
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluator.py      # BaseEvaluator
│       └── result.py         # EvaluationResult dataclass
│
├── dashboard/                 # Domain 2: Dashboard
│   ├── layouts/
│   │   └── settings/         # Split into focused modules
│   ├── services/
│   │   ├── notification/     # Split NotificationService
│   │   └── result.py         # ServiceResult dataclass
│   ├── callbacks/
│   │   ├── core/             # Decorators, utilities
│   │   └── domains/          # Organized by feature
│   └── tasks/
│       ├── base.py           # BaseTask with retry, limits
│       └── ...
│
├── data/                      # Domain 3: Data Engineering
│   ├── generation/
│   │   ├── signal_generator.py  # Orchestrator only
│   │   ├── fault_modeler.py
│   │   └── noise_generator.py
│   ├── datasets/
│   │   ├── base.py           # BaseSignalDataset ABC
│   │   └── ...
│   └── storage/
│       ├── cache_manager.py
│       └── ...
│
└── integration/               # Domain 6: Integration
    ├── adapters/
    │   ├── base.py           # Abstract adapter interface
    │   └── ...
    ├── registry/
    └── pipeline/
```

### What We Got Right (Don't Change)

| Pattern                  | Evidence                            | Keep Because                                 |
| ------------------------ | ----------------------------------- | -------------------------------------------- |
| Factory + Registry       | `model_factory.py`                  | Clean extensibility, case-insensitive lookup |
| Context Managers         | `get_db_session()`, `DeviceManager` | Guaranteed resource cleanup                  |
| Preset Configurations    | `*_small`, `*_base`, `*_large`      | Quick setup for common cases                 |
| Config-Hash Invalidation | `CacheManager`                      | Auto-cache invalidation on changes           |
| Division Guards          | `if x > 0 else 0.0`                 | Numerical stability                          |
| Thread-Local HDF5        | `threading.local()`                 | Multi-worker DataLoader safety               |
| Graceful Fallback        | Optional SHAP/UMAP                  | Degrades without crashing                    |
| Progress Callbacks       | `update_state()`                    | Real-time UI feedback                        |
| Multi-Seed Experiments   | Research scripts                    | Statistical validity                         |
| McNemar's Test           | `pinn_ablation.py`                  | Proper significance testing                  |

---

## Summary of Estimated Effort

| Domain                     | Quick Wins | Medium Tasks | Large Refactors | Total     |
| -------------------------- | ---------- | ------------ | --------------- | --------- |
| Core ML (1.x)              | 4h         | 12h          | 40h             | ~56h      |
| Dashboard (2.x)            | 2h         | 16h          | 40h             | ~58h      |
| Data Engineering (3.x)     | 3h         | 16h          | 40h             | ~59h      |
| Infrastructure (4.x)       | 4h         | 12h          | 32h             | ~48h      |
| Research/Integration (5-6) | 4h         | 16h          | 56h             | ~76h      |
| **Total**                  | **17h**    | **72h**      | **208h**        | **~297h** |

> [!NOTE]
> These estimates assume single-developer effort. Parallelization across 3-4 developers could reduce calendar time by 60%.

---

_Strategic insights synthesized from 5 Domain Consolidated Reports — 2026-01-24_
