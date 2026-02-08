# Interface Contracts Catalog — Cross-Domain APIs

**Compilation Date:** 2026-01-24  
**Source:** Section 7 (Cross-Domain Dependencies) from all 5 consolidated domain reports

---

## Purpose

Comprehensive documentation of boundaries where domains interact. This catalog maps every interface used for cross-domain communication in the LSTM_PFD system, documents contract details, and assesses stability and risk.

---

## 1. Interface Inventory

| From Domain     | To Domain     | Interface Name       | Location                                           | Stability | Risk   |
| --------------- | ------------- | -------------------- | -------------------------------------------------- | --------- | ------ |
| Core ML (1)     | Dashboard (2) | Model Factory        | `packages/core/models/model_factory.py`            | STABLE    | LOW    |
| Core ML (1)     | Dashboard (2) | Model `forward()`    | `packages/core/models/base_model.py`               | STABLE    | LOW    |
| Core ML (1)     | Dashboard (2) | Trainer `fit()`      | `packages/core/training/trainer.py`                | STABLE    | MEDIUM |
| Core ML (1)     | Dashboard (2) | Trainer History      | `packages/core/training/trainer.py`                | FRAGILE   | MEDIUM |
| Core ML (1)     | Dashboard (2) | Evaluator API        | `packages/core/evaluation/*.py`                    | STABLE    | LOW    |
| Core ML (1)     | Dashboard (2) | XAI Explainers       | `packages/core/explainability/*.py`                | STABLE    | LOW    |
| Dashboard (2)   | Core ML (1)   | DeepLearningAdapter  | `integrations/deep_learning_adapter.py`            | FRAGILE   | HIGH   |
| Dashboard (2)   | Core ML (1)   | Phase0Adapter        | `integrations/phase0_adapter.py`                   | FRAGILE   | HIGH   |
| Dashboard (2)   | Core ML (1)   | Phase1Adapter        | `integrations/phase1_adapter.py`                   | FRAGILE   | HIGH   |
| Data (3)        | Core ML (1)   | BearingFaultDataset  | `data/dataset.py`                                  | STABLE    | LOW    |
| Data (3)        | Core ML (1)   | StreamingHDF5Dataset | `data/streaming_hdf5_dataset.py`                   | STABLE    | LOW    |
| Data (3)        | Core ML (1)   | Transform Pipelines  | `data/transforms.py`, `data/cnn_transforms.py`     | STABLE    | LOW    |
| Data (3)        | Dashboard (2) | CacheService         | `packages/dashboard/services/cache_service.py`     | STABLE    | LOW    |
| Data (3)        | Dashboard (2) | ExplanationCache     | `packages/dashboard/services/explanation_cache.py` | STABLE    | LOW    |
| Infra (4)       | Dashboard (2) | Database Session API | `packages/dashboard/database/connection.py`        | STABLE    | LOW    |
| Infra (4)       | Dashboard (2) | ORM Models           | `packages/dashboard/models/*.py`                   | STABLE    | MEDIUM |
| Infra (4)       | Core ML (1)   | Config Classes       | `config/*.py`                                      | STABLE    | LOW    |
| Infra (4)       | All           | Constants            | `utils/constants.py`                               | STABLE    | LOW    |
| Research (5)    | Core ML (1)   | Model Imports        | `packages/core/models/*`                           | STABLE    | LOW    |
| Research (5)    | Data (3)      | SignalGenerator      | `data/signal_generator.py`                         | STABLE    | LOW    |
| Integration (6) | Dashboard (2) | Adapter APIs         | `integrations/*.py`                                | FRAGILE   | HIGH   |
| Integration (6) | All           | Utility Modules      | `utils/*.py`                                       | STABLE    | LOW    |

---

## 2. Core ML ↔ Dashboard Interfaces

### Interface 1: Model Factory

**API Signature:**

```python
def create_model(name: str, **kwargs) -> BaseModel:
    """Create model instance by registered name."""
```

**Location:** `packages/core/models/model_factory.py`

**Contract Details:**

- **Input:**
  - `name` (str): Registered model name (case-insensitive)
  - `**kwargs`: Model-specific configuration
- **Output:** PyTorch `nn.Module` instance inheriting from `BaseModel`
- **Exceptions:**
  - `KeyError`: Model name not registered
  - `TypeError`: Invalid configuration arguments

**Consumers:**

- `packages/dashboard/services/experiment_service.py`
- `packages/dashboard/tasks/training_tasks.py`
- `integrations/deep_learning_adapter.py`

**Stability:** ✅ STABLE — Well-tested factory pattern

**Testing:**

```python
from packages.core.models import create_model
model = create_model("cnn1d", num_classes=11)
assert isinstance(model, BaseModel)
assert hasattr(model, 'forward')
assert hasattr(model, 'get_config')
```

**Change Protocol:**

1. New models: Register in `MODEL_REGISTRY`
2. Signature changes: Add deprecation warnings first
3. Breaking changes: Coordinate with Dashboard team

---

### Interface 2: Model Forward Pass

**API Signature:**

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass through the model."""
```

**Location:** `packages/core/models/base_model.py`

**Contract Details:**

- **Input:** `[B, C, T]` tensor where B=batch, C=channels, T=time
- **Output:** `[B, num_classes]` logits (NOT softmax!)
- **Guarantees:**
  - Deterministic in eval mode
  - Gradient-enabled in train mode

**Consumers:** All trainers, evaluators, explainers

**Stability:** ✅ STABLE — Abstract method contract

> [!CAUTION]
> **PINN Exception:** Dual-input models (`HybridPINN`) require `(x, physics_params)` input. Standard trainers cannot handle these without modification.

---

### Interface 3: Trainer Interface

**API Signature:**

```python
class Trainer:
    def fit(self, num_epochs: int) -> Dict[str, List[float]]:
        """Train model for specified epochs."""

    def train_epoch(self) -> Dict[str, float]:
        """Single training epoch."""
        # Returns: {"train_loss": float, "train_acc": float}

    def validate_epoch(self) -> Dict[str, float]:
        """Single validation epoch."""
        # Returns: {"val_loss": float, "val_acc": float, "lr": float}
```

**Location:** `packages/core/training/trainer.py`

**Contract Details:**

- **`fit()` returns:** History dict with keys `train_loss`, `train_acc`, `val_loss`, `val_acc`
- **State properties:**
  - `trainer.best_val_acc` — Best validation accuracy
  - `trainer.current_epoch` — Current epoch number
  - `trainer.history` — Training history dictionary

**Consumers:**

- Dashboard training progress UI
- Celery training tasks

**Stability:** ⚠️ FRAGILE — History key names are implicit

**Risk:** Dashboard plots break if history key names change

**Mitigation:** Document required history keys explicitly

---

### Interface 4: Evaluator Interface

**API Signature:**

```python
class CNNEvaluator:
    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Evaluate model on test data."""
        # Returns: {"accuracy": float, "confusion_matrix": np.ndarray, "per_class_metrics": Dict}

    def predict(self, inputs: torch.Tensor) -> np.ndarray:
        """Get predicted labels."""

    def save_results(self, path: str) -> None:
        """Save results to JSON."""
```

**Location:** `packages/core/evaluation/cnn_evaluator.py`

**Contract Details:**

- **Percentage convention:** Values in 0-100 range
- **Confusion matrix:** Shape `[num_classes, num_classes]`
- **Per-class metrics:** Dict with `precision`, `recall`, `f1` per class

**Consumers:**

- Dashboard results display
- Research analysis scripts

**Stability:** ✅ STABLE

> [!WARNING]
> **Percentage Scale Risk:** Some evaluators return 0-1, others 0-100. Standardize on 0-100 for consistency.

---

### Interface 5: XAI Explainer Interface

**API Signature:**

```python
class IntegratedGradientsExplainer:
    def explain(self, input: torch.Tensor, target_class: int) -> np.ndarray:
        """Generate attribution for input signal."""
        # Returns: Attribution array same shape as input

    def visualize(self, attributions: np.ndarray) -> matplotlib.figure.Figure:
        """Create visualization of attributions."""
```

**Location:** `packages/core/explainability/integrated_gradients.py`

**Available Explainers:**

- `IntegratedGradientsExplainer`
- `SHAPExplainer` (GradientSHAP, DeepSHAP, KernelSHAP)
- `LIMEExplainer`
- `MCDropoutExplainer`
- `CAVExplainer` (TCaVs)

**Consumers:**

- Dashboard XAI tab
- Research visualization scripts

**Stability:** ✅ STABLE — Well-defined interface

---

## 3. Dashboard ↔ Core ML Adapters

> [!CAUTION]
> **HIGH COUPLING AREA** — Independence Score: 4/10  
> These adapters use `sys.path.insert()` which is fragile. Recommend proper Python packaging.

### Interface 6: DeepLearningAdapter

**API Signature:**

```python
class DeepLearningAdapter:
    @staticmethod
    def train(config: Dict[str, Any], progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute deep learning training from Dashboard."""
```

**Location:** `integrations/deep_learning_adapter.py`

**Contract Details:**

- **Input Config Required Keys:**
  - `model_type` (str): Model name for factory
  - `cache_path` (str): Path to HDF5 data cache
  - `num_epochs` (int): Training epochs
  - `batch_size` (int): Batch size
  - `learning_rate` (float): Initial LR

- **Input Config Optional Keys:**
  - `device` (str): `"cuda"` or `"cpu"`
  - `early_stopping_patience` (int)
  - `optimizer` (str), `scheduler` (str)
  - `hyperparameters` (Dict): Model-specific params

- **Output on Success:**

  ```python
  {"success": True, "test_accuracy": float, "history": Dict, ...}
  ```

- **Output on Failure:**
  ```python
  {"success": False, "error": str}
  ```

**Progress Callback Protocol:**

```python
def callback(epoch: int, metrics: Dict[str, Any]) -> None:
    # Called each epoch with: train_loss, val_loss, val_accuracy
```

**Consumers:** Celery training tasks, Dashboard training UI

**Stability:** ⚠️ FRAGILE — Uses `sys.path.insert()`

**Risk:** Import failures on path changes

**Mitigation:** Replace with proper Python packaging

---

### Interface 7: Phase0Adapter (Signal Generation)

**API Signature:**

```python
class Phase0Adapter:
    @staticmethod
    def generate(config: Dict[str, Any], progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Generate synthetic signals from Dashboard."""
```

**Location:** `integrations/phase0_adapter.py`

**Contract Details:**

- **Output:** HDF5 file with `train`, `val`, `test` groups
- **Returns:** `{"success": True, "cache_path": str, ...}`

**Stability:** ⚠️ FRAGILE — Placeholder implementation

---

### Interface 8: Phase1Adapter (Classical ML)

**API Signature:**

```python
class Phase1Adapter:
    @staticmethod
    def train(config: Dict[str, Any], progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute classical ML training from Dashboard."""
```

**Location:** `integrations/phase1_adapter.py`

**Contract Details:**

- Delegates to classical ML pipeline
- Returns standard success/error dict

**Stability:** ⚠️ FRAGILE — Uses `sys.path.insert()`

---

## 4. Data ↔ Core ML Interfaces

### Interface 9: BearingFaultDataset

**API Signature:**

```python
class BearingFaultDataset(Dataset):
    @classmethod
    def from_hdf5(cls, path: str, split: str = 'train', transform: Optional[Callable] = None) -> 'BearingFaultDataset':
        """Load dataset from HDF5 file."""

    @classmethod
    def from_mat_file(cls, path: str, ...) -> 'BearingFaultDataset':
        """Load dataset from MATLAB file."""

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get (signal, label) tuple."""
```

**Location:** `data/dataset.py`

**Contract Details:**

- **Output format:** `Tuple[torch.Tensor, int]`
- **Signal shape:** `[C, T]` where C=channels, T=signal_length
- **Label:** Integer class index (0 to 10)

**Consumers:** All Core ML trainers

**Stability:** ✅ STABLE — Factory method pattern

---

### Interface 10: StreamingHDF5Dataset

**API Signature:**

```python
class StreamingHDF5Dataset(Dataset):
    def __init__(self, hdf5_path: str, split: str = 'train', transform: Optional[Callable] = None):
        """Memory-efficient streaming dataset."""
```

**Location:** `data/streaming_hdf5_dataset.py`

**Contract Details:**

- **Thread-safety:** Uses thread-local HDF5 handles
- **Multi-worker safe:** SWMR mode enabled
- **Memory:** Only loads requested samples

**Stability:** ✅ STABLE

---

### Interface 11: Transform Pipeline

**API Signature:**

```python
def get_train_transforms() -> Compose:
    """Get training data augmentation transforms."""

def get_test_transforms() -> Compose:
    """Get test data preprocessing transforms."""
```

**Location:** `data/transforms.py`, `data/cnn_transforms.py`

**Contract Details:**

- **Input:** NumPy array or torch.Tensor
- **Output:** torch.Tensor

**Stability:** ✅ STABLE

> [!WARNING]
> **Duplicate `Compose` class** in two files. Use `transforms.py` version only.

---

## 5. Infrastructure ↔ All Domains Interfaces

### Interface 12: Database Session API

**API Signature:**

```python
@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Get database session with auto-commit/rollback."""
```

**Location:** `packages/dashboard/database/connection.py`

**Contract Details:**

- **Usage:** `with get_db_session() as session:`
- **Auto-commit:** On successful exit
- **Auto-rollback:** On exception
- **Pooling:** 30 connections + 30 overflow

**Consumers:** All Dashboard services, callbacks

**Stability:** ✅ STABLE — Well-established pattern

**Testing:**

```python
with get_db_session() as session:
    result = session.query(Experiment).filter_by(id=1).first()
# Auto-commit on success, rollback on exception
```

---

### Interface 13: Configuration Classes

**API Signature:**

```python
@dataclass
class TrainingConfig(BaseConfig):
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001

    @classmethod
    def from_yaml(cls, path: Path) -> 'TrainingConfig':
        """Load from YAML file."""

    def validate(self) -> bool:
        """Run JSON Schema validation."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MLflow logging."""
```

**Location:** `config/training_config.py`

**Config Classes Available:**

- `TrainingConfig`, `OptimizerConfig`, `SchedulerConfig`
- `ModelConfig`, `CNNConfig`, `TransformerConfig`
- `DataConfig`, `SignalConfig`, `GenerationConfig`
- `ExperimentConfig`, `MLflowConfig`

**Consumers:** All domains

**Stability:** ✅ STABLE — Dataclass pattern

> [!IMPORTANT]
> **No environment variable support** currently. Secrets must be YAML-loaded.

---

### Interface 14: Constants Module

**API Signature:**

```python
# Signal parameters
SIGNAL_LENGTH: int = 102400
SAMPLING_RATE: int = 20480
NUM_CLASSES: int = 11

# Fault types
FAULT_TYPES: List[str] = ['Normal', 'Misalignment', ...]
```

**Location:** `utils/constants.py`

**Contract Details:**

- **629 lines** of centralized constants
- **50+ constants** covering signals, models, training
- **5 utility functions** for validation

**Consumers:** All domains (50+ import sites)

**Stability:** ✅ STABLE — Single source of truth

**Change Protocol:**

1. Changes affect ALL domains
2. Requires full regression testing
3. Document breaking changes

---

## 6. Services ↔ Callbacks Interfaces

> [!CAUTION]
> **HIGH COUPLING AREA** — Callbacks Independence Score: 4/10  
> The callback layer creates tight coupling between UI layouts and backend services.

### Interface 15: Service Return Pattern

**Standard Return Structures:**

```python
# Success case (Services)
{"status": "success", "data": Any, "message": str}

# Success case (Tuple pattern)
(True, data, None)

# Failure case
{"status": "error", "error": str}
(False, None, error_message)
```

**Problem:** Inconsistent patterns across services. Recommend unifying to:

```python
@dataclass
class ServiceResult:
    success: bool
    data: Optional[Any] = None
    error: Optional[ServiceError] = None
```

---

### Interface 16: Celery Task Return Pattern

**Standard Return Structure:**

```python
{
    "success": bool,
    "error": Optional[str],
    "traceback": Optional[str],
    # Task-specific data...
}
```

**Progress Update Pattern:**

```python
self.update_state(
    state='PROGRESS',
    meta={
        "progress": float,  # 0.0 - 1.0
        "status": str,
        "current": int,
        "total": int
    }
)
```

**Location:** All task files in `packages/dashboard/tasks/`

**Stability:** ✅ STABLE — Consistent pattern

---

## 7. Integration Risk Assessment

### High-Risk Interfaces (Many Dependencies)

| Interface            | Consumers                     | Risk                     | Mitigation                           |
| -------------------- | ----------------------------- | ------------------------ | ------------------------------------ |
| `utils/constants.py` | 50+ files                     | Changes break everything | Version field, full regression tests |
| `model_factory`      | Training, Dashboard, Research | Registry incompleteness  | Complete registration                |
| Trainer history keys | Dashboard plots               | Silent UI breakage       | Document required keys               |
| Checkpoint format    | Evaluation, Dashboardload     | Model unloadable         | Version field + migration            |
| HDF5 cache format    | All data consumers            | Data corruption          | Cache versioning                     |

### Medium-Risk Interfaces

| Interface                                 | Risk                       | Mitigation                   |
| ----------------------------------------- | -------------------------- | ---------------------------- |
| Evaluator percentage scale (0-1 vs 0-100) | Incorrect displays         | Standardize on 0-100         |
| Config schema changes                     | Valid experiments rejected | Complete validation coverage |
| ORM model relationships                   | Dashboard query failures   | Add relationship tests       |

### Stable Interfaces (Safe to Depend On)

- **Model Factory:** Used widely, well-tested factory pattern
- **BearingFaultDataset:** Standard PyTorch Dataset interface
- **Database Session API:** Context manager pattern
- **Constants Module:** Single source of truth

---

## 8. Deprecated Interfaces

| Interface              | Location                | Replacement                  | Removal Date             |
| ---------------------- | ----------------------- | ---------------------------- | ------------------------ |
| `from_matlab_struct()` | `config/base_config.py` | `from_yaml()`                | TBD                      |
| `declarative_base()`   | `database/base.py`      | `DeclarativeBase` class      | SQLAlchemy 2.0 migration |
| `datetime.utcnow`      | ORM models              | `datetime.now(timezone.utc)` | Python 3.12 migration    |

---

## 9. Versioning Strategy

For future interface changes:

1. **Semantic Versioning for Major APIs**
   - Model Factory: `v1.x` → `v2.0` for breaking changes
   - Config Schema: Version field in YAML files

2. **Deprecation Warnings Before Removal**

   ```python
   import warnings
   warnings.warn("This API is deprecated. Use X instead.", DeprecationWarning)
   ```

3. **Support Period**
   - Major APIs: 2-version support period
   - Internal utilities: No guarantee

4. **Migration Paths**
   - Provide upgrade scripts for checkpoint formats
   - Document breaking changes in CHANGELOG

---

## 10. Testing Contracts

### Critical Contract Tests Required

| Interface                     | Test Type   | Priority |
| ----------------------------- | ----------- | -------- |
| Model Factory → Trainer       | Integration | P0       |
| Trainer → Dashboard history   | Contract    | P0       |
| Evaluator → Dashboard display | Contract    | P1       |
| Adapter → Celery tasks        | Integration | P0       |
| Constants → All consumers     | Validation  | P1       |

### Recommended Test Pattern

```python
# tests/contracts/test_trainer_dashboard_contract.py
def test_trainer_returns_required_history_keys():
    """Ensure trainer history has keys Dashboard expects."""
    trainer = create_test_trainer()
    history = trainer.fit(epochs=1)

    REQUIRED_KEYS = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
    for key in REQUIRED_KEYS:
        assert key in history, f"Dashboard requires '{key}' in trainer history"
```

---

_Interface Contracts Catalog compiled from 5 domain consolidated reports — 2026-01-24_
