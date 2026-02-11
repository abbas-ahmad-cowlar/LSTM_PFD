# IDB 6.0: Integration Layer Best Practices

**Domain:** Cross-Cutting Concerns  
**Date:** 2026-01-23  
**Source:** `integration/`, `packages/dashboard/integrations/`, `utils/`

---

## 1. Integration Patterns

### 1.1 Adapter Pattern for Cross-Domain Bridge

Use static-method adapters to bridge Dashboard tasks to Core ML pipelines:

```python
# packages/dashboard/integrations/phase1_adapter.py
class Phase1Adapter:
    """Adapter for Phase 1 classical ML training."""

    @staticmethod
    def train(config: dict, progress_callback=None) -> dict:
        """
        Train classical ML model using Phase 1 pipeline.

        Args:
            config: Training configuration dict
            progress_callback: Optional callback(epoch, metrics)

        Returns:
            Training results dictionary with success flag
        """
        try:
            from pipelines.classical_ml_pipeline import ClassicalMLPipeline
            # ... implementation
            return {"success": True, "test_accuracy": results["test_accuracy"]}
        except Exception as e:
            logger.error(f"Phase 1 training failed: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
```

**Key Points:**

- Static methods allow stateless invocation from Celery tasks
- Always return dict with `success` boolean flag
- Accept optional `progress_callback` for UI updates
- Use try/except with structured error responses

---

### 1.2 Progress Callback Convention

Implement consistent progress reporting across all adapters:

```python
# Standard progress callback signature
ProgressCallback = Callable[[int, Dict[str, Any]], None]

def train(config: dict, progress_callback: ProgressCallback = None):
    for epoch in range(1, num_epochs + 1):
        # Training logic...

        if progress_callback:
            progress_callback(epoch, {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": current_lr,
                "epoch_time": time.time() - epoch_start,
            })
```

**Standard metrics dict keys:**

- `epoch`, `train_loss`, `val_loss`
- `train_accuracy`, `val_accuracy`
- `learning_rate`, `epoch_time`
- `status` (string for phase descriptions)

---

### 1.3 Model Registry Pattern

Centralized model tracking with SQLite backend:

```python
# integration/model_registry.py
class ModelRegistry:
    def __init__(self, db_path: str = 'models/model_registry.db'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def register_model(
        self,
        model_name: str,
        phase: str,
        accuracy: float,
        model_path: str,
        **kwargs  # precision, recall, hyperparameters, etc.
    ) -> int:
        # Store hyperparameters as JSON
        hyperparameters_json = json.dumps(kwargs.get('hyperparameters', {}))
        # ... SQL insert
        return model_id
```

**Best practices:**

- Use `**kwargs` for optional metadata
- Serialize complex objects (dicts) to JSON
- Auto-create parent directories
- Return generated IDs for chaining

---

## 2. Utility Function Conventions

### 2.1 Constants Centralization

All magic numbers go in `utils/constants.py`:

```python
# utils/constants.py - 629 LOC of centralized constants

# Signal parameters (with units in comments)
SIGNAL_LENGTH: int = 102400        # samples
SAMPLING_RATE: int = 20480         # Hz
SIGNAL_DURATION: float = 5.0       # seconds
NYQUIST_FREQUENCY: float = 10240.0 # Hz

# Fault classification
NUM_CLASSES: int = 11
FAULT_TYPES: List[str] = [
    'sain', 'desalignement', 'desequilibre', ...
]

# Provide lookup utilities
def get_fault_id(fault_name: str) -> int:
    """Get numeric ID for a fault type name."""
    if fault_name.lower() not in FAULT_TYPE_TO_ID:
        raise ValueError(f"Unknown fault type: {fault_name}")
    return FAULT_TYPE_TO_ID[fault_name.lower()]
```

**Organization pattern:**

- Group by category with `# ==== CATEGORY ====` headers
- Document units in comments
- Provide utility functions for lookups
- Export via `__init__.py`

---

### 2.2 Clean `__init__.py` Exports

Structure exports for easy imports:

```python
# utils/__init__.py
from .constants import (
    SIGNAL_LENGTH, SAMPLING_RATE, NUM_CLASSES, FAULT_TYPES,
    get_fault_id, get_fault_name,
)
from .reproducibility import set_seed, make_deterministic
from .device_manager import get_device, DeviceManager
from .file_io import save_json, load_json, ensure_dir

__all__ = [
    # Constants
    'SIGNAL_LENGTH', 'SAMPLING_RATE', ...
    # Reproducibility
    'set_seed', 'make_deterministic', ...
    # Device management
    'get_device', 'DeviceManager', ...
]
```

**Benefits:**

- `from utils import get_device, set_seed` works cleanly
- IDE autocompletion for all utilities
- Explicit `__all__` prevents accidental exports

---

### 2.3 Context Manager Pattern

Use context managers for resource management:

```python
# utils/device_manager.py
class DeviceManager:
    """Context manager for device operations."""

    def __init__(self, prefer_gpu: bool = True, gpu_id: Optional[int] = None):
        self.prefer_gpu = prefer_gpu
        self.gpu_id = gpu_id
        self.device = None

    def __enter__(self) -> torch.device:
        self.device = get_device(self.prefer_gpu, self.gpu_id)
        return self.device

    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_gpu_memory(self.device)

# Usage
with DeviceManager(prefer_gpu=True) as device:
    model = MyModel().to(device)
    # ... training
# GPU memory automatically cleared
```

---

## 3. Cross-Domain Communication Patterns

### 3.1 Config-Driven Interface

Use flat config dicts for cross-domain communication:

```python
# Dashboard -> Adapter config structure
config = {
    # Required
    "model_type": "cnn1d",
    "cache_path": "data/processed/signals_cache.h5",

    # Training params
    "num_epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,

    # Optional with defaults
    "device": "auto",
    "early_stopping_patience": 15,
    "optimizer": "adam",
    "scheduler": "plateau",

    # Model-specific (nested)
    "hyperparameters": {
        "num_layers": 4,
        "hidden_dim": 256,
    }
}
```

**Conventions:**

- Flat structure for common params
- Nested `hyperparameters` dict for model-specific
- Use `.get()` with sensible defaults
- Document expected keys in docstrings

---

### 3.2 Result Dict Convention

Standardized return format:

```python
# Success case
return {
    "success": True,
    "model_type": config["model_type"],
    "test_accuracy": test_acc,
    "test_f1": f1_score,
    "training_time": total_time,
    "history": training_history,  # Lists of per-epoch metrics
}

# Failure case
return {
    "success": False,
    "error": str(e),
}
```

**Required keys:**

- `success: bool` — Always present
- `error: str` — Present when `success=False`

---

## 4. Error Propagation Patterns

### 4.1 Structured Exception Handling

```python
# Pattern: Catch, log, return structured error
@staticmethod
def train(config: dict, progress_callback=None):
    try:
        # ... training logic
        return {"success": True, ...}
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
```

**Key elements:**

- Use `exc_info=True` for full traceback in logs
- Return structured dict, never raise to caller
- Include error message in return dict

---

### 4.2 Graceful Import Fallbacks

For optional dependencies:

```python
# integration/unified_pipeline.py
try:
    from utils.constants import SAMPLING_RATE
except ImportError:
    SAMPLING_RATE = 20480  # Fallback value
    logger.warning("Using fallback SAMPLING_RATE=20480")
```

> [!WARNING]
> Silent fallbacks can mask real issues. Log at WARNING level.

---

### 4.3 Validation Before Execution

```python
# integration/configuration_validator.py
def validate_config(config: Dict[str, Any]) -> bool:
    """Validate master configuration file."""
    try:
        _validate_required_sections(config)
        _validate_value_ranges(config)
        _validate_file_paths(config)
        _validate_hyperparameters(config)
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False
```

**Pattern:** Separate validation functions, aggregate in main validator.

---

## 5. Logging Conventions for Cross-Domain Flows

### 5.1 Structured Logging Setup

```python
# utils/logging.py
def setup_logging(
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True
) -> None:
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
```

**Format:** `2026-01-23 14:30:25 | my_module           | INFO     | Message`

---

### 5.2 Logger Per Module

```python
# At module top
from utils.logging import get_logger
logger = get_logger(__name__)

# Usage
logger.info(f"Starting training: {config['model_type']}")
logger.warning(f"Low memory: {memory_usage:.1%}")
logger.error(f"Failed to load: {path}", exc_info=True)
```

---

### 5.3 Phase Transition Logging

Use visual separators for major phases:

```python
# integration/unified_pipeline.py
logger.info("=" * 60)
logger.info("Starting Unified ML Pipeline")
logger.info("=" * 60)

logger.info("\n[Phase 0] Generating synthetic dataset...")
# ... phase logic
logger.info(f"✓ Generated {count} signals")
logger.info(f"✗ Phase 0 failed: {error}")  # On failure
```

**Icons:**

- `✓` — Success
- `✗` — Failure
- `⚠️` — Warning

---

### 5.4 System Information Logging

```python
# utils/logging.py
def log_system_info() -> None:
    """Log system information for reproducibility."""
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    # ... GPU details
```

**Call at pipeline start** for experiment reproducibility.

---

## 6. Quick Reference

| Pattern                    | Location                        | Purpose               |
| -------------------------- | ------------------------------- | --------------------- |
| Adapter with callbacks     | `integrations/*.py`             | Dashboard ↔ Core ML   |
| Config dict interface      | All adapters                    | Standardized inputs   |
| Result dict with `success` | All adapters                    | Standardized outputs  |
| Constants centralization   | `utils/constants.py`            | No magic numbers      |
| Structured logging         | `utils/logging.py`              | Consistent log format |
| Context managers           | `utils/device_manager.py`       | Resource cleanup      |
| Model registry             | `integration/model_registry.py` | Track all models      |

---

_Best practices extracted from IDB 6.0 Integration Layer_
