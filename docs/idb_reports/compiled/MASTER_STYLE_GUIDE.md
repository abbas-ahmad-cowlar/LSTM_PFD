# Master Style Guide — LSTM_PFD Project

**Compilation Date:** 2026-01-24  
**Source Reports:** 5 Domain Consolidated Analyses  
**Coverage:** ~43,000 LOC across 200+ Python files

---

## Purpose

Unified code standards across all domains for **consistency**, **maintainability**, and **onboarding**. This document distills best practices observed across Core ML, Dashboard, Data Engineering, Infrastructure, and Research domains.

---

## 1. Python Code Conventions

### Naming Conventions

| Element       | Pattern               | Examples                                                 | Notes                               |
| ------------- | --------------------- | -------------------------------------------------------- | ----------------------------------- |
| **Classes**   | `PascalCase`          | `SignalGenerator`, `CNNTrainer`, `APIKey`                | All 5 domains consistent            |
| **Functions** | `snake_case`          | `create_model()`, `extract_features()`, `run_ablation()` | Prefer `verb_noun()` pattern        |
| **Variables** | `snake_case`          | `train_loader`, `best_accuracy`                          | Descriptive names                   |
| **Constants** | `UPPER_SNAKE_CASE`    | `NUM_CLASSES`, `SIGNAL_LENGTH`, `SAMPLING_RATE`          | Centralized in `utils/constants.py` |
| **Private**   | `_leading_underscore` | `_initialize_weights()`, `_process_batch()`              | Internal methods                    |
| **Files**     | `snake_case.py`       | `signal_generator.py`, `model_factory.py`                | All 5 domains consistent            |
| **DB Tables** | `snake_case`          | `api_keys`, `training_runs`                              | Infrastructure domain               |
| **Configs**   | `<Component>Config`   | `TrainingConfig`, `ModelConfig`                          | Infrastructure domain               |

**Dashboard-Specific Naming:**

- **Component IDs:** `{page}-{component}-{element}` (e.g., `settings-api-key-table`)
- **Callbacks:** `register_*_callbacks(app)` pattern
- **Services:** `*_service.py` suffix (e.g., `notification_service.py`)

### Import Organization

**Standard Order (All Domains):**

```python
# 1. Standard library
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# 2. Third-party
import numpy as np
import torch
import torch.nn as nn

# 3. Project-level (absolute imports)
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
from config.training_config import TrainingConfig

# 4. Local (relative imports within package)
from .base_model import BaseModel
from .model_factory import create_model
```

**Optional Dependency Pattern (Observed in XAI, Research, Integration):**

```python
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("SHAP not installed. Using native implementation.")
```

### Type Hints

**Usage:** ~85% of public methods across all domains

**Standard Format:**

```python
def create_model(name: str, num_classes: int = NUM_CLASSES, **kwargs) -> BaseModel:
    """Create a model instance by name."""
    ...

def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
    """Evaluate model on dataloader."""
    ...
```

**Dataclass Fields:** 100% typed (Infrastructure Domain 4.4)

```python
@dataclass
class TrainingConfig:
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: str = "cuda"
```

### Docstrings

**Format:** Google-style (most common across all domains)

```python
def feature_extractor(signal: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    """Extract features from signal.

    Args:
        signal: Input signal array of shape (T,) or (C, T)
        fs: Sampling frequency in Hz

    Returns:
        Feature vector of shape (num_features,)

    Raises:
        ValueError: If signal contains NaN or Inf values

    Example:
        >>> features = feature_extractor(signal, fs=20480)
        >>> print(features.shape)
        (52,)
    """
```

**For Scientific Methods (XAI, Features):** Include formulas and references

```python
def compute_spectral_entropy(signal: np.ndarray) -> float:
    """Compute spectral entropy of signal.

    Formula: H = -sum(p * log(p)) where p is normalized PSD

    Reference:
        Crepeau, J.C., & Isaacson, L.K. (1991). Spectral entropy measures.
    """
```

---

## 2. Architecture Patterns

### Model Patterns

**1. Abstract Base Class Pattern (Domain 1 — Core ML)**

```python
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        ...

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for checkpointing."""
        ...

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters."""
        ...
```

**2. Factory Pattern with Registry (Domain 1, 4)**

```python
MODEL_REGISTRY = {
    'cnn1d': create_cnn1d,
    'resnet18': create_resnet18_1d,
    'transformer': create_vision_transformer_1d,
}

def create_model(name: str, **kwargs) -> BaseModel:
    """Factory function with case-insensitive lookup."""
    if name.lower() not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return MODEL_REGISTRY[name.lower()](**kwargs)
```

**3. Preset Configurations (Domain 1)**

```python
def cnn_transformer_small(num_classes: int = NUM_CLASSES, **kwargs):
    """Small preset for quick experiments."""
    return create_cnn_transformer_hybrid(d_model=256, num_heads=4, **kwargs)

def cnn_transformer_base(num_classes: int = NUM_CLASSES, **kwargs):
    """Base preset for production."""
    return create_cnn_transformer_hybrid(d_model=512, num_heads=8, **kwargs)
```

### Service Patterns

**1. Context Manager for Resources (Domain 2, 3)**

```python
from contextlib import contextmanager

@contextmanager
def get_db_session():
    """Database session with automatic commit/rollback."""
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

# Usage
with get_db_session() as session:
    session.query(Model).filter_by(...).first()
```

**2. Cache-Aside Pattern (Domain 2, 3)**

```python
def get_cached(key: str, compute_fn: Callable) -> Any:
    """Get from cache or compute and cache."""
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = compute_fn()
    cache.set(key, result, ttl=3600)
    return result
```

**3. Factory Pattern for Providers (Domain 2)**

```python
class NotificationProviderFactory:
    _providers = {
        'email': EmailProvider,
        'slack': SlackProvider,
        'teams': TeamsProvider,
    }

    @classmethod
    def create(cls, provider_type: str, **kwargs) -> NotificationProvider:
        return cls._providers[provider_type](**kwargs)
```

### Callback Patterns

**1. Dash Callback Registration (Domain 2)**

```python
def register_settings_callbacks(app: Dash):
    """Register all settings-related callbacks."""

    @app.callback(
        Output('settings-output', 'children'),
        Input('settings-submit', 'n_clicks'),
        State('settings-input', 'value'),
        prevent_initial_call=True
    )
    def handle_settings_submit(n_clicks, value):
        ...
```

**2. Adapter Pattern with Callbacks (Domain 6)**

```python
class DeepLearningAdapter:
    @staticmethod
    def train(
        config: Dict[str, Any],
        progress_callback: Optional[Callable[[int, Dict], None]] = None
    ) -> Dict[str, Any]:
        """Train with optional progress reporting."""
        for epoch in range(num_epochs):
            metrics = train_epoch(...)
            if progress_callback:
                progress_callback(epoch, metrics)
        return {"success": True, "test_accuracy": best_acc}
```

---

## 3. Testing Standards

### Test Organization

```
tests/
├── conftest.py                    # Shared fixtures (session/module scope)
├── unit/
│   ├── core/
│   │   ├── test_models.py         # Mirror source structure
│   │   ├── test_trainers.py
│   │   └── test_evaluators.py
│   ├── dashboard/
│   │   ├── test_callbacks.py      # ⚠️ Currently missing (P0)
│   │   └── test_services.py
│   └── data/
│       ├── test_signal_generator.py
│       └── test_datasets.py
├── integration/
│   ├── test_comprehensive.py      # End-to-end pipelines
│   └── test_pipelines.py
├── benchmarks/
│   └── test_performance.py
└── stress_tests.py                # Memory/concurrency tests
```

### Fixture Patterns

| Scope      | Use Case                          | Example                            |
| ---------- | --------------------------------- | ---------------------------------- |
| `session`  | Device detection, expensive setup | `@pytest.fixture(scope="session")` |
| `module`   | Model creation, shared data       | Datasets shared across test class  |
| `function` | Mutable state, test isolation     | Fresh model instance per test      |

```python
# conftest.py
@pytest.fixture(scope="session")
def device():
    """Single CUDA/CPU detection per test session."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="module")
def sample_dataset():
    """Shared dataset for module tests."""
    return BearingFaultDataset.from_hdf5(CACHE_PATH)

@pytest.fixture
def temp_dir():
    """Function-scoped temp directory with cleanup."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)
```

### Mock Patterns

**When to Mock:**

- External services (Redis, database, APIs)
- File I/O in unit tests
- GPU operations in CI without GPU

**Preferred Libraries:** `pytest-mock`, `unittest.mock`

```python
def test_cache_service(mocker):
    mock_redis = mocker.patch('services.cache_service.redis_client')
    mock_redis.get.return_value = None

    result = cache_service.get('key')

    mock_redis.get.assert_called_once_with('key')
```

### Coverage Requirements

| Domain          | Minimum | Current Gap                     |
| --------------- | ------- | ------------------------------- |
| Core ML (1.x)   | 85%     | In-module `__main__` tests only |
| Dashboard (2.x) | 70%     | **Callbacks and Tasks at 0%**   |
| Data (3.x)      | 80%     | Cache/import/validator untested |
| Infrastructure  | 75%     | Adequate                        |
| Research        | 60%     | Script-level testing only       |

---

## 4. Interface Contract Standards

### API Design Principles

1. **Input:** Accept dictionaries or dataclasses with explicit validation
2. **Output:** Return structured types (dataclasses, TypedDicts, or dicts with documented schema)
3. **Errors:** Raise domain-specific exceptions, not generic `Exception`

### Model Interface Contract

| Method               | Input              | Output                                | Notes               |
| -------------------- | ------------------ | ------------------------------------- | ------------------- |
| `forward(x)`         | `[B, C, T]` tensor | `[B, num_classes]` logits             | NOT softmax         |
| `get_config()`       | None               | Dict with `model_type`, `num_classes` | For checkpointing   |
| `count_parameters()` | None               | Dict with `total`, `trainable`        | Inherited from base |

### Trainer Interface Contract

| Method             | Signature                   | Notes                       |
| ------------------ | --------------------------- | --------------------------- |
| `fit(num_epochs)`  | `-> Dict[str, List[float]]` | Returns training history    |
| `train_epoch()`    | `-> Dict[str, float]`       | `train_loss`, `train_acc`   |
| `validate_epoch()` | `-> Dict[str, float]`       | `val_loss`, `val_acc`, `lr` |

### Service Return Contract

```python
# Standard service result pattern (Domain 2)
@dataclass
class ServiceResult:
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None

# Celery task return pattern (Domain 2)
def task_result(success: bool, **kwargs) -> Dict:
    return {"success": success, "error": None, **kwargs}
```

### Configuration Contract

```python
# All configs must support (Domain 4)
config = TrainingConfig.from_yaml(Path('config.yaml'))
config.validate()    # JSON Schema validation
config.to_dict()     # For logging
config.to_yaml()     # For saving
```

### Error Handling

```python
# Domain-specific exceptions
class ModelNotFoundError(ValueError):
    """Raised when model name not in registry."""
    pass

class CacheCorruptionError(RuntimeError):
    """Raised when cache file is corrupted."""
    pass

# Usage
try:
    model = create_model(name)
except ModelNotFoundError:
    logger.error(f"Model '{name}' not found in registry")
    raise
```

---

## 5. Documentation Standards

### Code Comments

**When to Comment:**

- Complex algorithms (physics formulas, ML techniques)
- Non-obvious design decisions
- Performance-critical code with specific optimizations
- TODO/FIXME with ticket numbers

**When NOT to Comment:**

- Self-explanatory code
- Obvious variable assignments
- Standard library usage

### Module Docstrings

**Required in every module:**

```python
"""Signal generation module for bearing fault diagnosis.

This module provides physics-based synthetic vibration signal generation
with 11 fault classes and 8-layer noise modeling.

Classes:
    SignalGenerator: Main orchestrator for dataset generation
    FaultModeler: Implements fault signature patterns
    NoiseGenerator: Multi-layer noise synthesis

Usage:
    from data.signal_generator import SignalGenerator
    generator = SignalGenerator(config)
    dataset = generator.generate_dataset()
"""
```

### README Requirements

Each package directory should have a `README.md` with:

1. **Purpose** — What this package/module does
2. **Quick Start** — Minimal usage example
3. **API Reference** — Link to docstrings or generated docs
4. **Dependencies** — Required packages
5. **Testing** — How to run tests

---

## 6. Domain-Specific Conventions

### Core ML (Domain 1)

- All models inherit from `BaseModel`
- Use `model_factory` for instantiation
- Model configs use dataclasses with presets (`*_small`, `*_base`, `*_large`)
- All models must support ONNX export
- Division guards for numerical stability: `return x / y if y > 0 else 0.0`

### Dashboard (Domain 2)

- Component IDs: `{page}-{component}-{action}`
- Callbacks use registration pattern: `register_*_callbacks(app)`
- Services are stateless (no global state)
- Database access only through service layer
- Celery tasks return `{"success": bool, ...}` dict

### Data Engineering (Domain 3)

- Dataset classes inherit from `torch.utils.data.Dataset`
- Factory methods: `from_hdf5()`, `from_mat_file()`, `from_generator_output()`
- Thread-local HDF5 handles for multi-worker DataLoader safety
- Config-hash caching for automatic invalidation
- Graceful Redis fallback (no crash on cache failure)

### Infrastructure (Domain 4)

- Constants centralized in `utils/constants.py` (629 LOC)
- `BaseModel` (ORM) provides `id`, `created_at`, `updated_at`, `to_dict()`
- JSON Schema validation for all config classes
- Environment variable support for secrets
- Multi-stage Docker builds with non-root user

### Research (Domain 5)

- All scripts have CLI via `argparse`
- `--quick` flag for development mode
- Multi-seed experiments with mean±std reporting
- Timestamped output directories
- Publication-quality: DPI=300, `viridis` colormap

---

## 7. Anti-Patterns to Avoid

### 1. God Classes

**Problem:** `settings.py` at 42KB (1,005 lines), `NotificationService` at 713 lines  
**Solution:** Split into focused modules (max 500 lines)

```
# Before: layouts/settings.py (1,005 lines)
# After:
layouts/settings/
├── __init__.py
├── api_keys.py
├── profile.py
├── security.py
├── notifications.py
└── webhooks.py
```

### 2. Circular Dependencies

**Problem:** Dashboard callbacks import from both UI layouts and backend services  
**Solution:** Use dependency injection, service layer abstraction

### 3. Hardcoded Magic Numbers

**Problem:** `102400`, `20480`, `5000` scattered across files  
**Solution:** Import from `utils/constants.py`

```python
# Bad
signal_length = 102400

# Good
from utils.constants import SIGNAL_LENGTH
signal_length = SIGNAL_LENGTH
```

### 4. `sys.path` Manipulation

**Problem:** `sys.path.append('/home/user/LSTM_PFD')` in 8+ files  
**Solution:** Proper package structure with relative imports

```python
# Bad
import sys
sys.path.insert(0, '/home/user/LSTM_PFD')
from packages.core.models import BaseModel

# Good
from packages.core.models import BaseModel
```

### 5. Bare `except:` Clauses

**Problem:** 70+ occurrences swallowing all exceptions  
**Solution:** Catch specific exceptions

```python
# Bad
try:
    result = risky_operation()
except:
    pass

# Good
try:
    result = risky_operation()
except (ValueError, IOError) as e:
    logger.error(f"Operation failed: {e}")
    raise
```

### 6. In-Memory State in Multi-Process Environments

**Problem:** 2FA rate limiting uses in-memory dict  
**Solution:** Use Redis for shared state

### 7. Silent Fallbacks

**Problem:** `HybridPINN` import fails silently, falls back to `ResNet18`  
**Solution:** Fail-fast or log explicit warning

```python
# Bad
try:
    from models.pinn import HybridPINN
except ImportError:
    HybridPINN = ResNet18  # Silent substitution

# Good
try:
    from models.pinn import HybridPINN
except ImportError:
    warnings.warn("HybridPINN not available, using ResNet18 fallback")
    HybridPINN = ResNet18
```

### 8. Duplicate Implementations

**Problem:** `FocalLoss`, `Compose`, `PositionalEncoding` defined 2-3 times  
**Solution:** Single canonical implementation, import everywhere else

### 9. Empty `__init__.py` Files

**Problem:** No public API defined, import confusion  
**Solution:** Explicit `__all__` exports

```python
# __init__.py
from .trainer import Trainer
from .callbacks import EarlyStopping, ModelCheckpoint

__all__ = ['Trainer', 'EarlyStopping', 'ModelCheckpoint']
```

### 10. Inconsistent Colormap (`jet`)

**Problem:** `jet` colormap not colorblind-accessible  
**Solution:** Use `viridis` (perceptually uniform)

---

## 8. Onboarding Checklist

Before contributing to this codebase:

- [ ] Read this Master Style Guide
- [ ] Read [EXECUTIVE_DASHBOARD.md](./EXECUTIVE_DASHBOARD.md) for project overview
- [ ] Read relevant domain consolidated report:
  - [ ] [DOMAIN_1_CORE_ML_CONSOLIDATED.md](./DOMAIN_1_CORE_ML_CONSOLIDATED.md)
  - [ ] [DOMAIN_2_DASHBOARD_CONSOLIDATED.md](./DOMAIN_2_DASHBOARD_CONSOLIDATED.md)
  - [ ] [DOMAIN_3_DATA_CONSOLIDATED.md](./DOMAIN_3_DATA_CONSOLIDATED.md)
  - [ ] [DOMAIN_4_INFRASTRUCTURE_CONSOLIDATED.md](./DOMAIN_4_INFRASTRUCTURE_CONSOLIDATED.md)
  - [ ] [DOMAIN_5_RESEARCH_INTEGRATION_CONSOLIDATED.md](./DOMAIN_5_RESEARCH_INTEGRATION_CONSOLIDATED.md)
- [ ] Review [INTERFACE_CONTRACTS_CATALOG.md](./INTERFACE_CONTRACTS_CATALOG.md) (if available)
- [ ] Review [CRITICAL_ISSUES_MATRIX.md](./CRITICAL_ISSUES_MATRIX.md) for known issues
- [ ] Run existing tests to verify setup: `pytest tests/ -v`
- [ ] Set up pre-commit hooks for linting

---

## Appendix: Quick Reference

### Import Template

```python
"""Module docstring."""

# Standard library
import os
from pathlib import Path
from typing import Dict, List, Optional

# Third-party
import numpy as np
import torch

# Project
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
from config import TrainingConfig

# Local
from .base import BaseClass
```

### Dataclass Config Template

```python
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ComponentConfig:
    """Configuration for component.

    Attributes:
        param1: Description of param1
        param2: Description of param2
    """
    param1: int = 100
    param2: float = 1e-3
    optional_list: List[int] = field(default_factory=list)

    def validate(self) -> bool:
        """Validate configuration."""
        ...

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
```

### Test Template

```python
"""Tests for module_name."""

import pytest
from utils.constants import NUM_CLASSES

from module_name import TargetClass


class TestTargetClass:
    """Test suite for TargetClass."""

    @pytest.fixture
    def instance(self):
        """Create instance for testing."""
        return TargetClass()

    def test_initialization(self, instance):
        """Test default initialization."""
        assert instance is not None

    def test_method_with_valid_input(self, instance):
        """Test method with valid input."""
        result = instance.method(valid_input)
        assert result == expected

    @pytest.mark.parametrize("invalid_input", [None, "", -1])
    def test_method_with_invalid_input(self, instance, invalid_input):
        """Test method raises on invalid input."""
        with pytest.raises(ValueError):
            instance.method(invalid_input)
```

---

_Master Style Guide compiled from 5 Domain Consolidated Reports — 2026-01-24_
