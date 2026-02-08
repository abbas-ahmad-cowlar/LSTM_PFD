# IDB 4.4: Configuration Sub-Block — Best Practices

**Extracted:** 2026-01-23  
**Source:** [IDB_4_4_CONFIG_ANALYSIS.md](file:///c:/Users/COWLAR/projects/LSTM_PFD/docs/idb_reports/IDB_4_4_CONFIG_ANALYSIS.md)

---

## 1. Configuration Structure Patterns

### 1.1 Use Dataclass-Based Configuration ✅

**Pattern:** Define all configs as `@dataclass` classes inheriting from `BaseConfig`.

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any
from config.base_config import BaseConfig

@dataclass
class MyConfig(BaseConfig):
    """Configuration with clear docstring and example usage."""

    # Simple fields with defaults
    learning_rate: float = 0.001
    batch_size: int = 32

    # Complex fields use field(default_factory=...)
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])

    def get_schema(self) -> Dict[str, Any]:
        return {...}
```

**Benefits:**

- Type hints provide IDE autocomplete and static analysis
- Immutable-friendly design
- Automatic `__init__`, `__repr__`, `__eq__`

### 1.2 Hierarchical Composition Pattern ✅

**Pattern:** Use nested dataclasses for complex configurations.

```python
@dataclass
class OptimizerConfig(BaseConfig):
    name: str = 'adamw'
    lr: float = 0.001

@dataclass
class SchedulerConfig(BaseConfig):
    name: str = 'cosine'
    T_max: int = 100

@dataclass
class TrainingConfig(BaseConfig):
    """Master config aggregating sub-configs."""
    num_epochs: int = 100
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
```

### 1.3 Factory Pattern for Model Selection ✅

**Pattern:** Use a selector field with `get_active_config()` method.

```python
@dataclass
class ModelConfig(BaseConfig):
    model_type: str = 'cnn1d'  # Selector

    # All possible configs
    cnn1d: CNN1DConfig = field(default_factory=CNN1DConfig)
    resnet1d: ResNet1DConfig = field(default_factory=ResNet1DConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)

    def get_active_config(self) -> BaseConfig:
        """Return config for currently selected model type."""
        config_map = {
            'cnn1d': self.cnn1d,
            'resnet1d': self.resnet1d,
            'transformer': self.transformer,
        }
        if self.model_type not in config_map:
            raise ValueError(f"Unknown model type: {self.model_type}")
        return config_map[self.model_type]
```

---

## 2. Validation Patterns

### 2.1 JSON Schema Validation ✅

**Pattern:** Every config class implements `get_schema()` returning a JSON Schema.

```python
def get_schema(self) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string", "enum": ["adam", "adamw", "sgd"]},
            "lr": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "batch_size": {"type": "integer", "minimum": 1, "maximum": 1024},
        },
        "required": ["name", "lr"]
    }
```

**Common Schema Patterns:**

| Type               | Schema                                                 |
| ------------------ | ------------------------------------------------------ |
| Enum               | `{"type": "string", "enum": ["a", "b", "c"]}`          |
| Bounded number     | `{"type": "number", "minimum": 0, "maximum": 1}`       |
| Positive integer   | `{"type": "integer", "minimum": 1}`                    |
| String with length | `{"type": "string", "minLength": 1, "maxLength": 255}` |
| Array of integers  | `{"type": "array", "items": {"type": "integer"}}`      |

### 2.2 Post-Init Validation (Recommended) ⚠️

**Pattern:** Use `__post_init__` for cross-field validation.

```python
@dataclass
class CNN1DConfig(BaseConfig):
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_sizes: List[int] = field(default_factory=lambda: [15, 11, 7])

    def __post_init__(self):
        if len(self.conv_channels) != len(self.kernel_sizes):
            raise ValueError(
                f"conv_channels ({len(self.conv_channels)}) and "
                f"kernel_sizes ({len(self.kernel_sizes)}) must have same length"
            )
```

### 2.3 Validation Utilities ⚠️

**Pattern:** Centralized validation helpers (currently unused but available).

```python
from config.base_config import ConfigValidator

# Validate positive values
ConfigValidator.validate_positive(learning_rate, "learning_rate")

# Validate range
ConfigValidator.validate_range(dropout, 0.0, 1.0, "dropout")

# Validate probability
ConfigValidator.validate_probability(noise_level, "noise_level")
```

---

## 3. Environment Variable Conventions

> [!WARNING]
> Current codebase has **zero environment variable support**. The following are **recommended patterns** to adopt.

### 3.1 Environment Variable with Fallback (Recommended)

```python
import os
from dataclasses import dataclass, field

@dataclass
class MLflowConfig(BaseConfig):
    tracking_uri: str = field(
        default_factory=lambda: os.getenv('MLFLOW_TRACKING_URI', './mlruns')
    )
    experiment_name: str = field(
        default_factory=lambda: os.getenv('MLFLOW_EXPERIMENT', 'default_experiment')
    )
```

### 3.2 Naming Conventions (Recommended)

| Config Field     | Environment Variable      |
| ---------------- | ------------------------- |
| `tracking_uri`   | `MLFLOW_TRACKING_URI`     |
| `device`         | `LSTM_PFD_DEVICE`         |
| `checkpoint_dir` | `LSTM_PFD_CHECKPOINT_DIR` |
| `seed`           | `LSTM_PFD_SEED`           |

**Prefix all project-specific env vars with `LSTM_PFD_`.**

### 3.3 Sensitive Values Pattern (Recommended)

```python
@dataclass
class DatabaseConfig(BaseConfig):
    # Never hardcode secrets!
    connection_string: str = field(
        default_factory=lambda: os.environ['DATABASE_URL']  # Required, no fallback
    )

    def __post_init__(self):
        if not self.connection_string:
            raise ValueError("DATABASE_URL environment variable is required")
```

---

## 4. Default Value Conventions

### 4.1 Import Constants from Centralized Location ✅

```python
# config/data_config.py
from utils.constants import SAMPLING_RATE, SIGNAL_DURATION

@dataclass
class SignalConfig(BaseConfig):
    fs: int = SAMPLING_RATE        # 20480
    T: float = SIGNAL_DURATION     # 5.0
```

**Constants file:** [utils/constants.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/utils/constants.py) (629 lines)

### 4.2 Default Value Guidelines

| Field Type   | Convention                                 | Example                                    |
| ------------ | ------------------------------------------ | ------------------------------------------ |
| **Paths**    | Relative to project root                   | `checkpoint_dir='checkpoints'`             |
| **Device**   | Prefer auto-detection                      | `device='auto'` (then detect cuda/mps/cpu) |
| **Seeds**    | Consistent default                         | `seed=42` (project-wide)                   |
| **Booleans** | Safe default                               | `enabled=False` for experimental features  |
| **Lists**    | Use `field(default_factory=lambda: [...])` | Never use mutable default                  |

### 4.3 Avoiding Mutable Default Antipattern ✅

```python
# ❌ WRONG - Mutable default shared across instances
@dataclass
class BadConfig:
    layers: List[int] = [32, 64]  # DANGER!

# ✅ CORRECT - Factory creates new list per instance
@dataclass
class GoodConfig(BaseConfig):
    layers: List[int] = field(default_factory=lambda: [32, 64])
```

---

## 5. Documentation Requirements

### 5.1 Class-Level Docstring ✅

```python
@dataclass
class TransformerConfig(BaseConfig):
    """
    Configuration for Transformer model for time series.

    Architecture:
        Embedding -> Positional Encoding -> Transformer Encoder -> FC

    Example:
        >>> config = TransformerConfig(
        ...     d_model=256,
        ...     nhead=8,
        ...     num_layers=6
        ... )
    """
```

**Required Sections:**

1. Brief description
2. Architecture/purpose (if applicable)
3. Example usage with realistic values

### 5.2 Module-Level Docstring ✅

```python
"""
Model architecture configurations for deep learning models.

Purpose:
    Configuration classes for all model architectures:
    - CNN-1D models
    - ResNet-1D models
    - Transformer models

Author: Syed Abbas Ahmad
Date: 2025-11-19
"""
```

### 5.3 Field Documentation via Comments

```python
@dataclass
class OptimizerConfig(BaseConfig):
    # Optimizer type
    name: str = 'adamw'  # 'adam', 'adamw', 'sgd', 'rmsprop'

    # Learning rate
    lr: float = 0.001

    # Adam/AdamW parameters
    betas: tuple = (0.9, 0.999)  # (beta1, beta2)
    eps: float = 1e-8            # Numerical stability
```

### 5.4 Type Package Marker ✅

Include `py.typed` marker file for PEP 561 compliance:

```
config/
├── __init__.py
├── base_config.py
├── py.typed          # ← Empty file, enables type checking
└── ...
```

---

## 6. Serialization Patterns

### 6.1 YAML Load/Save ✅

```python
# Load from YAML
config = TrainingConfig.from_yaml(Path('configs/training.yaml'))

# Save to YAML
config.to_yaml(Path('configs/training_export.yaml'))
```

### 6.2 Dictionary Conversion ✅

```python
# Convert to dict (for logging, MLflow params)
config_dict = config.to_dict()

# Merge multiple configs (later overrides earlier)
merged = TrainingConfig.merge_configs(base_config, override_config)
```

---

## 7. Quick Reference Checklist

When creating a new configuration class:

- [ ] Inherit from `BaseConfig`
- [ ] Use `@dataclass` decorator
- [ ] Implement `get_schema()` with complete JSON Schema
- [ ] Use `field(default_factory=...)` for mutable defaults
- [ ] Import constants from `utils/constants.py`
- [ ] Add class docstring with example
- [ ] Add to `config/__init__.py` exports
- [ ] Add cross-field validation in `__post_init__` if needed
- [ ] Consider environment variable support for deployment values

---

## Appendix: File Templates

### Minimal Config Class

```python
from dataclasses import dataclass
from typing import Dict, Any
from config.base_config import BaseConfig

@dataclass
class NewFeatureConfig(BaseConfig):
    """Configuration for new feature."""

    enabled: bool = False
    threshold: float = 0.5

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "threshold": {"type": "number", "minimum": 0, "maximum": 1}
            }
        }
```

### Config with Nested Sub-Config

```python
@dataclass
class ParentConfig(BaseConfig):
    """Parent config with nested child."""

    name: str = 'default'
    child: NewFeatureConfig = field(default_factory=NewFeatureConfig)

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 1}
            }
        }
```
