# IDB 1.1: Models Sub-Block Best Practices

**IDB ID:** 1.1  
**Domain:** Core ML Engine  
**Scope:** `packages/core/models/`  
**Purpose:** Onboarding guide, cross-team consistency reference, and style guide contribution  
**Last Updated:** 2026-01-23

---

## Table of Contents

1. [Patterns Worth Preserving](#1-patterns-worth-preserving)
2. [Code Style & Conventions](#2-code-style--conventions)
3. [Interface Contracts](#3-interface-contracts)
4. [Testing Patterns](#4-testing-patterns)
5. [Future Developer Recommendations](#5-future-developer-recommendations)
6. [Cross-Team Coordination Notes](#6-cross-team-coordination-notes)

---

## 1. Patterns Worth Preserving

### 1.1 Model Registration via Factory

**Pattern:** Centralized model registry with factory functions.

```python
# MODEL_REGISTRY maps string names to creation functions
MODEL_REGISTRY = {
    'cnn1d': create_cnn1d,
    'cnn_1d': create_cnn1d,  # Alias for convenience
    'resnet18': create_resnet18_1d,
    'resnet18_1d': create_resnet18_1d,
    # ...
}

def register_model(name: str, creation_fn: callable):
    """Register a new model type."""
    MODEL_REGISTRY[name.lower()] = creation_fn
```

**Why This Works:**

- Case-insensitive lookup (`name.lower()`)
- Supports aliases (same model, multiple names)
- Dynamic registration for extensions

**When Adding a New Model:**

```python
# Step 1: Create your model class
class MyNewModel(BaseModel):
    ...

# Step 2: Create factory function
def create_my_new_model(num_classes: int = NUM_CLASSES, **kwargs) -> MyNewModel:
    return MyNewModel(num_classes=num_classes, **kwargs)

# Step 3: Register in MODEL_REGISTRY
MODEL_REGISTRY['my_new_model'] = create_my_new_model
```

---

### 1.2 Preset Configuration Pattern

**Pattern:** Named presets for common configurations.

```python
def cnn_transformer_small(num_classes: int = NUM_CLASSES, **kwargs):
    """Small CNN-Transformer hybrid (fast, ~10M params)."""
    return create_cnn_transformer_hybrid(
        num_classes=num_classes,
        d_model=256,
        num_heads=4,
        num_layers=2,
        **kwargs
    )

def cnn_transformer_base(num_classes: int = NUM_CLASSES, **kwargs):
    """Base CNN-Transformer hybrid (recommended, ~15M params)."""
    return create_cnn_transformer_hybrid(
        num_classes=num_classes,
        d_model=512,
        num_heads=8,
        num_layers=4,
        **kwargs
    )
```

**Why This Works:**

- Clear performance/size trade-offs
- Consistent naming: `{model}_{size}` (small, base, large, tiny)
- All presets still accept `**kwargs` for fine-tuning

---

### 1.3 Configuration via get_config()

**Pattern:** Every model returns its hyperparameters as a dictionary.

```python
def get_config(self) -> dict:
    """Return model configuration."""
    return {
        'model_type': 'CNN1D',
        'num_classes': self.num_classes,
        'input_channels': self.input_channels,
        'num_parameters': self.get_num_params()
    }
```

**Required Keys:**
| Key | Type | Description |
|-----|------|-------------|
| `model_type` | `str` | Human-readable model name |
| `num_classes` | `int` | Number of output classes |
| `num_parameters` | `int` | Total parameter count |

**Optional Keys:**
| Key | Type | Description |
|-----|------|-------------|
| `input_channels` | `int` | Input channel count |
| `dropout` | `float` | Dropout probability |
| `backbone` | `str` | For hybrid models |

---

### 1.4 Checkpoint Structure

**Pattern:** Standardized checkpoint dictionary format.

```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),  # Optional
    'epoch': 50,  # Optional
    'metrics': {'val_acc': 0.95, 'val_loss': 0.15},  # Optional
    'model_config': model.get_config(),  # Optional but recommended
}
```

**Loading Best Practice:**

```python
checkpoint = torch.load(path, map_location=device)

# Handle multiple formats
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint  # Raw state_dict
```

---

### 1.5 Weight Initialization Pattern

**Pattern:** He initialization for ReLU networks.

```python
def _initialize_weights(self):
    """Initialize weights using He initialization for ReLU networks."""
    for m in self.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)
```

**Call Order:** Always call `self._initialize_weights()` at the end of `__init__()`.

---

## 2. Code Style & Conventions

### 2.1 Naming Conventions

| Element           | Convention                         | Example                                      |
| ----------------- | ---------------------------------- | -------------------------------------------- |
| Model Classes     | `PascalCase`                       | `CNN1D`, `VisionTransformer1D`, `HybridPINN` |
| Factory Functions | `snake_case` with `create_` prefix | `create_cnn1d`, `create_vit_1d`              |
| Preset Functions  | `snake_case` with size suffix      | `vit_small_1d`, `cnn_transformer_base`       |
| Private Methods   | `_snake_case` prefix               | `_initialize_weights`, `_make_layer`         |
| Constants         | `UPPER_SNAKE_CASE`                 | `NUM_CLASSES`, `SIGNAL_LENGTH`               |
| Files             | `snake_case.py`                    | `cnn_1d.py`, `vision_transformer_1d.py`      |
| Directories       | `snake_case/`                      | `spectrogram_cnn/`, `efficientnet/`          |

### 2.2 Import Ordering

```python
"""
Module docstring with Purpose, Author, Date.
"""

# 1. Standard library imports
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json

# 2. Third-party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 3. Project absolute imports
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
from utils.logging import get_logger

# 4. Local relative imports
from .base_model import BaseModel
from .cnn_1d import CNN1D
```

### 2.3 Docstring Format (Google Style)

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of the model.

    Args:
        x: Input tensor of shape [B, C, T] where:
            - B: Batch size
            - C: Number of channels (typically 1)
            - T: Signal length

    Returns:
        Output tensor of shape [B, num_classes] containing logits

    Raises:
        ValueError: If input tensor has wrong number of dimensions

    Example:
        >>> model = CNN1D(num_classes=11)
        >>> x = torch.randn(32, 1, 102400)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 11])
    """
```

### 2.4 Type Hint Usage

**Required For:**

- Function signatures (all arguments and return types)
- Class attributes that are public

**Examples:**

```python
# Function with full type hints
def create_model(
    model_name: str,
    num_classes: int = NUM_CLASSES,
    **kwargs
) -> BaseModel:

# Use Optional for nullable parameters
def load_checkpoint(
    path: Path,
    device: Optional[torch.device] = None
) -> Tuple['BaseModel', Dict[str, Any]]:

# Use Union for multiple types (prefer Optional when one is None)
from typing import Union
def process(data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
```

### 2.5 Module Organization

Every model file should follow this structure:

```python
"""
Module docstring with:
- One-line description
- Architecture summary
- Input/Output shapes
- References (papers, sections)
- Author & Date
"""

# Imports (ordered as above)

# Logger instance
logger = get_logger(__name__)

# Helper classes (ConvBlock, AttentionModule, etc.)
class ConvBlock(nn.Module):
    ...

# Main model class
class MyModel(BaseModel):
    ...

# Factory function
def create_my_model(num_classes: int = NUM_CLASSES, **kwargs) -> MyModel:
    ...

# Preset configurations (if applicable)
def my_model_small(num_classes: int = NUM_CLASSES, **kwargs):
    ...

def my_model_base(num_classes: int = NUM_CLASSES, **kwargs):
    ...

# Test function (to be moved to tests/ in future)
if __name__ == '__main__':
    ...
```

---

## 3. Interface Contracts

### 3.1 BaseModel Abstract Interface

**MUST Implement:**

```python
class MyModel(BaseModel):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  x with shape [B, C, T] or [B, T]
        Output: logits with shape [B, num_classes]
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """Return dictionary with hyperparameters."""
        pass
```

**Inherited Methods (DO NOT Override Unless Necessary):**
| Method | Purpose |
|--------|---------|
| `count_parameters()` | Returns dict with total/trainable/non-trainable |
| `get_num_params()` | Returns int of total parameters |
| `get_model_size_mb()` | Returns float of model size in MB |
| `summary()` | Returns formatted string summary |
| `save_checkpoint()` | Saves model to disk |
| `freeze_backbone()` | Freezes all parameters |
| `unfreeze_backbone()` | Unfreezes all parameters |

### 3.2 Input/Output Tensor Contract

```python
# Standard input shape
x = torch.randn(batch_size, 1, SIGNAL_LENGTH)  # [B, C, T]
# Where:
#   B = batch_size (any positive int)
#   C = channels (typically 1 for raw vibration signal)
#   T = SIGNAL_LENGTH (102400 from constants)

# Standard output shape
output = model(x)  # [B, NUM_CLASSES]
# Where:
#   NUM_CLASSES = 11 (from constants)
#   output contains raw logits (NOT softmax/sigmoid)
```

**2D to 3D Conversion:**

```python
# Models SHOULD handle 2D input gracefully
if x.dim() == 2:
    x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]
```

### 3.3 Device Handling Convention

**Pattern:** Never hardcode device; accept it as parameter or infer from inputs.

```python
# GOOD: Infer device from input
device = x.device

# GOOD: Accept device parameter
def forward(self, x: torch.Tensor, device: str = 'cuda'):
    ...

# GOOD: Use next(parameters) to get current device
device = next(model.parameters()).device

# BAD: Hardcoded device
device = torch.device('cuda')  # Don't do this!
```

### 3.4 State Dict Structure

Standard PyTorch naming for layers:

```python
model.state_dict().keys()
# Should produce nested dot notation:
# 'conv1.conv.weight'
# 'conv1.conv.bias'
# 'conv1.bn.weight'
# 'conv1.bn.bias'
# 'fc1.weight'
# 'fc1.bias'
```

**DO NOT:**

- Use dynamic layer names based on config
- Use list indices in names (e.g., `layers.0.weight`)

---

## 4. Testing Patterns

### 4.1 In-Module Test Structure (Current)

```python
if __name__ == '__main__':
    from utils.constants import NUM_CLASSES, SIGNAL_LENGTH

    print("Testing MyModel...")

    # Test 1: Instantiation
    model = MyModel(num_classes=NUM_CLASSES)
    print(f"✓ Model created with {model.get_num_params():,} parameters")

    # Test 2: Forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, SIGNAL_LENGTH)
    output = model(x)
    assert output.shape == (batch_size, NUM_CLASSES)
    print(f"✓ Forward pass: {x.shape} -> {output.shape}")

    # Test 3: Configuration retrieval
    config = model.get_config()
    assert 'num_classes' in config
    print(f"✓ Config: {config}")

    # Test 4: Gradient flow
    loss = output.sum()
    loss.backward()
    print("✓ Backward pass successful")

    print("\n✅ All tests passed!")
```

### 4.2 Expected Test Coverage

| Test Category         | What to Test                                   |
| --------------------- | ---------------------------------------------- |
| **Instantiation**     | Model creates without errors with default args |
| **Forward Pass**      | Correct output shape for standard input        |
| **2D Input Handling** | Model handles [B, T] input by unsqueezing      |
| **Gradient Flow**     | `loss.backward()` completes without error      |
| **Config Retrieval**  | `get_config()` returns required keys           |
| **Device Transfer**   | `model.to('cuda')` works if GPU available      |
| **Checkpoint**        | Save and load produces identical outputs       |

### 4.3 Future Testing Pattern (Recommended)

Tests should be moved to `tests/test_models/` with pytest:

```python
# tests/test_models/test_cnn_1d.py
import pytest
import torch
from packages.core.models import CNN1D, create_cnn1d
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH

class TestCNN1D:
    @pytest.fixture
    def model(self):
        return create_cnn1d(num_classes=NUM_CLASSES)

    @pytest.fixture
    def sample_input(self):
        return torch.randn(4, 1, SIGNAL_LENGTH)

    def test_forward_shape(self, model, sample_input):
        output = model(sample_input)
        assert output.shape == (4, NUM_CLASSES)

    def test_config_contains_required_keys(self, model):
        config = model.get_config()
        assert 'model_type' in config
        assert 'num_classes' in config
```

---

## 5. Future Developer Recommendations

### 5.1 ALWAYS Do When Adding a New Model

| #   | Action                                                       | Rationale                             |
| --- | ------------------------------------------------------------ | ------------------------------------- |
| 1   | Inherit from `BaseModel`                                     | Required for interface consistency    |
| 2   | Implement `forward()` and `get_config()`                     | Abstract methods that must be defined |
| 3   | Use `NUM_CLASSES` and `SIGNAL_LENGTH` from `utils.constants` | Avoid magic numbers                   |
| 4   | Create factory function `create_xyz()`                       | Enables registry pattern              |
| 5   | Register in `MODEL_REGISTRY`                                 | Makes model discoverable              |
| 6   | Call `self._initialize_weights()` in `__init__`              | Consistent initialization             |
| 7   | Add type hints to all public methods                         | Code quality and IDE support          |
| 8   | Write docstring with Args/Returns/Example                    | Documentation consistency             |
| 9   | Handle 2D input by unsqueezing                               | Flexibility for callers               |
| 10  | Test with `if __name__ == '__main__':` block                 | Quick verification                    |

### 5.2 NEVER Do

| #   | Anti-Pattern                                            | Consequence                          |
| --- | ------------------------------------------------------- | ------------------------------------ |
| 1   | **Hardcode `sys.path.append()`**                        | Breaks on other machines             |
| 2   | **Hardcode device (`'cuda'`)**                          | Fails on CPU-only machines           |
| 3   | **Use magic numbers**                                   | Hard to maintain, inconsistent       |
| 4   | **Skip `get_config()`**                                 | Breaks checkpoint metadata           |
| 5   | **Return probabilities from `forward()`**               | Loss functions expect logits         |
| 6   | **Create duplicate model files**                        | Import confusion, maintenance burden |
| 7   | **Import with absolute paths containing `/home/user/`** | Machine-specific paths               |
| 8   | **Use `print()` for logging**                           | No log level control; use `logger`   |

### 5.3 Be CAREFUL About

| Area                        | Warning                                                  |
| --------------------------- | -------------------------------------------------------- |
| **Input shapes**            | Always document expected shape in docstring              |
| **Output interpretation**   | Clarify if returning logits, probabilities, or features  |
| **Memory in forward**       | Avoid creating tensors that aren't detached in inference |
| **Backward compatibility**  | New parameters should have defaults                      |
| **Ensemble models**         | Verify all sub-models are on same device                 |
| **Physics-informed models** | Metadata parameters may break standard training loops    |
| **Transfer learning**       | Use `freeze_backbone()` before fine-tuning               |

---

## 6. Cross-Team Coordination Notes

### 6.1 Training Sub-Block (IDB 1.2) Integration

**Models provide:**
| Interface | Description |
|-----------|-------------|
| `model.forward(x)` | Standard forward pass |
| `model.parameters()` | For optimizer construction |
| `model.train() / model.eval()` | Mode switching |
| `model.save_checkpoint()` | Checkpoint saving |

**Training expects:**

- Output shape: `[B, num_classes]`
- Output type: **Logits** (not softmax)
- Gradient-enabled parameters

**Coordination Required For:**

- Adding new loss functions (models may need to return additional outputs)
- Multi-task models (need to clarify which head to train)

### 6.2 Evaluation Sub-Block (IDB 1.3) Integration

**Evaluation expects:**
| Method | Expected Behavior |
|--------|-------------------|
| `model.eval()` | Disables dropout, batchnorm uses running stats |
| `model(x)` | Returns logits for accuracy calculation |
| `model.get_config()` | For logging experiment metadata |

**Models should NOT:**

- Apply softmax in forward() — evaluation computes metrics on logits

### 6.3 Dashboard (IDB 2.1) Integration

**Dashboard reads:**
| Interface | Usage |
|-----------|-------|
| `model.get_config()` | Display model hyperparameters |
| `model.get_num_params()` | Show parameter count |
| `model.get_model_size_mb()` | Display model size |
| `list_available_models()` | Populate model selector dropdown |

**Dashboard calls:**
| Interface | Usage |
|-----------|-------|
| `create_model()` | Train new model from UI |
| `load_pretrained()` | Load saved model for inference |

### 6.4 Changes Requiring Cross-Team Coordination

| Change                       | Affected Teams       | Coordination Steps                           |
| ---------------------------- | -------------------- | -------------------------------------------- |
| New model architecture       | Training, Dashboard  | Add to registry, update UI dropdown          |
| Changed forward() signature  | Training, Evaluation | Update training loop, metrics calculation    |
| New output head (multi-task) | Training, Evaluation | New loss function, new metrics               |
| Changed checkpoint format    | All                  | Version checkpoint, provide migration script |
| New BaseModel method         | All                  | Document, test backward compatibility        |
| Removed deprecated model     | Dashboard            | Update UI, remove from dropdown              |

---

## Appendix: Quick Reference Card

### New Model Checklist

```
□ Inherit from BaseModel
□ Implement forward() -> [B, num_classes] logits
□ Implement get_config() with model_type, num_classes, num_parameters
□ Create create_xyz() factory function
□ Add to MODEL_REGISTRY in model_factory.py
□ Use constants from utils.constants (NUM_CLASSES, SIGNAL_LENGTH)
□ Add _initialize_weights() using He/Kaiming
□ Write Google-style docstrings
□ Add type hints
□ Include if __name__ == '__main__' test block
□ Handle 2D input with unsqueeze
□ Export from __init__.py
```

### File Template

```python
"""
{ModelName} for Bearing Fault Diagnosis

Architecture:
    - ...

Input: [B, 1, {SIGNAL_LENGTH}]
Output: [B, {NUM_CLASSES}]

Author: {Your Name}
Date: {Date}
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import torch
import torch.nn as nn
from typing import Dict, Any

from .base_model import BaseModel


class {ModelName}(BaseModel):
    def __init__(self, num_classes: int = NUM_CLASSES, **kwargs):
        super().__init__()
        # ... layers ...
        self._initialize_weights()

    def _initialize_weights(self):
        # He initialization
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        # ... forward pass ...
        return logits

    def get_config(self) -> Dict[str, Any]:
        return {
            'model_type': '{ModelName}',
            'num_classes': self.num_classes,
            'num_parameters': self.get_num_params()
        }


def create_{model_name}(num_classes: int = NUM_CLASSES, **kwargs):
    return {ModelName}(num_classes=num_classes, **kwargs)


if __name__ == '__main__':
    model = create_{model_name}()
    x = torch.randn(4, 1, SIGNAL_LENGTH)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Config: {model.get_config()}")
    print("✅ Test passed!")
```

---

_Document generated by IDB 1.1 Best Practices Extractor_
