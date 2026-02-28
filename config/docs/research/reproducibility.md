# Reproducibility

This document covers how to ensure reproducible experiments in LSTM PFD.

## Random Seed Control

All randomness is controlled via centralized seed setting:

```python
from utils.reproducibility import set_seed

# Sets seed for: Python random, NumPy, PyTorch (CPU+CUDA)
set_seed(42)
```

### Implementation Details

| Library | Method                                      |
| ------- | ------------------------------------------- |
| Python  | `random.seed(42)`                           |
| NumPy   | `np.random.seed(42)`                        |
| PyTorch | `torch.manual_seed(42)`                     |
| CUDA    | `torch.cuda.manual_seed_all(42)`            |
| cuDNN   | `torch.backends.cudnn.deterministic = True` |

!!! warning "Performance Trade-off"
Setting `cudnn.deterministic = True` may reduce training speed by 10-20%.

---

## Configuration Management

All experiments are configured via YAML files:

```yaml
# config/experiment.yaml
model:
  name: resnet18
  num_classes: 11
  dropout: 0.3

training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001

data:
  path: data/processed/signals_cache.h5
  split_ratio: [0.7, 0.15, 0.15]
```

---

## Data Versioning (DVC)

For dataset versioning, we use DVC:

```bash
# Initialize DVC
dvc init

# Track datasets
dvc add data/processed/signals_cache.h5

# Push to remote storage
dvc push
```

---

## Experiment Tracking

Configuration files and results are automatically logged:

```
experiments/
├── 2026-01-13_resnet18/
│   ├── config.yaml
│   ├── metrics.json
│   ├── model.pth
│   └── logs/
```

---

## See Also

- [First Experiment](../getting-started/first-experiment.md)
- [PINN Theory](pinn-theory.md)
