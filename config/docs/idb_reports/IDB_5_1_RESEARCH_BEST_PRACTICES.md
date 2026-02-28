# IDB 5.1: Research Scripts Best Practices

**Extracted from:** `scripts/research/` (9 scripts)  
**Date:** 2026-01-23  
**Audience:** All teams writing experiment scripts

---

## 1. Experiment Script Patterns

### 1.1 Script Structure Template

```python
#!/usr/bin/env python3
"""
Script Title

Brief description of what the experiment does.

Usage:
    python scripts/research/script_name.py --data path/to/data.h5
    python scripts/research/script_name.py --quick  # Development mode

Author: Team/Person Name
Date: YYYY-MM-DD
"""

import sys
from pathlib import Path

# Add project root to path BEFORE other imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Standard library imports
import argparse
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

# Third-party imports (with optional checks)
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Project imports
from utils.reproducibility import set_seed
from utils.device_manager import get_device
from utils.logging import get_logger

logger = get_logger(__name__)

# Constants
OUTPUT_DIR = Path('results/script_name')


def main():
    args = parse_args()
    # ... implementation


if __name__ == '__main__':
    main()
```

### 1.2 Dataclass for Configuration

**Reference:** [ablation_study.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/scripts/research/ablation_study.py#L93-L123)

```python
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    description: str

    # Hyperparameters
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001

    # Flags
    use_augmentation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    config_name: str
    accuracy_mean: float
    accuracy_std: float
    f1_mean: float
    f1_std: float
    training_time_seconds: float
    seed_results: List[Dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
```

### 1.3 Graceful Optional Dependencies

**Reference:** [ablation_study.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/scripts/research/ablation_study.py#L45-L76)

```python
# Pattern for optional imports
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not installed. Some features will be limited.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Use in code
def plot_results(results):
    if not HAS_PLOTTING:
        logger.warning("Plotting not available, skipping visualization")
        return
    # ... plotting code
```

---

## 2. CLI Argument Conventions

### 2.1 Standard Argument Set

All research scripts **MUST** support these arguments:

| Argument       | Type | Default            | Description                      |
| -------------- | ---- | ------------------ | -------------------------------- |
| `--data`       | str  | required           | Path to HDF5 dataset             |
| `--epochs`     | int  | 30-50              | Training epochs                  |
| `--batch-size` | int  | 32                 | Batch size                       |
| `--output-dir` | str  | `results/{script}` | Output directory                 |
| `--seed`       | int  | 42                 | Random seed                      |
| `--device`     | str  | auto               | `cpu` or `cuda`                  |
| `--quick`      | flag | False              | Development mode (reduced scope) |

### 2.2 ArgumentParser Template

**Reference:** [transformer_benchmark.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/scripts/research/transformer_benchmark.py#L341-L365)

```python
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Experiment description here',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Shows defaults
    )

    # Required arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to HDF5 dataset file')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                       help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='results/experiment',
                       help='Output directory for results')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path')

    # Device selection
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda'],
                       help='Device to use (default: auto-detect)')

    # Quick mode
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode with reduced epochs/search space')

    return parser.parse_args()
```

### 2.3 Advanced CLI with Examples

**Reference:** [ablation_study.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/scripts/research/ablation_study.py#L1065-L1102)

```python
parser = argparse.ArgumentParser(
    description='Comprehensive Ablation Study Framework',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
    # Run full ablation study (all components)
    python scripts/research/ablation_study.py

    # Quick test with 3 configurations
    python scripts/research/ablation_study.py --quick

    # Ablate specific component
    python scripts/research/ablation_study.py --component physics

    # Run with specific number of seeds
    python scripts/research/ablation_study.py --seeds 10
    """
)

parser.add_argument('--component', '-c', type=str, default='all',
                   choices=['all', 'physics', 'attention', 'loss'],
                   help='Component to ablate')
```

---

## 3. Logging Conventions

### 3.1 Use Project Logger

**ALWAYS** use the project's centralized logger:

```python
from utils.logging import get_logger

logger = get_logger(__name__)

# Usage
logger.info("Starting experiment...")
logger.warning("GPU not available, using CPU")
logger.error("Dataset not found: %s", path)
```

### 3.2 Structured Progress Logging

```python
def run_experiments(configs: List[Config]):
    logger.info("=" * 60)
    logger.info("EXPERIMENT NAME")
    logger.info("=" * 60)
    logger.info(f"Configurations: {len(configs)}")
    logger.info(f"Device: {device}")

    for i, config in enumerate(configs):
        logger.info(f"\n--- Running {i+1}/{len(configs)}: {config.name} ---")
        # ... run experiment
        logger.info(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 60)
```

### 3.3 Epoch Progress Logging

Log every N epochs to avoid output spam:

```python
for epoch in range(epochs):
    # ... training code

    if (epoch + 1) % 10 == 0:  # Every 10 epochs
        logger.info(f"  Epoch {epoch+1}/{epochs}: "
                   f"Train {train_acc:.4f}, Val {val_acc:.4f}")
```

---

## 4. Reproducibility Requirements

### 4.1 Seed Setting

**ALWAYS** use the shared utility:

```python
from utils.reproducibility import set_seed

def main():
    args = parse_args()
    set_seed(args.seed)
    # ... rest of code
```

### 4.2 Multi-Seed Experiments

For publication-quality results, run with multiple seeds:

**Reference:** [ablation_study.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/scripts/research/ablation_study.py#L760-L800)

```python
def run_with_seeds(config, num_seeds: int = 5):
    """Run experiment with multiple seeds for statistical validity."""
    seed_results = []

    for seed in range(num_seeds):
        logger.info(f"  Seed {seed + 1}/{num_seeds}")
        set_seed(seed)

        # Run experiment
        metrics = run_single_experiment(config)
        seed_results.append(metrics)

    # Aggregate results
    return {
        'accuracy_mean': np.mean([r['accuracy'] for r in seed_results]),
        'accuracy_std': np.std([r['accuracy'] for r in seed_results]),
        'f1_mean': np.mean([r['f1'] for r in seed_results]),
        'f1_std': np.std([r['f1'] for r in seed_results]),
        'seed_results': seed_results
    }
```

### 4.3 Statistical Significance Testing

**Reference:** [pinn_ablation.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/scripts/research/pinn_ablation.py#L189-L212)

```python
from scipy import stats

def mcnemar_test(preds_a: List[int], preds_b: List[int], labels: List[int]) -> float:
    """
    McNemar's test between two models.
    Returns p-value for null hypothesis of equal error rates.
    """
    a_correct = np.array(preds_a) == np.array(labels)
    b_correct = np.array(preds_b) == np.array(labels)

    # Contingency: A correct & B wrong, A wrong & B correct
    b = np.sum(a_correct & ~b_correct)
    c = np.sum(~a_correct & b_correct)

    if b + c == 0:
        return 1.0  # No discordant pairs

    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return p_value
```

---

## 5. Output Format Conventions

### 5.1 Timestamped Filenames

**ALWAYS** include timestamp to prevent overwrites:

```python
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Examples
csv_path = output_dir / f"results_{timestamp}.csv"
json_path = output_dir / f"benchmark_{timestamp}.json"
plot_path = output_dir / f"comparison_{timestamp}.png"
```

### 5.2 JSON Results Structure

```python
results = {
    'timestamp': datetime.now().isoformat(),
    'config': {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'data_path': str(args.data)
    },
    'dataset': {
        'train_samples': len(train_set),
        'val_samples': len(val_set),
        'test_samples': len(test_set),
        'num_classes': num_classes
    },
    'results': {
        'model_name': {
            'accuracy': 0.9534,
            'f1_macro': 0.9421,
            'num_parameters': 1234567
        }
    }
}

# Save with indent for readability
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
```

### 5.3 CSV Results for Tables

```python
import pandas as pd

results_df = pd.DataFrame([
    {'name': 'Model A', 'accuracy': 0.95, 'f1': 0.94},
    {'name': 'Model B', 'accuracy': 0.93, 'f1': 0.92},
])

# Save with index=False for cleaner output
results_df.to_csv(output_dir / f'results_{timestamp}.csv', index=False)
```

### 5.4 Publication-Ready Figures

```python
import matplotlib.pyplot as plt
import seaborn as sns

def save_publication_figure(fig, path, dpi=150):
    """Save figure with publication-quality settings."""
    fig.savefig(path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    logger.info(f"Saved figure to {path}")

# Usage
fig, ax = plt.subplots(figsize=(10, 6))
# ... plotting code
save_publication_figure(fig, output_dir / f'results_{timestamp}.png')
```

### 5.5 Output Directory Creation

```python
def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ... rest of code
```

---

## Quick Reference Checklist

Before committing a research script, verify:

- [ ] Uses `from utils.reproducibility import set_seed`
- [ ] Has `--data`, `--epochs`, `--seed`, `--quick` CLI arguments
- [ ] Uses `from utils.logging import get_logger`
- [ ] Outputs have timestamps in filenames
- [ ] Results saved as JSON with metadata (config, dataset info)
- [ ] Optional dependencies handled with `HAS_*` pattern
- [ ] Progress logged every 10 epochs (not every epoch)
- [ ] Output directory created with `mkdir(parents=True, exist_ok=True)`

---

_Best practices extracted from IDB 5.1 Research Scripts Sub-Block_
