# IDB 1.3 Evaluation Sub-Block — Best Practices

**For:** Cross-team consistency and quality assurance  
**Audience:** ML Engineers, Data Scientists, QA Engineers  
**Last Updated:** 2026-01-24

---

## Table of Contents

1. [Metric Computation Patterns](#1-metric-computation-patterns)
2. [Results Serialization Conventions](#2-results-serialization-conventions)
3. [Visualization Styling Requirements](#3-visualization-styling-requirements)
4. [Cross-Team Result Format Contracts](#4-cross-team-result-format-contracts)
5. [Testing Patterns for Evaluators](#5-testing-patterns-for-evaluators)

---

## 1. Metric Computation Patterns

### 1.1 Standard Evaluation Loop

**Pattern:** Always use `@torch.no_grad()` decorator and set model to eval mode.

```python
@torch.no_grad()
def evaluate(self, dataloader: DataLoader) -> Dict[str, any]:
    """Evaluate model on dataloader."""
    self.model.eval()

    all_predictions = []
    all_targets = []
    all_probs = []

    for inputs, targets in tqdm(dataloader, desc="Evaluating"):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        # Forward pass
        outputs = self.model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)

        # Collect results (on CPU to avoid GPU memory issues)
        all_predictions.append(predicted.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    probs = np.concatenate(all_probs)

    # Compute metrics
    return self._compute_metrics(predictions, targets, probs)
```

> [!IMPORTANT]
> **Why this pattern?**
>
> - `@torch.no_grad()` disables gradient tracking → 50% memory reduction
> - `.cpu()` before collecting prevents GPU OOM on large test sets
> - `tqdm` provides progress feedback for long evaluations

### 1.2 Per-Class Metrics Computation

**Pattern:** Use `sklearn.metrics.precision_recall_fscore_support` with `zero_division=0`.

```python
from sklearn.metrics import precision_recall_fscore_support

def compute_per_class_metrics(
    self,
    predictions: np.ndarray,
    targets: np.ndarray,
    probs: np.ndarray,
    class_names: Optional[list] = None
) -> Dict:
    """Compute precision, recall, F1 for each class."""
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        average=None,
        zero_division=0  # Handle classes with no predictions
    )

    num_classes = len(precision)
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    per_class = {}
    for i in range(num_classes):
        per_class[class_names[i]] = {
            'precision': float(precision[i]),  # Convert to Python float
            'recall': float(recall[i]),
            'f1_score': float(f1[i]),
            'support': int(support[i])
        }

    return per_class
```

> [!TIP]
> **Best Practice:** Always include `zero_division=0` to handle edge cases where a class has no predictions (avoids division by zero warnings).

### 1.3 Macro vs. Weighted Averages

**When to use each:**

| Metric Type          | Use Case                                     | Example             |
| -------------------- | -------------------------------------------- | ------------------- |
| **Macro Average**    | Balanced datasets, treat all classes equally | Research benchmarks |
| **Weighted Average** | Imbalanced datasets, weight by support       | Production systems  |

```python
# Macro averages (treat all classes equally)
precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)

# Weighted averages (weight by class frequency)
precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)
recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)
f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
```

### 1.4 Device Detection Pattern

**Pattern:** Auto-detect CUDA availability with fallback.

```python
def __init__(
    self,
    model: nn.Module,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    self.model = model.to(device)
    self.device = device
    self.model.eval()
```

> [!NOTE]
> This pattern makes code portable across systems with/without GPUs.

---

## 2. Results Serialization Conventions

### 2.1 JSON Export Pattern

**Pattern:** Convert numpy arrays to lists, ensure JSON serializability.

```python
def save_results(
    self,
    metrics: Dict,
    save_path: Path,
    include_cm: bool = True
):
    """Save evaluation results to JSON file."""
    results = {}

    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            # Only include confusion matrix if requested
            if include_cm or key != 'confusion_matrix':
                results[key] = value.tolist()
        elif isinstance(value, (np.int64, np.int32)):
            results[key] = int(value)
        elif isinstance(value, (np.float64, np.float32)):
            results[key] = float(value)
        else:
            results[key] = value

    # Create parent directories if needed
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with indentation for readability
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {save_path}")
```

### 2.2 Standard Output Dictionary Schema

**All evaluators MUST return a dictionary with these keys:**

```python
results = {
    # Core metrics (REQUIRED)
    'accuracy': float,              # Overall accuracy (0-100)
    'confusion_matrix': np.ndarray, # [num_classes, num_classes]
    'per_class_metrics': Dict,      # Per-class precision/recall/F1

    # Aggregated metrics (REQUIRED)
    'precision_macro': float,       # Macro-averaged precision
    'recall_macro': float,          # Macro-averaged recall
    'f1_macro': float,              # Macro-averaged F1

    # Raw predictions (REQUIRED for further analysis)
    'predictions': np.ndarray,      # Predicted labels
    'targets': np.ndarray,          # Ground truth labels
    'probabilities': np.ndarray,    # Class probabilities [N, num_classes]

    # Optional metrics
    'loss': float,                  # Test loss (if criterion provided)
}
```

> [!WARNING]
> **Breaking Changes:** If you add/remove required keys, update this document and notify all teams.

---

## 3. Visualization Styling Requirements

### 3.1 Matplotlib Configuration

**Pattern:** Use consistent styling across all visualizations.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style at module level
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
```

### 3.2 Confusion Matrix Visualization

**Standard pattern:**

```python
def plot_confusion_matrix(
    self,
    cm: np.ndarray,
    save_path: Path,
    class_names: List[str]
):
    """Plot confusion matrix with standard styling."""
    plt.figure(figsize=(12, 10))

    sns.heatmap(
        cm,
        annot=True,           # Show values
        fmt='d',              # Integer format
        cmap='Blues',         # Standard colormap
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )

    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Important: close to free memory
```

> [!TIP]
> **Always use `plt.close()`** after saving to prevent memory leaks in batch plotting.

### 3.3 Colormap Standards

| Visualization Type | Colormap    | Rationale                                   |
| ------------------ | ----------- | ------------------------------------------- |
| Confusion Matrix   | `'Blues'`   | Sequential, intuitive                       |
| Attention Heatmaps | `'viridis'` | Perceptually uniform, colorblind-safe       |
| Diverging Metrics  | `'RdYlGn'`  | Red (bad) → Yellow (neutral) → Green (good) |
| Error Analysis     | `'Reds'`    | Error intensity                             |

### 3.4 Attention Visualization Best Practices

**From `attention_visualization.py`:**

```python
def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    patch_size: int = 512,
    head_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    cmap: str = 'viridis',
    title: Optional[str] = None
) -> plt.Figure:
    """Plot attention weights as a heatmap."""

    # Handle different tensor shapes
    if attention_weights.dim() == 4:
        attention_weights = attention_weights[0]  # Remove batch dimension
    if attention_weights.dim() == 3:
        if head_idx is not None:
            attention_weights = attention_weights[head_idx]
        else:
            attention_weights = attention_weights.mean(dim=0)  # Average over heads

    attention_np = attention_weights.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(attention_np, cmap=cmap, aspect='auto', interpolation='nearest')

    # Add descriptive labels
    ax.set_xlabel('Key Patch Index', fontsize=12)
    ax.set_ylabel('Query Patch Index', fontsize=12)
    ax.set_title(title or 'Attention Heatmap', fontsize=14, fontweight='bold')

    # Colorbar with label
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
```

---

## 4. Cross-Team Result Format Contracts

### 4.1 Standard Metric Names

**Use these exact names for consistency across all teams:**

| Metric            | Key Name           | Type         | Range                      |
| ----------------- | ------------------ | ------------ | -------------------------- |
| Accuracy          | `accuracy`         | `float`      | 0-100 (percentage)         |
| Precision (macro) | `precision_macro`  | `float`      | 0-100                      |
| Recall (macro)    | `recall_macro`     | `float`      | 0-100                      |
| F1 Score (macro)  | `f1_macro`         | `float`      | 0-100                      |
| Confusion Matrix  | `confusion_matrix` | `np.ndarray` | [num_classes, num_classes] |
| Per-Class F1      | `per_class_f1`     | `np.ndarray` | [num_classes]              |
| AUC (macro)       | `auc_macro`        | `float`      | 0-1                        |

> [!IMPORTANT]
> **Percentage Convention:** All percentages are stored as 0-100, not 0-1.

### 4.2 Per-Class Metrics Schema

```python
per_class_metrics = {
    "Normal": {
        "precision": 98.5,
        "recall": 97.2,
        "f1_score": 97.8,
        "support": 150
    },
    "Ball Fault": {
        "precision": 95.3,
        "recall": 96.1,
        "f1_score": 95.7,
        "support": 142
    },
    # ... for all classes
}
```

### 4.3 Ensemble Results Format

**From `ensemble_evaluator.py`:**

```python
ensemble_results = {
    # Standard metrics
    'accuracy': float,
    'f1_score': float,

    # Diversity metrics (ensemble-specific)
    'diversity': {
        'mean_disagreement': float,      # 0-1, higher = more diverse
        'mean_q_statistic': float,       # -1 to 1
        'mean_correlation': float,       # 0-1, lower = more diverse
        'disagreement_matrix': np.ndarray  # [num_models, num_models]
    },

    # Individual model comparison
    'individual_performances': [
        {'model_name': str, 'accuracy': float, 'f1_score': float},
        # ... for each model
    ],

    'improvement_over_best': float  # Percentage points
}
```

---

## 5. Testing Patterns for Evaluators

### 5.1 Dummy Data Generation

**Pattern:** Create reproducible test data with known properties.

```python
def test_cnn_evaluator():
    """Test CNN evaluator with dummy data."""
    import numpy as np
    from models.cnn.cnn_1d import CNN1D

    # Create dummy data with fixed seed
    np.random.seed(42)
    num_samples = 100
    signal_length = 102400
    num_classes = 11

    signals = np.random.randn(num_samples, signal_length).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_samples)

    # Create datasets and loaders
    _, _, test_ds = create_cnn_datasets_from_arrays(signals, labels)
    loaders = create_cnn_dataloaders(
        test_dataset=test_ds,
        batch_size=8,
        num_workers=0  # Avoid multiprocessing in tests
    )

    # Create evaluator
    model = CNN1D(num_classes=num_classes)
    evaluator = CNNEvaluator(model, device='cpu', class_names=class_names)

    # Run evaluation
    metrics = evaluator.evaluate(loaders['test'], verbose=True)

    # Assertions
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 100
    assert metrics['confusion_matrix'].shape == (num_classes, num_classes)
```

### 5.2 **main** Block Pattern

**All evaluator files should include a test in `__main__`:**

```python
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Evaluator")
    print("=" * 60)

    # Test 1: Basic evaluation
    print("\n1. Testing basic evaluation...")
    test_basic_evaluation()

    # Test 2: Per-class metrics
    print("\n2. Testing per-class metrics...")
    test_per_class_metrics()

    # Test 3: Result serialization
    print("\n3. Testing result serialization...")
    test_save_results()

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
```

### 5.3 Test Coverage Checklist

Every evaluator should test:

- [ ] **Basic evaluation** - Accuracy, F1, precision, recall
- [ ] **Per-class metrics** - All classes have metrics
- [ ] **Confusion matrix** - Correct shape and values
- [ ] **Single prediction** - `predict_single()` works
- [ ] **Batch prediction** - `predict()` with `return_labels` and `return_probs`
- [ ] **Result serialization** - JSON export works
- [ ] **Edge cases** - Empty test set, single-class predictions

### 5.4 Memory Leak Testing

**Pattern:** Test on increasing batch sizes to detect memory issues.

```python
def test_memory_scalability():
    """Test evaluator doesn't leak memory on large batches."""
    import gc
    import torch

    model = create_test_model()
    evaluator = Evaluator(model, device='cpu')

    for batch_size in [16, 32, 64, 128, 256]:
        # Create large test set
        loader = create_test_loader(num_samples=1000, batch_size=batch_size)

        # Evaluate
        metrics = evaluator.evaluate(loader)

        # Force garbage collection
        del metrics
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"✓ Batch size {batch_size} passed")
```

---

## 6. Common Pitfalls to Avoid

### ❌ Anti-Pattern 1: Storing Full Tensors

```python
# BAD: Stores all signals in memory
misclassified = []
for signal, true_label, pred_label in zip(signals, true_labels, predictions):
    if true_label != pred_label:
        misclassified.append({
            'signal': signal,  # ❌ Huge memory usage!
            'true_label': true_label,
            'predicted_label': pred_label
        })
```

```python
# GOOD: Store only indices
misclassified_indices = []
for idx, (true_label, pred_label) in enumerate(zip(true_labels, predictions)):
    if true_label != pred_label:
        misclassified_indices.append(idx)

# Retrieve signals later when needed
hard_signals = [dataset[idx] for idx in misclassified_indices[:10]]
```

### ❌ Anti-Pattern 2: Inconsistent Percentage Scaling

```python
# BAD: Mixing 0-1 and 0-100
results = {
    'accuracy': 0.95,      # 0-1 scale
    'f1_score': 95.3,      # 0-100 scale ❌ Inconsistent!
}
```

```python
# GOOD: All percentages as 0-100
results = {
    'accuracy': 95.0,
    'f1_score': 95.3,
}
```

### ❌ Anti-Pattern 3: Not Closing Plots

```python
# BAD: Memory leak when generating many plots
for i in range(100):
    fig = plot_confusion_matrix(cm)
    fig.savefig(f'cm_{i}.png')
    # ❌ Figure not closed, memory accumulates
```

```python
# GOOD: Always close figures
for i in range(100):
    fig = plot_confusion_matrix(cm)
    fig.savefig(f'cm_{i}.png')
    plt.close(fig)  # ✅ Frees memory
```

---

## 7. Advanced Patterns

### 7.1 Physics-Aware Evaluation (PINN)

**From `pinn_evaluator.py`:**

```python
def evaluate_with_physics_metrics(
    self,
    dataloader: DataLoader,
    metadata: Optional[Dict[str, torch.Tensor]] = None
) -> Dict:
    """Evaluate with physics-aware metrics."""

    # Standard metrics
    standard_results = super().evaluate(dataloader)

    # Physics-specific metrics
    freq_consistency = self._compute_frequency_consistency(
        signals, predictions, metadata
    )

    prediction_plausibility = self._compute_prediction_plausibility(
        signals, predictions, metadata
    )

    # Add physics metrics to results
    standard_results['physics_metrics'] = {
        'frequency_consistency': freq_consistency,
        'prediction_plausibility': prediction_plausibility
    }

    return standard_results
```

### 7.2 Lazy Metric Computation

**Recommended for extensibility:**

```python
from functools import cached_property

class EvaluationResult:
    """Lazy evaluation result container."""

    def __init__(self, predictions, targets, probabilities):
        self._predictions = predictions
        self._targets = targets
        self._probabilities = probabilities

    @cached_property
    def accuracy(self) -> float:
        """Compute accuracy only when accessed."""
        return accuracy_score(self._targets, self._predictions) * 100

    @cached_property
    def confusion_matrix(self) -> np.ndarray:
        """Compute confusion matrix only when accessed."""
        return confusion_matrix(self._targets, self._predictions)

    @cached_property
    def per_class_f1(self) -> np.ndarray:
        """Compute per-class F1 only when accessed."""
        return f1_score(self._targets, self._predictions, average=None)
```

---

## 8. Quick Reference

### Metric Computation

- Always use `@torch.no_grad()` and `model.eval()`
- Move tensors to CPU before collecting
- Use `zero_division=0` in sklearn metrics

### Serialization

- Convert numpy to Python types for JSON
- Use `indent=2` for readability
- Store percentages as 0-100, not 0-1

### Visualization

- Use `cmap='viridis'` for attention, `'Blues'` for confusion
- Always `plt.close()` after saving
- Set `dpi=300` for publication quality

### Testing

- Include `__main__` block with tests
- Test edge cases (empty sets, single class)
- Check for memory leaks with large batches

---

## References

- [IDB 1.3 Evaluation Analysis](./IDB_1_3_EVALUATION_ANALYSIS.md)
- [evaluator.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/core/evaluation/evaluator.py)
- [cnn_evaluator.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/core/evaluation/cnn_evaluator.py)
- [attention_visualization.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/core/evaluation/attention_visualization.py)
