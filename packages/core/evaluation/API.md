# Evaluation API Reference

> Complete API documentation for all classes and functions in the Evaluation sub-block.

---

## Core Evaluators

### `ModelEvaluator`

> Base evaluator for comprehensive model assessment on test data. File: `evaluator.py`

**Constructor:**

```python
ModelEvaluator(model: nn.Module, device: str = 'cuda')
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | required | Trained model to evaluate |
| `device` | `str` | `'cuda'` | Device to run evaluation on |

**Methods:**

#### `evaluate(dataloader, class_names=None) -> Dict[str, any]`

Evaluate model on a DataLoader. Computes accuracy, confusion matrix, per-class metrics, and collects all predictions/probabilities.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataloader` | `DataLoader` | required | Test data loader |
| `class_names` | `Optional[list]` | `None` | Human-readable class names |

**Returns:** Dictionary with keys: `accuracy` (float), `confusion_matrix` (ndarray), `per_class_metrics` (dict), `predictions` (ndarray), `targets` (ndarray), `probabilities` (ndarray).

#### `compute_per_class_metrics(predictions, targets, probs, class_names=None) -> Dict`

Compute precision, recall, and F1 for each class using `sklearn.metrics.precision_recall_fscore_support`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `predictions` | `np.ndarray` | required | Predicted labels |
| `targets` | `np.ndarray` | required | Ground truth labels |
| `probs` | `np.ndarray` | required | Prediction probabilities |
| `class_names` | `Optional[list]` | `None` | Class names |

**Returns:** `{class_name: {'precision': float, 'recall': float, 'f1_score': float, 'support': int}}`

#### `generate_classification_report(predictions, targets, class_names=None) -> str`

Generate a detailed classification report string via `sklearn.metrics.classification_report`.

**Example:**

```python
evaluator = ModelEvaluator(model, device='cuda')
results = evaluator.evaluate(test_loader, class_names=['Normal', 'Fault A', 'Fault B'])
report = evaluator.generate_classification_report(
    results['predictions'], results['targets'], ['Normal', 'Fault A', 'Fault B']
)
print(report)
```

---

### `CNNEvaluator`

> Feature-rich evaluator for CNN fault diagnosis models. File: `cnn_evaluator.py`

**Constructor:**

```python
CNNEvaluator(
    model: nn.Module,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    class_names: Optional[List[str]] = None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | required | Trained CNN model |
| `device` | `str` | auto | Device for evaluation |
| `class_names` | `Optional[List[str]]` | `None` | Class names for reporting |

**Methods:**

#### `evaluate(test_loader, criterion=None, verbose=True) -> Dict[str, float]`

Full evaluation with accuracy, precision, recall, F1 (macro and weighted), and loss.

#### `get_classification_report(test_loader, output_dict=False)`

Generate sklearn classification report.

#### `get_confusion_matrix(test_loader, normalize=None) -> np.ndarray`

Compute confusion matrix with optional normalization (`'true'`, `'pred'`, `'all'`, or `None`).

#### `predict(test_loader, return_labels=False, return_probs=False)`

Generate predictions. Optionally return ground truth labels and/or class probabilities.

#### `predict_single(signal, return_prob=False)`

Predict fault class for a single signal tensor.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signal` | `torch.Tensor` | required | Input signal `[1, signal_length]` or `[signal_length]` |
| `return_prob` | `bool` | `False` | Also return confidence probability |

#### `save_results(metrics, save_path, include_cm=True)`

Save evaluation results to JSON file.

#### `get_per_class_accuracy(test_loader) -> Dict[int, float]`

Compute per-class accuracy. Returns `{class_index: accuracy_percentage}`.

**Example:**

```python
evaluator = CNNEvaluator(model, class_names=['Normal', 'Ball', 'Inner', 'Outer'])
metrics = evaluator.evaluate(test_loader)
evaluator.save_results(metrics, Path('results/eval.json'))
prediction = evaluator.predict_single(signal_tensor, return_prob=True)
```

---

### `PINNEvaluator`

> Specialized evaluator for Physics-Informed Neural Networks with physics-aware metrics. File: `pinn_evaluator.py`

**Constructor:**

```python
PINNEvaluator(model: nn.Module, device: str = 'cuda', sample_rate: int = 20480)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `nn.Module` | required | PINN model to evaluate |
| `device` | `str` | `'cuda'` | Device |
| `sample_rate` | `int` | `20480` | Signal sampling rate for physics loss computation |

**Methods:**

#### `evaluate_with_physics_metrics(dataloader, metadata=None, class_names=None) -> Dict`

Evaluate with standard metrics plus physics-aware metrics (frequency consistency, prediction plausibility).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataloader` | `DataLoader` | required | Test data loader |
| `metadata` | `Optional[Dict[str, Tensor]]` | `None` | Operating conditions (rpm, load, etc.) |
| `class_names` | `Optional[List[str]]` | `None` | Class names |

**Returns:** Dictionary with standard metrics plus `physics_metrics` sub-dict containing `frequency_consistency` and `prediction_plausibility`.

#### `test_sample_efficiency(train_dataset, val_dataset, train_function, sample_sizes=None, num_trials=3) -> Dict`

Test how model performance scales with training set size.

#### `test_ood_generalization(ood_dataloaders, class_names=None) -> Dict[str, Dict]`

Test out-of-distribution generalization to unseen operating conditions. Returns `{condition_name: metrics_dict}`.

#### `compare_with_baseline(baseline_model, test_dataloader, class_names=None) -> Dict`

Compare PINN performance against a baseline CNN model.

**Convenience function:**

```python
from packages.core.evaluation.pinn_evaluator import evaluate_pinn_model

results = evaluate_pinn_model(model, test_loader, device='cuda', metadata=meta, class_names=names)
```

---

### `SpectrogramEvaluator`

> Evaluator for spectrogram-based (2D CNN) models with Grad-CAM support. File: `spectrogram_evaluator.py`. Extends `ModelEvaluator`.

**Constructor:**

```python
SpectrogramEvaluator(
    model: nn.Module,
    device: str = 'cuda',
    class_names: Optional[List[str]] = None
)
```

**Methods:**

#### `visualize_predictions(spectrograms, targets=None, save_path=None, figsize=(15, 10))`

Visualize spectrograms annotated with model predictions and ground truth.

#### `compute_grad_cam(spectrogram, target_layer='layer4', target_class=None) -> Tuple[np.ndarray, int]`

Compute Grad-CAM heatmap for a single spectrogram. Returns `(heatmap [H, W], predicted_class)`.

#### `visualize_grad_cam(spectrograms, targets=None, save_path=None, figsize=(15, 10))`

Visualize spectrograms with Grad-CAM overlays.

#### `analyze_frequency_contributions(spectrograms, num_freq_bins=10) -> Dict`

Analyze which frequency bins contribute most to predictions.

#### `compare_tfr_types(dataloaders, tfr_names=None) -> Dict`

Compare model performance on different time-frequency representations (e.g., STFT vs. CWT vs. WVD).

---

## Analysis Tools

### `ConfusionAnalyzer`

> Deep analysis of confusion patterns. File: `confusion_analyzer.py`

**Constructor:**

```python
ConfusionAnalyzer(confusion_matrix: np.ndarray, class_names: List[str])
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `confusion_matrix` | `np.ndarray` | required | Confusion matrix `[num_classes, num_classes]` |
| `class_names` | `List[str]` | required | Class names |

**Methods:**

#### `find_most_confused_pairs(top_k=5) -> List[Tuple[str, str, int]]`

Returns list of `(true_class, predicted_class, count)` tuples, sorted descending by count.

#### `compute_error_concentration() -> float`

Returns percentage of total errors concentrated in the single most-error-prone class.

#### `analyze_per_class_errors() -> pd.DataFrame`

Returns DataFrame with columns: `class`, `total`, `correct`, `errors`, `accuracy`.

---

### `ROCAnalyzer`

> ROC curve and AUC computation for multi-class classification. File: `roc_analyzer.py`

**Constructor:**

```python
ROCAnalyzer(probabilities: np.ndarray, targets: np.ndarray, class_names: List[str])
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `probabilities` | `np.ndarray` | required | Predicted probabilities `[N, num_classes]` |
| `targets` | `np.ndarray` | required | Ground truth labels `[N]` |
| `class_names` | `List[str]` | required | Class names |

**Methods:**

#### `compute_roc_curves() -> Dict[str, Tuple[np.ndarray, np.ndarray]]`

Returns `{class_name: (fpr_array, tpr_array)}` using one-vs-rest strategy.

#### `compute_auc_scores() -> Dict[str, float]`

Returns `{class_name: auc_score}`. Handles classes not present in targets (returns 0.0).

#### `compute_macro_auc() -> float`

Returns macro-averaged AUC across all classes.

---

### `ErrorAnalyzer`

> Comprehensive misclassification analysis with confidence and hard example mining. File: `error_analysis.py`

**Constructor:**

```python
ErrorAnalyzer(model: nn.Module, class_names: List[str], device: str = 'cpu')
```

**Methods:**

#### `analyze_misclassifications(test_loader) -> Dict`

Analyze all misclassifications. Returns dictionary with confusion matrix, per-sample details, and statistics.

#### `find_most_confused_pairs(confusion_matrix, top_k=10) -> List[Tuple]`

Find top-*k* confused class pairs.

#### `analyze_confidence_distribution(analysis_results) -> Dict`

Analyze confidence distribution for correct vs. incorrect predictions.

#### `find_hard_examples(analysis_results, criterion='low_confidence', top_k=20) -> List[Dict]`

Find hard examples. Criteria: `'low_confidence'` or `'margin'`.

#### `plot_confusion_matrix(confusion_matrix, save_path=None)`

Plot confusion matrix as a heatmap.

#### `plot_confidence_distribution(confidence_stats, save_path=None)`

Plot confidence histograms for correct vs. incorrect predictions.

#### `compare_model_errors(models, model_names, test_loader) -> Dict`

Compare errors across multiple models. Identifies samples that all models get wrong, and complementary errors useful for ensemble construction.

#### `generate_report(analysis_results, save_path=None) -> str`

Generate a comprehensive text report of the error analysis.

---

### `RobustnessTester`

> Test model robustness to sensor noise and missing features. File: `robustness_tester.py`

**Constructor:**

```python
RobustnessTester(model: nn.Module, device: str = 'cuda')
```

**Methods:**

#### `test_sensor_noise(dataloader, noise_levels=[0.01, 0.05, 0.1, 0.2]) -> Dict[float, float]`

Add Gaussian noise at various levels and measure accuracy.

**Returns:** `{noise_level: accuracy_percentage}`

#### `test_missing_features(dataloader, dropout_rates=[0.1, 0.2, 0.3, 0.5]) -> Dict[float, float]`

Zero out random features at various dropout rates and measure accuracy.

**Returns:** `{dropout_rate: accuracy_percentage}`

---

## Ensemble Evaluation

### `EnsembleEvaluator`

> Comprehensive ensemble evaluation with diversity metrics. File: `ensemble_evaluator.py`

**Constructor:**

```python
EnsembleEvaluator(num_classes: int = NUM_CLASSES, class_names: Optional[List[str]] = None)
```

**Methods:**

#### `evaluate_ensemble(ensemble, test_loader, device='cuda', save_dir=None) -> Dict`

Full ensemble evaluation: accuracy, F1, precision, recall, confusion matrix, per-class metrics. Optionally saves plots.

#### `evaluate_ensemble_diversity(models, test_loader, device='cuda') -> Dict`

Compute diversity metrics: pairwise disagreement, Q-statistic, correlation coefficient.

#### `compare_ensemble_vs_individuals(ensemble, individual_models, test_loader, device='cuda', model_names=None) -> Dict`

Compare ensemble performance against each individual model.

**Convenience function:**

```python
from packages.core.evaluation.ensemble_evaluator import evaluate_ensemble_performance

results = evaluate_ensemble_performance(ensemble, test_loader, device='cuda')
print(f"Accuracy: {results['accuracy']:.2f}%")
```

---

### `EnsembleVoting`

> Combine predictions from multiple models via voting. File: `ensemble_voting.py`

**Constructor:**

```python
EnsembleVoting(
    models: List[nn.Module],
    weights: Optional[List[float]] = None,
    device: str = 'cpu'
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `List[nn.Module]` | required | List of trained models |
| `weights` | `Optional[List[float]]` | `None` | Model weights for soft voting (uniform if None) |
| `device` | `str` | `'cpu'` | Device |

**Methods:**

#### `soft_voting(inputs) -> Tuple[Tensor, Tensor]`

Weighted average of predicted probabilities. Returns `(predictions [B], probabilities [B, num_classes])`.

#### `hard_voting(inputs) -> Tensor`

Majority vote of predicted classes. Returns predictions `[B]`.

#### `evaluate(test_loader, voting_method='soft') -> Dict`

Evaluate ensemble on test set. Returns accuracy, precision, recall, F1.

---

### `StackingEnsemble`

> Two-level stacking ensemble with a meta-learner. File: `ensemble_voting.py`

**Constructor:**

```python
StackingEnsemble(
    base_models: List[nn.Module],
    meta_learner: Optional = None,
    device: str = 'cpu'
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_models` | `List[nn.Module]` | required | Level-1 models |
| `meta_learner` | `Optional` | `None` | Level-2 learner (defaults to `LogisticRegression`) |
| `device` | `str` | `'cpu'` | Device |

**Methods:**

#### `get_base_predictions(inputs) -> np.ndarray`

Get concatenated predictions from all base models. Returns `[B, num_models * num_classes]`.

#### `train(train_loader)`

Train the meta-learner on base model predictions.

#### `predict(inputs) -> np.ndarray`

Predict using the stacking ensemble. Returns predictions `[B]`.

#### `evaluate(test_loader) -> Dict`

Evaluate stacking ensemble. Returns accuracy, precision, recall, F1.

**Standalone comparison function:**

```python
from packages.core.evaluation.ensemble_voting import compare_ensemble_methods

comparison_df = compare_ensemble_methods(models, train_loader, test_loader, device='cuda')
print(comparison_df)
```

---

## Comparison & Visualization

### `TimeVsFrequencyComparator`

> Compare time-domain (1D CNN) and frequency-domain (2D CNN) models. File: `time_vs_frequency_comparison.py`

**Constructor:**

```python
TimeVsFrequencyComparator(class_names: List[str], device: str = 'cuda')
```

**Methods:**

#### `evaluate_model(model, dataloader, model_name) -> Dict`

Evaluate a single model with accuracy, per-class F1, confusion matrix, and inference timing.

#### `compare_models(models, dataloaders) -> Dict[str, Dict]`

Compare multiple models. Each model can have its own dataloader (for different input representations).

#### `create_comparison_table(results) -> pd.DataFrame`

Create a summary DataFrame with accuracy, macro F1, weighted F1, and inference time per model.

#### `analyze_per_fault_performance(results, save_path=None) -> pd.DataFrame`

Analyze which faults benefit from frequency-domain processing.

#### `identify_frequency_sensitive_faults(results, time_models, freq_models, threshold=0.02) -> Dict`

Identify faults where frequency-domain models outperform time-domain models by at least `threshold`.

#### `plot_confusion_matrices(results, save_dir=None)`

Plot confusion matrices for all models side-by-side.

#### `generate_comparison_report(results, output_dir)`

Generate comprehensive comparison report with figures.

**Convenience function:**

```python
from packages.core.evaluation.time_vs_frequency_comparison import compare_time_vs_frequency

results = compare_time_vs_frequency(models, dataloaders, class_names, output_dir=Path('results/'))
```

---

### `PhysicsInterpreter`

> Visualize how PINN models use physics knowledge. File: `physics_interpretability.py`

**Constructor:**

```python
PhysicsInterpreter(model, device: str = 'cuda', sample_rate: int = 51200)
```

**Methods:**

#### `plot_learned_vs_expected_frequencies(signal, true_label, predicted_label, rpm=3600.0, save_path=None)`

Compare observed, expected, and predicted frequency distributions.

#### `visualize_knowledge_graph(kg_pinn_model, signal=None, save_path=None)`

Visualize fault relationship graph with learned attention weights.

#### `plot_physics_feature_importance(hybrid_pinn_model, test_samples, test_labels, save_path=None)`

Analyze importance of physics features by perturbation.

#### `plot_operating_condition_sensitivity(signal, rpm_range, load_range, num_points=20, save_path=None)`

Visualize how predictions change across operating conditions.

#### `plot_sommerfeld_reynolds_distribution(test_loader, save_path=None)`

Plot distribution of Sommerfeld and Reynolds numbers in dataset.

---

### Standalone Functions — `architecture_comparison.py`

#### `count_parameters(model) -> int`

Count total trainable parameters.

#### `compute_flops(model, input_shape=(1, 1, SIGNAL_LENGTH), device='cpu') -> int`

Estimate FLOPs (simplified, via hooks on Conv1d and Linear layers).

#### `measure_inference_time(model, input_shape, num_runs=100, warmup_runs=10, device='cpu') -> Dict`

Returns `{'mean': float, 'std': float, 'min': float, 'max': float}` in seconds.

#### `measure_memory_usage(model, input_shape, device='cpu') -> Dict`

Returns memory statistics in MB.

#### `evaluate_model_accuracy(model, test_loader, device='cpu') -> Dict`

Returns accuracy and per-class F1.

#### `compare_architectures(model_dict, test_loader=None, input_shape=(1,1,SIGNAL_LENGTH), device='cpu', save_path=None) -> pd.DataFrame`

Comprehensive comparison of multiple architectures. Returns DataFrame with all metrics.

#### `plot_accuracy_vs_params(df, save_path=None)`

Plot accuracy vs. parameter count.

#### `plot_accuracy_vs_inference_time(df, save_path=None)`

Plot accuracy vs. inference time.

#### `plot_pareto_frontier(df, save_path=None)`

Plot multi-dimensional Pareto frontier.

#### `print_comparison_summary(df)`

Print formatted comparison summary to console.

---

### Standalone Functions — `attention_visualization.py`

#### `plot_attention_heatmap(attention_weights, patch_size=512, head_idx=None, figsize=(10,8), save_path=None, cmap='viridis', title=None)`

Plot attention weights as a heatmap. Accepts shapes `[B, n_heads, n_patches, n_patches]`, `[n_heads, n_patches, n_patches]`, or `[n_patches, n_patches]`.

#### `plot_signal_with_attention(signal, attention_weights, patch_size=512, true_label=None, predicted_label=None, figsize=(15,8), save_path=None, label_names=None)`

Signal waveform with attention importance overlay.

#### `attention_rollout(all_attention_weights, discard_ratio=0.1) -> torch.Tensor`

Compute attention rollout across all transformer layers. Recursively multiplies attention matrices.

#### `find_most_attended_patches(attention_weights, top_k=10) -> List`

Find patches receiving the most attention. Returns list of `(patch_index, importance_score)`.

#### `compare_attention_heads(attention_weights, patch_size=512, figsize=(15,10), save_path=None)`

Compare attention patterns across heads.

#### `analyze_attention_entropy(attention_weights) -> Dict`

Compute attention entropy. Returns `{'mean_entropy': float, 'per_head_entropy': list, ...}`.

---

### Standalone Functions — `benchmark.py`

#### `benchmark_against_classical(dl_model, test_loader, classical_results, device='cuda') -> pd.DataFrame`

Compare a deep learning model against classical ML baselines. `classical_results` is `{model_name: accuracy}`.

#### `generate_comparison_table(results) -> pd.DataFrame`

Sort results by accuracy descending.

**Example:**

```python
from packages.core.evaluation.benchmark import benchmark_against_classical

classical = {'SVM': 85.2, 'Random Forest': 82.1, 'KNN': 78.5}
comparison = benchmark_against_classical(model, test_loader, classical, device='cuda')
print(comparison)
```
