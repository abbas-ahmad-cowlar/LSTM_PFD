# Evaluation Metrics Guide

> Reference for every metric computed in the Evaluation sub-block, with formulas, interpretation, and code examples.

## Classification Metrics

These metrics are computed by `ModelEvaluator` (`evaluator.py`) and `CNNEvaluator` (`cnn_evaluator.py`) using `sklearn.metrics`.

### Accuracy

**Formula:**

$$\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Predictions}} \times 100$$

**When to use:** Quick overall performance snapshot. Less informative when classes are imbalanced.

**Interpretation:** Higher is better. A value of 100% means perfect classification. For _N_-class problems with balanced classes, random chance is 100/_N_%.

**Computed in:** `ModelEvaluator.evaluate()`, `CNNEvaluator.evaluate()`, `CNNEvaluator.get_per_class_accuracy()`

```python
from packages.core.evaluation.evaluator import ModelEvaluator

evaluator = ModelEvaluator(model, device='cuda')
results = evaluator.evaluate(test_loader)
print(f"Accuracy: {results['accuracy']:.2f}%")
```

---

### Precision

**Formula (per-class):**

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}$$

**When to use:** When false positives are costly (e.g., raising a false alarm for a bearing fault that doesn't exist).

**Interpretation:** Of all samples predicted as class _c_, what fraction actually belongs to class _c_. Higher is better.

**Computed in:** `ModelEvaluator.compute_per_class_metrics()`, `CNNEvaluator._compute_metrics()`

```python
results = evaluator.evaluate(test_loader, class_names=['Normal', 'Inner', 'Outer'])
for cls, metrics in results['per_class_metrics'].items():
    print(f"{cls}: Precision={metrics['precision']:.4f}")
```

---

### Recall (Sensitivity)

**Formula (per-class):**

$$\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}$$

**When to use:** When false negatives are costly (e.g., missing an actual bearing fault is dangerous).

**Interpretation:** Of all samples that actually belong to class _c_, what fraction was correctly identified. Higher is better.

**Computed in:** `ModelEvaluator.compute_per_class_metrics()`, `CNNEvaluator._compute_metrics()`

---

### F1 Score

**Formula (per-class):**

$$F1_c = 2 \times \frac{\text{Precision}_c \times \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

**When to use:** When you need a balanced metric that considers both false positives and false negatives. Preferred over accuracy for imbalanced datasets.

**Interpretation:** Harmonic mean of precision and recall. Range [0, 1]. A value of 1 means perfect precision and recall.

**Computed in:** `ModelEvaluator.compute_per_class_metrics()`, `CNNEvaluator._compute_metrics()`, `EnsembleEvaluator.evaluate_ensemble()`

**Averaging modes** (used by `CNNEvaluator`):
| Mode | Description |
|------|-------------|
| `macro` | Unweighted mean of per-class F1 scores |
| `weighted` | Weighted mean by class support (sample count) |

---

## ROC / AUC Metrics

These metrics are computed by `ROCAnalyzer` (`roc_analyzer.py`).

### ROC Curve (Receiver Operating Characteristic)

**What it is:** A plot of True Positive Rate (TPR) vs. False Positive Rate (FPR) at various classification thresholds, computed per-class using the one-vs-rest strategy.

**When to use:** To evaluate classifier performance across all threshold settings, especially when the operating threshold hasn't been decided.

**Interpretation:** A curve closer to the top-left corner is better. The diagonal line represents random chance.

**Computed in:** `ROCAnalyzer.compute_roc_curves()` — returns `{class_name: (fpr_array, tpr_array)}`.

```python
from packages.core.evaluation.roc_analyzer import ROCAnalyzer

analyzer = ROCAnalyzer(probabilities, targets, class_names)
roc_curves = analyzer.compute_roc_curves()

for cls_name, (fpr, tpr) in roc_curves.items():
    print(f"{cls_name}: {len(fpr)} threshold points")
```

---

### AUC (Area Under the ROC Curve)

**Formula:** Area under the ROC curve, computed numerically via the trapezoidal rule.

**When to use:** Single-number summary of ROC curve quality. Threshold-independent.

**Interpretation:** Range [0, 1]. AUC = 1.0 is perfect, AUC = 0.5 is random chance.

**Computed in:**

- `ROCAnalyzer.compute_auc_scores()` — per-class AUC using `sklearn.metrics.roc_auc_score`
- `ROCAnalyzer.compute_macro_auc()` — macro-averaged AUC across all classes

```python
auc_scores = analyzer.compute_auc_scores()
macro_auc = analyzer.compute_macro_auc()
print(f"Macro AUC: {macro_auc:.4f}")
```

---

## Confusion Matrix Metrics

These metrics are computed by `ConfusionAnalyzer` (`confusion_analyzer.py`).

### Most Confused Pairs

**What it is:** The top-_k_ pairs of classes (true class, predicted class) with the highest misclassification counts, excluding diagonal elements.

**When to use:** To identify which fault types the model struggles to distinguish.

**Computed in:** `ConfusionAnalyzer.find_most_confused_pairs(top_k=5)`

```python
from packages.core.evaluation.confusion_analyzer import ConfusionAnalyzer

analyzer = ConfusionAnalyzer(confusion_matrix, class_names)
top_confused = analyzer.find_most_confused_pairs(top_k=5)
# Returns: [('Inner Race', 'Outer Race', 23), ...]
```

---

### Error Concentration

**Formula:**

$$\text{Error Concentration} = \frac{\max(\text{errors per class})}{\text{total errors}} \times 100$$

**When to use:** To check whether errors are spread across classes or concentrated in one class.

**Interpretation:** High concentration means one class dominates the errors. Low concentration means errors are distributed.

**Computed in:** `ConfusionAnalyzer.compute_error_concentration()`

---

### Per-Class Error Analysis

**What it is:** A DataFrame with `class`, `total`, `correct`, `errors`, and `accuracy` columns for each class.

**Computed in:** `ConfusionAnalyzer.analyze_per_class_errors()`

---

## Ensemble Diversity Metrics

These metrics are computed by `EnsembleEvaluator.evaluate_ensemble_diversity()` in `ensemble_evaluator.py`.

### Pairwise Disagreement

**Formula:**

$$D_{ij} = \frac{1}{N} \sum_{k=1}^{N} \mathbb{1}[\hat{y}_i^{(k)} \neq \hat{y}_j^{(k)}]$$

**Interpretation:** Fraction of samples where models _i_ and _j_ produce different predictions. Higher disagreement generally indicates better ensemble potential.

---

### Q-Statistic

**What it is:** Yule's Q-statistic measuring the association between two classifiers. Computed from the 2×2 contingency table of correct/incorrect predictions.

**Interpretation:** Range [-1, 1]. Values near 0 indicate independence (high diversity). Positive values indicate agreement.

---

### Correlation Coefficient

**What it is:** Pearson correlation of binary correctness vectors between pairs of models.

**Interpretation:** Lower correlation = more diverse ensemble members = better ensemble potential.

---

## Robustness Metrics

These metrics are computed by `RobustnessTester` (`robustness_tester.py`).

### Noise-Degraded Accuracy

**What it is:** Model accuracy measured after adding Gaussian noise to inputs at various standard deviation levels.

**Default noise levels:** `[0.01, 0.05, 0.1, 0.2]`

**When to use:** To assess how well the model tolerates sensor noise in real-world deployments.

**Computed in:** `RobustnessTester.test_sensor_noise(dataloader, noise_levels)`

```python
from packages.core.evaluation.robustness_tester import RobustnessTester

tester = RobustnessTester(model, device='cuda')
noise_results = tester.test_sensor_noise(test_loader, noise_levels=[0.01, 0.05, 0.1])
# Returns: {0.01: 98.5, 0.05: 95.2, 0.1: 88.1}
```

---

### Feature-Dropout Accuracy

**What it is:** Model accuracy measured after randomly zeroing out input features at various dropout rates.

**Default dropout rates:** `[0.1, 0.2, 0.3, 0.5]`

**When to use:** To assess resilience to missing or corrupted sensor channels.

**Computed in:** `RobustnessTester.test_missing_features(dataloader, dropout_rates)`

---

## Physics-Aware Metrics

These metrics are computed by `PINNEvaluator` (`pinn_evaluator.py`).

### Frequency Consistency

**What it is:** Fraction of predictions that are consistent with the characteristic fault frequencies observed in the signal's frequency spectrum, given the operating conditions (RPM, load).

**Interpretation:** Range [0, 1]. Higher means the model's predictions align with physics expectations. Uses `BearingDynamics` and `FaultSignatureDatabase` for expected frequencies.

**Computed in:** `PINNEvaluator._compute_frequency_consistency(signals, predictions, metadata)`

---

### Prediction Plausibility

**What it is:** Score reflecting whether predictions are physically plausible given operating conditions. For example, high-speed + low-load conditions are unlikely to produce severe bearing faults.

**Interpretation:** Range [0, 1]. Higher is better. Penalizes predictions that contradict known physics relationships.

**Computed in:** `PINNEvaluator._compute_prediction_plausibility(signals, predictions, metadata)`

```python
from packages.core.evaluation.pinn_evaluator import PINNEvaluator

evaluator = PINNEvaluator(pinn_model, device='cuda', sample_rate=20480)
results = evaluator.evaluate_with_physics_metrics(test_loader, metadata=meta)
print(f"Frequency Consistency: {results['physics_metrics']['frequency_consistency']:.3f}")
```

---

## Architecture Comparison Metrics

These metrics are computed by functions in `architecture_comparison.py`.

| Metric               | Function                                      | Description                                      |
| -------------------- | --------------------------------------------- | ------------------------------------------------ |
| Trainable Parameters | `count_parameters(model)`                     | Total number of trainable parameters             |
| FLOPs                | `compute_flops(model, input_shape)`           | Estimated floating-point operations (simplified) |
| Inference Time       | `measure_inference_time(model, input_shape)`  | Mean, std, min, max latency over multiple runs   |
| Memory Usage         | `measure_memory_usage(model, input_shape)`    | Model size, input size, output size, total (MB)  |
| Accuracy             | `evaluate_model_accuracy(model, test_loader)` | Test set accuracy and per-class F1               |

**Combined comparison:** `compare_architectures(model_dict, test_loader)` returns a DataFrame with all metrics.

**Visualization:** `plot_pareto_frontier(df)` plots accuracy vs. parameters vs. inference time.

---

## Performance Results

> ⚠️ **Results pending.** Performance metrics below will be populated
> after experiments are run on the current codebase.

| Metric         | Value       |
| -------------- | ----------- |
| Accuracy       | `[PENDING]` |
| F1 Score       | `[PENDING]` |
| Precision      | `[PENDING]` |
| Recall         | `[PENDING]` |
| AUC            | `[PENDING]` |
| Inference Time | `[PENDING]` |
