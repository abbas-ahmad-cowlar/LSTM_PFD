# Explainability API Reference

## Classes

---

### `IntegratedGradientsExplainer`

> Gradient-path attribution explainer (Sundararajan et al., 2017).

**Constructor:**

```python
IntegratedGradientsExplainer(model: nn.Module, device: str = 'cuda')
```

| Parameter | Type        | Default  | Description              |
| --------- | ----------- | -------- | ------------------------ |
| `model`   | `nn.Module` | required | PyTorch model to explain |
| `device`  | `str`       | `'cuda'` | Computation device       |

**Methods:**

#### `explain(input_signal, target_class=None, baseline=None, steps=50, internal_batch_size=8) → torch.Tensor`

Compute Integrated Gradients attributions.

| Parameter             | Type                     | Default  | Description                     |
| --------------------- | ------------------------ | -------- | ------------------------------- |
| `input_signal`        | `torch.Tensor`           | required | `[1, C, T]` or `[C, T]`         |
| `target_class`        | `Optional[int]`          | `None`   | Target class (None → argmax)    |
| `baseline`            | `Optional[torch.Tensor]` | `None`   | Baseline input (None → zeros)   |
| `steps`               | `int`                    | `50`     | Integration steps               |
| `internal_batch_size` | `int`                    | `8`      | Gradient computation batch size |

**Returns:** `torch.Tensor` — attributions with same shape as input.

**Example:**

```python
explainer = IntegratedGradientsExplainer(model, device='cpu')
attributions = explainer.explain(signal, target_class=3, steps=50)
```

#### `explain_batch(input_signals, target_classes=None, baseline=None, steps=50) → torch.Tensor`

Compute Integrated Gradients for a batch of inputs.

| Parameter        | Type                     | Default  | Description          |
| ---------------- | ------------------------ | -------- | -------------------- |
| `input_signals`  | `torch.Tensor`           | required | `[B, C, T]`          |
| `target_classes` | `Optional[torch.Tensor]` | `None`   | `[B]` target classes |
| `baseline`       | `Optional[torch.Tensor]` | `None`   | Baseline input       |
| `steps`          | `int`                    | `50`     | Integration steps    |

**Returns:** `torch.Tensor [B, C, T]`

#### `compute_convergence_delta(input_signal, attributions, target_class, baseline=None) → float`

Verify attribution quality via completeness property. Returns absolute difference between attribution sum and `F(x) − F(x')`.

---

### `SHAPExplainer`

> Shapley-value explanations with multiple backends.

**Constructor:**

```python
SHAPExplainer(
    model: nn.Module,
    background_data: Optional[torch.Tensor] = None,
    device: str = 'cuda',
    use_shap_library: bool = True
)
```

| Parameter          | Type                     | Default  | Description                    |
| ------------------ | ------------------------ | -------- | ------------------------------ |
| `model`            | `nn.Module`              | required | PyTorch model                  |
| `background_data`  | `Optional[torch.Tensor]` | `None`   | Background dataset `[N, C, T]` |
| `device`           | `str`                    | `'cuda'` | Device                         |
| `use_shap_library` | `bool`                   | `True`   | Try to import `shap` library   |

**Methods:**

#### `explain(input_signal, method='gradient', n_samples=100) → torch.Tensor`

Compute SHAP values.

| Parameter      | Type           | Default      | Description                           |
| -------------- | -------------- | ------------ | ------------------------------------- |
| `input_signal` | `torch.Tensor` | required     | `[1, C, T]` or `[C, T]`               |
| `method`       | `str`          | `'gradient'` | `'gradient'`, `'deep'`, or `'kernel'` |
| `n_samples`    | `int`          | `100`        | Approximation samples                 |

**Returns:** `torch.Tensor` — SHAP values with same shape as input.

**Example:**

```python
explainer = SHAPExplainer(model, background_data=bg, device='cpu')
shap_values = explainer.explain(signal, method='gradient', n_samples=100)
```

---

### `LIMEExplainer`

> Segment-perturbation local linear model.

**Constructor:**

```python
LIMEExplainer(
    model: nn.Module,
    device: str = 'cuda',
    num_segments: int = 20,
    kernel_width: float = 0.25
)
```

| Parameter      | Type        | Default  | Description                  |
| -------------- | ----------- | -------- | ---------------------------- |
| `model`        | `nn.Module` | required | PyTorch model                |
| `device`       | `str`       | `'cuda'` | Device                       |
| `num_segments` | `int`       | `20`     | Number of signal segments    |
| `kernel_width` | `float`     | `0.25`   | Exponential kernel bandwidth |

**Methods:**

#### `explain(input_signal, target_class=None, num_samples=1000, distance_metric='cosine', model_regressor=None) → Tuple[np.ndarray, List[Tuple[int, int]]]`

Explain prediction using LIME.

| Parameter         | Type            | Default    | Description                 |
| ----------------- | --------------- | ---------- | --------------------------- |
| `input_signal`    | `torch.Tensor`  | required   | `[1, C, T]` or `[C, T]`     |
| `target_class`    | `Optional[int]` | `None`     | None → argmax               |
| `num_samples`     | `int`           | `1000`     | Perturbed samples           |
| `distance_metric` | `str`           | `'cosine'` | Distance metric             |
| `model_regressor` | `Optional`      | `None`     | Linear model (None → Ridge) |

**Returns:** `(segment_weights: np.ndarray, segment_boundaries: List[Tuple[int, int]])`

**Example:**

```python
explainer = LIMEExplainer(model, device='cpu', num_segments=20)
weights, boundaries = explainer.explain(signal, num_samples=1000)
```

---

### `UncertaintyQuantifier`

> Monte Carlo Dropout uncertainty estimation.

**Constructor:**

```python
UncertaintyQuantifier(model: nn.Module, device: str = 'cuda')
```

| Parameter | Type        | Default  | Description               |
| --------- | ----------- | -------- | ------------------------- |
| `model`   | `nn.Module` | required | Model with dropout layers |
| `device`  | `str`       | `'cuda'` | Device                    |

**Methods:**

#### `predict_with_uncertainty(input_signal, n_samples=50, return_all=False) → Tuple`

MC Dropout inference.

| Parameter      | Type           | Default  | Description               |
| -------------- | -------------- | -------- | ------------------------- |
| `input_signal` | `torch.Tensor` | required | `[B, C, T]`               |
| `n_samples`    | `int`          | `50`     | MC forward passes         |
| `return_all`   | `bool`         | `False`  | Return all MC predictions |

**Returns:** `(mean_prediction [B, C], uncertainty [B, C], all_predictions [n, B, C] if return_all)`

#### `entropy_based_uncertainty(mean_prediction) → torch.Tensor`

Entropy of mean prediction distribution. Returns `[B]`.

#### `mutual_information(predictions) → torch.Tensor`

Epistemic uncertainty via MI = H(E[p]) − E[H(p)]. Input: `[n_samples, B, num_classes]`. Returns `[B]`.

#### `reject_uncertain_predictions(mean_prediction, uncertainty, threshold=0.2) → Tuple`

Flag high-uncertainty predictions. Returns `(confident_preds, rejected_indices)`.

**Example:**

```python
uq = UncertaintyQuantifier(model, device='cpu')
mean, unc, all_p = uq.predict_with_uncertainty(signal, n_samples=50, return_all=True)
entropy = uq.entropy_based_uncertainty(mean)
```

---

### `ConceptActivationVector`

> Dataclass representing a learned concept vector.

| Field          | Type         | Description                    |
| -------------- | ------------ | ------------------------------ |
| `concept_name` | `str`        | Human-readable concept name    |
| `layer_name`   | `str`        | Layer where CAV was computed   |
| `vector`       | `np.ndarray` | CAV vector (hyperplane normal) |
| `accuracy`     | `float`      | Classifier test accuracy       |
| `classifier`   | `any`        | Trained classifier object      |

---

### `CAVGenerator`

> Generates Concept Activation Vectors.

**Constructor:**

```python
CAVGenerator(
    model: nn.Module,
    device: str = 'cuda',
    classifier_type: str = 'linear_svm'
)
```

| Parameter         | Type        | Default        | Description                    |
| ----------------- | ----------- | -------------- | ------------------------------ |
| `model`           | `nn.Module` | required       | PyTorch model                  |
| `device`          | `str`       | `'cuda'`       | Device                         |
| `classifier_type` | `str`       | `'linear_svm'` | `'linear_svm'` or `'logistic'` |

**Methods:**

#### `generate_cav(concept_examples, random_examples, layer_name, concept_name='concept', test_size=0.2, random_state=42) → ConceptActivationVector`

Train a CAV at a specific layer.

| Parameter          | Type           | Default     | Description         |
| ------------------ | -------------- | ----------- | ------------------- |
| `concept_examples` | `torch.Tensor` | required    | `[N_concept, C, T]` |
| `random_examples`  | `torch.Tensor` | required    | `[N_random, C, T]`  |
| `layer_name`       | `str`          | required    | Target layer name   |
| `concept_name`     | `str`          | `'concept'` | Human-readable name |

**Returns:** `ConceptActivationVector`

---

### `TCAVAnalyzer`

> Tests concept influence on model predictions.

**Constructor:**

```python
TCAVAnalyzer(model: nn.Module, cav: ConceptActivationVector, device: str = 'cuda')
```

**Methods:**

#### `compute_tcav_score(test_examples, target_class, n_random_runs=10) → Dict`

Compute TCAV score with statistical significance.

| Parameter       | Type           | Default  | Description                 |
| --------------- | -------------- | -------- | --------------------------- |
| `test_examples` | `torch.Tensor` | required | `[N, C, T]`                 |
| `target_class`  | `int`          | required | Target class                |
| `n_random_runs` | `int`          | `10`     | Random CAV runs for p-value |

**Returns:** `{'tcav_score': float, 'p_value': float, 'significant': bool}`

---

### `PartialDependenceAnalyzer`

> Computes PDP and ICE for feature effect analysis.

**Constructor:**

```python
PartialDependenceAnalyzer(
    model: nn.Module,
    feature_extractor: Optional[Callable] = None,
    device: str = 'cuda'
)
```

| Parameter           | Type                 | Default  | Description                         |
| ------------------- | -------------------- | -------- | ----------------------------------- |
| `model`             | `nn.Module`          | required | PyTorch model                       |
| `feature_extractor` | `Optional[Callable]` | `None`   | signal → features (None = identity) |
| `device`            | `str`                | `'cuda'` | Device                              |

**Methods:**

#### `partial_dependence_1d(X, feature_idx, grid_resolution=50, grid_range=None, target_class=None, percentile_range=(5, 95)) → Tuple`

**Returns:** `(grid_values: np.ndarray, pd_values: np.ndarray)`

#### `ice_plot_1d(X, feature_idx, grid_resolution=50, grid_range=None, target_class=None, percentile_range=(5, 95), max_samples=100) → Tuple`

**Returns:** `(grid_values: np.ndarray, ice_curves: np.ndarray [N_samples, N_grid])`

#### `partial_dependence_2d(X, feature_idx1, feature_idx2, grid_resolution=30, target_class=None) → Tuple`

**Returns:** `(grid1, grid2, pd_values: np.ndarray [len(grid1), len(grid2)])`

---

### `Predicate`

> Dataclass — a single condition in an anchor rule.

| Field           | Type              | Description                   |
| --------------- | ----------------- | ----------------------------- |
| `feature_name`  | `str`             | Feature name                  |
| `feature_idx`   | `int`             | Feature index                 |
| `operator`      | `str`             | `'>'`, `'<'`, `'between'`     |
| `threshold`     | `float`           | Threshold value               |
| `threshold_max` | `Optional[float]` | Upper bound (for `'between'`) |

#### `evaluate(feature_value: float) → bool`

Check if feature satisfies predicate.

---

### `Anchor`

> Dataclass — a set of predicates that anchor a prediction.

| Field          | Type              | Description                   |
| -------------- | ----------------- | ----------------------------- |
| `predicates`   | `List[Predicate]` | Rule conditions               |
| `precision`    | `float`           | P(same class \| anchor holds) |
| `coverage`     | `float`           | P(anchor holds)               |
| `target_class` | `int`             | Predicted class               |

#### `evaluate(features: np.ndarray) → bool`

Check if all predicates are satisfied.

---

### `AnchorExplainer`

> Beam-search anchor generator.

**Constructor:**

```python
AnchorExplainer(
    model: nn.Module,
    feature_extractor: Callable,
    feature_names: List[str],
    device: str = 'cuda',
    precision_threshold: float = 0.95,
    beam_size: int = 5,
    max_predicates: int = 5
)
```

| Parameter             | Type        | Default  | Description              |
| --------------------- | ----------- | -------- | ------------------------ |
| `model`               | `nn.Module` | required | PyTorch model            |
| `feature_extractor`   | `Callable`  | required | signal → feature vector  |
| `feature_names`       | `List[str]` | required | Feature names            |
| `precision_threshold` | `float`     | `0.95`   | Minimum anchor precision |
| `beam_size`           | `int`       | `5`      | Beam search width        |
| `max_predicates`      | `int`       | `5`      | Max rules per anchor     |

**Methods:**

#### `explain(input_signal, n_samples=1000, verbose=True) → Anchor`

Generate anchor explanation.

**Example:**

```python
explainer = AnchorExplainer(model, feat_fn, feat_names, device='cpu')
anchor = explainer.explain(signal, n_samples=1000)
print(anchor)  # Human-readable IF-THEN rules
```

---

## Standalone Functions

### Integrated Gradients Visualization

| Function                   | Signature                                                                                                    |
| -------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `plot_attribution_map`     | `(signal, attributions, predicted_class, true_class=None, class_names=None, save_path=None, show_plot=True)` |
| `plot_attribution_heatmap` | `(attributions, predicted_class, class_names=None, save_path=None)`                                          |

### SHAP Visualization

| Function              | Signature                                                                                        |
| --------------------- | ------------------------------------------------------------------------------------------------ |
| `plot_shap_waterfall` | `(shap_values, base_value, predicted_value, feature_names=None, max_display=20, save_path=None)` |
| `plot_shap_summary`   | `(shap_values_batch, signals_batch, max_display=20, save_path=None)`                             |

### LIME Visualization

| Function                | Signature                                                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------- |
| `plot_lime_explanation` | `(signal, segment_weights, segment_boundaries, predicted_class, true_class=None, class_names=None, save_path=None)` |
| `plot_lime_bar_chart`   | `(segment_weights, top_k=10, class_name=None, save_path=None)`                                                      |

### Uncertainty Visualization

| Function                           | Signature                                                                       |
| ---------------------------------- | ------------------------------------------------------------------------------- |
| `calibrate_model`                  | `(model, dataloader, device='cuda', num_bins=10) → (prob_true, prob_pred, ece)` |
| `plot_calibration_curve`           | `(prob_true, prob_pred, ece, model_name='Model', save_path=None)`               |
| `plot_uncertainty_distribution`    | `(uncertainties, correct_mask, save_path=None)`                                 |
| `plot_prediction_with_uncertainty` | `(signal, mean_prediction, uncertainty, class_names=None, save_path=None)`      |

### CAV/TCAV Visualization

| Function              | Signature   |
| --------------------- | ----------- |
| `plot_tcav_results`   | `(results)` |
| `plot_cav_comparison` | `(cavs)`    |

### Partial Dependence Visualization

| Function                     | Signature                                                                                                              |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `plot_partial_dependence`    | `(grid_values, pd_values, feature_name='Feature', target_class=None, class_names=None, save_path=None)`                |
| `plot_ice_curves`            | `(grid_values, ice_curves, feature_name='Feature', show_pd=True, target_class=None, class_names=None, save_path=None)` |
| `plot_partial_dependence_2d` | `(grid1, grid2, pd_values, feature_name1, feature_name2, target_class=None, class_names=None, save_path=None)`         |
| `detect_interactions`        | `(grid1, grid2, pd_2d, pd_1d_feature1, pd_1d_feature2) → float`                                                        |

### Anchor Visualization

| Function                  | Signature                                                    |
| ------------------------- | ------------------------------------------------------------ |
| `plot_anchor_explanation` | `(anchor, original_features, feature_names, save_path=None)` |
| `compare_anchors`         | `(anchors, save_path=None)`                                  |
