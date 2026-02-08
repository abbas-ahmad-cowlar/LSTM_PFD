# XAI Method Guide

> Deep-dive reference for every explainability method implemented in `packages/core/explainability/`.

---

## 1. Integrated Gradients

**File:** `integrated_gradients.py`  
**Class:** `IntegratedGradientsExplainer`  
**Status:** Complete  
**Reference:** Sundararajan, Taly & Yan (2017) — *Axiomatic Attribution for Deep Networks*, ICML.

### Theory

Integrated Gradients computes attributions by integrating the model's gradients along a straight-line path from a baseline input x' to the actual input x:

```
IG_i(x) = (x_i - x'_i) × ∫[α=0→1] ∂F(x' + α(x - x')) / ∂x_i  dα
```

The integral is approximated using the trapezoidal rule with `steps` evenly-spaced α values (default 50).

### Axiomatic Guarantees

| Property | Meaning |
|---|---|
| **Sensitivity** | Any feature that changes the prediction receives non-zero attribution |
| **Implementation Invariance** | Functionally equivalent networks produce identical attributions |
| **Completeness** | Sum of attributions = F(x) − F(x') |

### Usage

```python
from packages.core.explainability import IntegratedGradientsExplainer, plot_attribution_map

explainer = IntegratedGradientsExplainer(model, device='cpu')

# Single-sample explanation
attributions = explainer.explain(
    input_signal,           # [1, C, T] or [C, T]
    target_class=3,         # None → uses argmax
    baseline=None,          # None → zero baseline
    steps=50,
    internal_batch_size=8
)

# Batch explanation
batch_attr = explainer.explain_batch(signals, target_classes=labels, steps=50)

# Convergence verification
delta = explainer.compute_convergence_delta(signal, attributions, target_class=3)
# delta < 0.01 indicates good convergence
```

### Output Format

- **`attributions`**: `torch.Tensor` with same shape as input (`[1, C, T]`)
- **`delta`** (convergence): `float` — absolute difference between attribution sum and F(x)−F(x')

### Visualization

| Function | Description |
|---|---|
| `plot_attribution_map(signal, attributions, predicted_class, ...)` | Three-panel plot: signal, attributions, overlay |
| `plot_attribution_heatmap(attributions, predicted_class, ...)` | Channel × Time heatmap (multi-channel) |

### Limitations

- Baseline choice affects results (default is zero, which may not be meaningful for all signals)
- Computation scales linearly with `steps` — more steps = slower but more accurate
- Requires gradient access (only differentiable PyTorch models)

---

## 2. SHAP (SHapley Additive exPlanations)

**File:** `shap_explainer.py`  
**Class:** `SHAPExplainer`  
**Status:** Complete (3 backends)  
**Reference:** Lundberg & Lee (2017) — *A Unified Approach to Interpreting Model Predictions*, NeurIPS.

### Theory

SHAP values are Shapley values from coalitional game theory. They satisfy:

| Axiom | Meaning |
|---|---|
| **Local accuracy** | f(x) = φ₀ + Σφᵢ |
| **Missingness** | Missing features get φ = 0 |
| **Consistency** | If a feature's marginal contribution increases, its Shapley value does not decrease |

### Supported Backends

| Backend | Method arg | Requires `shap` lib | Speed | Notes |
|---|---|---|---|---|
| **GradientSHAP** | `'gradient'` | No | Fast | Native PyTorch; samples random baselines from `background_data` |
| **DeepSHAP** | `'deep'` | Yes | Medium | Wraps `shap.DeepExplainer` |
| **KernelSHAP** | `'kernel'` | No | Slow | Model-agnostic; segments signal into 20 patches, fits linear regression |

### Usage

```python
from packages.core.explainability import SHAPExplainer, plot_shap_waterfall, plot_shap_summary

explainer = SHAPExplainer(
    model,
    background_data=bg_data,  # [N, C, T] — optional but recommended
    device='cpu',
    use_shap_library=True      # tries `import shap`, falls back gracefully
)

shap_values = explainer.explain(
    input_signal,        # [1, C, T] or [C, T]
    method='gradient',   # 'gradient' | 'deep' | 'kernel'
    n_samples=100
)
```

### Output Format

- **`shap_values`**: `torch.Tensor` with same shape as input

### Visualization

| Function | Description |
|---|---|
| `plot_shap_waterfall(shap_values, base_value, predicted_value, ...)` | Waterfall chart showing per-feature contribution |
| `plot_shap_summary(shap_values_batch, signals_batch, ...)` | Swarm plot across batch (feature × SHAP value, colored by feature value) |

### Limitations

- `background_data` is required for `_deep_shap`; recommended for `_gradient_shap` (uses zeros otherwise)
- KernelSHAP segments the signal into a fixed 20 patches — resolution is limited
- DeepSHAP requires `pip install shap`

---

## 3. LIME (Local Interpretable Model-agnostic Explanations)

**File:** `lime_explainer.py`  
**Class:** `LIMEExplainer`  
**Status:** Complete  
**Reference:** Ribeiro, Singh & Guestrin (2016) — *"Why Should I Trust You?"*, KDD.

### Theory

LIME divides the input signal into `num_segments` segments, generates perturbed copies by randomly masking segments (replacing with zeros), observes how predictions change, and fits a weighted linear model (Ridge regression) to approximate the local decision boundary.

### Usage

```python
from packages.core.explainability import LIMEExplainer, plot_lime_explanation

explainer = LIMEExplainer(
    model,
    device='cpu',
    num_segments=20,
    kernel_width=0.25
)

segment_weights, segment_boundaries = explainer.explain(
    input_signal,          # [1, C, T] or [C, T]
    target_class=None,     # None → uses argmax
    num_samples=1000,
    distance_metric='cosine',
    model_regressor=None   # None → Ridge(alpha=1.0)
)
```

### Output Format

- **`segment_weights`**: `np.ndarray [num_segments]` — linear model coefficients (importance of each segment)
- **`segment_boundaries`**: `List[Tuple[int, int]]` — `(start, end)` index pairs

### Visualization

| Function | Description |
|---|---|
| `plot_lime_explanation(signal, weights, boundaries, ...)` | Three-panel: signal, segment bars, color overlay |
| `plot_lime_bar_chart(segment_weights, top_k=10, ...)` | Horizontal bar chart of top-K segment importances |

### Limitations

- Masking with zeros may not be a neutral baseline for all signal types
- Explanation depends on `num_segments` and `kernel_width` choices
- Cosine distance metric assumes meaningful notion of direction in segment space

---

## 4. Uncertainty Quantification (MC Dropout)

**File:** `uncertainty_quantification.py`  
**Class:** `UncertaintyQuantifier`  
**Status:** Complete  
**Reference:** Gal & Ghahramani (2016) — *Dropout as a Bayesian Approximation*.

### Theory

Monte Carlo Dropout performs `n_samples` stochastic forward passes with dropout enabled at inference time. The variance across predictions gives calibrated uncertainty. Additionally:

- **Entropy**: `H(E[p])` — high for uniform (uncertain) distributions
- **Mutual Information**: `MI = H(E[p]) − E[H(p)]` — isolates epistemic uncertainty (model uncertainty from insufficient data)

### Usage

```python
from packages.core.explainability import (
    UncertaintyQuantifier, calibrate_model,
    plot_calibration_curve, plot_prediction_with_uncertainty
)

uq = UncertaintyQuantifier(model, device='cpu')

mean_pred, uncertainty, all_preds = uq.predict_with_uncertainty(
    signal, n_samples=50, return_all=True
)

# Entropy-based uncertainty
entropy = uq.entropy_based_uncertainty(mean_pred)

# Epistemic uncertainty (mutual information)
mi = uq.mutual_information(all_preds)

# Reject uncertain predictions
confident_preds, rejected_idx = uq.reject_uncertain_predictions(
    mean_pred, uncertainty, threshold=0.2
)

# Calibration analysis
prob_true, prob_pred, ece = calibrate_model(model, dataloader, device='cpu')
```

### Output Format

| Output | Type | Shape |
|---|---|---|
| `mean_pred` | `torch.Tensor` | `[B, num_classes]` |
| `uncertainty` | `torch.Tensor` | `[B, num_classes]` |
| `all_preds` | `torch.Tensor` | `[n_samples, B, num_classes]` (if `return_all=True`) |
| `entropy` | `torch.Tensor` | `[B]` |
| `mi` | `torch.Tensor` | `[B]` |
| `ece` | `float` | scalar |

### Visualization

| Function | Description |
|---|---|
| `plot_calibration_curve(prob_true, prob_pred, ece, ...)` | Reliability diagram |
| `plot_uncertainty_distribution(uncertainties, correct_mask, ...)` | Correct vs. incorrect uncertainty histogram |
| `plot_prediction_with_uncertainty(signal, mean_pred, uncertainty, ...)` | Signal + bar chart with error bars |

### Limitations

- Requires the model to have dropout layers (dropout is forced on at inference via `model.train()` during MC sampling)
- Quality depends on `n_samples` — more forward passes = better estimates but slower
- Deep Ensembles are mentioned in module docstring but not implemented as a separate class

---

## 5. Concept Activation Vectors (CAV) & TCAV

**File:** `concept_activation_vectors.py`  
**Classes:** `ConceptActivationVector`, `CAVGenerator`, `TCAVAnalyzer`  
**Status:** Complete  
**Reference:** Kim et al. (2018) — *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)*, ICML.

### Theory

A CAV is the normal vector to the hyperplane that separates concept examples from random examples in a layer's activation space (trained via Linear SVM or Logistic Regression). TCAV score measures the fraction of test examples where the directional derivative in the CAV direction is positive.

### Usage

```python
from packages.core.explainability import CAVGenerator, TCAVAnalyzer, plot_tcav_results

# Step 1: Generate CAV
cav_gen = CAVGenerator(model, device='cpu', classifier_type='linear_svm')
cav = cav_gen.generate_cav(
    concept_examples,   # [N_concept, C, T]
    random_examples,    # [N_random, C, T]
    layer_name='conv1',
    concept_name='high_frequency_noise'
)

# Step 2: TCAV testing
tcav = TCAVAnalyzer(model, cav, device='cpu')
results = tcav.compute_tcav_score(
    test_examples,      # [N, C, T]
    target_class=3,
    n_random_runs=10
)
# results = {'tcav_score': float, 'p_value': float, 'significant': bool}
```

### Output Format

| Output | Type |
|---|---|
| `cav.vector` | `np.ndarray` — normal vector |
| `cav.accuracy` | `float` — classifier test accuracy |
| `results['tcav_score']` | `float` in [0, 1] |
| `results['p_value']` | `float` — statistical significance |

### Visualization

| Function | Description |
|---|---|
| `plot_tcav_results(results)` | TCAV score bar chart |
| `plot_cav_comparison(cavs)` | Compare CAVs across concepts |

### Limitations

- Requires user to supply concept examples and random examples — quality depends on dataset curation
- Layer selection significantly affects results
- Classifier accuracy below ~60% indicates the concept may not be learnable at that layer

---

## 6. Partial Dependence (PDP) & ICE

**File:** `partial_dependence.py`  
**Class:** `PartialDependenceAnalyzer`  
**Status:** Complete

### Theory

Partial Dependence shows the average effect of a feature on predictions:

```
PD(x_j) = E_X[f(x_j, X_C)]
```

ICE (Individual Conditional Expectation) shows per-sample curves, revealing heterogeneous effects that PD averaging may hide.

### Usage

```python
from packages.core.explainability import (
    PartialDependenceAnalyzer, plot_partial_dependence,
    plot_ice_curves, plot_partial_dependence_2d, detect_interactions
)

pd_analyzer = PartialDependenceAnalyzer(
    model,
    feature_extractor=my_feature_fn,  # signal → feature vector (or None for identity)
    device='cpu'
)

# 1D PDP
grid_values, pd_values = pd_analyzer.partial_dependence_1d(
    X, feature_idx=0, grid_resolution=50, target_class=3
)

# ICE curves
grid_values, ice_curves = pd_analyzer.ice_plot_1d(
    X, feature_idx=0, grid_resolution=50, max_samples=100
)

# 2D PDP (feature interactions)
grid1, grid2, pd_2d = pd_analyzer.partial_dependence_2d(
    X, feature_idx1=0, feature_idx2=1, grid_resolution=30
)

# H-statistic for interaction detection
h_stat = detect_interactions(grid1, grid2, pd_2d, pd_1d_f1, pd_1d_f2)
```

### Visualization

| Function | Description |
|---|---|
| `plot_partial_dependence(grid, pd_values, ...)` | 1D PDP line plot |
| `plot_ice_curves(grid, ice_curves, ...)` | ICE with optional PD overlay |
| `plot_partial_dependence_2d(grid1, grid2, pd_2d, ...)` | Contour / heatmap for 2 features |

### Limitations

- Feature extractor must be provided if the model operates on raw signals (not features)
- Computational cost scales with `grid_resolution × N_samples × N_features`
- Assumes features are independent when computing marginal effects (standard PD assumption)

---

## 7. Anchors (Rule-Based Explanations)

**File:** `anchors.py`  
**Classes:** `Predicate`, `Anchor`, `AnchorExplainer`  
**Status:** Complete  
**Reference:** Ribeiro, Singh & Guestrin (2018) — *Anchors: High-Precision Model-Agnostic Explanations*, AAAI.

### Theory

An Anchor is a set of IF-THEN predicates such that, if all predicates hold, the model's prediction is the same with high probability (≥ `precision_threshold`). The algorithm uses beam search to find minimal, high-precision anchors.

### Usage

```python
from packages.core.explainability import AnchorExplainer, plot_anchor_explanation

explainer = AnchorExplainer(
    model,
    feature_extractor=my_feature_fn,
    feature_names=['mean_amplitude', 'dominant_freq', 'rms', ...],
    device='cpu',
    precision_threshold=0.95,
    beam_size=5,
    max_predicates=5
)

anchor = explainer.explain(signal, n_samples=1000, verbose=True)
print(anchor)
# Anchor (precision=0.97, coverage=0.34, class=3):
#   mean_amplitude > 1.5 AND dominant_freq in [80.0, 120.0]
```

### Output Format

- **`Anchor`** dataclass with:
  - `predicates: List[Predicate]` — each `Predicate` has `feature_name`, `operator`, `threshold`
  - `precision: float` — P(same prediction | anchor holds)
  - `coverage: float` — P(anchor holds)
  - `target_class: int`

### Visualization

| Function | Description |
|---|---|
| `plot_anchor_explanation(anchor, features, feature_names, ...)` | Bar chart of anchor predicates |
| `compare_anchors(anchors, ...)` | Multi-anchor comparison |

### Limitations

- Requires a user-supplied `feature_extractor` function and `feature_names` list
- Perturbation uses Gaussian noise with varying levels — may not be realistic for all domains
- Beam search is approximate; larger `beam_size` improves quality but increases computation
