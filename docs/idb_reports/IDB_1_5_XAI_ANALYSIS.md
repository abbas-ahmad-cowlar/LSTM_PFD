# IDB 1.5: Explainability Sub-Block Analysis

> **Analysis Date**: 2026-01-23
> **Analyst**: AI Agent (Explainability Sub-Block Analyst)
> **Domain**: Core ML Engine
> **Primary Directory**: `packages/core/explainability/`

---

## Executive Summary

The Explainability Sub-Block implements **7 distinct XAI methodologies** across 8 Python files (~3,074 total lines). The implementation demonstrates strong scientific grounding with proper academic references. However, several architectural and performance concerns exist, including the **absence of the expected caching mechanism** and lack of a unified explainer interface.

### At a Glance

| Metric                  | Value                         |
| ----------------------- | ----------------------------- |
| Total Files             | 8 (including `__init__.py`)   |
| Total LOC               | ~3,074                        |
| XAI Methods             | 7 (with 3 SHAP variants)      |
| Visualization Functions | 12                            |
| Missing Components      | `explanation_cache.py`        |
| Test Coverage           | Inline `__main__` blocks only |

---

## Task 1: Current State Assessment

### 1.1 XAI Methods Inventory

| Method                   | File                            | Class/Function                   | Scientific Reference                   |
| ------------------------ | ------------------------------- | -------------------------------- | -------------------------------------- |
| **Integrated Gradients** | `integrated_gradients.py`       | `IntegratedGradientsExplainer`   | Sundararajan et al., 2017 (ICML)       |
| **GradientSHAP**         | `shap_explainer.py`             | `SHAPExplainer._gradient_shap()` | Lundberg & Lee, 2017 (NeurIPS)         |
| **DeepSHAP**             | `shap_explainer.py`             | `SHAPExplainer._deep_shap()`     | Lundberg & Lee, 2017 (NeurIPS)         |
| **KernelSHAP**           | `shap_explainer.py`             | `SHAPExplainer._kernel_shap()`   | Lundberg & Lee, 2017 (NeurIPS)         |
| **LIME**                 | `lime_explainer.py`             | `LIMEExplainer`                  | Ribeiro et al., 2016 (KDD)             |
| **MC Dropout**           | `uncertainty_quantification.py` | `UncertaintyQuantifier`          | Gal & Ghahramani, 2016 (ICML)          |
| **CAVs/TCAV**            | `concept_activation_vectors.py` | `CAVGenerator`, `TCAVAnalyzer`   | Kim et al., 2018 (ICML)                |
| **Partial Dependence**   | `partial_dependence.py`         | `PartialDependenceAnalyzer`      | Friedman, 2001; Goldstein et al., 2015 |
| **ICE Plots**            | `partial_dependence.py`         | `ice_plot_1d()`                  | Goldstein et al., 2015                 |
| **Anchors**              | `anchors.py`                    | `AnchorExplainer`                | Ribeiro et al., 2018 (AAAI)            |

### 1.2 Caching Mechanism Status

> [!CAUTION]
> **`explanation_cache.py` does not exist.** The expected caching mechanism is completely absent from the codebase.

**Impact**:

- Repeated explanations for same inputs require full recomputation
- No memoization of expensive gradient calculations
- Performance penalty for dashboard integration

### 1.3 Visualization Outputs

| Visualization            | Function                                                                         | Location                        |
| ------------------------ | -------------------------------------------------------------------------------- | ------------------------------- |
| Attribution Map          | `plot_attribution_map()`                                                         | `integrated_gradients.py`       |
| Attribution Heatmap      | `plot_attribution_heatmap()`                                                     | `integrated_gradients.py`       |
| SHAP Waterfall           | `plot_shap_waterfall()`                                                          | `shap_explainer.py`             |
| SHAP Summary             | `plot_shap_summary()`                                                            | `shap_explainer.py`             |
| LIME Explanation         | `plot_lime_explanation()`                                                        | `lime_explainer.py`             |
| LIME Bar Chart           | `plot_lime_bar_chart()`                                                          | `lime_explainer.py`             |
| Calibration Curve        | `plot_calibration_curve()`                                                       | `uncertainty_quantification.py` |
| Uncertainty Distribution | `plot_uncertainty_distribution()`                                                | `uncertainty_quantification.py` |
| Prediction + Uncertainty | `plot_prediction_with_uncertainty()`                                             | `uncertainty_quantification.py` |
| TCAV Results             | `plot_tcav_results()`                                                            | `concept_activation_vectors.py` |
| CAV Comparison           | `plot_cav_comparison()`                                                          | `concept_activation_vectors.py` |
| Partial Dependence       | `plot_partial_dependence()`, `plot_ice_curves()`, `plot_partial_dependence_2d()` | `partial_dependence.py`         |
| Anchor Explanation       | `plot_anchor_explanation()`, `compare_anchors()`                                 | `anchors.py`                    |

### 1.4 Implementation Compliance with XAI Best Practices

| Best Practice                    | Status     | Notes                                     |
| -------------------------------- | ---------- | ----------------------------------------- |
| Proper baseline selection        | ⚠️ Partial | Zero baseline default; configurable in IG |
| Convergence checks               | ✅ Yes     | `compute_convergence_delta()` in IG       |
| Statistical significance testing | ✅ Yes     | TCAV p-value computation                  |
| Uncertainty quantification       | ✅ Yes     | Full MC Dropout + calibration             |
| Human-interpretable output       | ✅ Yes     | Anchors produce IF-THEN rules             |
| Model-agnostic methods           | ✅ Yes     | LIME, Anchors, KernelSHAP                 |
| Gradient-based methods           | ✅ Yes     | IG, GradientSHAP                          |

---

## Task 2: Critical Issues Identification

### P0 (Critical) Issues

#### 2.1 Missing Caching Infrastructure

```
Severity: P0 (Critical)
Location: Entire package
Impact: Performance, Production Readiness
```

**Problem**: No caching mechanism exists. Each explanation recomputes from scratch:

- Integrated Gradients: 50+ forward passes per sample
- SHAP: 100+ samples per explanation
- LIME: 1000 perturbation samples default
- CAVs: Linear classifier retraining per concept

**Recommendation**: Implement `explanation_cache.py` with:

```python
class ExplanationCache:
    def __init__(self, cache_dir: Path, max_size_mb: int = 1000):
        ...

    def get_or_compute(self, key: str, compute_fn: Callable) -> Any:
        ...

    def invalidate(self, model_hash: str):
        ...
```

---

#### 2.2 KernelSHAP Linear Regression Without Proper Weighting

```
Severity: P0 (Critical)
Location: shap_explainer.py:253-261
Impact: Incorrect Attributions
```

**Problem** (Lines 253-261):

```python
# Fit linear model (simplified SHAP)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(masks, target_predictions)

# Segment-level SHAP values
segment_shap = lr.coef_
```

This is **scientifically incorrect**. True KernelSHAP requires:

1. SHAP kernel weighting: `w(z) = (M-1) / (C(M,|z|) × |z| × (M-|z|))`
2. Weighted least squares regression

**Current Implementation**: Uses unweighted `LinearRegression()` which violates SHAP's theoretical guarantees.

**Fix**:

```python
from sklearn.linear_model import Ridge

# Compute proper SHAP kernel weights
n_features = masks.sum(axis=1)
kernel_weights = (n_segments - 1) / (scipy.special.comb(n_segments, n_features) * n_features * (n_segments - n_features) + 1e-8)

# Weighted regression
ridge = Ridge(alpha=0.01)
ridge.fit(masks, target_predictions, sample_weight=kernel_weights)
```

---

### P1 (High) Issues

#### 2.3 GradientSHAP Accumulates Wrong Gradient

```
Severity: P1 (High)
Location: shap_explainer.py:169
Impact: Attribution Accuracy
```

**Problem** (Line 169):

```python
total_gradients += interpolated.grad
```

Uses `interpolated.grad` but `interpolated` is recreated each loop. The gradient is correctly computed but the final multiplication uses `input_signal - mean_baseline` instead of per-sample baseline difference.

**Lines 174-176**:

```python
mean_baseline = baselines.mean(dim=0, keepdim=True)
shap_values = avg_gradients * (input_signal - mean_baseline)
```

This approximation is acceptable for GradientSHAP but differs from the canonical formulation which uses per-sample baseline differences.

---

#### 2.4 LIME Segment Boundary Edge Case

```
Severity: P1 (High)
Location: lime_explainer.py:146-152
Impact: Off-by-one in final segment
```

**Problem**:

```python
for i in range(self.num_segments):
    start = i * segment_size
    end = (i + 1) * segment_size if i < self.num_segments - 1 else signal_length
```

When `signal_length % num_segments != 0`, the last segment may be significantly larger than others, biasing importance scores.

**Example**: Signal length 10240, 20 segments → segment_size = 512, but last segment could be 512+ if there's remainder.

---

#### 2.5 MC Dropout Mode Not Restored on Exception

```
Severity: P1 (High)
Location: uncertainty_quantification.py:81-100
Impact: Model State Corruption
```

**Problem**:

```python
# Enable dropout during inference
self.model.train()  # Line 81

# ... (no try/finally)

# Return to eval mode
self.model.eval()  # Line 100
```

If an exception occurs between lines 81-100, the model remains in training mode, affecting all subsequent predictions.

**Fix**:

```python
try:
    self.model.train()
    # ... MC sampling
finally:
    self.model.eval()
```

---

#### 2.6 CAV Gradient Hook Memory Leak Risk

```
Severity: P1 (High)
Location: concept_activation_vectors.py:355-379
Impact: Memory Leak
```

**Problem**: In `_compute_directional_derivatives()`, a hook is registered per sample in a loop:

```python
for input_sample in inputs:
    ...
    handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        ...
    finally:
        handle.remove()
```

While the hook is removed, the inner function closes over `activation_grads` list which persists. For large N, this causes memory growth.

---

### P2 (Medium) Issues

#### 2.7 Hardcoded Parameters

| Location                     | Parameter             | Value | Issue                                         |
| ---------------------------- | --------------------- | ----- | --------------------------------------------- |
| `shap_explainer.py:223`      | `n_segments`          | 20    | Not configurable in KernelSHAP                |
| `lime_explainer.py:71`       | `num_samples`         | 1000  | Default may be too high for fast explanations |
| `integrated_gradients.py:63` | `internal_batch_size` | 8     | Not adaptive to GPU memory                    |
| `anchors.py:132`             | `precision_threshold` | 0.95  | May be too strict for noisy data              |

---

#### 2.8 sys.path Manipulation Anti-Pattern

```
Severity: P2 (Medium)
Location: All files
Impact: Import Hygiene
```

Every file contains:

```python
sys.path.append(str(Path(__file__).parent.parent))
```

This pollutes `sys.path` and can cause import conflicts. Should use proper package structure with `__init__.py` and relative imports.

---

#### 2.9 Missing Input Validation

| Method                                             | Missing Validation                           |
| -------------------------------------------------- | -------------------------------------------- |
| `IntegratedGradientsExplainer.explain()`           | `steps` must be > 0                          |
| `SHAPExplainer._gradient_shap()`                   | `n_samples` must be > 0                      |
| `LIMEExplainer.explain()`                          | `num_segments` must divide signal reasonably |
| `UncertaintyQuantifier.predict_with_uncertainty()` | Model must have dropout layers               |

---

#### 2.10 Visualization Defaults Can Be Misleading

```
Severity: P2 (Medium)
Location: integrated_gradients.py:302-305
```

**Problem**:

```python
for i in range(len(time) - 1):
    color_intensity = attr_normalized[i]
    color = plt.cm.Reds(color_intensity)
    axes[2].axvspan(time[i], time[i+1], alpha=color_intensity * 0.5, color=color, linewidth=0)
```

Using `Reds` colormap for all attributions (positive and negative) can mislead users. Negative attributions should use a diverging colormap (e.g., `RdBu`).

---

## Task 3: "If I Could Rewrite This" Retrospective

### 3.1 Unified Explainer Interface

**Current State**: Each explainer has its own API:

- `IntegratedGradientsExplainer.explain(input_signal, target_class, baseline, steps)`
- `SHAPExplainer.explain(input_signal, method, n_samples)`
- `LIMEExplainer.explain(input_signal, target_class, num_samples)`
- `AnchorExplainer.explain(input_signal, n_samples)`

**Proposed Unified Interface**:

```python
class ExplainerProtocol(Protocol):
    def explain(
        self,
        input_signal: torch.Tensor,
        target_class: Optional[int] = None,
        **kwargs
    ) -> ExplanationResult:
        ...

    def visualize(
        self,
        explanation: ExplanationResult,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        ...

@dataclass
class ExplanationResult:
    method: str
    attributions: Optional[torch.Tensor]  # For IG, SHAP
    segments: Optional[List[Tuple]]  # For LIME, Anchors
    uncertainty: Optional[Dict]  # For MC Dropout
    metadata: Dict[str, Any]
```

---

### 3.2 Caching Strategy Recommendations

**Hierarchical Cache Architecture**:

```
                    ┌─────────────────────────┐
                    │   Result Cache (L1)     │
                    │   TTL: 1 hour           │
                    │   Key: input_hash       │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ Intermediate Cache (L2) │
                    │ Gradients, Activations  │
                    │ TTL: 24 hours           │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Model Cache (L3)      │
                    │   CAVs, Background      │
                    │   Invalidate on retrain │
                    └─────────────────────────┘
```

**Cache Invalidation Triggers**:

1. Model weights change (hash-based detection)
2. Background data changes (for SHAP)
3. Concept examples change (for CAVs)
4. TTL expiration

---

### 3.3 Scientific Soundness Recommendations

| Method          | Current                     | Recommended Fix                                    |
| --------------- | --------------------------- | -------------------------------------------------- |
| **KernelSHAP**  | Plain LinearRegression      | Weighted Ridge with SHAP kernel                    |
| **IG Baseline** | Zero default                | Offer "informative" baselines (mean, random, blur) |
| **LIME**        | Fixed segment size          | SLIC-like adaptive segmentation                    |
| **MC Dropout**  | Softmax averaging           | Log-softmax for better numerical stability         |
| **Anchors**     | Gaussian noise perturbation | Domain-specific perturbation (preserve physics)    |

---

### 3.4 Performance Optimization Opportunities

| Optimization                      | Impact                   | Effort |
| --------------------------------- | ------------------------ | ------ |
| Batch IG computation              | 10x speedup              | Medium |
| GPU-parallel LIME perturbations   | 5x speedup               | Low    |
| Pre-computed background summaries | 3x speedup for SHAP      | Low    |
| Lazy CAV computation              | On-demand only           | Low    |
| Explanation caching               | Eliminates recomputation | Medium |

---

## Technical Debt Inventory

### Priority Matrix

| ID        | Issue                          | Severity | Effort | Impact   |
| --------- | ------------------------------ | -------- | ------ | -------- |
| TD-1.5.01 | Missing `explanation_cache.py` | P0       | High   | Critical |
| TD-1.5.02 | KernelSHAP incorrect weighting | P0       | Medium | Critical |
| TD-1.5.03 | MC Dropout mode not restored   | P1       | Low    | High     |
| TD-1.5.04 | CAV gradient hook memory       | P1       | Low    | High     |
| TD-1.5.05 | GradientSHAP baseline handling | P1       | Medium | Medium   |
| TD-1.5.06 | LIME segment edge case         | P1       | Low    | Medium   |
| TD-1.5.07 | sys.path manipulation          | P2       | Medium | Low      |
| TD-1.5.08 | Hardcoded parameters           | P2       | Low    | Medium   |
| TD-1.5.09 | Missing input validation       | P2       | Low    | Medium   |
| TD-1.5.10 | Visualization colormap issues  | P2       | Low    | Low      |

---

## Best Practices Identified

### ✅ What's Done Well

1. **Comprehensive Academic References**: Each method has proper citations
2. **Convergence Validation**: IG includes `compute_convergence_delta()` for completeness axiom
3. **Statistical Testing**: TCAV includes proper p-value computation with random baselines
4. **Multiple SHAP Backends**: Supports native PyTorch and official SHAP library
5. **Graceful Degradation**: Falls back to gradient methods if libraries unavailable
6. **Self-contained Testing**: Each file has `__main__` validation blocks
7. **Type Hints**: Consistent use of typing annotations
8. **Docstrings**: Every class and method is documented

### Patterns Worth Adopting

```python
# Good: Graceful library fallback (shap_explainer.py:73-86)
try:
    import shap
    self.shap = shap
    self.shap_available = True
except ImportError:
    warnings.warn("SHAP library not installed. Using native PyTorch implementation.")
    self.shap_available = False

# Good: Convergence validation (integrated_gradients.py:202-242)
def compute_convergence_delta(...) -> float:
    """Verify attribution quality via completeness axiom."""
    attribution_sum = attributions.sum().item()
    expected_diff = score_input - score_baseline
    delta = abs(attribution_sum - expected_diff)
    return delta
```

---

## Recommendations Summary

### Immediate Actions (Week 1)

1. **Fix KernelSHAP weighting** - Scientific correctness issue
2. **Add try/finally to MC Dropout** - Prevent model state corruption
3. **Implement basic in-memory cache** - LRU with hash key

### Short-Term (Sprint)

4. Create unified `ExplainerProtocol` interface
5. Fix LIME segment boundary edge case
6. Add input validation across all methods

### Medium-Term (Quarter)

7. Implement full `explanation_cache.py` with disk persistence
8. Refactor to remove `sys.path` manipulation
9. Add proper unit tests (not just `__main__` blocks)
10. Implement diverging colormap for +/- attributions

---

## Appendix: File Metrics

| File                            | Lines | Classes | Functions | Complexity |
| ------------------------------- | ----- | ------- | --------- | ---------- |
| `__init__.py`                   | 81    | 0       | 0         | Low        |
| `integrated_gradients.py`       | 432   | 1       | 3         | Medium     |
| `shap_explainer.py`             | 442   | 1       | 3         | High       |
| `lime_explainer.py`             | 418   | 1       | 3         | Medium     |
| `uncertainty_quantification.py` | 452   | 1       | 5         | Medium     |
| `concept_activation_vectors.py` | 572   | 3       | 3         | High       |
| `partial_dependence.py`         | 581   | 1       | 5         | Medium     |
| `anchors.py`                    | 577   | 3       | 3         | High       |

---

_Report generated by AI Agent IDB 1.5 Analyst_
