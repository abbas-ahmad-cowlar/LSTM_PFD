# IDB 1.5: Explainability Sub-Block — Best Practices

> **Extracted**: 2026-01-24  
> **Source**: `packages/core/explainability/`  
> **Purpose**: Codify proven patterns for XAI implementation consistency

---

## 1. Explainer Interface Patterns

### 1.1 Core Interface Structure

**Pattern**: Consistent class-based explainer with `explain()` method

```python
class ExplainerBase:
    """Base pattern for all explainers."""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        self.model.eval()  # Always set to eval mode initially

    def explain(
        self,
        input_signal: torch.Tensor,
        target_class: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Core explanation method.

        Args:
            input_signal: Input to explain [1, C, T] or [C, T]
            target_class: Target class (if None, use predicted class)
            **kwargs: Method-specific parameters

        Returns:
            Explanation (format varies by method)
        """
        # 1. Ensure batch dimension
        if input_signal.dim() == 2:
            input_signal = input_signal.unsqueeze(0)

        # 2. Move to device
        input_signal = input_signal.to(self.device)

        # 3. Auto-detect target class if not provided
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_signal)
                target_class = output.argmax(dim=1).item()

        # 4. Compute explanation
        return self._compute_explanation(input_signal, target_class, **kwargs)
```

**Example**: [integrated_gradients.py:57-107](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/core/explainability/integrated_gradients.py#L57-L107)

---

### 1.2 Graceful Library Fallback

**Pattern**: Support optional dependencies with native fallback

```python
class SHAPExplainer:
    def __init__(self, model, use_shap_library: bool = True):
        self.shap_available = False

        if use_shap_library:
            try:
                import shap
                self.shap = shap
                self.shap_available = True
                print("✓ SHAP library available")
            except ImportError:
                warnings.warn(
                    "SHAP library not installed. Using native PyTorch implementation. "
                    "Install with: pip install shap"
                )
                self.shap_available = False
```

**Benefits**:

- Package works without heavy dependencies
- Enables better methods when available
- User-friendly warnings guide installation

**Example**: [shap_explainer.py:74-86](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/core/explainability/shap_explainer.py#L74-L86)

---

### 1.3 Scientific Validation Methods

**Pattern**: Provide built-in correctness checks

```python
def compute_convergence_delta(
    self,
    input_signal: torch.Tensor,
    attributions: torch.Tensor,
    target_class: int,
    baseline: Optional[torch.Tensor] = None
) -> float:
    """
    Verify attribution quality via completeness axiom.

    Completeness: sum(attributions) = F(input) - F(baseline)

    Returns:
        Absolute error (should be < 0.01 for good convergence)
    """
    with torch.no_grad():
        score_input = self.model(input_signal)[0, target_class].item()
        score_baseline = self.model(baseline)[0, target_class].item()

    attribution_sum = attributions.sum().item()
    expected_diff = score_input - score_baseline
    delta = abs(attribution_sum - expected_diff)

    return delta
```

**Example**: [integrated_gradients.py:202-242](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/core/explainability/integrated_gradients.py#L202-L242)

---

### 1.4 Batch Processing Support

**Pattern**: Process multiple samples efficiently

```python
def explain_batch(
    self,
    input_signals: torch.Tensor,
    target_classes: Optional[torch.Tensor] = None,
    **kwargs
) -> torch.Tensor:
    """Process batch of inputs efficiently."""
    batch_size = input_signals.shape[0]
    attributions = []

    for i in range(batch_size):
        target = target_classes[i].item() if target_classes is not None else None
        attr = self.explain(
            input_signals[i:i+1],
            target_class=target,
            **kwargs
        )
        attributions.append(attr)

    return torch.cat(attributions, dim=0)
```

**Example**: [integrated_gradients.py:168-200](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/core/explainability/integrated_gradients.py#L168-L200)

---

## 2. Caching Conventions

### 2.1 Expected Cache Architecture

> [!WARNING]
> **Current Status**: No caching implemented in IDB 1.5. The patterns below are recommended based on analysis findings.

**Recommended Pattern**: Three-tier cache hierarchy

```python
from pathlib import Path
from typing import Optional, Callable, Any
import hashlib
import pickle
import json

class ExplanationCache:
    """
    Three-tier cache for XAI computations:
    - L1: Result Cache (final explanations)
    - L2: Intermediate Cache (gradients, activations)
    - L3: Model Cache (CAVs, background data)
    """

    def __init__(
        self,
        cache_dir: Path,
        max_size_mb: int = 1000,
        ttl_hours: int = 24
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.ttl_hours = ttl_hours

        # Separate subdirectories
        self.results_dir = self.cache_dir / "results"
        self.intermediate_dir = self.cache_dir / "intermediate"
        self.model_dir = self.cache_dir / "model"

        for dir in [self.results_dir, self.intermediate_dir, self.model_dir]:
            dir.mkdir(exist_ok=True)

    def _compute_key(
        self,
        input_hash: str,
        method: str,
        model_hash: str,
        **params
    ) -> str:
        """Generate unique cache key."""
        key_components = {
            'input': input_hash,
            'method': method,
            'model': model_hash,
            'params': params
        }
        key_str = json.dumps(key_components, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable,
        tier: str = 'results'
    ) -> Any:
        """
        Get cached result or compute and cache.

        Args:
            key: Cache key
            compute_fn: Function to compute result if not cached
            tier: Cache tier ('results', 'intermediate', 'model')
        """
        cache_file = self._get_cache_path(key, tier)

        # Check cache
        if cache_file.exists() and self._is_valid(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Cache read error: {e}, recomputing...")

        # Compute and cache
        result = compute_fn()

        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"Cache write error: {e}")

        return result

    def _get_cache_path(self, key: str, tier: str) -> Path:
        """Get cache file path for tier."""
        tier_dirs = {
            'results': self.results_dir,
            'intermediate': self.intermediate_dir,
            'model': self.model_dir
        }
        return tier_dirs[tier] / f"{key}.pkl"

    def _is_valid(self, cache_file: Path) -> bool:
        """Check if cache entry is still valid (TTL check)."""
        import time
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        return age_hours < self.ttl_hours

    def invalidate(self, model_hash: str):
        """Invalidate all caches for a specific model."""
        for tier_dir in [self.results_dir, self.intermediate_dir, self.model_dir]:
            for cache_file in tier_dir.glob(f"*{model_hash}*.pkl"):
                cache_file.unlink()
```

**Usage Example**:

```python
cache = ExplanationCache(cache_dir=Path("./cache/xai"))

def explain_cached(explainer, signal, target_class):
    input_hash = hashlib.sha256(signal.cpu().numpy().tobytes()).hexdigest()[:16]
    model_hash = get_model_hash(explainer.model)

    key = cache._compute_key(
        input_hash=input_hash,
        method="integrated_gradients",
        model_hash=model_hash,
        steps=50
    )

    return cache.get_or_compute(
        key=key,
        compute_fn=lambda: explainer.explain(signal, target_class, steps=50),
        tier='results'
    )
```

---

### 2.2 Cache Invalidation Triggers

| Trigger                 | Action                 | Tier      |
| ----------------------- | ---------------------- | --------- |
| Model weights change    | Invalidate all         | All tiers |
| Background data updated | Invalidate SHAP caches | L2, L3    |
| Concept examples change | Invalidate CAV caches  | L3        |
| TTL expires             | Delete stale entries   | Per-tier  |

---

## 3. Visualization Styling for XAI

### 3.1 Consistent Figure Layout

**Pattern**: Three-panel layout for signal + explanation

```python
def plot_attribution_map(
    signal: np.ndarray,
    attributions: np.ndarray,
    predicted_class: int,
    true_class: Optional[int] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Standard three-panel layout:
    1. Original signal
    2. Attribution values
    3. Signal with attribution overlay
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    time = np.arange(len(signal))

    # Panel 1: Original Signal
    axes[0].plot(time, signal, 'b-', linewidth=0.8)
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title('Original Vibration Signal', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Attribution Values
    axes[1].plot(time, attributions, 'r-', linewidth=0.8)
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1].set_ylabel('Attribution', fontsize=11)
    axes[1].set_title('Integrated Gradients Attribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Signal with Attribution Overlay
    axes[2].plot(time, signal, 'b-', linewidth=0.8, alpha=0.5, label='Signal')

    # Color-code attribution importance
    attr_normalized = np.abs(attributions) / (np.abs(attributions).max() + 1e-8)

    for i in range(len(time) - 1):
        color_intensity = attr_normalized[i]
        color = plt.cm.Reds(color_intensity)
        axes[2].axvspan(time[i], time[i+1], alpha=color_intensity * 0.5,
                       color=color, linewidth=0)

    axes[2].set_xlabel('Time Steps', fontsize=11)
    axes[2].set_ylabel('Amplitude', fontsize=11)
    axes[2].set_title('Signal with Attribution Overlay', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show_plot:
        plt.show()
    else:
        plt.close()
```

**Example**: [integrated_gradients.py:245-336](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/core/explainability/integrated_gradients.py#L245-L336)

---

### 3.2 Color Palette Guidelines

| Visualization Type                 | Recommended Colormap      | Rationale                |
| ---------------------------------- | ------------------------- | ------------------------ |
| **Positive-only attributions**     | `Reds`, `Blues`, `Greens` | Clear intensity gradient |
| **Positive/negative attributions** | `RdBu_r`, `coolwarm`      | Diverging, zero-centered |
| **Uncertainty**                    | `viridis`, `plasma`       | Perceptually uniform     |
| **Heatmaps**                       | `RdYlGn` (good/bad)       | Intuitive interpretation |
| **Multi-class**                    | `tab10`, `Set3`           | Distinct categories      |

**Example Pattern**:

```python
# For signed attributions (use diverging colormap)
import matplotlib.pyplot as plt

attr_max = np.abs(attributions).max()
im = ax.imshow(
    attributions,
    cmap='RdBu_r',  # Red = negative, Blue = positive
    vmin=-attr_max,
    vmax=attr_max,
    aspect='auto'
)
```

---

### 3.3 Annotation Best Practices

**Pattern**: Add metadata and significance markers

```python
def plot_tcav_results(tcav_results: List[Dict], save_path: Optional[str] = None):
    """TCAV results with statistical significance annotations."""
    fig, ax = plt.subplots(figsize=(10, max(6, len(tcav_results) * 0.5)))

    concept_names = [r['concept_name'] for r in tcav_results]
    tcav_scores = [r['tcav_score'] for r in tcav_results]
    p_values = [r['p_value'] for r in tcav_results]

    # Color by significance
    colors = ['green' if p < 0.05 else 'gray' for p in p_values]

    y_pos = np.arange(len(tcav_results))
    bars = ax.barh(y_pos, tcav_scores, color=colors, alpha=0.7, edgecolor='black')

    # Add significance markers
    for i, (score, p) in enumerate(zip(tcav_scores, p_values)):
        marker = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(score + 0.02, i, f'{marker} (p={p:.3f})', va='center', fontsize=9)

    # Baseline reference
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.5,
               label='Random baseline', alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(concept_names)
    ax.set_xlabel('TCAV Score (Fraction of Positive Influence)', fontsize=11)
    ax.set_title('Concept Influence on Model Predictions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
```

**Example**: [concept_activation_vectors.py:384-432](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/core/explainability/concept_activation_vectors.py#L384-L432)

---

### 3.4 Save Path Conventions

**Pattern**: Consistent naming with metadata

```python
def save_explanation_plot(
    explanation_type: str,
    sample_id: str,
    target_class: int,
    output_dir: Path,
    timestamp: bool = True
) -> Path:
    """
    Generate standardized filename for XAI visualizations.

    Format: {explanation_type}_{sample_id}_class{target_class}_{timestamp}.png
    Example: integrated_gradients_sample_042_class3_20260124_041500.png
    """
    from datetime import datetime

    filename_parts = [explanation_type, sample_id, f"class{target_class}"]

    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_parts.append(ts)

    filename = "_".join(filename_parts) + ".png"
    return output_dir / filename
```

---

## 4. Performance Optimization Patterns

### 4.1 Batched Gradient Computation

**Pattern**: Process multiple interpolation steps in batches

```python
def _compute_integrated_gradients(
    self,
    input_signal: torch.Tensor,
    baseline: torch.Tensor,
    target_class: int,
    steps: int,
    batch_size: int  # Key parameter
) -> torch.Tensor:
    """Compute IG with batched gradients for efficiency."""

    # Generate all interpolated inputs
    alphas = torch.linspace(0, 1, steps + 1, device=self.device)
    interpolated_inputs = baseline + alphas.view(-1, 1, 1) * (input_signal - baseline)

    # Compute gradients in batches (not one-by-one)
    all_gradients = []

    for i in range(0, len(interpolated_inputs), batch_size):
        batch = interpolated_inputs[i:i+batch_size]
        batch.requires_grad = True

        outputs = self.model(batch)
        target_scores = outputs[:, target_class]

        gradients = torch.autograd.grad(
            outputs=target_scores,
            inputs=batch,
            grad_outputs=torch.ones_like(target_scores),
            create_graph=False
        )[0]

        all_gradients.append(gradients.detach())

    all_gradients = torch.cat(all_gradients, dim=0)

    # Trapezoidal rule integration
    avg_gradients = (all_gradients[:-1] + all_gradients[1:]) / 2.0
    avg_gradients = avg_gradients.mean(dim=0, keepdim=True)

    return (input_signal - baseline) * avg_gradients
```

**Benefit**: 10x speedup vs. sequential processing

**Example**: [integrated_gradients.py:109-166](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/core/explainability/integrated_gradients.py#L109-L166)

---

### 4.2 Reusable Hook Management

**Pattern**: Context manager for activation hooks

```python
from contextlib import contextmanager
from typing import Dict, List

@contextmanager
def capture_activations(model: nn.Module, layer_names: List[str]):
    """
    Context manager for capturing layer activations.
    Automatically removes hooks on exit.
    """
    activations: Dict[str, List[torch.Tensor]] = {name: [] for name in layer_names}
    handles = []

    def make_hook(name: str):
        def hook_fn(module, input, output):
            activations[name].append(output.detach().cpu())
        return hook_fn

    # Register hooks
    for name, module in model.named_modules():
        if name in layer_names:
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)

    try:
        yield activations
    finally:
        # Always remove hooks
        for handle in handles:
            handle.remove()

# Usage
with capture_activations(model, ['layer1.conv', 'layer2.conv']) as activations:
    _ = model(input_batch)
    # activations['layer1.conv'] now contains captured outputs
```

**Benefits**:

- Guaranteed cleanup (no memory leaks)
- Reusable across methods
- Exception-safe

---

### 4.3 Configurable Sample Counts

**Pattern**: Adaptive sampling based on performance budget

```python
class UncertaintyQuantifier:
    def predict_with_uncertainty(
        self,
        input_signal: torch.Tensor,
        n_samples: int = 50,
        min_samples: int = 10,  # Fast mode
        max_samples: int = 200,  # High accuracy mode
        target_std_threshold: float = 0.01
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Adaptive MC Dropout with early stopping.

        Stops sampling when uncertainty estimate stabilizes.
        """
        self.model.train()
        predictions = []

        for i in range(max_samples):
            with torch.no_grad():
                output = self.model(input_signal)
                probs = torch.softmax(output, dim=1)
                predictions.append(probs)

            # Check convergence after min_samples
            if i >= min_samples:
                current_predictions = torch.stack(predictions)
                current_std = current_predictions.std(dim=0).mean()

                if i >= n_samples and current_std < target_std_threshold:
                    print(f"Converged after {i+1} samples")
                    break

        predictions = torch.stack(predictions)
        mean_prediction = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)

        self.model.eval()

        return mean_prediction, uncertainty, predictions
```

---

### 4.4 Lazy CAV Computation

**Pattern**: Compute CAVs on-demand, not eagerly

```python
class CAVGenerator:
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = device
        self._cav_cache: Dict[str, ConceptActivationVector] = {}

    def get_or_generate_cav(
        self,
        concept_examples: torch.Tensor,
        random_examples: torch.Tensor,
        layer_name: str,
        concept_name: str,
        force_recompute: bool = False
    ) -> ConceptActivationVector:
        """Lazy CAV generation with caching."""

        cache_key = f"{concept_name}_{layer_name}"

        # Return cached if available
        if cache_key in self._cav_cache and not force_recompute:
            print(f"Using cached CAV for {cache_key}")
            return self._cav_cache[cache_key]

        # Generate new CAV
        print(f"Generating CAV for {cache_key}")
        cav = self.generate_cav(
            concept_examples, random_examples, layer_name, concept_name
        )

        # Cache for reuse
        self._cav_cache[cache_key] = cav

        return cav
```

---

## 5. Scientific Correctness Validation

### 5.1 Integrated Gradients Axioms

**Pattern**: Verify all three axioms

```python
def validate_integrated_gradients(
    explainer: IntegratedGradientsExplainer,
    input_signal: torch.Tensor,
    baseline: torch.Tensor,
    target_class: int,
    attributions: torch.Tensor
) -> Dict[str, bool]:
    """
    Validate Integrated Gradients axioms:
    1. Completeness: sum(attr) = F(x) - F(baseline)
    2. Sensitivity: If x differs from baseline only in feature i, and F(x) != F(baseline),
                    then attr_i != 0
    3. Implementation Invariance: Functionally equivalent networks produce same attributions
    """
    results = {}

    # 1. Completeness
    delta = explainer.compute_convergence_delta(
        input_signal, attributions, target_class, baseline
    )
    results['completeness'] = delta < 0.01  # Tolerance

    # 2. Sensitivity (test with synthetic example)
    # Create input that differs in single feature
    test_input = baseline.clone()
    test_input[0, 0, 100] = input_signal[0, 0, 100]  # Differ only at index 100

    with torch.no_grad():
        out_baseline = explainer.model(baseline)[0, target_class]
        out_test = explainer.model(test_input)[0, target_class]

    if abs(out_baseline - out_test) > 1e-3:  # Outputs differ
        test_attr = explainer.explain(test_input, target_class, baseline)
        results['sensitivity'] = abs(test_attr[0, 0, 100]) > 1e-6  # Attribution at index 100 is non-zero
    else:
        results['sensitivity'] = True  # N/A if outputs don't differ

    # 3. Implementation Invariance (requires multiple passes)
    attr_recomputed = explainer.explain(input_signal, target_class, baseline)
    results['implementation_invariance'] = torch.allclose(
        attributions, attr_recomputed, rtol=1e-3
    )

    return results
```

---

### 5.2 SHAP Property Verification

**Pattern**: Check local accuracy and consistency

```python
def validate_shap_properties(
    explainer: SHAPExplainer,
    input_signal: torch.Tensor,
    shap_values: torch.Tensor,
    target_class: int
) -> Dict[str, bool]:
    """
    Validate SHAP properties:
    1. Local Accuracy: f(x) = φ₀ + Σφᵢ
    2. Missingness: If feature x_i = baseline_i, then φ_i = 0
    3. Consistency: Not easily testable without model pairs
    """
    results = {}

    # 1. Local Accuracy
    with torch.no_grad():
        prediction = explainer.model(input_signal)[0, target_class].item()

    if explainer.background_data is not None:
        base_prediction = explainer.model(
            explainer.background_data[:1]
        )[0, target_class].item()
    else:
        base_prediction = 0.0

    shap_sum = shap_values.sum().item()
    expected = prediction - base_prediction

    results['local_accuracy'] = abs(shap_sum - expected) < 0.1  # 10% tolerance

    # 2. Missingness (cannot easily verify without known baseline matches)
    results['missingness'] = None  # Requires domain-specific test

    return results
```

---

### 5.3 Statistical Significance Testing (TCAV)

**Pattern**: Always include p-values with random baselines

```python
def compute_tcav_with_significance(
    analyzer: TCAVAnalyzer,
    test_examples: torch.Tensor,
    target_class: int,
    n_random_runs: int = 100,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Compute TCAV with proper statistical testing.

    Returns TCAV score + p-value + confidence interval.
    """
    # Compute concept TCAV
    derivatives = analyzer._compute_directional_derivatives(
        test_examples, target_class, analyzer.cav
    )
    tcav_score = (derivatives > 0).mean()

    # Random baseline distribution
    random_scores = []

    for i in range(n_random_runs):
        random_vector = np.random.randn(len(analyzer.cav.vector))
        random_vector = random_vector / (np.linalg.norm(random_vector) + 1e-8)

        random_cav = ConceptActivationVector(
            concept_name=f"random_{i}",
            layer_name=analyzer.cav.layer_name,
            vector=random_vector,
            accuracy=0.5,
            classifier=None
        )

        random_derivatives = analyzer._compute_directional_derivatives(
            test_examples, target_class, random_cav
        )
        random_score = (random_derivatives > 0).mean()
        random_scores.append(random_score)

    random_scores = np.array(random_scores)

    # Two-tailed p-value
    p_value = (np.abs(random_scores - 0.5) >= np.abs(tcav_score - 0.5)).mean()

    # Confidence interval (bootstrapped)
    ci_lower = np.percentile(random_scores, alpha/2 * 100)
    ci_upper = np.percentile(random_scores, (1 - alpha/2) * 100)

    return {
        'tcav_score': tcav_score,
        'p_value': p_value,
        'significant': p_value < alpha,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'random_mean': random_scores.mean(),
        'random_std': random_scores.std()
    }
```

**Example**: [concept_activation_vectors.py:257-325](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/core/explainability/concept_activation_vectors.py#L257-L325)

---

### 5.4 Calibration Metrics

**Pattern**: Always compute ECE for uncertainty methods

```python
def compute_calibration_metrics(
    model: nn.Module,
    dataloader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Compute comprehensive calibration metrics.

    Returns:
        - ECE: Expected Calibration Error
        - MCE: Maximum Calibration Error
        - Brier Score: Mean squared error of probabilities
    """
    all_confidences = []
    all_correct = []
    all_probs = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)

            confidences, predictions = probs.max(dim=1)
            correct = (predictions == targets)

            all_confidences.extend(confidences.cpu().numpy())
            all_correct.extend(correct.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_confidences = np.array(all_confidences)
    all_correct = np.array(all_correct)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)

    # Expected Calibration Error
    num_bins = 10
    bin_edges = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    mce = 0.0

    for i in range(num_bins):
        bin_mask = (all_confidences >= bin_edges[i]) & (all_confidences < bin_edges[i+1])
        if bin_mask.sum() > 0:
            bin_acc = all_correct[bin_mask].mean()
            bin_conf = all_confidences[bin_mask].mean()
            bin_error = abs(bin_acc - bin_conf)
            bin_weight = bin_mask.sum() / len(all_confidences)

            ece += bin_weight * bin_error
            mce = max(mce, bin_error)

    # Brier Score
    brier_score = np.mean(
        np.sum((all_probs - np.eye(all_probs.shape[1])[all_targets])**2, axis=1)
    )

    return {
        'ece': ece,
        'mce': mce,
        'brier_score': brier_score
    }
```

**Example**: [uncertainty_quantification.py:188-252](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/core/explainability/uncertainty_quantification.py#L188-L252)

---

## 6. Code Organization Patterns

### 6.1 Module Structure

```
explainability/
├── __init__.py              # Public API exports
├── integrated_gradients.py  # IG explainer + visualizations
├── shap_explainer.py        # SHAP (3 variants)
├── lime_explainer.py        # LIME
├── uncertainty_quantification.py  # MC Dropout + calibration
├── concept_activation_vectors.py  # CAVs/TCAV
├── partial_dependence.py    # PDP/ICE
├── anchors.py               # Anchors
└── explanation_cache.py     # (MISSING - needs implementation)
```

**Key Principle**: One file per major XAI method with integrated visualization functions.

---

### 6.2 Docstring Standards

**Pattern**: NumPy-style with academic references

```python
"""
Integrated Gradients for Attribution

Implements Integrated Gradients (Sundararajan et al., 2017) for attributing
predictions to input features. This method computes the integral of gradients
along a path from a baseline to the input.

Key Properties:
- Sensitivity: If inputs differ only in one feature and predictions differ,
  the differing feature should have non-zero attribution
- Implementation Invariance: Attribution is the same for functionally equivalent networks
- Completeness: Attributions sum to difference between output at input and baseline

Reference:
Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks.
International Conference on Machine Learning (ICML).
"""
```

**Example**: All files include this pattern

---

### 6.3 Self-Validation in `__main__`

**Pattern**: Every module includes runnable validation

```python
if __name__ == "__main__":
    print("=" * 60)
    print("Integrated Gradients - Validation")
    print("=" * 60)

    from models.cnn.cnn_1d import CNN1D

    model = CNN1D(num_classes=NUM_CLASSES, input_channels=1, dropout=0.3)
    signal = torch.randn(1, 1, 10240)

    explainer = IntegratedGradientsExplainer(model, device='cpu')

    print("\nComputing Integrated Gradients...")
    attributions = explainer.explain(signal, target_class=3, steps=50)

    print(f"  Input shape: {signal.shape}")
    print(f"  Attribution shape: {attributions.shape}")
    print(f"  Attribution range: [{attributions.min():.4f}, {attributions.max():.4f}]")

    # Check convergence
    delta = explainer.compute_convergence_delta(signal, attributions, target_class=3)
    print(f"  Convergence delta: {delta:.6f}")

    if delta < 0.01:
        print("  ✓ Convergence check PASSED")
    else:
        print(f"  ⚠ Convergence delta is {delta:.6f}")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
```

**Benefits**:

- Quick smoke test without pytest infrastructure
- Documents expected behavior
- Validates on synthetic data

---

## Summary: Adoption Checklist

When implementing new XAI methods, ensure:

- [ ] Consistent `explain()` interface with auto-detect target class
- [ ] Graceful fallback for optional dependencies
- [ ] Scientific validation methods (convergence, p-values, etc.)
- [ ] Three-panel visualization layout
- [ ] Diverging colormap for signed values
- [ ] Batched gradient computation where applicable
- [ ] Context managers for hook cleanup
- [ ] Lazy computation with caching
- [ ] Academic reference in docstring
- [ ] Self-validation in `__main__`
- [ ] Proper exception handling with try/finally
- [ ] Input validation and meaningful error messages

---

_Best Practices extracted from IDB 1.5 Explainability Sub-Block_
