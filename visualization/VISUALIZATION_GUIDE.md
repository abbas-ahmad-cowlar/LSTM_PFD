# Visualization Guide

> Detailed usage guide for every visualization tool in the `visualization/` module.

---

## Table of Contents

1. [Signal Display](#1-signal-display)
2. [Spectrogram Visualization](#2-spectrogram-visualization)
3. [Model Performance Plots](#3-model-performance-plots)
4. [Feature Analysis & Dimensionality Reduction](#4-feature-analysis--dimensionality-reduction)
5. [Latent Space Comparison](#5-latent-space-comparison)
6. [CNN Inspection (1D)](#6-cnn-inspection-1d)
7. [CNN Inspection (2D — Spectrogram Models)](#7-cnn-inspection-2d--spectrogram-models)
8. [Saliency Maps](#8-saliency-maps)
9. [Counterfactual Explanations](#9-counterfactual-explanations)
10. [Attention Visualization](#10-attention-visualization)
11. [Interactive XAI Dashboard](#11-interactive-xai-dashboard)
12. [Publication-Quality Settings](#12-publication-quality-settings)
13. [Integration with Research Scripts](#13-integration-with-research-scripts)
14. [Matplotlib Conventions](#14-matplotlib-conventions)

---

## 1. Signal Display

**File:** `signal_plots.py`  
**Class:** `SignalPlotter`

### `SignalPlotter.plot_signal_examples`

Plots time-domain waveform, frequency spectrum (FFT), and spectrogram side-by-side for multiple fault classes.

```python
from visualization import SignalPlotter

fig = SignalPlotter.plot_signal_examples(
    signals,          # np.ndarray — shape (n_samples, signal_length)
    labels,           # np.ndarray — integer class labels
    fs=20480,         # float — sampling frequency in Hz
    n_examples=3,     # int — number of classes to display
    save_path=None    # Optional[Path] — save location
)
```

**Parameters:**

| Parameter    | Type             | Default  | Description                        |
| ------------ | ---------------- | -------- | ---------------------------------- |
| `signals`    | `np.ndarray`     | required | Signal matrix `(N, signal_length)` |
| `labels`     | `np.ndarray`     | required | Label array `(N,)`                 |
| `fs`         | `float`          | required | Sampling frequency (Hz)            |
| `n_examples` | `int`            | `3`      | Number of classes to plot          |
| `save_path`  | `Optional[Path]` | `None`   | File path to save figure           |

### `SignalPlotter.plot_signal_comparison`

Side-by-side comparison of two signals in time and frequency domains.

```python
fig = SignalPlotter.plot_signal_comparison(
    signal1, signal2,
    fs=20480,
    labels=["Normal", "Ball Fault"],
    save_path="comparison.png"
)
```

---

## 2. Spectrogram Visualization

**File:** `spectrogram_plots.py`  
**Functions:** `plot_spectrogram`, `plot_spectrogram_comparison`, `plot_fault_spectrograms_grid`, `plot_spectrogram_with_prediction`, `plot_frequency_evolution`, `plot_spectrogram_statistics`

**Dependencies:** `data.spectrogram_generator.SpectrogramGenerator`, `data.wavelet_transform.WaveletTransform`, `data.wigner_ville.generate_wvd`

### `plot_spectrogram`

Plot a single spectrogram as a heatmap.

```python
from visualization.spectrogram_plots import plot_spectrogram

ax = plot_spectrogram(
    spectrogram,      # np.ndarray — shape [H, W]
    fs=20480,         # float — sampling frequency
    title="STFT",     # str — plot title
    ax=None,          # Optional[plt.Axes] — existing axes
    cmap='viridis',   # str — colormap
    colorbar=True     # bool — show colorbar
)
```

### `plot_spectrogram_comparison`

Compare STFT, CWT, and WVD representations of the same signal.

```python
from visualization.spectrogram_plots import plot_spectrogram_comparison

plot_spectrogram_comparison(
    signal,                           # np.ndarray — time-domain signal
    fs=20480,
    tfr_types=['STFT', 'CWT', 'WVD'],  # List[str] — TFR types
    save_path="tfr_comparison.png",
    figsize=(15, 10)
)
```

### `plot_fault_spectrograms_grid`

Grid of spectrograms for all fault types.

```python
from visualization.spectrogram_plots import plot_fault_spectrograms_grid

plot_fault_spectrograms_grid(
    signals_by_fault,   # Dict[str, np.ndarray] — {fault_name: signal}
    fs=20480,
    tfr_type='STFT',
    save_path="fault_grid.png",
    figsize=(20, 15)
)
```

### `plot_spectrogram_with_prediction`

Spectrogram with model prediction overlay showing true label, predicted label, and confidence.

```python
from visualization.spectrogram_plots import plot_spectrogram_with_prediction

plot_spectrogram_with_prediction(
    signal, fs=20480,
    true_label="Ball-007",
    predicted_label="Ball-014",
    confidence=0.87,
    save_path="prediction.png"
)
```

### `plot_frequency_evolution`

Track energy in specific frequency bands over time.

```python
from visualization.spectrogram_plots import plot_frequency_evolution

plot_frequency_evolution(
    signal, fs=20480,
    freq_bands=[(100, 500), (1000, 3000)],  # List[Tuple[float, float]]
    save_path="freq_evolution.png"
)
```

### `plot_spectrogram_statistics`

Per-class spectrogram statistics (mean, std, etc.).

```python
from visualization.spectrogram_plots import plot_spectrogram_statistics

plot_spectrogram_statistics(
    spectrograms,    # np.ndarray — [N, H, W]
    labels,          # np.ndarray — [N]
    class_names=["Normal", "Ball-007", ...],
    save_path="stats.png"
)
```

---

## 3. Model Performance Plots

**File:** `performance_plots.py`  
**Class:** `PerformancePlotter`

### `plot_model_comparison`

Bar chart comparing validation accuracy across models.

```python
from visualization import PerformancePlotter

fig = PerformancePlotter.plot_model_comparison(
    results,         # Dict — {model_name: {'accuracy': float, ...}}
    save_path="model_comparison.png"
)
```

### `plot_confusion_matrix`

Confusion matrix heatmap with optional row-wise normalization.

```python
fig = PerformancePlotter.plot_confusion_matrix(
    cm,                  # np.ndarray — (n_classes, n_classes)
    class_names=None,    # Optional[List[str]]
    normalize=True,      # bool — normalize by true label
    save_path="cm.png"
)
```

### `plot_roc_curves`

One-vs-Rest ROC curves with per-class AUC.

```python
fig = PerformancePlotter.plot_roc_curves(
    y_true,             # np.ndarray — true labels (N,)
    y_proba,            # np.ndarray — predicted probabilities (N, n_classes)
    class_names=None,
    save_path="roc.png"
)
```

---

## 4. Feature Analysis & Dimensionality Reduction

**File:** `feature_visualization.py`  
**Class:** `FeatureVisualizer`

All methods are `@staticmethod`.

### `plot_correlation_matrix`

Feature-to-feature Pearson correlation heatmap.

```python
from visualization import FeatureVisualizer

FeatureVisualizer.plot_correlation_matrix(
    features,           # np.ndarray — (N, D)
    feature_names,      # List[str] — D feature names
    save_path=None
)
```

### `plot_feature_distributions`

Per-class violin/box distributions for each feature.

```python
FeatureVisualizer.plot_feature_distributions(
    features, labels, feature_names,
    save_path="distributions.png"
)
```

### `plot_tsne_clusters`

t-SNE 2D projection colored by class label.

```python
FeatureVisualizer.plot_tsne_clusters(
    features,           # np.ndarray — (N, D)
    labels,             # np.ndarray — (N,)
    save_path=None,
    perplexity=30,      # int — t-SNE perplexity
    title='t-SNE Feature Clustering'
)
```

### `plot_umap_clusters`

UMAP 2D projection (requires `umap-learn`).

```python
FeatureVisualizer.plot_umap_clusters(
    features, labels,
    save_path=None,
    n_neighbors=15,     # int — UMAP neighbor count
    min_dist=0.1,       # float — UMAP minimum distance
    title='UMAP Feature Clustering'
)
```

### `compare_embeddings`

Side-by-side comparison of two feature sets (e.g., physics branch vs CNN branch).

```python
FeatureVisualizer.compare_embeddings(
    features_a,                  # np.ndarray — (N, D1)
    features_b,                  # np.ndarray — (N, D2)
    labels,
    name_a='Physics Branch',
    name_b='CNN Branch',
    method='tsne',               # 'tsne' | 'umap'
    save_path="comparison.png"
)
```

### `compute_clustering_metrics`

Returns silhouette score, Davies-Bouldin index, and Calinski-Harabasz index.

```python
metrics = FeatureVisualizer.compute_clustering_metrics(features, labels)
# metrics: {'silhouette': float, 'davies_bouldin': float, 'calinski_harabasz': float}
```

---

## 5. Latent Space Comparison

**File:** `latent_space_analysis.py`  
**Class:** `LatentSpaceAnalyzer`

Designed for comparing physics-informed vs pure data-driven feature representations.

### Constructor

```python
from visualization.latent_space_analysis import LatentSpaceAnalyzer

analyzer = LatentSpaceAnalyzer(device='cpu')
```

### `extract_features`

Extract intermediate features from a PyTorch model using forward hooks.

```python
features, labels = analyzer.extract_features(
    model,           # nn.Module
    dataloader,      # DataLoader yielding (signals, labels)
    layer_name=None  # Optional[str] — target layer (None → penultimate)
)
```

### `plot_comparison`

Side-by-side 2D projection of two feature sets.

```python
analyzer.plot_comparison(
    features_physics, features_cnn, labels,
    name_a="Physics Branch",
    name_b="Data Branch",
    method='tsne',         # 'tsne' | 'umap' | 'pca'
    save_path="latent.png"
)
```

### `print_comparison_report`

Print clustering metrics for both feature sets to stdout.

```python
analyzer.print_comparison_report(
    features_physics, features_cnn, labels,
    name_a="Physics Branch", name_b="Data Branch"
)
```

---

## 6. CNN Inspection (1D)

### CNNVisualizer

**File:** `cnn_visualizer.py`

```python
from visualization.cnn_visualizer import CNNVisualizer

viz = CNNVisualizer(model, device=None)  # auto-detects CUDA/CPU
```

| Method                          | Description                                   | Key Parameters                                  |
| ------------------------------- | --------------------------------------------- | ----------------------------------------------- |
| `plot_conv_filters`             | Visualize Conv1d filter weights as 1D signals | `layer_name`, `max_filters=64`                  |
| `plot_feature_maps`             | Feature map activations at a layer            | `input_signal`, `layer_name`, `max_channels=16` |
| `plot_activation_distributions` | Activation stats across all layers            | `input_signal`                                  |
| `plot_filter_heatmap`           | Filter weights as a 2D heatmap                | `layer_name`                                    |
| `plot_receptive_field`          | Receptive field size per layer                | —                                               |

### CNNAnalyzer

**File:** `cnn_analysis.py`

```python
from visualization.cnn_analysis import CNNAnalyzer

analyzer = CNNAnalyzer(model, device=None)
```

| Method                  | Description                                                  | Key Parameters                                                   |
| ----------------------- | ------------------------------------------------------------ | ---------------------------------------------------------------- |
| `analyze_gradient_flow` | Gradient magnitude per layer (vanishing/exploding detection) | `dataloader`, `num_batches=10`                                   |
| `compute_saliency_map`  | Input-gradient saliency                                      | `input_signal`, `target_class`                                   |
| `occlusion_sensitivity` | Sliding-window occlusion                                     | `input_signal`, `target_class`, `window_size=1024`, `stride=512` |
| `analyze_failure_cases` | Misclassified sample analysis                                | `dataloader`, `class_names`, `n_cases=10`                        |
| `layer_ablation_study`  | Remove each layer, measure accuracy drop                     | `dataloader`                                                     |

---

## 7. CNN Inspection (2D — Spectrogram Models)

**File:** `activation_maps_2d.py`  
**Functions:** module-level (not class-based)

```python
from visualization.activation_maps_2d import (
    visualize_filters,
    visualize_feature_maps,
    generate_grad_cam,
    visualize_grad_cam,
    analyze_layer_responses,
    visualize_filter_responses_to_frequency
)
```

### `visualize_filters`

```python
visualize_filters(
    model,               # nn.Module — trained 2D CNN
    layer_name='layer1', # str — Conv2d layer name
    num_filters=64,
    save_path=None,
    figsize=(16, 16)
)
```

### `visualize_grad_cam`

Grad-CAM heatmap overlaid on an input spectrogram.

```python
visualize_grad_cam(
    model, spectrogram,       # [1, 1, H, W] tensor
    target_class=None,        # None → predicted class
    class_name=None,
    save_path=None,
    figsize=(12, 6),
    device='cuda'
)
```

### `analyze_layer_responses`

Summary statistics of activations across multiple layers.

```python
analyze_layer_responses(
    model, spectrogram,
    layers=['layer1', 'layer2', 'layer3', 'layer4'],
    save_path=None
)
```

---

## 8. Saliency Maps

**File:** `saliency_maps.py`  
**Class:** `SaliencyMapGenerator`  
**Convenience functions:** `plot_saliency_map`, `compare_saliency_methods`

### Constructor

```python
from visualization import SaliencyMapGenerator

gen = SaliencyMapGenerator(model, device='cuda')
```

### Methods

| Method                 | Description                         | Key Parameters                    |
| ---------------------- | ----------------------------------- | --------------------------------- |
| `vanilla_gradient`     | ∂output/∂input                      | `input_signal`, `target_class`    |
| `smooth_grad`          | Averaged gradient over noisy inputs | `noise_level=0.1`, `n_samples=50` |
| `gradient_times_input` | Gradient × input value              | `input_signal`, `target_class`    |
| `grad_cam_1d`          | GradCAM for 1D conv layers          | `input_signal`, `target_layer`    |

### Plotting

```python
from visualization import plot_saliency_map, compare_saliency_methods

# Single method
saliency = gen.vanilla_gradient(signal, target_class=0)
plot_saliency_map(signal, saliency, predicted_class=0,
                  method_name="Vanilla Gradient", save_path="sal.png")

# Compare all methods side-by-side
compare_saliency_methods(signal, gen, target_class=0, save_path="comparison.png")
```

---

## 9. Counterfactual Explanations

**File:** `counterfactual_explanations.py`  
**Class:** `CounterfactualGenerator`  
**Convenience functions:** `plot_counterfactual_explanation`, `plot_optimization_history`

### Constructor

```python
from visualization import CounterfactualGenerator

cf_gen = CounterfactualGenerator(model, device='cuda')
```

### `generate`

Find minimal perturbation to change the model's prediction.

**Objective:** `minimize: ||δ||₂ + λ_l1·||δ||₁ + loss_class(x + δ, target)`

```python
counterfactual, info = cf_gen.generate(
    original_signal,              # Tensor [1, C, T]
    target_class=3,
    lambda_l2=0.1,
    lambda_l1=0.01,
    learning_rate=0.01,
    max_iterations=1000,
    confidence_threshold=0.9
)
# info: {'success': bool, 'perturbation_l2': float, 'iterations': int, ...}
```

### `generate_diverse_counterfactuals`

Generate multiple diverse counterfactuals with a diversity penalty.

```python
results = cf_gen.generate_diverse_counterfactuals(
    original_signal, target_class=3,
    num_counterfactuals=5,
    diversity_weight=0.5
)
```

### Plotting

```python
from visualization import plot_counterfactual_explanation, plot_optimization_history

plot_counterfactual_explanation(
    original_signal, counterfactual, info,
    class_names=["Normal", "Ball-007", ...],
    save_path="cf.png"
)

plot_optimization_history(info['history'], save_path="opt_history.png")
```

---

## 10. Attention Visualization

**File:** `attention_viz.py`  
**Class:** `AttentionVisualizer`

Supports `SignalTransformer`, `PatchTST`, and any model with `get_attention_maps()`.

### Constructor

```python
from visualization.attention_viz import AttentionVisualizer

attn_viz = AttentionVisualizer(model, device='cpu')
```

### Methods

| Method                                                         | Description                                                     |
| -------------------------------------------------------------- | --------------------------------------------------------------- |
| `get_attention_weights(x, layer_idx=-1)`                       | Extract attention weights `[n_heads, seq, seq]`                 |
| `attention_rollout(attention_weights, discard_ratio=0.0)`      | Compute attention rollout across layers (Abnar & Zuidema, 2020) |
| `plot_attention_heatmap(x, layer_idx, head_idx, ...)`          | Heatmap of attention weights                                    |
| `plot_signal_attention(signal, x, layer_idx, patch_size, ...)` | Signal with attention overlay                                   |
| `plot_head_comparison(x, layer_idx, ...)`                      | Per-head attention patterns                                     |

### Example

```python
# Attention heatmap for the last layer
attn_viz.plot_attention_heatmap(
    x,                    # [1, channels, seq_len]
    layer_idx=-1,
    head_idx=None,        # None → average across heads
    title="Attention Weights",
    save_path="attention.png"
)

# Signal with attention overlay
attn_viz.plot_signal_attention(
    raw_signal,           # np.ndarray [seq_len]
    x,                    # preprocessed tensor [1, C, T]
    patch_size=64,        # patch length for PatchTST
    save_path="signal_attn.png"
)
```

---

## 11. Interactive XAI Dashboard

**File:** `xai_dashboard.py`  
**Framework:** Streamlit

An interactive web application that consolidates all XAI methods into a single interface.

### Supported Methods

- Integrated Gradients
- SHAP (via `explainability.shap_explainer`)
- LIME (via `explainability.lime_explainer`)
- Saliency Maps (vanilla gradient, SmoothGrad, GradCAM-1D)
- Uncertainty Quantification (via `explainability.uncertainty_quantification`)
- Counterfactual Explanations
- Concept Activation Vectors (optional, if `explainability.concept_activation_vectors` is available)

### Running the Dashboard

```bash
streamlit run visualization/xai_dashboard.py
```

The dashboard automatically detects CUDA availability and provides:

- Signal type selection (healthy, ball fault, inner race, outer race)
- RPM and signal length configuration
- Per-method visualization panels with interactive controls

---

## 12. Publication-Quality Settings

The module sets global matplotlib defaults in `feature_visualization.py`:

```python
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (10, 8),
    'figure.dpi': 150
})
```

### Customization Per Call

Every plotting method accepts `figsize` and `save_path` parameters:

```python
# Override default figure size
viz.plot_conv_filters(figsize=(20, 12), save_path="filters.pdf")
```

### Color Palettes

- **Fault class colors** are defined in `latent_space_analysis.py` and `feature_visualization.py` using 11-color palettes mapping to: Normal, Ball (3 severities), Inner Race (3 severities), Outer Race (3 severities), Combined.
- **Attention maps** use a custom `LinearSegmentedColormap` (`white → yellow → orange → red`) defined in `attention_viz.py`.

---

## 13. Integration with Research Scripts

The visualization module is designed to be called from research experiment scripts in `scripts/research/`. Typical workflow:

```python
# In a research experiment script
from visualization import FeatureVisualizer, PerformancePlotter
from visualization.latent_space_analysis import LatentSpaceAnalyzer

# After training...
analyzer = LatentSpaceAnalyzer()
physics_features, labels = analyzer.extract_features(model, test_loader)

# Generate publication figures
analyzer.plot_comparison(
    physics_features, cnn_features, labels,
    save_path="results/latent_comparison.png"
)

PerformancePlotter.plot_confusion_matrix(
    cm, class_names=class_names,
    save_path="results/confusion_matrix.pdf"
)
```

---

## 14. Matplotlib Conventions

- All plotting functions return matplotlib `Figure` objects (or `Axes` where noted).
- Figures are **not** shown automatically (`plt.show()` is not called) — the caller controls display.
- Saving is done via `fig.savefig(save_path, dpi=150, bbox_inches='tight')`.
- `plt.tight_layout()` is called before saving in all functions.
- Grid lines use `alpha=0.3` for subtlety.
- DPI defaults to **150** for saved figures.
