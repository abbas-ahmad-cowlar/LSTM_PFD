# IDB 5.2: Visualization Sub-Block Analysis

**Analyst:** AI Agent  
**Date:** 2026-01-23  
**Scope:** `visualization/` directory (13 files)  
**Purpose:** Publication-quality visualizations: attention maps, latent space (t-SNE/UMAP), feature importance, physics embeddings

---

## Executive Summary

The visualization module provides a comprehensive suite of 13 Python files (~170KB total) covering attention visualization, CNN analysis, saliency maps, latent space projections, spectrograms, and an interactive XAI dashboard. While the module offers **excellent functional coverage**, it suffers from **critical styling inconsistencies** that compromise publication-readiness and maintainability.

| Metric                   | Value              |
| ------------------------ | ------------------ |
| Total Files              | 13                 |
| Total Lines              | ~4,300             |
| Centralized Style Config | ❌ Only 1 file     |
| Colorblind Accessible    | ❌ No              |
| DPI Standard             | ❌ Mixed (150/300) |
| Memory Management        | ⚠️ Inconsistent    |

---

## Task 1: Current State Assessment

### 1.1 Visualization Types Inventory

| File                             | Visualization Type                | Lines | Key Classes/Functions                    |
| -------------------------------- | --------------------------------- | ----- | ---------------------------------------- |
| `attention_viz.py`               | Transformer attention heatmaps    | 429   | `AttentionVisualizer`                    |
| `activation_maps_2d.py`          | 2D CNN Grad-CAM, feature maps     | 532   | `visualize_filters`, `generate_grad_cam` |
| `cnn_analysis.py`                | Gradient flow, ablation, saliency | 600   | `CNNAnalyzer`                            |
| `cnn_visualizer.py`              | 1D CNN filters, activations       | 482   | `CNNVisualizer`                          |
| `counterfactual_explanations.py` | Counterfactual generation         | 419   | `CounterfactualGenerator`                |
| `feature_visualization.py`       | t-SNE/UMAP clusters               | 493   | `FeatureVisualizer`                      |
| `latent_space_analysis.py`       | Physics vs CNN branch comparison  | 429   | `LatentSpaceAnalyzer`                    |
| `performance_plots.py`           | Confusion matrix, ROC curves      | 131   | `PerformancePlotter`                     |
| `saliency_maps.py`               | Gradient-based XAI                | 484   | `SaliencyMapGenerator`                   |
| `signal_plots.py`                | Time/frequency domain signals     | 132   | `SignalPlotter`                          |
| `spectrogram_plots.py`           | STFT, CWT, WVD comparison         | 454   | `plot_spectrogram_comparison`            |
| `xai_dashboard.py`               | Interactive Streamlit dashboard   | 573   | `main()`, render functions               |
| `__init__.py`                    | Module exports                    | 37    | -                                        |

### 1.2 Styling Consistency Audit

#### DPI Settings (Critical Inconsistency)

| DPI Value          | Files Using                                                                                                                                                                 | Count |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| **150**            | `signal_plots.py`, `saliency_maps.py`, `performance_plots.py`, `latent_space_analysis.py`, `feature_visualization.py`, `counterfactual_explanations.py`, `attention_viz.py` | 7     |
| **300**            | `spectrogram_plots.py`, `cnn_visualizer.py`, `cnn_analysis.py`, `activation_maps_2d.py`                                                                                     | 4     |
| **None (default)** | `xai_dashboard.py`                                                                                                                                                          | 1     |

> [!CAUTION]
> **Publication Standard Violation**: Academic publications typically require 300-600 DPI. Half the codebase uses 150 DPI which is insufficient for print.

#### Font Configuration

Only **1 of 13 files** has centralized `plt.rcParams` settings:

```python
# feature_visualization.py (lines 53-62) - ONLY file with this
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

**Impact**: Other files rely on matplotlib defaults, causing inconsistent appearance across visualizations.

#### Colormap Usage

| Colormap   | Files Using                                                                            | Suitability                |
| ---------- | -------------------------------------------------------------------------------------- | -------------------------- |
| `viridis`  | `spectrogram_plots.py`, `signal_plots.py`, `attention_viz.py`, `activation_maps_2d.py` | ✅ Perceptually uniform    |
| `Blues`    | `performance_plots.py`                                                                 | ✅ Good for sequential     |
| `coolwarm` | `feature_visualization.py`                                                             | ✅ Good for diverging      |
| `RdBu_r`   | `cnn_visualizer.py`                                                                    | ✅ Good for diverging      |
| `jet`      | `activation_maps_2d.py`                                                                | ❌ **Not colorblind-safe** |
| `hot`      | `activation_maps_2d.py`                                                                | ⚠️ Limited accessibility   |

### 1.3 Input/Output Format Mapping

| Function Category | Input Format                     | Output Format |
| ----------------- | -------------------------------- | ------------- |
| Model visualizers | PyTorch model + tensor           | PNG/figure    |
| Feature plots     | NumPy arrays                     | PNG/figure    |
| Signal plots      | NumPy + sampling rate            | PNG/figure    |
| Spectrogram plots | NumPy 1D signal                  | PNG/figure    |
| Performance plots | Confusion matrix / probabilities | PNG/figure    |

**Output path handling**: All functions use optional `save_path` parameter with `bbox_inches='tight'`.

---

## Task 2: Critical Issues Identification

### P0 - Critical (Must Fix)

| ID   | Issue                       | Location                                                     | Impact                                        |
| ---- | --------------------------- | ------------------------------------------------------------ | --------------------------------------------- |
| P0-1 | **DPI inconsistency**       | All files                                                    | Publications rejected; figures look different |
| P0-2 | **No unified style config** | Missing                                                      | 13 files with independent styling             |
| P0-3 | **Duplicate CLASS_COLORS**  | `feature_visualization.py:65`, `latent_space_analysis.py:49` | Maintenance nightmare; risk of drift          |
| P0-4 | **Duplicate CLASS_NAMES**   | Same files                                                   | Same issue                                    |

### P1 - High (Should Fix)

| ID   | Issue                          | Location                           | Impact                           |
| ---- | ------------------------------ | ---------------------------------- | -------------------------------- |
| P1-1 | **`jet` colormap used**        | `activation_maps_2d.py:310,322`    | Not colorblind-accessible        |
| P1-2 | **No colorblind options**      | All files                          | Excludes ~8% of male researchers |
| P1-3 | **Font settings not global**   | 12 of 13 files                     | Inconsistent label sizes         |
| P1-4 | **`plt.close()` inconsistent** | Multiple                           | Memory leaks in batch processing |
| P1-5 | **Bare except clauses**        | `feature_visualization.py:337-348` | Swallows errors silently         |

### P2 - Medium (Nice to Have)

| ID   | Issue                                | Location                                               | Impact                       |
| ---- | ------------------------------------ | ------------------------------------------------------ | ---------------------------- |
| P2-1 | Missing figure size standardization  | Multiple                                               | Inconsistent plot dimensions |
| P2-2 | No PDF/SVG vector output option      | All save calls                                         | Limited for publications     |
| P2-3 | Hardcoded axis labels                | Multiple                                               | Not i18n-ready               |
| P2-4 | Missing progress bars for t-SNE/UMAP | `feature_visualization.py`, `latent_space_analysis.py` | Poor UX for large datasets   |

---

## Task 3: "If I Could Rewrite This" Retrospective

### 3.1 Unified Style Configuration Needed?

**YES - Critical Gap**

The module would benefit enormously from a centralized style module:

```python
# visualization/style_config.py (proposed)
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class PublicationStyle:
    dpi: int = 300
    font_size: int = 12
    axes_labelsize: int = 14
    figure_width: float = 10.0
    colormap_sequential: str = 'viridis'
    colormap_diverging: str = 'coolwarm'
    colormap_categorical: List[str] = ...  # CLASS_COLORS

PUBLICATION_STYLE = PublicationStyle()
COLORBLIND_STYLE = PublicationStyle(colormap_sequential='cividis')
```

**Why not done originally?** Files were developed incrementally without architectural planning.

### 3.2 Are Visualization Functions Composable?

**Partially**

| Composability    | Status       | Notes                                  |
| ---------------- | ------------ | -------------------------------------- |
| Figure creation  | ❌ Coupled   | Each function creates its own figure   |
| Subplot addition | ❌ Coupled   | No `ax` injection pattern consistently |
| Color assignment | ❌ Coupled   | Inline `CLASS_COLORS[label]`           |
| Save logic       | ⚠️ Partially | Common pattern but copy-pasted         |

**Recommendation**: Adopt a functional approach where plotting functions accept `ax` parameter:

```python
# Current (bad)
def plot_tsne_clusters(...):
    fig, ax = plt.subplots()  # Creates figure internally

# Proposed (good)
def plot_tsne_clusters(..., ax=None, fig=None):
    if ax is None:
        fig, ax = plt.subplots()
```

### 3.3 Consistent Color Palette?

**NO - Duplicated Definitions**

Two files define `CLASS_COLORS` independently:

| File                             | Colors    | Notes                         |
| -------------------------------- | --------- | ----------------------------- |
| `feature_visualization.py:65-77` | 11 colors | Same color for Ball faults    |
| `latent_space_analysis.py:49-61` | 11 colors | Different shades per severity |

**Problem**: These two definitions **are not identical**:

- `feature_visualization.py` uses `#e74c3c` for all 3 Ball fault severities
- `latent_space_analysis.py` uses `#e74c3c`, `#c0392b`, `#a93226` for different severities

This creates visual inconsistency between t-SNE plots and latent space plots.

---

## Good Practices Worth Preserving

| Practice                       | Example Location                 | Recommendation                   |
| ------------------------------ | -------------------------------- | -------------------------------- |
| UMAP availability check        | `feature_visualization.py:43-47` | Keep optional dependency pattern |
| Docstrings with usage examples | Most docstrings                  | Maintain quality                 |
| Type hints                     | All function signatures          | Expand to return types           |
| `bbox_inches='tight'`          | All save calls                   | Keep for clean exports           |
| Standalone demo functions      | `demo_visualization()` pattern   | Document and test                |
| Clustering metrics computation | `compute_clustering_metrics()`   | Useful for automated analysis    |

---

## Recommended Actions

### Immediate (Week 1)

1. **Create `visualization/style_config.py`** with:
   - Unified DPI (300 for publication, 150 for screen)
   - Single `CLASS_COLORS` and `CLASS_NAMES` definition
   - `apply_publication_style()` function
   - Colorblind-safe palette option

2. **Replace `jet` with `viridis`** in `activation_maps_2d.py`

3. **Standardize DPI to 300** across all files

### Short-term (Week 2-3)

4. **Refactor all plotting functions** to accept optional `ax` parameter

5. **Add `plt.close()` consistently** after all save operations

6. **Fix bare except clauses** in `feature_visualization.py`

### Medium-term (Month 1)

7. **Add vector output option** (PDF/SVG) for publications

8. **Implement colorblind mode** toggle

9. **Add progress bars** for dimensionality reduction operations

---

## Technical Debt Inventory

| Category                | Count  | Priority |
| ----------------------- | ------ | -------- |
| Styling inconsistencies | 5      | P0-P1    |
| Code duplication        | 3      | P0       |
| Accessibility gaps      | 2      | P1       |
| Memory management       | 1      | P1       |
| Missing features        | 4      | P2       |
| **Total**               | **15** | -        |

---

## Appendix: File-by-File Summary

### `__init__.py`

- **Status**: ✅ Clean
- **Exports**: 9 public symbols
- **Missing**: `LatentSpaceAnalyzer`, `CNNAnalyzer` not exported

### `feature_visualization.py`

- **Status**: ⚠️ Best-styled but has duplicate definitions
- **Unique**: Only file with `plt.rcParams` configuration
- **Issues**: Bare except, duplicate CLASS_COLORS

### `latent_space_analysis.py`

- **Status**: ⚠️ Good functionality, duplicate CLASS_COLORS
- **Unique**: Physics vs Data branch comparison
- **Issues**: Different CLASS_COLORS than feature_visualization.py

### `attention_viz.py`

- **Status**: ✅ Well-structured
- **Unique**: Attention rollout implementation
- **Issues**: 150 DPI, no centralized style

### `activation_maps_2d.py`

- **Status**: ⚠️ Uses `jet` colormap
- **Unique**: 2D CNN Grad-CAM
- **Issues**: jet colormap not colorblind-safe

### `cnn_analysis.py` & `cnn_visualizer.py`

- **Status**: ✅ Well-structured, 300 DPI
- **Unique**: Gradient flow analysis, ablation studies
- **Issues**: No shared style config

### `counterfactual_explanations.py`

- **Status**: ✅ Clean
- **Unique**: Optimization-based counterfactuals
- **Issues**: 150 DPI

### `performance_plots.py`

- **Status**: ⚠️ Minimal, needs expansion
- **Lines**: Only 131 lines
- **Issues**: 150 DPI, limited plot types

### `saliency_maps.py`

- **Status**: ✅ Comprehensive XAI methods
- **Unique**: SmoothGrad, GradCAM-1D
- **Issues**: 150 DPI

### `signal_plots.py`

- **Status**: ⚠️ Minimal
- **Lines**: Only 132 lines
- **Issues**: 150 DPI

### `spectrogram_plots.py`

- **Status**: ✅ Well-structured, 300 DPI
- **Unique**: STFT/CWT/WVD comparison
- **Issues**: No colorblind option

### `xai_dashboard.py`

- **Status**: ✅ Interactive Streamlit app
- **Unique**: Unified XAI interface
- **Issues**: No standardized DPI for saved figures
