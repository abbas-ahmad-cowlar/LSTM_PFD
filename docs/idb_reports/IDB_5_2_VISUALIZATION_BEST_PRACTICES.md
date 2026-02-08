# IDB 5.2: Visualization Best Practices

**Extracted from:** Visualization Sub-Block Analysis  
**Date:** 2026-01-23  
**Scope:** Publication-quality visualization conventions for the LSTM_PFD project

---

## 1. Visualization Styling Conventions

### Recommended Pattern: Centralized Style Configuration

```python
# visualization/style_config.py (create this file)
import matplotlib.pyplot as plt

def apply_publication_style():
    """Apply consistent styling across all visualizations."""
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })
```

### Usage in All Visualization Files

```python
from visualization.style_config import apply_publication_style
apply_publication_style()  # Call at module level
```

---

## 2. Figure Size & DPI Conventions

### Publication Standards

| Output Type          | DPI     | Figure Size     | Use Case            |
| -------------------- | ------- | --------------- | ------------------- |
| **Print/Journal**    | 300-600 | (10, 8) inches  | Final submission    |
| **Screen/Dashboard** | 150     | (12, 8) inches  | Interactive viewing |
| **Poster**           | 300     | (20, 16) inches | Large format        |

### Recommended Save Pattern

```python
def save_figure(fig, save_path, dpi=300):
    """Standardized figure saving with memory cleanup."""
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)  # Always close after saving
```

> [!IMPORTANT]
> Always call `plt.close(fig)` after saving to prevent memory leaks in batch processing.

---

## 3. Color Palette Conventions

### Single Source of Truth

```python
# visualization/style_config.py
CLASS_COLORS = [
    '#2ecc71',  # 0: Normal - green
    '#e74c3c',  # 1: Ball-007 - red
    '#c0392b',  # 2: Ball-014 - darker red
    '#a93226',  # 3: Ball-021 - darkest red
    '#3498db',  # 4: IR-007 - blue
    '#2980b9',  # 5: IR-014 - darker blue
    '#1f618d',  # 6: IR-021 - darkest blue
    '#f39c12',  # 7: OR-007 - orange
    '#d68910',  # 8: OR-014 - darker orange
    '#b9770e',  # 9: OR-021 - darkest orange
    '#9b59b6'   # 10: Combined - purple
]

CLASS_NAMES = [
    'Normal', 'Ball-007', 'Ball-014', 'Ball-021',
    'IR-007', 'IR-014', 'IR-021',
    'OR-007', 'OR-014', 'OR-021', 'Combined'
]
```

### Colormap Selection Guide

| Data Type              | Recommended                    | Avoid            |
| ---------------------- | ------------------------------ | ---------------- |
| Sequential (intensity) | `viridis`, `plasma`, `cividis` | `jet`, `rainbow` |
| Diverging (±deviation) | `coolwarm`, `RdBu_r`           | `jet`            |
| Categorical            | Custom palette above           | Any sequential   |
| Heatmaps               | `viridis`                      | `hot`, `jet`     |

---

## 4. Font Conventions

### Journal-Ready Typography

```python
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 12,           # Base font
    'axes.labelsize': 14,      # Axis labels (x, y)
    'axes.titlesize': 16,      # Subplot titles
    'figure.titlesize': 18,    # Figure suptitle
    'xtick.labelsize': 11,     # Tick labels
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'legend.title_fontsize': 12
})
```

### Minimum Font Sizes for Print

| Element     | Minimum Size | Recommended |
| ----------- | ------------ | ----------- |
| Axis labels | 10pt         | 12-14pt     |
| Tick labels | 8pt          | 10-11pt     |
| Legend      | 8pt          | 10-11pt     |
| Title       | 12pt         | 14-16pt     |

---

## 5. Accessibility Considerations

### Colorblind-Safe Mode

```python
# visualization/style_config.py
COLORBLIND_COLORS = [
    '#000000',  # Black
    '#E69F00',  # Orange
    '#56B4E9',  # Sky blue
    '#009E73',  # Bluish green
    '#F0E442',  # Yellow
    '#0072B2',  # Blue
    '#D55E00',  # Vermillion
    '#CC79A7'   # Reddish purple
]

def get_colors(colorblind_mode=False):
    return COLORBLIND_COLORS if colorblind_mode else CLASS_COLORS
```

### Best Practices

1. **Never rely on color alone** — use markers, line styles, or patterns
2. **Use perceptually uniform colormaps** — `viridis`, `cividis` (designed for colorblindness)
3. **Avoid red-green combinations** — affects ~8% of males
4. **Test with simulators** — Use tools like Coblis or Color Oracle

### Marker + Color Pattern

```python
MARKERS = ['o', 's', '^', 'v', 'D', 'p', 'h', '*', 'x', '+', 'd']

for i, label in enumerate(unique_labels):
    ax.scatter(x, y,
              color=CLASS_COLORS[i],
              marker=MARKERS[i % len(MARKERS)],  # Different shape per class
              label=CLASS_NAMES[i])
```

---

## 6. Output Format Conventions

### Recommended Formats

| Format  | Use Case       | Pros                    | Cons                      |
| ------- | -------------- | ----------------------- | ------------------------- |
| **PNG** | Web, dashboard | Universal, transparency | Raster, large at high DPI |
| **PDF** | Publications   | Vector, scalable        | Larger file size          |
| **SVG** | Editing, web   | Vector, editable        | Limited support           |
| **EPS** | LaTeX papers   | Vector, legacy support  | No transparency           |

### Multi-Format Save Pattern

```python
def save_publication_figure(fig, base_path, formats=['png', 'pdf']):
    """Save figure in multiple formats for different uses."""
    from pathlib import Path
    base = Path(base_path).with_suffix('')

    for fmt in formats:
        fig.savefig(f"{base}.{fmt}",
                   dpi=300 if fmt == 'png' else None,
                   bbox_inches='tight',
                   format=fmt)
    plt.close(fig)
```

### Function Signature Standard

```python
def plot_something(
    data: np.ndarray,
    labels: np.ndarray,
    *,  # Force keyword arguments after this
    ax: Optional[plt.Axes] = None,          # Allow subplot injection
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
    dpi: int = 300,
    colorblind_mode: bool = False
) -> plt.Figure:
    """
    Standardized function signature for all visualization functions.

    Returns:
        Figure object (caller responsible for plt.close())
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # ... plotting logic ...

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')

    return fig
```

---

## Quick Reference Checklist

Before committing any visualization code:

- [ ] Uses centralized `style_config.py` settings
- [ ] DPI is 300 for saved figures
- [ ] Colors from `CLASS_COLORS` (not hardcoded)
- [ ] `bbox_inches='tight'` on all saves
- [ ] `plt.close(fig)` after saving
- [ ] Colormap is `viridis`/`coolwarm` (not `jet`)
- [ ] Function accepts optional `ax` parameter
- [ ] Markers used in addition to colors (accessibility)
