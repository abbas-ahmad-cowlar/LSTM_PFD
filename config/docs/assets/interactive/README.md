# Interactive Plots Directory

This directory contains interactive HTML visualizations embedded in the MkDocs documentation.

## Contents

Place generated Plotly HTML files here:

- `confusion_matrix.html` - Interactive confusion matrix
- `roc_curves.html` - Multi-class ROC curves
- `training_curves.html` - Training/validation loss curves
- `shap_summary.html` - SHAP feature importance
- `hpo_surface.html` - HPO parameter surface

## Generation

Run the export script to generate these files:

```bash
python scripts/export_interactive_plots.py
```
