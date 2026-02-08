# Interactive Visualizations

This page contains interactive Plotly visualizations that demonstrate the model performance and analysis results.

!!! tip "Interactive Features"
All plots below support zoom, pan, and hover interactions. Click on legend items to toggle visibility.

---

## Confusion Matrix

Interactive confusion matrix for the ensemble model achieving [PENDING — run experiment to fill] accuracy across all 11 fault classes.

<iframe src="../../assets/interactive/confusion_matrix.html" 
        width="100%" height="750px" frameBorder="0"
        style="border: 1px solid #ddd; border-radius: 8px;">
</iframe>

---

## ROC Curves

Multi-class ROC curves (One-vs-Rest) showing classification performance with average AUC of [PENDING — run experiment to fill].

<iframe src="../../assets/interactive/roc_curves.html" 
        width="100%" height="650px" frameBorder="0"
        style="border: 1px solid #ddd; border-radius: 8px;">
</iframe>

---

## Training Progress

Training and validation curves showing stable convergence over 150 epochs.

<iframe src="../../assets/interactive/training_curves.html" 
        width="100%" height="500px" frameBorder="0"
        style="border: 1px solid #ddd; border-radius: 8px;">
</iframe>

---

## Feature Importance (SHAP)

SHAP summary plot showing the most important features for fault classification.

<iframe src="../../assets/interactive/shap_summary.html" 
        width="100%" height="650px" frameBorder="0"
        style="border: 1px solid #ddd; border-radius: 8px;">
</iframe>

---

## Hyperparameter Optimization Surface

3D visualization of the HPO search space showing the relationship between learning rate, batch size, and accuracy.

<iframe src="../../assets/interactive/hpo_surface.html" 
        width="100%" height="750px" frameBorder="0"
        style="border: 1px solid #ddd; border-radius: 8px;">
</iframe>

---

## Regenerating Plots

To regenerate these interactive plots with your own data:

```bash
python scripts/export_interactive_plots.py
```

This will create/overwrite the HTML files in `docs/assets/interactive/`.

---

## See Also

- [PINN Theory](pinn-theory.md) - Physics-informed neural network details
- [XAI Methods](xai-methods.md) - Explainability techniques
- [Ablation Studies](ablation-studies.md) - Model comparison results
