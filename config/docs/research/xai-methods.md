# Explainable AI (XAI) Methods

This document describes the explainability methods implemented in LSTM PFD for model interpretability.

## Overview

Understanding _why_ a model predicts a particular fault type is crucial for:

- **Trust**: Engineers need to verify predictions align with domain knowledge
- **Debugging**: Identifying when models rely on spurious correlations
- **Compliance**: Meeting regulatory requirements for transparent AI

---

## Attribution Methods

### 1. SHAP (SHapley Additive exPlanations)

Based on game theory, SHAP values quantify each feature's contribution:

\[
\phi*i(f, x) = \sum*{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} \left[ f(S \cup \{i\}) - f(S) \right]
\]

**Usage:**

```python
from packages.core.explainability import SHAPExplainer

explainer = SHAPExplainer(model, background_data[:100])
shap_values = explainer.explain(signal)
explainer.plot_waterfall(shap_values)
```

### 2. LIME (Local Interpretable Model-agnostic Explanations)

LIME fits a local linear model around each prediction:

\[
\xi(x) = \arg\min\_{g \in G} \mathcal{L}(f, g, \pi_x) + \Omega(g)
\]

Where $\pi_x$ is a proximity measure and $\Omega(g)$ is the complexity of $g$.

### 3. Integrated Gradients

For neural networks, we compute path integrals:

\[
IG_i(x) = (x_i - x'\_i) \int_0^1 \frac{\partial f(x' + \alpha(x - x'))}{\partial x_i} d\alpha
\]

### 4. Grad-CAM

For CNNs, we visualize class activation maps:

\[
L^c\_{Grad-CAM} = ReLU\left( \sum_k \alpha^c_k A^k \right)
\]

Where:
\[
\alpha^c*k = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A^k*{ij}}
\]

---

## Comparison

| Method   | Type         | Speed  | Global | Local |
| -------- | ------------ | ------ | ------ | ----- |
| SHAP     | Game Theory  | Slow   | ✅     | ✅    |
| LIME     | Perturbation | Medium | ❌     | ✅    |
| IG       | Gradient     | Fast   | ❌     | ✅    |
| Grad-CAM | Activation   | Fast   | ❌     | ✅    |

---

## Quality Metrics

### Faithfulness

Does the explanation reflect actual model behavior?

\[
\text{Faithfulness} = \text{Corr}\left( \phi_i, \Delta f_i \right)
\]

### Stability

Are explanations consistent for similar inputs?

\[
\text{Stability} = 1 - \frac{\| \phi(x) - \phi(x + \epsilon) \|}{\| \phi(x) \|}
\]

---

## Dashboard Integration

The XAI Dashboard provides interactive visualizations:

1. Navigate to **XAI Dashboard** in the sidebar
2. Select a trained model
3. Choose an explanation method
4. Upload or select a test signal
5. View interactive attribution plots

---

## See Also

- [PINN Theory](pinn-theory.md)
- [User Guide: XAI Dashboard](../user-guide/dashboard/xai.md)
