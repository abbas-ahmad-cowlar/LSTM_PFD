# Ensemble Strategies

This document describes the ensemble learning approaches that achieve 98-99% accuracy.

## Overview

Ensemble methods combine multiple models to improve prediction accuracy and robustness.

```mermaid
graph TB
    subgraph "Diversity"
        A[ResNet-34<br/>96.8%]
        B[Transformer<br/>96.5%]
        C[PINN<br/>97.8%]
        D[EfficientNet<br/>96.2%]
    end

    subgraph "Combination"
        A --> E[Voting]
        B --> E
        C --> E
        D --> E

        A --> F[Stacking]
        B --> F
        C --> F
        D --> F
    end

    E --> G[98.1%]
    F --> H[98.4%]
```

---

## Methods

### 1. Voting Ensemble

Simple but effective combination:

**Hard Voting:**
\[
\hat{y} = \arg\max*c \sum*{i=1}^{M} \mathbf{1}[h_i(x) = c]
\]

**Soft Voting (Weighted):**
\[
\hat{y} = \arg\max*c \sum*{i=1}^{M} w_i \cdot P_i(y = c | x)
\]

```python
from packages.core.models.ensemble import VotingEnsemble

ensemble = VotingEnsemble(
    models=[resnet, transformer, pinn, efficientnet],
    voting='soft',
    weights=[0.2, 0.25, 0.3, 0.25]
)
```

### 2. Stacked Generalization

Use a meta-learner on base model predictions:

\[
\hat{y} = g\left( h_1(x), h_2(x), \ldots, h_M(x) \right)
\]

```python
from packages.core.models.ensemble import StackedEnsemble

ensemble = StackedEnsemble(
    base_models=[resnet, transformer, pinn],
    meta_learner='xgboost'
)
```

### 3. Mixture of Experts (MoE)

Gating network selects experts per input:

\[
\hat{y} = \sum\_{i=1}^{M} g_i(x) \cdot h_i(x)
\]

Where $g_i(x)$ is the gating weight for expert $i$.

---

## Results

| Strategy               | Accuracy  | Latency | Model Size |
| ---------------------- | --------- | ------- | ---------- |
| Single Best (PINN)     | 97.8%     | 25ms    | 47MB       |
| Hard Voting            | 97.9%     | 85ms    | 188MB      |
| Soft Voting            | 98.1%     | 85ms    | 188MB      |
| **Stacking (XGBoost)** | **98.4%** | 95ms    | 195MB      |
| MoE                    | 98.2%     | 90ms    | 200MB      |

---

## Diversity Analysis

Ensemble performance depends on model diversity:

| Model Pair            | Correlation | Error Overlap |
| --------------------- | ----------- | ------------- |
| ResNet vs Transformer | 0.72        | 45%           |
| ResNet vs PINN        | 0.68        | 38%           |
| Transformer vs PINN   | 0.65        | 35%           |

!!! tip "Best Practice"
Combine models with **low error correlation** for maximum ensemble benefit.

---

## See Also

- [PINN Theory](pinn-theory.md)
- [API: Ensemble Models](../api/core/models.md)
