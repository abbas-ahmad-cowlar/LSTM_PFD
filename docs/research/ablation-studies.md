# Ablation Studies

This document outlines the ablation study methodology for validating model contributions.

## Overview

Ablation studies systematically remove or modify model components to understand their contribution.

---

## Study Design

### 1. Architecture Ablations

| Variant       | Modification            | Δ Accuracy                          |
| ------------- | ----------------------- | ----------------------------------- |
| Full Model    | Baseline                | [PENDING — run experiment to fill]  |
| No Residual   | Remove skip connections | [PENDING — run experiment to fill]  |
| No BatchNorm  | Remove BN layers        | [PENDING — run experiment to fill]  |
| No Dropout    | Remove regularization   | [PENDING — run experiment to fill]  |
| Smaller Width | Half channels           | [PENDING — run experiment to fill]  |

### 2. Physics Loss Ablations (PINN)

| Constraint  | Removed                  | Δ Accuracy                          |
| ----------- | ------------------------ | ----------------------------------- |
| Full PINN   | Baseline                 | [PENDING — run experiment to fill]  |
| No Energy   | $\mathcal{L}_{energy}$   | [PENDING — run experiment to fill]  |
| No Momentum | $\mathcal{L}_{momentum}$ | [PENDING — run experiment to fill]  |
| No Bearing  | $\mathcal{L}_{bearing}$  | [PENDING — run experiment to fill]  |
| Data Only   | All physics              | [PENDING — run experiment to fill]  |

### 3. Ensemble Component Ablations

| Removed Model | Δ Accuracy                          |
| ------------- | ----------------------------------- |
| None (Full)   | [PENDING — run experiment to fill]  |
| -ResNet       | [PENDING — run experiment to fill]  |
| -Transformer  | [PENDING — run experiment to fill]  |
| -PINN         | [PENDING — run experiment to fill]  |
| -EfficientNet | [PENDING — run experiment to fill]  |

---

## Statistical Significance

We report results with 95% confidence intervals:

\[
\bar{x} \pm 1.96 \frac{s}{\sqrt{n}}
\]

McNemar's test for paired comparison:

\[
\chi^2 = \frac{(b - c)^2}{b + c}
\]

Where $b$ and $c$ are discordant pairs.

---

## Running Ablations

```bash
python scripts/run_ablations.py \
    --model resnet18 \
    --ablations all \
    --seeds 42,123,456
```

---

## See Also

- [PINN Theory](pinn-theory.md)
- [Ensemble Strategies](ensemble-strategies.md)
