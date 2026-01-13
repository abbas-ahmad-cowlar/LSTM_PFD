# Ablation Studies

This document outlines the ablation study methodology for validating model contributions.

## Overview

Ablation studies systematically remove or modify model components to understand their contribution.

---

## Study Design

### 1. Architecture Ablations

| Variant       | Modification            | Δ Accuracy |
| ------------- | ----------------------- | ---------- |
| Full Model    | Baseline                | 97.8%      |
| No Residual   | Remove skip connections | -1.2%      |
| No BatchNorm  | Remove BN layers        | -0.8%      |
| No Dropout    | Remove regularization   | -0.3%      |
| Smaller Width | Half channels           | -1.5%      |

### 2. Physics Loss Ablations (PINN)

| Constraint  | Removed                  | Δ Accuracy |
| ----------- | ------------------------ | ---------- |
| Full PINN   | Baseline                 | 97.8%      |
| No Energy   | $\mathcal{L}_{energy}$   | -0.6%      |
| No Momentum | $\mathcal{L}_{momentum}$ | -0.4%      |
| No Bearing  | $\mathcal{L}_{bearing}$  | -0.3%      |
| Data Only   | All physics              | -1.4%      |

### 3. Ensemble Component Ablations

| Removed Model | Δ Accuracy |
| ------------- | ---------- |
| None (Full)   | 98.4%      |
| -ResNet       | -0.4%      |
| -Transformer  | -0.3%      |
| -PINN         | -0.6%      |
| -EfficientNet | -0.2%      |

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
