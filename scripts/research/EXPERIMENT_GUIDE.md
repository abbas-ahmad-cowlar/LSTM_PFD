# Experiment Guide

> Detailed methodology, configuration, and interpretation guide for each research experiment script.

---

## 1. Ablation Study (`ablation_study.py`)

### Purpose

Systematically removes or modifies model components to quantify their individual contribution. Supports five ablation categories: physics loss, attention mechanism, multi-scale processing, data augmentation, and loss function design.

### Methodology

1. Define a **baseline configuration** with all components enabled.
2. For each component category, create variant configurations with that component disabled or modified.
3. Train each configuration across multiple random seeds (default: 5).
4. Report mean accuracy and standard deviation, plus Δ from baseline.

### Inputs

- **Data:** Synthetic vibration signals generated internally, or real HDF5 data via `--data-path`.
- **Model:** `AblationModel` — a configurable 1D CNN with optional multi-scale blocks, squeeze-excitation, and physics loss.

### CLI Reference

| Argument              | Type | Default                  | Description                                                                              |
| --------------------- | ---- | ------------------------ | ---------------------------------------------------------------------------------------- |
| `--component` / `-c`  | str  | `all`                    | Component to ablate: `all`, `physics`, `attention`, `multiscale`, `augmentation`, `loss` |
| `--seeds` / `-s`      | int  | `5`                      | Number of random seeds per configuration                                                 |
| `--epochs` / `-e`     | int  | `50`                     | Training epochs per configuration                                                        |
| `--output-dir` / `-o` | str  | `results/ablation_study` | Output directory                                                                         |
| `--quick`             | flag | —                        | Run with 3 minimal configurations for testing                                            |
| `--device`            | str  | auto                     | `cpu` or `cuda`                                                                          |
| `--data-path`         | str  | None                     | Path to real dataset (HDF5)                                                              |

### Expected Outputs

- Console summary table: configuration name, mean accuracy, Δ accuracy
- JSON results file in the output directory

### Interpreting Results

- A negative Δ accuracy indicates that removing the component **hurts** performance — the larger the drop, the more important that component.
- Use at least 5 seeds for statistically meaningful comparisons.

### Performance

> ⚠️ **Results pending.** Metrics will be populated after experiments are run on the current codebase.

| Configuration             | Accuracy                             | Δ Accuracy                           |
| ------------------------- | ------------------------------------ | ------------------------------------ |
| Baseline (all components) | `[PENDING — run experiment to fill]` | —                                    |
| Component ablations       | `[PENDING — run experiment to fill]` | `[PENDING — run experiment to fill]` |

---

## 2. Contrastive Physics Pretraining (`contrastive_physics.py`)

### Purpose

Evaluates whether contrastive pretraining using physics-based similarity improves downstream classification compared to purely supervised training.

### Methodology

1. **Pretraining phase:** Train a `SignalEncoder` using `PhysicsInfoNCELoss` / `NTXentLoss` on contrastive pairs, where positive pairs are determined by physics parameter similarity (threshold-based).
2. **Fine-tuning phase:** Attach a classification head (`ContrastiveClassifier`) and evaluate three approaches:
   - Supervised baseline (trained from scratch)
   - Contrastive pretrained + frozen encoder (linear probe)
   - Contrastive pretrained + full fine-tuning
3. Repeat across multiple seeds and report mean ± std for accuracy and F1.

### Inputs

- Synthetic signals with physics parameters generated internally.

### CLI Reference

| Argument            | Type | Default               | Description                    |
| ------------------- | ---- | --------------------- | ------------------------------ |
| `--num-samples`     | int  | `1000`                | Number of training samples     |
| `--pretrain-epochs` | int  | `50`                  | Contrastive pretraining epochs |
| `--finetune-epochs` | int  | `30`                  | Fine-tuning/supervised epochs  |
| `--num-seeds`       | int  | `3`                   | Number of random seeds         |
| `--device`          | str  | auto                  | `cpu` or `cuda`                |
| `--output-dir`      | str  | `results/contrastive` | Output directory for plots     |

### Expected Outputs

- Training curves plot (PNG)
- Benchmark comparison bar chart (PNG)
- Console summary of accuracy/F1 per method

### Interpreting Results

- Compare contrastive methods against the supervised baseline.
- If freezing the encoder yields competitive results, the learned representations are transferable.

### Performance

> ⚠️ **Results pending.**

| Method                   | Accuracy                             | F1 Score                             |
| ------------------------ | ------------------------------------ | ------------------------------------ |
| Supervised baseline      | `[PENDING — run experiment to fill]` | `[PENDING — run experiment to fill]` |
| Contrastive (frozen)     | `[PENDING — run experiment to fill]` | `[PENDING — run experiment to fill]` |
| Contrastive (fine-tuned) | `[PENDING — run experiment to fill]` | `[PENDING — run experiment to fill]` |

---

## 3. Failure Analysis (`failure_analysis.py`)

### Purpose

Analyzes misclassified samples to identify systematic failure modes. Computes signal characteristics (SNR, RMS) of misclassified samples and generates manuscript-ready tables and visualizations.

### Methodology

1. Compare ground-truth labels against predictions.
2. Identify misclassified samples and compute per-sample features: SNR and RMS.
3. Group failures by class and confusion pattern.
4. Generate summary tables and (optionally) Plotly visualizations.

### Inputs

- Predictions and ground truth (provided programmatically, or synthetic in `--demo` mode).
- Raw signals for feature computation.

### CLI Reference

| Argument       | Type | Default        | Description                  |
| -------------- | ---- | -------------- | ---------------------------- |
| `--output-dir` | str  | `docs/figures` | Output directory for figures |
| `--demo`       | flag | —              | Run with synthetic demo data |

### Expected Outputs

- Console: per-class failure rates, confusion pattern summary
- Figures: failure distribution plots (PNG or HTML if Plotly available)
- Manuscript-ready LaTeX table fragments (console output)

### Interpreting Results

- High failure rates for specific class pairs indicate systematic confusion patterns.
- Low-SNR failures suggest the model struggles with noisy signals.

---

## 4. Hyperparameter Sensitivity (`hyperparameter_sensitivity.py`)

### Purpose

Performs grid search over physics loss weight hyperparameters to understand their effect on PINN model performance.

### Methodology

1. Load or generate training data.
2. Define a grid of physics loss weights (energy, momentum, bearing).
3. Train a PINN model for each weight combination.
4. Plot sensitivity surfaces and generate a report.

### Inputs

- Synthetic data or real HDF5 data via `--data-path`.

### CLI Reference

| Argument       | Type | Default               | Description                       |
| -------------- | ---- | --------------------- | --------------------------------- |
| `--data-path`  | str  | None                  | Path to real dataset              |
| `--epochs`     | int  | `30`                  | Training epochs per configuration |
| `--batch-size` | int  | `64`                  | Batch size                        |
| `--output-dir` | str  | `results/sensitivity` | Output directory                  |
| `--quick`      | flag | —                     | Reduced grid for quick testing    |

### Expected Outputs

- Sensitivity surface plots (PNG)
- JSON report with per-configuration results
- Console summary of best weight combinations

### Interpreting Results

- Flat sensitivity surfaces indicate robustness to that hyperparameter.
- Sharp peaks indicate the model is sensitive — tune carefully.
- The optimal region identifies recommended weight ranges.

### Performance

> ⚠️ **Results pending.**

| Best Weight Config                   | Accuracy                             |
| ------------------------------------ | ------------------------------------ |
| `[PENDING — run experiment to fill]` | `[PENDING — run experiment to fill]` |

---

## 5. Out-of-Distribution Testing (`ood_testing.py`)

### Purpose

Tests model robustness against two types of distribution shift: severity shifts (same fault type, different severity) and novel fault types (unseen classes at test time).

### Methodology

1. **Severity shift:** Train on a subset of severities, test on held-out severity levels.
2. **Novel faults:** Train on a subset of fault classes, test on entirely unseen classes.
3. Evaluate using accuracy, maximum softmax probability (confidence calibration), and prediction entropy.

### Inputs

- Synthetic data or `--demo` mode.

### CLI Reference

| Argument       | Type | Default       | Description                             |
| -------------- | ---- | ------------- | --------------------------------------- |
| `--ood-type`   | str  | `severity`    | Type of OOD test: `severity` or `novel` |
| `--num-seeds`  | int  | `3`           | Number of random seeds                  |
| `--epochs`     | int  | `30`          | Training epochs                         |
| `--output-dir` | str  | `results/ood` | Output directory                        |
| `--demo`       | flag | —             | Run with synthetic demo data            |

### Expected Outputs

- JSON results with per-seed metrics
- Console summary: in-distribution vs OOD accuracy, confidence, entropy

### Interpreting Results

- A large accuracy drop on OOD data indicates fragile generalization.
- High confidence (max softmax) on OOD samples indicates poor calibration — the model is confidently wrong.
- High entropy on OOD samples is desirable — indicates the model is uncertain about unfamiliar inputs.

### Performance

> ⚠️ **Results pending.**

| OOD Type       | In-Dist Accuracy                     | OOD Accuracy                         |
| -------------- | ------------------------------------ | ------------------------------------ |
| Severity shift | `[PENDING — run experiment to fill]` | `[PENDING — run experiment to fill]` |
| Novel faults   | `[PENDING — run experiment to fill]` | `[PENDING — run experiment to fill]` |

---

## 6. PINN Ablation (`pinn_ablation.py`)

### Purpose

Validates the contribution of individual physics-informed components (energy conservation, momentum conservation, bearing dynamics) through systematic ablation.

### Methodology

1. Define ablation configurations: Full PINN, No Energy, No Momentum, No Bearing, Data Only.
2. Load data and create PINN models (with fallback to CNN1D if PINN import fails).
3. Train and evaluate each configuration.
4. Compare against data-only baseline.

### CLI Reference

| Argument  | Type | Default | Description                       |
| --------- | ---- | ------- | --------------------------------- |
| `--quick` | flag | —       | Reduced epochs and configurations |

### Expected Outputs

- Console: per-configuration accuracy and comparison table
- Evaluates which physics constraints contribute most to performance

### Interpreting Results

- Compare each ablation against the full PINN and data-only baseline.
- The component whose removal causes the largest accuracy drop is most important.

### Performance

> ⚠️ **Results pending.**

| Configuration | Accuracy                             |
| ------------- | ------------------------------------ |
| Full PINN     | `[PENDING — run experiment to fill]` |
| No Energy     | `[PENDING — run experiment to fill]` |
| No Momentum   | `[PENDING — run experiment to fill]` |
| No Bearing    | `[PENDING — run experiment to fill]` |
| Data Only     | `[PENDING — run experiment to fill]` |

---

## 7. PINN Comparison (`pinn_comparison.py`)

### Purpose

Compares three PINN formulations side-by-side: Raissi et al. (2019), Karniadakis et al. (2021), and the project's custom approach, against a pure data-driven baseline.

### Methodology

1. Implement distinct physics loss functions for each formulation.
2. Train each formulation on the same data with identical hyperparameters.
3. Evaluate accuracy, physics loss, and total loss.
4. Output comparison as JSON.

### CLI Reference

| Argument      | Type | Default                        | Description                      |
| ------------- | ---- | ------------------------------ | -------------------------------- |
| `--data-path` | str  | None                           | Path to real dataset             |
| `--epochs`    | int  | `50`                           | Training epochs                  |
| `--output`    | str  | `results/pinn_comparison.json` | Output JSON file                 |
| `--quick`     | flag | —                              | Reduced epochs for quick testing |

### Expected Outputs

- JSON file with per-formulation metrics
- Console summary table

### Interpreting Results

- Compare accuracy improvements over the baseline for each formulation.
- Lower physics loss indicates better physical consistency.
- The best formulation balances accuracy and physics adherence.

### Performance

> ⚠️ **Results pending.**

| Formulation             | Accuracy                             | Physics Loss                         |
| ----------------------- | ------------------------------------ | ------------------------------------ |
| Baseline (data only)    | `[PENDING — run experiment to fill]` | N/A                                  |
| Raissi formulation      | `[PENDING — run experiment to fill]` | `[PENDING — run experiment to fill]` |
| Karniadakis formulation | `[PENDING — run experiment to fill]` | `[PENDING — run experiment to fill]` |
| Our approach            | `[PENDING — run experiment to fill]` | `[PENDING — run experiment to fill]` |

---

## 8. Transformer Benchmark (`transformer_benchmark.py`)

### Purpose

Benchmarks Transformer-based architectures against a CNN1D baseline for vibration signal classification.

### Methodology

1. **Model registry** contains: `SignalTransformer`, `PatchTST`, `TSMixer`, and `CNN1D` baseline.
2. Train each model with identical data, hyperparameters, and seed.
3. Report accuracy, parameter count, and training time.

### CLI Reference

| Argument       | Type       | Default                              | Description                           |
| -------------- | ---------- | ------------------------------------ | ------------------------------------- |
| `--data-path`  | str        | None                                 | Path to real dataset                  |
| `--models`     | str (list) | all registered                       | Models to benchmark (space-separated) |
| `--epochs`     | int        | `50`                                 | Training epochs                       |
| `--batch-size` | int        | `64`                                 | Batch size                            |
| `--seed`       | int        | `42`                                 | Random seed                           |
| `--output`     | str        | `results/transformer_benchmark.json` | Output JSON file                      |
| `--quick`      | flag       | —                                    | Reduced epochs                        |

### Expected Outputs

- JSON file with per-model results (accuracy, params, training time)
- Console comparison table

### Interpreting Results

- Compare Transformer architectures against CNN1D baseline.
- Consider the accuracy-efficiency trade-off: Transformers may improve accuracy but at higher parameter/time cost.

### Performance

> ⚠️ **Results pending.**

| Model             | Accuracy                             | Parameters                           | Training Time                        |
| ----------------- | ------------------------------------ | ------------------------------------ | ------------------------------------ |
| CNN1D             | `[PENDING — run experiment to fill]` | `[PENDING — run experiment to fill]` | `[PENDING — run experiment to fill]` |
| SignalTransformer | `[PENDING — run experiment to fill]` | `[PENDING — run experiment to fill]` | `[PENDING — run experiment to fill]` |
| PatchTST          | `[PENDING — run experiment to fill]` | `[PENDING — run experiment to fill]` | `[PENDING — run experiment to fill]` |
| TSMixer           | `[PENDING — run experiment to fill]` | `[PENDING — run experiment to fill]` | `[PENDING — run experiment to fill]` |

---

## 9. XAI Metrics (`xai_metrics.py`)

### Purpose

Implements quantitative evaluation metrics for explainability methods (SHAP, LIME, Integrated Gradients, etc.) and generates an expert validation survey template.

### Methodology

The `XAIEvaluator` class computes four quality metrics:

| Metric           | Definition                                                                                             |
| ---------------- | ------------------------------------------------------------------------------------------------------ |
| **Faithfulness** | Correlation between attribution values and actual feature importance (measured by perturbation impact) |
| **Stability**    | Consistency of explanations for similar inputs: `1 - ‖φ(x) - φ(x+ε)‖ / ‖φ(x)‖`                         |
| **Sparsity**     | Fraction of near-zero attributions — sparser explanations are more interpretable                       |
| **Infidelity**   | Measures how well the explanation approximates the model's behavior under perturbation                 |

### Inputs

- A trained model and test signals (provided programmatically).
- No CLI arguments — run as a module.

### Expected Outputs

- Console: metric values per XAI method
- `create_survey_template()` generates a structured expert validation questionnaire

### Interpreting Results

- **Faithfulness > 0.7** indicates the explanation reliably reflects model behavior.
- **Stability > 0.8** indicates robust explanations.
- **Higher sparsity** is better for interpretability but may sacrifice completeness.
- **Lower infidelity** is better — the explanation closely matches actual model behavior.
