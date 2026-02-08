# Research Scripts

> Experiment scripts for ablation studies, benchmarks, and research validation of the LSTM_PFD bearing fault diagnosis system.

## Overview

This directory contains 9 standalone research scripts. Each script is self-contained, uses synthetic data by default (no dataset required), and produces results in a configurable output directory. Scripts share common patterns: argparse CLIs, `--quick` mode for fast validation, deterministic seeding, and automatic device selection (CPU/CUDA).

## Script Catalog

| Script                          | Purpose                                                                                        | Key CLI Args                                                                                         | Default Output                       |
| ------------------------------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `ablation_study.py`             | Systematic component ablation (physics, attention, multiscale, augmentation, loss)             | `--component`, `--seeds`, `--epochs`, `--quick`, `--device`, `--data-path`, `--output-dir`           | `results/ablation_study/`            |
| `contrastive_physics.py`        | Compares contrastive pretraining (physics-informed) vs supervised baseline                     | `--num-samples`, `--pretrain-epochs`, `--finetune-epochs`, `--num-seeds`, `--device`, `--output-dir` | `results/contrastive/`               |
| `failure_analysis.py`           | Analyzes misclassified samples to identify failure modes                                       | `--output-dir`, `--demo`                                                                             | `docs/figures/`                      |
| `hyperparameter_sensitivity.py` | Grid search over physics loss weights for PINN models                                          | `--data-path`, `--epochs`, `--batch-size`, `--output-dir`, `--quick`                                 | `results/sensitivity/`               |
| `ood_testing.py`                | Out-of-distribution robustness testing (severity shifts, novel faults)                         | `--ood-type`, `--num-seeds`, `--epochs`, `--output-dir`, `--demo`                                    | `results/ood/`                       |
| `pinn_ablation.py`              | PINN-specific ablation: validates contribution of physics-informed components                  | `--quick`                                                                                            | Console output                       |
| `pinn_comparison.py`            | Compares PINN formulations (Raissi, Karniadakis, Ours) against baseline                        | `--data-path`, `--epochs`, `--output`, `--quick`                                                     | `results/pinn_comparison.json`       |
| `transformer_benchmark.py`      | Benchmarks Transformer architectures (SignalTransformer, PatchTST, TSMixer) vs CNN1D           | `--data-path`, `--models`, `--epochs`, `--batch-size`, `--seed`, `--output`, `--quick`               | `results/transformer_benchmark.json` |
| `xai_metrics.py`                | XAI quality metrics (faithfulness, stability, sparsity, infidelity) and expert survey template | _(no CLI args — run as module)_                                                                      | Console output                       |

## Running Experiments

### Quick Start

Every script with a `--quick` flag supports rapid validation with reduced configurations:

```bash
# Quick ablation study (3 configs instead of full suite)
python scripts/research/ablation_study.py --quick

# Quick PINN comparison (fewer epochs)
python scripts/research/pinn_comparison.py --quick

# Quick hyperparameter sensitivity
python scripts/research/hyperparameter_sensitivity.py --quick
```

### Full Experiments

```bash
# Full ablation study — all components, 5 seeds, 50 epochs each
python scripts/research/ablation_study.py --component all --seeds 5 --epochs 50

# Contrastive pretraining benchmark
python scripts/research/contrastive_physics.py --num-samples 2000 --pretrain-epochs 50 --finetune-epochs 30

# PINN formulation comparison
python scripts/research/pinn_comparison.py --epochs 50

# Transformer architecture benchmark
python scripts/research/transformer_benchmark.py --models SignalTransformer PatchTST TSMixer CNN1D --epochs 50

# Hyperparameter sensitivity analysis
python scripts/research/hyperparameter_sensitivity.py --epochs 30 --batch-size 64

# Out-of-distribution testing
python scripts/research/ood_testing.py --ood-type severity --num-seeds 3 --epochs 30
python scripts/research/ood_testing.py --ood-type novel --num-seeds 3 --epochs 30

# Failure analysis (demo mode with synthetic data)
python scripts/research/failure_analysis.py --demo

# PINN ablation
python scripts/research/pinn_ablation.py

# XAI metrics and survey template
python scripts/research/xai_metrics.py
```

### Using Real Data

Scripts that accept `--data-path` can use the CWRU bearing dataset:

```bash
python scripts/research/ablation_study.py --data-path data/processed/signals_cache.h5 --epochs 100
python scripts/research/pinn_comparison.py --data-path data/processed/signals_cache.h5 --epochs 100
```

> **Note:** If `--data-path` is not provided, scripts generate synthetic vibration data internally for testing purposes. Results from synthetic data are for validation only and should not be reported as benchmark results.

## Output Locations

| Script                          | Output Format                     | Location                             |
| ------------------------------- | --------------------------------- | ------------------------------------ |
| `ablation_study.py`             | JSON + console summary            | `results/ablation_study/`            |
| `contrastive_physics.py`        | PNG plots + console summary       | `results/contrastive/`               |
| `failure_analysis.py`           | PNG/HTML figures + console tables | `docs/figures/`                      |
| `hyperparameter_sensitivity.py` | PNG plots + JSON report           | `results/sensitivity/`               |
| `ood_testing.py`                | JSON + console summary            | `results/ood/`                       |
| `pinn_ablation.py`              | Console output                    | _(stdout only)_                      |
| `pinn_comparison.py`            | JSON                              | `results/pinn_comparison.json`       |
| `transformer_benchmark.py`      | JSON                              | `results/transformer_benchmark.json` |
| `xai_metrics.py`                | Console output                    | _(stdout only)_                      |

## Reproducibility

- **Seeding:** All scripts set seeds for `random`, `numpy`, and `torch` (CPU + CUDA). Most scripts accept `--seeds` or `--seed` to control the number of repetitions or the seed value.
- **Determinism:** PyTorch deterministic mode is used where applicable.
- **Quick mode:** The `--quick` flag reduces the number of configurations and epochs, producing faster but less statistically robust results. Do not use `--quick` results for publication.

## Dependencies

All scripts depend on:

- `torch`, `numpy`, `scikit-learn` (core ML)
- `matplotlib`, `plotly` (visualization, optional for some scripts)
- Project packages: `packages.core.models`, `packages.core.training` (imported with fallbacks to built-in implementations)

## Related Documentation

- [Experiment Guide](EXPERIMENT_GUIDE.md) — Detailed per-script methodology and interpretation
- [docs/research/](../../docs/research/) — Research-grade supporting documentation
