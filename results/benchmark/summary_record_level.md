# Benchmark Summary — RECORD LEVEL (P6 remediation Step 2)

Generated 2026-06-14T15:36:58.825505+00:00 @ `0d25d205` on DESKTOP-M7RBFOO.

> **Statistical unit corrected.** The Phase-4 table and significance tests
> (`summary.md`) used 2,640 one-second windows as if independent. They are
> not: 5 windows per 5-second record share fault, severity, operating point
> and noise. Below, each record is aggregated by **soft vote** (mean class
> score over its 5 windows) and all statistics use the **528 records** as the
> independent unit (external audit Finding 12).
>
> **This is a recomputation of the CLASSIFICATION benchmark only.** It does
> not re-label or re-train. Rows flagged below kept their Phase-4 training
> reality (pc_cnn = CE-only/architecture, multitask = single-task, hybrid =
> rolling-element branch + constant metadata). No "physics-family" average is
> computed. This is **not** a test of physics-informed training (Steps 4-5).

| Model | Record acc (mean ± std over seeds) | 95% CI (record bootstrap of seed-mean) | Window acc (recorded) | Note |
|---|---|---|---|---|
| random_forest | 98.74 ± 0.09 | [97.79, 99.56] | 94.61 |  |
| svm | 97.73 ± 0.00 | [96.40, 98.86] | 94.05 |  |
| gradient_boosting | 97.92 ± 0.15 | [96.65, 98.99] | 94.05 |  |
| cnn1d | 93.43 ± 4.26 | [91.92, 94.89] | 91.94 |  |
| attention_cnn | 93.94 ± 4.27 | [92.36, 95.39] | 89.37 |  |
| cnn_lstm | 99.43 ± 0.15 | [98.80, 99.94] | 96.12 |  |
| resnet18 | 99.18 ± 0.09 | [98.48, 99.75] | 96.14 |  |
| patchtst | 90.40 ± 0.47 | [88.57, 92.17] | 89.85 |  |
| hybrid_pinn **(physics-labeled)** | 90.34 ± 0.94 | [88.13, 92.49] | 90.04 | rolling-element branch + constant metadata |
| physics_constrained_cnn **(physics-labeled)** | 98.99 ± 0.45 | [98.23, 99.62] | 95.98 | CE-only (architecture; physics loss OFF) |
| multitask_pinn **(physics-labeled)** | 90.47 ± 0.62 | [88.51, 92.36] | 90.28 | single-task (aux heads unused) |

Classical bar (RandomForest, record level): 98.74%

## Record-level significance

Best vanilla (deep): **cnn_lstm** (seed-mean 99.43%). Best physics-labeled: **physics_constrained_cnn** (seed-mean 98.99%) — *CE-only (architecture; physics loss OFF)*.

Paired tests below use each model's **best-validation seed** (single model, no ensembling): cnn_lstm seed0 = 99.62%, physics_constrained_cnn seed2 = 99.62%.

Gap (best physics-labeled − best vanilla, best-val seeds): +0.00 pts, 95% CI [+0.00, +0.00] (cluster bootstrap by record).

Wilcoxon over seeds (n=3): p = 0.500 — n=3 seeds; min attainable two-sided p = 0.25 — McNemar is the powered test.

McNemar (exact, record level):
- cnn_lstm_vs_resnet18: p = 0.5
- cnn_lstm_vs_physics_constrained_cnn: p = 1
- resnet18_vs_physics_constrained_cnn: p = 0.5
- physics_constrained_cnn_vs_cnn_lstm: p = 1
- physics_constrained_cnn_vs_resnet18: p = 0.5

_Source: `scripts/aggregate_benchmark_record_level.py`; per-window score cache under `results/benchmark/record_level/_cache/`. Each model passed a sanity gate: argmax of cached scores reproduces its recorded window accuracy._