# Benchmark Summary — Dataset v2, frozen protocol (WINDOW-LEVEL)

> ⚠ **SIGNIFICANCE SUPERSEDED (P6 remediation, 2026-06-14).** Accuracies below are window-level descriptive numbers (2,640 correlated windows). All significance (McNemar/Wilcoxon) is invalid at the window level and is **superseded by record-level statistics** — see `results/benchmark/summary_record_level.md` (528 independent records). The annotated rows are **NOT physics-informed results** (pc_cnn CE-only, multitask single-task, hybrid rolling-element branch + constant metadata; audit Findings 7-9). Do not cite them as physics.

Generated 2026-06-14T16:10:43.059913+00:00 @ 0d25d205

| Model | Test acc (mean ± std) | Macro-F1 | Seeds | Note |
|---|---|---|---|---|
| random_forest | 94.61 ± 0.05 | 0.9463 | 3 |  |
| svm | 94.05 ± 0.00 | 0.9416 | 3 |  |
| gradient_boosting | 94.05 ± 0.03 | 0.9411 | 3 |  |
| cnn1d | 91.94 ± 2.84 | 0.9170 | 3 |  |
| attention_cnn | 89.37 ± 4.82 | 0.8939 | 3 |  |
| cnn_lstm | 96.12 ± 0.16 | 0.9609 | 3 |  |
| resnet18 | 96.14 ± 0.28 | 0.9608 | 3 |  |
| patchtst | 89.85 ± 0.19 | 0.8821 | 3 |  |
| hybrid_pinn | 90.04 ± 0.51 | 0.8948 | 3 | rolling-element branch + constant metadata |
| physics_constrained_cnn | 95.98 ± 0.36 | 0.9595 | 3 | CE-only (architecture; physics loss OFF) |
| multitask_pinn | 90.28 ± 0.64 | 0.8946 | 3 | single-task (aux heads unused) |
| voting_ensemble | 96.48 ± 0.00 | 0.9641 | 1 |  |

Classical bar (RF): 94.61%

## Window-level significance — SUPERSEDED (descriptive only)

Top deep by accuracy: resnet18 (vanilla) vs physics_constrained_cnn (*CE-only (architecture; physics loss OFF)*); Wilcoxon p=0.500 (n=3). **Use `summary_record_level.md` instead.**

McNemar (paired, WINDOW-level — invalid, superseded):
- cnn_lstm_vs_physics_constrained_cnn: p = 0.2188
- cnn_lstm_vs_resnet18: p = 0.4531
- physics_constrained_cnn_vs_resnet18: p = 1