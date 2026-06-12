# Benchmark Summary — Dataset v2, frozen protocol

Generated 2026-06-12T05:53:09.417777+00:00 @ e498deb0

| Model | Test acc (mean ± std) | Macro-F1 | Seeds |
|---|---|---|---|
| random_forest | 94.61 ± 0.05 | 0.9463 | 3 |
| svm | 94.05 ± 0.00 | 0.9416 | 3 |
| gradient_boosting | 94.05 ± 0.03 | 0.9411 | 3 |
| cnn1d | 91.94 ± 2.84 | 0.9170 | 3 |
| attention_cnn | 89.37 ± 4.82 | 0.8939 | 3 |
| cnn_lstm | 96.12 ± 0.16 | 0.9609 | 3 |
| resnet18 | 96.14 ± 0.28 | 0.9608 | 3 |
| patchtst | 89.85 ± 0.19 | 0.8821 | 3 |
| hybrid_pinn **(physics)** | 90.04 ± 0.51 | 0.8948 | 3 |
| physics_constrained_cnn **(physics)** | 95.98 ± 0.36 | 0.9595 | 3 |
| multitask_pinn **(physics)** | 90.28 ± 0.64 | 0.8946 | 3 |
| voting_ensemble | 96.48 ± 0.00 | 0.9641 | 1 |

Classical bar (RF): 94.61%
Best physics: physics_constrained_cnn | Best vanilla: resnet18 | Wilcoxon p=0.500 (n=3, see note)
McNemar (paired, exact):
- cnn_lstm_vs_physics_constrained_cnn: p = 0.2188
- cnn_lstm_vs_resnet18: p = 0.4531
- physics_constrained_cnn_vs_resnet18: p = 1