# results/ — index (what each folder is)

Every committed folder holds small evidence files (json/md/png). Model
checkpoints (`*.pth`, ~45 MB each) are NOT in git; full archives with
checkpoints live at `D:\Libraries\` (see bottom).

| Folder | Contribution | What it is |
|---|---|---|
| `FINDINGS.md` | — | Phase-5 synthesis + frozen claims list (owner-ratified at Gate 5) |
| `dataset_v2_validation/` | C1 | dataset_v2.h5 validation (stratification, leakage, splits) |
| `benchmark/` | C2 | frozen 11-model pre-registered benchmark (Phase 4) + deployment-feeding numbers |
| `deployment/` | C5 | ONNX export / latency / INT8 appendix |
| `noise_robustness/` | C3 §8.1 | 24 frozen checkpoints × SNR-20/10/5 + degradation summary |
| `phase5/` | C3 §8.2–8.5 | the 45 INERT-loss Colab runs (data-efficiency, severity-OOD, ablation-before, true-metadata) |
| `phase5_fixed/` | C3 §8.4 | 9 runs with the FIXED (differentiable) loss + `before_after_8_4.md` |
| `phase5_dataeff_fixed/` | C3 §8.2 | 21 runs, fixed-loss low-data test + `before_after_8_2.md` |
| `xai_alignment/` | C4 §8.6a | IG attribution-vs-physics-band alignment + `findings_8_6.md` |
| `uncertainty/` | C4 §8.6b | MC-dropout calibration (ECE, reliability, reject curve) |
| `cnn1d_v1_baseline/` | history | Phase-1 first real artifact (86.48%) |
| `cnn1d_v2_baseline/` | history | Phase-3 dataset-v2 baseline (90.53%) |

**Headline:** physics-informed learning gives no accuracy advantage on this
clean synthetic data (C3 negative, all regimes) but a modest interpretability/
calibration gain (C4). See `FINDINGS.md`.

**Full checkpoint archives (off-repo, `D:\Libraries\`):**
- `results_phase5-20260613T100807Z-3-001` — 45 inert runs + 45 ckpts (= `phase5/`)
- `results_phase5_fixed_full-20260613T221621Z-3-001` — 9 fixed §8.4 runs + ckpts (= `phase5_fixed/`)
- `results_phase5_dataeff_fixed` — 21 fixed §8.2 runs + ckpts (= `phase5_dataeff_fixed/`)
