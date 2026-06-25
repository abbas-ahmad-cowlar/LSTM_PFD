# Reproducibility / provenance manifest

> One row per paper-facing result: the command, the code commit it was produced at,
> the frozen inputs (content-hashed), the committed output artifact, and where the
> (gitignored) checkpoints live. Closes the chain command → commit → dataset-hash →
> checkpoint → result for a cold referee (audits 2026-06-24, M1). **Authoritative
> verdict: `results/FINDINGS.md` §0 (complete negative).**

## Frozen inputs (content hashes — SHA-256)

| Artifact | SHA-256 | Provenance |
|---|---|---|
| `data/generated/dataset_v2.h5` | `f72ad35b733c5649fc0ed729826c0dc95a85e3c0fd2d3053c425443fe5afcee0` | `scripts/generate_dataset_v2.py`, global seed 42, generated 2026-06-11 |
| `packages/core/models/physics/healthy_reference.json` | `3274296e517393b899a6e95b96e1ecf900fcf2e418e5aea9f89f9853cf667fc3` | `scripts/compute_healthy_reference.py` (1120 train healthy windows) |
| `packages/core/models/physics/random_reference.json` | `946590a4f9fe147ddd42ae25b49e82d43345951143360774ec91ba3588f15d82` | `scripts/compute_random_reference.py`, RNG seed 20260623 (§8.8 control) |

Recompute any hash with `sha256sum <path>`. The two reference JSONs regenerate
byte-identical (the auditor confirmed, 2026-06-24).

## Environment
- **Training:** Google Colab **Tesla T4** (GPU). Frozen budget for every run:
  Adam lr 1e-3, batch 64, ≤60 epochs, patience 10, AMP.
- **Analysis / eval / aggregation:** laptop, **Python 3.14.0, torch 2.9.1+cpu**
  (`requirements.lock.txt`), CPU-only. Set `PYTHONIOENCODING=utf-8`.

## Results

| Result (paper) | Reproduce command | Code commit | Output artifact (committed) | Checkpoints (gitignored → Zenodo) |
|---|---|---|---|---|
| **C1 dataset** | `python scripts/generate_dataset_v2.py` | — | `results/dataset_v2_validation/`; CI `tests/test_physics_signatures.py` | — (dataset hash above) |
| **C2 benchmark (record-level)** | `python scripts/aggregate_benchmark_record_level.py` | runs `e498deb` | `results/benchmark/summary_record_level.{json,md}` | `results/benchmark/deep/*/seed*/best_model.pth` |
| **C3 §8.8 n=12 noise — the headline NEGATIVE** | runner `python scripts/run_phase5_gpu.py --out-root results_p7_strengthen --seeds 0 1 2 3 4 5 6 7 8 9 10 11 {--only pinn_ablation --weights 0.0 1.0 \| --control f9_scramble \| --control random_bands}` → analysis `python scripts/p7_strengthen_record_level.py` | `4a2063d` | `results/p7_strengthen/p7_strengthen_record_level.json` + 48 `metrics.json` | `results/p7_strengthen/**/best_model.pth` (48) |
| §8.2 data-eff / §8.3 OOD (record-level, negative) | `python scripts/phase5_bandenergy_record_level.py` | runs `ce344d1` | `results/phase5_bandenergy/summary_record_level.json` | `results/phase5_bandenergy/{data_efficiency,severity_ood}/**/best_model.pth` |
| §8.7 F9 scramble (n=3, **superseded by §8.8**) | `python scripts/f9_scramble_record_level.py` | runs `70f623f` | `results/phase5_bandenergy/f9_scramble_record_level.json` | `results/phase5_bandenergy/pinn_ablation_scramble/**/best_model.pth` |
| §8.4 band-energy noise (n=3, **SUPERSEDED by §8.8**) | `python scripts/phase5_bandenergy_record_level.py` | runs `ce344d1` | `results/phase5_bandenergy/findings_bandenergy.md` (banner-marked superseded) | as above |
| §8.6a XAI / §8.6b calibration (negative) | `python scripts/run_xai_calibration.py` | — | `results/xai_alignment/alignment.json`, `results/uncertainty/calibration.json` | (eval of the §8.4 checkpoints) |
| **C5 deployment** | (ONNX export appendix) | — | `results/deployment/appendix.md` | — |

> The §8.8 row is the load-bearing one. `metrics.json` for those 48 runs carry the
> old `ops_aware` field (= the **eval** flag; training used per-sample rpm — the
> runner now writes `eval_ops_aware` + `train_metadata_rpm_used`, see `928e689`).

## Checkpoint archive (Zenodo)
The `*.pth` checkpoints are gitignored (large, binary). For external reproducibility
they must be deposited on Zenodo:
- **DOI: `<TBD — deposit pending>`** (owner to upload).
- Contents: `results/p7_strengthen/**/best_model.pth` (48, ~2.1 GB), the Phase-5
  benchmark/band-energy `best_model.pth`, each beside its `metrics.json`.
- The record-level analysis scripts reconstruct every number from these + the dataset
  (hash above); a per-checkpoint `.npy` cache under
  `results/phase5_bandenergy/_record_cache/` is gitignored and regenerates on demand.

## Independent verification
The 2026-06-24 audit round reproduced the §8.8 numbers **cache-free, from the raw
checkpoints**, with its own code (not the maintainers' analysis or cache), all 48
window-accuracy sanity gates passing:
`scripts/audit_independent_recompute.py`, `scripts/audit_verify_random_control.py`
(`audit_reports/INDEPENDENT_AUDIT_2026-06-24_CLAUDE.md`). Suite: `pytest -q` → 268
passed, 6 deselected.
