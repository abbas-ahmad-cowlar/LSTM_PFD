# Reproducibility / provenance manifest

> One row per paper-facing result: the command, the code commit it was produced at,
> the frozen inputs (content-hashed), the committed output artifact, and where the
> (gitignored) checkpoints live. Closes the chain command → commit → dataset-hash →
> checkpoint → result for a cold referee (audits 2026-06-24, M1). **Authoritative
> verdict: `results/FINDINGS.md` §0 (complete negative).**

> **Path note (2026-06 repo tidy).** Result directories were renamed to
> content-keyed names (file contents — hence all SHA-256 below — are unchanged):
> `p7_strengthen → noise_seed_robustness` (and its JSON `p7_strengthen_record_level
> → noise_seed_robustness_record_level`), `phase5_bandenergy → band_energy_reruns`,
> `xai_alignment → interpretability`, `uncertainty → calibration`. Pre-remediation
> dirs (`phase5*`, `noise_robustness`, `cnn1d_v{1,2}_baseline`) were removed
> (in git history). The §8.8 runner command below records the literal Colab
> `--out-root results_p7_strengthen` used at the time; the downloaded checkpoints
> now live under `results/noise_seed_robustness/`. Dated audit reports under
> `audit_reports/` keep the pre-tidy paths as snapshots.
>
> **Commit-SHA note (2026-06-26).** The per-result **code-commit SHAs** in the table
> below (e.g. `e498deb`, `4a2063d`, `ce344d1`, `70f623f`, `928e689`) predate the
> 2026-06-26 git-history rewrite (Claude co-author-trailer strip + fabricated-paper
> purge), which reassigned every commit SHA — so they **no longer resolve**. Treat
> them as historical labels; the current code is on `main`@`cf92673` (`git log`). The
> frozen **input** SHA-256 hashes below are content hashes and are UNAFFECTED.

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
| **C3 §8.8 n=12 noise — the headline NEGATIVE** | runner `python scripts/run_phase5_gpu.py --out-root results_p7_strengthen --seeds 0 1 2 3 4 5 6 7 8 9 10 11 {--only pinn_ablation --weights 0.0 1.0 \| --control f9_scramble \| --control random_bands}` → analysis `python scripts/p7_strengthen_record_level.py` | `4a2063d` | `results/noise_seed_robustness/noise_seed_robustness_record_level.json` + 48 `metrics.json` | `results/noise_seed_robustness/**/best_model.pth` (48) |
| §8.2 data-eff / §8.3 OOD (record-level, negative) | `python scripts/phase5_bandenergy_record_level.py` | runs `ce344d1` | `results/band_energy_reruns/summary_record_level.json` | `results/band_energy_reruns/{data_efficiency,severity_ood}/**/best_model.pth` |
| §8.7 F9 scramble (n=3, **superseded by §8.8**) | `python scripts/f9_scramble_record_level.py` | runs `70f623f` | `results/band_energy_reruns/f9_scramble_record_level.json` | `results/band_energy_reruns/pinn_ablation_scramble/**/best_model.pth` |
| §8.4 band-energy noise (n=3, **SUPERSEDED by §8.8**) | `python scripts/phase5_bandenergy_record_level.py` | runs `ce344d1` | `results/band_energy_reruns/findings_bandenergy.md` (banner-marked superseded) | as above |
| §8.6a XAI / §8.6b calibration (negative) | `python scripts/run_xai_calibration.py` | — | `results/interpretability/alignment.json`, `results/calibration/calibration.json` | (eval of the §8.4 checkpoints) |
| **C5 deployment** | (ONNX export appendix) | — | `results/deployment/appendix.md` | — |

> The §8.8 row is the load-bearing one. `metrics.json` for those 48 runs carry the
> old `ops_aware` field (= the **eval** flag; training used per-sample rpm — the
> runner now writes `eval_ops_aware` + `train_metadata_rpm_used`, see `928e689`).

## Checkpoint archive (Zenodo)
The `*.pth` checkpoints are gitignored (large, binary). For external reproducibility
they must be deposited on Zenodo:
- **DOI: `<TBD — deposit pending>`** (owner to upload).
- Contents: `results/noise_seed_robustness/**/best_model.pth` (48, ~2.1 GB), the Phase-5
  benchmark/band-energy `best_model.pth`, each beside its `metrics.json`.
- The record-level analysis scripts reconstruct every number from these + the dataset
  (hash above); a per-checkpoint `.npy` cache under
  `results/band_energy_reruns/_record_cache/` is gitignored and regenerates on demand.

## Independent verification
The 2026-06-24 audit round reproduced the §8.8 numbers **cache-free, from the raw
checkpoints**, with its own code (not the maintainers' analysis or cache), all 48
window-accuracy sanity gates passing:
`scripts/audit_independent_recompute.py`, `scripts/audit_verify_random_control.py`
(`audit_reports/INDEPENDENT_AUDIT_2026-06-24_CLAUDE.md`). Suite: `pytest -q` → 268
passed, 6 deselected.
