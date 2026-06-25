# results/ — index (what each folder is)

Every committed folder holds small evidence files (json/md/png). Model
checkpoints (`*.pth`, ~45 MB each) are NOT in git; for external reproducibility
they will be deposited on Zenodo (see `PROVENANCE_MANIFEST.md`). Folder names are
**content-keyed** — renamed from internal phase/process names in a 2026-06 repo
tidy, and pre-remediation / superseded dirs were removed (recoverable from git
history).

| Folder | Contribution | What it is |
|---|---|---|
| `FINDINGS.md` | — | ratified synthesis + frozen claims list (owner-ratified §0) |
| `PROVENANCE_MANIFEST.md` | — | reproduce chain: command → commit → dataset-hash → checkpoint → result |
| `dataset_v2_validation/` | C1 | `dataset_v2.h5` validation (stratification, leakage, splits) |
| `benchmark/` | C2 | frozen 11-model pre-registered benchmark; `summary_record_level.*` = the record-level table cited in the paper |
| `noise_seed_robustness/` | C3 §8.8 | **the headline n=12 grid** (4 arms × 12 seeds; `noise_seed_robustness_record_level.json`) + 48 checkpoints |
| `band_energy_reruns/` | C3 §8.2/§8.3 | band-energy reruns: data-efficiency (§8.2) + severity-OOD (§8.3), record-level. Also holds the superseded n=3 noise ablation (§8.4) + §8.7 F9 scramble, banner-marked in-file |
| `interpretability/` | C3 §8.6a | IG attribution-vs-physics-band alignment + `findings_8_6.md` |
| `calibration/` | C3 §8.6b | MC-dropout calibration (ECE, reliability, reject curve) |
| `deployment/` | C5 | ONNX export / latency / INT8 appendix |

**Headline (ratified — see `FINDINGS.md` §0):** a synthetic classification
benchmark, **near-ceiling at the record level**, **no row showing a physics
accuracy advantage**, and a **rigorous, complete NEGATIVE** on physics-informed
learning. The last candidate positive — a 5 dB noise-robustness benefit that
looked significant at **n=3** — **did not replicate** in the pre-registered
**n=12** grid (§8.8, `noise_seed_robustness/`): correct physics ties CE-only
(degr 3.47 vs 3.54, seed-level **Wilcoxon p=0.79**; the random non-fault-band arm
has the lowest mean degradation numerically yet still fails the bar; no arm robust
≥10/12). The n=3 McNemar 14–0 (p=1.2e-4) was a **seed artifact** (the confounded
within-/representative-seed statistic; the seed-level Wilcoxon governs). Clean
accuracy, data-efficiency, severity-OOD, XAI alignment (§8.6a *reverses*), and
calibration (a wash) also did **not** survive. No "physics helps" claim of any
kind.

**Removed in the 2026-06 repo tidy** (pre-remediation / superseded; recoverable
from git history): `phase5/`, `phase5_fixed/`, `phase5_dataeff_fixed/`,
`noise_robustness/`, `cnn1d_v1_baseline/`, `cnn1d_v2_baseline/`. Pre-remediation
checkpoint archives remain off-repo at `D:\Libraries\`.
