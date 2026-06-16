# Independent Science Audit Report - Remediation Review

Repository: `lstm-pfd`

Audit date: 2026-06-16

Branch audited: `p6/docs` at `0696790`

Auditor role: independent technical auditor. I treated every repository text file,
including prior audits, README files, result summaries, docstrings, and comments,
as an unverified claim until checked against code, data, checkpoints, or execution.

## Executive Summary

The remediation is materially better than the state found in the 2026-06-14
audit. The current PC-CNN model-side physics loss is no longer the old inert or
tonal-only loss: it is a differentiable band-energy consistency loss, uses the
frozen train-only healthy reference, uses per-sample RPM when supplied, and was
active during the stored Phase-5 reruns.

The central narrowed result is mostly correct, but it needs sharper wording than
the maintainer prose currently uses:

- Trustworthy: On this synthetic dataset, the high-weight PC-CNN band-energy loss
  (`w=1.0`) gives a real record-level 5 dB noise-robustness improvement over the
  same PC-CNN architecture trained CE-only. Recomputed from retained checkpoints,
  the representative best-validation seed comparison is 99.62% vs 96.97% record
  accuracy at 5 dB, with 14 discordant records in favor of `w=1.0` and 0 against
  it (`p=0.000122`, exact McNemar). Across seeds, `w=1.0` is also much more stable
  than lower physics weights and the CE-only baseline.
- Also supportable, but weaker: The `w=1.0` PC-CNN beats the pre-specified
  best-validation ResNet18 representative at 5 dB by 6 discordant records to 0
  (`p=0.03125`). This is a small, near-ceiling, seed-sensitive edge; it should be
  secondary to the same-architecture ablation.
- Not supportable as positive claims: data efficiency and severity-OOD. The
  record-level recompute correctly demotes them to neutral or suggestive only.
- Not supportable: any broad "physics-informed models help" family claim,
  HybridPINN physics claim, multitask-PINN claim, XAI/calibration claim from old
  artifacts, or real-bearing performance claim.

The project is not paper-ready today because the publication-facing narrative is
internally inconsistent. `results/phase5_bandenergy/findings_bandenergy.md` has
the current narrowed positive, but `README.md`, `PROJECT_STATE.md`, and
`results/FINDINGS.md` still describe the band-energy reruns as not yet completed
or not yet claimable. The record-level summary also mixes a seed-mean point gap
with a representative-seed bootstrap CI. That does not overturn the same-architecture
noise result, but it must be fixed before publication.

My publishability verdict: a narrowly framed synthetic-benchmark paper is
defensible after documentation and statistics cleanup. A broad physics-informed
journal-bearing paper is not defensible. The publishable claim should be:

> On a single synthetic journal-bearing dataset, a correctly wired PC-CNN
> band-energy consistency loss at high weight improved 5 dB noise robustness in
> an architecture-matched ablation, with no clean-accuracy gain and a small
> near-ceiling clean tradeoff. Data-efficiency and severity-OOD advantages were
> not statistically supported at record level.

## Commands Executed

All commands were run in PowerShell from `C:\Users\COWLAR\projects\lstm-pfd`.
`PYTHONIOENCODING=utf-8` was set for Python commands that print Unicode.

```powershell
git status --short --branch
git log --oneline --decorate -n 20
$env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe -m pytest -q
$env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe scripts\verify_physics_consistency.py
$env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe scripts\audit_physics_penalties.py
$env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe scripts\phase5_bandenergy_record_level.py
```

I also ran independent Python probes against `data/generated/dataset_v2.h5`,
`packages/core/models/physics/healthy_reference.json`, result JSON files, cached
per-window probability arrays, and retained `best_model.pth` checkpoints.

Key observed outputs:

```text
git:
## p6/docs...origin/p6/docs
0696790 (HEAD -> p6/docs, origin/p6/docs) P6 Step 5a: record-level confirmation...

pytest:
261 passed, 6 deselected, 49 warnings in 82.59s
TOTAL coverage: 15.13%

phase5 record-level recompute:
test records: 528
w=0.0: clean 98.99+/-0.45 | 5dB 94.70+/-3.08
w=0.1: clean 99.12+/-0.09 | 5dB 94.13+/-2.96
w=0.3: clean 99.68+/-0.09 | 5dB 93.75+/-3.95
w=1.0: clean 98.61+/-0.62 | 5dB 98.55+/-0.76
resnet18 5dB 97.66+/-1.30
data-efficiency and severity-OOD tables reproduced as in summary_record_level.json
```

## Finding 1 - Dataset v2 and split integrity hold up

Severity: minor positive finding

Evidence:

- `scripts/generate_dataset_v2.py:58-75` implements record-level stratified
  splitting by `(class, severity)`.
- `scripts/generate_dataset_v2.py:136-138` checks duplicate record hashes before
  writing.
- `scripts/generate_dataset_v2.py:145-170` writes split-local labels,
  severities, metadata, and record hashes for clean splits.
- `data/dataset.py:609-651` implements the non-overlapping five-window view and
  exposes `record_index`.

Independent HDF5 inspection of `data/generated/dataset_v2.h5` observed:

```text
dataset sha256 prefix: f72ad35b733c5649
attrs: dataset_version=v2, rng_seed=42, fs=20480, signal_length=102400
train: 2464 records, 224/class, 56/class/severity
val:    528 records,  48/class, 12/class/severity
test:   528 records,  48/class, 12/class/severity
hash overlaps: train-val 0, train-test 0, val-test 0
test_snr20/10/5 labels match test order: True
metadata ranges: train/val/test rpm approximately 3240-3959, load 30-100%, temp 40-80 C
```

Conclusion:

Dataset v2 is internally coherent as a synthetic, group-split benchmark. I found
no evidence of train/validation/test duplicate leakage.

Caveat:

The `test_snr*` groups preserve label order but do not store their own metadata
or record hashes (`scripts/generate_dataset_v2.py:164-170`). That is acceptable
for the current inference-only noise evaluations, but a reproduction manifest
should state that SNR groups inherit clean-test ordering.

## Finding 2 - The frozen healthy reference is train-only, reproducible, and actually frozen

Severity: minor positive finding

Evidence:

- `scripts/compute_healthy_reference.py:64-75` opens only `f['train']`, selects
  the healthy class, and iterates healthy records.
- `scripts/compute_healthy_reference.py:80-94` computes per-window PSD fractions
  with tonal bands placed at each healthy record's RPM.
- `scripts/compute_healthy_reference.py:103-126` writes the JSON artifact.
- `packages/core/models/physics/healthy_reference.json` provenance reports
  `split: train`, `healthy_class: sain`, and `n_healthy_windows: 1120`.

Independent recomputation from `dataset_v2.h5` produced:

```text
healthy_ref nwin: 1120
max_abs_diff vs packages/core/models/physics/healthy_reference.json: 0
```

Conclusion:

The healthy reference is correctly computed from train healthy windows only. I
found no validation/test leakage into the loss baseline.

## Finding 3 - The current PC-CNN physics loss is correctly implemented and active

Severity: major positive finding

Evidence:

- `packages/core/models/pinn/physics_constrained_cnn.py:180-186` refuses to run
  if the healthy reference is absent.
- `packages/core/models/pinn/physics_constrained_cnn.py:194-204` computes
  differentiable softmax probabilities and per-sample RPM.
- `packages/core/models/pinn/physics_constrained_cnn.py:206-239` computes
  band-energy penalties for tonal and absolute-Hz bands against the healthy
  reference.
- `packages/core/models/pinn/physics_constrained_cnn.py:241-248` returns
  `(probs * pen).sum(dim=1).mean()`, so gradient flows through logits.
- `tests/test_physics_band_energy_loss.py:64-72` asserts gradient to logits.
- `tests/test_physics_band_energy_loss.py:100-107` asserts per-sample RPM shifts
  tonal bands.
- `scripts/run_phase5_gpu.py:145-155` unpacks per-record RPM/load/viscosity.
- `scripts/run_phase5_gpu.py:178-197` adds `physics_w * phys` when the model has
  `compute_physics_loss`.
- `scripts/run_phase5_gpu.py:343-403` builds the Phase-5 PC-CNN train loaders
  with `ops=True`, so RPM metadata flows during physics-loss training.

My direct gradient probe on a real train batch:

```text
phys_true 0.08993591368198395
requires_grad True
grad_fn MeanBackward0
phys_default 0.15128779411315918
abs_diff true-rpm vs default-rpm 0.06135188043117523
rpm [3924.51, 3678.53, 3315.45, 3635.67]
param_grad_l1_sum 904.8771517574787
```

Stored run histories also show the loss was active:

```text
pinn_ablation w=1.0 seed0/1/2: physics_loss nonzero in 52/60/60 epochs
pinn_ablation w=0.1 and w=0.3: physics_loss nonzero in every completed training epoch
resnet18 controls: physics_loss 0.0
```

Conclusion:

The earlier inert-loss failure is fixed for the PC-CNN model-method path used by
the new `results/phase5_bandenergy` reruns. The corrected loss was genuinely
engaged during training.

## Finding 4 - The same-architecture high-weight noise result is real in the artifacts

Severity: major positive finding

Evidence:

`scripts/phase5_bandenergy_record_level.py` recomputes from retained checkpoints
and cached probabilities:

- `scripts/phase5_bandenergy_record_level.py:55-96` builds per-record soft-vote
  probabilities and sanity-checks window-level accuracy against each
  `metrics.json`.
- `scripts/phase5_bandenergy_record_level.py:139-185` recomputes the `w` sweep
  and the ResNet18 reference.
- `scripts/phase5_bandenergy_record_level.py:178-183` writes McNemar p-values
  and bootstrap CIs.

Independent recomputation from the cached probability arrays:

```text
5 dB record accuracy, PC-CNN CE-only w=0:
seed0 96.78% (17 errors)
seed1 90.34% (51 errors)
seed2 96.97% (16 errors)
mean 94.70%, std 3.08
best-validation seed: seed2

5 dB record accuracy, PC-CNN band-energy w=1.0:
seed0 97.92% (11 errors)
seed1 98.11% (10 errors)
seed2 99.62% (2 errors)
mean 98.55%, std 0.76
best-validation seed: seed2

Representative exact McNemar, w=1.0 seed2 vs CE-only seed2:
w1 correct / w0 wrong: 14 records
w0 correct / w1 wrong: 0 records
p = 0.0001220703125
representative record gap: +2.65 points
seed-mean record gap: +3.85 points
```

Class-level error audit:

```text
Best-val CE-only w=0 seed2 errors at 5 dB:
lubrification: 14 misclassified as mixed_wear_lube
mixed_wear_lube: 2 misclassified as sain

Best-val w=1.0 seed2 errors at 5 dB:
mixed_wear_lube: 2 misclassified as sain

All 14 records rescued by w=1.0 were lubrification records that CE-only
misclassified as mixed_wear_lube.
```

This pattern is physically plausible for the implemented loss. On real
`lubrification` test windows, the mean penalty was:

```text
true lubrification signal:
sain penalty               0.000
lubrification penalty      0.000
mixed_wear_lube penalty    0.569
desequilibre penalty       0.911
```

Conclusion:

The same-architecture high-weight noise-robustness improvement is not a prose
artifact. It reproduces from retained checkpoints, passes the script's sanity
gate, and has a plausible class-level mechanism.

## Finding 5 - The vs-ResNet claim is statistically real under the pre-specified seed rule, but fragile

Severity: major limitation

Evidence:

The record-level script selects representative models by best validation
accuracy (`scripts/phase5_bandenergy_record_level.py:119-125`). Under that rule:

```text
w=1.0 PC-CNN seed2: 99.62% at 5 dB, 2 errors
ResNet18 seed2:     98.48% at 5 dB, 8 errors
w1 correct / resnet wrong: 6
resnet correct / w1 wrong: 0
p = 0.03125
```

But the edge is near-ceiling and seed-sensitive:

```text
ResNet18 seed1 has 98.67% at 5 dB, better than ResNet18 seed2 on the noise test.
w1 seed2 vs ResNet18 seed1: gap +0.95 points, 5 vs 0 discordants, p = 0.0625.
```

The best-validation seed rule is acceptable if pre-specified, and it does not
use the noise test for selection. Still, the small discordant count means this
should not be sold as a broad model-family result.

Conclusion:

The strongest publishable result is the PC-CNN w=1.0 vs PC-CNN CE-only
architecture-matched ablation. The ResNet comparison can be reported as a
secondary, small, near-ceiling comparison with explicit seed-count caveats.

## Finding 6 - The record-level summary mixes estimators for gaps and CIs

Severity: major

Evidence:

`scripts/phase5_bandenergy_record_level.py:178-183` stores:

- point gap: difference of seed-mean accuracies, e.g. `mean(w1 seeds) - mean(w0
  seeds) = 98.55 - 94.70 = +3.85 points`;
- CI: bootstrap of the best-validation representative correctness vectors, not
  the same seed-mean estimator.

The same mismatch appears in the text claim in
`results/phase5_bandenergy/findings_bandenergy.md`, which reports a seed-mean
gap with a representative-seed CI.

Concrete example:

```text
w=1.0 vs CE-only at 5 dB:
seed-mean gap:          +3.85 points
representative-seed gap:+2.65 points
reported CI:            [1.33, 4.17] from representative vectors

w=1.0 vs ResNet18 at 5 dB:
seed-mean gap:          +0.88 points
representative-seed gap:+1.14 points
reported CI:            [0.38, 2.08] from representative vectors
```

Impact:

This does not invalidate the McNemar p-values or the positive same-architecture
result, but the reporting is statistically sloppy. A skeptical reviewer will
notice that the point estimate and CI are not for the same estimand.

Recommendation:

Report either:

1. representative-seed gap plus representative-seed CI and McNemar, or
2. seed-mean gap plus a cluster bootstrap/seed bootstrap that matches the
   seed-mean estimator.

Do not mix them in the same sentence.

## Finding 7 - Data-efficiency and severity-OOD are correctly demoted at record level

Severity: minor positive finding

Evidence:

Rerunning `scripts/phase5_bandenergy_record_level.py` reproduced:

```text
Data efficiency, record-level clean accuracy:
PC-CNN w0.3: 10% 96.28+/-1.39, 25% 98.48+/-0.15, 50% 98.80+/-0.79, 100% 99.49+/-0.32
ResNet18:    10% 96.28+/-1.10, 25% 97.22+/-0.62, 50% 98.80+/-0.24, 100% 99.18+/-0.09

Severity-OOD, record-level:
A train low test severe: PC-CNN 100.00+/-0.00, ResNet18 100.00+/-0.00
B train high test incipient: PC-CNN 82.58+/-0.62, ResNet18 76.01+/-3.41
direction B McNemar p = 0.3877, bootstrap CI spans zero
```

Conclusion:

The maintainers' narrowed interpretation is correct here. Data efficiency is not
a win; severity-OOD direction B is suggestive but not statistically supported at
record level.

## Finding 8 - The physics loss is useful but not a complete physics classifier

Severity: major limitation

Evidence:

Full test-set penalty audit against the frozen healthy reference:

```text
class                       own_mean   healthy_mean   margin
desalignement                 0.000       0.263       +0.263
desequilibre                  0.000       0.615       +0.615
jeu                           0.000       0.401       +0.401
lubrification                 0.000       0.343       +0.343
cavitation                    0.130       0.252       +0.122
usure                         0.000       0.438       +0.438
oilwhirl                      0.000       0.329       +0.329
mixed_misalign_imbalance      0.000       0.380       +0.380
mixed_wear_lube               0.014       0.406       +0.392
mixed_cavit_jeu               0.203       0.399       +0.196
```

The loss premise holds on average for all non-healthy classes, but cavitation and
mixed cavitation/clearance are weakly separated from healthy. The repository's
own physics consistency script also reports no narrow expected frequencies for
`lubrification` and `cavitation` and a 15/16 tonal expected-frequency match rate.

More importantly, `sain` has no bands, so `compute_physics_loss` assigns the
healthy class zero penalty by construction (`physics_constrained_cnn.py:215-239`
leaves `pen[:, sain] = 0`). For faulty samples, the physics term cannot penalize
assigning probability to `sain`; only cross-entropy can. That limitation is
visible in the remaining 5 dB errors:

```text
Best-val w=1.0 seed2 still misclassifies 2 mixed_wear_lube records as sain.
For mixed_wear_lube signals:
sain penalty             0.000
lubrification penalty    0.011
mixed_wear_lube penalty  0.014
usure penalty            0.016
```

Conclusion:

The loss contains real physics content, especially for suppressing
spectrally-inconsistent mixed-class predictions on pure lubrication records. It
does not prove that physics priors generally help, and it is not a complete
class-disambiguating physics model. It should be described as a band-consistency
regularizer, not as a mechanistic diagnostic solver.

## Finding 9 - Attribution to "physics" is plausible but not isolated from regularization confounds

Severity: major limitation

Evidence:

The same architecture and same protocol are used for the key ablation:

- `scripts/run_benchmark.py:87-114` trains the Phase-4 PC-CNN CE-only baseline.
- `scripts/run_phase5_gpu.py:178-197` adds the physics term for the rerun.
- `git diff e498deb0..ce344d1` over architecture/training files showed no
  ResNet backbone or benchmark-protocol changes; the meaningful model change is
  `PhysicsConstrainedCNN.compute_physics_loss` and the new Phase-5 runner.

However, the experiment does not include controls for:

- generic entropy/logit regularization at comparable strength;
- non-physics spectral-band regularization;
- randomized or permuted band references;
- a healthy-reference-only regularizer with wrong class mapping;
- more than three training seeds.

Conclusion:

The class-level rescued errors make a physics-content explanation plausible, but
the experiment does not fully isolate physics from "extra regularization at high
weight." The paper should say the result is attributable to the implemented
band-energy consistency term, not that it proves the general value of physics
priors.

## Finding 10 - Quarantined paths are excluded from the new positive claim, but not hard-blocked

Severity: major for future reproducibility, minor for the current noise claim

Evidence:

- `packages/core/training/physics_loss_functions.py:25-40` documents
  `FrequencyConsistencyLoss` as non-differentiable and quarantined.
- `packages/core/training/physics_loss_functions.py:90-91` still uses
  `torch.argmax(predictions)`.
- `packages/core/training/physics_loss_functions.py:284-293` documents
  `PhysicalConstraintLoss` as inert.
- `packages/core/training/pinn_trainer.py:49-55` warns that the trainer's physics
  path is inert, but `pinn_trainer.py:237-245` will still compute and add that
  inert physics loss if used.
- `packages/core/models/pinn/hybrid_pinn.py:40-49` clearly quarantines the
  rolling-element physics branch.
- `packages/core/models/pinn/hybrid_pinn.py:244-256` still computes rolling-element
  FTF/BPFO/BPFI/BSF/shaft-frequency features.
- `tests/test_physics_quarantine.py:33-52` pins the old generic loss as inert.
- `results/phase5_bandenergy/findings_bandenergy.md:151-152` excludes Sec. 8.5
  HybridPINN from the current claim.

Conclusion:

The current Phase-5 band-energy claim does not silently use the broken generic
loss or HybridPINN. That is good. But the quarantine is mostly by warning and
documentation. A future user can still instantiate `PINNTrainer` with
`lambda_physics > 0` and believe it is doing physics. That should raise a
runtime error until replaced.

## Finding 11 - The benchmark recomputes correctly at record level and no clean physics advantage exists

Severity: minor positive finding

Evidence:

`scripts/aggregate_benchmark_record_level.py` is methodologically improved:

- `aggregate_benchmark_record_level.py:35-46` explicitly states the benchmark is
  classification-only and not a physics test.
- `aggregate_benchmark_record_level.py:221-222` soft-votes the five windows per
  record.
- `aggregate_benchmark_record_level.py:225-230` separates seed-mean headline
  accuracy from best-validation representative paired tests.

Independent recomputation from `results/benchmark/record_level/_cache/*.npy`
matched `results/benchmark/summary_record_level.json` exactly:

```text
random_forest            98.74+/-0.09
cnn_lstm                 99.43+/-0.15
resnet18                 99.18+/-0.09
physics_constrained_cnn  98.99+/-0.45 (CE-only architecture row)
hybrid_pinn              90.34+/-0.94 (rolling-element branch + constant metadata)
multitask_pinn           90.47+/-0.62 (single-task)
max mean/std diff vs summary: 0
```

Conclusion:

The benchmark backbone is reproducible from artifacts and supports a
near-ceiling synthetic classification result. It does not show a clean-accuracy
benefit from physics-informed training.

## Finding 12 - Top-level repo documentation is stale and internally inconsistent

Severity: major

Evidence:

- `README.md:44-52` still says no physics-informed benefit is supportable and
  that the ratified band-energy experiment has not been run.
- `PROJECT_STATE.md:17-27` and `PROJECT_STATE.md:300-306` say the record-level
  recompute is still running and the noise/OOD signal is not yet a claim.
- `results/FINDINGS.md:34-45` says the stored artifacts show no physics advantage
  and that no physics benefit may be claimed yet.
- `results/FINDINGS.md:62-65` says the band-energy rerun is still next.
- In contrast, `results/phase5_bandenergy/findings_bandenergy.md:75-146` and
  `results/phase5_bandenergy/summary_record_level.json` contain the completed
  record-level rerun verdict.

Conclusion:

This is not evidence that the rerun is fake; the artifacts and recomputation
check out. But it is a serious publication-readiness problem. The repo currently
has two live narratives: "no benefit claim yet" and "noise benefit survives."
A paper draft must not proceed until the root README, `PROJECT_STATE.md`, and
`results/FINDINGS.md` are rewritten and reconciled with the record-level evidence.

## Finding 13 - Test suite is green but does not by itself prove scientific validity

Severity: minor

Evidence:

```text
pytest: 261 passed, 6 deselected
coverage: 15.13%
```

The suite now includes important targeted tests:

- band-energy loss gradient/RPM tests;
- signature DB consistency tests;
- quarantine tests for inert paths.

But many experiment and analysis paths are still low coverage or artifact-driven.
For a publication repository, the current CI is a useful guardrail, not a
complete reproducibility proof.

Recommendation:

Add non-smoke tests that assert the record-level summary uses consistent
estimands, that `PINNTrainer(lambda_physics>0)` hard-fails until fixed, and that
all paper-facing claims have a machine-readable manifest.

## Finding 14 - Synthetic data limitations are mostly honestly stated

Severity: minor positive finding

Evidence:

- `docs/PHYSICS.md:11-15` states the simulator has physically structured
  signals but no real-rig validation.
- `docs/PHYSICS.md:60-79` documents operating-condition sampling.
- `docs/PHYSICS.md:80-167` documents class signatures.
- `docs/PHYSICS.md:169-180` states optional advanced physics effects are off for
  v2.
- `docs/PHYSICS.md:182-191` lists limitations, including no real-data validation.
- `config/data_config.py:245-281` sets advanced physics toggles to `False` by
  default.

Conclusion:

The synthetic-only limitation is clear in the physics documentation. The paper
should keep it front and center. The result is about a controlled synthetic
generator, not field diagnosis.

## Independent Verdict on Current Claims

Claim: "Correctly implemented physics yields a statistically significant
noise-robustness benefit at high physics weight."

Verdict: Supported, if narrowed to the PC-CNN band-energy loss on this synthetic
dataset, especially as a same-architecture ablation. The representative
record-level McNemar result against CE-only is strong (14 to 0 discordants,
`p=0.000122`), the seed means are consistently better at `w=1.0`, and the rescued
errors are physically interpretable.

Required wording limits:

- Say "PC-CNN band-energy consistency loss at `w=1.0` improved 5 dB robustness"
  rather than "physics-informed learning improves robustness."
- Make the same-architecture ablation the headline.
- Describe the ResNet edge as secondary and near-ceiling.
- State the small clean tradeoff: record-level clean accuracy at `w=1.0` is
  98.61+/-0.62 vs 98.99+/-0.45 for CE-only and 99.68+/-0.09 for `w=0.3`.
- State `n=3` training seeds and 528 test records.
- State that no real-data validation exists.
- State that data-efficiency and severity-OOD did not survive record-level
  significance.

Claim: "Severity-OOD and data-efficiency do not survive significance."

Verdict: Supported. The record-level method correctly demotes these results.

Claim: "The remediation is sound and sufficient."

Verdict: Partly. The key PC-CNN loss remediation is sound. The total remediation
is not complete because documentation is stale, the generic trainer path remains
runtime-usable despite being inert, HybridPINN remains physically wrong, XAI is
not recomputed, and statistical reporting needs estimator cleanup.

Publishability:

Not publishable as-is. Publishable after:

1. statistics reporting is corrected;
2. root docs and `results/FINDINGS.md` are rewritten;
3. invalid old XAI/HybridPINN/generic-PINN claims remain excluded;
4. limitations and controls are stated honestly.

## Prioritized Recommendations

1. Fix `scripts/phase5_bandenergy_record_level.py` and
   `summary_record_level.json` to align point estimates and CIs to the same
   estimand. Report representative gaps with representative CIs, and seed-mean
   gaps with seed-mean CIs.

2. Rewrite `README.md`, `PROJECT_STATE.md`, and `results/FINDINGS.md` now. The
   current top-level narrative is stale and conflicts with the newest artifact.

3. Use the following claim boundary in the paper:
   "High-weight PC-CNN band-energy consistency improved 5 dB noise robustness in
   an architecture-matched ablation on synthetic data; no clean-accuracy,
   data-efficiency, or statistically significant severity-OOD gain was shown."

4. Do not claim XAI/calibration benefits until Sec. 8.6 is recomputed against
   the corrected bands and checkpoints.

5. Hard-block quarantined paths. `PINNTrainer` with a positive physics weight
   should raise `RuntimeError` until it uses the validated band-energy loss.
   HybridPINN should either be rebuilt on journal-bearing features or excluded
   from physics sections.

6. Add controls before making a stronger causal statement: entropy/logit
   regularization, random-band regularization, wrong-band regularization, and
   permuted healthy-reference controls.

7. Increase seeds if compute allows. With `n=3`, seed-level inference is weak;
   record-level McNemar is useful but does not characterize training-seed
   variability well.

8. Preserve artifact provenance. Every table in the paper should map to one
   command, one commit SHA, one dataset hash, one result JSON, and one retained
   checkpoint or score cache.

9. Add metadata/hashes to future SNR groups or document their inherited ordering
   explicitly in the reproduction manifest.

10. Keep the synthetic-only limitation in every abstract/conclusion version. The
    current result is defensible as controlled synthetic evidence, not as a
    field-bearing diagnostic claim.

