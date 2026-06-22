# Independent Technical Audit - 2026-06-22 - GPT-5

Repository: `C:\Users\COWLAR\projects\lstm-pfd`  
Branch audited: `p6/docs`  
HEAD observed: `b9d748c P6: remove prior audit reports to keep a fresh pair of independent audits uncontaminated`  
Auditor constraint followed: I did not read prior audit reports. I read the current claim docs, current result artifacts, the loss implementation, the two record-level scripts, and selected small side-result files. I did not bulk-read the repo, data files, or checkpoints.

## Executive Summary

This repository is not just polishing a pure null into a fake positive, but it is also not sitting on a strong "physics-informed learning works" result. The defensible result is much narrower:

> On this one synthetic dataset, for the retained checkpoints, a high-weight band-energy spectral-consistency term improves 5 dB record-level robustness in a same-architecture PC-CNN ablation. A wrong-physics scrambled-reference control reproduces that robustness in the representative seed and in 2 of 3 seeds, so the result should not be sold as evidence that correct journal-bearing physics priors caused the gain.

The headline record-level result is real in the artifacts I recomputed:

| Arm | Clean record acc | 5 dB record acc | Clean to 5 dB degradation |
|---|---:|---:|---:|
| CE-only `physics_constrained_cnn` w=0 | 98.99 +/- 0.45 | 94.70 +/- 3.08 | 4.29 |
| Correct band-energy w=1.0 | 98.61 +/- 0.62 | 98.55 +/- 0.76 | 0.06 |
| Scrambled-reference band-energy w=1.0 | 98.42 +/- 0.91 | 95.58 +/- 5.31 | 2.84 |

Representative best-validation seed, record-level 5 dB:

- Correct w=1.0 vs CE-only: gap +2.65 points, bootstrap CI [1.33, 4.17], exact McNemar 14-0, p=0.0001220703125.
- Scramble vs CE-only: gap +2.46 points, exact McNemar 14-1, p=0.0009765625.
- Scramble vs correct w=1.0: gap -0.19 points, exact McNemar 0-1, p=1.0.

The 14 records rescued by correct w=1.0 over CE-only are all `lubrification`; CE-only predicts all 14 as `mixed_wear_lube`, while w=1.0 predicts all 14 as `lubrification`. That is coherent with the claimed mechanism, but the F9 scramble shows the mechanism is not established as correct physics.

No other claimed advantage survives my audit pass. Data efficiency is neutral by the preregistered rule. Severity-OOD is direction-only and non-significant. XAI reverses against the physics model even in the band-aware metric. Calibration is a small, stochastic, single-seed/window-level wash. The benchmark is useful as a classification benchmark, not as evidence of physics-informed superiority.

My publication verdict: conditionally publishable as a synthetic dataset/benchmark plus a narrow spectral-regularization robustness finding and a cautionary negative result. Not publishable as "physics-informed learning improves fault diagnosis" or anything close to that. I would stake my name only on the narrow regularizer claim, with the limitations and F9 control in the abstract-level story.

## Commands And Evidence Base

Commands run:

- `git branch --show-current` -> `p6/docs`.
- `git log --oneline -n 12` -> top commit `b9d748c`, followed by the F9/remediation commits.
- `$env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe -m pytest -q` -> `263 passed, 6 deselected, 49 warnings in 77.08s`.
- `$env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe scripts\phase5_bandenergy_record_level.py` -> reproduced `summary_record_level.json` numbers.
- `$env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe scripts\f9_scramble_record_level.py` -> reproduced `f9_scramble_record_level.json` numbers.
- Independent cache-only Python recomputation from `results/phase5_bandenergy/_record_cache/*.npy` and `data/generated/dataset_v2.h5`, without trusting the emitted JSON.
- Direct physics-loss checks: gradient flow to parameters, rpm sensitivity, and scramble-vs-correct loss/gradient difference.
- Direct healthy-reference recomputation from `data/generated/dataset_v2.h5` train split only.

Key files read:

- `results/FINDINGS.md` lines 18-117 for the current draft verdict and wording limits.
- `results/phase5_bandenergy/findings_bandenergy.md` lines 73-186 for record-level and F9 claims.
- `results/phase5_bandenergy/summary_record_level.json`.
- `results/phase5_bandenergy/f9_scramble_record_level.json`.
- `scripts/phase5_bandenergy_record_level.py`.
- `scripts/f9_scramble_record_level.py`.
- `packages/core/models/pinn/physics_constrained_cnn.py`.
- `scripts/run_phase5_gpu.py` relevant training/F9 sections.
- `packages/core/models/physics/healthy_reference.json`.
- `scripts/compute_healthy_reference.py`.
- `results/xai_alignment/alignment.json`, `results/xai_alignment/findings_8_6.md`, `results/uncertainty/calibration.json`.
- `results/benchmark/summary_record_level.md`.
- `tests/test_physics_quarantine.py`.

## Prioritized Findings

### Finding 1 - Major - The same-architecture 5 dB robustness result is real, but narrow and near-ceiling

The surviving w=1.0 noise result reproduces from both the official script and my independent cache-only calculation.

Evidence:

- Record-level machinery: `scripts/phase5_bandenergy_record_level.py` uses split-specific cache keys at lines 55-63, sanity-gates cached window predictions against recorded metrics at lines 91-96, soft-votes records at line 95, computes exact McNemar at lines 103-117, and computes representative-seed bootstrap CIs at lines 187-190.
- Official recompute output:
  - w=0 clean 98.99 +/- 0.45, 5 dB 94.70 +/- 3.08.
  - w=1.0 clean 98.61 +/- 0.62, 5 dB 98.55 +/- 0.76.
  - w=1.0 vs w=0 representative seed: 14-0 McNemar, p=0.0001220703125.
- Independent cache-only recompute:
  - w=0 seed 5 dB record accuracies: [96.78, 90.34, 96.97].
  - w=1.0 seed 5 dB record accuracies: [97.92, 98.11, 99.62].
  - same-seed w1 vs w0 McNemar:
    - seed0: 10-4, p=0.1795654296875, gap +1.14 points.
    - seed1: 48-7, p=1.3087586125948292e-08, gap +7.77 points.
    - seed2: 14-0, p=0.0001220703125, gap +2.65 points.
  - The representative seed is seed2 for both w0 and w1, so the headline is not relying on the especially bad w0 seed1.
  - The 14 rescued seed2 records are all true `lubrification`; CE-only predicts all 14 as `mixed_wear_lube`; w=1.0 predicts all 14 correctly.

Interpretation:

- This is not a window-level artifact. Window aggregation to records actually sharpens the same-architecture noise result.
- It is also not just a tiny one-record fluke: 14 one-way discordants at 528 records is enough for exact McNemar at the record level.
- But it remains near-ceiling and conditional on retained trained checkpoints. It is not a strong seed-level theorem with n=3. The statistical unit for McNemar is records, not training runs.
- There is a clean-accuracy trade-off: w=1.0 clean record accuracy is 98.61, below CE-only 98.99 and below w=0.3 at 99.68. The draft correctly says no clean gain and should keep saying "small near-ceiling clean trade-off," not "free robustness."

### Finding 2 - Major - The F9 scrambled-reference control kills the physics-specific causal story

The scramble is implemented and active. It is a genuine wrong-reference control, not a no-op tag.

Evidence:

- `scripts/run_phase5_gpu.py` defines a seeded derangement at lines 321-333: class 0 healthy maps to itself; every fault class maps to a different class.
- The F9 queue sets that permutation for training at lines 381-396.
- `PhysicsConstrainedCNN.reference_permutation` is documented at lines 89-95 and applied inside `compute_physics_loss` at lines 223-229.
- The recorded permutation is `[0, 10, 5, 9, 6, 2, 8, 4, 7, 1, 3]`.
- Direct check on identical signals/logits:
  - no fault class is fixed except healthy.
  - correct-vs-scramble loss on the same logits/signals differed: 0.3606455 vs 0.3663030.
  - logit-gradient L1 norms differed, and the correct-vs-scramble gradient L1 difference was 0.2673551.
- Training histories also show the F9 loss was active:
  - correct w=1.0 physics loss mean about 0.033.
  - scramble w=1.0 physics loss mean about 0.485.

Record-level F9 evidence:

- `scripts/f9_scramble_record_level.py` compares CE-only, correct w=1.0, and scramble w=1.0 at lines 36-58 and computes representative-seed McNemar at lines 77-95.
- Official and independent recompute match:
  - Scramble 5 dB per seed: [88.07, 99.24, 99.43].
  - Scramble seed-mean degradation: 2.84 points.
  - Correct seed-mean degradation: 0.06 points.
  - CE-only seed-mean degradation: 4.29 points.
  - Representative seed2 scramble vs CE-only: 14-1, p=0.0009765625, gap +2.46 points.
  - Representative seed2 scramble vs correct: 0-1, p=1.0, gap -0.19 points.

Interpretation:

- The correct conclusion is not "physics caused the robustness." The wrong targets can produce the same representative-seed rescue of the same `lubrification` records.
- The most precise conclusion is: physics-specificity is not established; a high-weight spectral regularizer is sufficient to reproduce the representative-seed effect, while correct targets may improve cross-seed stability.
- The current draft mostly says this, but the word "reproduces" should be qualified whenever possible. Seed0 scramble collapses to 88.07 at 5 dB, so scramble does not reproduce the correct model's cross-seed stability. It reproduces the effect in the representative seed and in 2 of 3 seeds.
- Also, "same strength" means same lambda and same code structure, not same numerical penalty scale. The scrambled training loss is much larger than the correct-reference loss. That does not rescue the physics claim; it just means F9 is a strong confound against claiming the correct class references are the cause.

### Finding 3 - Major - The record-level statistics are mostly sound, but the inference is checkpoint-conditional, not run-distribution-level

The current JSON now keeps point estimate, CI, and McNemar aligned to the same representative-seed estimand. That fixes the earlier kind of estimator mismatch described in `summary_record_level.json` lines 3-9 and `scripts/phase5_bandenergy_record_level.py` lines 120-124.

Evidence:

- `summary_record_level.json` reports:
  - `repseed_gap_pts`: 2.65.
  - `repseed_gap_ci95`: [1.3257575757575757, 4.166666666666666].
  - `mcnemar_p`: 0.0001220703125.
  - `mcnemar_discordant_w1better_w0better`: [14, 0].
  - `seedmean_gap_pts`: 3.85, separately labeled.
- My independent bootstrap with the same seed and record vectors reproduced the same CI.

Limits:

- Exact McNemar is appropriate for paired record predictions from two fixed checkpoints. It does not test whether future training runs drawn from the same training procedure will reliably show the same effect.
- n=3 seeds is not enough to support a seed-level generalization claim. The paper must not imply that the p-value covers training randomness.
- The vs-ResNet result is secondary and fragile: w1 seed2 vs resnet seed2 is 6-0, p=0.03125, but w1 seed2 vs resnet's best-noise seed1 is 5-0, p=0.0625. The draft correctly demotes it.

### Finding 4 - Major - The physics loss is currently active, differentiable in the intended way, and uses per-sample rpm

The repaired `compute_physics_loss` is not the old inert argmax path.

Evidence:

- `packages/core/models/pinn/physics_constrained_cnn.py`:
  - loads the healthy reference at lines 84-87.
  - computes softmax probabilities at lines 202-203.
  - reads per-sample rpm metadata at lines 205-212.
  - computes spectral penalties under `torch.no_grad()` at lines 216-252.
  - returns `(probs * pen).sum(dim=1).mean()` at lines 253-260, so gradients flow through probabilities/logits.
- `scripts/run_phase5_gpu.py`:
  - unpacks rpm/load/viscosity metadata at lines 145-155.
  - adds `physics_w * phys` to CE during training at lines 191-198.
- Direct execution check:
  - `phys.requires_grad` was `True`.
  - parameter-gradient absolute sum from physics loss alone was `3619.6537917107344`.
  - loss changed with rpm metadata: default 3600 rpm 0.4065763, actual per-sample rpms 0.4267646, forced 1800 rpm 0.4866378.
  - training histories for w=0.1, w=0.3, w=1.0, and scramble contain nonzero physics-loss values.

Minor issue:

- The `metrics.json` field `ops_aware` is misleading for PC-CNN reruns. The w1 metrics have `ops_aware: false`, but the training loader is built with `ops=True`, and `run_epoch` still receives metadata and passes it into `compute_physics_loss`. In this script, `ops_aware` mainly controls evaluation metadata routing. This should be renamed or supplemented with a `train_ops_metadata: true` provenance field.

### Finding 5 - Major - The frozen healthy reference is train-only and exactly reproducible

The healthy reference does not leak validation/test signals into the loss baseline.

Evidence:

- `scripts/compute_healthy_reference.py` reads `f['train']` at lines 64-68.
- It selects healthy train records at line 74.
- It iterates 1-second windows from those records at lines 80-95.
- It writes the artifact at lines 103-127.
- `healthy_reference.json` provenance lines 3-15 says split `train`, healthy class `sain`, `n_healthy_windows` 1120, window 20480, n_fft 20480.
- My recomputation from `dataset_v2.h5`:
  - train records: 2464, healthy train records: 224.
  - 224 records * 5 windows = 1120 healthy windows.
  - val healthy records: 48; test healthy records: 48; not used.
  - max absolute difference between recomputed reference and `healthy_reference.json`: 0.

### Finding 6 - Major - Data-efficiency and severity-OOD are correctly set aside

The side results do not contain a buried robust positive.

Evidence from `summary_record_level.json` and official recompute:

- Data efficiency:
  - PC-CNN vs ResNet at 10 percent: 96.28 +/- 1.39 vs 96.28 +/- 1.10.
  - 25 percent: 98.48 +/- 0.15 vs 97.22 +/- 0.62.
  - 50 percent: 98.80 +/- 0.79 vs 98.80 +/- 0.24.
  - 100 percent: 99.49 +/- 0.32 vs 99.18 +/- 0.09.
  - Only one reduced fraction is a non-overlapping PC-CNN edge, so the preregistered data-efficiency rule fails.
- Severity-OOD:
  - Direction A severe: both 100.00 +/- 0.00.
  - Direction B incipient: PC-CNN 82.58 +/- 0.62 vs ResNet 76.01 +/- 3.41, but representative-seed gap CI [-2.27, +8.33] spans zero and McNemar p=0.3876953125.

Interpretation:

- Direction B may be worth future study, but it is not a paper claim.
- The draft's demotion is appropriate.

### Finding 7 - Major - XAI and calibration do not support a physics mechanism

The current XAI/calibration artifacts support the maintainers' decision to remove the C4 positive.

Evidence:

- `results/xai_alignment/alignment.json`:
  - tonal specificity: vanilla 1.041957, PC-CNN 0.856227.
  - band-aware in-band fraction: vanilla 0.145836, PC-CNN 0.099080.
  - `lubrification`, the class driving the noise rescue: vanilla 0.007192, PC-CNN 0.002388.
- `results/xai_alignment/findings_8_6.md` lines 45-67 correctly states that the band-aware rescue attempt fails and that both models put essentially zero attribution in the lube band.
- `results/uncertainty/calibration.json`:
  - vanilla clean ECE 0.023637; PC-CNN clean ECE 0.018180.
  - vanilla 5 dB ECE 0.026157; PC-CNN 5 dB ECE 0.022772.
  - But this is one representative seed, 220 windows, and MC-dropout is stochastic.
- `scripts/run_xai_calibration.py`:
  - uses seed 0 for the stratified window subset at lines 90-99.
  - turns dropout on for MC passes while leaving BatchNorm eval at lines 217-224.
  - does not set a fixed dropout-mask seed inside the MC loop at lines 246-253.

Interpretation:

- The XAI evidence actively argues against the "physics attention" story.
- Calibration is too small and too stochastic to claim.
- The draft is correct to say no interpretability or calibration advantage survives.

### Finding 8 - Major - The benchmark is useful, but only as a classification benchmark

The benchmark record-level correction and row relabeling are important and appear honest.

Evidence:

- `results/benchmark/summary_record_level.md` lines 5-16 explains that the original window-level unit was wrong, and that the recompute is a classification benchmark only.
- Lines 28-30 label:
  - `hybrid_pinn`: rolling-element branch + constant metadata.
  - `physics_constrained_cnn`: CE-only architecture; physics loss off.
  - `multitask_pinn`: single-task; auxiliary heads unused.
- Lines 34-48 show no physics-labeled clean benchmark advantage; best vanilla `cnn_lstm` and CE-only PC-CNN tie in best-val record accuracy, McNemar p=1.

Interpretation:

- This benchmark is a defensible dataset/classifier baseline.
- It must not be used as evidence that physics-informed models outperform vanilla models.

### Finding 9 - Minor - Quarantine of dead physics paths is now in place

The old inert generic physics-loss paths are hard-blocked rather than silently callable.

Evidence:

- `tests/test_physics_quarantine.py` asserts:
  - `FrequencyConsistencyLoss` raises, lines 33-37.
  - `PhysicalConstraintLoss` raises, lines 40-44.
  - `PINNTrainer(lambda_physics>0)` raises, lines 47-60.
  - stale `scripts/research/pinn_ablation.py` raises, lines 63-67.
- The full suite passed these tests.

Interpretation:

- This is the correct remediation posture. Do not remove these tests.

### Finding 10 - Major - Reproducibility is good enough for audit, but not yet paper-grade

The audit could reproduce the central numbers because the local machine has retained caches/checkpoints. A skeptical reviewer will need more.

Evidence:

- The CE-only arm is from benchmark commit `e498deb0ca74b8930c85e881ee7550e024981f98`; correct w1 is from `ce344d1fb73f6558939c9eeb51354e6e4363b7fd`; scramble is from `70f623ff82e1feaed5402e04caed724d194c053b`.
- Training budgets are effectively aligned: benchmark `run_benchmark.py` lines 43-46 and 124-127 use batch 64, max 60, patience 10, Adam lr 1e-3; phase5 runner uses the same constants at `scripts/run_phase5_gpu.py` lines 51-56 and Adam at line 254.
- But the exact paper table currently depends on off-git `best_model.pth` files and gitignored `.npy` probability caches.
- The result JSONs have useful provenance, but there is no single manifest tying every table row to command, dataset hash, checkpoint hash, cache hash, code SHA, and host.

Interpretation:

- The current repo is reproducible on this machine.
- It is not yet reviewer-grade reproducible unless the paper release includes a manifest and either publishes the retained checkpoints/caches or gives exact rerun instructions plus hashes.

## Big Picture: Answer To The Owner

Are we polishing a null result into a publication so we can publish any scrap?

Not exactly. There is one real result here. It is not large in absolute record count, but it is coherent, paired, record-level, reproduced from cache, and not an obvious aggregation or inert-loss artifact. The high-weight band-energy term really did make the retained PC-CNN more robust at 5 dB on this synthetic dataset.

But the result is much less glamorous than "physics-informed learning works." The scramble control is decisive enough to block that claim. If wrong per-class physics can rescue the same representative-seed records, the paper cannot claim correct journal-bearing physics caused the robustness. At best, correct physics may make the high-weight spectral regularizer more stable across seeds; with n=3, even that stability statement should be cautious.

What are we doing?

The honest project is now a synthetic benchmark paper plus a cautionary ML methodology paper:

- You built a controlled synthetic journal-bearing dataset and benchmark.
- You found that clean classification is near ceiling.
- You found that most hoped-for physics benefits do not survive record-level scrutiny.
- You found one narrow robustness effect from a band-energy regularizer.
- You then ran the right negative control and learned the effect is not physics-specific.

Is this the right direction?

Yes, if the goal is an honest modest publication. No, if the goal is to recover a flagship physics-informed-learning story. The strongest paper is not "physics wins"; it is "a careful synthetic benchmark shows how easy it is to overclaim physics-informed gains; after record-level statistics and a wrong-physics control, only a narrow spectral-regularization robustness effect remains."

Have we found anything real and defensible?

Yes:

- Dataset/generator as a synthetic benchmark, with stated limitations.
- Record-level classification benchmark.
- Narrow same-architecture 5 dB robustness improvement from high-weight band-energy regularization.
- Methodological caution: broken/inert physics losses can pass ordinary tests and contaminate a whole research story.

What is not defensible:

- "Physics-informed learning improves fault diagnosis."
- "Correct journal-bearing physics caused the noise robustness."
- "The model attends to physics bands."
- "Physics improves calibration."
- "Physics improves data efficiency or severity-OOD."
- Any real-bearing or field-deployment generalization claim.

Would I stake my name on the surviving claim as written?

I would stake my name on this wording:

> On this synthetic dataset, for retained checkpoints and record-level soft-voted evaluation, a high-weight band-energy spectral-consistency regularizer improved 5 dB robustness over a same-architecture CE-only PC-CNN. A scrambled-reference control reproduced the representative-seed robustness, so the effect should be interpreted as generic spectral regularization rather than evidence for physics-specific priors.

I would not stake my name on wording that calls the result a physics-informed benefit without immediately stating the F9 control result.

## Publishability Verdict

Conditionally publishable, but only under a strict honesty bar.

Suitable framing:

- Synthetic journal-bearing dataset and benchmark.
- Record-level reanalysis and reproducibility discipline.
- Negative results for broad physics-informed claims.
- One narrow, controlled 5 dB robustness effect from a band-energy spectral regularizer.
- Scrambled-reference control showing physics-specificity is not established.

Likely venue tier:

- IEEE Access, Sensors, Measurement-type venue, applied ML workshop, or arXiv.
- Not a top mechanical-systems venue unless paired with real-bearing validation or a much stronger physics-specific result.

Minimum honesty bar before submission:

- Synthetic-only in title/abstract/introduction limitations.
- No abstract-level "physics-informed learning improves..." claim.
- F9 control in the main results, not buried in appendix.
- n=3 and near-ceiling discordant-count limitations stated with the headline.
- Clean trade-off stated.
- All non-surviving results explicitly negative or neutral.
- Full artifact/provenance manifest.

## Prioritized Recommendations

1. Tighten the central wording. Replace any unqualified "scramble reproduces robustness" with "scramble reproduces the representative-seed robustness and is robust in 2 of 3 seeds, but has high seed variance; correct references may improve stability." Never call the surviving result a physics-specific gain.

2. Add a paper-grade provenance manifest. For every table row: command, code SHA, dataset hash, checkpoint path and hash, cache path and hash, metrics path, host/device, and exact script version.

3. Rerun the minimal decisive grid with more seeds if compute allows: CE-only same current runner, correct w=1.0, scramble w=1.0, same protocol, ideally 10+ seeds. Pre-register the seed-level estimand before running. This is the highest-value future compute.

4. Rerun the CE-only PC-CNN control in the same `run_phase5_gpu.py` code path if feasible. The current comparison is fair enough, but a same-runner CE-only arm removes the older-commit objection.

5. Keep record-level statistics as the only inferential basis. Do not report window-level p-values except as explicitly descriptive diagnostics.

6. Put the 14-record mechanism in the paper, but do not overinterpret it. It is useful that all rescued records are `lubrification` vs `mixed_wear_lube`; F9 proves that this alone does not establish a correct-physics mechanism.

7. Keep XAI/calibration/data-efficiency/OOD demoted. Do not resurrect C4 as a side-positive.

8. Fix the `ops_aware` provenance ambiguity in future metrics. Add a separate `train_metadata_rpm_used: true` or similar field for PC-CNN physics runs.

9. Add a small unit test for `reference_permutation`: assert derangement, assert loss/gradient changes on fixed logits/signals, and assert the permutation is recorded in F9 metrics.

10. Preserve the quarantine tests permanently. The old inert path is exactly the kind of code that can silently poison a future rerun.

## Final Bottom Line

This is real enough to publish honestly, but not in the story the project originally wanted. The defensible paper is a rigorous synthetic benchmark and correction story with one narrow spectral-regularization robustness result. If the manuscript says "physics-informed learning helps," it is overclaiming. If it says "after fixing broken physics losses and applying record-level tests, only a high-weight band-energy regularizer improved 5 dB robustness, and a wrong-physics control shows the benefit is not physics-specific," that is defensible.
