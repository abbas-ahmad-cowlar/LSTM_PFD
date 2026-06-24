# Independent Technical Audit - LSTM-PFD fourth round - GPT-5

Date: 2026-06-24  
Repo: `C:\Users\COWLAR\projects\lstm-pfd`  
Branch observed: `p7/strengthen`  
Audit scope: decisive n=12 P7 replication first, then methodology, then publishability spot checks.  
Prior audit reports: not read.

## Plain Answers Up Front

1. **Trustworthy?** Yes for the current n=12 conclusion. The decisive P7 numbers reproduced. The maintainer script, my independent cache aggregation, all window-level sanity gates, and targeted cold checkpoint inference agree. Important limitation: I did not cold-run all 96 checkpoint/split forward passes because one cold ResNet split took 131 seconds; a full cold pass would be hours on this CPU. I validated the per-window probability cache against cold inference on representative first batches and one full CE seed0 clean split, then recomputed the seed-level estimand independently from those checkpoint-derived probabilities.

2. **Publishable, as what?** Publishable only as a **synthetic dataset + frozen record-level benchmark + rigorous negative result on these physics-informed mechanisms**, with a useful methodological caution: a result that looked significant at n=3 disappeared at n=12. It is not publishable as "physics-informed learning beats data-driven models." It is not publishable as "correct journal-bearing physics improves noise robustness." At n=12, correct physics is statistically indistinguishable from CE-only.

3. **Honest venue/bar.** The honest venue is a dataset/benchmark or applied ML/reliability venue that accepts synthetic-only benchmark papers and negative results. Minimum honesty bar before submission: update every stale findings/paper draft, make record-level statistical units explicit, state that the P7 n=12 grid kills the surviving n=3 positive, include exact artifact/provenance manifest, and remove all CWRU/98.1%/expert-validated/PINN-helps claims.

4. **What I would not sign my name to.** I would not sign any manuscript claiming a physics-specific benefit, a replicated noise-robustness benefit, clean-accuracy improvement, data-efficiency improvement, severity-OOD improvement, interpretability gain, or calibration gain. I also would not sign the current `config/docs/paper/main.tex`; it describes a different, non-existent study.

One-line final verdict: **The results are trustworthy enough for a synthetic benchmark/null-result paper, but the n=3 noise-robustness positive was a seed artifact and the repository still contains stale paper-shaped claims that would make a submission misleading.**

## Priority 1 - n=12 Replication

### Maintainer script

Command:

```powershell
$env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe scripts\p7_strengthen_record_level.py
```

Key output:

```text
test records: 528  | seeds: 12

  w0_CEonly     : clean 99.12+/-0.62 | 5dB 95.58+/-4.14 | degr 3.54+/-3.85 (median 2.46) | robust 4/12
  w1.0_correct  : clean 98.72+/-0.77 | 5dB 95.25+/-3.72 | degr 3.47+/-3.42 (median 2.84) | robust 5/12
  w1.0_scramble : clean 98.45+/-0.97 | 5dB 93.56+/-6.64 | degr 4.89+/-6.08 (median 0.47) | robust 7/12
  w1.0_random   : clean 98.78+/-0.64 | 5dB 96.64+/-3.31 | degr 2.15+/-3.34 (median 0.38) | robust 9/12

  seed-level Wilcoxon signed-rank vs CE-only (paired per-seed degradation, n=12):
    w1.0_correct  : median Delta degr +0.28 | p(2-sided) 0.791 | p(more-robust) 0.3955 | 7/12 seeds beat CE-only
    w1.0_scramble : median Delta degr +0.09 | p(2-sided) 0.7505 | p(more-robust) 0.6418 | 6/12 seeds beat CE-only
    w1.0_random   : median Delta degr +1.89 | p(2-sided) 0.2061 | p(more-robust) 0.103 | 8/12 seeds beat CE-only

  robust-seed counts (degr<1.0): w0_CEonly 4/12 | w1.0_correct 5/12 | w1.0_scramble 7/12 | w1.0_random 9/12
```

This already answers the decisive question: no arm clears the preregistered bar of `>=10/12` robust seeds plus Wilcoxon significance.

### Independent recomputation

I first attempted a fully cold independent evaluator. It correctly revealed that the P7 checkpoints are ResNet-backed, not `cnn1d`-backed:

```text
RuntimeError loading with PhysicsConstrainedCNN(backbone='cnn1d'):
Unexpected key(s) in state_dict: "backbone.conv1.weight", "backbone.layer1..."
```

A cold single-checkpoint clean split with `PhysicsConstrainedCNN(backbone='resnet18')` took 131 seconds:

```text
windows (2640, 20480) batch 128
first batch seconds 6.62
done seconds 131.21 window_acc 95.61 record_acc 98.67
```

That makes 96 cold checkpoint/split passes an hours-scale job here. I therefore recomputed the P7 seed-level statistics independently from the per-window probability cache under `results/phase5_bandenergy/_record_cache`, after validating it:

```text
independent cache aggregation: 528 records, 2640 windows, seeds=12
w0_CEonly      clean 99.12+/-0.62 | 5dB 95.58+/-4.14 | degr 3.54+/-3.85 median 2.46 | robust 4/12
  degr_per_seed: 1.89 8.33 2.65 3.22 2.27 0.00 0.00 2.65 11.17 10.04 0.00 0.19
w1.0_correct   clean 98.72+/-0.77 | 5dB 95.25+/-3.72 | degr 3.47+/-3.42 median 2.84 | robust 5/12
  degr_per_seed: 0.00 0.38 -0.19 4.92 3.41 9.66 0.57 2.27 7.20 4.73 8.71 0.00
w1.0_scramble  clean 98.45+/-0.97 | 5dB 93.56+/-6.64 | degr 4.89+/-6.08 median 0.47 | robust 7/12
  degr_per_seed: 9.09 -0.38 -0.19 0.57 10.61 10.80 -0.57 17.23 0.38 -0.19 0.38 10.98
w1.0_random    clean 98.78+/-0.64 | 5dB 96.64+/-3.31 | degr 2.15+/-3.34 median 0.38 | robust 9/12
  degr_per_seed: 8.71 0.76 -0.19 0.38 0.19 0.00 -0.38 0.19 0.95 8.33 6.44 0.38
window-metric sanity max_abs_diff=0.000000; failures_gt_0.2=0
seed-level Wilcoxon vs CE-only:
w1.0_correct   median_reduction +0.28 mean_reduction +0.06 p_two 0.791016 p_more_robust 0.395508 seeds_better 7/12
w1.0_scramble  median_reduction +0.09 mean_reduction -1.36 p_two 0.750488 p_more_robust 0.641846 seeds_better 6/12
w1.0_random    median_reduction +1.89 mean_reduction +1.39 p_two 0.206055 p_more_robust 0.103027 seeds_better 8/12
```

Targeted cold cache validation:

```text
CE seed0 test: first_batch_max_abs_prob_diff=0; pred_equal=True
correct seed0 test: first_batch_max_abs_prob_diff=0; pred_equal=True
scramble seed0 test: first_batch_max_abs_prob_diff=0; pred_equal=True
random seed0 test: first_batch_max_abs_prob_diff=0; pred_equal=True
CE seed0 snr5: first_batch_max_abs_prob_diff=0; pred_equal=True
```

Conclusion: **the n=3 noise-robustness benefit does not replicate at n=12.** Correct physics vs CE-only is a null: mean degradation 3.47 vs 3.54 points, median paired degradation reduction +0.28 points, Wilcoxon two-sided p=0.791, one-sided p=0.3955, robust seeds 5/12 vs 4/12. Random bands are numerically best but still fail the preregistered bar: 9/12 robust, Wilcoxon p=0.206 two-sided / 0.103 one-sided.

## Priority 2 - Methodology of the n=12 Grid

### Preregistration timing

`experiments/PROTOCOL.md` contains the P7 section and fixed decision rule: section 8.8 starts at `experiments/PROTOCOL.md:197`, defines the all-ResNet same-budget arms at `experiments/PROTOCOL.md:212`, the seed-level degradation estimand at `experiments/PROTOCOL.md:226`, and the robust-seed/Wilcoxon decision rule at `experiments/PROTOCOL.md:229`.

Git history:

```text
4a2063d99387989042031fcd62111d4962d9966d 2026-06-23T01:43:55+05:00 P7 section 8.8: random-band non-physics control + n=12 grid (pre-registered)
```

Structured metrics scan:

```text
n_metrics 48
unique_git_sha ['4a2063d99387989042031fcd62111d4962d9966d']
finished_min 2026-06-22T21:22:29.343061+00:00
finished_max 2026-06-23T09:29:44.140749+00:00
```

The first run finished at 2026-06-23 02:22:29 +05:00, about 39 minutes after the preregistration commit at 01:43:55 +05:00. The grid artifacts all record that same preregistration commit. This passes.

### Fair grid

Structured scan:

```text
n 48
model {'physics_constrained_cnn': 48}
experiment {'pinn_ablation': 24, 'pinn_ablation_random': 12, 'pinn_ablation_scramble': 12}
physics_weight {'0.0': 12, '1.0': 36}
budgets {"(('batch', 64), ('lr', 0.001), ('max_epochs', 60), ('patience', 10))": 48}
devices {'cuda': 48}
best_epoch_minmax 6 57
state dict: has_resnet_layer1 True, has_cnn_block False, nkeys 122
```

The runner uses one `train_run` path for all arms (`scripts/run_phase5_gpu.py:239`), `create_model` for `physics_constrained_cnn` (`scripts/run_phase5_gpu.py:248`), identical budget provenance (`scripts/run_phase5_gpu.py:313`), a shared `--out-root` option for one folder (`scripts/run_phase5_gpu.py:361`), control-arm queues at `scripts/run_phase5_gpu.py:415` and `scripts/run_phase5_gpu.py:436`, and the regular `pinn_ablation` queue for both `w=0.0` and `w=1.0` at `scripts/run_phase5_gpu.py:484`.

This is methodologically sound: same architecture, same budget, same code path, same commit. The CE-only arm is not an old borrowed Phase-4 result in this P7 grid; it was rerun under `results/p7_strengthen/pinn_ablation/w0.0`.

### Random-band control

The random control is a real matched-structure non-physics control:

- The generator states the identical loss form and frozen artifact at `scripts/compute_random_reference.py:20` and `scripts/compute_random_reference.py:23`.
- The generated artifact path is fixed at `scripts/compute_random_reference.py:48`.
- Loading returns `random_signature` and `random_reference` objects (`packages/core/models/physics/fault_signatures.py:165`).
- The model stores opt-in random-control attributes at `packages/core/models/pinn/physics_constrained_cnn.py:97` and uses the random path at `packages/core/models/pinn/physics_constrained_cnn.py:200`.
- Random bands enter the same penalty and softmax-weighted loss form at `packages/core/models/pinn/physics_constrained_cnn.py:239`, `packages/core/models/pinn/physics_constrained_cnn.py:265`, and `packages/core/models/pinn/physics_constrained_cnn.py:276`.
- Contract tests assert count matching, non-overlap, default-off behavior, and differentiability in `tests/test_physics_random_band_control.py:34`, `tests/test_physics_random_band_control.py:98`, and `tests/test_physics_random_band_control.py:110`.

Generator rerun into a scratch path:

```text
Wrote audit_reports\random_reference.regenerated.json  (1120 healthy windows)
original_payload_sha256 4595070f1dc8f76e3a332d6d99ee6cba7c7fab3e1742474e3af11b23ca7ca5cf
regenerated_payload_sha256 4595070f1dc8f76e3a332d6d99ee6cba7c7fab3e1742474e3af11b23ca7ca5cf
payload_equal_ignoring_provenance True
scratch_removed True
```

Loss-path sanity:

```text
class desequilibre
validated_60Hz_loss 1.477160171958758e-08
random_control_60Hz_loss 1.0
uniform_logits_random_loss 0.9090909957885742 requires_grad True grad_abs_sum 0.16528917849063873
```

One minor reproducibility gap: the random-reference payload is deterministic, but the random-arm `metrics.json` files do not record a random-reference payload hash. I would add that to any paper artifact manifest.

### Analysis correctness

The reusable record-level evaluator uses `WindowedView` and cache keys by checkpoint/split (`scripts/phase5_bandenergy_record_level.py:56` to `scripts/phase5_bandenergy_record_level.py:70`), sanity-gates window accuracy against recorded metrics (`scripts/phase5_bandenergy_record_level.py:91` to `scripts/phase5_bandenergy_record_level.py:94`), and soft-votes five windows per record (`scripts/phase5_bandenergy_record_level.py:95`). P7 then computes seed-level degradation, Wilcoxon, robust-seed counts, and the decision rule at `scripts/p7_strengthen_record_level.py:103` to `scripts/p7_strengthen_record_level.py:152`.

Targeted random-control tests:

```text
tests\test_physics_random_band_control.py .....                          [ 41%]
tests\test_physics_band_energy_loss.py .......                           [100%]
12 passed, 1 warning in 57.67s
```

Full suite:

```text
collected 274 items / 6 deselected / 268 selected
========= 268 passed, 6 deselected, 49 warnings in 228.20s (0:03:48) ==========
```

This passes the requested bar.

## Priority 3 - Overall Publishability

### Clean benchmark and prior negatives

Record-level benchmark recompute:

```text
Record-level aggregation on cpu @ b64502a9
test split: 2640 windows, 5 windows/record, 528 records
...
Wrote results/benchmark/summary_record_level.{json,md} in 6s
  best vanilla cnn_lstm 99.43% | best physics-labeled physics_constrained_cnn 98.99%
  gap +0.00 pts CI [+0.00, +0.00]
```

Extracted top rows:

```text
cnn_lstm mean 99.43 std 0.15 bestval 99.62
resnet18 mean 99.18 std 0.09 bestval 99.24
physics_constrained_cnn mean 98.99 std 0.45 bestval 99.62 note CE-only (architecture; physics loss OFF)
hybrid_pinn mean 90.34 std 0.94 bestval 91.10 note rolling-element branch + constant metadata
multitask_pinn mean 90.47 std 0.62 bestval 91.29 note single-task (aux heads unused)
```

The clean benchmark is sound as a classification benchmark, but it does not support a physics advantage. The best physics-labeled row is explicitly CE-only, so it is not evidence for physics.

Phase-5 record-level recompute:

```text
section 8.4 ablation:
  w=0.0 clean 98.99+/-0.45 | 5dB 94.70+/-3.08
  w=1.0 clean 98.61+/-0.62 | 5dB 98.55+/-0.76

section 8.2 data efficiency:
  pc_cnn 10% 96.28+/-1.39; resnet18 10% 96.28+/-1.10
  pc_cnn 25% 98.48+/-0.15; resnet18 25% 97.22+/-0.62
  pc_cnn 50% 98.80+/-0.79; resnet18 50% 98.80+/-0.24

section 8.3 severity-OOD:
  A: pc_cnn 100.00+/-0.00; resnet18 100.00+/-0.00
  B: pc_cnn 82.58+/-0.62; resnet18 76.01+/-3.41
```

The old n=3 section 8.4 positive is superseded by P7 n=12. Data efficiency has only one reduced fraction clearly ahead and therefore fails the preregistered ">=2 reduced fractions with non-overlapping +/-1 std" rule (`experiments/PROTOCOL.md:116` to `experiments/PROTOCOL.md:120`). Severity-OOD direction A is a ceiling tie; direction B is descriptive only because representative-seed CI crosses zero and McNemar p=0.388.

XAI/calibration spot check:

```text
XAI tonal:
resnet18_vanilla in/control/ratio 0.1875 0.1799 1.0420
pc_cnn_physics   in/control/ratio 0.1259 0.1470 0.8562

XAI band-aware:
resnet18_vanilla 0.1458 n 200
pc_cnn_physics   0.0991 n 200

Calibration ECE:
resnet18 clean 0.02364, snr5 0.02616
pc_cnn   clean 0.01818, snr5 0.02277
```

XAI is not a positive: vanilla has higher physics-band attribution than the physics-trained model. Calibration is a single-checkpoint MC-dropout comparison; even where the current JSON favors pc_cnn in ECE, it is not seed-level and the project notes run-to-run direction instability. I would not claim calibration.

### Dataset/generator and leakage

Dataset spot check:

```text
file_attrs {'dataset_version': 'v2', 'num_classes': 11, 'records_per_class': 320,
            'records_per_class_per_severity': 80, 'rng_seed': 42,
            'sampling_rate': 20480, 'signal_length': 102400,
            'snr_variants_db': [20, 10, 5], 'split_ratios': [0.7, 0.15, 0.15]}
train n 2464 classes [224] severities {'incipient': 616, 'mild': 616, 'moderate': 616, 'severe': 616}
val n 528 classes [48] severities {'incipient': 132, 'mild': 132, 'moderate': 132, 'severe': 132}
test n 528 classes [48] severities {'incipient': 132, 'mild': 132, 'moderate': 132, 'severe': 132}
overlap_train_val 0
overlap_train_test 0
overlap_val_test 0
test_snr5_labels_equal_test True
```

The generator is designed around record-level stratified splits and hash leakage gates (`scripts/generate_dataset_v2.py:7`, `scripts/generate_dataset_v2.py:13`, `scripts/generate_dataset_v2.py:59`, `scripts/generate_dataset_v2.py:136`, `scripts/generate_dataset_v2.py:154`). `WindowedView` maps windows back to records (`data/dataset.py:609`, `data/dataset.py:650`), so the independent unit is the record. The dataset is sound as a **synthetic** journal-bearing benchmark. I did not find leakage in the HDF5.

### Stale paper-shaped landmines

This is the main publishability blocker.

`results/FINDINGS.md` still contains stale n=3 language: a "noise-robustness improvement" at `results/FINDINGS.md:24`, "one surviving physics positive" at `results/FINDINGS.md:57`, and allowed wording for "implemented PC-CNN band-energy consistency loss ... improved 5 dB robustness" at `results/FINDINGS.md:112`. Those are no longer valid after P7 n=12. The same file also contains an older historical snapshot claiming "modest interpretability and calibration gain" (`results/FINDINGS.md:177`, `results/FINDINGS.md:230`, `results/FINDINGS.md:252`). Even if intended as historical, it is dangerous in a submission repo.

`config/docs/paper/main.tex` is worse: it claims CWRU data, 98.1% accuracy, improved accuracy and physical consistency, expert-validated explanations, significant ablations, and physics-constraint benefits (`config/docs/paper/main.tex:54` to `config/docs/paper/main.tex:60`, `config/docs/paper/main.tex:119` to `config/docs/paper/main.tex:120`, `config/docs/paper/main.tex:350`, `config/docs/paper/main.tex:411`, `config/docs/paper/main.tex:422`, `config/docs/paper/main.tex:480`). That draft must not be submitted or cited as representing these results.

## Severity-Tagged Findings

### S0 - Decisive scientific finding: P7 n=12 kills the surviving positive

Correct physics does not beat CE-only at n=12. Correct physics has degradation 3.47+/-3.42 vs CE-only 3.54+/-3.85; median paired degradation reduction +0.28 points; Wilcoxon p=0.791 two-sided / 0.3955 one-sided; robust seeds 5/12 vs CE-only 4/12. No correct/scramble/random arm reaches the preregistered bar. The n=3 result was seed-fragile.

### S1 - Stale result narratives would mislead a referee

`results/FINDINGS.md` and `config/docs/paper/main.tex` contradict the verified n=12 result and broader negatives. Any paper draft must be rebuilt from current recomputed results, not patched around these files.

### S1 - Do not claim "physics-informed learning helps"

The evidence supports a dataset/benchmark and a negative result for these mechanisms. It does not support physics-specific robustness, accuracy, data-efficiency, severity-OOD, interpretability, or calibration claims.

### S2 - Reproducibility manifest needs tightening

The P7 grid itself is well-provenanced by commit and finished_at timestamps, but the random-arm metrics should record the random-reference payload hash. The audit also exposed that full cold CPU recomputation of every P7 checkpoint/split is expensive; publishable replication should document cache generation, cache invalidation, and expected runtime, or provide an optional faster inference path.

### S2 - Record-level unit must remain prominent

The scripts now do the right thing, but the paper must repeatedly state that accuracy statistics use 528 records, not 2640 windows, and that cross-run claims use seeds, not windows.

## Commands Run

High-signal commands and outcomes:

- `.\venv\Scripts\python.exe scripts\p7_strengthen_record_level.py`: reproduced the n=12 null and wrote `results/p7_strengthen/p7_strengthen_record_level.json`.
- Independent cache aggregation from checkpoint-derived per-window probabilities: reproduced all P7 clean/5dB/degradation/Wilcoxon values; all 96 window-accuracy sanity checks had max absolute diff 0.
- Cold inference spot checks: one full CE seed0 clean split matched cached/record stats; five first-batch probability comparisons had max abs diff 0.
- `git log -- experiments/PROTOCOL.md` plus metrics scan: prereg commit 4a2063 predates all runs; all 48 metrics point at 4a2063.
- Random-reference generator rerun into scratch path: payload hash matched exactly ignoring provenance.
- `.\venv\Scripts\python.exe -m pytest -q tests\test_physics_random_band_control.py tests\test_physics_band_energy_loss.py`: 12 passed.
- `.\venv\Scripts\python.exe -m pytest -q`: 268 passed, 6 deselected.
- `.\venv\Scripts\python.exe scripts\aggregate_benchmark_record_level.py`: benchmark record-level recompute succeeded.
- `.\venv\Scripts\python.exe scripts\phase5_bandenergy_record_level.py`: Phase-5 record-level recompute succeeded.
- HDF5 split/hash probe: exact balance and zero train/val/test record-hash overlap.

## Minimum Honesty Bar Before Submission

1. Replace `results/FINDINGS.md` section 0 with the n=12 null result.
2. Delete or quarantine `config/docs/paper/main.tex`; it describes a different study.
3. State the main result as: "On this synthetic benchmark, the tested physics-informed mechanisms do not outperform data-driven baselines; a promising n=3 noise result failed to replicate at n=12."
4. Include a manifest with dataset hash, git commit, checkpoint list, random-reference payload hash, commands, caches, and record-level statistical scripts.
5. Make every figure/table record-level unless explicitly labeled otherwise.
6. Label the contribution as synthetic-only. No real-bearing or CWRU performance claims.

## Final Verdict

**Trust the recomputed null. Publish only as a synthetic dataset/frozen benchmark plus a rigorous complete negative and a methodological caution; do not publish as a physics-informed-learning win.**
