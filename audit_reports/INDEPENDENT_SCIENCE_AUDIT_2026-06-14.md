# Independent Science Audit Report

Repository: `lstm-pfd`

Audit date: 2026-06-14

Auditor role: independent technical auditor. All repository text, including
README files, project state documents, previous audit reports, and comments, was
treated as an unverified claim until checked against code, data, or execution.

## Executive summary

The project is not ready to publish as a physics-informed bearing
fault-diagnosis study in its current state.

The synthetic dataset v2 is mostly internally coherent: class-balanced,
group-split by 5-second records, duplicate-free under full-signal hashing, and
the main journal-bearing fault classes do show the intended spectral signatures.
The test suite also passes. Those are real strengths.

However, the scientific claims around "physics-informed" models remain
untrustworthy. The current repository is in a transitional state where the
maintainers have found and documented part of the problem, but not all of it.
The physics loss used by generic trainer code is still non-differentiable and
therefore inert. The main fixed PC-CNN physics loss is differentiable, but still
uses only tonal expected frequencies and does not yet implement the planned
band-energy formulation required for broadband and mixed-fault classes. The
HybridPINN physics branch uses rolling-element bearing characteristic
frequencies for an SKF 6205 ball bearing, while the generator and current
dataset model journal/hydrodynamic bearing faults. That makes the HybridPINN
"physics" pathway physically inconsistent with the data.

The headline benchmark accuracies are plausible as synthetic classification
numbers, but several labels and comparisons are misleading. In the Phase-4
benchmark script, `physics_constrained_cnn` is trained with cross-entropy only,
not physics loss. `multitask_pinn` is also trained only on the fault-label
cross-entropy path, not with its auxiliary multitask losses. The reported
"physics family" noise-robustness comparison is especially weak because it
groups models that were not actually trained with valid journal-bearing physics,
and its average is materially affected by a single poor vanilla model
(`attention_cnn`).

My independent verdict: the repository can support a narrower claim that it
contains a reproducible synthetic journal-bearing classification benchmark with
mostly coherent generated signals. It cannot yet support a publishable claim
that physics-informed neural modeling has been validly tested or shown to help.
The paper should not be submitted until the physics modules are fixed or
quarantined, the affected experiments are rerun, the statistics are recomputed at
record level, and the claims are rewritten around verified evidence only.

## Commands executed

The following commands were run in PowerShell from the repository root
`C:\Users\COWLAR\projects\lstm-pfd`.

```powershell
git status --short --branch
git log --oneline --decorate -n 30
.\venv\Scripts\python.exe --version
$env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe -m pytest -q
$env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe scripts\verify_physics_consistency.py
```

I also ran independent Python probes against `data/generated/dataset_v2.h5`,
the JSON result artifacts, and the physics-loss/model code paths. The important
observed outputs are quoted in the findings below.

Baseline execution results:

```text
git status:
## p6/docs...origin/p6/docs

python:
Python 3.14.0

pytest:
251 passed, 6 deselected, 49 warnings in 66.88s
TOTAL coverage: 14.72%
```

The test suite passing is useful, but the 14.72% coverage means the tests do not
meaningfully exercise most experiment, training, physics, and analysis code.

## Repository state and provenance

Current HEAD at audit time:

```text
9337e77 P6: ratify band-energy physics loss (PROTOCOL sec 7); STOP for independent external audit
4b632c7 P6 remediation Tier A+B: rebuild signature DB, add tests, document loss replacement
af4d864 P6 AUDIT: physics-loss uses rolling-element signature DB, Section 8.4 contaminated
```

The result artifacts were not all produced at current HEAD. I independently
checked JSON metadata and found multiple source commits, including:

```text
results/benchmark/*:                 e498deb0...
results/phase5_fixed/pinn_ablation:  e894389...
results/phase5_dataeff_fixed:        d39c219...
results/phase5_true_metadata:        4f637a2...
results/noise_robustness:            b868338...
results/xai_calibration:             0d2e3d8...
current HEAD:                        9337e77...
```

This is not automatically invalid, but it is a reproducibility risk. A reader
cannot assume current code exactly reproduces every result artifact. A
publication needs an experiment manifest that pins the code commit, dataset
hash, command, config, seed list, hardware, and output file for every reported
number.

## Dataset and synthetic signal physics

### Finding 1: Dataset v2 split integrity is good

Severity: minor positive finding

Evidence:

`scripts/generate_dataset_v2.py` implements a replacement dataset writer with
per-split metadata and group-aware stratification. Relevant code:

- `scripts/generate_dataset_v2.py:1-18`: states the design goals, including
  split-local metadata and no leakage.
- `scripts/generate_dataset_v2.py:100-119`: generates 11 fault classes across
  four severity slots.
- `scripts/generate_dataset_v2.py:132-139`: performs split assignment and hash
  uniqueness checks.
- `scripts/generate_dataset_v2.py:145-186`: writes split-local signals, labels,
  severities, hashes, metadata, and SNR variants.

Independent HDF5 inspection of `data/generated/dataset_v2.h5` observed:

```text
dataset_version: v2
num_classes: 11
records_per_class: 320
records_per_class_per_severity: 80
rng_seed: 42
sampling_rate: 20480
signal_length: 102400
split_ratios: [0.7, 0.15, 0.15]

train: 2464 records, 12320 one-second windows
val:    528 records,  2640 one-second windows
test:   528 records,  2640 one-second windows

train class count: 224 per class
val class count:    48 per class
test class count:   48 per class

full-signal hash overlaps:
train-val:  0
train-test: 0
val-test:   0
```

Conclusion: dataset v2 appears properly balanced and duplicate-free at the
record level. The main split-leakage issue in the old HDF5 writer has been fixed
for v2.

### Finding 2: The main synthetic fault signatures are mostly present

Severity: minor positive finding with caveats

The generator is a journal/hydrodynamic synthetic fault model, not a
rolling-element bearing model. Relevant code:

- `data/signal_generation/fault_modeler.py:61-258`: implements healthy,
  misalignment, imbalance, clearance, lubrication, cavitation, wear, oil whirl,
  and mixed journal-bearing fault signals.
- `data/signal_generation/generator.py:150-239`: composes the full signal from
  operating conditions, severity, transients, the fault modeler, optional
  advanced physics, and noise.
- `config/data_config.py:24-31`: sets `fs=20480`, 5-second records, and nominal
  shaft frequency near 60 Hz.
- `config/data_config.py:245-312`: advanced physics effects are disabled by
  default in the production v2 data config.

I ran the repository's physics consistency script:

```powershell
$env:PYTHONIOENCODING='utf-8'; .\venv\Scripts\python.exe scripts\verify_physics_consistency.py
```

Observed output:

```text
Signal length: 102400 samples
Frequency resolution: 0.20 Hz
Tonal expected frequencies with detected matching peaks: 15/16 = 94%

desalignement: expected 119, 179 Hz -> OK
desequilibre:  expected 59 Hz -> OK
jeu:           expected 27, 60, 121 Hz -> OK
oilwhirl:      expected 27 Hz -> OK
mixed_wear_lube: expected 60 Hz OK, 121x missed
lubrification/cavitation: no characteristic tonal frequencies in this script
```

I then independently sampled 20 test records per class from
`data/generated/dataset_v2.h5` and computed full-record FFT and statistics. Key
observations:

```text
class                     rms     crest  kurtosis  signature-band fraction  top frequencies
sain                      0.0497  3.94   2.91      0.0000                  1.2, 1.0, 51.4 Hz
desalignement             0.1427  3.00   2.32      0.3914                  119.8, 109.4, 111.2 Hz
desequilibre              0.1419  2.85   1.93      0.5320                  64.6, 61.0, 63.4 Hz
jeu                       0.1166  3.32   2.55      0.6775                  25.8, 27.6, 61.2 Hz
lubrification             0.2037  2.89   1.83      0.9257                  4.4, 3.2, 2.2 Hz
cavitation                0.0498  6.21   3.72      0.0504                  1.0, 56.2, 1.2 Hz
usure                     0.1325  5.27   3.43      0.0868                  56.0, 54.2, 57.4 Hz
oilwhirl                  0.2880  2.31   1.76      0.8960                  25.8, 26.8, 27.0 Hz
mixed_misalign_imbalance  0.0843  3.63   2.74      0.6124                  60.2, 120.2, 180.4 Hz
mixed_wear_lube           0.0862  4.32   2.73      0.5230                  4.0, 3.8, 2.6 Hz
mixed_cavit_jeu           0.0779  4.09   2.72      0.5068                  27.4, 25.8, 26.2 Hz
```

Conclusion: the generated data is not random label noise. Most classes exhibit
the expected synthetic signatures. This supports the dataset as an internally
coherent synthetic benchmark.

### Finding 3: Cavitation is weakly expressed spectrally

Severity: major if cavitation-specific claims are made; otherwise minor

Evidence:

`data/signal_generation/fault_modeler.py:115-134` implements cavitation as
high-frequency bursts between 1500 and 2500 Hz. In my independent sample, the
1400-2600 Hz energy fraction was:

```text
cavitation: 0.0504
healthy:    0.0335
```

Cavitation did show elevated crest factor and kurtosis:

```text
cavitation crest factor: 6.21
healthy crest factor:    3.94

cavitation kurtosis:     3.72
healthy kurtosis:        2.91
```

Impact:

The cavitation class is detectable statistically, but the intended high-frequency
spectral signature is only modestly above healthy in the sampled records. Any
paper claim that cavitation is strongly represented by the intended
high-frequency band should be qualified or revalidated across all records and
severity levels.

Recommendation:

Report cavitation as an impulse/statistical synthetic class unless a full
per-severity spectral audit shows robust high-frequency separation. If
cavitation is important to the publication, tune the generator and regenerate the
dataset under a frozen protocol.

### Finding 4: Some "advanced physics" effects are mislabeled or simplified

Severity: minor for current v2 dataset; major if enabled in future experiments

Evidence:

Advanced physics options are disabled by default in `config/data_config.py:245-312`,
so this does not invalidate dataset v2. But the code does not always do what its
names imply:

- `data/signal_generation/fault_modeler.py:374-385`: `_apply_speed_fluctuation`
  is described as frequency modulation, but line 383 says "Apply as amplitude
  modulation" and line 385 returns `x * modulation`.
- `data/signal_generation/generator.py:320-347`: `_configure_transients` uses a
  speed-ramp transient as an amplitude modulation window from 0.85 to 1.15, not
  as true speed-varying phase for the fault frequencies.

Impact:

The current production data avoids most of this because advanced physics is off.
But the code names and comments could mislead future maintainers into believing
they have speed-varying physics when they mostly have amplitude modulation.

Recommendation:

Rename these effects or implement true phase/frequency modulation. Add tests
that inspect instantaneous frequency when speed transients are enabled.

## Physics inside the models

### Finding 5: Generic physics loss is non-differentiable and inert

Severity: critical

Evidence:

`packages/core/training/physics_loss_functions.py` defines
`FrequencyConsistencyLoss`. It uses the predicted class by `argmax`:

- `packages/core/training/physics_loss_functions.py:78-90`: computes
  `predicted_classes = torch.argmax(predictions, dim=1)`.
- `packages/core/training/physics_loss_functions.py:106-133`: loops over those
  discrete classes and computes losses from expected frequencies.
- `packages/core/training/physics_loss_functions.py:332-337`:
  `PhysicalConstraintLoss` includes that frequency-consistency loss.
- `packages/core/training/pinn_trainer.py:121-127`: `PINNTrainer` instantiates
  `PhysicalConstraintLoss`.
- `packages/core/training/pinn_trainer.py:229-237`: the trainer adds this loss
  during training when physics lambda is positive.

I executed a direct gradient check:

```text
--- Standalone FrequencyConsistencyLoss ---
loss 0.6533333659172058 requires_grad False grad_fn None
backward error RuntimeError element 0 of tensors does not require grad and does not have a grad_fn

--- Combined PhysicalConstraintLoss ---
loss 7.573333263397217 requires_grad False grad_fn None
backward error RuntimeError element 0 of tensors does not require grad and does not have a grad_fn
```

Impact:

Any experiment using `PhysicalConstraintLoss` through `PINNTrainer` does not
train with a functioning differentiable physics signal. This is not a subtle
regularization weakness; it is a hard autograd failure/inert path. The passing
pytest suite did not catch this.

Recommendation:

Quarantine or delete this loss path until it is replaced by a differentiable
band-energy loss with gradient tests. Add tests that assert
`loss.requires_grad`, nonzero gradients to logits or features, and class-specific
behavior for tonal, broadband, and mixed classes.

### Finding 6: PC-CNN's model-specific loss is differentiable but still incomplete

Severity: critical for affected physics-loss experiments

Evidence:

`packages/core/models/pinn/physics_constrained_cnn.py` contains a separate
model-specific `compute_physics_loss`:

- `packages/core/models/pinn/physics_constrained_cnn.py:123-220`: computes a
  softmax-weighted expected-frequency loss.
- `packages/core/models/pinn/physics_constrained_cnn.py:172-181`: uses a scalar
  mean RPM if batch metadata is not supplied.
- `packages/core/models/pinn/physics_constrained_cnn.py:188-191`: uses
  `get_expected_frequencies(...)`, not `get_expected_bands(...)`.
- `packages/core/models/physics/fault_signatures.py:95-102`:
  `get_expected_frequencies` returns only tonal frequency entries.
- `packages/core/models/physics/fault_signatures.py:104-113`:
  `get_expected_bands` exists but is not used by this loss.

My gradient check showed this path is differentiable:

```text
--- PhysicsConstrainedCNN.compute_physics_loss ---
loss 13.386616706848145 requires_grad True grad_fn MeanBackward0
logits grad norm 2.2972042560577393
```

But because it uses only tonal expected frequencies, it does not supervise pure
broadband or absolute-band classes correctly. In the current signature database:

- `lubrification` is represented primarily by a 1-6 Hz absolute band.
- `cavitation` is represented primarily by high-frequency broadband bands.
- `usure` includes broadband behavior.
- mixed faults include absolute/broadband components.

Impact:

The current PC-CNN physics loss is better than the generic inert loss, but it is
not yet the band-energy physics loss needed for the actual journal-bearing data.
Loss-weight ablations and data-efficiency/OOD conclusions that rely on this
intermediate loss remain scientifically unsafe.

Recommendation:

Implement the ratified band-energy loss before rerunning any physics-loss
experiments. The implementation should use `get_expected_bands`, support
per-sample RPM metadata, handle empty tonal classes, and include class-level
gradient tests.

### Finding 7: HybridPINN uses rolling-element bearing physics against journal-bearing data

Severity: critical

Evidence:

The dataset generator and current signature DB are journal/hydrodynamic:

- `data/signal_generation/fault_modeler.py:61-258`: journal/hydrodynamic fault
  classes such as oil whirl, lubrication, cavitation, wear, imbalance,
  misalignment, clearance.
- `packages/core/models/physics/fault_signatures.py:2`: current signature DB is
  explicitly journal/hydrodynamic.

But HybridPINN's physics branch uses `BearingDynamics`, whose defaults are
rolling-element SKF 6205 ball-bearing parameters:

- `packages/core/models/physics/bearing_dynamics.py:30-52`: default parameters
  include `n_balls`, `ball_diameter`, `pitch_diameter`, contact angle, and SKF
  6205 geometry.
- `packages/core/models/physics/bearing_dynamics.py:57-138`: computes FTF, BPFO,
  BPFI, and BSF.
- `packages/core/models/pinn/hybrid_pinn.py:78-80`: HybridPINN instantiates
  `BearingDynamics`.
- `packages/core/models/pinn/hybrid_pinn.py:126-134`: the physics feature branch
  expects seven physics features, including characteristic frequencies.
- `packages/core/models/pinn/hybrid_pinn.py:220-239`: HybridPINN computes and
  feeds FTF/BPFO/BPFI/BSF/shaft frequencies into the physics branch.

Impact:

This is physically wrong for the synthetic data being classified. Results under
`results/phase5_true_metadata/` cannot be interpreted as a valid test of
journal-bearing physics metadata. They are at best a test of whether a neural
network can exploit, ignore, or survive a mismatched rolling-element feature
branch.

This also means the repository's own prior statement that section 8.5 is
"independent" or "sound" is too broad. It may be independent of the specific
fault-signature database bug, but it is not independent of physics/model
validity.

Recommendation:

Either remove HybridPINN from the paper's physics-informed claims or rebuild its
physics branch around journal-bearing quantities actually used by the generator:
shaft speed, oil-whirl subsynchronous ratio, 1X/2X/3X harmonics, low-frequency
lubrication band, cavitation high-frequency band, and wear broadband metrics.
Then rerun the relevant experiments from scratch.

### Finding 8: "Multitask PINN" benchmark is not trained as multitask

Severity: major

Evidence:

`packages/core/models/pinn/multitask_pinn.py` defines auxiliary task heads and a
`compute_multitask_loss` method:

- `packages/core/models/pinn/multitask_pinn.py:170-207`: default `forward`
  returns only fault logits unless `return_all_tasks=True`.
- `packages/core/models/pinn/multitask_pinn.py:209-281`: multitask loss exists.

But `scripts/run_benchmark.py` uses generic cross-entropy training:

- `scripts/run_benchmark.py:87-114`: `run_epoch` calls `logits = model(xb)` and
  applies `CrossEntropyLoss`.
- `scripts/run_benchmark.py:117-209`: benchmark loop trains all listed models
  through that same path.

The shared Colab helper has the same issue:

- `scripts/colab/_train_utils.py:175-199`: trains with `outputs = model(signals)`
  and `CrossEntropyLoss` only.
- `scripts/colab/04_train_batch2_physics.py:21-24` and `44-50`: labels these as
  physics-informed training runs but uses the generic helper.

Impact:

The reported `multitask_pinn` benchmark is not evidence about multitask
physics-informed learning. It is a fault-classifier trained through the default
single-task path.

Recommendation:

Relabel existing results or remove them. If multitask learning is part of the
paper, create a dedicated trainer that passes `return_all_tasks=True`, supplies
valid labels for speed/load/severity, optimizes the documented auxiliary losses,
and reports task-specific metrics.

## Experimental validity and reported results

### Finding 9: Phase-4 "physics constrained CNN" benchmark is physics-off

Severity: major

Evidence:

`scripts/run_benchmark.py` trains all models with cross-entropy only:

- `scripts/run_benchmark.py:87-114`: no physics loss is added in `run_epoch`.
- `scripts/run_benchmark.py:117-209`: `physics_constrained_cnn` is included in
  the model list but receives the same CE-only training as ordinary classifiers.

By contrast, `scripts/run_phase5_gpu.py` contains a special path that does add
`model.compute_physics_loss` if `physics_w > 0`:

- `scripts/run_phase5_gpu.py:178-216`: training loop.
- `scripts/run_phase5_gpu.py:195-197`: adds `physics_w * phys` only when the
  model has `compute_physics_loss`.

Impact:

The Phase-4 benchmark result for `physics_constrained_cnn` is not a result for a
physics-constrained training objective. It is a CNN architecture result under
ordinary CE training. The maintainers do acknowledge this in some documents, but
the model name and result tables can still mislead a skeptical reader.

Recommendation:

In all tables, rename this row to something like `pc_cnn_architecture_ce_only`
unless it is trained with a verified physics loss. Do not cite Phase-4 PC-CNN as
evidence for or against physics-informed training.

### Finding 10: The headline benchmark supports "no accuracy gain", not a positive physics claim

Severity: major

Evidence:

I recomputed the summary from `results/benchmark/summary.json`. Key values:

```text
resnet18:                 96.136 +/- 0.283
cnn_lstm:                 96.124 +/- 0.156
physics_constrained_cnn:  95.985 +/- 0.365
voting ensemble:          96.477 single-seed
random_forest:            94.609
hybrid_pinn:              90.038
multitask_pinn:           90.278
attention_cnn:            89.369 +/- 4.824

Wilcoxon p, best physics vs best vanilla: 0.5
McNemar p, PC-CNN vs ResNet18 windows:    1.0
```

Impact:

These numbers do not show a physics-informed accuracy advantage. The strongest
defensible statement is that the synthetic benchmark is easy for strong vanilla
models and that the tested physics-labeled models do not improve clean accuracy.
Given Findings 5-9, even that statement should be framed carefully because the
physics-labeled models were not all using valid physics.

Recommendation:

Make the negative result the center of the story only after relabeling the
models honestly and separating architecture-only, invalid-physics, and
verified-physics runs.

### Finding 11: Noise robustness "physics family" comparison is misleading

Severity: major

Evidence:

From `results/noise_robustness/summary.json`, the family-level degradation is
reported as:

```text
physics family degradation: 8.5059
vanilla family degradation: 15.5404
```

But per-model values show a different story:

```text
resnet18 clean:                 96.14
resnet18 SNR 5 dB:              94.43
resnet18 degradation:            1.70

physics_constrained_cnn clean:  95.98
physics_constrained_cnn SNR 5:  91.00
physics_constrained_cnn degradation: 4.99

patchtst degradation:            1.81
attention_cnn degradation:       50.59
```

Impact:

The family comparison is dominated by grouping choices and the weak
`attention_cnn` result. It does not establish that physics-informed models are
more noise robust. The best vanilla model, ResNet18, is more robust than the
PC-CNN row under the reported SNR-5 condition.

Recommendation:

Do not use the family average as evidence of physics robustness. Report
per-model noise curves, predefine fair architecture-matched comparisons, and
avoid grouping models that were not trained with valid physics.

### Finding 12: Statistical testing treats correlated windows as independent

Severity: major

Evidence:

The dataset consists of 5-second records that are evaluated as five
non-overlapping one-second windows:

- `data/dataset.py:609-651`: `WindowedView` creates the per-window view.

The benchmark aggregation uses the window-level targets and predictions:

- `scripts/aggregate_benchmark.py:122-125`: extracts targets from the
  `WindowedView`.
- `scripts/aggregate_benchmark.py:194-200`: computes McNemar tests on those
  window-level predictions.

Test split size is:

```text
528 independent records
2640 one-second windows
```

Impact:

Using 2640 windows as if they were independent overstates the effective sample
size because windows from the same 5-second record share operating condition,
fault realization, severity, and noise process. McNemar p-values, confidence
intervals, and any "test touched once" statements need to be interpreted at the
record level or with clustered resampling.

Recommendation:

Recompute all statistical comparisons at record level using majority vote or
mean logits per 5-second record. Use cluster bootstrap by record for confidence
intervals. Window-level metrics may be reported only as secondary descriptive
metrics.

### Finding 13: Existing tests do not cover the most important scientific failure modes

Severity: major

Evidence:

The full pytest run passed:

```text
251 passed, 6 deselected, 49 warnings in 66.88s
TOTAL coverage: 14.72%
```

The passing suite did not catch:

- non-differentiable generic physics loss;
- wrong rolling-element physics inside HybridPINN;
- benchmark training physics-labeled models with CE only;
- multitask PINN not trained as multitask;
- stale/broken research scripts;
- window-level statistical dependence.

Impact:

The test suite is useful for smoke checks, but not a sufficient reproducibility
or scientific-validity gate.

Recommendation:

Add targeted tests around physics gradients, model/trainer wiring, per-split
metadata, record-level aggregation, and script dry-runs for every command that
appears in the paper's methods section.

## Reproducibility and artifact concerns

### Finding 14: The project is partially reproducible but not one-command reproducible

Severity: major

Evidence:

Positive evidence:

- Dataset v2 has root metadata with seed, split ratios, class counts, severity
  levels, and SNR variants.
- Result JSONs contain useful metrics and some source metadata.
- The test suite passes in the provided virtual environment.

Negative evidence:

- Results come from multiple historical SHAs, not current HEAD.
- The repository contains stale or broken scripts, including
  `scripts/research/pinn_ablation.py`.
- Some physics-labeled Colab scripts use generic CE-only training.

Specific stale/broken script evidence:

- `scripts/research/pinn_ablation.py:89-99` attempts to construct
  `HybridPINN(lambda_physics=..., lambda_boundary=...)`, but the current
  `HybridPINN` constructor does not expose those arguments.
- `scripts/research/pinn_ablation.py:121-140` trains with cross-entropy only
  even though it is described as a PINN ablation.

Impact:

A skeptical reviewer cannot take the methods section, script names, and result
artifacts at face value. They need a curated, pinned reproduction path.

Recommendation:

Create a `reproduce/` or `experiments/frozen/` directory with only the commands
used for the paper. Every report table should map to one command, one manifest,
one commit, and one output artifact. Move stale scripts to an archive or mark
them explicitly as non-authoritative.

## Assessment of maintainers' remediation plan

### Finding 15: The current remediation plan is directionally right but insufficient

Severity: major

Evidence:

The repository has an internal physics-loss audit and remediation notes:

- `audit_reports/PHYSICS_LOSS_AUDIT_2026-06-14.md:9-27`: identifies
  non-differentiability, wrong bearing-family signatures, missing mixed classes,
  and broadband mismatch in the previous physics loss.
- `audit_reports/PHYSICS_LOSS_AUDIT_2026-06-14.md:61-87`: proposes rebuilding
  the signature DB, adding tests, replacing the loss with band-energy, rerunning
  affected sections, and quarantining dead/wrong modules.
- `PROJECT_STATE.md:13-26`: says the project is paused for external audit after
  DB rebuild and band-energy ratification.
- `experiments/PROTOCOL.md:75-76`: records amendments around band-energy loss
  and the original PC-CNN argmax defect.

I agree with the diagnosis that the PC-CNN loss based on old rolling-element
signatures was contaminated and that a differentiable band-energy loss is the
right replacement direction.

But I disagree with the plan's implied blast-radius boundary. The plan and state
files understate or miss the following:

1. `PINNTrainer` and `PhysicalConstraintLoss` remain non-differentiable and must
   be fixed or quarantined.
2. HybridPINN still uses rolling-element `BearingDynamics`, so section 8.5
   "true metadata" is not a sound journal-bearing physics experiment.
3. Benchmark and noise-robustness tables still contain physics-labeled models
   trained without verified physics.
4. Window-level statistics need record-level recomputation.
5. Stale research and Colab scripts can still mislead future reproduction
   attempts.

Impact:

The maintainers have done the right thing by pausing before publication, but the
scope of required remediation is broader than the current documents state.

Recommendation:

Expand the remediation plan before rerunning experiments. Treat all
physics-forward claims as quarantined until every physics path is either
validated or removed from the paper.

## Big-picture scientific verdict

### What is trustworthy

The following claims are currently supportable:

- The repository contains a synthetic 11-class journal-bearing fault dataset.
- Dataset v2 is balanced by class and severity slot, has split-local metadata,
  and has no exact full-record hash overlap across train/validation/test.
- Most generated fault classes exhibit plausible synthetic journal-bearing
  spectral/statistical signatures.
- Strong vanilla models reach about 96% window-level accuracy on the clean
  synthetic test split.
- Existing physics-labeled models do not demonstrate a clean-accuracy advantage
  over strong vanilla baselines in the stored benchmark results.

### What is not trustworthy

The following claims are not currently supportable:

- That the repository has valid, fully wired physics-informed training for the
  reported PC-CNN/PINN experiments.
- That HybridPINN validates physics metadata for journal-bearing faults.
- That MultitaskPINN results validate multitask physics-informed learning.
- That the "physics family" is more noise robust than vanilla models.
- That window-level statistical tests provide reliable independent-sample
  inference.
- That all paper tables can be reproduced from current HEAD without a pinned
  manifest.

### Publishability verdict

Not publishable as currently framed.

A defensible paper may be possible after remediation, but the framing must be
honest and narrower: a synthetic journal-bearing benchmark with negative or
limited results for physics-informed models unless new reruns with valid
physics losses demonstrate otherwise. Any claim that physics-informed learning
improves accuracy, noise robustness, data efficiency, interpretability, or OOD
severity behavior must be supported by rerun experiments using verified
physics-consistent model code and record-level statistics.

## Prioritized recommendations

1. Freeze claims now. Do not submit or circulate a paper draft that presents the
   current physics-informed experiments as valid evidence.

2. Quarantine invalid physics paths. Mark `PhysicalConstraintLoss`,
   `PINNTrainer` physics mode, rolling-element `BearingDynamics` usage in
   HybridPINN, and stale PINN scripts as non-authoritative until fixed.

3. Implement the ratified differentiable band-energy loss. It must use
   journal-bearing bands, support mixed and broadband classes, accept per-sample
   RPM metadata, and have gradient tests.

4. Rebuild or remove HybridPINN. If retained, replace rolling-element
   BPFO/BPFI/BSF/FTF features with journal-bearing features aligned to the
   generator.

5. Rerun every affected experiment from a clean manifest: physics-loss ablation,
   data efficiency, severity OOD, true-metadata/metadata ablation, XAI and
   calibration if those depend on PC-CNN checkpoints, and any noise-robustness
   claim involving physics-labeled models.

6. Recompute statistical tests at record level. Use 528 test records, not 2640
   correlated windows, as the independent units. Keep window-level metrics only
   as descriptive secondary metrics.

7. Replace family-average claims with fair comparisons. Report per-model curves
   and architecture-matched comparisons; do not average together invalid,
   CE-only, and verified-physics models.

8. Curate the reproduction surface. Provide one frozen command and manifest per
   paper table. Archive or clearly mark stale scripts and Colab helpers that do
   not train the models as their names imply.

9. Add scientific CI tests. Minimum tests should include physics-loss gradient
   flow, class-specific band targeting, HybridPINN feature semantics, trainer
   wiring, metadata alignment, record-level aggregation, and script dry-runs.

10. Strengthen or disclose weak synthetic classes. Cavitation in particular
    needs a per-severity spectral audit or more cautious wording.

## Final conclusion

This repository is closer to being scientifically honest than many research
codebases because the maintainers have already identified some serious defects
and paused for audit. But honesty about a defect is not the same as remediation.

The current data generation is mostly coherent as a synthetic benchmark. The
current physics-informed modeling evidence is not valid enough for publication.
The safest and most accurate near-term conclusion is negative: on this synthetic
dataset, the stored results do not show a verified benefit from
physics-informed neural models, and several physics-labeled experiments were not
actually valid tests of the claimed physics.

