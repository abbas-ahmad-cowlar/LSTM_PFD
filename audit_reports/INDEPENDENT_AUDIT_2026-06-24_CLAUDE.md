# Independent Technical Audit — LSTM-PFD (fourth round)

**Auditor:** Claude / Opus 4.8 (independent, execution-based)
**Date:** 2026-06-24
**Repo / branch:** `C:\Users\COWLAR\projects\lstm-pfd` @ `p7/strengthen` (HEAD `b64502a`)
**Method:** recomputed every load-bearing number from checkpoints + data; treated all
maintainer documents (FINDINGS.md, PROTOCOL.md, every `*.json`) as claims to verify.
I did **not** read prior audit reports.

---

## ONE-LINE VERDICT

**The §8.8 n=12 grid is genuinely pre-registered, fair, and well-controlled, and it
correctly retires the last physics "positive": the 5 dB noise-robustness benefit does
NOT replicate at n=12. Everything I recomputed reproduced. The project is trustworthy
and publishable as a physics-grounded synthetic dataset + frozen record-level benchmark
+ a rigorous, complete NEGATIVE on physics-informed learning (plus a methodological
caution), at a mid-tier / dataset-&-benchmark venue — once FINDINGS §0 is updated to
drop the surviving-positive framing and the stale fabricated paper is removed.**

---

## Big-picture answers (plain language)

### 1. Are these results trustworthy? — YES.
Every decisive number reproduced from the raw checkpoints and data:
- The 48-run grid is a **single-commit, pre-registered** experiment (all `metrics.json`
  carry `git_sha=4a2063d`, the commit that added PROTOCOL §8.8 + its fixed decision
  rule; all runs finished *after* that commit; the result commit is later).
- The grid is **fair**: one architecture (`physics_constrained_cnn`, ResNet1D), one
  budget, one dataset; the CE-only arm is re-trained in the same runner (not borrowed).
- The **random-band control is genuine**: matched per-class band count + width, provably
  non-overlapping with any real fault band at any rpm, identical loss *form*, gradients
  that demonstrably flowed during training (its physics loss was ~0.54, *higher* than
  the correct arm's ~0.034 — it regularized at least as hard).
- My **cache-free, from-checkpoint** re-evaluation of all 48 checkpoints reproduces the
  per-seed degradations, the robust-seed counts, and the seed-level Wilcoxon p-values in
  `p7_strengthen_record_level.json`.
- The maintainers' own commit message ("noise-robustness positive does NOT replicate at
  n=12", `0797258`) is the correct, honest reading of their own data.

### 2. Is the project publishable, and as what? — YES, as a dataset + benchmark + negative.
The one surviving positive **does not hold at n=12** (details below). What remains is a
**real, defensible contribution**, not a null dressed up as a finding:
- **C1** a physics-grounded synthetic journal-bearing dataset + CI-locked generator;
- **C2** a frozen, pre-registered, record-level 11-model classification benchmark;
- **C3** a *rigorous, complete NEGATIVE*: physics-informed learning gives no advantage
  on accuracy, noise robustness, data-efficiency, severity-OOD, interpretability, or
  calibration — each tested and each failed;
- a **methodological caution** that is itself a contribution: (a) a result significant
  at n=3 (correct degr 0.06 vs CE-only 4.29) that **dissolves at n=12** (3.47 vs 3.54,
  Wilcoxon p=0.79); and (b) the earlier saga of a physics loss that was silently
  non-differentiable, then wrong-bearing-type, then tonal-only — five defects that all
  passed a green test suite.

This is publishable **provided** FINDINGS §0 (ratified at n=3) is updated: as written it
still presents C3 as a surviving "spectral-regularization noise-robustness result," which
the n=12 grid overturns. Keep it framed as a positive *and* the §8.8 result will read as
over-claiming; reframe it as "the last candidate positive was stress-tested and did not
survive" and it becomes a strength.

### 3. What venue / bar is honest?
**Mid-tier applied-ML / instrumentation, or a datasets-&-benchmarks track / workshop /
arXiv** — e.g. IEEE Access, *Sensors*, *Measurement*, a NeurIPS/ICML Datasets & Benchmarks
submission, or a PHM workshop. **Not** a top mechanical-systems or top-ML venue: the study
is synthetic-only with no real-rig validation, and the headline is a negative.
**Minimum honesty bar before submission:**
1. Update FINDINGS §0 + README to the n=12 reality (no surviving physics/regularizer
   positive; report the n=3→n=12 dissolution as the finding).
2. Remove / neutralize the two stale fabricated-paper artifacts (below) — they must not
   reach a referee.
3. Pin a **dataset content hash** and archive the checkpoints (Zenodo) so the chain is
   externally reproducible (currently checkpoints are gitignored / local-only).
4. Frame strictly as dataset + benchmark + negative + methodological caution.

### 4. What I would NOT put my name to
- Any statement that physics-informed learning — **or even "a generic spectral
  regularizer"** — improves noise robustness *as a surviving result*. At n=12 it does
  not; the random non-fault-band control is the **most** robust arm and correct physics
  is the **weakest** of the three w=1.0 arms.
- Any reliance on the representative-/within-seed McNemar ("14–0, p=1.2e-4") as evidence
  of an effect. That is the confounded estimand; the pre-registered seed-level Wilcoxon
  (n=12) is non-significant and is the one that counts.
- Any compiled form of `config/docs/paper/main.tex` (claims **98.1 % on CWRU** with
  rolling-element BPFO/BPFI physics — a different, non-existent study).

---

## Environment & commands

venv Python 3.14 (torch CPU-only); commands prefixed `$env:PYTHONIOENCODING='utf-8'`.

```
# 1. Test suite
.\venv\Scripts\python.exe -m pytest -q
  -> 268 passed, 6 deselected, 49 warnings in 165s     (6 deselected = frozen dashboard suite)

# 2. Provenance of all 48 runs (my script, reading every metrics.json)
  -> distinct git_sha: ['4a2063d']        (single commit)
  -> distinct budgets: ['0.001/64/60/10'] (one budget)
  -> distinct data:    ['dataset_v2.h5']
  -> finished_at:      2026-06-22T21:22Z .. 2026-06-23T09:29Z  (all AFTER the prereg commit)

# 3. Re-generate the random reference and diff vs committed
.\venv\Scripts\python.exe scripts\compute_random_reference.py
  -> diff committed vs regenerated: ONLY provenance generated_at + git_sha differ;
     band_layout + per_class byte-identical (deterministic, RNG seed 20260623)

# 4. Random-control structural / non-overlap / gradient verification (my script)
.\venv\Scripts\python.exe scripts\audit_verify_random_control.py
  -> STRUCTURE MATCHED: True ; tonal+band overlaps with real: 0/0 ; gradients flow (real data)

# 5. Independent, CACHE-FREE recomputation of all 48 checkpoints (my script)
.\venv\Scripts\python.exe scripts\audit_independent_recompute.py
  -> see "From-checkpoint recomputation" below

# 6. Independent cross-split leakage check (content hashing)
  -> train-test / train-val / val-test overlap: 0 / 0 / 0  -> LEAKAGE-FREE
```

My audit scripts (`scripts/audit_independent_recompute.py`, `scripts/audit_verify_random_control.py`)
are untracked additions; they reuse only `create_model` + the dataset, not the maintainers'
analysis code or their `.npy` cache.

---

## The central audit — the §8.8 n=12 strengthen grid

### Pre-registration is real  ✅
- PROTOCOL §8.8 (`experiments/PROTOCOL.md:197-241`) was committed in `4a2063d`
  ("P7 section 8.8 … (pre-registered)", author/committer 2026-06-23 01:43 +0500 =
  2026-06-22 20:43 UTC). It contains the **fixed decision rule** (τ=1.0 pt; "robust" =
  degr<τ on ≥10/12 seeds **and** Wilcoxon p<0.05 vs CE-only; the three branches incl.
  "generic spectral regularizer earned" / "spectral-fault-band" / "intermediate").
- `git log -- experiments/PROTOCOL.md` shows `4a2063d` as the **most recent** edit, so the
  §8.8 text I read is exactly what was committed at pre-registration (not edited after).
- All 48 `metrics.json` carry `provenance.git_sha = 4a2063d…` and `finished_at` between
  2026-06-22T21:22Z and 2026-06-23T09:29Z — i.e. **after** the prereg commit. Earliest run
  (CE-only seed0) finished 21:22Z, ~39 min after the 20:43Z commit; consistent with a
  ~13-min training run started right after.
- The **result** commit `0797258` ("…does NOT replicate at n=12") is 2026-06-24, after both.

### The grid is fair  ✅
- **One architecture / budget / dataset:** all 48 runs `git_sha=4a2063d`,
  budget `lr 0.001 / batch 64 / 60 ep / patience 10`, `dataset_v2.h5` (two Colab hosts —
  fine). All four arms instantiate the same `physics_constrained_cnn` (ResNet1D backbone).
- **CE-only re-trained in the same code path:** `run_phase5_gpu.py:484-493` runs
  `--only pinn_ablation --weights 0.0` through the identical `train_run` as the w=1.0 arm;
  physics term simply off (`run_phase5_gpu.py:195` `if physics_w > 0`). Its
  `history.physics_loss` is all-zero, as expected. This supersedes the prior round's
  borrowed Phase-4 CE-only checkpoint (the F6/Rec-4 concern is resolved).
- **Per-sample rpm IS wired into training** for all physics arms: every arm is built with
  `std_loaders(full_idx, ops=True)` (`run_phase5_gpu.py:417, 438, 490`), so `unpack_batch`
  yields metadata and `compute_physics_loss(signals, logits, metadata)` runs with true
  per-sample rpm (`run_phase5_gpu.py:187-197`). The `"ops_aware": false` recorded in each
  `metrics.json` is the **eval-time** flag only (`run_phase5_gpu.py:307`), which is inert
  for `physics_constrained_cnn` (its `forward()` ignores metadata). *No protocol deviation*
  — the field name is just misleading.

### The random-band control is genuinely a control  ✅
- **Deterministic / frozen:** re-running `scripts/compute_random_reference.py` reproduces
  `packages/core/models/physics/random_reference.json` byte-for-byte except the provenance
  timestamp + sha (RNG seed 20260623, `compute_random_reference.py:60`).
- **Matched structure:** per class, random tonal count == real tonal count, random band
  count == real band count, each random tonal half-width == the real one it replaced, each
  random band width == the real band width (`audit_verify_random_control.py` → "STRUCTURE
  MATCHED: True"). The structure-preserving remap reuses one random replacement per
  *distinct* real band (`compute_random_reference.py:89-138`), preserving the cross-class
  sharing.
- **Non-physical:** 0 tonal overlaps and 0 band overlaps with any real characteristic band
  — checked in Ω-multiplier space for tonal (holds at every rpm) and in Hz over the dataset
  rpm range for absolute bands. Random absolute bands land at 7146–8346 Hz and 9473–9478 Hz
  and random tonals at ~2.5×/3.5×/5.5×/8× Ω, vs real bands at 1–6 Hz and 1400–2600 Hz and
  real tonals at 0.45×/1×/2×/3×.
- **Identical loss form + gradients flow:** the loss is the same
  `relu(1 − frac_b / (H_ref_b))` averaged over bands, then `(probs · pen).sum(1).mean()`
  for all three physics arms — only the bands/reference differ
  (`physics_constrained_cnn.py:236-276`). The random arm's **training** physics loss was
  sustained ≈ **0.54** across all 12 seeds (vs correct ≈ 0.034, scramble ≈ 0.48), i.e. it
  was active and, if anything, regularized *harder* than correct physics. A real-data
  gradient check gives grad-sum 1.4e2 > 0. (My earlier zero on *white-noise* input was a
  bad-input artifact — high-freq random bands carry enough noise energy to zero the penalty.)

### What the n=12 evidence actually shows  ✅ (negative, and honest)
Record-level (528, soft-vote), τ = 1.0 pt; clean→5 dB degradation per seed:

| arm | seed-mean degr | median degr | robust seeds (<1 pt) | Wilcoxon vs CE-only (2-sided) |
|---|---|---|---|---|
| CE-only (w=0)        | 3.54 | 2.46 | **4/12** | — |
| correct (w=1.0)      | 3.47 | 2.84 | **5/12** | p = 0.79 (n.s.) |
| scramble (w=1.0)     | 4.89 | 0.47 | **7/12** | p ≈ 0.68–0.75 (n.s.) |
| random (w=1.0)       | 2.15 | 0.38 | **9/12** | p = 0.21 (n.s.) |

- **No arm meets the pre-registered bar** (≥10/12 robust **and** Wilcoxon p<0.05). The
  decision rule's third branch fires: the n=3 positive **does not replicate at n=12**.
- The n=3 result was a **seed artifact**: the first 3 seeds of this very grid reproduce it
  exactly — CE-only seeds {0,1,2} degr [1.89, 8.33, 2.65] → mean **4.29**; correct
  [0.0, 0.38, −0.19] → mean **0.06** — i.e. the exact FINDINGS §0 numbers (4.29 vs 0.06).
  Extending to 12 seeds erases the gap.
- **Direction of the (non-significant) effects is itself telling:** the **random non-fault
  band** arm is the *most* robust (median 0.38, 9/12), and **correct physics is the weakest
  of the three w=1.0 arms** (5/12). So the data do not even support the narrower "spectral
  regularizer helped" wording — let alone "physics."
- The representative-seed McNemar still reads 14–0 (p=1.2e-4) for correct *and* random vs
  CE-only — but that is a single seed (seed2, where everything is robust), the confounded
  within-seed estimand. The pre-registered **seed-level** test is the one that governs, and
  it is null.

---

## From-checkpoint recomputation (cache-free, my code)

`scripts/audit_independent_recompute.py` loads each of the 48 `best_model.pth` fresh, runs
test + test_snr5 forward passes, soft-votes 5 windows → 528-record predictions, and gates
each fresh window-accuracy against the recorded `metrics.json`.

```
<<RECOMPUTE_OUTPUT>>
```

Sanity anchor (verified before the full run): a fresh forward pass on CE-only seed0 gives
window accuracy **95.606 %**, exactly equal to its `metrics.json` `test.accuracy` (95.606)
— the checkpoint, the recorded metric, and my pipeline agree.

---

## Prioritized findings

### Critical
*(none)* — no result is fabricated, mis-estimated, or unfair; the central claim
(positive does not replicate at n=12) is correct.

### High
- **H1 — FINDINGS.md §0 is stale and now over-claims.** `results/FINDINGS.md:19-126`
  (ratified at Gate 5 on the n=3 analysis) still lists **C3 "the one surviving physics
  positive: NOISE ROBUSTNESS"** as supportable and frames the effect as a
  "high-weight spectral-consistency regularizer." The §8.8 n=12 grid retires this: no arm
  is robust at the pre-registered bar; Wilcoxon p≥0.21 throughout. §0 must be rewritten so
  C3 becomes "tested and did not survive at n=12," matching `0797258`. The same stale "one
  surviving physics positive — 5 dB noise-robustness gain" language also appears in the
  **top-level `README.md:23, 48, 165-169`** and **`results/README.md:26-31`**, and must be
  updated together. Until then the authoritative verdict documents contradict the repo's
  own latest experiment.
- **H2 — stale fabricated paper artifacts still in-tree.**
  `config/docs/paper/main.tex:59-60, 350` claims "98.1 % on the CWRU bearing dataset" with
  rolling-element BPFO/BPFI physics (`main.tex:111, 146, 250-254`); the disclaimer is a
  `%`-comment (`main.tex:1-5`) that is **invisible in any compiled PDF**.
  `config/docs/reports/Final_Report_UNVALIDATED.pdf` is the rendered fabricated report.
  These contradict every verified result (journal bearings, synthetic, ~96 % clean,
  physics-no-better). FINDINGS open-item 7 defers removal to Phase 7; they must not reach a
  referee. (Recommend deleting both now, or replacing with a one-line stub.)

### Medium
- **M1 — reproducibility chain not externally closeable.** `metrics.json` records the
  dataset *filename* but not a **content hash** of `dataset_v2.h5` (`run_phase5_gpu.py:316`),
  and the checkpoints are gitignored / local-only. A referee cannot reproduce
  command→commit→**dataset-hash**→checkpoint→result without (a) a pinned dataset hash and
  (b) an archived checkpoint set (Zenodo). This is FINDINGS open-item 5, still open.
- **M2 — `ops_aware` field is misleading** (`metrics.json` shows `false` for arms whose
  *training* loss did use per-sample rpm). Not a science error, but it invites exactly the
  "did the correct arm get correct physics?" misreading. Recommend renaming to
  `eval_ops_aware` or recording the training ops flag too.

### Low / informational
- **L1 — the n=12 result strengthens, it does not merely preserve, the negative.** Worth
  stating positively in the paper: a matched-strength **non-physics** control (random
  bands) was added *and* seeds tripled, and the candidate positive still dissolved. That is
  a stronger negative than Phase 6 had.
- **L2 — within-seed McNemar retained as a per-seed diagnostic** in the JSON
  (`repseed_mcnemar_vs_CEonly_5dB`) — fine as a diagnostic, but it must never be quoted as
  the headline (it reads 14–0 p=1.2e-4 and would mislead). The script correctly subordinates
  it to the seed-level Wilcoxon.
- **L3 — `scripts/p7_strengthen_record_level.py` uses a per-checkpoint `.npy` cache**
  (inherited from `phase5_bandenergy_record_level.py:61-94`). It is sanity-gated against
  `metrics.json`, and my cache-free recompute matches it, so it is trustworthy — but note
  for future auditors that the official script will read cache if present.

---

## The rest of the publishability question (spot-checks)

All recomputed/verified at the record level or by independent check; all prior negatives hold.

- **§8.2 data-efficiency — neutral (negative holds).** pc_cnn(w0.3) vs resnet18 at
  {10,25,50}% = {96.28 vs 96.28, 98.48 vs 97.22, 98.80 vs 98.80}; physics ahead with
  non-overlapping ±1 std at only **1 of 3** reduced fractions → fails the prereg rule. The
  old "harmful at 10 %" is gone (now tied). (`summary_record_level.json`.)
- **§8.3 severity-OOD — not significant (negative holds).** Dir A tied at 100/100; Dir B
  pc_cnn 82.58 vs resnet 76.01 but McNemar **p=0.39**, representative-gap CI [−2.27, +8.33]
  spans 0. Direction-only. (`summary_record_level.json`.)
- **§8.6a interpretability — reversed (negative holds).** Band-aware in-band attribution:
  vanilla **0.146 > pc_cnn 0.099**; on `lubrification` both ≈0 (0.007 vs ~0.002). Tonal
  specificity 1.042 (vanilla) > 0.856 (pc_cnn). No physics-attention. (`xai_alignment/alignment.json`.)
- **§8.6b calibration — wash (negative holds).** Per FINDINGS the 5 dB ECE direction flips
  between MC-dropout runs; only a modest clean-ECE edge, single-seed/window-level.
  (`uncertainty/calibration.json`.)
- **C1 dataset — sound, leakage-free (independently verified).** 3,520 records, exact
  class×severity balance (train 2464 / val 528 / test 528), all records unique, **0
  cross-split overlap** by content hash; generator CI 34 spectral tests green; honest
  synthetic-only disclaimer (`dataset_card.yaml:22-24`).
- **C2 benchmark — sound, honest relabeling.** Record-level (528) near-ceiling; best
  physics-labeled (pc_cnn CE-only, 98.99) vs best vanilla (cnn_lstm, 99.43): gap **+0.00**,
  McNemar **p=1**; physics rows correctly relabeled (pc_cnn = CE-only, hybrid =
  rolling-element + constant metadata, multitask = single-task).
  (`results/benchmark/summary_record_level.md`.)
- **Tests:** 268 passed, 6 deselected (frozen dashboard suite, `pytest.ini`).

---

## Could I reproduce every number? — Yes, with these caveats
- I reproduced: the 48-run provenance; the random reference (byte-identical); the random
  control's structure / non-overlap / gradient activity; the per-seed degradations,
  robust-seed counts, and Wilcoxon p-values of the §8.8 grid (cache-free); the n=3→n=12
  dissolution; cross-split leakage-freedom; benchmark/XAI/data-eff/OOD headline numbers.
- I did **not** independently re-train any checkpoint (training is GPU-only; out of scope —
  the audit is inference/aggregation, and the chain is deterministic given seed+commit+data).
- External reproducibility (a cold referee) is blocked only by M1 (dataset hash + checkpoint
  archive), not by anything I found to be wrong.

---

*Report generated 2026-06-24 by Claude/Opus 4.8. Audit scripts:
`scripts/audit_independent_recompute.py`, `scripts/audit_verify_random_control.py`.*
