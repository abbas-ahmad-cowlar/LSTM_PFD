# Phase-5 physics reruns — BAND-ENERGY loss vs the frozen healthy reference

> ⚠️ **SUPERSEDED 2026-06-24 — the noise-robustness result in this doc was based on
> n=3 seeds and does NOT replicate at n=12.** A pre-registered n=12 grid (PROTOCOL
> §8.8, `results/noise_seed_robustness/`, analysis `scripts/p7_strengthen_record_level.py`)
> with a matched-strength non-physics control shows correct physics tied with CE-only
> (degr 3.47 vs 3.54, seed-level **Wilcoxon p=0.79**; no arm robust on ≥10/12 seeds).
> This grid's own first three seeds reproduce the n=3 "win" below (4.29 vs 0.06), and
> the other nine erase it. **The "NOISE-ROBUSTNESS survives" verdict here is retired**
> — treat this whole doc as historical provenance for the n=3 analysis. Authoritative
> verdict: `results/FINDINGS.md` §0 (complete negative). The §8.7 F9 control section
> remains valid as far as it goes (wrong real bands reproduced the n=3 effect), but
> the §8.8 random-band control + n=12 supersede it: the effect is not real at n=12.
>
> **What this is.** The §8.2/§8.3/§8.4 experiments rerun with the *corrected*
> physics: journal-bearing signature DB, the differentiable **band-energy
> `compute_physics_loss` judged against the frozen healthy-class reference**, and
> **per-sample rpm**. 42 runs, Colab T4, code @`ce344d1` (`p6/docs`). Replaces the
> contaminated tonal-only/inert results in `results/phase5*`.
>
> **Status: RECORD-LEVEL CONFIRMATION DONE (2026-06-16, P6 Step 5a).** The
> window-level signal below was recomputed at the **record level** (528 records,
> soft-vote per record, cluster-bootstrap + exact McNemar) →
> `summary_record_level.json`. **Verdict (see the "Record-level verdict" section
> below): the NOISE-ROBUSTNESS result survives and is statistically significant;
> the severity-OOD and data-efficiency results do NOT survive as significant.**
> The window-level tables in this section are retained as recorded; the
> record-level numbers are authoritative for any claim. Checkpoints retained
> (`results/band_energy_reruns/**/best_model.pth`, off-git).

## Headline (window-level, mean ± std over 3 seeds)

The corrected loss shows **no clean-accuracy advantage** (the clean benchmark is
near-ceiling) but **emerging benefits in the two stress regimes physics is
theorized to help — noise and severity-OOD** — with **no clean-accuracy cost**.
This is a notable reversal of the contaminated result, which showed physics as
neutral-to-harmful in every regime.

### §8.4 — physics-weight ablation + noise (pc_cnn, clean & 5 dB)

| w (physics) | clean test | 5 dB test | clean→5 dB degradation |
|---|---|---|---|
| 0 (CE-only, Phase-4) | 95.98 ± 0.36 | 90.997 ± 2.90 | 4.99 |
| 0.1 | 95.92 ± 0.19 | 90.77 ± 3.59 | ~5.2 |
| 0.3 | 96.16 ± 0.18 | 89.46 ± 4.27 | ~6.7 |
| **1.0** | 96.15 ± 0.15 | **95.64 ± 0.22** | **0.51** |
| _resnet18 (vanilla, same backbone)_ | _96.14 ± 0.28_ | _94.43 ± 1.42_ | _1.70_ |

- **Clean accuracy is flat (~96%) across all w** — physics costs nothing on clean data.
- **At w=1.0 the model is the most noise-robust of anything tested**: it loses only
  **0.51 pts** at 5 dB (all 3 seeds 95.45/95.53/95.95), beating its CE-only self
  (degr. 4.99) **and** the best vanilla baseline resnet18 (degr. 1.70).
- Low weights (0.1/0.3) are *unstable* under noise (some seeds collapse to ~84-88);
  **the robustness emerges specifically at high physics weight.**
- This **reverses** the contaminated §8.4 finding (tonal/flat loss: "harmful at
  w=1.0, 5 dB 83.1"). The defect, not physics, drove the earlier negative here.

### §8.3 — severity-OOD (held-out-severity test accuracy)

| direction | pc_cnn (band-E, w0.3) | resnet18 (vanilla) |
|---|---|---|
| A: train incipient+mild+mod → test **severe** | 97.32 ± 0.07 | 97.37 ± 0.07 |
| B: train mild+mod+severe → test **incipient** (hard) | **79.04 ± 0.61** | 73.43 ± 3.29 |

- Direction A (extrapolate to *severe*) is easy and **tied**.
- Direction B (extrapolate to *incipient*, near the noise floor) — **physics +5.6
  pts, and far more stable** (std 0.61 vs 3.29). A genuine OOD benefit where it is
  hardest.

### §8.2 — data efficiency (clean test accuracy vs train fraction)

| fraction | pc_cnn (band-E, w0.3) | resnet18 (vanilla) | winner |
|---|---|---|---|
| 10% | 92.60 ± 0.37 | 93.60 ± 0.78 | vanilla (+1.0) |
| 25% | 95.23 ± 0.20 | 94.71 ± 0.42 | pc_cnn (+0.5, overlapping) |
| 50% | 95.72 ± 0.16 | 95.37 ± 0.12 | **pc_cnn (+0.35, non-overlapping)** |
| 100% | 96.20 ± 0.21 | 96.14 (Phase-4) | tied |

- Prereg rule (physics "wins" the regime = ahead at ≥2 fractions, non-overlapping
  ±1σ): met at **only 50%** → by the strict rule, **not a data-efficiency win**.
- But it is **~neutral and much improved** over the contaminated result (tonal/flat
  loss *hurt* at 10%: 91.11 ± 3.29). At low data, band-energy is far more stable
  (std 0.37 vs 3.29) and ahead in the mid regime.

## Record-level verdict (P6 Step 5a, 2026-06-16)

Recompute: `scripts/phase5_bandenergy_record_level.py` → `summary_record_level.json`
(soft-vote the 5 windows of each record; cluster-bootstrap CIs over 528 records;
exact McNemar on the best-val seed). A sanity gate (window-level acc reproduces
each run's recorded `metrics.json`) passed for every re-eval, or the script aborts.

### §8.4 ablation + noise (record level)

| w | clean | 5 dB | degr. |
|---|---|---|---|
| 0 (CE-only) | 98.99 ± 0.45 | 94.70 ± 3.08 | 4.29 |
| 0.1 | 99.12 ± 0.09 | 94.13 ± 2.96 | 4.99 |
| 0.3 | 99.68 ± 0.09 | 93.75 ± 3.95 | 5.93 |
| **1.0** | 98.61 ± 0.62 | **98.55 ± 0.76** | **0.06** |
| _resnet18 (vanilla)_ | _99.18 ± 0.09_ | _97.66 ± 1.30_ | _1.52_ |

- **w=1.0 vs CE-only (same architecture) @5 dB — McNemar 14–0, p = 1.2e-4.**
  Representative best-val-seed gap **+2.65 pts, CI95 [1.33, 4.17]** (excludes 0).
- **w=1.0 vs resnet18 @5 dB — McNemar 6–0, p = 0.031.** Representative gap
  **+1.14 pts, CI95 [0.38, 2.08]** (excludes 0) — but **fragile/near-ceiling**: vs
  resnet's best-*noise* seed it is 5–0, **p = 0.0625** (not significant). Secondary
  to the same-architecture ablation.

> **Estimand note (audit 2026-06-16 F6):** the paired gap, CI, and McNemar above
> are all the **representative best-val seed** (one estimand). The **seed-mean**
> gaps (w0: +3.85, resnet: +0.88) are descriptive only and must NOT be paired with
> these representative-seed CIs. `summary_record_level.json` now reports both
> separately (`repseed_gap_pts` / `seedmean_gap_pts`).

### §8.2 data efficiency (record level, clean acc)

| frac | pc_cnn (w0.3) | resnet18 | gap |
|---|---|---|---|
| 10% | 96.28 ± 1.39 | 96.28 ± 1.10 | 0.00 (tied) |
| 25% | 98.48 ± 0.15 | 97.22 ± 0.62 | +1.26 (non-overlapping) |
| 50% | 98.80 ± 0.79 | 98.80 ± 0.24 | 0.00 (tied) |
| 100% | 99.49 ± 0.32 | 99.18 ± 0.09 | +0.31 |

### §8.3 severity-OOD (record level)

| direction | pc_cnn (w0.3) | resnet18 | McNemar | gap CI95 |
|---|---|---|---|---|
| A → severe | 100.00 ± 0.00 | 100.00 ± 0.00 | 1.0 | [0.00, 0.00] |
| B → incipient | 82.58 ± 0.62 | 76.01 ± 3.41 | **0.388** | **[−2.27, +8.33]** |

### Verdict — what survives at the record level

- **NOISE ROBUSTNESS — SURVIVES (significant).** w=1.0 degrades **0.06 pt** vs the
  identical-architecture CE-only model's **4.29 pt**. Representative best-val seed:
  **McNemar 14–0, p=1.2e-4, gap +2.65 [1.33, 4.17]** (mechanism: w=1.0 rescues 14
  noisy `lubrification` records CE-only mislabels `mixed_wear_lube`). This is the
  one defensible C3 positive: correctly-implemented journal-bearing physics at high
  weight earns a **real noise-robustness benefit**, strongest as a same-architecture
  ablation; it reverses the contaminated "harmful at w=1.0 (83.1)" negative. vs the
  best vanilla resnet18 it is significant but **small/fragile** (6–0, p=0.031, gap
  +1.14 [0.38, 2.08]; flips to p=0.0625 vs resnet's best-noise seed) → secondary.
- **SEVERITY-OOD — DOES NOT SURVIVE as significant.** Dir A tied at the ceiling
  (100%). Dir B favors physics (seed-mean 82.58 vs 76.01; representative-seed gap
  +3.03) and is low-variance, but **McNemar p=0.39** and the representative gap CI
  **[−2.27, +8.33] spans zero** (too few incipient records to resolve the gap).
  Direction-only / suggestive; **not a claim.**
- **DATA-EFFICIENCY — NEUTRAL, no win.** Ahead non-overlapping at only 1 of 3
  reduced fractions (25%) → fails the prereg rule. The contaminated "hurts at 10%"
  is gone (now exactly tied). No advantage, no harm.
- **Caveat:** at the record level w=1.0's *clean* accuracy (98.61) is ~0.4–1 pt
  below the lower weights (98.99–99.68) — so the "no clean cost" softens to a
  marginal, near-ceiling clean cost traded for large noise robustness. State it.
- **Soft-voting compressed** the cross-model margins toward the ceiling (vs-vanilla
  noise gap 1.21→0.88; data-eff 10% edge → tie) and **widened the OOD CI through
  zero** — but the within-architecture noise robustness is *stronger* at record
  level (degradation 0.51→0.06).

## F9 control — physics-specific or generic regularization? (§8.7, 2026-06-22)

The surviving §8.4 w=1.0 noise result was tested against the pre-registered
**scrambled-reference control** (PROTOCOL §8.7): the same band-energy loss at
w=1.0, but each fault judged against a **different** fault's bands + healthy
reference (fixed derangement `[0,10,5,9,6,2,8,4,7,1,3]`) — identical loss
strength/structure, **wrong physics**. 3 seeds, Colab T4 @`70f623f`. Record-level
recompute: `scripts/f9_scramble_record_level.py` → `f9_scramble_record_level.json`.

| arm (record level, 528) | clean | 5 dB | degradation |
|---|---|---|---|
| w=0 CE-only | 98.99 ± 0.45 | 94.70 ± 3.08 | 4.29 |
| **w=1.0 correct physics** | 98.61 ± 0.62 | **98.55 ± 0.76** | **0.06** |
| **w=1.0 SCRAMBLED** | 98.42 ± 0.91 | **95.58 ± 5.31** | **2.84** |

Representative best-val seed (seed2 all arms), exact McNemar @5 dB:
- scramble vs CE-only: **14–1, p = 9.8e-4** (scramble significantly more robust than plain CE).
- scramble vs correct-w1.0: **0–1, p = 1.0** (at the representative seed, indistinguishable from correct physics).

### Verdict (§8.7 decision rule): LARGELY GENERIC REGULARIZATION — not physics-specific.

Scrambling the per-class physics did **not** destroy the robustness: at the
representative (best-val) seed the scrambled model is **as robust as correct
physics** (0–1, p=1) and **beats CE-only** (14–1, p≈1e-3); 2 of its 3 seeds are
robust. So a **high-weight band-energy regularizer produces the noise robustness
even with the WRONG per-class targets** — the effect is **not specific to correct
journal-bearing physics**.

What correct physics *does* buy is **stability**: all 3 correct-physics seeds are
robust (5 dB std **0.76**), whereas the scramble is **seed-fragile** (std **5.31** —
one seed collapses ~13 pt), giving a seed-mean degradation of 2.84 vs correct's
0.06. Correct per-class targets make the robustness *reliable*, but are not what
*creates* it.

**Consequence (the §8.7 rule mandates the conservative reading for an intermediate
result):** the surviving benefit must be reported as **"a high-weight band-energy
spectral-consistency regularizer improves 5 dB noise robustness in a
same-architecture ablation"** — a **spectral-regularization** effect, **NOT**
evidence that correct physics priors help (correct physics only improves cross-seed
stability). This **confirms the FINDINGS draft's narrow wording**; the word
"physics" may **not** be claimed for the noise result.

## Was this expected?

**Partly — and it shifts the story.** We expected no clean-accuracy gain (the
benchmark is near-ceiling; confirmed). We did **not** expect the strong, stable
**noise robustness at high physics weight** — that is new, and it is exactly the
kind of benefit physics priors are supposed to confer. The earlier "decisive
negative across all regimes" now looks substantially like an artifact of the
broken loss (inert → tonal-only → flat baseline). With the loss correctly
implemented (band-energy vs the real healthy reference, per-sample rpm), physics
earns its keep in the **noise** regime (record-level significant, same-architecture
ablation) while staying ~free on clean accuracy. The **severity-OOD** signal did
**not** survive record-level significance (direction-only), and data-efficiency is
neutral — see the record-level verdict above.

## What it does NOT yet establish (guardrails)

- The record-level recompute (above) is now **authoritative**. The only surviving
  physics-benefit claim is **noise robustness** (§8.4). **Severity-OOD (§8.3) and
  data-efficiency (§8.2) did NOT survive** as significant record-level results —
  report direction-only / neutral, not as benefits.
- §8.5 (HybridPINN) is **not** included — its physics branch is still
  rolling-element (quarantined). No multi-mechanism claim.
- Single dataset, synthetic-only; n=3 seeds; the vs-vanilla 5 dB edge and the
  surviving noise result are near-ceiling (the discordant-record count is small).

## Next steps

1. **Record-level recompute** of §8.2/§8.3/§8.4 — **DONE 2026-06-16**
   (`scripts/phase5_bandenergy_record_level.py` → `summary_record_level.json`;
   soft-vote, cluster-bootstrap CIs, McNemar). Verdict in the section above.
2. **§8.6a XAI recompute** against the corrected bands (does the w=1.0 model attend
   to the characteristic frequency bands more?) — owner-gated, NEXT after sign-off.
3. **Rewrite + re-ratify FINDINGS** (owner-gated) — the verdict moves from
   "no physics advantage" to **"no clean-accuracy gain, but a real, statistically
   significant NOISE-ROBUSTNESS benefit at high physics weight (strongest as a
   same-architecture ablation) from correctly-implemented journal-bearing
   physics"** — with severity-OOD and data-efficiency reported as direction-only /
   neutral (they did not survive record-level significance).
4. Optionally rebuild §8.5 HybridPINN on journal-bearing features (separate task).

_Source runs: `results/band_energy_reruns/{pinn_ablation,data_efficiency,severity_ood}/`
(42 metrics.json + checkpoints). Generated from code @`ce344d1`._
