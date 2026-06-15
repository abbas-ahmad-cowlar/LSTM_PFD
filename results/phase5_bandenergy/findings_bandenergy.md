# Phase-5 physics reruns — BAND-ENERGY loss vs the frozen healthy reference

> **What this is.** The §8.2/§8.3/§8.4 experiments rerun with the *corrected*
> physics: journal-bearing signature DB, the differentiable **band-energy
> `compute_physics_loss` judged against the frozen healthy-class reference**, and
> **per-sample rpm**. 42 runs, Colab T4, code @`ce344d1` (`p6/docs`). Replaces the
> contaminated tonal-only/inert results in `results/phase5*`.
>
> **Status: WINDOW-LEVEL, 3 seeds — a promising signal, NOT yet a claim.** Per the
> guardrails, no physics-benefit claim is made until these are recomputed at the
> **record level** (528 records) with McNemar. The checkpoints are retained
> (`results/phase5_bandenergy/**/best_model.pth`, off-git) for that recompute.

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

## Was this expected?

**Partly — and it shifts the story.** We expected no clean-accuracy gain (the
benchmark is near-ceiling; confirmed). We did **not** expect the strong, stable
**noise robustness at high physics weight** — that is new, and it is exactly the
kind of benefit physics priors are supposed to confer. The earlier "decisive
negative across all regimes" now looks substantially like an artifact of the
broken loss (inert → tonal-only → flat baseline). With the loss correctly
implemented (band-energy vs the real healthy reference, per-sample rpm), physics
earns its keep in the **noise** and **severity-OOD** regimes while staying free on
clean accuracy.

## What it does NOT yet establish (guardrails)

- All numbers are **window-level**, **n=3 seeds**. The headline (§8.4 w=1.0 noise;
  §8.3 dir B) **must be confirmed at the record level (528 records) with McNemar**
  before any physics-benefit claim. Until then this is a *promising signal*.
- §8.5 (HybridPINN) is **not** included — its physics branch is still
  rolling-element (quarantined). No multi-mechanism claim.
- Single dataset, synthetic-only.

## Next steps

1. **Record-level recompute** of §8.2/§8.3/§8.4 from the retained checkpoints
   (per `scripts/aggregate_benchmark_record_level.py`'s method): per-record
   soft-vote, cluster-bootstrap CIs, McNemar (w=0 vs w=1.0 at 5 dB; pc_cnn w=1.0 vs
   resnet18 at 5 dB; dir B). This confirms or retracts the headline.
2. **§8.6a XAI recompute** against the corrected bands (does the w=1.0 model attend
   to the characteristic frequency bands more?).
3. **Rewrite + re-ratify FINDINGS** — if record-level holds, the verdict moves from
   "no physics advantage" to "no clean-accuracy gain, but a real noise-robustness
   and severity-OOD benefit from correctly-implemented journal-bearing physics."
4. Optionally rebuild §8.5 HybridPINN on journal-bearing features (separate task).

_Source runs: `results/phase5_bandenergy/{pinn_ablation,data_efficiency,severity_ood}/`
(42 metrics.json + checkpoints). Generated from code @`ce344d1`._
