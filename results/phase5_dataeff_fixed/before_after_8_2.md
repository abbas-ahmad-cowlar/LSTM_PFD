# §8.2 data-efficiency — fixed physics loss (the low-data test)

The one principled "fair shot" for physics: low data is the regime where
physics priors are theoretically most valuable, and §8.4 only tested full data.
pc_cnn re-run with the differentiable loss (branch `p5/physics-loss-fix`
@d39c219) at {10,25,50,100}% × 3 seeds, vs the inert "before" (`results/phase5/`)
and vs vanilla resnet18 (reproduced here as a cross-check — identical to before,
since resnet18 never uses the physics loss).

## Test accuracy vs train fraction (mean ± std, 3 seeds)

| frac | pc_cnn FIXED | pc_cnn inert (before) | resnet18 vanilla |
|---|---|---|---|
| 10% | **91.11 ± 3.29** | 93.55 ± 0.61 | 93.60 ± 0.96 |
| 25% | 95.11 ± 0.10 | 94.99 ± 0.35 | 94.71 ± 0.52 |
| 50% | 95.54 ± 0.17 | 95.48 ± 0.21 | 95.37 ± 0.15 |
| 100% | 96.12 ± 0.38 | 95.98 ± 0.45 | 96.14 (Phase-4) |

pc_cnn FIXED 10% per-seed: **91.89 / 87.50 / 93.94** (one seed crashed ~6 pts;
the inert version was 93.55 ± 0.61, rock-steady).

## What we expected vs what we got

- **Expected (prereg §8.2)**: physics_constrained_cnn loses *less* accuracy than
  resnet18 as data shrinks — physics priors should compensate for scarce data.
- **Got**: the opposite at the extreme. At 10% the fixed physics loss made pc_cnn
  **worse than both** the inert version and vanilla (91.11 vs 93.60), with 3–5×
  the seed variance. At 25/50/100% it's within noise of vanilla.
- **Pre-registered decision rule** (physics wins the regime if mean > vanilla at
  ≥2 of 3 reduced fractions with non-overlapping ±1 std): physics wins **0 of 3**.
  At 25% and 50% pc_cnn's mean is marginally higher but the std bands overlap;
  at 10% it loses outright. **Hypothesis REJECTED.**

## What it means

This is the decisive test for the physics-advantage claim (C3): low data is
where physics-informed learning is *supposed* to shine, and a correctly-
implemented, gradient-carrying physics loss not only failed to help — it
**degraded low-data performance and destabilized training** (one seed −6 pts).
Mechanistic reading consistent with §8.4: the frequency-consistency penalty adds
a competing objective that, with few samples, pulls the model toward sparse/
noisy spectral peaks instead of letting it fit the limited data — more
constraints + less data = higher variance, worse fit. The data-driven CNN
already captures the discriminative structure; the physics prior is redundant at
best, harmful at worst.

## Bottom line for the paper (C3)

C3 is now a **complete, decisive negative** across every regime tested:
noise (§8.1, fragile family edge only) · data-efficiency (§8.2, no help, hurts
at 10%) · severity-OOD (§8.3, split/architecture) · weight ablation (§8.4,
neutral-to-harmful) · metadata (§8.5, null). Physics-informed learning does not
beat data-driven on clean synthetic journal-bearing data — tested rigorously,
including in its most favorable regime. The contribution shifts to the honest
negative + the C4 interpretability gain (§8.6) + the C2 benchmark / C1 dataset.
