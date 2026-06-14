# §8.4 PINN ablation — before vs after the differentiable-loss fix

Physics-weight sweep on `physics_constrained_cnn`, frozen budget, seeds {0,1,2},
clean test + 5 dB test. "Before" = inert argmax loss (`results/phase5/`);
"after" = differentiable softmax-weighted loss (branch `p5/physics-loss-fix`
@e894389, `results/phase5_fixed/`). w=0 = Phase-4 pure-CNN baseline
(`results/noise_robustness/`).

## Accuracy vs physics weight (mean ± std over 3 seeds)

| w | clean — before | clean — after | 5 dB — before | 5 dB — after |
|---|---|---|---|---|
| 0 (physics off) | 95.98 ± 0.36 | — (baseline) | 91.00 ± 2.90 | — (baseline) |
| 0.1 | 95.99 (inert) | **96.11 ± 0.44** | 91.00 (inert) | **92.54 ± 5.16** |
| 0.3 | 95.99 (inert) | **96.12 ± 0.38** | 91.00 (inert) | **92.21 ± 6.11** |
| 1.0 | 95.99 (inert) | **95.68 ± 0.50** | 91.00 (inert) | **83.12 ± 12.02** |

Per-seed 5 dB (after-fix), showing the instability:

| w | seed0 | seed1 | seed2 |
|---|---|---|---|
| 0.1 | 86.59 | 95.15 | 95.87 |
| 0.3 | 85.15 | 95.91 | 95.57 |
| 1.0 | 82.05 | **71.67** | 95.64 |

## What we expected vs what we got

- **Expected (prereg §8.4 hypothesis)**: physics loss contributes measurably;
  w=0 underperforms w>0 under stress (5 dB) even if tied on clean data.
- **Before fix**: w had ZERO effect (loss non-differentiable) — every w byte-
  identical. Hypothesis untestable; the experiment was void (§8.0-bis).
- **After fix**: w now changes training (fix verified), but the hypothesis is
  **REJECTED**:
  - Clean accuracy is **flat** across w (~96.0); physics gives no clean gain.
  - 5 dB is **not improved** at low w (92.5/92.2 vs 91.0 — within the ±5–6
    seed variance, overlapping) and **actively degraded** at high w
    (w=1.0 → 83.1, *below* w=0's 91.0, with one seed collapsing to 71.7).
  - The trend is monotone the WRONG way: more physics weight → worse, more
    unstable noise robustness.

## What it means

The frequency-consistency physics loss provides **no benefit** to journal-
bearing fault classification, and heavy weighting **hurts** noise robustness.
Mechanistic reading: the penalty anchors predictions to specific spectral
peaks; under heavy noise those peaks are exactly what's corrupted, so enforcing
physics-consistency drags the model toward unreliable features — the more so
the higher the weight. The data-driven CNN already captures the discriminative
spectral structure, leaving the explicit constraint redundant at best.

This is an honest, complete negative result for the loss-based physics route
(C3). The value here is twofold: (1) the methodological before/after — a naive
physics loss was silently inert, a real and common pitfall; (2) corrected, it
still doesn't help, which is itself an informative finding about this problem.

## McNemar (prereg) — deferred

The prereg decision rule asks for McNemar (w=0 vs best-w at 5 dB). The
per-window predictions aren't stored in `metrics.json` (only confusion matrix +
accuracy), so this needs a quick laptop re-eval of the w=0 (benchmark pc_cnn)
and best-w=0.1 (fixed) checkpoints capturing paired predictions. Given the
effect is within-noise / non-favorable, the descriptive conclusion already
holds; McNemar will be run when assembling FINDINGS.md for completeness.

## Decision (owner rule: rerun §8.2/8.3 only if the fix moves the needle)

The fix moved the needle **but not favorably** — physics-on is neutral-to-
harmful vs physics-off. There is no advantage to chase, so **skip the §8.2/8.3
fixed-loss rerun.** (Rerunning the pc_cnn arms with physics-on has no reason to
reveal a data-efficiency/OOD advantage the ablation shows doesn't exist.)
