# §8.6 — Physics-consistent XAI (8.6a) & calibration (8.6b)

Best-vanilla = `resnet18` (benchmark seed0). Best-physics = fixed `pc_cnn`
(w=0.3, seed1; differentiable-loss branch). **Both share the identical
ResNet1D backbone (3,853,195 params each)** — so every difference below is
attributable to physics-loss training, not architecture. Laptop/CPU, 220
stratified test windows; IG 32 steps; MC-dropout 30 passes. Artifacts:
`results/xai_alignment/alignment.json`, `results/uncertainty/calibration.json`.

## 8.6a — Attribution alignment with physics frequency bands

Fraction of Integrated-Gradients attribution *energy* (FFT of the attribution)
falling in the true class's PHYSICS.md characteristic-frequency bands (±15%),
vs equally-wide CONTROL bands off the harmonic grid.

| model | in-band frac | control frac | specificity ratio (in/ctrl) |
|---|---|---|---|
| resnet18 (vanilla) | 0.132 | 0.184 | **0.716** |
| pc_cnn (physics) | 0.136 | 0.160 | **0.849** |

**Read:** the physics-trained model's attributions are ~19% more concentrated
in characteristic-frequency bands *relative to control* than the identical-
backbone vanilla model (0.849 vs 0.716) — it puts more energy in physics bands
(0.136 vs 0.132) and less in off-band control (0.160 vs 0.184). Directional
support for C4: **physics-loss training shifts what the model attends to toward
physically-meaningful frequencies, even though it did not improve accuracy
(§8.4).**

**Honest caveats:** (1) both ratios are < 1 — neither model puts the *majority*
of attribution energy in the physics bands, so this is a relative, not
absolute, alignment claim; (2) the control-band definition (×1.37 off-grid)
affects the absolute fractions — a different control would shift them; (3)
per-class is mixed (pc_cnn higher for classes 2,4,5; vanilla higher for 1,3).
The robust, defensible statement is the relative one.

## 8.6b — MC-dropout calibration (clean + 5 dB)

Correct MC-dropout (dropout layers on, BatchNorm in eval — the stock
`UncertaintyQuantifier` used `model.train()`, which corrupts BN and was fixed).

| model | clean acc | clean ECE | 5 dB acc | 5 dB ECE |
|---|---|---|---|---|
| resnet18 (vanilla) | 0.955 | 0.0278 | 0.941 | 0.0272 |
| pc_cnn (physics) | 0.964 | **0.0215** | 0.964 | **0.0176** |

Reject option: both reach **100% accuracy at ~50% coverage** (drop the least-
confident half → perfect on the rest) — a strong deployment property for both.

**Read:** pc_cnn is modestly better calibrated than the same-backbone vanilla
model at both SNRs (lower ECE), and the gap widens slightly under noise
(0.0176 vs 0.0272 at 5 dB). Consistent with the C4 theme.

**Honest caveats:** (1) single checkpoint per model (not seed-averaged), so
treat as indicative, not statistically established — for the paper, average ECE
over the 3 seeds; (2) both models are already well-calibrated (ECE < 0.03);
(3) this pc_cnn checkpoint is the noise-robust seed1 (5 dB ≈ clean for it),
which flatters its 5 dB numbers — a seed-averaged comparison is the honest one.

## Synthesis (feeds FINDINGS.md / paper C4)

Physics-informed training is **not an accuracy lever** on clean synthetic data
(§8.2–8.5 negative), **but it is an interpretability/trust lever**: with the
identical backbone, the physics-trained model produces attributions more
aligned with known fault frequencies and is modestly better calibrated. That is
a coherent, honest C4 contribution — and a more interesting paper claim than a
forced accuracy win.

**TODO before the claim is paper-ready:** seed-average the calibration (ECE over
3 seeds each); add a sensitivity check on the control-band choice for 8.6a;
generate the reliability-diagram + reject-curve + per-class-alignment figures
(data already in the JSONs) during FINDINGS assembly.
