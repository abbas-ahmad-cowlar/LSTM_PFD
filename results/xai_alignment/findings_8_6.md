# §8.6 — Physics-consistent XAI (8.6a) & calibration (8.6b)

> **CORRECTED RECOMPUTE — 2026-06-17 (P6 Step 5c).** Recomputed against the
> **corrected journal-bearing signature DB** and the **band-energy w=1.0
> checkpoint** (the noise-robust arm), same-backbone vanilla, both best-val seed2 —
> the representative seeds of the record-level surviving-result analysis.
> Artifacts: `results/xai_alignment/alignment.json`,
> `results/uncertainty/calibration.json` (provenance git `a2e09d9`,
> resnet18 seed2 vs pc_cnn band-energy w1.0 seed2, 160 windows, IG 32 steps,
> MC-dropout 30). **The §1–§2 tables below the line are the OLD contaminated run
> (broken rolling-element DB + the tonal-only w0.3 seed1 checkpoint) — SUPERSEDED,
> kept for provenance.**

## VERDICT — the C4 interpretability/calibration positive does NOT survive

Against the corrected bands and the band-energy checkpoint, **neither §8.6a nor
§8.6b supports a physics advantage.** The earlier "physics attributions are more
physics-aligned, and physics is better calibrated" was substantially an artifact
of the broken DB + contaminated checkpoint. The only surviving physics benefit in
the whole study is the record-level **noise robustness** (§8.4); C4 is **not** a
second positive.

### 8.6a — attribution alignment with physics frequency bands (corrected)

Fraction of IG-attribution spectral energy in the true class's PHYSICS.md
**tonal** characteristic frequencies (±15%) vs equally-wide control bands (×1.37
off-grid). Specificity ratio = in-band / control.

| model | in-band frac | control frac | specificity ratio |
|---|---|---|---|
| resnet18 (vanilla) | 0.1875 | 0.1799 | **1.042** |
| pc_cnn (band-energy w1.0) | 0.1259 | 0.1470 | **0.856** |

**Read (honest):** the result **reverses** — the *vanilla* model is now slightly
*more* tonally physics-aligned (ratio 1.042, puts more energy in-band than
control) than the band-energy model (0.856, less in-band than control). The old
0.849 > 0.716 in physics's favor does not reproduce once the DB is correct and the
checkpoint is the validated one.

**Decisive caveat — the metric is structurally blind to the physics model's
mechanism.** `get_expected_frequencies` is **tonal-only**, so it **excludes
`lubrification` (class 4) and `cavitation` (class 5)** entirely (no tonal
signature → 0 of the 160 scored windows are from them; only classes 1,2,3,6,7,8,9,10
are scored). But the band-energy loss operates on **absolute Hz bands** for exactly
those broadband classes, and the surviving noise result is driven by **lubrification**
(the 14 rescued records, §8.4). So this metric cannot see — and likely penalizes —
the band the physics model actually learned to attend to. **This is "not more
*tonally* aligned," NOT "attends less to physics."** A fair test needs a
**band-aware** variant (`get_expected_bands`: tonal harmonics + the 1–6 Hz lube /
1.4–2.6 kHz cavitation absolute bands) with a documented control — flagged as a
methodology decision for the owner, not made unilaterally here.

Per-class in-band fraction is mixed (physics higher only for class 1; vanilla
higher for 2,3,6,7,8,9,10) — consistent with the loss pushing attention toward
broadband bands this tonal metric does not measure.

### 8.6b — MC-dropout calibration (corrected; clean + 5 dB)

MC-dropout 30 passes (dropout on, BatchNorm eval), 160-window stratified subsample.

| model | clean acc | clean ECE | 5 dB acc | 5 dB ECE |
|---|---|---|---|---|
| resnet18 (vanilla) | 0.964 | 0.0241 | 0.964 | **0.0225** |
| pc_cnn (band-energy w1.0) | 0.964 | **0.0182** | 0.959 | 0.0258 |

**Read (honest):** a **wash, not a physics win.** pc_cnn is better calibrated on
**clean** (ECE 0.0182 vs 0.0241) but **worse at 5 dB** (0.0258 vs 0.0225) — the
old "better calibrated, gap *widens* under noise" actually **inverts**. Both
models are well-calibrated (ECE < 0.03). Single checkpoint each, window-level
subsample → indicative only, not seed-averaged. The plain-eval accuracy sanity
gate (both 0.964 clean) confirms the correct checkpoints loaded.

### Synthesis (feeds the FINDINGS rewrite / paper C4)

C4 as previously framed (a modest interpretability + calibration positive) **does
not survive the correction.** Corrected: tonal attribution alignment reverses
(vanilla ahead), and calibration is a clean/5 dB wash. The defensible study now
has **one** physics positive — the record-level **noise robustness** (§8.4,
same-architecture ablation) — and **no** XAI/calibration advantage to claim. The
honest framing is stronger for it: a single, well-supported result rather than a
bundle of fragile ones.

**Open (owner decision):** a **band-aware** §8.6a recompute (including the lube /
cavitation absolute bands) is the only way to fairly test whether the band-energy
model attends *more* to its broadband targets; the tonal-only metric cannot. This
is the XAI analogue of the F9 controls question — it would corroborate (or refute)
the physics-content explanation of the noise result. Until run, **no XAI/calibration
benefit may be claimed.**

---

> ⚠️ **SUPERSEDED / CONTAMINATED (pre-2026-06-17).** The tables below used the
> broken **rolling-element** signature DB and the tonal-only **pc_cnn w0.3 seed1**
> checkpoint (resnet18 seed0). Retained for provenance only — **do not cite.**

Best-vanilla = `resnet18` (benchmark seed0). Best-physics = fixed `pc_cnn`
(w=0.3, seed1; differentiable-loss branch). **Both share the identical
ResNet1D backbone (3,853,195 params each)** — so every difference below is
attributable to physics-loss training, not architecture. Laptop/CPU, 220
stratified test windows; IG 32 steps; MC-dropout 30 passes.

## 8.6a — Attribution alignment with physics frequency bands (SUPERSEDED)

| model | in-band frac | control frac | specificity ratio (in/ctrl) |
|---|---|---|---|
| resnet18 (vanilla) | 0.132 | 0.184 | **0.716** |
| pc_cnn (physics) | 0.136 | 0.160 | **0.849** |

(Old read: physics ~19% more concentrated relative to control. **Reversed by the
corrected recompute above.**)

## 8.6b — MC-dropout calibration (SUPERSEDED)

| model | clean acc | clean ECE | 5 dB acc | 5 dB ECE |
|---|---|---|---|---|
| resnet18 (vanilla) | 0.955 | 0.0278 | 0.941 | 0.0272 |
| pc_cnn (physics) | 0.964 | 0.0215 | 0.964 | 0.0176 |

(Old read: physics modestly better calibrated, gap widens under noise. **The
corrected recompute inverts the 5 dB direction — superseded.** This old pc_cnn
checkpoint was the noise-robust seed1, which flattered its 5 dB numbers.)
