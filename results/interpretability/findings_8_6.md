# §8.6 — Physics-consistent XAI (8.6a) & calibration (8.6b)

> **CORRECTED RECOMPUTE — 2026-06-17 (P6 Step 5c).** Recomputed against the
> **corrected journal-bearing signature DB** and the **band-energy w=1.0
> checkpoint** (the noise-robust arm), same-backbone vanilla, both best-val seed2 —
> the representative seeds of the record-level surviving-result analysis.
> Artifacts: `results/interpretability/alignment.json`,
> `results/calibration/calibration.json` (provenance git `a2e09d9`,
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

The tonal metric is **tonal-only** — it excludes `lubrification` and `cavitation`
(no tonal signature → only classes 1,2,3,6,7,8,9,10 scored), i.e. it cannot see the
broadband bands the band-energy loss operates on. To remove that blind spot we ran
the **band-aware** variant below.

#### 8.6a band-aware (corrected bands incl. broadband) — the rescue attempt FAILS

In-band = fraction of IG-attribution energy in the class's **full corrected
expected bands** (`get_expected_bands`: tonal harmonics ±6% **and** the absolute
1–6 Hz lube / 1.4–2.6 kHz cavitation bands), so **all 10 non-healthy classes** are
scored (n=200). The **physics-vs-vanilla comparison is on identical bands → no
control band, no control-sensitivity caveat.**

| model | band-aware in-band frac (n=200) |
|---|---|
| resnet18 (vanilla) | **0.1458** |
| pc_cnn (band-energy w1.0) | **0.0991** |

**Read (honest): the interpretability angle does NOT survive even band-aware.**
Vanilla puts *more* attribution energy in the physics bands than the physics model
(0.146 vs 0.099). Per-class, physics is higher in only **2 of 10** classes
(desalignement +0.042, cavitation +0.018) and lower in the other 8. **The decisive
one: `lubrification` — the exact class driving the §8.4 noise rescue — has vanilla
0.007 vs physics 0.002**, i.e. **both models put essentially zero attribution in the
1–6 Hz lube band, and physics even less.** So the noise-robustness benefit is **not**
explained by the physics model attending to physics bands more; IG attribution gives
**no** independent support for a "physics-attention" mechanism. (`mixed_wear_lube`
−0.128 and `desequilibre` −0.165 are the largest vanilla-favoring gaps.)

### 8.6b — MC-dropout calibration (corrected; clean + 5 dB)

MC-dropout 30 passes (dropout on, BatchNorm eval), 200-window stratified subsample.
**MC-dropout is stochastic (no fixed mask seed), so ECE varies run-to-run** — two
runs of this same script gave, at 5 dB: pc_cnn 0.0258 then 0.0228; vanilla 0.0225
then 0.0262. **The 5 dB direction flips between runs → the calibration comparison
is within MC-dropout noise, i.e. a non-result.** Latest run:

| model | clean acc | clean ECE | 5 dB acc | 5 dB ECE |
|---|---|---|---|---|
| resnet18 (vanilla) | 0.964 | 0.0236 | 0.964 | 0.0262 |
| pc_cnn (band-energy w1.0) | 0.964 | 0.0182 | 0.959 | 0.0228 |

**Read (honest):** a **wash / non-result.** Both models are well-calibrated
(ECE < 0.03) and the only stable difference (clean ECE modestly lower for pc_cnn)
is single-checkpoint, window-level, and not seed-averaged. The 5 dB direction is
not reproducible across runs. **No calibration advantage may be claimed.** The
plain-eval accuracy sanity gate (both 0.964 clean) confirms correct checkpoint load.

### Synthesis (feeds the FINDINGS rewrite / paper C4)

C4 as previously framed (a modest interpretability + calibration positive) **does
not survive the correction — in any form.** Corrected: tonal attribution alignment
reverses (vanilla ahead), the **band-aware** variant (the rescue attempt, run here)
**also** has vanilla ahead (0.146 vs 0.099) and shows *both* models ignore the lube
band, and calibration is a run-to-run-noise wash. The defensible study has **one**
physics positive — the record-level **noise robustness** (§8.4, same-architecture
ablation) — and **no** XAI/calibration advantage to claim. The honest framing is
stronger for it: a single, well-supported result rather than a bundle of fragile
ones.

**Implication for the F9 / "is it physics?" question.** We hoped IG attribution
would corroborate that the noise benefit comes from the model attending to physics
bands. It does **not** — even band-aware, and even on `lubrification` (the rescue
class), the physics model does not attend to its bands more than vanilla. So XAI
provides **no** independent evidence that the noise result is "physics" rather than
"a spectral regularizer that happened to harden the decision boundary." If the paper
wants the word **physics**, the **F9 training controls** (entropy/random-band/
permuted-reference) are now the **only** remaining way to establish it; otherwise the
claim must stay "the implemented band-energy term improved noise robustness," with
the mechanism left open. (Artifacts: `alignment.json` band_aware block,
`scripts/run_xai_calibration.py`.)

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
