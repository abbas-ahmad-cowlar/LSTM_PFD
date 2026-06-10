# Physics of the Synthetic Journal-Bearing Fault Simulator

> **Status**: Draft for owner sign-off (Convergence Plan P3.1).
> **Scope**: Documents exactly what `data/signal_generation/` implements — every
> equation below is traceable to code (file:function cited per section). Where a
> coefficient is empirical rather than first-principles, it is flagged **[E]**
> and defended. This document is the normative reference for the spectral-signature
> validation tests (`tests/test_physics_signatures.py`) — each test cites a section
> here (§n).
>
> **Honesty note**: This simulator produces *physically structured* synthetic
> signals — correct characteristic frequencies, physically motivated parameter
> couplings — but its amplitude coefficients are plausible engineering values,
> not solutions of the Reynolds equation. Signal fidelity against a real test
> rig has **not** been validated; that is future work (BACKLOG: sim-to-real).

---

## 1. Signal composition model

Every signal (`generator.py:generate_single_signal`) is assembled as:

```
x(t) = n_base(t) + Σ noise_layers(t) + s_fault(t)
```

- **Baseline**: `n_base = amp_base · 0.05 · N(0,1)` — broadband machine floor,
  where `amp_base = (0.2 + 0.1·U) · operating_factor` couples the floor level
  to load and temperature.
- **Noise layers** (`noise_generator.py`, 7 independent layers, each toggleable):
  measurement (Gaussian, σ=0.03), EMI (50/60 Hz mains tone, 0.01), pink 1/f
  (0.02), low-frequency thermal drift (0.015), ADC quantization (step 0.001),
  sensor calibration drift (0.001/s), sporadic impulses (2/s), plus an aliasing
  artifact with probability 0.10. Defaults in `utils/physics_constants.py::NoiseDefaults`.
- **Fault signature** `s_fault(t)`: per-class models of §4, scaled by a severity
  curve (§2) and a transient modulation envelope.

Sampling: fs = 20,480 Hz, T = 5 s, N = 102,400 samples. Nominal shaft speed
Ω₀ = 60 Hz (3,600 RPM).

## 2. Severity model

(`generator.py:_configure_severity`, `config/data_config.py::SeverityConfig`)

Four labeled levels with **non-overlapping** factor ranges:

| Level | Severity factor s |
|---|---|
| incipient | 0.20–0.45 |
| mild | 0.45–0.70 |
| moderate | 0.70–0.90 |
| severe | 0.90–1.00 |

The factor multiplies the fault signature pointwise via `severity_curve(t)`.
30% of signals carry **temporal evolution**: the curve grows linearly from s to
min(1, s+0.3) across the record, simulating progressing degradation.
The level label and initial factor are stored in metadata → this enables the
severity-shift OOD experiments (Phase 5.2).

## 3. Operating conditions & fluid-film physics

(`generator.py:_configure_operating_conditions`, `_configure_physics`)

Per signal, sampled independently and stored in metadata:

- **Speed**: Ω = Ω₀·(1+δ), δ ~ U(−10%, +10%) → Ω ∈ [54, 66] Hz.
- **Load**: 30–100% of rated; `load_factor = 0.3 + 0.7·(load%/100)`.
- **Temperature**: T ∈ [40, 80] °C; `temp_factor = 0.9 + 0.2·(T−40)/40`.
- **Viscosity** [E]: exponential law μ(T) = μ_ref·exp(−0.03·(T−60)) — the
  standard Reynolds-type viscosity-temperature exponential with a decay rate
  typical of ISO VG 32–68 turbine oils over 40–80 °C.
- **Sommerfeld number**: S = S_base · (μ/μ_ref) · (Ω/Ω₀) / load_factor,
  clipped to [0.05, 0.50] (`BearingPhysics`). This is the standard
  proportionality S ∝ μN/P (viscosity × speed / unit load) anchored at
  S_base = 0.15 **[E]** — a representative value for a moderately loaded
  industrial journal bearing (ISO 7902 regime charts).
- **Reynolds number**: sampled in [500, 5000] (laminar → transition film flow).
- **Clearance ratio**: C/R ∈ [0.001, 0.003] (typical journal-bearing practice).

## 4. Fault signature models

All in `fault_modeler.py:generate_fault_signal`. Amplitude coefficients
(`FaultAmplitudes`) are **[E]** throughout: they set plausible *relative*
signature strengths (fault component vs ~0.05·amp_base noise floor) rather than
absolute g-levels. Their defense: (a) each fault's *frequency structure* — the
diagnostic content — follows established vibration-analysis practice cited per
class below; (b) relative amplitudes were chosen so no class is trivially
separable by RMS alone (verified: classical-ML baseline does not saturate);
(c) all coefficients are centralized and frozen (`physics_constants.py`), so
experiments are reproducible and the values are auditable.

### 4.1 `sain` (healthy)
`s_fault = 0`. The signal is the operating-condition-scaled noise floor only.
**Test signature**: lowest RMS of all classes; no dominant tonal peaks above the
noise floor besides EMI.

### 4.2 `desalignement` (misalignment)
`0.35·sin(2ωt+φ₂) + 0.20·sin(3ωt+φ₃)` — energy at **2× and 3× shaft speed**,
2X dominant. This is the textbook misalignment signature (parallel misalignment
→ 2X radial; angular → axial 1X/2X/3X; e.g. ISO 13373-2, Piotrowski).
Couplings: severity curve, transient envelope. Phases random per signal.
**Test signature (§T2)**: PSD peaks at 2Ω and 3Ω; energy(2Ω) > energy(1Ω);
energy(2Ω) > energy(3Ω).

### 4.3 `desequilibre` (imbalance)
`0.5·load_factor·sin(ωt+φ)·δ_speed²` — pure **1× tone** whose amplitude grows
with the **square of speed** (centrifugal force F = m·r·ω²) and with load.
**Test signature (§T3)**: dominant single peak at 1Ω; amplitude monotone in
speed_variation² (two-point check).

### 4.4 `jeu` (bearing clearance / looseness)
Sub-synchronous component at **(0.43–0.48)·Ω** (0.25·temp_factor) + 1X (0.18)
+ 2X (0.10). Looseness in journal bearings manifests as sub-synchronous whirl-
like motion plus harmonic generation (API 684 sub-synchronous instability
discussion). Temperature coupling via clearance growth.
**Test signature (§T4)**: sub-synchronous peak in [0.40, 0.50]·Ω band, plus 1X
and 2X peaks.

### 4.5 `lubrification` (lubrication deficiency)
Stick-slip oscillation at **2–5 Hz** with amplitude `0.30·temp_factor·(0.3/S)` —
inversely proportional to Sommerfeld number (low S = thin film = boundary
contact), plus 1–4 random decaying friction impacts (length ~1 ms, exponential
decay). Captures the boundary-lubrication regime: low-frequency friction-driven
oscillation intensifying as the film thins.
**Test signature (§T5)**: low-frequency band (1–6 Hz) energy elevated; RMS
decreases as S increases (two-point check); impact spikes → kurtosis > 3.

### 4.6 `cavitation`
2–7 short bursts (8 ms Hann-windowed, exp(−100t) decay) of a **1.5–2.5 kHz**
carrier, amplitude 0.6·severity. Vapor-bubble collapse excites high-frequency
structural resonances in short transient bursts — the classic cavitation
signature in hydraulic machinery (broadband HF bursts rather than tones).
**Test signature (§T6)**: spectral energy fraction in 1.4–2.6 kHz band elevated
vs healthy; high kurtosis (burstiness).

### 4.7 `usure` (wear)
Broadband noise `0.25·operating·physics·N(0,1)` (surface roughness increase)
+ asperity harmonics `0.12·(sin ωt + 0.5·sin 2ωt)` + slow amplitude modulation
(0.5–2 Hz). Wear raises the broadband floor while asperity contact adds weak
shaft-synchronous tones.
**Test signature (§T7)**: spectral flatness higher than tonal faults; broadband
floor elevated vs healthy; 1X present.

### 4.8 `oilwhirl`
Tone at **(0.42–0.48)·Ω** with amplitude `0.40/√S`, slow sub-modulation at half
the whirl frequency. Oil whirl is the canonical journal-bearing instability:
the lubricant wedge rotates at slightly less than half shaft speed (classically
0.42–0.48×Ω; Muszynska). Lower S (lighter relative load ↔ here inverse
amplitude coupling [E]) promotes whirl.
**Test signature (§T8)**: dominant sub-synchronous peak in [0.40, 0.49]·Ω;
RMS decreases as S increases (two-point check).

### 4.9–4.11 Mixed faults
Superpositions with reduced per-component amplitudes (≈70% of single-fault
coefficients) so the combined RMS stays comparable:
- `mixed_misalign_imbalance`: 1X + 2X + 3X simultaneously (§4.2 + §4.3).
- `mixed_wear_lube`: broadband + asperity tones + stick-slip + contact impacts
  (§4.7 + §4.5).
- `mixed_cavit_jeu`: HF bursts + sub-synchronous + 1X (§4.6 + §4.4).
**Test signature (§T9–T11)**: each constituent signature detectable
simultaneously.

## 5. Advanced physics effects (optional toggles)

(`fault_modeler.py:apply_advanced_physics` — off by default in v1/v2 datasets;
documented for completeness)

Run-up/coast-down speed ramps; band-limited random-walk speed fluctuation
(simplified FM→AM); critical-speed resonance amplification
H = 1/√((1−r²)² + (2ζr)²) with ζ=0.05, f_crit=45 Hz (API 684 typical);
cross-coupled stiffness (Hilbert-shifted +90° component, Kxy/Kxx=0.3);
thermal-growth amplitude drift; axial vibration coupling. **Decision for v2**:
keep OFF — they add realism noise without serving the C1–C4 experiments, and
toggling them mid-study would invalidate comparisons.

## 6. Known limitations (stated, deliberate)

1. Amplitude coefficients are engineering-plausible, not Reynolds-equation
   solutions **[E]** — the *frequency structure* carries the physics.
2. No structural transfer path: signals are "at the bearing" — no housing/sensor
   transfer function (could be added as an FIR in future).
3. Speed fluctuation is amplitude-modulated, not true FM (simplification noted
   in code).
4. No real-data validation yet — this is the central limitation the paper must
   state; the dataset's value is controlled ground truth, not field fidelity.

---

*Owner sign-off (P3.1 DoD): pending.*
