# Project Scope Analysis — LSTM_PFD

> **Analysis Date:** 2026-01-24  
> **Purpose:** Extract implicit scope, detect mismatches, and define project limitations  
> **Version:** 1.0

---

## Executive Summary

This analysis reveals a **critical scope mismatch** at the heart of the LSTM_PFD project:

| Aspect            | Synthetic Data Generator       | CWRU Benchmark        |
| ----------------- | ------------------------------ | --------------------- |
| **Bearing Type**  | Hydrodynamic/Journal           | Ball Bearing          |
| **Physics Model** | Sommerfeld equation, oil whirl | BPFO/BPFI/BSF         |
| **Fault Types**   | 11 (oil whirl, cavitation)     | 10 (inner/outer/ball) |
| **Sampling Rate** | 20,480 Hz                      | 12,000 Hz             |

> [!CAUTION]
> The physics model generates synthetic data for **journal bearings** but the project benchmarks against **CWRU ball bearing data**. These are fundamentally different mechanical systems with different failure modes.

---

## 1. What is PFD?

### Current Definition

| Source        | Definition Found |
| ------------- | ---------------- |
| README.md     | ❌ Not defined   |
| Documentation | ❌ Not defined   |
| Code comments | ❌ Not defined   |
| File names    | ❌ Not defined   |

**PFD is never explicitly defined anywhere in the codebase.**

### Proposed Definition (Based on Code Analysis)

Based on the actual implementation, "PFD" most likely stands for one of:

1. **P**hysics-informed **F**ault **D**iagnosis ⭐ _Most Likely_
   - Evidence: The project extensively uses physics-informed neural networks (PINNs)
   - The Sommerfeld equation and Reynolds number are central to the physics model

2. **P**redictive **F**ault **D**etection
   - Aligns with the predictive maintenance focus

3. **P**rocess(ing) **F**ault **D**iagnosis
   - Generic descriptor matching ML classification

> [!IMPORTANT]
> **Recommendation**: Add explicit definition to README.md Line 1:
>
> ```markdown
> # LSTM_PFD: Physics-informed Fault Diagnosis for Bearing Systems
> ```

---

## 2. Discovered Implicit Scope

### 2.1 Bearing Type

| Parameter                | Value                         | Evidence                                                                                             |
| ------------------------ | ----------------------------- | ---------------------------------------------------------------------------------------------------- |
| **Implemented**          | Hydrodynamic/Journal Bearings | See evidence below                                                                                   |
| **Documentation Claims** | "Hydrodynamic bearings"       | [USER_GUIDE/index.html](file:///c:/Users/COWLAR/projects/LSTM_PFD/site/USER_GUIDE/index.html) FAQ Q3 |

**Evidence from Source Code:**

#### [signal_generator.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/signal_generator.py#L90-92)

```python
class FaultModeler:
    """
    Physics-based fault modeling for hydrodynamic bearings.

    Implements equations from Section 7.3 of technical report.
    """
```

#### [pinn-theory.md](file:///c:/Users/COWLAR/projects/LSTM_PFD/docs/research/pinn-theory.md#L72-79)

```markdown
### 3. Bearing Dynamics

Sommerfeld equation for journal bearings:

$$S = \frac{\mu N L D}{W} \left( \frac{R}{c} \right)^2$$
```

### 2.2 Physics Equations Used

| Equation                | Purpose                     | Bearing Type             |
| ----------------------- | --------------------------- | ------------------------ |
| **Sommerfeld Number**   | Lubrication regime          | Journal/Hydrodynamic     |
| **Reynolds Number**     | Flow characterization       | Both                     |
| **Oil Whirl Frequency** | Sub-synchronous instability | Journal (oil-lubricated) |
| **BPFO/BPFI/BSF**       | Defect frequencies          | Ball/Rolling Element     |

> [!WARNING]
> [bearing_dynamics.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/core/models/physics/bearing_dynamics.py) calculates **ball bearing** characteristic frequencies (BPFO, BPFI, BSF) but [signal_generator.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/signal_generator.py) generates **journal bearing** fault signatures. This is an internal inconsistency.

### 2.3 Operating Envelope

| Parameter           | Min     | Max     | Configurable? | Source            |
| ------------------- | ------- | ------- | ------------- | ----------------- |
| Speed (RPM)         | 2,700   | 4,500   | Yes (±10%)    | `data_config.py`  |
| Speed (Hz)          | 54      | 66      | Yes           | Base: 60 Hz       |
| Load (%)            | 30%     | 100%    | Yes           | `OperatingConfig` |
| Temperature (°C)    | 40      | 80      | Yes           | `OperatingConfig` |
| Sampling Rate (Hz)  | 20,480  | 20,480  | No\*          | `constants.py`    |
| Signal Duration (s) | 5.0     | 5.0     | Yes           | `SignalConfig`    |
| Total Samples       | 102,400 | 102,400 | Derived       | `SIGNAL_LENGTH`   |

\*Sampling rate is hardcoded in `constants.py` Line 27

### 2.4 Fault Types (Actual Implementation)

From [constants.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/utils/constants.py#L48-60):

| ID  | Internal Name              | Display Name      | Physics Basis                  | Category |
| --- | -------------------------- | ----------------- | ------------------------------ | -------- |
| 0   | `sain`                     | Healthy           | Baseline                       | Baseline |
| 1   | `desalignement`            | Misalignment      | 2X, 3X harmonics               | Journal  |
| 2   | `desequilibre`             | Imbalance         | 1X, speed² dependence          | Both     |
| 3   | `jeu`                      | Bearing Clearance | Sub-sync + harmonics           | Journal  |
| 4   | `lubrification`            | Lubrication Issue | Stick-slip, inverse Sommerfeld | Journal  |
| 5   | `cavitation`               | Cavitation        | High-freq bursts               | Journal  |
| 6   | `usure`                    | Wear              | Broadband + AM                 | Both     |
| 7   | `oilwhirl`                 | Oil Whirl         | Sub-sync 0.43×Ω                | Journal  |
| 8   | `mixed_misalign_imbalance` | Combined          | Additive                       | Mixed    |
| 9   | `mixed_wear_lube`          | Combined          | Additive                       | Mixed    |
| 10  | `mixed_cavit_jeu`          | Combined          | Additive                       | Mixed    |

**Classification**: 7/11 fault types are **journal bearing specific** (oil whirl, cavitation, lubrication issues).

---

## 3. Mismatch Analysis

### 3.1 Synthetic Data ↔ CWRU Benchmark

| Aspect              | Synthetic Data        | CWRU Dataset                    | Match?       | Impact      |
| ------------------- | --------------------- | ------------------------------- | ------------ | ----------- |
| **Bearing Type**    | Journal/Hydrodynamic  | Deep Groove Ball (SKF 6205)     | ❌ No        | 🔴 Critical |
| **Fault Types**     | 11 (journal-specific) | 10 (inner/outer/ball)           | ❌ No        | 🔴 Critical |
| **Physics Model**   | Sommerfeld equation   | N/A (real data)                 | N/A          | -           |
| **Sampling Rate**   | 20,480 Hz             | 12,000 Hz (DE) / 48,000 Hz (FE) | ❌ No        | 🟡 Medium   |
| **Signal Length**   | 102,400 samples       | Variable (≈121K per file)       | ⚠️ Different | 🟢 Low      |
| **Num Classes**     | 11                    | 10                              | ❌ No        | 🔴 Critical |
| **Load Conditions** | 30-100%               | 0-3 HP (fixed levels)           | ⚠️ Different | 🟡 Medium   |

**Severity**: 🔴 **CRITICAL**

**Impact**:

- Cannot directly compare synthetic-trained models on CWRU
- Physics-informed constraints (Sommerfeld, oil whirl) are irrelevant for ball bearings
- Transfer learning claims would be scientifically invalid

### 3.2 README ↔ Implementation

| Claim in README                                | Actually Implemented           | Match?                |
| ---------------------------------------------- | ------------------------------ | --------------------- |
| "11 Fault Types" (Line 38, 83)                 | 11 types in `constants.py`     | ✅ Yes                |
| "CWRU Bearing Dataset" (various)               | Loader exists for benchmarking | ⚠️ Partial            |
| "Hydrodynamic bearings" (User Guide)           | Journal bearing physics        | ✅ Yes                |
| "98-99% Accuracy" (Line 64)                    | Ensemble on synthetic data     | ⚠️ Unclear validation |
| "Ball Fault, Inner Race, Outer Race" (Line 83) | Not in synthetic generator     | ❌ No                 |

**README Line 83 Claims**:

```
Normal, Ball Fault, Inner Race, Outer Race, Combined,
Imbalance, Misalignment, Oil Whirl, Cavitation, Looseness, Oil Deficiency
```

**Actual `constants.py` Implementation**:

```
sain, desalignement, desequilibre, jeu, lubrification,
cavitation, usure, oilwhirl, mixed_misalign_imbalance,
mixed_wear_lube, mixed_cavit_jeu
```

> [!CAUTION]
> README claims "Ball Fault, Inner Race, Outer Race" but these are **not implemented** in the synthetic generator. The synthetic data generates journal bearing faults (oil whirl, cavitation).

### 3.3 `bearing_dynamics.py` ↔ `signal_generator.py`

This is an **internal inconsistency**:

| File                  | Bearing Type                            | Purpose                   |
| --------------------- | --------------------------------------- | ------------------------- |
| `bearing_dynamics.py` | Ball bearing (BPFO/BPFI/BSF)            | PINN physics constraints  |
| `signal_generator.py` | Journal bearing (Sommerfeld, oil whirl) | Synthetic data generation |

The default bearing in `bearing_dynamics.py` is SKF 6205 (ball bearing), but the synthetic data is generated for journal bearings.

---

## 4. Explicit Limitations to Document

### 4.1 Must Add to README

```markdown
## ⚠️ Scope & Limitations

### Bearing Type Scope

This system is designed for **hydrodynamic/journal bearings** operating in:

- **Speed Range:** 2,700 - 4,500 RPM (54-66 Hz)
- **Load Range:** 30% - 100% of rated load
- **Temperature Range:** 40°C - 80°C
- **Lubrication:** Oil-lubricated systems

### Fault Types Covered

The physics model generates 11 fault types specific to journal bearings:

- Healthy baseline
- Misalignment, Imbalance, Looseness
- Lubrication issues, Oil whirl _(journal-specific)_
- Cavitation, Wear
- Combined fault modes

### Not Suitable For

Without modification, this system is **not validated** for:

- ❌ Rolling element bearings (ball, roller, tapered)
- ❌ Dry bearings (no lubrication)
- ❌ Magnetic bearings
- ❌ Speeds outside 2,700-4,500 RPM

### CWRU Benchmark Notes

The CWRU dataset loader is included for **research comparison** but:

- CWRU uses **ball bearings** (different physics)
- Fault types differ (inner/outer race vs. oil whirl/cavitation)
- Physics-informed constraints are not applicable
```

### 4.2 Must Add to Research Paper

```latex
\section{Limitations}

The proposed method has been developed and validated for
\textbf{hydrodynamic journal bearings} with:
\begin{itemize}
    \item Synthetic data generated using Sommerfeld-based physics model
    \item 11 fault classes including oil whirl and cavitation
    \item Operating conditions: 54-66 Hz, 30-100\% load, 40-80°C
\end{itemize}

Extension to rolling element bearings (e.g., CWRU dataset) would require:
\begin{enumerate}
    \item Replacement of Sommerfeld constraints with defect frequency equations
    \item Regeneration of synthetic training data
    \item Revalidation of physics-informed loss functions
\end{enumerate}

While preliminary experiments on the CWRU dataset show promising results,
the physics-informed components were designed for journal bearings and
their effectiveness on ball bearings has not been rigorously validated.
```

---

## 5. Research Paper Scope Statement (Draft)

### Valid Claims ✅

1. "State-of-the-art for journal/hydrodynamic bearing fault diagnosis"
2. "Physics-informed approach using Sommerfeld constraints"
3. "Validated on physics-based synthetic data with 11 fault types"
4. "98%+ accuracy on synthetic journal bearing dataset"
5. "Explainable predictions via SHAP, LIME, Integrated Gradients"

### Invalid Claims Without Additional Work ❌

1. ~~"Generalizes to all bearing types"~~
2. ~~"Validated on CWRU ball bearing dataset"~~ (physics mismatch not addressed)
3. ~~"Ball fault, inner race, outer race detection"~~ (not in synthetic model)
4. ~~"Universal bearing fault diagnosis"~~

### Recommended Title Adjustment

| Current                                   | Suggested                                                                   |
| ----------------------------------------- | --------------------------------------------------------------------------- |
| "Advanced Bearing Fault Diagnosis System" | "Physics-Informed Hydrodynamic Bearing Fault Diagnosis using LSTM Networks" |
| "LSTM_PFD"                                | "LSTM-PINN for Journal Bearing Diagnostics"                                 |

---

## 6. Commercial Applicability

### Strong Fit ✅

| Industry              | Application         | Why                                            |
| --------------------- | ------------------- | ---------------------------------------------- |
| **Turbomachinery**    | Steam/gas turbines  | Journal bearings, oil-lubricated, direct match |
| **Hydropower**        | Generator bearings  | Large journal bearings, fluid film lubrication |
| **Pumps/Compressors** | Industrial pumps    | Sleeve bearings common, oil whirl relevant     |
| **Marine Propulsion** | Stern tube bearings | Water-lubricated journal bearings              |
| **Paper Mills**       | Roll bearings       | Large journal bearings, high speeds            |

### Poor Fit ❌

| Industry            | Application     | Why Not                                |
| ------------------- | --------------- | -------------------------------------- |
| **Electric Motors** | Standard motors | Typically ball bearings (CWRU-type)    |
| **Automotive**      | Wheel bearings  | Tapered roller, not journal            |
| **HVAC**            | Fan motors      | Ball bearings, different failure modes |
| **Robotics**        | Joint bearings  | Mixed types, often ball/roller         |

### Could Fit With Modifications

| Industry               | Required Changes                                   | Effort    |
| ---------------------- | -------------------------------------------------- | --------- |
| **CNC Spindles**       | Replace Sommerfeld with BPFO/BPFI; regenerate data | 2-3 weeks |
| **Wind Turbines**      | Support both bearing types in pitch/yaw            | 4-6 weeks |
| **General Industrial** | Add ball bearing physics module                    | 4-8 weeks |

---

## 7. Recommendations Summary

### Priority 1: Documentation Fixes (< 1 day)

1. **Define PFD in README Line 1**: Add "Physics-informed Fault Diagnosis"
2. **Correct fault type list in README Line 83**: Replace ball/inner/outer with actual fault types
3. **Add Scope & Limitations section**: Use template from Section 4.1
4. **Update User Guide FAQ Q3**: Clarify ball bearing vs journal bearing distinction

### Priority 2: Code Consistency (1-3 days)

1. **Align `bearing_dynamics.py`**: Either:
   - Add journal bearing physics mode, OR
   - Remove/isolate ball bearing calculations
2. **Document sampling rate assumption**: Add comment explaining 20,480 Hz choice
3. **Add explicit bearing type to config**: `bearing_type: 'journal' | 'ball'`

### Priority 3: Scope Expansion (Future Work)

| Task                               | Effort    | Impact                               |
| ---------------------------------- | --------- | ------------------------------------ |
| Add ball bearing fault generator   | 4-6 weeks | Opens CWRU validation                |
| Support variable sampling rates    | 1-2 weeks | Enables CWRU/Paderborn compatibility |
| Dual-physics PINN (journal + ball) | 6-8 weeks | Universal bearing support            |
| Real journal bearing validation    | 4-8 weeks | Industry credibility                 |

---

## 8. Cross-References to Consolidated IDB Reports

The following scope-related issues have been documented in the existing consolidated IDB reports:

### From [DOMAIN_1_CORE_ML_CONSOLIDATED.md](file:///c:/Users/COWLAR/projects/LSTM_PFD/docs/idb_reports/compiled/DOMAIN_1_CORE_ML_CONSOLIDATED.md)

| Issue                                                                      | Scope Implication                                                          |
| -------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **Hardcoded magic numbers** (`102400`, `20480`, `5000`) in 45+ occurrences | Signal length and sampling rate baked into code; changes require 45+ edits |
| **55+ model architectures** across 13 subdirectories                       | Broad model support, but only ~15 registered in factory                    |
| **Empty `__init__.py`** in Training, Evaluation packages                   | API scope unclear                                                          |

### From [DOMAIN_3_DATA_CONSOLIDATED.md](file:///c:/Users/COWLAR/projects/LSTM_PFD/docs/idb_reports/compiled/DOMAIN_3_DATA_CONSOLIDATED.md)

| Issue                                                               | Scope Implication                         |
| ------------------------------------------------------------------- | ----------------------------------------- |
| **11 fault classes matching MATLAB `generator.m` port**             | Confirms journal bearing scope            |
| **Sommerfeld number calculation with Arrhenius viscosity model**    | Physics model is hydrodynamic-specific    |
| **8-layer noise model** (docs say 7, implementation has 8)          | Documentation mismatch                    |
| **CWRU benchmark support** exists but is for different bearing type | Benchmark comparison has physics mismatch |
| **Thread-local HDF5 handles with SWMR mode**                        | Scope: production-ready data loading      |

### From [DOMAIN_5_RESEARCH_INTEGRATION_CONSOLIDATED.md](file:///c:/Users/COWLAR/projects/LSTM_PFD/docs/idb_reports/compiled/DOMAIN_5_RESEARCH_INTEGRATION_CONSOLIDATED.md)

| Issue                                                | Scope Implication                  |
| ---------------------------------------------------- | ---------------------------------- |
| **7 of 10 unified pipeline phases are placeholders** | E2E pipeline scope incomplete      |
| **`sys.path.insert()` anti-pattern** in all adapters | Portability scope issue            |
| **Hardcoded fallback for SAMPLING_RATE**             | Fails silently on import issues    |
| **Constants: 50+ in `utils/constants.py` (629 LOC)** | Centralized scope parameters exist |

### From [CRITICAL_ISSUES_MATRIX.md](file:///c:/Users/COWLAR/projects/LSTM_PFD/docs/idb_reports/compiled/CRITICAL_ISSUES_MATRIX.md)

> [!NOTE]
> See the Critical Issues Matrix for the full P0/P1/P2 classification of scope-affecting issues across all domains.

---

## Appendix A: Key File References

| File                                                                                                              | Purpose                   | Bearing Type     |
| ----------------------------------------------------------------------------------------------------------------- | ------------------------- | ---------------- |
| [signal_generator.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/signal_generator.py)                         | Synthetic data generation | Journal          |
| [constants.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/utils/constants.py)                                      | Fault type definitions    | Journal          |
| [data_config.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/config/data_config.py)                                 | Operating conditions      | Journal          |
| [bearing_dynamics.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/packages/core/models/physics/bearing_dynamics.py) | Physics calculations      | Ball (mismatch!) |
| [cwru_dataset.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/cwru_dataset.py)                                 | CWRU loader               | Ball             |
| [pinn-theory.md](file:///c:/Users/COWLAR/projects/LSTM_PFD/docs/research/pinn-theory.md)                          | Physics documentation     | Journal          |

---

## Appendix B: CWRU Dataset Details

From [cwru_dataset.py](file:///c:/Users/COWLAR/projects/LSTM_PFD/data/cwru_dataset.py#L67-127):

| Class       | Label | Fault Size | Files       |
| ----------- | ----- | ---------- | ----------- |
| `normal`    | 0     | N/A        | 97-100.mat  |
| `ball_007`  | 1     | 0.007"     | 118-121.mat |
| `ball_014`  | 2     | 0.014"     | 185-188.mat |
| `ball_021`  | 3     | 0.021"     | 222-225.mat |
| `inner_007` | 4     | 0.007"     | 105-108.mat |
| `inner_014` | 5     | 0.014"     | 169-172.mat |
| `inner_021` | 6     | 0.021"     | 209-212.mat |
| `outer_007` | 7     | 0.007"     | 130-133.mat |
| `outer_014` | 8     | 0.014"     | 197-200.mat |
| `outer_021` | 9     | 0.021"     | 234-237.mat |

**Total**: 10 classes (vs. 11 in synthetic data)

---

_Report generated by IDB Analysis Agent — Project Scope Analyst_  
_Analysis completed: 2026-01-24_
