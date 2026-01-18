# LSTM_PFD Project Inconsistencies Audit

> **Generated:** 2026-01-19  
> **Purpose:** Document architectural inconsistencies for future remediation  
> **Status:** Audit only (no fixes applied)

---

## Executive Summary

This audit reveals **significant architectural debt** across the LSTM_PFD project, primarily in:

1. **Fault Type Definitions** - 6+ different definitions across the codebase
2. **Magic Numbers** - 180+ hardcoded values that should use constants
3. **Duplicate Configuration** - Core and Dashboard have separate constant files
4. **Semantic Mismatches** - Mapping between French (Phase 0) and English (Dashboard) fault names is incorrect

---

## 1. Fault Type Inconsistencies (CRITICAL)

### 1.1 Summary Table

| Location                               | Count | Type                        |
| -------------------------------------- | ----- | --------------------------- |
| `dashboard_config.py`                  | 11    | English (Dashboard UI)      |
| `xai_dashboard.py`                     | 11    | English (Different!)        |
| `data_config.py` (Phase 0)             | 11    | French (Data Generation)    |
| `utils/constants.py` FAULT_TYPES       | 11    | French                      |
| `utils/constants.py` FAULT_CLASSES     | 11    | English (matches dashboard) |
| `utils/constants.py` FAULT_LABELS_PINN | 11    | English (Physics/PINN)      |

### 1.2 Dashboard Config (`dashboard_config.py`)

```python
FAULT_CLASSES = [
    "normal", "ball_fault", "inner_race", "outer_race", "combined",
    "imbalance", "misalignment", "oil_whirl", "cavitation", "looseness", "oil_deficiency"
]
```

### 1.3 XAI Dashboard (`visualization/xai_dashboard.py`)

```python
FAULT_CLASSES = [
    'Healthy', 'Misalignment', 'Imbalance', 'Looseness',
    'Bearing Outer Race', 'Bearing Inner Race', 'Bearing Ball',
    'Gear Fault', 'Shaft Bent', 'Rotor Rub', 'Combined Fault'
]
```

> [!CAUTION]
> **XAI Dashboard uses non-existent fault types!**  
> `Gear Fault`, `Shaft Bent`, `Rotor Rub` are **never generated** by the data generation system.
> This means XAI visualizations for these classes would fail or show garbage data.

### 1.4 Phase 0 Data Generation (`config/data_config.py`)

```python
# Single faults (French names)
single_faults = {
    'desalignement': True,  # Misalignment
    'desequilibre': True,   # Imbalance
    'jeu': True,            # Looseness
    'lubrification': True,  # Lubrication issues
    'cavitation': True,     # Cavitation
    'usure': True,          # Wear
    'oilwhirl': True,       # Oil whirl
}

# Generated fault list includes:
# sain, desalignement, desequilibre, jeu, lubrification,
# cavitation, usure, oilwhirl, mixed_misalign_imbalance,
# mixed_wear_lube, mixed_cavit_jeu
```

### 1.5 Mapping Errors in `DASHBOARD_TO_PHASE0_FAULT_MAP`

```python
DASHBOARD_TO_PHASE0_FAULT_MAP = {
    "normal": "sain",                    # ✓ Correct
    "misalignment": "desalignement",     # ✓ Correct
    "imbalance": "desequilibre",         # ✓ Correct
    "looseness": "jeu",                  # ✓ Correct
    "oil_deficiency": "lubrification",   # ✓ Correct
    "cavitation": "cavitation",          # ✓ Correct
    "ball_fault": "usure",               # ⚠️ Questionable - usure = wear, not ball fault
    "oil_whirl": "oilwhirl",             # ✓ Correct
    "inner_race": "mixed_wear_lube",     # ❌ WRONG - inner_race is NOT a mixed fault!
    "outer_race": "mixed_cavit_jeu",     # ❌ WRONG - outer_race is NOT a mixed fault!
    "combined": "mixed_misalign_imbalance"  # ⚠️ Questionable mapping
}
```

### 1.6 PINN Labels (`utils/constants.py`)

```python
FAULT_LABELS_PINN = {
    0: 'healthy',      # Phase 0: sain (index 0) ✓
    1: 'misalignment', # Phase 0: desalignement (index 1) ✓
    2: 'imbalance',    # Phase 0: desequilibre (index 2) ✓
    3: 'outer_race',   # Phase 0: jeu (index 3) ❌ MISMATCH!
    4: 'inner_race',   # Phase 0: lubrification (index 4) ❌ MISMATCH!
    5: 'ball',         # Phase 0: cavitation (index 5) ❌ MISMATCH!
    6: 'looseness',    # Phase 0: usure (index 6) ❌ MISMATCH!
    7: 'oil_whirl',    # Phase 0: oilwhirl (index 7) ✓
    8: 'cavitation',   # Phase 0: mixed_misalign_imbalance (index 8) ❌ MISMATCH!
    9: 'wear',         # Phase 0: mixed_wear_lube (index 9) ❌ MISMATCH!
    10: 'lubrication'  # Phase 0: mixed_cavit_jeu (index 10) ❌ MISMATCH!
}
```

> [!WARNING]
> **PINN label indices don't match Phase 0 indices!**  
> This would cause models trained on Phase 0 data to predict completely wrong classes when used with PINN-based interpretability.

### 1.7 Resolution Options

| Option                         | Description                                             | Effort | Risk                        |
| ------------------------------ | ------------------------------------------------------- | ------ | --------------------------- |
| **A. Unify to Phase 0 French** | Use French names everywhere, translate only for display | High   | Medium                      |
| **B. Unify to English**        | Update Phase 0 to use English names                     | Medium | High (breaks existing data) |
| **C. Create Mapping Layer**    | Central translation service                             | Medium | Low                         |
| **D. Dual Names**              | Store both French and English in datasets               | Low    | Low                         |

**Recommended:** Option C - Create a central `FaultTypeRegistry` class.

---

## 2. Duplicate Constant Definitions

### 2.1 Two Separate Constant Files

| File                                    | Location     | Purpose                      |
| --------------------------------------- | ------------ | ---------------------------- |
| `utils/constants.py`                    | Project root | Core ML/training constants   |
| `packages/dashboard/utils/constants.py` | Dashboard    | Dashboard-specific constants |

### 2.2 Conflicting Values

| Constant                         | Core (`utils/constants.py`) | Dashboard (`dashboard/utils/constants.py`) |
| -------------------------------- | --------------------------- | ------------------------------------------ |
| `RF_N_ESTIMATORS_MIN`            | 100                         | 10                                         |
| `RF_N_ESTIMATORS_MAX`            | 1000                        | 500                                        |
| `RF_MAX_DEPTH_MIN`               | 10                          | 2                                          |
| `RF_MAX_DEPTH_MAX`               | 100                         | 50                                         |
| `NN_FILTERS_MIN`                 | 32                          | 16                                         |
| `DEFAULT_ONNX_OPSET_VERSION`     | 14                          | 11                                         |
| `PROGRESSIVE_START_SIZE_DEFAULT` | 51200                       | 50                                         |
| `PROGRESSIVE_END_SIZE_DEFAULT`   | 102400                      | 100                                        |
| `SIGNALS_PER_MINUTE_GENERATION`  | 50                          | 100                                        |

> [!IMPORTANT]
> These differences mean the dashboard may configure training with parameters outside what the core training code expects!

### 2.3 Resolution Options

| Option                        | Description                                                            |
| ----------------------------- | ---------------------------------------------------------------------- |
| **A. Merge to Single File**   | One authoritative `constants.py`                                       |
| **B. Dashboard Imports Core** | Dashboard imports from `utils.constants`, adds only dashboard-specific |
| **C. Configuration YAML**     | Move all configurable values to YAML with schema validation            |

**Recommended:** Option B - Dashboard should import from core, not duplicate.

---

## 3. Hardcoded Magic Numbers

### 3.1 Signal Length (102400)

Found **50+ occurrences** of hardcoded `102400` instead of using `SIGNAL_LENGTH`:

```
visualization/cnn_visualizer.py:441:  input_length=102400
visualization/cnn_analysis.py:581:    torch.randn(1, 1, 102400)
scripts/train_cnn.py:196:             'input_length': 102400
scripts/inference_cnn.py:142:         input_length=102400
tests/test_data_generation.py:575:    shape[1], 102400
```

### 3.2 Sampling Rate (20480)

Found **40+ occurrences** of hardcoded `20480` instead of using `SAMPLING_RATE`:

```
visualization/spectrogram_plots.py:48:  fs: float = 20480
packages/core/training/physics_loss_functions.py:41: sample_rate: int = 20480
packages/core/models/pinn/physics_constrained_cnn.py:56: sample_rate: int = 20480
```

### 3.3 Number of Classes (11)

Found **70+ occurrences** of hardcoded `11` instead of using `NUM_CLASSES`:

```
tests/test_models.py:25:    model = CNN1D(num_classes=11)
scripts/research/ablation_study.py:358: num_classes: int = 11
packages/core/models/transformer/tsmixer.py:18: num_classes=11
```

### 3.4 Resolution

Replace all hardcoded values with constant imports:

```python
# Before
model = CNN1D(num_classes=11, input_length=102400)

# After
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
model = CNN1D(num_classes=NUM_CLASSES, input_length=SIGNAL_LENGTH)
```

---

## 4. Milestones Folder Duplication

### 4.1 Issue

The `milestones/` folder contains **3 copies** of utility files:

```
milestones/milestone-1/utils/constants.py
milestones/milestone-2/utils/constants.py
milestones/milestone-3/utils/constants.py
```

Each may have diverged from the main `utils/constants.py`.

### 4.2 Resolution Options

| Option                   | Description                                         |
| ------------------------ | --------------------------------------------------- |
| **A. Delete Milestones** | Archive/remove milestone folders                    |
| **B. Symlinks**          | Replace with symlinks to main utils                 |
| **C. Leave As-Is**       | Milestones are historical snapshots (document this) |

**Recommended:** Option A or C - Milestones should either be archived or clearly marked as frozen snapshots.

---

## 5. Model Type Inconsistencies

### 5.1 Model Registry in Dashboard

```python
# dashboard/utils/constants.py
MODEL_TYPES = [
    "random_forest", "svm", "cnn1d", "resnet18", "resnet34", "resnet50",
    "efficientnet", "transformer", "spectrogram_cnn", "pinn", "ensemble"
]
```

### 5.2 Actual Models Available

In `packages/core/models/`:

- `cnn/cnn_1d.py` - CNN1D ✓
- `cnn/resnet_1d.py` - ResNet1D (18, 34, 50) ✓
- `transformer/` - VisionTransformer1D, PatchTST, TSMixer ✓
- `spectrogram_cnn/` - SpectrogramCNN, DualStreamCNN ✓
- `pinn/` - PhysicsConstrainedCNN, MultitaskPINN ✓
- `ensemble/` - EnsembleClassifier ✓
- `hybrid/` - CNNTransformerHybrid ❌ **Not in MODEL_TYPES!**

### 5.3 Missing from MODEL_TYPES

- `cnn_transformer` (Hybrid model)
- `vit_1d` (Vision Transformer 1D)
- `patchtst` (PatchTST)
- `tsmixer` (TSMixer)
- `multitask_pinn` (Multitask PINN)
- `dual_stream_cnn` (Dual Stream Spectrogram CNN)

---

## 6. Import Path Inconsistencies

### 6.1 Issue

Some files import from relative paths, others from absolute:

```python
# Relative (fragile)
from ..utils.constants import NUM_CLASSES

# Absolute (preferred)
from utils.constants import NUM_CLASSES

# Dashboard-specific (creates coupling)
from dashboard_config import FAULT_CLASSES
```

### 6.2 Resolution

Standardize on absolute imports with proper `__init__.py` exports.

---

## 7. Summary of Recommendations

| Priority | Issue                                             | Recommended Fix                          |
| -------- | ------------------------------------------------- | ---------------------------------------- |
| **P0**   | PINN labels don't match Phase 0 indices           | Fix `FAULT_LABELS_PINN` to match Phase 0 |
| **P0**   | XAI Dashboard has non-existent faults             | Update to use actual fault types         |
| **P0**   | Mapping errors in `DASHBOARD_TO_PHASE0_FAULT_MAP` | Fix incorrect mappings                   |
| **P1**   | Duplicate constants in dashboard                  | Dashboard imports from core              |
| **P1**   | 180+ hardcoded magic numbers                      | Replace with constant references         |
| **P2**   | Missing models in MODEL_TYPES                     | Add hybrid/transformer variants          |
| **P2**   | Milestones folder duplication                     | Archive or document as frozen            |
| **P3**   | Import path inconsistencies                       | Standardize on absolute imports          |

---

## 8. Files Requiring Changes

### High Priority (Fault Types)

1. `packages/dashboard/dashboard_config.py` - Fix mapping
2. `visualization/xai_dashboard.py` - Use actual fault types
3. `utils/constants.py` - Fix FAULT_LABELS_PINN

### Medium Priority (Constants)

4. `packages/dashboard/utils/constants.py` - Import from core instead of duplicating
5. All files with hardcoded `102400`, `20480`, `11` - Use constants

### Low Priority (Documentation)

6. `milestones/` folder - Archive or document

---

## Appendix: Search Commands Used

```powershell
# Find fault class definitions
grep -r "FAULT_CLASSES" --include="*.py"

# Find hardcoded signal length
grep -r "102400" --include="*.py"

# Find hardcoded sampling rate
grep -r "20480" --include="*.py"

# Find hardcoded num_classes
grep -rE "num_classes.*=.*11" --include="*.py"
```
