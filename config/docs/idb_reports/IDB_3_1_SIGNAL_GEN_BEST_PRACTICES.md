# IDB 3.1: Signal Generation Best Practices

**Domain:** Data Engineering  
**Curator:** AI Agent  
**Date:** 2026-01-23  
**Source Files:** `signal_generator.py`, `signal_augmentation.py`, `spectrogram_generator.py`

---

## 1. Physics Parameter Conventions

### 1.1 Sommerfeld Number Calculation Pattern ⭐

**Pattern:** Calculate dimensionless bearing parameters from operating conditions.

```python
# CORRECT: Temperature-dependent viscosity with Arrhenius approximation
viscosity_factor = np.exp(-0.03 * (temperature_C - 60))  # ~3%/°C

# CORRECT: Sommerfeld proportional to speed, inverse to load
sommerfeld = (sommerfeld_base * viscosity_factor *
              speed_factor * (1.0 / load_factor))

# CORRECT: Clamp to physically realistic range
sommerfeld = np.clip(sommerfeld, 0.05, 0.5)
```

**Key Conventions:**
| Parameter | Relationship | Rationale |
|-----------|-------------|-----------|
| Temperature | Exponential decay of viscosity | Arrhenius approximation |
| Speed | Linear proportionality | Definition of Sommerfeld |
| Load | Inverse relationship | Higher load → thinner film |
| Clamping | [0.05, 0.5] | Avoid unphysical instabilities |

---

### 1.2 Fault Amplitude Conventions

**Pattern:** Scale fault signatures relative to baseline amplitude.

```python
# Base amplitudes by fault type (normalized to 1.0 baseline)
FAULT_AMPLITUDES = {
    'misalignment_2X': 0.35,  # Strong second harmonic
    'misalignment_3X': 0.20,  # Weaker third harmonic
    'imbalance_1X': 0.50,     # Dominant first harmonic
    'wear_broadband': 0.25,   # Distributed energy
    'cavitation_burst': 0.60, # High-energy events
}
```

**Convention:** Document amplitude ratios with physical justification in comments.

---

### 1.3 Frequency Conventions

**Pattern:** Express frequencies relative to shaft speed (Omega).

```python
# Sub-synchronous ratios
oil_whirl_ratio = 0.42 + 0.06 * np.random.rand()  # 0.42-0.48× shaft
clearance_sub_ratio = 0.43 + 0.05 * np.random.rand()  # Similar range

# Absolute frequencies (Hz) for non-rotational phenomena
stick_slip_freq = 2 + 3 * np.random.rand()  # 2-5 Hz
cavitation_burst_freq = 1500 + 1000 * np.random.rand()  # 1.5-2.5 kHz
```

**Convention:** Use `X` notation for harmonics (1X, 2X, 3X) and ratios for sub-synchronous.

---

## 2. Signal Generation Patterns

### 2.1 Layered Noise Architecture ⭐

**Pattern:** Apply independent noise sources sequentially with boolean toggles.

```python
class NoiseGenerator:
    def apply_noise_layers(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, bool]]:
        x_noisy = x.copy()
        applied = {}

        # Each layer independent and configurable
        if self.config.noise.measurement:
            x_noisy += self._measurement_noise()
            applied['measurement'] = True
        else:
            applied['measurement'] = False

        # ... repeat for each layer

        return x_noisy, applied  # Return tracking dict
```

**Benefits:**

- Ablation studies (toggle individual sources)
- Reproducible noise composition
- Metadata tracks what was applied

---

### 2.2 Severity-Curve Modulation Pattern ⭐

**Pattern:** Use time-varying severity curves multiplied with fault signatures.

```python
# Generate severity curve (constant or evolving)
if has_temporal_evolution:
    severity_curve = np.linspace(severity_start, severity_end, N)
else:
    severity_curve = severity_factor * np.ones(N)

# Apply to fault signature
x_fault = severity_curve * fault_signature * transient_modulation
```

**Convention:** Separate severity from fault physics for independent control.

---

### 2.3 Transient Modulation Pattern

**Pattern:** Multiplicative modulation for non-stationary behavior.

```python
transient_modulation = np.ones(N)  # Default: no modulation

if transient_type == 'speed_ramp':
    transient_modulation[start:end] = np.linspace(0.85, 1.15, end - start)

elif transient_type == 'load_step':
    transient_modulation[:step_idx] = 0.7
    transient_modulation[step_idx:] = 1.0

elif transient_type == 'thermal_expansion':
    transient_modulation = 0.9 + 0.2 * (1 - np.exp(-t / tau))
```

**Convention:** Keep modulation as separate array for easy visualization.

---

### 2.4 Mixed Fault Composition Pattern

**Pattern:** Additive combination of individual fault signatures.

```python
# CORRECT: Additive superposition (not multiplicative!)
def generate_mixed_fault(self, ...):
    fault_a = self._generate_component_a(severity_a)
    fault_b = self._generate_component_b(severity_b)

    # Linear superposition preserves fault characteristics
    return severity_curve * (fault_a + fault_b) * transient_modulation
```

**Rationale:** Physical signals superimpose linearly in the time domain.

---

## 3. Validation Patterns

### 3.1 Metadata Population Pattern ⭐

**Pattern:** Comprehensive dataclass capturing all generation parameters.

```python
@dataclass
class SignalMetadata:
    # Fault information
    fault: str
    severity: str
    severity_factor_initial: float

    # Operating conditions
    speed_rpm: float
    load_percent: float
    temperature_C: float

    # Physics parameters (for verification)
    sommerfeld_number: float
    sommerfeld_calculated: bool  # Track if calculated or random

    # Signal properties (post-generation validation)
    signal_rms: float
    signal_peak: float
    signal_crest_factor: float

    # Provenance
    generation_timestamp: str
    generator_version: str
    rng_seed: int
```

**Convention:** Include both input parameters AND computed signal properties.

---

### 3.2 Post-Generation Metrics Pattern

**Pattern:** Compute signal statistics after generation for validation.

```python
metadata = SignalMetadata(
    # ... other fields ...
    signal_rms=float(np.sqrt(np.mean(x**2))),
    signal_peak=float(np.max(np.abs(x))),
    signal_crest_factor=float(np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-10)),
)
```

**Convention:** Add epsilon (1e-10) to avoid division by zero.

---

### 3.3 Augmentation Tracking Pattern

**Pattern:** Return augmentation parameters as dict for reproducibility.

```python
def _apply_augmentation(self, x, is_augmented) -> Dict[str, Any]:
    aug_params = {'method': 'none'}

    if is_augmented:
        if aug_method == 'time_shift':
            shift = np.random.randint(-max_shift, max_shift)
            x[:] = np.roll(x, shift)
            aug_params = {'method': 'time_shift', 'shift_samples': int(shift)}

    return aug_params  # Stored in metadata
```

---

## 4. Documentation Requirements for Physics

### 4.1 Fault Model Documentation Template ⭐

```python
def generate_fault_signal(self, fault_type: str, ...) -> np.ndarray:
    """
    Generate fault-specific vibration signature.

    Physics Model:
        [fault_type]: [Brief description of physical mechanism]

    Frequency Content:
        - [Harmonic]: [Amplitude ratio] (reason)
        - [Sub-sync]: [Frequency ratio] (physical basis)

    Scaling Factors:
        - Sommerfeld: [relationship] (e.g., inverse for lubrication)
        - Speed: [relationship] (e.g., squared for imbalance)
        - Load: [relationship]

    References:
        - [Technical report section]
        - [Literature reference if applicable]
    """
```

---

### 4.2 Noise Layer Documentation Template

```python
def apply_noise_layers(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Apply N-layer noise model to signal.

    Noise Sources:
        1. Measurement: Gaussian, σ = config.levels['measurement']
        2. EMI: 50-60 Hz sinusoidal interference
        3. Pink: 1/f spectrum via cumulative sum
        ...

    Each layer independently toggleable via config.noise.[layer_name].
    """
```

---

### 4.3 Constant Documentation Convention

**Pattern:** Group physics constants with inline documentation.

```python
# Bearing Physics Constants (from Section 7.3 Technical Report)
SOMMERFELD_BASE = 0.25      # Typical design Sommerfeld for journal bearings
SOMMERFELD_RANGE = (0.05, 0.5)  # Stable operating range

# Fault Frequency Ratios (empirically validated)
OIL_WHIRL_RATIO = (0.42, 0.48)  # Sub-sync, depends on bearing geometry
CAGE_ROTATION_RATIO = 0.4       # Typical rolling element cage frequency
```

---

## 5. Reproducibility Patterns

### 5.1 Global Seed Initialization Pattern ⭐

**Pattern:** Set seed at generator initialization, with optional per-signal variation.

```python
class SignalGenerator:
    def __init__(self, config):
        if config.rng_seed is not None:
            set_seed(config.rng_seed)  # Global initialization

    def generate_dataset(self):
        for i, fault in enumerate(faults):
            if self.config.per_signal_seed_variation:
                set_seed(self.config.rng_seed + total_signals)  # Deterministic variation
            signal, meta = self.generate_single_signal(fault)
```

---

### 5.2 Unified Random Module Pattern ⭐

**CRITICAL:** Use only ONE random source throughout.

```python
# CORRECT: Use numpy.random consistently
rng = np.random.default_rng(seed)
value = rng.random()
normal = rng.standard_normal(N)

# AVOID: Mixing random sources
import random  # ❌ Python's random
value = random.random()  # Not controlled by np.random seed!
```

---

### 5.3 Version Tracking Pattern

**Pattern:** Embed generator version in metadata.

```python
metadata = SignalMetadata(
    generator_version="Python_v1.0",
    generation_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    rng_seed=self.config.rng_seed,
)
```

---

### 5.4 Stratified Split Pattern

**Pattern:** Ensure class balance in train/val/test splits.

```python
from sklearn.model_selection import train_test_split

# Stratified split to maintain class distribution
X_train, X_test, y_train, y_test = train_test_split(
    signals, labels,
    test_size=0.15,
    stratify=labels,  # ← Key parameter
    random_state=config.rng_seed
)
```

---

## Quick Reference Card

| Category        | Best Practice                      | See Section |
| --------------- | ---------------------------------- | ----------- |
| Physics         | Clamp Sommerfeld to [0.05, 0.5]    | 1.1         |
| Physics         | Use X notation for harmonics       | 1.3         |
| Generation      | Layered noise with tracking dict   | 2.1         |
| Generation      | Severity-curve multiplication      | 2.2         |
| Generation      | Additive mixed faults              | 2.4         |
| Validation      | Dataclass with computed metrics    | 3.1         |
| Validation      | Epsilon in division (1e-10)        | 3.2         |
| Docs            | Reference technical report section | 4.1         |
| Reproducibility | Single random source (np.random)   | 5.2         |
| Reproducibility | Stratified splits with seed        | 5.4         |

---

_Best practices extracted from IDB 3.1 Signal Generation Sub-Block_
