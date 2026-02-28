# IDB 1.4 Features Sub-Block Analysis

**Block ID:** IDB 1.4  
**Domain:** Core ML Engine  
**Primary Directory:** `packages/core/features/`  
**Files Analyzed:** 12 files  
**Analysis Date:** 2026-01-22  
**Author:** AI Agent (Features Sub-Block Analyst)

---

## Executive Summary

The Features Sub-Block is a well-architected feature engineering module for classical ML-based bearing fault diagnosis. The implementation extracts **36 base features** across 5 domains (time, frequency, envelope, wavelet, bispectrum), with an optional **16 advanced features** for research purposes. The code demonstrates strong scientific grounding, comprehensive documentation, and reasonable numerical stability practices.

**Strengths:**

- Clean separation of concerns with domain-specific extraction modules
- Comprehensive feature set scientifically aligned with vibration analysis literature
- Well-documented with docstrings and usage examples
- Solid validation utilities for NaN/Inf detection

**Areas for Improvement:**

- Hardcoded default parameters should use centralized constants
- Missing SHAP integration despite documentation claims
- No streaming/incremental extraction support
- Limited parallelization in batch extraction

---

## Task 1: Current State Assessment

### 1.1 File Inventory (12 Files)

| File                       | Purpose                   | Lines | Key Functions/Classes                                                                         |
| -------------------------- | ------------------------- | ----- | --------------------------------------------------------------------------------------------- |
| `__init__.py`              | Module exports            | 20    | Exports `FeatureExtractor`, `FeatureSelector`, `FeatureNormalizer`                            |
| `feature_extractor.py`     | Main orchestrator         | 315   | `FeatureExtractor` class (36 features)                                                        |
| `feature_selector.py`      | Selection algorithms      | 294   | `FeatureSelector` (MRMR, variance), `VarianceThresholdSelector`                               |
| `feature_importance.py`    | Importance analysis       | 235   | RF Gini importance, permutation importance, visualization                                     |
| `feature_normalization.py` | Normalization             | 164   | `FeatureNormalizer` (z-score, minmax), `RobustNormalizer` (IQR-based)                         |
| `feature_validator.py`     | Validation utilities      | 234   | NaN checks, distribution analysis, replacement strategies                                     |
| `time_domain.py`           | Time-domain features      | 168   | 7 features: RMS, Kurtosis, Skewness, CrestFactor, ShapeFactor, ImpulseFactor, ClearanceFactor |
| `frequency_domain.py`      | Frequency-domain features | 225   | 12 features: DominantFreq, SpectralCentroid, entropy, band energies, harmonic ratios          |
| `envelope_analysis.py`     | Envelope features         | 146   | 4 features: EnvelopeRMS, EnvelopeKurtosis, EnvelopePeak, ModulationFreq                       |
| `wavelet_features.py`      | Wavelet features          | 222   | 7 features: WaveletEnergyRatio, WaveletKurtosis, energy levels, cepstral features             |
| `bispectrum.py`            | Bispectrum features       | 229   | 6 features: BispectrumPeak, Mean, Entropy, PhaseCoupling, NonlinearityIndex                   |
| `advanced_features.py`     | Optional advanced         | 351   | 16 features: CWT (4), WPT (4), nonlinear dynamics (4), spectrogram (4)                        |

### 1.2 Feature Types Extracted

**Total Base Features: 36** (matches target of 36, not 52 as originally specified)

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION PIPELINE                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Signal (1D array, 102400 samples @ 20480 Hz)             │
│                         │                                        │
│          ┌──────────────┼──────────────┐                        │
│          ▼              ▼              ▼                        │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐                  │
│   │Time Domain │ │Freq Domain │ │  Envelope  │                  │
│   │ 7 features │ │12 features │ │ 4 features │                  │
│   └────────────┘ └────────────┘ └────────────┘                  │
│          │              │              │                        │
│          └──────────────┼──────────────┘                        │
│                         ▼                                        │
│          ┌──────────────┼──────────────┐                        │
│          ▼              ▼              ▼                        │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐                  │
│   │  Wavelet   │ │ Bispectrum │ │ [Advanced] │                  │
│   │ 7 features │ │ 6 features │ │16 features │                  │
│   └────────────┘ └────────────┘ └────────────┘                  │
│          │              │              │                        │
│          └──────────────┼──────────────┘                        │
│                         ▼                                        │
│              Feature Vector (36 or 52)                          │
│                         │                                        │
│          ┌──────────────┼──────────────┐                        │
│          ▼              ▼              ▼                        │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐                  │
│   │Normalization│ │ Selection  │ │ Validation │                  │
│   │(z-score/mm)│ │(MRMR→15)  │ │(NaN check) │                  │
│   └────────────┘ └────────────┘ └────────────┘                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### Feature Breakdown by Domain

**Time Domain (7 features):**
| Feature | Formula | Physical Meaning |
|---------|---------|------------------|
| RMS | √(mean(x²)) | Signal energy/amplitude |
| Kurtosis | 4th moment | Impulsiveness (bearing defects) |
| Skewness | 3rd moment | Distribution asymmetry |
| CrestFactor | peak/RMS | Peak detection sensitivity |
| ShapeFactor | RMS/mean(\|x\|) | Waveform shape indicator |
| ImpulseFactor | peak/mean(\|x\|) | Impulse detection |
| ClearanceFactor | peak/(mean(√\|x\|))² | Clearance monitoring |

**Frequency Domain (12 features):**
| Feature | Description |
|---------|-------------|
| DominantFreq | Peak frequency in spectrum |
| SpectralCentroid | Center of mass of spectrum |
| SpectralEntropy | Shannon entropy (tonal vs. noise) |
| LowBandEnergy | Energy in 0-500 Hz |
| MidBandEnergy | Energy in 500-2000 Hz |
| HighBandEnergy | Energy in 2000-5000 Hz |
| VeryHighBandEnergy | Energy in 5000-10000 Hz |
| TotalSpectralPower | Sum of PSD |
| SpectralStd | Standard deviation of PSD |
| Harmonic2X1X | 2X/1X ratio (misalignment indicator) |
| Harmonic3X1X | 3X/1X ratio (looseness indicator) |
| SpectralPeakiness | Peak/mean ratio |

**Envelope (4 features):**
| Feature | Description |
|---------|-------------|
| EnvelopeRMS | RMS of Hilbert envelope |
| EnvelopeKurtosis | Envelope impulsiveness |
| EnvelopePeak | Maximum envelope value |
| ModulationFreq | Dominant modulation frequency |

**Wavelet (7 features):**
| Feature | Description |
|---------|-------------|
| WaveletEnergyRatio | Detail/total energy ratio |
| WaveletKurtosis | Mean kurtosis of detail coefficients |
| WaveletEnergy_D1 | Highest frequency detail energy |
| WaveletEnergy_D3 | Mid-frequency detail energy |
| WaveletEnergy_D5 | Low-frequency detail energy |
| CepstralPeakRatio | Peak/mean in cepstrum |
| QuefrencyCentroid | Centroid of cepstrum |

**Bispectrum (6 features):**
| Feature | Description |
|---------|-------------|
| BispectrumPeak | Peak bispectrum value |
| BispectrumMean | Mean bispectrum value |
| BispectrumEntropy | Entropy of bispectrum |
| PhaseCoupling | Quadratic phase coupling strength |
| NonlinearityIndex | Deviation from Gaussianity |
| BispectrumPeakRatio | Peak/mean ratio |

### 1.3 Extraction Pipeline Flow

```python
# Main entry point: FeatureExtractor.extract_features()
1. Input: signal (1D np.ndarray)
2. Extract time domain features → 7 features
3. Extract frequency domain features (requires fs) → 12 features
4. Extract envelope features (requires fs) → 4 features
5. Extract wavelet features (requires fs) → 7 features
6. Extract bispectrum features → 6 features
7. Combine into ordered dict
8. Convert to numpy array using canonical feature_names_ ordering
9. Output: feature_vector (36,)
```

### 1.4 Normalization/Validation Logic

**Normalization Options:**

- `FeatureNormalizer`: Wraps sklearn's `StandardScaler` (z-score) or `MinMaxScaler`
- `RobustNormalizer`: Custom IQR-based normalization (median, Q25-Q75)

**Validation Utilities:**

- `validate_feature_vector()`: Checks dimensionality, NaN, Inf values
- `check_feature_distribution()`: Warns about low-variance features, outliers (>5 std)
- `check_for_nans()`: Returns dict of NaN counts per feature
- `replace_nans()`: Imputation strategies (mean, median, zero)
- `validate_feature_matrix()`: Comprehensive validation with class balance check

### 1.5 Feature Importance Tools

| Tool                              | Method                  | Status                 |
| --------------------------------- | ----------------------- | ---------------------- |
| `get_random_forest_importances()` | Gini importance (MDI)   | ✅ Working             |
| `get_permutation_importances()`   | Sklearn permutation     | ✅ Working             |
| `plot_feature_importances()`      | Bar chart visualization | ✅ Working             |
| `plot_permutation_importances()`  | Error bar visualization | ✅ Working             |
| `compare_importances()`           | Side-by-side comparison | ✅ Working             |
| SHAP integration                  | SHapley values          | ❌ **NOT IMPLEMENTED** |

> [!WARNING]
> **SHAP is NOT implemented in `feature_importance.py`** despite being claimed in `INDEPENDENT_DEVELOPMENT_BLOCKS.md`. SHAP functionality exists only in `packages/core/explainability/` module (XAI block), not in the Features block.

---

## Task 2: Critical Issues Identification

### P0 (Critical) Issues

| ID   | Issue                      | Location      | Impact                                                                                                                         |
| ---- | -------------------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| P0-1 | **Feature count mismatch** | Documentation | IDB doc claims 52 features but implementation provides 36 base (or 52 with optional advanced). Inconsistency causes confusion. |

### P1 (High) Issues

| ID   | Issue                            | Location                                                                            | Impact                     | Recommendation                                             |
| ---- | -------------------------------- | ----------------------------------------------------------------------------------- | -------------------------- | ---------------------------------------------------------- |
| P1-1 | **Hardcoded `fs=20480`**         | `feature_extractor.py:46`, `wavelet_features.py:169`, `advanced_features.py:27,293` | Violates DRY; magic number | Use `from utils.constants import SAMPLING_RATE` as default |
| P1-2 | **Hardcoded `nfft=512`**         | `bispectrum.py:20`                                                                  | Hardcoded FFT size         | Add `DEFAULT_BISPECTRUM_NFFT` to constants                 |
| P1-3 | **No parallel batch extraction** | `feature_extractor.py:207-228`                                                      | Slow batch processing      | Add `n_jobs` parameter with `joblib.Parallel`              |
| P1-4 | **SHAP claim mismatch**          | `INDEPENDENT_DEVELOPMENT_BLOCKS.md:111`                                             | Misleading documentation   | Remove SHAP from Features IDB scope or implement           |

### P2 (Medium) Issues

| ID   | Issue                                    | Location                                                 | Impact                                                               | Recommendation                         |
| ---- | ---------------------------------------- | -------------------------------------------------------- | -------------------------------------------------------------------- | -------------------------------------- |
| P2-1 | **Hardcoded band ranges**                | `frequency_domain.py:197-200`                            | Not configurable: (0, 500), (500, 2000), (2000, 5000), (5000, 10000) | Move to constants or make configurable |
| P2-2 | **Hardcoded wavelet 'db4'**              | `wavelet_features.py:195-196`, `advanced_features.py:90` | Could be suboptimal for some signals                                 | Add wavelet selection parameter        |
| P2-3 | **Hardcoded level=5**                    | `wavelet_features.py:195`                                | Fixed decomposition level                                            | Make configurable                      |
| P2-4 | **Sample entropy O(N²)**                 | `advanced_features.py:121-160`                           | Very slow for long signals                                           | Add downsampling or faster algorithm   |
| P2-5 | **No NaN handling in extraction**        | `feature_extractor.py`                                   | NaN propagates                                                       | Add input validation                   |
| P2-6 | **Duplicate feature naming conventions** | `feature_extractor.py:92-133`                            | Both 'RMS' and 'rms', 'Kurtosis' and 'kurtosis'                      | Standardize on one convention          |
| P2-7 | **Missing return type hint**             | `feature_selector.py:61-99`                              | `fit()` should return `-> 'FeatureSelector'`                         | Add type hints                         |

### P3 (Low) Issues

| ID   | Issue                               | Location                 | Impact                                  |
| ---- | ----------------------------------- | ------------------------ | --------------------------------------- |
| P3-1 | Missing `__all__` in domain modules | `time_domain.py`, etc.   | Limits `from x import *` usage          |
| P3-2 | Inconsistent docstring format       | Various                  | Some use Google style, some plain       |
| P3-3 | No logging                          | All files                | No debug tracing capability             |
| P3-4 | Unused `sp_signal` import           | `frequency_domain.py:17` | Dead import                             |
| P3-5 | No feature versioning               | All                      | Cannot track feature definition changes |

### Numerical Stability Analysis

| Computation             | Potential Issue                | Current Mitigation         | Status |
| ----------------------- | ------------------------------ | -------------------------- | ------ |
| Division by zero        | CrestFactor, ShapeFactor, etc. | `if x > 0 else 0.0` guards | ✅ OK  |
| Log of zero             | Spectral entropy               | `+ 1e-12` epsilon          | ✅ OK  |
| Near-zero normalization | IQR-based                      | `iqr[iqr == 0] = 1.0`      | ✅ OK  |
| FFT edge cases          | Small signals                  | Handled in bispectrum      | ✅ OK  |
| Kurtosis of constant    | All constant values            | scipy handles gracefully   | ✅ OK  |

---

## Task 3: "If I Could Rewrite This" Retrospective

### 3.1 Is the Feature Set Scientifically Sound?

**Yes, largely sound.** The feature set represents a comprehensive coverage of classical vibration analysis literature:

**Strengths:**

- Time-domain features (RMS, kurtosis, crest factor) are industry standards for condition monitoring
- Envelope analysis with Hilbert transform is the gold standard for bearing fault detection
- Wavelet features capture multi-scale transients effectively
- Bispectrum features detect phase coupling (nonlinear fault signatures)
- MRMR selection prevents redundancy

**Concerns:**

1. **Missing bearing-specific frequencies**: No direct BPFI, BPFO, BSF, FTF extraction (ball pass frequencies)
2. **No adaptive filtering**: Fixed band ranges may miss faults at unexpected frequencies
3. **No order tracking**: Rotation speed variations not compensated
4. **Cepstral features in wavelet module**: Questionable placement (cepstrum != wavelet)

### 3.2 Features to Remove

| Feature                | Reason                                                   | Recommendation          |
| ---------------------- | -------------------------------------------------------- | ----------------------- |
| `BispectrumPeakRatio`  | Highly correlated with `BispectrumPeak / BispectrumMean` | Remove or keep only one |
| Dual naming convention | 'RMS' vs 'rms' duplicates confuse API                    | Keep only lowercase     |

### 3.3 Features to Add

| Feature                                        | Scientific Justification                 | Priority |
| ---------------------------------------------- | ---------------------------------------- | -------- |
| **BPFI/BPFO ratios**                           | Bearing fault frequencies are diagnostic | High     |
| **Spectral Kurtosis**                          | Detects transients in frequency bands    | High     |
| **Zero-crossing rate**                         | Simple, robust, computationally cheap    | Medium   |
| **Autocorrelation features**                   | Periodicity detection                    | Medium   |
| **Mel-frequency cepstral coefficients (MFCC)** | Popular in audio/vibration               | Low      |

### 3.4 Pipeline Efficiency for Streaming Data

**Current State: NOT streaming-ready**

| Issue                       | Impact                                      |
| --------------------------- | ------------------------------------------- |
| Full signal required        | Cannot process real-time streams            |
| No incremental computation  | Must recompute all features for each update |
| No windowing support        | Batch-only processing                       |
| FFT/DWT require full signal | Cannot stream partial results               |

**Recommended Architecture for Streaming:**

```
┌─────────────────────────────────────────────────────────┐
│                  STREAMING PIPELINE                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Stream Input → Window Buffer (5s) → Feature Extract    │
│                     ↓                                    │
│              Sliding Window                              │
│           (overlap: 50-75%)                              │
│                     ↓                                    │
│         Incremental Statistics                           │
│         (Welford's algorithm)                            │
│                     ↓                                    │
│         Running FFT (overlap-add)                        │
│                     ↓                                    │
│         Feature Cache (reuse)                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 3.5 Proposed Refactored Architecture

```python
# Proposed: FeatureRegistry pattern
class FeatureRegistry:
    """Central registry for feature definitions."""

    _features: Dict[str, FeatureDefinition] = {}

    @classmethod
    def register(cls, name: str, extractor: Callable, domain: str):
        cls._features[name] = FeatureDefinition(
            name=name,
            extractor=extractor,
            domain=domain,
            version="1.0"
        )

    @classmethod
    def extract(cls, signal: np.ndarray, feature_names: List[str]) -> Dict:
        return {name: cls._features[name].extract(signal) for name in feature_names}

# Usage:
FeatureRegistry.register("RMS", compute_rms, domain="time")
FeatureRegistry.register("SpectralCentroid", compute_spectral_centroid, domain="frequency")
```

**Benefits:**

- Plugin architecture for adding features
- Feature versioning
- Selective extraction
- Dependency injection for parameters

---

## Good Practices to Adopt

The Features block demonstrates several practices worth adopting across the codebase:

### 1. Documentation Excellence

```python
def compute_rms(signal: np.ndarray) -> float:
    """
    Compute Root Mean Square (RMS) value.

    RMS = sqrt(mean(x^2))

    Args:
        signal: Input signal array

    Returns:
        RMS value
    """
```

- Clear formula in docstring
- Type hints on all functions
- Example usage in complex functions

### 2. Defensive Division

```python
# Good pattern: Always guard against division by zero
return peak / rms if rms > 0 else 0.0
```

### 3. Separation of Concerns

```
feature_extractor.py (orchestrator) → delegates to:
├── time_domain.py (pure computation)
├── frequency_domain.py (pure computation)
├── envelope_analysis.py (pure computation)
├── wavelet_features.py (pure computation)
└── bispectrum.py (pure computation)
```

### 4. Sklearn Compatibility

```python
class FeatureNormalizer(BaseEstimator, TransformerMixin):
    """Compatible with sklearn pipelines."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.scaler.transform(X)
```

### 5. Canonical Feature Ordering

```python
# Maintains consistent ordering across save/load
self.feature_names_ = self._get_feature_names()
feature_vector = np.array([all_features[name] for name in self.feature_names_])
```

---

## Test Coverage Assessment

| Test File                           | Tests   | Coverage                                   |
| ----------------------------------- | ------- | ------------------------------------------ |
| `tests/unit/test_features.py`       | 9 tests | FeatureExtractor, normalization, selection |
| `tests/test_feature_engineering.py` | 6 tests | Extraction shape, batch, save/load, NaN    |

**Missing Test Coverage:**

- Individual domain extractors (`time_domain.py`, etc.) — no direct unit tests
- Advanced features extraction
- Feature validator functions
- Edge cases: very short signals, all-zero signals
- Feature importance functions

---

## Recommendations Summary

### Immediate Actions (P0-P1)

1. **Fix documentation**: Clarify 36 vs 52 feature count
2. **Replace hardcoded `fs=20480`** with `SAMPLING_RATE` constant import
3. **Add `DEFAULT_BISPECTRUM_NFFT`** to `utils/constants.py`
4. **Correct SHAP claim** in IDB documentation

### Short-term Improvements (P2)

5. **Add parallel batch extraction** with joblib
6. **Centralize all magic numbers** (band ranges, wavelet types, levels)
7. **Standardize naming convention** to lowercase
8. **Add input validation** in `extract_features()`

### Long-term Enhancements (P3+)

9. **Add bearing-specific features** (BPFI, BPFO)
10. **Implement feature versioning**
11. **Add streaming support** with windowed extraction
12. **Create FeatureRegistry** for plugin architecture
13. **Add comprehensive logging**

---

## Appendix: Feature Names List

```python
# Canonical 36 feature names (in extraction order)
FEATURE_NAMES = [
    # Time-domain (7)
    'RMS', 'Kurtosis', 'Skewness', 'CrestFactor',
    'ShapeFactor', 'ImpulseFactor', 'ClearanceFactor',
    # Frequency-domain (12)
    'DominantFreq', 'SpectralCentroid', 'SpectralEntropy',
    'LowBandEnergy', 'MidBandEnergy', 'HighBandEnergy',
    'VeryHighBandEnergy', 'TotalSpectralPower', 'SpectralStd',
    'Harmonic2X1X', 'Harmonic3X1X', 'SpectralPeakiness',
    # Envelope (4)
    'EnvelopeRMS', 'EnvelopeKurtosis', 'EnvelopePeak', 'ModulationFreq',
    # Wavelet (7)
    'WaveletEnergyRatio', 'WaveletKurtosis', 'WaveletEnergy_D1',
    'WaveletEnergy_D3', 'WaveletEnergy_D5', 'CepstralPeakRatio',
    'QuefrencyCentroid',
    # Bispectrum (6)
    'BispectrumPeak', 'BispectrumMean', 'BispectrumEntropy',
    'PhaseCoupling', 'NonlinearityIndex', 'BispectrumPeakRatio',
]

# Optional advanced features (16)
ADVANCED_FEATURE_NAMES = [
    # CWT (4)
    'CWT_TotalEnergy', 'CWT_PeakEnergy', 'CWT_EnergyRatio', 'CWT_DominantFreq',
    # WPT (4)
    'WPT_MaxEnergy', 'WPT_EnergyEntropy', 'WPT_EnergyStd', 'WPT_EnergyRatio',
    # Nonlinear (4)
    'SampleEntropy', 'ApproximateEntropy', 'DFA_Alpha', 'HurstExponent',
    # Time-frequency (4)
    'Spectrogram_Entropy', 'TF_Peak', 'TF_Mean', 'TF_Std',
]
```

---

_Report generated by IDB 1.4 Features Sub-Block Analyst_  
_Analysis completed: 2026-01-22_
