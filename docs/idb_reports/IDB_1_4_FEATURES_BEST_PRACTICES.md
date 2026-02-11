# IDB 1.4 Features Sub-Block — Best Practices

**Block ID:** IDB 1.4  
**Domain:** Core ML Engine  
**Purpose:** Curated best practices from the Features module for adoption across codebase  
**Date:** 2026-01-23

---

## 1. Feature Naming Conventions

### 1.1 Canonical Naming Standard

```python
# GOOD: PascalCase for feature names (matches scientific literature)
FEATURE_NAMES = [
    'RMS',              # Acronyms in caps
    'Kurtosis',         # Statistical terms capitalized
    'SpectralCentroid', # Compound words in PascalCase
    'WaveletEnergy_D1', # Use underscore for subscripts/indices
    'Harmonic2X1X',     # Include multipliers in name
]
```

### 1.2 Naming Rules

| Rule                                          | Example                          | Rationale                                  |
| --------------------------------------------- | -------------------------------- | ------------------------------------------ |
| Use **PascalCase** for feature names          | `SpectralEntropy`                | Matches IEEE/signal processing conventions |
| Keep **acronyms uppercase**                   | `RMS`, `FFT`, `DWT`              | Standard abbreviations stay readable       |
| Use **underscore for indices**                | `WaveletEnergy_D1`               | Distinguishes decomposition levels         |
| Include **physical units in docs**, not names | `DominantFreq` (Hz in docstring) | Keeps names concise                        |
| Prefix domain for disambiguation              | `EnvelopeRMS` vs `RMS`           | Avoids feature name collisions             |

### 1.3 Dual Convention Pattern (for backward compatibility)

```python
def extract_time_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
    """Returns both conventions for backward compatibility."""
    result = dict(time_features)  # Original: 'RMS', 'Kurtosis'

    # Add lowercase aliases for Python convention
    if 'RMS' in result:
        result['rms'] = result['RMS']
    if 'Kurtosis' in result:
        result['kurtosis'] = result['Kurtosis']

    return result
```

> [!TIP]
> When transitioning naming conventions, maintain both for one major version before deprecating the old style.

---

## 2. Vectorization Patterns

### 2.1 Prefer NumPy Vectorized Operations

```python
# GOOD: Vectorized computation
def compute_rms(signal: np.ndarray) -> float:
    return np.sqrt(np.mean(signal ** 2))

# GOOD: Vectorized band energy
def compute_band_energy(psd: np.ndarray, freqs: np.ndarray,
                        band_range: Tuple[float, float]) -> float:
    mask = (freqs >= band_range[0]) & (freqs <= band_range[1])
    return np.sum(psd[mask])  # Boolean indexing, no loops
```

### 2.2 Batch Processing Pattern

```python
# GOOD: Pre-allocate output array
def extract_batch(self, signals: np.ndarray) -> np.ndarray:
    n_signals = signals.shape[0]
    features_matrix = np.zeros((n_signals, len(self.feature_names_)))

    for i in range(n_signals):
        features_matrix[i] = self.extract_features(signals[i])

    return features_matrix
```

### 2.3 FFT Optimization Pattern

```python
# GOOD: One-sided spectrum with proper normalization
def compute_fft(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    N = len(signal)
    fft_vals = fft(signal)
    psd = (2.0 / N) * np.abs(fft_vals[:N // 2])  # One-sided, normalized
    freqs = fftfreq(N, 1.0 / fs)[:N // 2]
    return freqs, psd
```

### 2.4 Avoid Python Loops for Numerical Operations

```python
# BAD: Python loop for element-wise operation
def compute_energy_bad(coeffs):
    energy = 0
    for c in coeffs:
        energy += c ** 2
    return energy

# GOOD: NumPy vectorized
def compute_energy_good(coeffs):
    return np.sum(np.array(coeffs) ** 2)
```

---

## 3. Numerical Stability Practices

### 3.1 Division Guard Pattern

```python
# GOOD: Always guard against division by zero
def compute_crest_factor(signal: np.ndarray) -> float:
    peak = np.max(np.abs(signal))
    rms = compute_rms(signal)
    return peak / rms if rms > 0 else 0.0  # Explicit guard

# GOOD: Alternative with epsilon
def compute_ratio_safe(numerator: float, denominator: float) -> float:
    return numerator / (denominator + 1e-12)  # Epsilon prevents div-by-zero
```

### 3.2 Epsilon Usage Guidelines

| Context                   | Epsilon Value      | Rationale                         |
| ------------------------- | ------------------ | --------------------------------- |
| Division guard            | `1e-12`            | Below float64 precision threshold |
| Log arguments             | `1e-12`            | Prevents log(0) = -inf            |
| Probability normalization | `1e-12`            | Keeps sum ≈ 1.0                   |
| IQR-based normalization   | Replace 0 with 1.0 | Stronger guard for scale factor   |

### 3.3 Entropy Computation Pattern

```python
# GOOD: Safe entropy with normalized probabilities
def compute_spectral_entropy(psd: np.ndarray) -> float:
    # Normalize to probability distribution
    psd_norm = psd / (np.sum(psd) + 1e-12)
    # Remove zeros before log
    psd_norm = psd_norm[psd_norm > 0]
    # Compute entropy with epsilon in log
    entropy = -np.sum(psd_norm * np.log(psd_norm + 1e-12))
    return entropy
```

### 3.4 IQR-Based Robust Normalization

```python
# GOOD: Handle zero IQR gracefully
def fit(self, X: np.ndarray, y=None):
    self.median_ = np.median(X, axis=0)
    q75 = np.percentile(X, 75, axis=0)
    q25 = np.percentile(X, 25, axis=0)
    self.iqr_ = q75 - q25
    # Prevent division by zero for constant features
    self.iqr_[self.iqr_ == 0] = 1.0
    return self
```

---

## 4. Feature Validation Patterns

### 4.1 Validate Before Use Pattern

```python
def validate_feature_vector(features: np.ndarray,
                            expected_dim: int = 36) -> Tuple[bool, str]:
    """Validate a single feature vector."""
    # Check shape
    if features.shape[0] != expected_dim:
        return False, f"Expected {expected_dim} features, got {features.shape[0]}"

    # Check for NaN
    if np.any(np.isnan(features)):
        nan_indices = np.where(np.isnan(features))[0]
        return False, f"Found NaN values at indices: {nan_indices.tolist()}"

    # Check for Inf
    if np.any(np.isinf(features)):
        inf_indices = np.where(np.isinf(features))[0]
        return False, f"Found Inf values at indices: {inf_indices.tolist()}"

    return True, "Valid"
```

### 4.2 Distribution Warning Pattern

```python
def check_feature_distribution(features: np.ndarray,
                               feature_names: Optional[List[str]] = None,
                               variance_threshold: float = 1e-6) -> List[str]:
    """Warn about problematic features."""
    warnings_list = []

    # Check for near-constant features (low variance)
    variances = np.var(features, axis=0)
    low_var_idx = np.where(variances < variance_threshold)[0]
    if len(low_var_idx) > 0:
        warnings_list.append(f"Low variance features: {low_var_idx.tolist()}")

    # Check for outliers (> 5 std from mean)
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    for i in range(features.shape[1]):
        if stds[i] > 0:
            outliers = np.abs(features[:, i] - means[i]) > 5 * stds[i]
            if np.sum(outliers) > 0.05 * features.shape[0]:  # >5% outliers
                warnings_list.append(f"Feature {i}: excessive outliers")

    return warnings_list
```

### 4.3 NaN Replacement Strategies

```python
def replace_nans(features: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """Replace NaN values with imputed values."""
    features_clean = features.copy()

    for i in range(features.shape[1]):
        col = features[:, i]
        if np.any(np.isnan(col)):
            if strategy == 'mean':
                replacement = np.nanmean(col)
            elif strategy == 'median':
                replacement = np.nanmedian(col)
            elif strategy == 'zero':
                replacement = 0.0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            features_clean[np.isnan(col), i] = replacement

    return features_clean
```

### 4.4 Comprehensive Validation Pipeline

```python
# Run all validations in sequence
def validate_feature_matrix(features: np.ndarray,
                           labels: Optional[np.ndarray] = None,
                           feature_names: Optional[List[str]] = None,
                           verbose: bool = True) -> bool:
    all_valid = True

    # 1. Check for NaNs
    nan_dict = check_for_nans(features, feature_names)
    if nan_dict:
        all_valid = False
        if verbose:
            print(f"WARNING: NaN values in features: {nan_dict}")

    # 2. Check for Infs
    if np.any(np.isinf(features)):
        all_valid = False

    # 3. Check distributions
    distribution_warnings = check_feature_distribution(features, feature_names)
    if distribution_warnings and verbose:
        for warning in distribution_warnings:
            print(f"WARNING: {warning}")

    # 4. Check class balance if labels provided
    if labels is not None:
        unique, counts = np.unique(labels, return_counts=True)
        imbalance_ratio = np.max(counts) / np.min(counts)
        if imbalance_ratio > 10 and verbose:
            print(f"WARNING: High class imbalance (ratio: {imbalance_ratio:.1f})")

    return all_valid
```

---

## 5. Documentation Requirements for Features

### 5.1 Required Docstring Elements

```python
def compute_spectral_centroid(psd: np.ndarray, freqs: np.ndarray) -> float:
    """
    Compute spectral centroid (center of mass of spectrum).

    Centroid = sum(f * P(f)) / sum(P(f))              # ← Formula

    High centroid indicates energy at higher frequencies.  # ← Interpretation

    Args:
        psd: Power spectral density                   # ← Precise type description
        freqs: Frequency bins (Hz)                    # ← Units when applicable

    Returns:
        Spectral centroid (Hz)                        # ← Units in return

    Example:                                          # ← Usage example
        >>> freqs, psd = compute_fft(signal, fs=20480)
        >>> centroid = compute_spectral_centroid(psd, freqs)
        >>> print(f"Centroid: {centroid:.2f} Hz")
    """
```

### 5.2 Documentation Checklist

| Element                     | Required                | Example                                        |
| --------------------------- | ----------------------- | ---------------------------------------------- |
| **Purpose statement**       | ✅                      | "Compute spectral centroid..."                 |
| **Mathematical formula**    | ✅ for derived features | `Centroid = sum(f * P(f)) / sum(P(f))`         |
| **Physical interpretation** | ✅                      | "High values indicate high-freq content"       |
| **Args with types**         | ✅                      | `psd: Power spectral density array`            |
| **Args with units**         | ✅ when applicable      | `fs: Sampling frequency (Hz)`                  |
| **Returns with units**      | ✅                      | `Spectral centroid (Hz)`                       |
| **Example usage**           | Recommended             | Doctest format preferred                       |
| **Reference to literature** | Recommended             | "Reference: Section 8.2.3 of technical report" |

### 5.3 Module-Level Documentation

```python
"""
Frequency-domain spectral features for vibration signal analysis.

Purpose:
    Compute 12 frequency-domain features including dominant frequency,
    spectral centroid, entropy, band energies, and harmonic ratios.

Reference: Section 8.2.3 of technical report

Author: Syed Abbas Ahmad
Date: 2025-11-19
"""
```

### 5.4 Feature Name Documentation Pattern

```python
def _get_feature_names(self) -> List[str]:
    """
    Define canonical ordering of feature names.

    Returns:
        List of 36 feature names in extraction order
    """
    # Time-domain (7) - Statistical moments and shape factors
    time_names = [
        'RMS',           # Root mean square amplitude
        'Kurtosis',      # 4th moment - impulsiveness indicator
        'Skewness',      # 3rd moment - distribution asymmetry
        'CrestFactor',   # Peak/RMS ratio
        'ShapeFactor',   # RMS/mean ratio
        'ImpulseFactor', # Peak/mean ratio
        'ClearanceFactor' # Peak/(mean sqrt)^2
    ]
    # ... continue for all domains
```

---

## 6. Sklearn Compatibility Pattern

```python
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureNormalizer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible normalizer for pipeline integration."""

    def __init__(self, method: str = 'standard'):
        self.method = method

    def fit(self, X: np.ndarray, y=None):
        """Fit on training data. Returns self for chaining."""
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features."""
        return self.scaler.transform(X)

    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility."""
        return {'method': self.method}

    def set_params(self, **params):
        """Set parameters for sklearn compatibility."""
        if 'method' in params:
            self.method = params['method']
        return self
```

---

## 7. Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURES BEST PRACTICES                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  NAMING           │  PascalCase for features (SpectralCentroid) │
│                   │  Uppercase acronyms (RMS, FFT)               │
│                   │  Underscore for indices (WaveletEnergy_D1)   │
│                                                                  │
│  NUMERICAL        │  Guard divisions: x / y if y > 0 else 0.0   │
│  STABILITY        │  Use epsilon 1e-12 for log/division          │
│                   │  Replace zero IQR with 1.0                   │
│                                                                  │
│  VECTORIZATION    │  Prefer np.sum/np.mean over Python loops     │
│                   │  Use boolean indexing for band selection     │
│                   │  Pre-allocate output arrays                  │
│                                                                  │
│  VALIDATION       │  Check NaN/Inf before use                    │
│                   │  Warn on low variance (< 1e-6)               │
│                   │  Report outliers (> 5 std)                   │
│                                                                  │
│  DOCUMENTATION    │  Include formula in docstring                │
│                   │  Specify units (Hz, seconds)                 │
│                   │  Add usage examples                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

_Best Practices curated from IDB 1.4 Features Sub-Block_  
_For adoption across codebase — 2026-01-23_
