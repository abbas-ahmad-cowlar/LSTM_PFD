# Features API Reference

## Classes

---

### `FeatureExtractor`

> Main orchestrator for extracting all 36 base features from vibration signals.

**File:** `feature_extractor.py`

**Constructor:**

```python
FeatureExtractor(fs: float = 20480)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `fs` | `float` | `20480` | Sampling frequency in Hz |

**Methods:**

#### `extract_features(signal: np.ndarray) → np.ndarray`

Extract all 36 features from a single signal.

| Parameter | Type | Description |
|---|---|---|
| `signal` | `np.ndarray` | Input vibration signal (1D array) |

**Returns:** Feature vector of shape `(36,)` in canonical order (see [FEATURE_CATALOG.md](FEATURE_CATALOG.md)).

```python
extractor = FeatureExtractor(fs=20480)
signal = np.random.randn(102400)
features = extractor.extract_features(signal)
assert features.shape == (36,)
```

---

#### `extract_features_dict(signal: np.ndarray) → Dict[str, float]`

Extract features and return as a named dictionary.

**Returns:** Dictionary with 36 feature name → value pairs.

```python
features_dict = extractor.extract_features_dict(signal)
print(features_dict['RMS'])
print(features_dict['SpectralEntropy'])
```

---

#### `extract_time_domain_features(signal: np.ndarray) → Dict[str, float]`

Extract only time-domain features with dual naming convention.

**Returns:** Dictionary with both capitalized (`'RMS'`) and lowercase (`'rms'`) keys, plus additional `'mean'`, `'std'`, `'peak'` statistics.

```python
features = extractor.extract_time_domain_features(signal)
print(features['rms'])    # lowercase alias
print(features['RMS'])    # original name — same value
print(features['mean'])   # additional basic stat
```

---

#### `extract_frequency_domain_features(signal: np.ndarray) → Dict[str, float]`

Extract only frequency-domain features with dual naming convention.

**Returns:** Dictionary with both capitalized and lowercase keys.

---

#### `extract_batch(signals: np.ndarray) → np.ndarray`

Extract features from a batch of signals.

| Parameter | Type | Description |
|---|---|---|
| `signals` | `np.ndarray` | Batch of signals, shape `(n_signals, signal_length)` |

**Returns:** Feature matrix of shape `(n_signals, 36)`.

```python
signals = np.random.randn(100, 102400)
features = extractor.extract_batch(signals)
assert features.shape == (100, 36)
```

---

#### `get_feature_names() → List[str]`

Get ordered list of 36 feature names in canonical order.

---

#### `save_features(features: np.ndarray, filepath: Path) → None`

Save extracted features to `.npz` or `.npy` file.

#### `load_features(filepath: Path) → np.ndarray`

Load features from file.

---

### `FeatureSelector`

> Feature selection via MRMR or variance threshold. sklearn-compatible (fit/transform API).

**File:** `feature_selector.py`

**Constructor:**

```python
FeatureSelector(
    method: str = 'mrmr',
    n_features: int = 15,
    threshold: float = 0.01,
    random_state: int = 42
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | `str` | `'mrmr'` | Selection method: `'mrmr'` or `'variance'` |
| `n_features` | `int` | `15` | Number of features to select (MRMR only) |
| `threshold` | `float` | `0.01` | Variance threshold (variance method only) |
| `random_state` | `int` | `42` | Random seed for reproducibility |

**Methods:**

#### `fit(X, y=None, feature_names=None) → self`

Fit the selector. For MRMR, `y` is required.

MRMR algorithm:
1. Compute relevance: I(feature; target) for all features
2. Select feature with max relevance
3. Iteratively select features maximizing: relevance − mean(redundancy with selected)

```python
selector = FeatureSelector(method='mrmr', n_features=15)
selector.fit(X_train, y_train, feature_names=extractor.get_feature_names())
```

#### `transform(X: np.ndarray) → np.ndarray`

Select features from a feature matrix.

#### `get_selected_features() → List[int]`

Get indices of selected features.

#### `get_feature_names() → List[str]`

Get names of selected features (if names were provided during fit).

---

### `VarianceThresholdSelector`

> Removes features with variance below a threshold (constant or near-constant features).

**File:** `feature_selector.py`

**Constructor:**

```python
VarianceThresholdSelector(threshold: float = 0.01)
```

**Methods:** `fit(X, y=None)`, `transform(X)`, `get_selected_features()`.

```python
selector = VarianceThresholdSelector(threshold=0.01)
selector.fit(X_train)
X_filtered = selector.transform(X_train)
```

---

### `FeatureNormalizer`

> Z-score or min-max normalization. sklearn-compatible (BaseEstimator, TransformerMixin).

**File:** `feature_normalization.py`

**Constructor:**

```python
FeatureNormalizer(method: str = 'standard')
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | `str` | `'standard'` | `'standard'`, `'zscore'` (alias), or `'minmax'` |

**Methods:** `fit(X, y=None)`, `transform(X)`, `inverse_transform(X)`, `get_params()`, `set_params()`.

```python
normalizer = FeatureNormalizer(method='standard')
normalizer.fit(X_train)
X_norm = normalizer.transform(X_train)
X_test_norm = normalizer.transform(X_test)
```

---

### `RobustNormalizer`

> Normalization using median and IQR. More robust to outliers than z-score.

**File:** `feature_normalization.py`

**Constructor:**

```python
RobustNormalizer()
```

**Methods:** `fit(X, y=None)`, `transform(X)`, `inverse_transform(X)`.

Formula: `(X − median) / IQR`

```python
normalizer = RobustNormalizer()
normalizer.fit(X_train)
X_robust = normalizer.transform(X_train)
```

---

## Standalone Functions

### Feature Extraction

#### `extract_time_domain_features(signal) → Dict[str, float]`

**File:** `time_domain.py` — Extracts 7 time-domain features. See [FEATURE_CATALOG.md](FEATURE_CATALOG.md) #1–7.

#### `extract_frequency_domain_features(signal, fs) → Dict[str, float]`

**File:** `frequency_domain.py` — Extracts 12 frequency-domain features. See catalog #8–19.

#### `extract_envelope_features(signal, fs) → Dict[str, float]`

**File:** `envelope_analysis.py` — Extracts 4 envelope features. See catalog #20–23.

#### `extract_wavelet_features(signal, fs=20480) → Dict[str, float]`

**File:** `wavelet_features.py` — Extracts 7 wavelet/cepstral features. See catalog #24–30.

#### `extract_bispectrum_features(signal) → Dict[str, float]`

**File:** `bispectrum.py` — Extracts 6 higher-order spectral features. See catalog #31–36.

#### `extract_advanced_features(signal, fs=20480) → Dict[str, float]`

**File:** `advanced_features.py` — Extracts 16 advanced features (~10× slower). See catalog #37–52.

---

### Feature Validation

**File:** `feature_validator.py`

#### `validate_feature_vector(features, expected_dim=36) → Tuple[bool, str]`

Validates a single feature vector for correct dimensionality, NaN, and Inf values.

```python
is_valid, msg = validate_feature_vector(features)
if not is_valid:
    print(f"Validation failed: {msg}")
```

#### `validate_feature_matrix(features, labels=None, feature_names=None, verbose=True) → bool`

Comprehensive validation: NaN/Inf checks, distribution warnings, class balance report.

#### `check_feature_distribution(features, feature_names=None, variance_threshold=1e-6) → List[str]`

Checks for low-variance and extreme-outlier features.

#### `check_for_nans(features, feature_names=None) → Dict[str, int]`

Returns a dictionary mapping feature names to their NaN counts.

#### `replace_nans(features, strategy='mean') → np.ndarray`

Replace NaN values using `'mean'`, `'median'`, or `'zero'` strategy.

---

### Feature Importance

**File:** `feature_importance.py`

#### `get_random_forest_importances(rf_model, feature_names=None) → Dict[str, float]`

Extract Gini importance from a trained `RandomForestClassifier`.

#### `get_permutation_importances(model, X_val, y_val, feature_names=None, n_repeats=10, random_state=42) → Dict[str, Tuple[float, float]]`

Compute permutation importances. Returns `{name: (mean, std)}`.

#### `plot_feature_importances(importances, top_n=20, figsize=(10,8), title="Feature Importances") → plt.Figure`

Horizontal bar chart of feature importances.

#### `plot_permutation_importances(importances, top_n=20, figsize=(10,8)) → plt.Figure`

Permutation importances with error bars.

#### `compare_importances(rf_importances, perm_importances, top_n=15) → plt.Figure`

Side-by-side comparison of RF Gini and permutation importances.

---

### Helper Functions (Frequency Domain)

**File:** `frequency_domain.py`

| Function | Description |
|---|---|
| `compute_fft(signal, fs)` | Returns `(freqs, psd)` tuple via FFT |
| `compute_dominant_frequency(psd, freqs)` | Peak frequency in spectrum |
| `compute_spectral_centroid(psd, freqs)` | Center of mass of spectrum |
| `compute_spectral_entropy(psd)` | Shannon entropy of PSD |
| `compute_band_energy(psd, freqs, band_range)` | Energy in a frequency band |
| `compute_harmonic_ratios(psd, freqs, f0, tolerance=5.0)` | 2×/1× and 3×/1× harmonic ratios |

### Helper Functions (Wavelet / Cepstral)

**File:** `wavelet_features.py`

| Function | Description |
|---|---|
| `compute_dwt_energy(signal, wavelet='db4', level=5)` | Energy per DWT level |
| `compute_wavelet_energy_ratio(energies)` | Detail-to-total energy ratio |
| `compute_wavelet_kurtosis(signal, wavelet='db4', level=5)` | Mean kurtosis of detail coefficients |
| `compute_cepstral_peak_ratio(signal, fs)` | Peak/mean in cepstrum |
| `compute_quefrency_centroid(signal, fs)` | Centroid of cepstrum |

### Helper Functions (Envelope)

**File:** `envelope_analysis.py`

| Function | Description |
|---|---|
| `compute_envelope(signal)` | Hilbert envelope (analytic signal magnitude) |
| `compute_envelope_rms(envelope)` | RMS of envelope |
| `compute_envelope_kurtosis(envelope)` | Kurtosis of envelope |
| `compute_envelope_peak(envelope)` | Peak envelope value |
| `compute_modulation_frequency(envelope, fs)` | Dominant modulation frequency |

### Helper Functions (Bispectrum)

**File:** `bispectrum.py`

| Function | Description |
|---|---|
| `compute_bispectrum(signal, nfft=512)` | Bispectrum magnitude via direct method |
| `compute_bispectrum_peak(bispectrum)` | Peak bispectrum value |
| `compute_bispectrum_mean(bispectrum)` | Mean bispectrum value |
| `compute_bispectrum_entropy(bispectrum)` | Entropy of bispectrum |
| `compute_phase_coupling(signal)` | Bicoherence-based coupling (0–1) |
| `compute_nonlinearity_index(signal)` | Gaussianity deviation index |

### Helper Functions (Advanced)

**File:** `advanced_features.py`

| Function | Description |
|---|---|
| `extract_cwt_features(signal, fs=20480)` | 4 CWT features |
| `extract_wpt_features(signal, level=4)` | 4 WPT features |
| `extract_nonlinear_features(signal)` | 4 nonlinear dynamics features |
| `compute_sample_entropy(signal, m=2, r=0.2)` | Sample entropy |
| `compute_approximate_entropy(signal, m=2, r=0.2)` | Approximate entropy |
| `compute_dfa(signal, min_window=4, max_window=None)` | DFA scaling exponent |
