# Features — Feature Engineering Module

> Extract, select, normalize, and validate vibration signal features for bearing fault diagnosis.

## Overview

The Features module is the feature engineering backbone of the classical ML pipeline. It transforms raw vibration time-series signals into a structured feature vector that downstream models (e.g., Random Forest, SVM) consume for fault classification.

The module extracts **36 base features** across five signal analysis domains (time-domain, frequency-domain, envelope analysis, wavelet transforms, and bispectrum/higher-order statistics). An additional **16 advanced features** are available for computationally intensive experiments (CWT, WPT, nonlinear dynamics, spectrogram). After extraction, features can be selected (MRMR or variance threshold), normalized (z-score, min-max, or robust), and validated for data quality.

## Architecture

```mermaid
graph TD
    A["Raw Vibration Signal<br/>(1D numpy array)"] --> B["FeatureExtractor<br/>(Orchestrator)"]

    B --> C["time_domain.py<br/>7 features"]
    B --> D["frequency_domain.py<br/>12 features"]
    B --> E["envelope_analysis.py<br/>4 features"]
    B --> F["wavelet_features.py<br/>7 features"]
    B --> G["bispectrum.py<br/>6 features"]

    C --> H["Feature Vector<br/>(36,) numpy array"]
    D --> H
    E --> H
    F --> H
    G --> H

    H --> I["FeatureSelector<br/>(MRMR / Variance)"]
    I --> J["Selected Features<br/>(n_selected,)"]
    J --> K["FeatureNormalizer<br/>(Z-score / MinMax / Robust)"]
    K --> L["Normalized Features<br/>→ ML Model"]

    style B fill:#4a90d9,color:#fff
    style H fill:#2d6a4f,color:#fff
    style L fill:#e76f51,color:#fff
```

### Optional Advanced Features

```mermaid
graph LR
    A["Raw Signal"] --> B["advanced_features.py"]
    B --> C["CWT (4)"]
    B --> D["WPT (4)"]
    B --> E["Nonlinear (4)"]
    B --> F["Spectrogram (4)"]
    C --> G["16 Advanced Features"]
    D --> G
    E --> G
    F --> G

    style B fill:#e9c46a,color:#333
    style G fill:#e76f51,color:#fff
```

> ⚠️ Advanced features are **~10× slower** to compute. Use only when needed.

## Quick Start

```python
import numpy as np
from packages.core.features import FeatureExtractor, FeatureSelector, FeatureNormalizer

# 1. Extract features
extractor = FeatureExtractor(fs=20480)
signal = np.random.randn(102400)  # Example vibration signal

# Single signal → 36-element vector
features = extractor.extract_features(signal)
assert features.shape == (36,)

# Batch extraction
signals = np.random.randn(100, 102400)
feature_matrix = extractor.extract_batch(signals)
assert feature_matrix.shape == (100, 36)

# 2. Select best features (MRMR)
selector = FeatureSelector(method='mrmr', n_features=15)
selector.fit(X_train, y_train, feature_names=extractor.get_feature_names())
X_selected = selector.transform(X_train)

# 3. Normalize
normalizer = FeatureNormalizer(method='standard')
normalizer.fit(X_selected)
X_normalized = normalizer.transform(X_selected)
```

## Key Components

| Component                           | Description                                               | File                       |
| ----------------------------------- | --------------------------------------------------------- | -------------------------- |
| `FeatureExtractor`                  | Main orchestrator — extracts all 36 base features         | `feature_extractor.py`     |
| `FeatureSelector`                   | MRMR and variance-threshold feature selection             | `feature_selector.py`      |
| `VarianceThresholdSelector`         | Simple variance-based feature filter                      | `feature_selector.py`      |
| `FeatureNormalizer`                 | Z-score / min-max normalization (sklearn-compatible)      | `feature_normalization.py` |
| `RobustNormalizer`                  | Median/IQR robust normalization                           | `feature_normalization.py` |
| `extract_time_domain_features`      | 7 statistical features (RMS, kurtosis, etc.)              | `time_domain.py`           |
| `extract_frequency_domain_features` | 12 spectral features (FFT-based)                          | `frequency_domain.py`      |
| `extract_envelope_features`         | 4 Hilbert envelope features                               | `envelope_analysis.py`     |
| `extract_wavelet_features`          | 7 DWT + cepstral features                                 | `wavelet_features.py`      |
| `extract_bispectrum_features`       | 6 higher-order spectral features                          | `bispectrum.py`            |
| `extract_advanced_features`         | 16 optional advanced features (CWT, WPT, nonlinear, STFT) | `advanced_features.py`     |
| Feature importance analysis         | RF Gini + permutation importance with plots               | `feature_importance.py`    |
| Feature validation utilities        | NaN/Inf checks, distribution validation, NaN replacement  | `feature_validator.py`     |

## API Summary

See [API.md](API.md) for full API reference. See [FEATURE_CATALOG.md](FEATURE_CATALOG.md) for the definitive list of all 52 features with formulas and source locations.

## Dependencies

- **Requires:**
  - `numpy`, `scipy` — signal processing and statistics
  - `pywt` — wavelet transforms (DWT, CWT, WPT)
  - `scikit-learn` — `mutual_info_classif`, `StandardScaler`, `MinMaxScaler`, `permutation_importance`
  - `matplotlib`, `seaborn` — importance visualization
  - `utils.constants` — `SAMPLING_RATE` (20480 Hz), `SIGNAL_LENGTH`
- **Provides:**
  - `FeatureExtractor` — main extraction class
  - `FeatureSelector` — feature selection
  - `FeatureNormalizer` — normalization

## Configuration

| Parameter                 | Default      | Description                                                |
| ------------------------- | ------------ | ---------------------------------------------------------- |
| `fs` (sampling frequency) | `20480` Hz   | Sampling rate of vibration signals                         |
| `n_features` (selector)   | `15`         | Number of features to select via MRMR                      |
| `method` (normalizer)     | `'standard'` | Normalization method: `'standard'`, `'zscore'`, `'minmax'` |
| `wavelet`                 | `'db4'`      | Wavelet type for DWT/WPT decomposition                     |
| `level` (DWT)             | `5`          | Wavelet decomposition depth                                |

## Performance

> ⚠️ **Results pending.** Performance metrics below will be populated
> after experiments are run on the current codebase.

| Metric                                  | Value       |
| --------------------------------------- | ----------- |
| Feature extraction time (single signal) | `[PENDING]` |
| Feature extraction time (batch of 100)  | `[PENDING]` |
| Advanced feature extraction time        | `[PENDING]` |
| MRMR selection time                     | `[PENDING]` |

## Testing

```bash
# Unit tests
pytest tests/unit/test_features.py -v

# Integration tests
pytest tests/test_feature_engineering.py -v
```

## Related Documentation

- [IDB 1.4 Features Analysis](../../docs/idb_reports/IDB_1_4_FEATURES_ANALYSIS.md)
- [IDB 1.4 Features Best Practices](../../docs/idb_reports/IDB_1_4_FEATURES_BEST_PRACTICES.md)
- [Models README](../models/README.md) — downstream consumers of feature vectors
- [Training README](../training/README.md) — training pipeline using features
