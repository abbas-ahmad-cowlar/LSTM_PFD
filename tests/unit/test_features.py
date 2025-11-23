"""
Unit Tests for Feature Extraction

Tests for features/feature_extractor.py

Author: Syed Abbas Ahmad
Date: 2025-11-20
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from features.feature_extractor import FeatureExtractor


@pytest.mark.unit
class TestFeatureExtractor:
    """Test suite for FeatureExtractor."""

    def test_initialization(self):
        """Test FeatureExtractor initialization."""
        extractor = FeatureExtractor(fs=20480)
        assert extractor.fs == 20480

    def test_extract_time_domain_features(self, sample_signal):
        """Test time domain feature extraction."""
        extractor = FeatureExtractor(fs=20480)
        features = extractor.extract_time_domain_features(sample_signal)

        assert isinstance(features, dict)
        assert 'mean' in features
        assert 'std' in features
        assert 'rms' in features
        assert 'peak' in features
        assert 'kurtosis' in features
        assert 'skewness' in features

        # Check feature values are reasonable
        assert np.isfinite(features['mean'])
        assert features['std'] >= 0
        assert features['rms'] >= 0

    def test_extract_frequency_domain_features(self, sample_signal):
        """Test frequency domain feature extraction."""
        extractor = FeatureExtractor(fs=20480)
        features = extractor.extract_frequency_domain_features(sample_signal)

        assert isinstance(features, dict)
        assert 'spectral_centroid' in features
        assert 'spectral_spread' in features
        assert 'spectral_entropy' in features

        # Check features are in valid range
        assert features['spectral_centroid'] >= 0
        assert features['spectral_spread'] >= 0
        assert 0 <= features['spectral_entropy'] <= 10

    def test_extract_features_batch(self, sample_batch_signals):
        """Test batch feature extraction."""
        signals, _ = sample_batch_signals
        extractor = FeatureExtractor(fs=20480)

        features_list = []
        for signal in signals:
            features = extractor.extract_features(signal)
            features_list.append(features)

        features_array = np.array(features_list)

        # Check shape
        assert features_array.shape[0] == len(signals)
        assert features_array.shape[1] > 0  # Has features

        # Check all features are finite
        assert np.all(np.isfinite(features_array))

    def test_extract_features_empty_signal(self):
        """Test extraction with empty signal."""
        extractor = FeatureExtractor(fs=20480)

        with pytest.raises((ValueError, IndexError)):
            extractor.extract_features(np.array([]))

    def test_extract_features_constant_signal(self):
        """Test extraction with constant signal."""
        extractor = FeatureExtractor(fs=20480)
        constant_signal = np.ones(1024)

        features = extractor.extract_features(constant_signal)

        # Mean should be 1.0, std should be 0
        assert np.isclose(features[0], 1.0, atol=0.1)  # Assuming mean is first feature
        # Some features will be 0 or undefined for constant signal


@pytest.mark.unit
class TestFeatureNormalization:
    """Test suite for feature normalization."""

    def test_zscore_normalization(self, sample_features):
        """Test Z-score normalization."""
        from features.feature_normalization import FeatureNormalizer

        X, _ = sample_features
        normalizer = FeatureNormalizer(method='zscore')

        X_norm = normalizer.fit_transform(X)

        # Check shape preserved
        assert X_norm.shape == X.shape

        # Check mean ~0, std ~1
        assert np.allclose(X_norm.mean(axis=0), 0, atol=0.1)
        assert np.allclose(X_norm.std(axis=0), 1, atol=0.1)

    def test_minmax_normalization(self, sample_features):
        """Test Min-Max normalization."""
        from features.feature_normalization import FeatureNormalizer

        X, _ = sample_features
        normalizer = FeatureNormalizer(method='minmax')

        X_norm = normalizer.fit_transform(X)

        # Check range [0, 1]
        assert np.all(X_norm >= 0)
        assert np.all(X_norm <= 1)
        assert np.allclose(X_norm.min(axis=0), 0, atol=0.1)
        assert np.allclose(X_norm.max(axis=0), 1, atol=0.1)

    def test_transform_without_fit(self, sample_features):
        """Test transform without fit raises error."""
        from features.feature_normalization import FeatureNormalizer

        X, _ = sample_features
        normalizer = FeatureNormalizer(method='zscore')

        with pytest.raises((RuntimeError, AttributeError, ValueError)):
            normalizer.transform(X)


@pytest.mark.unit
class TestFeatureSelection:
    """Test suite for feature selection."""

    def test_mrmr_selection(self, sample_features):
        """Test MRMR feature selection."""
        from features.feature_selector import FeatureSelector

        X, y = sample_features
        selector = FeatureSelector(method='mrmr', n_features=5)

        X_selected = selector.fit_transform(X, y)

        # Check shape
        assert X_selected.shape[0] == X.shape[0]
        assert X_selected.shape[1] == 5

    def test_variance_threshold_selection(self, sample_features):
        """Test variance threshold selection."""
        from features.feature_selector import FeatureSelector

        X, y = sample_features

        # Add constant feature
        X_with_constant = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

        selector = FeatureSelector(method='variance', threshold=0.1)
        X_selected = selector.fit_transform(X_with_constant, y)

        # Should remove constant feature
        assert X_selected.shape[1] < X_with_constant.shape[1]
