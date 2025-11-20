"""
Main feature extraction orchestrator for classical ML pipeline.

Purpose:
    Coordinates extraction of all 36 features from vibration signals.
    Combines time-domain, frequency-domain, envelope, wavelet, and
    bispectrum features.

Reference: Section 8.2 of technical report

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

from .time_domain import extract_time_domain_features
from .frequency_domain import extract_frequency_domain_features
from .envelope_analysis import extract_envelope_features
from .wavelet_features import extract_wavelet_features
from .bispectrum import extract_bispectrum_features


class FeatureExtractor:
    """
    Main orchestrator for extracting all 36 features from signals.

    Features:
    - 7 time-domain features
    - 12 frequency-domain features
    - 4 envelope features
    - 7 wavelet features
    - 6 bispectrum features
    Total: 36 features

    Example:
        >>> extractor = FeatureExtractor(fs=20480)
        >>> signal = np.random.randn(102400)
        >>> features = extractor.extract_features(signal)
        >>> print(f"Total features: {len(features)}")  # 36
    """

    def __init__(self, fs: float = 20480):
        """
        Initialize feature extractor.

        Args:
            fs: Sampling frequency (Hz)
        """
        self.fs = fs
        self.feature_names_ = self._get_feature_names()

    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """
        Extract all 36 features from a single signal.

        Args:
            signal: Input vibration signal (1D array)

        Returns:
            Feature vector (36,) as numpy array

        Example:
            >>> signal = np.random.randn(102400)
            >>> features = extractor.extract_features(signal)
            >>> assert features.shape == (36,)
        """
        # Extract features from each domain
        time_features = extract_time_domain_features(signal)
        freq_features = extract_frequency_domain_features(signal, self.fs)
        env_features = extract_envelope_features(signal, self.fs)
        wavelet_features = extract_wavelet_features(signal, self.fs)
        bispectrum_features = extract_bispectrum_features(signal)

        # Combine all features into a single dict
        all_features = {
            **time_features,
            **freq_features,
            **env_features,
            **wavelet_features,
            **bispectrum_features
        }

        # Convert to ordered array based on feature_names
        feature_vector = np.array([all_features[name] for name in self.feature_names_])

        return feature_vector

    def extract_time_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract time domain features only.

        Args:
            signal: Input vibration signal (1D array)

        Returns:
            Dictionary with time-domain features (RMS, Kurtosis, etc.)

        Example:
            >>> features = extractor.extract_time_domain_features(signal)
            >>> print(features['mean'], features['std'], features['rms'])
        """
        time_features = extract_time_domain_features(signal)
        # Add common aliases for test compatibility
        result = dict(time_features)
        if 'RMS' in result and 'mean' not in result:
            # Calculate mean for test expectations
            result['mean'] = np.mean(signal)
            result['std'] = np.std(signal)
            result['peak'] = np.max(np.abs(signal))
        # Map existing features to expected names
        if 'RMS' in result:
            result['rms'] = result['RMS']
        if 'Kurtosis' in result:
            result['kurtosis'] = result['Kurtosis']
        if 'Skewness' in result:
            result['skewness'] = result['Skewness']
        return result

    def extract_frequency_domain_features(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract frequency domain features only.

        Args:
            signal: Input vibration signal (1D array)

        Returns:
            Dictionary with frequency-domain features

        Example:
            >>> features = extractor.extract_frequency_domain_features(signal)
            >>> print(features['spectral_centroid'])
        """
        freq_features = extract_frequency_domain_features(signal, self.fs)
        # Map to expected names for tests
        result = dict(freq_features)
        if 'SpectralCentroid' in result:
            result['spectral_centroid'] = result['SpectralCentroid']
        if 'SpectralEntropy' in result:
            result['spectral_entropy'] = result['SpectralEntropy']
        # Calculate spectral spread if not present
        if 'spectral_spread' not in result:
            # Approximate spectral spread from spectral std
            if 'SpectralStd' in result:
                result['spectral_spread'] = result['SpectralStd']
            else:
                result['spectral_spread'] = 0.0
        return result

    def extract_features_dict(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Extract features and return as dictionary.

        Args:
            signal: Input vibration signal (1D array)

        Returns:
            Dictionary with all 36 features

        Example:
            >>> features_dict = extractor.extract_features_dict(signal)
            >>> print(features_dict['RMS'])
        """
        time_features = extract_time_domain_features(signal)
        freq_features = extract_frequency_domain_features(signal, self.fs)
        env_features = extract_envelope_features(signal, self.fs)
        wavelet_features = extract_wavelet_features(signal, self.fs)
        bispectrum_features = extract_bispectrum_features(signal)

        all_features = {
            **time_features,
            **freq_features,
            **env_features,
            **wavelet_features,
            **bispectrum_features
        }

        return all_features

    def extract_batch(self, signals: np.ndarray) -> np.ndarray:
        """
        Extract features from a batch of signals.

        Args:
            signals: Batch of signals (n_signals, signal_length)

        Returns:
            Feature matrix (n_signals, 36)

        Example:
            >>> signals = np.random.randn(100, 102400)
            >>> features = extractor.extract_batch(signals)
            >>> assert features.shape == (100, 36)
        """
        n_signals = signals.shape[0]
        features_matrix = np.zeros((n_signals, len(self.feature_names_)))

        for i in range(n_signals):
            features_matrix[i] = self.extract_features(signals[i])

        return features_matrix

    def get_feature_names(self) -> List[str]:
        """
        Get ordered list of feature names.

        Returns:
            List of 36 feature names
        """
        return self.feature_names_

    def _get_feature_names(self) -> List[str]:
        """
        Define canonical ordering of feature names.

        Returns:
            List of 36 feature names in extraction order
        """
        # Time-domain (7)
        time_names = [
            'RMS', 'Kurtosis', 'Skewness', 'CrestFactor',
            'ShapeFactor', 'ImpulseFactor', 'ClearanceFactor'
        ]

        # Frequency-domain (12)
        freq_names = [
            'DominantFreq', 'SpectralCentroid', 'SpectralEntropy',
            'LowBandEnergy', 'MidBandEnergy', 'HighBandEnergy',
            'VeryHighBandEnergy', 'TotalSpectralPower', 'SpectralStd',
            'Harmonic2X1X', 'Harmonic3X1X', 'SpectralPeakiness'
        ]

        # Envelope (4)
        envelope_names = [
            'EnvelopeRMS', 'EnvelopeKurtosis', 'EnvelopePeak', 'ModulationFreq'
        ]

        # Wavelet (7)
        wavelet_names = [
            'WaveletEnergyRatio', 'WaveletKurtosis', 'WaveletEnergy_D1',
            'WaveletEnergy_D3', 'WaveletEnergy_D5', 'CepstralPeakRatio',
            'QuefrencyCentroid'
        ]

        # Bispectrum (6)
        bispectrum_names = [
            'BispectrumPeak', 'BispectrumMean', 'BispectrumEntropy',
            'PhaseCoupling', 'NonlinearityIndex', 'BispectrumPeakRatio'
        ]

        all_names = (
            time_names + freq_names + envelope_names +
            wavelet_names + bispectrum_names
        )

        return all_names

    def save_features(self, features: np.ndarray, filepath: Path):
        """
        Save extracted features to file.

        Args:
            features: Feature matrix (n_samples, 36)
            filepath: Path to save file (.npz or .npy)
        """
        filepath = Path(filepath)
        if filepath.suffix == '.npz':
            np.savez(filepath, features=features, feature_names=self.feature_names_)
        else:
            np.save(filepath, features)

    def load_features(self, filepath: Path) -> np.ndarray:
        """
        Load features from file.

        Args:
            filepath: Path to feature file

        Returns:
            Feature matrix
        """
        filepath = Path(filepath)
        if filepath.suffix == '.npz':
            data = np.load(filepath)
            return data['features']
        else:
            return np.load(filepath)
