"""
Wavelet Transform Module for Phase 5: Time-Frequency Analysis

Implements Continuous Wavelet Transform (CWT) and Discrete Wavelet Transform (DWT)
for bearing fault diagnosis. Wavelets provide better time-frequency resolution than
STFT for transient signals.

Author: AI Assistant
Date: 2025-11-20
"""

import numpy as np
import pywt
from typing import Tuple, Optional, List, Dict
import warnings


class WaveletTransform:
    """
    Continuous Wavelet Transform (CWT) generator for time-frequency analysis.

    CWT provides adaptive time-frequency resolution:
    - High frequency resolution at low frequencies
    - High time resolution at high frequencies

    Ideal for bearing faults with transient impacts and frequency modulation.
    """

    def __init__(
        self,
        wavelet: str = 'morl',
        scales: int = 128,
        fs: int = 20480
    ):
        """
        Initialize CWT generator.

        Args:
            wavelet: Mother wavelet type
                - 'morl': Morlet wavelet (good for bearing faults)
                - 'cmor': Complex Morlet
                - 'mexh': Mexican hat
                - 'gaus': Gaussian derivatives
            scales: Number of wavelet scales (frequency bins)
            fs: Sampling frequency (Hz)
        """
        self.wavelet = wavelet
        self.n_scales = scales
        self.fs = fs

        # Validate wavelet type
        available_wavelets = pywt.wavelist(kind='continuous')
        if wavelet not in available_wavelets:
            raise ValueError(f"Wavelet '{wavelet}' not supported. "
                           f"Available: {available_wavelets}")

    def generate_cwt_scalogram(
        self,
        signal_data: np.ndarray,
        scale_spacing: str = 'log'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate CWT scalogram (time-frequency energy representation).

        Args:
            signal_data: Input signal [N_samples]
            scale_spacing: Scale distribution ('log' or 'linear')

        Returns:
            Tuple containing:
                - scalogram: CWT coefficients [n_scales, n_time]
                - frequencies: Corresponding frequencies [n_scales]
        """
        # Generate scales
        if scale_spacing == 'log':
            # Logarithmic spacing (more resolution at low frequencies)
            scales = np.logspace(
                np.log10(1),
                np.log10(min(len(signal_data) // 4, 512)),
                self.n_scales
            )
        elif scale_spacing == 'linear':
            scales = np.linspace(1, min(len(signal_data) // 4, 512), self.n_scales)
        else:
            raise ValueError(f"Unknown scale_spacing: {scale_spacing}")

        # Compute CWT
        coefficients, frequencies = pywt.cwt(
            signal_data,
            scales,
            wavelet=self.wavelet,
            sampling_period=1.0 / self.fs
        )

        # Compute magnitude (scalogram)
        scalogram = np.abs(coefficients)

        return scalogram, frequencies

    def generate_normalized_scalogram(
        self,
        signal_data: np.ndarray,
        normalization: str = 'log_standardize'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate normalized CWT scalogram.

        Args:
            signal_data: Input signal
            normalization: Normalization method
                - 'log_standardize': Log-scale + z-score
                - 'log_minmax': Log-scale + min-max to [0, 1]
                - 'standardize': Z-score only
                - 'minmax': Min-max to [0, 1]

        Returns:
            Normalized scalogram, frequencies
        """
        scalogram, frequencies = self.generate_cwt_scalogram(signal_data)

        # Apply log transformation if requested
        if 'log' in normalization:
            scalogram = np.log10(scalogram + 1e-10)

        # Apply normalization
        if 'standardize' in normalization:
            mean = np.mean(scalogram)
            std = np.std(scalogram)
            if std > 1e-10:
                scalogram = (scalogram - mean) / std
        elif 'minmax' in normalization:
            min_val = np.min(scalogram)
            max_val = np.max(scalogram)
            if max_val - min_val > 1e-10:
                scalogram = (scalogram - min_val) / (max_val - min_val)

        return scalogram, frequencies

    def generate_power_scalogram(
        self,
        signal_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate power scalogram (energy distribution).

        Args:
            signal_data: Input signal

        Returns:
            Power scalogram, frequencies
        """
        coefficients, frequencies = self.generate_cwt_scalogram(signal_data)
        power_scalogram = coefficients ** 2
        return power_scalogram, frequencies

    def batch_generate(
        self,
        signals: np.ndarray,
        normalization: str = 'log_standardize',
        verbose: bool = True
    ) -> np.ndarray:
        """
        Generate scalograms for a batch of signals.

        Args:
            signals: Array of signals [N, signal_length]
            normalization: Normalization method
            verbose: Print progress

        Returns:
            Scalograms [N, n_scales, signal_length]
        """
        n_signals, signal_length = signals.shape
        scalograms = np.zeros((n_signals, self.n_scales, signal_length), dtype=np.float32)

        for i in range(n_signals):
            scalogram, _ = self.generate_normalized_scalogram(
                signals[i],
                normalization=normalization
            )
            scalograms[i] = scalogram.astype(np.float32)

            if verbose and (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{n_signals} scalograms")

        return scalograms


class DiscreteWaveletTransform:
    """
    Discrete Wavelet Transform (DWT) for multi-resolution analysis.

    DWT decomposes signal into approximation (low-freq) and detail (high-freq) coefficients.
    Useful for denoising and feature extraction.
    """

    def __init__(
        self,
        wavelet: str = 'db4',
        level: int = 5
    ):
        """
        Initialize DWT.

        Args:
            wavelet: Mother wavelet ('db4', 'sym4', 'coif3', etc.)
            level: Decomposition level
        """
        self.wavelet = wavelet
        self.level = level

        # Validate wavelet
        available = pywt.wavelist(kind='discrete')
        if wavelet not in available:
            raise ValueError(f"Wavelet '{wavelet}' not supported. "
                           f"Available: {available}")

    def decompose(
        self,
        signal_data: np.ndarray
    ) -> List[np.ndarray]:
        """
        Perform DWT decomposition.

        Args:
            signal_data: Input signal

        Returns:
            List of coefficients [cA_n, cD_n, cD_n-1, ..., cD_1]
            where cA is approximation, cD are details
        """
        coeffs = pywt.wavedec(
            signal_data,
            wavelet=self.wavelet,
            level=self.level
        )
        return coeffs

    def reconstruct(
        self,
        coeffs: List[np.ndarray]
    ) -> np.ndarray:
        """
        Reconstruct signal from DWT coefficients.

        Args:
            coeffs: DWT coefficients

        Returns:
            Reconstructed signal
        """
        return pywt.waverec(coeffs, wavelet=self.wavelet)

    def denoise(
        self,
        signal_data: np.ndarray,
        threshold_mode: str = 'soft',
        threshold_scale: float = 1.0
    ) -> np.ndarray:
        """
        Wavelet-based denoising using thresholding.

        Args:
            signal_data: Noisy input signal
            threshold_mode: 'soft' or 'hard' thresholding
            threshold_scale: Threshold multiplier

        Returns:
            Denoised signal
        """
        # Decompose
        coeffs = self.decompose(signal_data)

        # Estimate noise level from finest detail coefficients
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745

        # Universal threshold
        threshold = threshold_scale * sigma * np.sqrt(2 * np.log(len(signal_data)))

        # Threshold detail coefficients
        coeffs_thresh = [coeffs[0]]  # Keep approximation
        for detail_coeffs in coeffs[1:]:
            if threshold_mode == 'soft':
                thresholded = pywt.threshold(detail_coeffs, threshold, mode='soft')
            elif threshold_mode == 'hard':
                thresholded = pywt.threshold(detail_coeffs, threshold, mode='hard')
            else:
                raise ValueError(f"Unknown threshold_mode: {threshold_mode}")
            coeffs_thresh.append(thresholded)

        # Reconstruct
        denoised = self.reconstruct(coeffs_thresh)

        # Match original length
        if len(denoised) > len(signal_data):
            denoised = denoised[:len(signal_data)]
        elif len(denoised) < len(signal_data):
            denoised = np.pad(denoised, (0, len(signal_data) - len(denoised)))

        return denoised

    def extract_dwt_features(
        self,
        signal_data: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract statistical features from DWT coefficients.

        Args:
            signal_data: Input signal

        Returns:
            Dictionary of wavelet features
        """
        coeffs = self.decompose(signal_data)

        features = {}

        # Energy in each decomposition level
        for i, coeff in enumerate(coeffs):
            level_name = f'cA{self.level}' if i == 0 else f'cD{self.level - i + 1}'
            features[f'energy_{level_name}'] = np.sum(coeff ** 2)
            features[f'entropy_{level_name}'] = self._shannon_entropy(coeff)

        # Total energy
        total_energy = sum(features[k] for k in features if 'energy' in k)

        # Relative energies
        for key in list(features.keys()):
            if 'energy' in key:
                rel_key = key.replace('energy', 'rel_energy')
                features[rel_key] = features[key] / (total_energy + 1e-10)

        return features

    @staticmethod
    def _shannon_entropy(signal: np.ndarray) -> float:
        """Calculate Shannon entropy of signal."""
        # Histogram-based entropy
        hist, _ = np.histogram(signal, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist))
        return entropy


class WaveletPacketTransform:
    """
    Wavelet Packet Transform (WPT) for complete time-frequency decomposition.

    Unlike DWT which only decomposes approximations, WPT decomposes both
    approximations and details at each level.
    """

    def __init__(
        self,
        wavelet: str = 'db4',
        level: int = 4
    ):
        """
        Initialize WPT.

        Args:
            wavelet: Mother wavelet
            level: Decomposition level
        """
        self.wavelet = wavelet
        self.level = level

    def decompose(
        self,
        signal_data: np.ndarray
    ) -> pywt.WaveletPacket:
        """
        Perform WPT decomposition.

        Args:
            signal_data: Input signal

        Returns:
            WaveletPacket object
        """
        wp = pywt.WaveletPacket(
            data=signal_data,
            wavelet=self.wavelet,
            maxlevel=self.level
        )
        return wp

    def extract_wpt_features(
        self,
        signal_data: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract features from WPT decomposition.

        Args:
            signal_data: Input signal

        Returns:
            Dictionary of WPT features
        """
        wp = self.decompose(signal_data)

        features = {}

        # Get all nodes at max level
        node_names = [node.path for node in wp.get_level(self.level, 'freq')]

        # Energy in each node
        energies = []
        for node_name in node_names:
            node = wp[node_name]
            energy = np.sum(node.data ** 2)
            energies.append(energy)
            features[f'wpt_energy_{node_name}'] = energy

        total_energy = sum(energies)

        # Relative energies
        for i, node_name in enumerate(node_names):
            rel_energy = energies[i] / (total_energy + 1e-10)
            features[f'wpt_rel_energy_{node_name}'] = rel_energy

        # Best basis energy concentration
        features['wpt_energy_concentration'] = max(energies) / (total_energy + 1e-10)

        return features


# Convenience functions
def generate_scalogram(
    signal_data: np.ndarray,
    wavelet: str = 'morl',
    scales: int = 128,
    fs: int = 20480,
    normalization: str = 'log_standardize'
) -> np.ndarray:
    """
    Convenience function to generate a single scalogram.

    Args:
        signal_data: Input signal
        wavelet: Wavelet type
        scales: Number of scales
        fs: Sampling frequency
        normalization: Normalization method

    Returns:
        Normalized scalogram [n_scales, n_time]
    """
    cwt = WaveletTransform(wavelet=wavelet, scales=scales, fs=fs)
    scalogram, _ = cwt.generate_normalized_scalogram(
        signal_data,
        normalization=normalization
    )
    return scalogram


def denoise_signal(
    signal_data: np.ndarray,
    wavelet: str = 'db4',
    level: int = 5
) -> np.ndarray:
    """
    Convenience function for wavelet denoising.

    Args:
        signal_data: Noisy signal
        wavelet: Wavelet type
        level: Decomposition level

    Returns:
        Denoised signal
    """
    dwt = DiscreteWaveletTransform(wavelet=wavelet, level=level)
    return dwt.denoise(signal_data)


if __name__ == "__main__":
    # Example usage
    print("Wavelet Transform - Example Usage\n")

    # Generate synthetic signal
    fs = 20480
    duration = 5.0
    t = np.linspace(0, duration, int(fs * duration))

    # Bearing fault with impact and noise
    signal_data = np.zeros_like(t)

    # Periodic impacts (100 Hz fault frequency)
    impact_freq = 100
    impact_times = np.arange(0, duration, 1 / impact_freq)
    for t_impact in impact_times:
        idx = int(t_impact * fs)
        if idx < len(signal_data):
            # Damped sinusoid (impact response)
            duration_impact = 0.005  # 5ms
            t_local = t[idx:min(idx + int(duration_impact * fs), len(t))] - t_impact
            impact = np.exp(-1000 * t_local) * np.sin(2 * np.pi * 5000 * t_local)
            signal_data[idx:idx + len(impact)] += impact

    # Add noise
    noisy_signal = signal_data + 0.2 * np.random.randn(len(signal_data))

    # 1. CWT Scalogram
    print("1. Continuous Wavelet Transform (CWT)")
    cwt = WaveletTransform(wavelet='morl', scales=128, fs=fs)
    scalogram, freqs = cwt.generate_normalized_scalogram(noisy_signal)
    print(f"   Scalogram shape: {scalogram.shape}")
    print(f"   Frequency range: {freqs.min():.1f} - {freqs.max():.1f} Hz")

    # 2. DWT Denoising
    print("\n2. Discrete Wavelet Transform (DWT) Denoising")
    dwt = DiscreteWaveletTransform(wavelet='db4', level=5)
    denoised = dwt.denoise(noisy_signal)
    print(f"   Original SNR: {10 * np.log10(np.var(signal_data) / np.var(noisy_signal - signal_data)):.2f} dB")
    print(f"   Denoised SNR: {10 * np.log10(np.var(signal_data) / np.var(denoised - signal_data)):.2f} dB")

    # 3. DWT Features
    print("\n3. DWT Feature Extraction")
    features = dwt.extract_dwt_features(noisy_signal)
    print(f"   Extracted {len(features)} wavelet features")
    print(f"   Sample features: {list(features.keys())[:5]}")

    # 4. WPT Features
    print("\n4. Wavelet Packet Transform (WPT)")
    wpt = WaveletPacketTransform(wavelet='db4', level=4)
    wpt_features = wpt.extract_wpt_features(noisy_signal)
    print(f"   Extracted {len(wpt_features)} WPT features")
    print(f"   Energy concentration: {wpt_features['wpt_energy_concentration']:.4f}")

    print("\n✓ Wavelet transform operations successful!")
