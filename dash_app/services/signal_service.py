"""
Signal processing service.
"""
import numpy as np
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats
from typing import Dict, Any, Tuple

from services.data_service import DataService
from utils.logger import setup_logger
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

logger = setup_logger(__name__)


class SignalService:
    """Service for signal processing operations."""

    @staticmethod
    def compute_fft(signal_data: np.ndarray, fs: int = SAMPLING_RATE) -> Tuple[np.ndarray, np.ndarray]:
        """Compute FFT of signal."""
        n = len(signal_data)
        freq = np.fft.rfftfreq(n, d=1/fs)
        fft = np.abs(np.fft.rfft(signal_data))
        return freq, fft

    @staticmethod
    def compute_spectrogram(
        signal_data: np.ndarray,
        fs: int = SAMPLING_RATE,
        nperseg: int = 256
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute spectrogram using STFT."""
        f, t, Sxx = scipy_signal.spectrogram(
            signal_data,
            fs=fs,
            nperseg=nperseg,
            noverlap=nperseg//2
        )
        return f, t, Sxx

    @staticmethod
    def extract_basic_features(signal_data: np.ndarray, fs: int = SAMPLING_RATE) -> Dict[str, float]:
        """Extract basic statistical features."""
        features = {
            "rms": float(np.sqrt(np.mean(signal_data**2))),
            "kurtosis": float(scipy_stats.kurtosis(signal_data)),
            "skewness": float(scipy_stats.skew(signal_data)),
            "peak_value": float(np.max(np.abs(signal_data))),
            "mean": float(np.mean(signal_data)),
            "std": float(np.std(signal_data))
        }

        # Frequency domain features
        freq, fft = SignalService.compute_fft(signal_data, fs)
        features["dominant_frequency"] = float(freq[np.argmax(fft)])
        features["spectral_centroid"] = float(np.sum(freq * fft) / np.sum(fft))

        return features

    @staticmethod
    def get_signal_with_features(dataset_id: int, signal_id: str) -> Dict[str, Any]:
        """Get signal data with computed features."""
        signal_data = DataService.load_signal_data(dataset_id, signal_id)
        features = SignalService.extract_basic_features(signal_data)

        return {
            "signal_id": signal_id,
            "signal_data": signal_data.tolist(),
            "features": features,
            "length": len(signal_data)
        }
