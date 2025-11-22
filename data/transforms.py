"""
Signal preprocessing transformations.

Purpose:
    PyTorch-style transforms for signal preprocessing:
    - Normalization (z-score, min-max)
    - Resampling
    - Filtering (bandpass, lowpass, highpass)
    - Composable transform pipeline

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import numpy as np
import torch
from typing import Optional, List, Callable, Union
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d
from utils.constants import SAMPLING_RATE

from utils.logging import get_logger

logger = get_logger(__name__)


class Compose:
    """
    Compose multiple transforms together.

    Example:
        >>> transform = Compose([
        ...     Normalize(method='zscore'),
        ...     ToTensor()
        ... ])
        >>> signal_transformed = transform(signal)
    """

    def __init__(self, transforms: List[Callable]):
        """
        Initialize composed transforms.

        Args:
            transforms: List of transform callables
        """
        self.transforms = transforms

    def __call__(self, signal: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        """Apply transforms sequentially."""
        for transform in self.transforms:
            signal = transform(signal)
        return signal


class Normalize:
    """
    Normalize signal amplitude.

    Methods:
    - 'zscore': Zero mean, unit variance
    - 'minmax': Scale to [0, 1]
    - 'robust': Robust scaling using IQR

    Example:
        >>> normalize = Normalize(method='zscore')
        >>> signal_norm = normalize(signal)
    """

    def __init__(self, method: str = 'zscore', eps: float = 1e-8):
        """
        Initialize normalizer.

        Args:
            method: Normalization method ('zscore', 'minmax', 'robust')
            eps: Small constant to avoid division by zero
        """
        self.method = method
        self.eps = eps

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        if self.method == 'zscore':
            mean = np.mean(signal)
            std = np.std(signal)
            return (signal - mean) / (std + self.eps)

        elif self.method == 'minmax':
            min_val = np.min(signal)
            max_val = np.max(signal)
            return (signal - min_val) / (max_val - min_val + self.eps)

        elif self.method == 'robust':
            # Robust scaling using IQR
            q25 = np.percentile(signal, 25)
            q75 = np.percentile(signal, 75)
            iqr = q75 - q25
            median = np.median(signal)
            return (signal - median) / (iqr + self.eps)

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")


class Resample:
    """
    Resample signal to different sampling rate.

    Example:
        >>> resample = Resample(original_fs=SAMPLING_RATE, target_fs=10240)
        >>> signal_resampled = resample(signal)
    """

    def __init__(
        self,
        original_fs: float,
        target_fs: float,
        method: str = 'scipy'
    ):
        """
        Initialize resampler.

        Args:
            original_fs: Original sampling frequency
            target_fs: Target sampling frequency
            method: Resampling method ('scipy', 'linear')
        """
        self.original_fs = original_fs
        self.target_fs = target_fs
        self.method = method

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply resampling."""
        if self.original_fs == self.target_fs:
            return signal

        if self.method == 'scipy':
            # Use scipy's resample (FFT-based)
            num_samples = int(len(signal) * self.target_fs / self.original_fs)
            return scipy_signal.resample(signal, num_samples)

        elif self.method == 'linear':
            # Linear interpolation
            original_time = np.arange(len(signal)) / self.original_fs
            target_time = np.arange(0, original_time[-1], 1/self.target_fs)
            interpolator = interp1d(original_time, signal, kind='linear')
            return interpolator(target_time)

        else:
            raise ValueError(f"Unknown resampling method: {self.method}")


class BandpassFilter:
    """
    Apply bandpass filter to signal.

    Example:
        >>> bp_filter = BandpassFilter(lowcut=10, highcut=500, fs=SAMPLING_RATE)
        >>> signal_filtered = bp_filter(signal)
    """

    def __init__(
        self,
        lowcut: float,
        highcut: float,
        fs: float,
        order: int = 4
    ):
        """
        Initialize bandpass filter.

        Args:
            lowcut: Low cutoff frequency (Hz)
            highcut: High cutoff frequency (Hz)
            fs: Sampling frequency (Hz)
            order: Filter order
        """
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

        # Design Butterworth filter
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        self.b, self.a = scipy_signal.butter(
            order, [low, high], btype='band'
        )

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply bandpass filter."""
        return scipy_signal.filtfilt(self.b, self.a, signal)


class LowpassFilter:
    """
    Apply lowpass filter to signal.

    Example:
        >>> lp_filter = LowpassFilter(cutoff=500, fs=SAMPLING_RATE)
        >>> signal_filtered = lp_filter(signal)
    """

    def __init__(self, cutoff: float, fs: float, order: int = 4):
        """
        Initialize lowpass filter.

        Args:
            cutoff: Cutoff frequency (Hz)
            fs: Sampling frequency (Hz)
            order: Filter order
        """
        self.cutoff = cutoff
        self.fs = fs
        self.order = order

        # Design Butterworth filter
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        self.b, self.a = scipy_signal.butter(
            order, normal_cutoff, btype='low'
        )

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply lowpass filter."""
        return scipy_signal.filtfilt(self.b, self.a, signal)


class HighpassFilter:
    """
    Apply highpass filter to signal.

    Example:
        >>> hp_filter = HighpassFilter(cutoff=10, fs=SAMPLING_RATE)
        >>> signal_filtered = hp_filter(signal)
    """

    def __init__(self, cutoff: float, fs: float, order: int = 4):
        """
        Initialize highpass filter.

        Args:
            cutoff: Cutoff frequency (Hz)
            fs: Sampling frequency (Hz)
            order: Filter order
        """
        self.cutoff = cutoff
        self.fs = fs
        self.order = order

        # Design Butterworth filter
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        self.b, self.a = scipy_signal.butter(
            order, normal_cutoff, btype='high'
        )

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply highpass filter."""
        return scipy_signal.filtfilt(self.b, self.a, signal)


class ToTensor:
    """
    Convert numpy array to PyTorch tensor.

    Example:
        >>> to_tensor = ToTensor()
        >>> signal_tensor = to_tensor(signal)
    """

    def __init__(self, dtype: torch.dtype = torch.float32):
        """
        Initialize converter.

        Args:
            dtype: Target tensor dtype
        """
        self.dtype = dtype

    def __call__(self, signal: np.ndarray) -> torch.Tensor:
        """Convert to tensor."""
        return torch.tensor(signal, dtype=self.dtype)


class Unsqueeze:
    """
    Add channel dimension to signal.

    Converts (N,) to (1, N) for CNN input.

    Example:
        >>> unsqueeze = Unsqueeze()
        >>> signal_2d = unsqueeze(signal)  # (N,) -> (1, N)
    """

    def __init__(self, dim: int = 0):
        """
        Initialize unsqueeze.

        Args:
            dim: Dimension to add
        """
        self.dim = dim

    def __call__(self, signal: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Add dimension."""
        if isinstance(signal, torch.Tensor):
            return signal.unsqueeze(self.dim)
        else:
            return np.expand_dims(signal, axis=self.dim)


class Detrend:
    """
    Remove linear trend from signal.

    Example:
        >>> detrend = Detrend()
        >>> signal_detrended = detrend(signal)
    """

    def __init__(self, type: str = 'linear'):
        """
        Initialize detrend.

        Args:
            type: Detrend type ('linear', 'constant')
        """
        self.type = type

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply detrending."""
        return scipy_signal.detrend(signal, type=self.type)


class Clip:
    """
    Clip signal values to range.

    Example:
        >>> clip = Clip(min_val=-3, max_val=3)
        >>> signal_clipped = clip(signal)
    """

    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        """
        Initialize clipper.

        Args:
            min_val: Minimum value (None = no clipping)
            max_val: Maximum value (None = no clipping)
        """
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply clipping."""
        return np.clip(signal, self.min_val, self.max_val)


class AddNoise:
    """
    Add Gaussian noise to signal.

    Example:
        >>> add_noise = AddNoise(snr_db=20)
        >>> signal_noisy = add_noise(signal)
    """

    def __init__(self, snr_db: float = 20.0):
        """
        Initialize noise adder.

        Args:
            snr_db: Signal-to-noise ratio in dB
        """
        self.snr_db = snr_db

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Add noise."""
        # Calculate signal power
        signal_power = np.mean(signal ** 2)

        # Calculate noise power
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear

        # Generate and add noise
        noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
        return signal + noise


class WindowSlice:
    """
    Extract window from signal.

    Example:
        >>> window = WindowSlice(start=0, length=10240)
        >>> signal_window = window(signal)
    """

    def __init__(
        self,
        start: Optional[int] = None,
        length: Optional[int] = None,
        random: bool = False
    ):
        """
        Initialize window slicer.

        Args:
            start: Start index (None = beginning)
            length: Window length (None = to end)
            random: Random window position
        """
        self.start = start
        self.length = length
        self.random = random

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Extract window."""
        N = len(signal)

        if self.random and self.length is not None:
            # Random window position
            max_start = N - self.length
            if max_start > 0:
                start = np.random.randint(0, max_start + 1)
            else:
                start = 0
        else:
            start = self.start if self.start is not None else 0

        if self.length is not None:
            end = min(start + self.length, N)
        else:
            end = N

        return signal[start:end]


def get_default_transform(
    normalize: bool = True,
    to_tensor: bool = True,
    add_channel_dim: bool = True
) -> Compose:
    """
    Get default transform pipeline.

    Args:
        normalize: Apply z-score normalization
        to_tensor: Convert to PyTorch tensor
        add_channel_dim: Add channel dimension for CNN

    Returns:
        Composed transform

    Example:
        >>> transform = get_default_transform()
        >>> signal_transformed = transform(signal)
    """
    transforms = []

    if normalize:
        transforms.append(Normalize(method='zscore'))

    if add_channel_dim:
        transforms.append(Unsqueeze(dim=0))

    if to_tensor:
        transforms.append(ToTensor())

    return Compose(transforms)
