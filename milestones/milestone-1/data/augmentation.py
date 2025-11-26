"""
Advanced data augmentation for vibration signals.

Purpose:
    Sophisticated augmentation techniques beyond basic operations:
    - Time warping
    - Magnitude warping
    - Window slicing
    - Mixup between signals
    - Adding synthetic noise profiles

Author: Author Name
Date: 2025-11-19
"""

import numpy as np
from typing import Tuple, Optional, List
from scipy.interpolate import CubicSpline
from scipy.signal import resample

from utils.logging import get_logger

logger = get_logger(__name__)


class SignalAugmenter:
    """
    Advanced augmentation techniques for vibration signals.

    Provides more sophisticated augmentations than basic transformations:
    - Time warping (non-uniform time stretching)
    - Magnitude warping (amplitude modulation along signal)
    - Window slicing (extract and pad random windows)
    - Mixup (linear combination of two signals)
    - Cutout (zero random segments)

    Example:
        >>> augmenter = SignalAugmenter()
        >>> augmented = augmenter.time_warp(signal, sigma=0.2)
        >>> mixed = augmenter.mixup(signal1, signal2, alpha=0.5)
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        """
        Initialize augmenter.

        Args:
            rng: Random number generator (for reproducibility)
        """
        self.rng = rng if rng is not None else np.random.default_rng()

    def time_warp(
        self,
        signal: np.ndarray,
        sigma: float = 0.2,
        num_knots: int = 4
    ) -> np.ndarray:
        """
        Apply time warping (non-uniform time stretching).

        Simulates variable speed conditions or sample rate variations.

        Args:
            signal: Input signal (N,)
            sigma: Warping strength (0.0 = no warp, 0.5 = strong warp)
            num_knots: Number of warping control points

        Returns:
            Time-warped signal (same length)

        Example:
            >>> warped = augmenter.time_warp(signal, sigma=0.2)
        """
        N = len(signal)

        # Create warping curve via cubic spline through random knots
        orig_steps = np.linspace(0, N - 1, num_knots)
        random_warps = self.rng.normal(loc=1.0, scale=sigma, size=num_knots)
        random_warps = np.cumsum(random_warps)
        random_warps = (random_warps - random_warps[0]) / (random_warps[-1] - random_warps[0]) * (N - 1)

        # Cubic spline interpolation
        warper = CubicSpline(orig_steps, random_warps)
        warped_indices = warper(np.arange(N))

        # Clip to valid range
        warped_indices = np.clip(warped_indices, 0, N - 1)

        # Interpolate signal at warped indices
        warped_signal = np.interp(warped_indices, np.arange(N), signal)

        return warped_signal

    def magnitude_warp(
        self,
        signal: np.ndarray,
        sigma: float = 0.2,
        num_knots: int = 4
    ) -> np.ndarray:
        """
        Apply magnitude warping (smooth amplitude modulation).

        Simulates varying sensor sensitivity or distance changes.

        Args:
            signal: Input signal (N,)
            sigma: Warping strength
            num_knots: Number of control points

        Returns:
            Magnitude-warped signal

        Example:
            >>> warped = augmenter.magnitude_warp(signal, sigma=0.3)
        """
        N = len(signal)

        # Create smooth magnitude curve
        orig_steps = np.linspace(0, N - 1, num_knots)
        random_mags = self.rng.normal(loc=1.0, scale=sigma, size=num_knots)

        # Cubic spline interpolation
        warper = CubicSpline(orig_steps, random_mags)
        magnitude_curve = warper(np.arange(N))

        # Apply magnitude modulation
        warped_signal = signal * magnitude_curve

        return warped_signal

    def window_slice(
        self,
        signal: np.ndarray,
        window_ratio: float = 0.9
    ) -> np.ndarray:
        """
        Extract random window and pad to original length.

        Simulates partial observation or measurement window shifts.

        Args:
            signal: Input signal (N,)
            window_ratio: Fraction of signal to keep (0.5 = 50%)

        Returns:
            Windowed signal (same length, padded with zeros)

        Example:
            >>> sliced = augmenter.window_slice(signal, window_ratio=0.8)
        """
        N = len(signal)
        window_length = int(N * window_ratio)

        # Random start position
        start = self.rng.integers(0, N - window_length + 1)
        end = start + window_length

        # Extract window
        windowed = np.zeros_like(signal)
        windowed[start:end] = signal[start:end]

        return windowed

    def mixup(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray,
        alpha: Optional[float] = None,
        beta_alpha: float = 0.2
    ) -> Tuple[np.ndarray, float]:
        """
        Mixup: linear combination of two signals.

        Creates synthetic training examples by blending signals.
        Commonly used for regularization in deep learning.

        Args:
            signal1: First signal
            signal2: Second signal (must be same length)
            alpha: Mixing coefficient (0.0 to 1.0). If None, sample from Beta distribution
            beta_alpha: Parameter for Beta distribution (if alpha=None)

        Returns:
            (mixed_signal, alpha) tuple

        Example:
            >>> mixed, alpha = augmenter.mixup(signal1, signal2)
            >>> # mixed = alpha * signal1 + (1 - alpha) * signal2
        """
        if len(signal1) != len(signal2):
            raise ValueError(f"Signal length mismatch: {len(signal1)} vs {len(signal2)}")

        # Sample mixing coefficient from Beta distribution if not provided
        if alpha is None:
            alpha = self.rng.beta(beta_alpha, beta_alpha)

        # Linear combination
        mixed = alpha * signal1 + (1 - alpha) * signal2

        return mixed, alpha

    def cutout(
        self,
        signal: np.ndarray,
        num_cutouts: int = 1,
        cutout_ratio: float = 0.1
    ) -> np.ndarray:
        """
        Zero out random segments of signal.

        Simulates sensor dropouts or data corruption.

        Args:
            signal: Input signal (N,)
            num_cutouts: Number of segments to zero
            cutout_ratio: Fraction of signal per cutout

        Returns:
            Signal with random segments zeroed

        Example:
            >>> corrupted = augmenter.cutout(signal, num_cutouts=2, cutout_ratio=0.05)
        """
        N = len(signal)
        cutout_length = int(N * cutout_ratio)

        output = signal.copy()

        for _ in range(num_cutouts):
            # Random position
            start = self.rng.integers(0, N - cutout_length + 1)
            end = start + cutout_length

            # Zero out
            output[start:end] = 0.0

        return output

    def jittering(
        self,
        signal: np.ndarray,
        sigma: float = 0.05
    ) -> np.ndarray:
        """
        Add random Gaussian jittering to signal.

        Args:
            signal: Input signal
            sigma: Standard deviation of jitter (relative to signal std)

        Returns:
            Jittered signal

        Example:
            >>> jittered = augmenter.jittering(signal, sigma=0.03)
        """
        signal_std = np.std(signal)
        jitter = self.rng.normal(0, sigma * signal_std, size=len(signal))
        return signal + jitter

    def scaling(
        self,
        signal: np.ndarray,
        sigma: float = 0.1
    ) -> np.ndarray:
        """
        Randomly scale signal amplitude.

        Args:
            signal: Input signal
            sigma: Standard deviation of scaling factor

        Returns:
            Scaled signal

        Example:
            >>> scaled = augmenter.scaling(signal, sigma=0.1)
        """
        scale_factor = self.rng.normal(1.0, sigma)
        return signal * scale_factor

    def rotation(
        self,
        signal: np.ndarray
    ) -> np.ndarray:
        """
        Randomly rotate (circular shift) signal.

        Args:
            signal: Input signal

        Returns:
            Rotated signal

        Example:
            >>> rotated = augmenter.rotation(signal)
        """
        shift = self.rng.integers(0, len(signal))
        return np.roll(signal, shift)

    def permutation(
        self,
        signal: np.ndarray,
        num_segments: int = 4
    ) -> np.ndarray:
        """
        Randomly permute segments of signal.

        Args:
            signal: Input signal
            num_segments: Number of segments to divide and permute

        Returns:
            Permuted signal

        Example:
            >>> permuted = augmenter.permutation(signal, num_segments=5)
        """
        N = len(signal)
        segment_length = N // num_segments

        # Split into segments
        segments = []
        for i in range(num_segments):
            start = i * segment_length
            end = start + segment_length if i < num_segments - 1 else N
            segments.append(signal[start:end])

        # Randomly permute
        self.rng.shuffle(segments)

        # Concatenate
        permuted = np.concatenate(segments)

        return permuted

    def augment_batch(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        augmentation_methods: List[str],
        augment_per_signal: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random augmentations to batch of signals.

        Args:
            signals: Array of signals (batch_size, signal_length)
            labels: Array of labels (batch_size,)
            augmentation_methods: List of method names to randomly apply
            augment_per_signal: Number of augmented versions per signal

        Returns:
            (augmented_signals, augmented_labels) tuple

        Example:
            >>> methods = ['time_warp', 'magnitude_warp', 'jittering']
            >>> aug_signals, aug_labels = augmenter.augment_batch(
            ...     signals, labels, methods, augment_per_signal=2
            ... )
        """
        method_map = {
            'time_warp': lambda s: self.time_warp(s, sigma=0.2),
            'magnitude_warp': lambda s: self.magnitude_warp(s, sigma=0.2),
            'window_slice': lambda s: self.window_slice(s, window_ratio=0.9),
            'jittering': lambda s: self.jittering(s, sigma=0.05),
            'scaling': lambda s: self.scaling(s, sigma=0.1),
            'rotation': lambda s: self.rotation(s),
            'permutation': lambda s: self.permutation(s, num_segments=4),
            'cutout': lambda s: self.cutout(s, num_cutouts=1, cutout_ratio=0.1)
        }

        augmented_signals = []
        augmented_labels = []

        for signal, label in zip(signals, labels):
            # Original
            augmented_signals.append(signal)
            augmented_labels.append(label)

            # Augmented versions
            for _ in range(augment_per_signal):
                # Randomly select augmentation method
                method_name = self.rng.choice(augmentation_methods)
                if method_name in method_map:
                    aug_signal = method_map[method_name](signal)
                    augmented_signals.append(aug_signal)
                    augmented_labels.append(label)

        return np.array(augmented_signals), np.array(augmented_labels)


def random_augment(
    signal: np.ndarray,
    methods: List[str] = ['time_warp', 'magnitude_warp', 'jittering'],
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Convenience function to apply random augmentation.

    Args:
        signal: Input signal
        methods: List of augmentation methods to choose from
        rng: Random number generator

    Returns:
        Augmented signal

    Example:
        >>> augmented = random_augment(signal, methods=['time_warp', 'jittering'])
    """
    augmenter = SignalAugmenter(rng=rng)
    method_name = (rng or np.random.default_rng()).choice(methods)

    method_map = {
        'time_warp': lambda: augmenter.time_warp(signal, sigma=0.2),
        'magnitude_warp': lambda: augmenter.magnitude_warp(signal, sigma=0.2),
        'window_slice': lambda: augmenter.window_slice(signal, window_ratio=0.9),
        'jittering': lambda: augmenter.jittering(signal, sigma=0.05),
        'scaling': lambda: augmenter.scaling(signal, sigma=0.1),
        'rotation': lambda: augmenter.rotation(signal),
        'permutation': lambda: augmenter.permutation(signal, num_segments=4),
        'cutout': lambda: augmenter.cutout(signal, num_cutouts=1, cutout_ratio=0.1)
    }

    if method_name in method_map:
        return method_map[method_name]()
    else:
        logger.warning(f"Unknown augmentation method: {method_name}, returning original signal")
        return signal
