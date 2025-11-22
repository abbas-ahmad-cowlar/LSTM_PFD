"""
Advanced signal augmentation techniques for time-series data.

Purpose:
    Sophisticated augmentation methods to improve model robustness:
    - Mixup: Interpolate between samples
    - Time warping: Non-linear time stretching
    - Magnitude warping: Non-linear amplitude changes
    - Window slicing: Random crops with overlap
    - Jittering: Add random noise
    - Scaling: Random amplitude scaling
    - Time shifting: Circular shift in time

    These augmentations help prevent overfitting and improve generalization
    to unseen fault conditions and sensor variations.

Author: LSTM_PFD Team
Date: 2025-11-20
"""

from utils.constants import SIGNAL_LENGTH

import torch
import numpy as np
from typing import Tuple, Optional
import random

from utils.logging import get_logger

logger = get_logger(__name__)


class SignalAugmentation:
    """
    Base class for signal augmentation techniques.
    """

    def __call__(self, signal: torch.Tensor, label: Optional[int] = None):
        """Apply augmentation to signal."""
        raise NotImplementedError


class Mixup(SignalAugmentation):
    """
    Mixup augmentation for signals.

    Creates synthetic training samples by linear interpolation:
    mixed_signal = lambda * signal1 + (1 - lambda) * signal2
    mixed_label = lambda * label1 + (1 - lambda) * label2

    Reference: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2017)

    Args:
        alpha: Beta distribution parameter (higher = more mixing)
        prob: Probability of applying mixup

    Example:
        >>> mixup = Mixup(alpha=0.2, prob=0.5)
        >>> signal1 = torch.randn(SIGNAL_LENGTH)
        >>> signal2 = torch.randn(SIGNAL_LENGTH)
        >>> mixed, lambda_val = mixup(signal1, signal2)
    """

    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(
        self,
        signal1: torch.Tensor,
        signal2: torch.Tensor,
        label1: Optional[torch.Tensor] = None,
        label2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], float]:
        """
        Apply mixup to two signals.

        Args:
            signal1: First signal
            signal2: Second signal
            label1: First label (optional)
            label2: Second label (optional)

        Returns:
            Mixed signal, mixed label (if labels provided), lambda value
        """
        if random.random() > self.prob:
            # No mixup
            return signal1, label1, 1.0

        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Mix signals
        mixed_signal = lam * signal1 + (1 - lam) * signal2

        # Mix labels if provided
        mixed_label = None
        if label1 is not None and label2 is not None:
            mixed_label = lam * label1 + (1 - lam) * label2

        return mixed_signal, mixed_label, lam


class TimeWarping(SignalAugmentation):
    """
    Time warping augmentation.

    Applies non-linear time stretching/compression to create variations
    while preserving the overall signal characteristics.

    Args:
        sigma: Standard deviation of warping magnitude
        num_knots: Number of control points for warping

    Example:
        >>> warp = TimeWarping(sigma=0.2, num_knots=4)
        >>> signal = torch.randn(SIGNAL_LENGTH)
        >>> warped = warp(signal)
    """

    def __init__(self, sigma: float = 0.2, num_knots: int = 4):
        self.sigma = sigma
        self.num_knots = num_knots

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply time warping to signal.

        Args:
            signal: Input signal [signal_length] or [batch, signal_length]

        Returns:
            Warped signal with same shape
        """
        orig_shape = signal.shape
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        batch_size, signal_length = signal.shape

        # Generate warping curve
        warp = self._generate_warp_curve(signal_length)

        # Apply warping
        warped = []
        for i in range(batch_size):
            # Interpolate signal at warped time points
            indices = torch.clamp(warp, 0, signal_length - 1).long()
            warped_signal = signal[i][indices]
            warped.append(warped_signal)

        result = torch.stack(warped)

        if len(orig_shape) == 1:
            result = result.squeeze(0)

        return result

    def _generate_warp_curve(self, length: int) -> torch.Tensor:
        """Generate smooth warping curve."""
        # Create control points
        knot_positions = np.linspace(0, length - 1, self.num_knots)
        knot_values = np.random.normal(0, self.sigma, self.num_knots)

        # Interpolate to full length
        warp = np.interp(np.arange(length), knot_positions, knot_values)

        # Add original time and normalize
        warp = np.arange(length) + warp * length
        warp = np.clip(warp, 0, length - 1)

        return torch.from_numpy(warp).float()


class MagnitudeWarping(SignalAugmentation):
    """
    Magnitude warping augmentation.

    Applies smooth, non-linear amplitude changes to the signal.

    Args:
        sigma: Standard deviation of magnitude changes
        num_knots: Number of control points for warping

    Example:
        >>> mag_warp = MagnitudeWarping(sigma=0.2, num_knots=4)
        >>> signal = torch.randn(SIGNAL_LENGTH)
        >>> warped = mag_warp(signal)
    """

    def __init__(self, sigma: float = 0.2, num_knots: int = 4):
        self.sigma = sigma
        self.num_knots = num_knots

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply magnitude warping to signal.

        Args:
            signal: Input signal

        Returns:
            Magnitude-warped signal
        """
        orig_shape = signal.shape
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)

        batch_size, signal_length = signal.shape

        # Generate magnitude curve
        mag_curve = self._generate_magnitude_curve(signal_length)

        # Apply to all batch elements
        warped = signal * mag_curve.unsqueeze(0)

        if len(orig_shape) == 1:
            warped = warped.squeeze(0)

        return warped

    def _generate_magnitude_curve(self, length: int) -> torch.Tensor:
        """Generate smooth magnitude curve."""
        # Create control points
        knot_positions = np.linspace(0, length - 1, self.num_knots)
        knot_values = np.random.normal(1.0, self.sigma, self.num_knots)

        # Interpolate to full length
        mag_curve = np.interp(np.arange(length), knot_positions, knot_values)
        mag_curve = np.maximum(mag_curve, 0.1)  # Prevent zeros

        return torch.from_numpy(mag_curve).float()


class Jittering(SignalAugmentation):
    """
    Add random Gaussian noise (jittering) to signal.

    Args:
        sigma: Standard deviation of noise relative to signal std

    Example:
        >>> jitter = Jittering(sigma=0.05)
        >>> signal = torch.randn(SIGNAL_LENGTH)
        >>> noisy = jitter(signal)
    """

    def __init__(self, sigma: float = 0.05):
        self.sigma = sigma

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Add jittering noise to signal.

        Args:
            signal: Input signal

        Returns:
            Noisy signal
        """
        noise_std = self.sigma * torch.std(signal)
        noise = torch.randn_like(signal) * noise_std
        return signal + noise


class Scaling(SignalAugmentation):
    """
    Random amplitude scaling.

    Args:
        sigma: Standard deviation of scaling factor

    Example:
        >>> scale = Scaling(sigma=0.1)
        >>> signal = torch.randn(SIGNAL_LENGTH)
        >>> scaled = scale(signal)
    """

    def __init__(self, sigma: float = 0.1):
        self.sigma = sigma

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply random scaling to signal.

        Args:
            signal: Input signal

        Returns:
            Scaled signal
        """
        scale_factor = np.random.normal(1.0, self.sigma)
        scale_factor = max(scale_factor, 0.5)  # Prevent extreme scaling
        return signal * scale_factor


class TimeShift(SignalAugmentation):
    """
    Circular time shift augmentation.

    Args:
        max_shift: Maximum shift amount as fraction of signal length

    Example:
        >>> shift = TimeShift(max_shift=0.1)
        >>> signal = torch.randn(SIGNAL_LENGTH)
        >>> shifted = shift(signal)
    """

    def __init__(self, max_shift: float = 0.1):
        self.max_shift = max_shift

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply random circular shift.

        Args:
            signal: Input signal

        Returns:
            Shifted signal
        """
        signal_length = signal.shape[-1]
        max_shift_samples = int(signal_length * self.max_shift)
        shift = random.randint(-max_shift_samples, max_shift_samples)

        return torch.roll(signal, shifts=shift, dims=-1)


class WindowSlicing(SignalAugmentation):
    """
    Extract random window from signal.

    Useful for signals longer than needed, introduces translation invariance.

    Args:
        window_size: Size of window to extract
        strategy: 'random' or 'center'

    Example:
        >>> slicer = WindowSlicing(window_size=SIGNAL_LENGTH)
        >>> long_signal = torch.randn(150000)
        >>> window = slicer(long_signal)
    """

    def __init__(self, window_size: int, strategy: str = 'random'):
        self.window_size = window_size
        self.strategy = strategy

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Extract window from signal.

        Args:
            signal: Input signal (must be >= window_size)

        Returns:
            Windowed signal of size window_size
        """
        signal_length = signal.shape[-1]

        if signal_length < self.window_size:
            # Pad if too short
            pad_size = self.window_size - signal_length
            signal = torch.nn.functional.pad(signal, (0, pad_size), mode='constant')
            return signal

        if signal_length == self.window_size:
            return signal

        # Extract window
        if self.strategy == 'center':
            start = (signal_length - self.window_size) // 2
        else:  # random
            start = random.randint(0, signal_length - self.window_size)

        return signal[..., start:start + self.window_size]


class ComposeAugmentations:
    """
    Compose multiple augmentations.

    Args:
        augmentations: List of augmentation instances
        prob: Probability of applying each augmentation

    Example:
        >>> augment = ComposeAugmentations([
        ...     Jittering(sigma=0.05),
        ...     Scaling(sigma=0.1),
        ...     TimeShift(max_shift=0.1)
        ... ], prob=0.5)
        >>> augmented = augment(signal)
    """

    def __init__(self, augmentations: list, prob: float = 0.5):
        self.augmentations = augmentations
        self.prob = prob

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations sequentially.

        Args:
            signal: Input signal

        Returns:
            Augmented signal
        """
        for aug in self.augmentations:
            if random.random() < self.prob:
                signal = aug(signal)

        return signal


def test_augmentations():
    """Test all augmentation techniques."""
    print("=" * 60)
    print("Testing Signal Augmentations")
    print("=" * 60)

    # Create test signal
    signal_length = SIGNAL_LENGTH
    signal = torch.randn(signal_length)

    print(f"\nOriginal signal shape: {signal.shape}")
    print(f"Original signal stats: mean={signal.mean():.4f}, std={signal.std():.4f}")

    print("\n1. Testing Mixup...")
    mixup = Mixup(alpha=0.2, prob=1.0)
    signal2 = torch.randn(signal_length)
    mixed, _, lam = mixup(signal, signal2)
    print(f"   Mixed signal shape: {mixed.shape}, lambda={lam:.3f}")

    print("\n2. Testing Time Warping...")
    time_warp = TimeWarping(sigma=0.2, num_knots=4)
    warped = time_warp(signal)
    print(f"   Warped signal shape: {warped.shape}")
    print(f"   Warped stats: mean={warped.mean():.4f}, std={warped.std():.4f}")

    print("\n3. Testing Magnitude Warping...")
    mag_warp = MagnitudeWarping(sigma=0.2, num_knots=4)
    mag_warped = mag_warp(signal)
    print(f"   Mag-warped signal shape: {mag_warped.shape}")
    print(f"   Mag-warped stats: mean={mag_warped.mean():.4f}, std={mag_warped.std():.4f}")

    print("\n4. Testing Jittering...")
    jitter = Jittering(sigma=0.05)
    jittered = jitter(signal)
    print(f"   Jittered signal shape: {jittered.shape}")
    print(f"   Jittered stats: mean={jittered.mean():.4f}, std={jittered.std():.4f}")

    print("\n5. Testing Scaling...")
    scale = Scaling(sigma=0.1)
    scaled = scale(signal)
    print(f"   Scaled signal shape: {scaled.shape}")
    print(f"   Scaled stats: mean={scaled.mean():.4f}, std={scaled.std():.4f}")

    print("\n6. Testing Time Shift...")
    shift = TimeShift(max_shift=0.1)
    shifted = shift(signal)
    print(f"   Shifted signal shape: {shifted.shape}")
    print(f"   Shifted stats: mean={shifted.mean():.4f}, std={shifted.std():.4f}")

    print("\n7. Testing Window Slicing...")
    long_signal = torch.randn(150000)
    slicer = WindowSlicing(window_size=SIGNAL_LENGTH, strategy='random')
    sliced = slicer(long_signal)
    print(f"   Sliced from {long_signal.shape} to {sliced.shape}")

    print("\n8. Testing Composed Augmentations...")
    compose = ComposeAugmentations([
        Jittering(sigma=0.05),
        Scaling(sigma=0.1),
        TimeShift(max_shift=0.05)
    ], prob=1.0)  # Apply all for testing

    composed = compose(signal)
    print(f"   Composed augmentation shape: {composed.shape}")
    print(f"   Composed stats: mean={composed.mean():.4f}, std={composed.std():.4f}")

    print("\n9. Testing batch processing...")
    batch_signal = torch.randn(8, signal_length)
    batch_warped = time_warp(batch_signal)
    print(f"   Batch warped shape: {batch_warped.shape}")

    print("\n" + "=" * 60)
    print("✅ All augmentation tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_augmentations()
