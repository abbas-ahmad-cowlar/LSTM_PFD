"""
Signal preprocessing transforms for CNN training.

Purpose:
    Transform raw vibration signals into CNN-ready format:
    - ToTensor1D: Convert NumPy → PyTorch tensor
    - Normalize1D: Z-score normalization per sample
    - RandomCrop1D: Extract random subsequence (data augmentation)
    - RandomAmplitudeScale: Multiply by random factor
    - AddGaussianNoise: Inject noise for robustness

Author: LSTM_PFD Team
Date: 2025-11-20
"""

import numpy as np
import torch
from typing import Optional, Callable, Union
from utils.constants import SIGNAL_LENGTH


class ToTensor1D:
    """
    Convert NumPy array to PyTorch tensor.

    Ensures correct data type (float32) and shape [1, T] for CNN input.

    Example:
        >>> transform = ToTensor1D()
        >>> signal = np.random.randn(SIGNAL_LENGTH)  # NumPy array
        >>> tensor = transform(signal)
        >>> print(tensor.shape)  # torch.Size([1, SIGNAL_LENGTH])
    """

    def __call__(self, signal: np.ndarray) -> torch.Tensor:
        """
        Convert NumPy array to tensor.

        Args:
            signal: NumPy array of shape (T,) or (1, T)

        Returns:
            PyTorch tensor of shape (1, T)
        """
        # Ensure float32
        if signal.dtype != np.float32:
            signal = signal.astype(np.float32)

        # Convert to tensor
        tensor = torch.from_numpy(signal)

        # Add channel dimension if needed: (T,) → (1, T)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        return tensor


class Normalize1D:
    """
    Z-score normalization: (x - mean) / std.

    Normalizes each signal independently to zero mean and unit variance.
    Prevents issues from varying signal amplitudes across samples.

    Args:
        eps: Small constant to prevent division by zero

    Example:
        >>> transform = Normalize1D()
        >>> signal = np.random.randn(SIGNAL_LENGTH) * 10 + 5
        >>> normalized = transform(signal)
        >>> print(f"Mean: {normalized.mean():.6f}, Std: {normalized.std():.6f}")
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, signal: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Normalize signal to zero mean and unit variance.

        Args:
            signal: NumPy array or tensor

        Returns:
            Normalized signal (same type as input)
        """
        if isinstance(signal, torch.Tensor):
            mean = signal.mean()
            std = signal.std()
            return (signal - mean) / (std + self.eps)
        else:
            mean = signal.mean()
            std = signal.std()
            return (signal - mean) / (std + self.eps)


class RandomCrop1D:
    """
    Extract random subsequence from signal (data augmentation).

    Useful for:
    - Training on shorter sequences (faster training)
    - Augmentation: multiple crops from same signal
    - Testing invariance to temporal position

    Args:
        crop_size: Length of cropped subsequence
        padding_mode: How to handle crops larger than signal ('reflect', 'constant')

    Example:
        >>> transform = RandomCrop1D(crop_size=50000)
        >>> signal = np.random.randn(SIGNAL_LENGTH)
        >>> cropped = transform(signal)
        >>> print(cropped.shape)  # (50000,)
    """

    def __init__(self, crop_size: int, padding_mode: str = 'reflect'):
        self.crop_size = crop_size
        self.padding_mode = padding_mode

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """
        Randomly crop signal.

        Args:
            signal: Input signal of shape (T,) or (1, T)

        Returns:
            Cropped signal of shape (crop_size,)
        """
        # Flatten to 1D if needed
        if signal.ndim == 2 and signal.shape[0] == 1:
            signal = signal.squeeze(0)

        signal_length = len(signal)

        # If crop size equals signal length, return as-is
        if self.crop_size == signal_length:
            return signal

        # If crop size larger than signal, pad
        if self.crop_size > signal_length:
            pad_width = self.crop_size - signal_length
            if self.padding_mode == 'reflect':
                signal = np.pad(signal, (0, pad_width), mode='reflect')
            else:
                signal = np.pad(signal, (0, pad_width), mode='constant', constant_values=0)
            return signal

        # Random crop
        start_idx = np.random.randint(0, signal_length - self.crop_size + 1)
        return signal[start_idx:start_idx + self.crop_size]


class RandomAmplitudeScale:
    """
    Multiply signal by random scaling factor (data augmentation).

    Simulates variations in sensor gain and signal strength.

    Args:
        scale_range: Tuple (min_scale, max_scale) for uniform sampling
        p: Probability of applying this transform

    Example:
        >>> transform = RandomAmplitudeScale(scale_range=(0.8, 1.2), p=0.5)
        >>> signal = np.ones(1000)
        >>> scaled = transform(signal)
        >>> print(f"Scale factor: {scaled.mean():.2f}")  # Between 0.8 and 1.2
    """

    def __init__(self, scale_range: tuple = (0.8, 1.2), p: float = 0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, signal: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply random amplitude scaling.

        Args:
            signal: Input signal

        Returns:
            Scaled signal (or original if not applied)
        """
        if np.random.rand() > self.p:
            return signal

        # Sample scale factor
        scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])

        return signal * scale_factor


class AddGaussianNoise:
    """
    Add Gaussian noise to signal (data augmentation).

    Improves robustness to sensor noise and measurement uncertainty.

    Args:
        noise_level: Standard deviation of noise as fraction of signal std
        p: Probability of applying this transform

    Example:
        >>> transform = AddGaussianNoise(noise_level=0.05, p=0.5)
        >>> signal = np.random.randn(1000)
        >>> noisy = transform(signal)
    """

    def __init__(self, noise_level: float = 0.05, p: float = 0.5):
        self.noise_level = noise_level
        self.p = p

    def __call__(self, signal: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        Add Gaussian noise to signal.

        Args:
            signal: Input signal

        Returns:
            Noisy signal (or original if not applied)
        """
        if np.random.rand() > self.p:
            return signal

        # Compute signal standard deviation
        if isinstance(signal, torch.Tensor):
            signal_std = signal.std().item()
            noise = torch.randn_like(signal) * (self.noise_level * signal_std)
        else:
            signal_std = signal.std()
            noise = np.random.randn(*signal.shape) * (self.noise_level * signal_std)

        return signal + noise


class Compose:
    """
    Compose multiple transforms into a single pipeline.

    Args:
        transforms: List of transform objects

    Example:
        >>> transform = Compose([
        ...     Normalize1D(),
        ...     RandomAmplitudeScale(p=0.5),
        ...     AddGaussianNoise(p=0.3),
        ...     ToTensor1D()
        ... ])
        >>> signal = np.random.randn(SIGNAL_LENGTH)
        >>> tensor = transform(signal)
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, signal):
        """Apply all transforms sequentially."""
        for transform in self.transforms:
            signal = transform(signal)
        return signal

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string


def get_train_transforms(augment: bool = True) -> Compose:
    """
    Get standard training transforms.

    Args:
        augment: Whether to include data augmentation

    Returns:
        Composed transform pipeline

    Example:
        >>> train_transform = get_train_transforms(augment=True)
        >>> signal = np.random.randn(SIGNAL_LENGTH)
        >>> tensor = train_transform(signal)
    """
    if augment:
        return Compose([
            Normalize1D(),
            RandomAmplitudeScale(scale_range=(0.9, 1.1), p=0.5),
            AddGaussianNoise(noise_level=0.03, p=0.3),
            ToTensor1D()
        ])
    else:
        return Compose([
            Normalize1D(),
            ToTensor1D()
        ])


def get_test_transforms() -> Compose:
    """
    Get standard test transforms (no augmentation).

    Returns:
        Composed transform pipeline

    Example:
        >>> test_transform = get_test_transforms()
        >>> signal = np.random.randn(SIGNAL_LENGTH)
        >>> tensor = test_transform(signal)
    """
    return Compose([
        Normalize1D(),
        ToTensor1D()
    ])


def test_transforms():
    """Test all transform classes."""
    print("=" * 60)
    print("Testing CNN Transforms")
    print("=" * 60)

    # Create test signal
    signal = np.random.randn(SIGNAL_LENGTH).astype(np.float32)

    # Test ToTensor1D
    print("\n1. Testing ToTensor1D...")
    to_tensor = ToTensor1D()
    tensor = to_tensor(signal)
    print(f"   Input shape: {signal.shape}, dtype: {signal.dtype}")
    print(f"   Output shape: {tensor.shape}, dtype: {tensor.dtype}")
    assert tensor.shape == (1, SIGNAL_LENGTH)
    assert tensor.dtype == torch.float32

    # Test Normalize1D
    print("\n2. Testing Normalize1D...")
    normalize = Normalize1D()
    normalized = normalize(signal)
    print(f"   Original - Mean: {signal.mean():.4f}, Std: {signal.std():.4f}")
    print(f"   Normalized - Mean: {normalized.mean():.4f}, Std: {normalized.std():.4f}")
    assert abs(normalized.mean()) < 1e-6
    assert abs(normalized.std() - 1.0) < 1e-6

    # Test RandomCrop1D
    print("\n3. Testing RandomCrop1D...")
    crop = RandomCrop1D(crop_size=50000)
    cropped = crop(signal)
    print(f"   Cropped shape: {cropped.shape}")
    assert cropped.shape == (50000,)

    # Test RandomAmplitudeScale
    print("\n4. Testing RandomAmplitudeScale...")
    scale = RandomAmplitudeScale(scale_range=(0.8, 1.2), p=1.0)
    scaled = scale(signal)
    scale_factor = np.abs(scaled).mean() / np.abs(signal).mean()
    print(f"   Scale factor: {scale_factor:.4f}")
    assert 0.7 < scale_factor < 1.3

    # Test AddGaussianNoise
    print("\n5. Testing AddGaussianNoise...")
    add_noise = AddGaussianNoise(noise_level=0.1, p=1.0)
    noisy = add_noise(signal)
    noise_power = np.mean((noisy - signal) ** 2)
    print(f"   Noise power: {noise_power:.6f}")
    assert noise_power > 0

    # Test Compose
    print("\n6. Testing Compose...")
    train_transform = get_train_transforms(augment=True)
    output = train_transform(signal)
    print(f"   Pipeline output shape: {output.shape}, dtype: {output.dtype}")
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, SIGNAL_LENGTH)

    # Test test transforms
    print("\n7. Testing test transforms...")
    test_transform = get_test_transforms()
    test_output = test_transform(signal)
    print(f"   Test output shape: {test_output.shape}")
    assert test_output.shape == (1, SIGNAL_LENGTH)

    print("\n" + "=" * 60)
    print("✅ All transform tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_transforms()
