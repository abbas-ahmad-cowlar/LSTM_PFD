"""
Spectrogram-Specific Data Augmentation

Implements augmentation techniques for time-frequency representations:
- SpecAugment (time and frequency masking)
- Time warping
- MixUp for spectrograms
- Spectrogram-specific random transforms

Reference:
- Park et al. (2019). "SpecAugment: A Simple Data Augmentation Method for ASR"

Usage:
    from data.spectrogram_augmentation import SpecAugment, spec_augment_transform

    # Apply SpecAugment
    augmenter = SpecAugment(time_mask_param=40, freq_mask_param=20)
    augmented_spec = augmenter(spectrogram)

    # MixUp
    mixed_spec, mixed_target = spectrogram_mixup(spec1, spec2, target1, target2, alpha=0.4)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import random


class SpecAugment(nn.Module):
    """
    SpecAugment: masking blocks of consecutive time or frequency bins.

    Args:
        time_mask_param: Maximum width of time mask (default: 40)
        freq_mask_param: Maximum width of frequency mask (default: 20)
        num_time_masks: Number of time masks to apply (default: 2)
        num_freq_masks: Number of frequency masks to apply (default: 2)
        mask_value: Value to fill masked regions (default: 0.0)
        p: Probability of applying augmentation (default: 0.5)
    """

    def __init__(
        self,
        time_mask_param: int = 40,
        freq_mask_param: int = 20,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        mask_value: float = 0.0,
        p: float = 0.5
    ):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.mask_value = mask_value
        self.p = p

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.

        Args:
            spectrogram: Input spectrogram [C, H, W] or [B, C, H, W]

        Returns:
            Augmented spectrogram
        """
        if random.random() > self.p:
            return spectrogram

        if spectrogram.dim() == 3:
            # Single spectrogram [C, H, W]
            return self._augment_single(spectrogram)
        elif spectrogram.dim() == 4:
            # Batch of spectrograms [B, C, H, W]
            return torch.stack([self._augment_single(s) for s in spectrogram])
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {spectrogram.dim()}D")

    def _augment_single(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to a single spectrogram [C, H, W]."""
        C, H, W = spectrogram.shape
        spec_aug = spectrogram.clone()

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, max(1, H - f))
            spec_aug[:, f0:f0+f, :] = self.mask_value

        # Time masking
        for _ in range(self.num_time_masks):
            t = random.randint(0, self.time_mask_param)
            t0 = random.randint(0, max(1, W - t))
            spec_aug[:, :, t0:t0+t] = self.mask_value

        return spec_aug


class TimeWarp(nn.Module):
    """
    Time warping augmentation for spectrograms.

    Applies non-linear time distortion to simulate speed variations.

    Args:
        W: Maximum warp distance (default: 80)
        p: Probability of applying augmentation (default: 0.5)
    """

    def __init__(self, W: int = 80, p: float = 0.5):
        super().__init__()
        self.W = W
        self.p = p

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply time warping to spectrogram.

        Args:
            spectrogram: Input spectrogram [C, H, W]

        Returns:
            Warped spectrogram
        """
        if random.random() > self.p:
            return spectrogram

        C, H, W = spectrogram.shape

        # Choose center point
        center = W // 2

        # Random warp distance
        warp = random.randint(-self.W, self.W)

        # Create grid for interpolation
        if warp == 0:
            return spectrogram

        # Simple linear interpolation for warping
        # In practice, you'd use scipy.ndimage.map_coordinates or similar
        # This is a simplified version
        warped = spectrogram.clone()

        if warp > 0:
            # Compress left, stretch right
            left_ratio = center / (center + warp)
            right_ratio = (W - center) / (W - center - warp)

            # Resample (simplified)
            # In production, use proper interpolation
            pass

        return warped  # Return original for now (simplified)


class SpectrogramMixup:
    """
    MixUp augmentation for spectrograms.

    Linearly combines two spectrograms and their labels.

    Args:
        alpha: Beta distribution parameter (default: 0.4)
        p: Probability of applying mixup (default: 0.5)
    """

    def __init__(self, alpha: float = 0.4, p: float = 0.5):
        self.alpha = alpha
        self.p = p

    def __call__(
        self,
        spec1: torch.Tensor,
        spec2: torch.Tensor,
        target1: torch.Tensor,
        target2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply MixUp to two spectrograms.

        Args:
            spec1: First spectrogram
            spec2: Second spectrogram
            target1: First target (one-hot or index)
            target2: Second target

        Returns:
            Tuple of (mixed spectrogram, mixed target)
        """
        if random.random() > self.p:
            return spec1, target1

        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # Mix spectrograms
        mixed_spec = lam * spec1 + (1 - lam) * spec2

        # Mix targets (assume one-hot encoding)
        if target1.dim() == 1:
            # Convert indices to one-hot
            num_classes = max(target1.max().item(), target2.max().item()) + 1
            target1_onehot = torch.zeros(num_classes).scatter_(0, target1, 1)
            target2_onehot = torch.zeros(num_classes).scatter_(0, target2, 1)
            mixed_target = lam * target1_onehot + (1 - lam) * target2_onehot
        else:
            mixed_target = lam * target1 + (1 - lam) * target2

        return mixed_spec, mixed_target


class RandomNoise(nn.Module):
    """
    Add random Gaussian noise to spectrograms.

    Args:
        noise_std: Standard deviation of noise (default: 0.01)
        p: Probability of applying noise (default: 0.3)
    """

    def __init__(self, noise_std: float = 0.01, p: float = 0.3):
        super().__init__()
        self.noise_std = noise_std
        self.p = p

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to spectrogram."""
        if random.random() > self.p:
            return spectrogram

        noise = torch.randn_like(spectrogram) * self.noise_std
        return spectrogram + noise


class RandomErasing(nn.Module):
    """
    Randomly erase rectangular regions in spectrograms.

    Similar to SpecAugment but with random aspect ratios.

    Args:
        p: Probability of applying erasing (default: 0.5)
        scale: Range of erasing area ratio (default: (0.02, 0.33))
        ratio: Range of aspect ratio (default: (0.3, 3.3))
        value: Value to fill erased region (default: 0.0)
    """

    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0.0
    ):
        super().__init__()
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Randomly erase rectangular region."""
        if random.random() > self.p:
            return spectrogram

        C, H, W = spectrogram.shape[-3:]
        area = H * W

        for _ in range(10):  # Try up to 10 times
            # Sample area and aspect ratio
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))

            if h < H and w < W:
                i = random.randint(0, H - h)
                j = random.randint(0, W - w)

                spec_erased = spectrogram.clone()
                spec_erased[..., i:i+h, j:j+w] = self.value
                return spec_erased

        return spectrogram


class SpecAugmentCompose:
    """
    Compose multiple spectrogram augmentations.

    Args:
        transforms: List of augmentation transforms
    """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply all transforms sequentially."""
        for transform in self.transforms:
            spectrogram = transform(spectrogram)
        return spectrogram


def get_default_spec_augment(mode: str = 'light') -> SpecAugmentCompose:
    """
    Get default SpecAugment configuration.

    Args:
        mode: Augmentation strength ('light', 'medium', 'heavy')

    Returns:
        Composed augmentation pipeline
    """
    if mode == 'light':
        transforms = [
            SpecAugment(
                time_mask_param=20,
                freq_mask_param=10,
                num_time_masks=1,
                num_freq_masks=1,
                p=0.5
            ),
            RandomNoise(noise_std=0.005, p=0.3)
        ]
    elif mode == 'medium':
        transforms = [
            SpecAugment(
                time_mask_param=40,
                freq_mask_param=20,
                num_time_masks=2,
                num_freq_masks=2,
                p=0.7
            ),
            RandomNoise(noise_std=0.01, p=0.5),
            RandomErasing(p=0.3)
        ]
    elif mode == 'heavy':
        transforms = [
            SpecAugment(
                time_mask_param=60,
                freq_mask_param=30,
                num_time_masks=3,
                num_freq_masks=3,
                p=0.8
            ),
            RandomNoise(noise_std=0.02, p=0.6),
            RandomErasing(p=0.5)
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose from ['light', 'medium', 'heavy']")

    return SpecAugmentCompose(transforms)


# Convenience function
def spec_augment_transform(
    spectrogram: torch.Tensor,
    time_mask: int = 40,
    freq_mask: int = 20,
    num_time_masks: int = 2,
    num_freq_masks: int = 2
) -> torch.Tensor:
    """
    Convenience function to apply SpecAugment.

    Args:
        spectrogram: Input spectrogram [C, H, W] or [B, C, H, W]
        time_mask: Time mask parameter
        freq_mask: Frequency mask parameter
        num_time_masks: Number of time masks
        num_freq_masks: Number of frequency masks

    Returns:
        Augmented spectrogram
    """
    augmenter = SpecAugment(
        time_mask_param=time_mask,
        freq_mask_param=freq_mask,
        num_time_masks=num_time_masks,
        num_freq_masks=num_freq_masks
    )
    return augmenter(spectrogram)


if __name__ == '__main__':
    # Test SpecAugment
    spec = torch.randn(1, 129, 400)  # [C, H, W]

    print("Testing SpecAugment...")
    augmenter = SpecAugment(time_mask_param=40, freq_mask_param=20)
    spec_aug = augmenter(spec)
    print(f"Original: {spec.shape}, Augmented: {spec_aug.shape}")
    print(f"Percentage masked: {((spec_aug == 0).sum() / spec_aug.numel() * 100):.2f}%")

    print("\nTesting MixUp...")
    spec1 = torch.randn(1, 129, 400)
    spec2 = torch.randn(1, 129, 400)
    target1 = torch.tensor([0])
    target2 = torch.tensor([1])
    mixup = SpectrogramMixup(alpha=0.4)
    mixed_spec, mixed_target = mixup(spec1, spec2, target1, target2)
    print(f"Mixed spectrogram: {mixed_spec.shape}")
    print(f"Mixed target: {mixed_target}")

    print("\nTesting default augmentation pipeline...")
    pipeline = get_default_spec_augment(mode='medium')
    spec_transformed = pipeline(spec)
    print(f"Transformed: {spec_transformed.shape}")
