"""
Advanced Data Augmentation for Deep Learning

Implements stronger augmentation techniques for Phase 3 deeper models:
- CutMix: Cut-and-paste augmentation with label mixing
- Adversarial Augmentation: FGSM-based perturbations
- AutoAugment: Learned augmentation policies
- Time Masking: Random time segment masking

Reference:
- Yun et al. (2019). "CutMix: Regularization Strategy to Train Strong Classifiers"
- Goodfellow et al. (2014). "Explaining and Harnessing Adversarial Examples"
- Cubuk et al. (2019). "AutoAugment: Learning Augmentation Policies"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Callable
import random
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH


def cutmix(
    signal1: torch.Tensor,
    signal2: torch.Tensor,
    label1: torch.Tensor,
    label2: torch.Tensor,
    alpha: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CutMix augmentation: Cut and paste segments between signals.

    Randomly cuts a segment from signal2 and pastes it into signal1,
    mixing labels proportionally.

    Args:
        signal1: First signal tensor [B, C, T]
        signal2: Second signal tensor [B, C, T]
        label1: Labels for signal1 [B] or [B, num_classes]
        label2: Labels for signal2 [B] or [B, num_classes]
        alpha: Beta distribution parameter (higher = more mixing)

    Returns:
        mixed_signal: Augmented signal
        mixed_label: Mixed labels
    """
    batch_size, channels, length = signal1.shape

    # Sample mixing ratio from Beta distribution
    lam = np.random.beta(alpha, alpha)

    # Calculate cut length
    cut_length = int(lam * length)

    # Random cut position
    cut_start = np.random.randint(0, length - cut_length + 1)
    cut_end = cut_start + cut_length

    # Create mixed signal
    mixed_signal = signal1.clone()
    mixed_signal[:, :, cut_start:cut_end] = signal2[:, :, cut_start:cut_end]

    # Mix labels proportionally
    # If labels are class indices, convert to one-hot
    if label1.dim() == 1:
        num_classes = label1.max().item() + 1
        label1_onehot = F.one_hot(label1, num_classes=num_classes).float()
        label2_onehot = F.one_hot(label2, num_classes=num_classes).float()
    else:
        label1_onehot = label1
        label2_onehot = label2

    # Mix labels
    mixed_label = lam * label1_onehot + (1 - lam) * label2_onehot

    return mixed_signal, mixed_label


def cutmix_batch(
    signals: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 1.0,
    prob: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply CutMix augmentation to a batch with specified probability.

    Args:
        signals: Batch of signals [B, C, T]
        labels: Batch of labels [B] or [B, num_classes]
        alpha: Beta distribution parameter
        prob: Probability of applying CutMix

    Returns:
        Augmented signals and labels
    """
    if np.random.rand() > prob:
        return signals, labels

    # Shuffle batch to create pairs
    batch_size = signals.shape[0]
    indices = torch.randperm(batch_size, device=signals.device)

    shuffled_signals = signals[indices]
    shuffled_labels = labels[indices]

    # Apply CutMix
    mixed_signals, mixed_labels = cutmix(
        signals, shuffled_signals,
        labels, shuffled_labels,
        alpha=alpha
    )

    return mixed_signals, mixed_labels


def adversarial_augmentation(
    model: nn.Module,
    signal: torch.Tensor,
    label: torch.Tensor,
    epsilon: float = 0.01,
    criterion: Optional[Callable] = None
) -> torch.Tensor:
    """
    Adversarial augmentation using Fast Gradient Sign Method (FGSM).

    Creates adversarial examples by perturbing input in direction of gradient.

    Args:
        model: Neural network model
        signal: Input signal [B, C, T]
        label: True label [B]
        epsilon: Perturbation magnitude
        criterion: Loss function (default: CrossEntropyLoss)

    Returns:
        Adversarial signal
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Ensure model is in eval mode for generating adversarial examples
    model.eval()

    # Require gradient for input
    signal_adv = signal.clone().detach().requires_grad_(True)

    # Forward pass
    output = model(signal_adv)

    # Calculate loss
    if label.dim() > 1:  # One-hot labels
        label = label.argmax(dim=1)
    loss = criterion(output, label)

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Generate adversarial example
    # FGSM: x_adv = x + epsilon * sign(gradient)
    with torch.no_grad():
        perturbation = epsilon * signal_adv.grad.sign()
        signal_adv = signal + perturbation

    return signal_adv.detach()


def time_masking(
    signal: torch.Tensor,
    mask_param: int = 1000,
    num_masks: int = 1
) -> torch.Tensor:
    """
    Time masking augmentation: Mask random time segments.

    Similar to SpecAugment but for time-domain signals.

    Args:
        signal: Input signal [B, C, T]
        mask_param: Maximum length of masked segment
        num_masks: Number of masks to apply

    Returns:
        Masked signal
    """
    batch_size, channels, length = signal.shape
    masked_signal = signal.clone()

    for _ in range(num_masks):
        # Random mask length
        mask_length = np.random.randint(0, min(mask_param, length // 4))

        # Random mask start position
        mask_start = np.random.randint(0, length - mask_length + 1)

        # Apply mask (set to zero)
        masked_signal[:, :, mask_start:mask_start+mask_length] = 0

    return masked_signal


def gaussian_noise_augmentation(
    signal: torch.Tensor,
    noise_std: float = 0.01
) -> torch.Tensor:
    """
    Add Gaussian noise to signal.

    Args:
        signal: Input signal [B, C, T]
        noise_std: Standard deviation of noise

    Returns:
        Noisy signal
    """
    noise = torch.randn_like(signal) * noise_std
    return signal + noise


def amplitude_scaling(
    signal: torch.Tensor,
    scale_range: Tuple[float, float] = (0.8, 1.2)
) -> torch.Tensor:
    """
    Random amplitude scaling.

    Args:
        signal: Input signal [B, C, T]
        scale_range: Min and max scaling factors

    Returns:
        Scaled signal
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return signal * scale


def time_shift(
    signal: torch.Tensor,
    shift_max: int = 1000
) -> torch.Tensor:
    """
    Random time shift (circular shift).

    Args:
        signal: Input signal [B, C, T]
        shift_max: Maximum shift amount

    Returns:
        Shifted signal
    """
    shift = np.random.randint(-shift_max, shift_max)
    return torch.roll(signal, shifts=shift, dims=-1)


class AutoAugment:
    """
    AutoAugment-style learned augmentation policy.

    Applies a sequence of augmentation operations with learned probabilities.
    """

    def __init__(self, policy: Optional[List] = None):
        """
        Args:
            policy: List of augmentation operations. If None, uses default policy.
        """
        if policy is None:
            # Default policy for vibration signals
            self.policy = [
                [('gaussian_noise', 0.3, 0.01), ('amplitude_scale', 0.5, 0.1)],
                [('time_shift', 0.5, 500), ('time_mask', 0.3, 1000)],
                [('gaussian_noise', 0.5, 0.02), ('time_shift', 0.5, 1000)],
            ]
        else:
            self.policy = policy

        self.operations = {
            'gaussian_noise': self._apply_noise,
            'amplitude_scale': self._apply_scale,
            'time_shift': self._apply_shift,
            'time_mask': self._apply_mask,
        }

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentation policy.

        Args:
            signal: Input signal [B, C, T]

        Returns:
            Augmented signal
        """
        # Randomly select one sub-policy
        sub_policy = random.choice(self.policy)

        # Apply operations in sequence
        augmented = signal
        for operation_name, prob, magnitude in sub_policy:
            if random.random() < prob:
                operation = self.operations[operation_name]
                augmented = operation(augmented, magnitude)

        return augmented

    def _apply_noise(self, signal: torch.Tensor, magnitude: float) -> torch.Tensor:
        return gaussian_noise_augmentation(signal, noise_std=magnitude)

    def _apply_scale(self, signal: torch.Tensor, magnitude: float) -> torch.Tensor:
        scale_range = (1.0 - magnitude, 1.0 + magnitude)
        return amplitude_scaling(signal, scale_range=scale_range)

    def _apply_shift(self, signal: torch.Tensor, magnitude: float) -> torch.Tensor:
        return time_shift(signal, shift_max=int(magnitude))

    def _apply_mask(self, signal: torch.Tensor, magnitude: float) -> torch.Tensor:
        return time_masking(signal, mask_param=int(magnitude), num_masks=1)


class MixupAugmentation:
    """
    Mixup augmentation: Linear interpolation between samples.

    Reference: Zhang et al. (2018). "mixup: Beyond Empirical Risk Minimization"
    """

    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Beta distribution parameter
        """
        self.alpha = alpha

    def __call__(
        self,
        signal1: torch.Tensor,
        signal2: torch.Tensor,
        label1: torch.Tensor,
        label2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mix two samples linearly.

        Args:
            signal1: First signal [B, C, T]
            signal2: Second signal [B, C, T]
            label1: Labels for signal1 [B] or [B, num_classes]
            label2: Labels for signal2 [B] or [B, num_classes]

        Returns:
            Mixed signal and labels
        """
        lam = np.random.beta(self.alpha, self.alpha)

        # Mix signals
        mixed_signal = lam * signal1 + (1 - lam) * signal2

        # Mix labels
        if label1.dim() == 1:
            num_classes = label1.max().item() + 1
            label1_onehot = F.one_hot(label1, num_classes=num_classes).float()
            label2_onehot = F.one_hot(label2, num_classes=num_classes).float()
        else:
            label1_onehot = label1
            label2_onehot = label2

        mixed_label = lam * label1_onehot + (1 - lam) * label2_onehot

        return mixed_signal, mixed_label


class CompositeAugmentation:
    """
    Composite augmentation pipeline combining multiple techniques.
    """

    def __init__(
        self,
        use_cutmix: bool = True,
        use_mixup: bool = False,
        use_autoaugment: bool = True,
        cutmix_prob: float = 0.5,
        cutmix_alpha: float = 1.0,
        mixup_alpha: float = 1.0
    ):
        """
        Args:
            use_cutmix: Whether to use CutMix
            use_mixup: Whether to use Mixup
            use_autoaugment: Whether to use AutoAugment
            cutmix_prob: Probability of applying CutMix
            cutmix_alpha: CutMix alpha parameter
            mixup_alpha: Mixup alpha parameter
        """
        self.use_cutmix = use_cutmix
        self.use_mixup = use_mixup
        self.use_autoaugment = use_autoaugment

        self.cutmix_prob = cutmix_prob
        self.cutmix_alpha = cutmix_alpha

        if use_autoaugment:
            self.autoaugment = AutoAugment()

        if use_mixup:
            self.mixup = MixupAugmentation(alpha=mixup_alpha)

    def __call__(
        self,
        signals: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply composite augmentation.

        Args:
            signals: Batch of signals [B, C, T]
            labels: Batch of labels [B] or [B, num_classes]

        Returns:
            Augmented signals and labels
        """
        # Apply AutoAugment first (basic augmentations)
        if self.use_autoaugment:
            signals = self.autoaugment(signals)

        # Apply mixing augmentations
        if self.use_cutmix and random.random() < self.cutmix_prob:
            signals, labels = cutmix_batch(
                signals, labels,
                alpha=self.cutmix_alpha,
                prob=1.0  # Already checked probability
            )
        elif self.use_mixup:
            # Shuffle batch for mixup
            batch_size = signals.shape[0]
            indices = torch.randperm(batch_size, device=signals.device)
            signals, labels = self.mixup(
                signals, signals[indices],
                labels, labels[indices]
            )

        return signals, labels


# Test augmentations
if __name__ == "__main__":
    print("Testing advanced augmentation techniques...")

    # Create dummy data
    signals = torch.randn(4, 1, 10240)
    labels = torch.randint(0, 11, (4,))

    print(f"\nOriginal shape: {signals.shape}")

    # Test CutMix
    print("\nTesting CutMix...")
    mixed_signals, mixed_labels = cutmix_batch(signals, labels, alpha=1.0, prob=1.0)
    print(f"CutMix output: {mixed_signals.shape}, {mixed_labels.shape}")

    # Test Time Masking
    print("\nTesting Time Masking...")
    masked = time_masking(signals, mask_param=1000, num_masks=2)
    print(f"Time masking output: {masked.shape}")

    # Test AutoAugment
    print("\nTesting AutoAugment...")
    autoaug = AutoAugment()
    augmented = autoaug(signals)
    print(f"AutoAugment output: {augmented.shape}")

    # Test Composite
    print("\nTesting Composite Augmentation...")
    composite = CompositeAugmentation()
    aug_signals, aug_labels = composite(signals, labels)
    print(f"Composite output: {aug_signals.shape}, {aug_labels.shape}")

    print("\n✓ All augmentation tests passed!")
