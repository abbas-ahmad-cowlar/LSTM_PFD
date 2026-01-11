"""
Patch-Based Augmentation for Transformer Models

Implements augmentation techniques specific to patch-based transformer models:
- Patch Dropout: Randomly drop patches during training (regularization)
- Patch Mixup: Mix patches from two signals
- Temporal Shift: Shift patches in time
- Patch Cutout: Mask out random patches

These techniques help transformers generalize better and prevent overfitting.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def patch_dropout(
    patches: torch.Tensor,
    drop_prob: float = 0.1,
    training: bool = True
) -> torch.Tensor:
    """
    Randomly drop patches during training.

    Similar to DropBlock, this regularizes the model by forcing it to not rely
    on specific patches for classification.

    Args:
        patches: Patch embeddings of shape [B, n_patches, d_model]
        drop_prob: Probability of dropping each patch
        training: Only apply during training

    Returns:
        Patches with some randomly dropped (set to zero)

    Example:
        >>> patches = torch.randn(4, 200, 256)  # 4 samples, 200 patches, 256-dim
        >>> augmented = patch_dropout(patches, drop_prob=0.1)
    """
    if not training or drop_prob == 0:
        return patches

    batch_size, n_patches, d_model = patches.shape

    # Create dropout mask [B, n_patches, 1]
    keep_prob = 1 - drop_prob
    mask = torch.bernoulli(torch.full((batch_size, n_patches, 1), keep_prob, device=patches.device))

    # Apply mask and scale to maintain expected value
    patches = patches * mask / keep_prob

    return patches


def patch_mixup(
    patches1: torch.Tensor,
    patches2: torch.Tensor,
    labels1: torch.Tensor,
    labels2: torch.Tensor,
    alpha: float = 0.4
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Mix patches from two signals with interpolation.

    Patch-level mixup mixes corresponding patches from two signals,
    creating a hybrid signal for better generalization.

    Args:
        patches1: First batch of patches [B, n_patches, d_model]
        patches2: Second batch of patches [B, n_patches, d_model]
        labels1: Labels for first batch [B]
        labels2: Labels for second batch [B]
        alpha: Beta distribution parameter (higher = more mixing)

    Returns:
        Tuple of (mixed patches, mixed labels)

    Example:
        >>> patches1 = torch.randn(4, 200, 256)
        >>> patches2 = torch.randn(4, 200, 256)
        >>> labels1 = torch.tensor([0, 1, 2, 3])
        >>> labels2 = torch.tensor([4, 5, 6, 7])
        >>> mixed_patches, mixed_labels = patch_mixup(patches1, patches2, labels1, labels2)
    """
    batch_size = patches1.size(0)
    num_classes = labels1.max().item() + 1

    # Sample mixing coefficient from beta distribution
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    # Mix patches
    mixed_patches = lam * patches1 + (1 - lam) * patches2

    # Mix labels (convert to one-hot first)
    labels1_onehot = F.one_hot(labels1, num_classes=num_classes).float()
    labels2_onehot = F.one_hot(labels2, num_classes=num_classes).float()
    mixed_labels = lam * labels1_onehot + (1 - lam) * labels2_onehot

    return mixed_patches, mixed_labels


def temporal_shift_patches(
    patches: torch.Tensor,
    max_shift: int = 5,
    circular: bool = True
) -> torch.Tensor:
    """
    Shift patches in time dimension.

    This augments the temporal relationship between patches, helping the model
    become invariant to small temporal shifts.

    Args:
        patches: Patch embeddings [B, n_patches, d_model]
        max_shift: Maximum number of patches to shift
        circular: If True, use circular shift; else pad with zeros

    Returns:
        Temporally shifted patches

    Example:
        >>> patches = torch.randn(4, 200, 256)
        >>> shifted = temporal_shift_patches(patches, max_shift=3)
    """
    if max_shift == 0:
        return patches

    batch_size, n_patches, d_model = patches.shape

    # Random shift for each sample in batch
    shifts = torch.randint(-max_shift, max_shift + 1, (batch_size,))

    shifted_patches = []
    for i, shift in enumerate(shifts):
        if shift == 0:
            shifted_patches.append(patches[i])
        elif circular:
            # Circular shift
            shifted_patches.append(torch.roll(patches[i], shifts=shift.item(), dims=0))
        else:
            # Shift with zero padding
            if shift > 0:
                # Shift right (pad left)
                pad = torch.zeros(shift, d_model, device=patches.device)
                shifted = torch.cat([pad, patches[i][:-shift]], dim=0)
            else:
                # Shift left (pad right)
                pad = torch.zeros(-shift, d_model, device=patches.device)
                shifted = torch.cat([patches[i][-shift:], pad], dim=0)
            shifted_patches.append(shifted)

    return torch.stack(shifted_patches)


def patch_cutout(
    patches: torch.Tensor,
    n_holes: int = 10,
    hole_size: int = 5
) -> torch.Tensor:
    """
    Apply cutout to patches (mask out random regions).

    Similar to Cutout augmentation for images, this masks out contiguous
    regions of patches.

    Args:
        patches: Patch embeddings [B, n_patches, d_model]
        n_holes: Number of cutout regions
        hole_size: Size of each hole (in number of patches)

    Returns:
        Patches with cutout applied

    Example:
        >>> patches = torch.randn(4, 200, 256)
        >>> augmented = patch_cutout(patches, n_holes=5, hole_size=3)
    """
    batch_size, n_patches, d_model = patches.shape
    augmented = patches.clone()

    for b in range(batch_size):
        for _ in range(n_holes):
            # Random starting position
            start_idx = np.random.randint(0, max(1, n_patches - hole_size))
            end_idx = min(start_idx + hole_size, n_patches)

            # Mask out
            augmented[b, start_idx:end_idx, :] = 0

    return augmented


def patch_permutation(
    patches: torch.Tensor,
    permute_prob: float = 0.1
) -> torch.Tensor:
    """
    Randomly permute (swap) patch positions.

    This forces the model to learn more robust temporal relationships.

    Args:
        patches: Patch embeddings [B, n_patches, d_model]
        permute_prob: Probability of permuting each patch pair

    Returns:
        Patches with some positions permuted

    Example:
        >>> patches = torch.randn(4, 200, 256)
        >>> permuted = patch_permutation(patches, permute_prob=0.05)
    """
    batch_size, n_patches, d_model = patches.shape
    augmented = patches.clone()

    for b in range(batch_size):
        # Number of swaps to perform
        n_swaps = int(n_patches * permute_prob / 2)

        for _ in range(n_swaps):
            # Random pair to swap
            idx1, idx2 = np.random.choice(n_patches, size=2, replace=False)
            augmented[b, [idx1, idx2]] = augmented[b, [idx2, idx1]]

    return augmented


def patch_jitter(
    patches: torch.Tensor,
    noise_std: float = 0.01
) -> torch.Tensor:
    """
    Add small random noise to patch embeddings.

    Args:
        patches: Patch embeddings [B, n_patches, d_model]
        noise_std: Standard deviation of Gaussian noise

    Returns:
        Patches with added noise

    Example:
        >>> patches = torch.randn(4, 200, 256)
        >>> jittered = patch_jitter(patches, noise_std=0.01)
    """
    noise = torch.randn_like(patches) * noise_std
    return patches + noise


class PatchAugmentation:
    """
    Composable patch augmentation pipeline.

    Combines multiple augmentation techniques with configurable probabilities.

    Args:
        use_dropout: Enable patch dropout
        dropout_prob: Dropout probability
        use_cutout: Enable patch cutout
        cutout_holes: Number of cutout holes
        cutout_size: Size of each hole
        use_shift: Enable temporal shift
        max_shift: Maximum shift amount
        use_jitter: Enable patch jitter
        jitter_std: Jitter noise std
        use_permutation: Enable patch permutation
        permute_prob: Permutation probability

    Example:
        >>> aug = PatchAugmentation(
        ...     use_dropout=True, dropout_prob=0.1,
        ...     use_cutout=True, cutout_holes=5
        ... )
        >>> patches = torch.randn(4, 200, 256)
        >>> augmented = aug(patches, training=True)
    """

    def __init__(
        self,
        use_dropout: bool = True,
        dropout_prob: float = 0.1,
        use_cutout: bool = False,
        cutout_holes: int = 5,
        cutout_size: int = 3,
        use_shift: bool = True,
        max_shift: int = 3,
        use_jitter: bool = False,
        jitter_std: float = 0.01,
        use_permutation: bool = False,
        permute_prob: float = 0.05
    ):
        self.use_dropout = use_dropout
        self.dropout_prob = dropout_prob
        self.use_cutout = use_cutout
        self.cutout_holes = cutout_holes
        self.cutout_size = cutout_size
        self.use_shift = use_shift
        self.max_shift = max_shift
        self.use_jitter = use_jitter
        self.jitter_std = jitter_std
        self.use_permutation = use_permutation
        self.permute_prob = permute_prob

    def __call__(self, patches: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Apply augmentation pipeline.

        Args:
            patches: Input patches [B, n_patches, d_model]
            training: Only apply during training

        Returns:
            Augmented patches
        """
        if not training:
            return patches

        # Apply augmentations in sequence
        if self.use_dropout:
            patches = patch_dropout(patches, self.dropout_prob, training=True)

        if self.use_cutout:
            patches = patch_cutout(patches, self.cutout_holes, self.cutout_size)

        if self.use_shift:
            patches = temporal_shift_patches(patches, self.max_shift)

        if self.use_jitter:
            patches = patch_jitter(patches, self.jitter_std)

        if self.use_permutation:
            patches = patch_permutation(patches, self.permute_prob)

        return patches


# Preset augmentation configurations
def get_light_augmentation() -> PatchAugmentation:
    """Light augmentation for faster training."""
    return PatchAugmentation(
        use_dropout=True, dropout_prob=0.05,
        use_shift=True, max_shift=2,
        use_cutout=False,
        use_jitter=False,
        use_permutation=False
    )


def get_medium_augmentation() -> PatchAugmentation:
    """Medium augmentation (recommended for most cases)."""
    return PatchAugmentation(
        use_dropout=True, dropout_prob=0.1,
        use_cutout=True, cutout_holes=5, cutout_size=3,
        use_shift=True, max_shift=3,
        use_jitter=False,
        use_permutation=False
    )


def get_heavy_augmentation() -> PatchAugmentation:
    """Heavy augmentation for preventing overfitting."""
    return PatchAugmentation(
        use_dropout=True, dropout_prob=0.15,
        use_cutout=True, cutout_holes=10, cutout_size=5,
        use_shift=True, max_shift=5,
        use_jitter=True, jitter_std=0.02,
        use_permutation=True, permute_prob=0.1
    )


if __name__ == '__main__':
    # Test all augmentation functions
    print("Testing patch augmentation functions...\n")

    batch_size = 4
    n_patches = 200
    d_model = 256

    # Create sample patches
    patches = torch.randn(batch_size, n_patches, d_model)
    print(f"Input patches shape: {patches.shape}")

    # Test patch dropout
    print("\n1. Testing patch_dropout...")
    dropped = patch_dropout(patches, drop_prob=0.1, training=True)
    print(f"   Output shape: {dropped.shape}")
    print(f"   Fraction of zero patches: {(dropped.abs().sum(dim=-1) == 0).float().mean():.3f}")

    # Test patch mixup
    print("\n2. Testing patch_mixup...")
    patches2 = torch.randn(batch_size, n_patches, d_model)
    labels1 = torch.randint(0, 11, (batch_size,))
    labels2 = torch.randint(0, 11, (batch_size,))
    mixed, mixed_labels = patch_mixup(patches, patches2, labels1, labels2, alpha=0.4)
    print(f"   Mixed patches shape: {mixed.shape}")
    print(f"   Mixed labels shape: {mixed_labels.shape}")

    # Test temporal shift
    print("\n3. Testing temporal_shift_patches...")
    shifted = temporal_shift_patches(patches, max_shift=5, circular=True)
    print(f"   Shifted shape: {shifted.shape}")

    # Test cutout
    print("\n4. Testing patch_cutout...")
    cutout = patch_cutout(patches, n_holes=5, hole_size=3)
    print(f"   Cutout shape: {cutout.shape}")
    print(f"   Fraction of zero patches: {(cutout.abs().sum(dim=-1) == 0).float().mean():.3f}")

    # Test permutation
    print("\n5. Testing patch_permutation...")
    permuted = patch_permutation(patches, permute_prob=0.1)
    print(f"   Permuted shape: {permuted.shape}")

    # Test jitter
    print("\n6. Testing patch_jitter...")
    jittered = patch_jitter(patches, noise_std=0.01)
    print(f"   Jittered shape: {jittered.shape}")

    # Test augmentation pipeline
    print("\n7. Testing PatchAugmentation pipeline...")
    aug = get_medium_augmentation()
    augmented = aug(patches, training=True)
    print(f"   Augmented shape: {augmented.shape}")

    # Test presets
    print("\n8. Testing preset configurations...")
    for name, aug_fn in [
        ('light', get_light_augmentation),
        ('medium', get_medium_augmentation),
        ('heavy', get_heavy_augmentation)
    ]:
        aug = aug_fn()
        result = aug(patches, training=True)
        print(f"   {name}: output shape = {result.shape}")

    print("\n✅ All tests passed!")
