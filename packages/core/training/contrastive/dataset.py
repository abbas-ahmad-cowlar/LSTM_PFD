"""
Contrastive Learning Datasets

Dataset classes for both 1D signal (physics-based) and 2D spectrogram
(SimCLR-style) contrastive learning.

Extracted from:
- scripts/research/contrastive_physics.py (PhysicsContrastiveDataset, FineTuneDataset)
- data/contrast_learning_tfr.py (ContrastiveSpectrogramDataset)
"""

import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable, Dict, List, Optional, Tuple

from .physics_similarity import select_positive_negative_pairs

logger = logging.getLogger(__name__)


class PhysicsContrastiveDataset(Dataset):
    """
    Dataset that yields triplets (anchor, positive, negatives) based on
    physics similarity rather than class labels.

    This enables the model to learn representations that capture
    underlying physics relationships.
    """

    def __init__(self,
                 signals: np.ndarray,
                 physics_params: List[Dict[str, float]],
                 similarity_threshold: float = 0.8,
                 num_negatives: int = 5,
                 augment: bool = True):
        """
        Args:
            signals: Time-series signals (N, C, L) or (N, L)
            physics_params: List of physics parameter dicts for each signal
            similarity_threshold: Min similarity for positive pairs
            num_negatives: Number of negative samples per anchor
            augment: Whether to apply random augmentations
        """
        self.signals = torch.FloatTensor(signals)
        if self.signals.dim() == 2:
            self.signals = self.signals.unsqueeze(1)  # Add channel dim

        self.physics_params = physics_params
        self.similarity_threshold = similarity_threshold
        self.num_negatives = num_negatives
        self.augment = augment

        # Precompute pairs
        logger.info("Precomputing positive/negative pairs based on physics similarity...")
        self.pairs = select_positive_negative_pairs(
            physics_params, similarity_threshold, num_negatives
        )
        logger.info(f"Created {len(self.pairs)} triplets")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_idx, pos_idx, neg_idxs = self.pairs[idx]

        anchor = self.signals[anchor_idx]
        positive = self.signals[pos_idx]
        negatives = self.signals[neg_idxs]  # (num_neg, C, L)

        # Apply augmentations
        if self.augment:
            anchor = self._augment(anchor)
            positive = self._augment(positive)
            negatives = torch.stack([self._augment(n) for n in negatives])

        return anchor, positive, negatives

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations."""
        # Random noise
        if np.random.random() < 0.5:
            noise = torch.randn_like(x) * 0.02
            x = x + noise

        # Random scaling
        if np.random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            x = x * scale

        # Random temporal shift
        if np.random.random() < 0.5:
            shift = np.random.randint(-50, 50)
            x = torch.roll(x, shift, dims=-1)

        return x


class ContrastiveSpectrogramDataset(Dataset):
    """
    Dataset for contrastive learning on spectrograms (SimCLR-style).

    Generates pairs of augmented views from each spectrogram.

    Args:
        spectrograms: Array of spectrograms [N, H, W]
        augmentation: Augmentation function that takes spectrogram and returns augmented version
        return_original: Whether to also return original spectrogram (default: False)
    """

    def __init__(
        self,
        spectrograms: np.ndarray,
        augmentation: Optional[Callable] = None,
        return_original: bool = False
    ):
        self.spectrograms = spectrograms
        self.augmentation = augmentation
        self.return_original = return_original

        if augmentation is None:
            # Default: identity (user should supply real augmentation)
            self.augmentation = lambda x: x

    def __len__(self) -> int:
        return len(self.spectrograms)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get two augmented views of a spectrogram.

        Returns:
            Tuple of (view1, view2) where both are augmented versions of the same spectrogram
        """
        spec = self.spectrograms[idx]  # [H, W]

        # Convert to tensor
        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)  # [1, H, W]

        # Generate two augmented views
        view1 = self.augmentation(spec_tensor)
        view2 = self.augmentation(spec_tensor)

        if self.return_original:
            return view1, view2, spec_tensor
        else:
            return view1, view2


class FineTuneDataset(Dataset):
    """Dataset for fine-tuning on downstream classification."""

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.signals = torch.FloatTensor(signals)
        if self.signals.dim() == 2:
            self.signals = self.signals.unsqueeze(1)
        self.labels = torch.LongTensor(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.signals[idx], self.labels[idx]
