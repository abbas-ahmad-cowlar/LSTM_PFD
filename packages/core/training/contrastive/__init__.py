"""
Contrastive Learning Training Package

Physics-informed contrastive pretraining for bearing fault diagnosis.
Supports both 1D signal (physics-based pairs) and 2D spectrogram (SimCLR-style)
contrastive learning.

Components:
- physics_similarity: Physics parameter similarity and pair selection
- losses: PhysicsInfoNCELoss, NTXentLoss, SimCLRLoss
- dataset: PhysicsContrastiveDataset, ContrastiveSpectrogramDataset, FineTuneDataset
- pretrainer: ContrastivePretrainer, ContrastiveFineTuner, pretrain_contrastive
"""

from .physics_similarity import compute_physics_similarity, select_positive_negative_pairs
from .losses import PhysicsInfoNCELoss, NTXentLoss, SimCLRLoss
from .dataset import PhysicsContrastiveDataset, ContrastiveSpectrogramDataset, FineTuneDataset
from .pretrainer import ContrastivePretrainer, ContrastiveFineTuner, pretrain_contrastive

__all__ = [
    # Physics similarity
    'compute_physics_similarity',
    'select_positive_negative_pairs',
    # Losses
    'PhysicsInfoNCELoss',
    'NTXentLoss',
    'SimCLRLoss',
    # Datasets
    'PhysicsContrastiveDataset',
    'ContrastiveSpectrogramDataset',
    'FineTuneDataset',
    # Training
    'ContrastivePretrainer',
    'ContrastiveFineTuner',
    'pretrain_contrastive',
]
