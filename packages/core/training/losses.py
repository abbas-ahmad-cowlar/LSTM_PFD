"""
Loss Functions — backward-compat re-export shim.

All loss implementations are now in ``training.losses``:
- Classification: training.losses.classification
- Distillation: training.losses.distillation
- Physics: training.physics_loss_functions
- Contrastive: training.contrastive.losses
"""

# Re-export classification losses
from .losses.classification import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
    SupConLoss,
    create_criterion,
)

# Re-export physics losses
from .physics_loss_functions import PhysicalConstraintLoss as PhysicsInformedLoss

# Utility function
from .losses.classification import LabelSmoothingCrossEntropy as LabelSmoothingLoss

import torch
import numpy as np


def compute_class_weights(labels, num_classes=None):
    """
    Compute class weights inversely proportional to class frequency.

    Args:
        labels: Array or tensor of class labels
        num_classes: Number of classes (auto-detected if None)

    Returns:
        Tensor of class weights
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    labels = np.array(labels)

    if num_classes is None:
        num_classes = len(np.unique(labels))

    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts[counts == 0] = 1.0  # Avoid division by zero

    weights = len(labels) / (num_classes * counts)

    return torch.FloatTensor(weights)


__all__ = [
    "FocalLoss",
    "LabelSmoothingCrossEntropy",
    "LabelSmoothingLoss",
    "SupConLoss",
    "PhysicsInformedLoss",
    "compute_class_weights",
    "create_criterion",
]
