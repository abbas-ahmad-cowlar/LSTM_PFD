"""
Loss Functions for Fault Diagnosis

Consolidated loss module — re-exports canonical implementations from cnn_losses
and adds physics-specific losses.

Available losses:
- FocalLoss (from cnn_losses)
- LabelSmoothingCrossEntropy (from cnn_losses)
- SupConLoss (from cnn_losses)
- PhysicsInformedLoss (defined here)
- compute_class_weights (defined here)
- create_criterion (from cnn_losses)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from utils.constants import NUM_CLASSES

# Re-export canonical implementations (avoid duplication)
from training.cnn_losses import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
    SupConLoss,
    create_criterion,
)


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss for PINN models.

    Combines:
    - Data loss (cross-entropy)
    - Physics constraint loss

    Args:
        data_weight: Weight for data loss (default: 1.0)
        physics_weight: Weight for physics loss (default: 0.1)
    """
    def __init__(self, data_weight: float = 1.0, physics_weight: float = 0.1):
        super().__init__()
        self.data_weight = data_weight
        self.physics_weight = physics_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        physics_loss: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            predictions: Model predictions [B, num_classes]
            targets: Ground truth labels [B]
            physics_loss: Optional physics constraint loss

        Returns:
            Total loss
        """
        # Data-driven loss
        data_loss = self.ce_loss(predictions, targets)

        # Total loss
        total_loss = self.data_weight * data_loss

        if physics_loss is not None:
            total_loss += self.physics_weight * physics_loss

        return total_loss


def compute_class_weights(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.

    Args:
        targets: All training labels [N]
        num_classes: Number of classes

    Returns:
        Class weights [num_classes]
    """
    class_counts = torch.bincount(targets, minlength=num_classes).float()
    total = class_counts.sum()

    # Inverse frequency weighting
    weights = total / (num_classes * class_counts + 1e-10)

    return weights
