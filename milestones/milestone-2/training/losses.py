"""
Loss Functions for Fault Diagnosis

Implements various loss functions:
- Focal Loss (for class imbalance)
- Label Smoothing Cross Entropy
- Physics-Informed Loss (for PINN models)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Reference: Lin et al. (2017). "Focal Loss for Dense Object Detection"

    Args:
        alpha: Weighting factor for classes (default: None)
        gamma: Focusing parameter (default: 2.0)
        reduction: 'none', 'mean', or 'sum' (default: 'mean')
    """
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions [B, num_classes]
            targets: Ground truth labels [B]

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy loss with label smoothing.

    Args:
        smoothing: Label smoothing factor (default: 0.1)
        reduction: 'none', 'mean', or 'sum' (default: 'mean')
    """
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions [B, num_classes]
            targets: Ground truth labels [B]

        Returns:
            Label smoothing loss
        """
        log_probs = F.log_softmax(inputs, dim=1)
        num_classes = inputs.size(1)

        # One-hot encode targets
        targets_one_hot = F.one_hot(targets, num_classes).float()

        # Apply label smoothing
        smooth_targets = targets_one_hot * (1 - self.smoothing) + self.smoothing / num_classes

        # Compute loss
        loss = -(smooth_targets * log_probs).sum(dim=1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


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


def create_loss_function(
    loss_name: str = 'cross_entropy',
    num_classes: int = NUM_CLASSES,
    label_smoothing: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    Create loss function by name.

    Args:
        loss_name: Name of loss function ('cross_entropy', 'label_smoothing', 'focal')
        num_classes: Number of classes
        label_smoothing: Label smoothing factor (for label_smoothing loss)
        **kwargs: Additional loss-specific arguments

    Returns:
        Loss function instance

    Example:
        >>> criterion = create_loss_function('cross_entropy')
        >>> criterion = create_loss_function('label_smoothing', label_smoothing=0.1)
        >>> criterion = create_loss_function('focal', gamma=2.0)
    """
    loss_name = loss_name.lower()

    if loss_name == 'cross_entropy':
        return nn.CrossEntropyLoss()

    elif loss_name == 'label_smoothing':
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing)

    elif loss_name == 'focal':
        alpha = kwargs.get('alpha', None)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif loss_name == 'physics_informed':
        data_weight = kwargs.get('data_weight', 1.0)
        physics_weight = kwargs.get('physics_weight', 0.1)
        return PhysicsInformedLoss(data_weight=data_weight, physics_weight=physics_weight)

    else:
        raise ValueError(
            f"Unknown loss function: {loss_name}. "
            f"Available: 'cross_entropy', 'label_smoothing', 'focal', 'physics_informed'"
        )
