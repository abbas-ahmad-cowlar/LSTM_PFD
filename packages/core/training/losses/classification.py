"""
Classification Loss Functions

Canonical implementations for fault-diagnosis classification:
- LabelSmoothingCrossEntropy: Softened hard labels for calibration
- FocalLoss: Addresses class imbalance (focus on hard examples)
- SupConLoss: Supervised contrastive learning
- create_criterion: Factory function

Moved from: training/cnn_losses.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from utils.constants import NUM_CLASSES


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing regularization.

    Args:
        smoothing: Smoothing factor ε (0 = no smoothing, 0.1 = 10% smoothing)
        reduction: Loss reduction ('mean', 'sum', 'none')
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        assert 0.0 <= smoothing < 1.0, "Smoothing must be in [0, 1)"
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / num_classes)
            true_dist.scatter_(
                1, targets.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / num_classes
            )

        loss = torch.sum(-true_dist * log_probs, dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    FL = -α(1-p)^γ · log(p)

    Args:
        alpha: Class weight for positive class (None = no weighting)
        gamma: Focusing parameter (0 = standard CE, 2 = strong focus)
        reduction: Loss reduction ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1))
        p_t = (probs * targets_one_hot).sum(dim=-1)

        focal_weight = (1 - p_t) ** self.gamma
        ce_loss = -torch.log(p_t + 1e-8)
        loss = focal_weight * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[targets]
                loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Args:
        temperature: Temperature parameter for scaling
        base_temperature: Base temperature (default 0.07)
    """

    def __init__(self, temperature: float = 0.1, base_temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T), self.temperature
        )

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


def create_criterion(
    criterion_type: str = "label_smoothing",
    num_classes: int = NUM_CLASSES,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create loss criterion.

    Args:
        criterion_type: 'ce', 'label_smoothing', 'focal', 'supcon'
        num_classes: Number of classes
        **kwargs: Additional kwargs

    Returns:
        Loss criterion module
    """
    if criterion_type == "ce":
        return nn.CrossEntropyLoss()
    elif criterion_type == "label_smoothing":
        smoothing = kwargs.get("smoothing", 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    elif criterion_type == "focal":
        gamma = kwargs.get("gamma", 2.0)
        alpha = kwargs.get("alpha", None)
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif criterion_type == "supcon":
        temperature = kwargs.get("temperature", 0.1)
        return SupConLoss(temperature=temperature)
    else:
        raise ValueError(f"Unknown criterion type: {criterion_type}")
