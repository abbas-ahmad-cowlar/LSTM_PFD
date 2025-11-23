"""
Loss functions for CNN training.

Purpose:
    Custom loss functions tailored for fault diagnosis CNN:
    - LabelSmoothingCrossEntropy: Prevent overconfident predictions
    - FocalLoss: Address class imbalance (focus on hard examples)
    - SupConLoss: Supervised contrastive learning (optional)

Author: Syed Abbas Ahmad
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing regularization.

    Label smoothing prevents overfitting by softening hard labels:
        Hard labels:  [0, 0, 0, 1, 0, ...]  (100% confidence)
        Soft labels:  [ε/K, ε/K, ε/K, 1-ε+ε/K, ε/K, ...]

    Benefits:
    - Improves calibration (predicted probabilities match actual accuracy)
    - Prevents overconfident predictions
    - Slight regularization effect (< 0.5% accuracy cost)

    Args:
        smoothing: Smoothing factor ε (0 = no smoothing, 0.1 = 10% smoothing)
        reduction: Loss reduction ('mean', 'sum', 'none')

    Example:
        >>> criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        >>> logits = torch.randn(8, 11)  # Batch of 8, 11 classes
        >>> targets = torch.randint(0, 11, (8,))
        >>> loss = criterion(logits, targets)
        >>> print(loss.item())
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert 0.0 <= smoothing < 1.0, "Smoothing must be in [0, 1)"
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed cross-entropy loss.

        Args:
            logits: Predicted logits [B, num_classes]
            targets: Ground truth labels [B] (integer class indices)

        Returns:
            Scalar loss value
        """
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Create soft labels
        with torch.no_grad():
            # Initialize all with smoothing / num_classes
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / num_classes)

            # Set correct class to 1 - smoothing + smoothing/num_classes
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing + self.smoothing / num_classes)

        # KL divergence between soft labels and predictions
        loss = torch.sum(-true_dist * log_probs, dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.

    Focuses training on hard examples by down-weighting easy examples.
    Useful when some fault classes are rare (class imbalance).

    Loss formula: FL = -α(1-p)^γ * log(p)
        - α: Class weighting factor
        - γ: Focusing parameter (γ=0 → standard CE, γ=2 → strong focus on hard examples)
        - p: Predicted probability for true class

    Args:
        alpha: Class weight for positive class (None = no weighting)
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: Loss reduction ('mean', 'sum', 'none')

    Example:
        >>> criterion = FocalLoss(gamma=2.0)
        >>> logits = torch.randn(8, 11)
        >>> targets = torch.randint(0, 11, (8,))
        >>> loss = criterion(logits, targets)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Predicted logits [B, num_classes]
            targets: Ground truth labels [B]

        Returns:
            Scalar loss value
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)

        # Get probabilities for true class
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1))
        p_t = (probs * targets_one_hot).sum(dim=-1)  # [B]

        # Compute focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross-entropy: -log(p_t)
        ce_loss = -torch.log(p_t + 1e-8)

        # Compute focal loss
        loss = focal_weight * ce_loss

        # Apply class weighting (alpha)
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[targets]
                loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (optional, advanced).

    Encourages embeddings of same class to be close, different classes to be far.
    Requires model to output embeddings (before final classification layer).

    Reference: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)

    Args:
        temperature: Temperature parameter for scaling
        base_temperature: Base temperature (default 0.07)

    Example:
        >>> criterion = SupConLoss(temperature=0.1)
        >>> features = F.normalize(model.get_features(x), dim=1)  # [B, D]
        >>> labels = torch.randint(0, 11, (B,))
        >>> loss = criterion(features, labels)
    """

    def __init__(self, temperature: float = 0.1, base_temperature: float = 0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute supervised contrastive loss.

        Args:
            features: Normalized embeddings [B, D]
            labels: Class labels [B]

        Returns:
            Scalar loss value
        """
        device = features.device
        batch_size = features.shape[0]

        # Create mask for positive pairs (same class)
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Compute similarity matrix: dot product of normalized features
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # For numerical stability, subtract max
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Exclude self-contrast (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # Compute log probabilities
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


def create_criterion(
    criterion_type: str = 'label_smoothing',
    num_classes: int = NUM_CLASSES,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss criterion.

    Args:
        criterion_type: Type of criterion ('ce', 'label_smoothing', 'focal', 'supcon')
        num_classes: Number of classes (for focal loss alpha)
        **kwargs: Additional arguments for specific criterion

    Returns:
        Loss criterion module

    Example:
        >>> criterion = create_criterion('label_smoothing', smoothing=0.1)
        >>> criterion = create_criterion('focal', gamma=2.0)
    """
    if criterion_type == 'ce':
        return nn.CrossEntropyLoss()

    elif criterion_type == 'label_smoothing':
        smoothing = kwargs.get('smoothing', 0.1)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)

    elif criterion_type == 'focal':
        gamma = kwargs.get('gamma', 2.0)
        alpha = kwargs.get('alpha', None)
        return FocalLoss(alpha=alpha, gamma=gamma)

    elif criterion_type == 'supcon':
        temperature = kwargs.get('temperature', 0.1)
        return SupConLoss(temperature=temperature)

    else:
        raise ValueError(f"Unknown criterion type: {criterion_type}")


def test_losses():
    """Test loss functions."""
    print("=" * 60)
    print("Testing CNN Loss Functions")
    print("=" * 60)

    batch_size = 16
    num_classes=NUM_CLASSES

    # Create dummy data
    logits = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Test standard CrossEntropy
    print("\n1. Testing CrossEntropyLoss...")
    ce_loss = nn.CrossEntropyLoss()
    loss = ce_loss(logits, targets)
    print(f"   Loss: {loss.item():.4f}")
    assert loss.item() > 0

    # Test LabelSmoothingCrossEntropy
    print("\n2. Testing LabelSmoothingCrossEntropy...")
    ls_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss = ls_loss(logits, targets)
    print(f"   Loss: {loss.item():.4f}")
    assert loss.item() > 0

    # Test FocalLoss
    print("\n3. Testing FocalLoss...")
    focal_loss = FocalLoss(gamma=2.0)
    loss = focal_loss(logits, targets)
    print(f"   Loss: {loss.item():.4f}")
    assert loss.item() > 0

    # Test SupConLoss
    print("\n4. Testing SupConLoss...")
    features = F.normalize(torch.randn(batch_size, 128), dim=1)
    supcon_loss = SupConLoss(temperature=0.1)
    loss = supcon_loss(features, targets)
    print(f"   Loss: {loss.item():.4f}")
    assert loss.item() > 0

    # Test create_criterion factory
    print("\n5. Testing create_criterion factory...")
    criterion1 = create_criterion('label_smoothing', smoothing=0.1)
    criterion2 = create_criterion('focal', gamma=2.0)
    print(f"   Created: {type(criterion1).__name__}, {type(criterion2).__name__}")

    # Test gradient flow
    print("\n6. Testing gradient flow...")
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    loss = criterion(logits, targets)
    loss.backward()
    assert logits.grad is not None
    print(f"   Gradients computed successfully")

    print("\n" + "=" * 60)
    print("✅ All loss function tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_losses()
