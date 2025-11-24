"""
Training Metrics Computation

Provides utilities for computing and tracking metrics during training:
- Accuracy
- F1 score (macro, micro, weighted)
- Precision and recall
- Confusion matrix
- Top-k accuracy
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix as sklearn_confusion_matrix
)
from typing import Dict, Optional, Tuple
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH


class MetricsTracker:
    """
    Tracks and aggregates metrics across batches.

    Example:
        >>> tracker = MetricsTracker()
        >>> for batch in dataloader:
        ...     preds, targets = model(batch), batch.labels
        ...     tracker.update(preds, targets)
        >>> metrics = tracker.compute()
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all tracked values."""
        self.predictions = []
        self.targets = []

    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update with new predictions and targets.

        Args:
            predictions: Model predictions [B, num_classes] or [B]
            targets: Ground truth labels [B]
        """
        # Convert logits to class predictions if needed
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=1)

        # Move to CPU and convert to numpy
        predictions = predictions.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        self.predictions.append(predictions)
        self.targets.append(targets)

    def compute(self, average: str = 'macro') -> Dict[str, float]:
        """
        Compute metrics from accumulated predictions.

        Args:
            average: Averaging strategy for multi-class metrics
                    ('macro', 'micro', 'weighted')

        Returns:
            Dictionary of metrics
        """
        if not self.predictions:
            return {}

        # Concatenate all batches
        preds = np.concatenate(self.predictions)
        targets = np.concatenate(self.targets)

        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(targets, preds),
            'f1_score': f1_score(targets, preds, average=average, zero_division=0),
            'precision': precision_score(targets, preds, average=average, zero_division=0),
            'recall': recall_score(targets, preds, average=average, zero_division=0),
        }

        return metrics


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy.

    Args:
        predictions: Model predictions [B, num_classes] or [B]
        targets: Ground truth labels [B]

    Returns:
        Accuracy (0-100)
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)

    correct = (predictions == targets).sum().item()
    total = targets.size(0)

    return 100.0 * correct / total


def compute_f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    average: str = 'macro'
) -> float:
    """
    Compute F1 score.

    Args:
        predictions: Model predictions [B, num_classes] or [B]
        targets: Ground truth labels [B]
        average: Averaging strategy ('macro', 'micro', 'weighted')

    Returns:
        F1 score
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)

    preds_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    return f1_score(targets_np, preds_np, average=average, zero_division=0)


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        predictions: Model predictions [B, num_classes] or [B]
        targets: Ground truth labels [B]
        num_classes: Number of classes (optional)

    Returns:
        Confusion matrix [num_classes, num_classes]
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)

    preds_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    labels = None
    if num_classes is not None:
        labels = list(range(num_classes))

    return sklearn_confusion_matrix(targets_np, preds_np, labels=labels)


def compute_top_k_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5
) -> float:
    """
    Compute top-k accuracy.

    Args:
        predictions: Model predictions [B, num_classes]
        targets: Ground truth labels [B]
        k: Top-k to consider

    Returns:
        Top-k accuracy (0-100)
    """
    with torch.no_grad():
        batch_size = targets.size(0)

        # Get top-k predictions
        _, pred_topk = predictions.topk(k, dim=1, largest=True, sorted=True)

        # Check if target is in top-k
        targets = targets.view(-1, 1).expand_as(pred_topk)
        correct = pred_topk.eq(targets).any(dim=1).sum().item()

        return 100.0 * correct / batch_size


def compute_per_class_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int
) -> Dict[int, Dict[str, float]]:
    """
    Compute per-class precision, recall, and F1 score.

    Args:
        predictions: Model predictions [B, num_classes] or [B]
        targets: Ground truth labels [B]
        num_classes: Number of classes

    Returns:
        Dictionary mapping class index to metrics
    """
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)

    preds_np = predictions.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    # Compute per-class metrics
    precision = precision_score(
        targets_np,
        preds_np,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0
    )
    recall = recall_score(
        targets_np,
        preds_np,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0
    )
    f1 = f1_score(
        targets_np,
        preds_np,
        average=None,
        labels=list(range(num_classes)),
        zero_division=0
    )

    per_class_metrics = {}
    for i in range(num_classes):
        per_class_metrics[i] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i]
        }

    return per_class_metrics
