"""
CNN Loss Functions — backward-compat re-export shim.

All implementations have been moved to ``training.losses.classification``.
Import from ``training.losses`` instead.
"""

from .losses.classification import (
    LabelSmoothingCrossEntropy,
    FocalLoss,
    SupConLoss,
    create_criterion,
)

# Legacy alias for backward compat
LabelSmoothingLoss = LabelSmoothingCrossEntropy

__all__ = [
    "LabelSmoothingCrossEntropy",
    "LabelSmoothingLoss",
    "FocalLoss",
    "SupConLoss",
    "create_criterion",
]
