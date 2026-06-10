"""
Loss Functions — consolidated re-exports.

    from packages.core.training.losses import FocalLoss, create_criterion
"""

# Classification losses
from .classification import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
    SupConLoss,
    create_criterion,
)

# Physics losses (stay in their canonical location)
from ..physics_loss_functions import (
    FrequencyConsistencyLoss,
    SommerfeldConsistencyLoss,
    TemporalSmoothnessLoss,
    PhysicalConstraintLoss,
    SpectralDistanceLoss,
)

__all__ = [
    # Classification
    "FocalLoss",
    "LabelSmoothingCrossEntropy",
    "SupConLoss",
    "create_criterion",
    # Physics
    "FrequencyConsistencyLoss",
    "SommerfeldConsistencyLoss",
    "TemporalSmoothnessLoss",
    "PhysicalConstraintLoss",
    "SpectralDistanceLoss",
]
