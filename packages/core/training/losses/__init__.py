"""
Loss Functions — consolidated re-exports.

All loss functions are available through this package:
    from training.losses import FocalLoss, DistillationLoss, create_criterion
"""

# Classification losses
from .classification import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
    SupConLoss,
    create_criterion,
)

# Distillation loss
from .distillation import DistillationLoss

# Physics losses (stay in their canonical location)
from ..physics_loss_functions import (
    FrequencyConsistencyLoss,
    SommerfeldConsistencyLoss,
    TemporalSmoothnessLoss,
    PhysicalConstraintLoss,
    SpectralDistanceLoss,
)

# Contrastive losses (stay in their canonical location)
from ..contrastive.losses import (
    PhysicsInfoNCELoss,
    NTXentLoss,
    SimCLRLoss,
)

# Legacy alias
PhysicsInformedLoss = None  # removed; use PhysicalConstraintLoss directly

__all__ = [
    # Classification
    "FocalLoss",
    "LabelSmoothingCrossEntropy",
    "SupConLoss",
    "create_criterion",
    # Distillation
    "DistillationLoss",
    # Physics
    "FrequencyConsistencyLoss",
    "SommerfeldConsistencyLoss",
    "TemporalSmoothnessLoss",
    "PhysicalConstraintLoss",
    "SpectralDistanceLoss",
    # Contrastive
    "PhysicsInfoNCELoss",
    "NTXentLoss",
    "SimCLRLoss",
]
