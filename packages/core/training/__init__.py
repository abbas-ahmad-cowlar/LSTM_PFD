"""Training infrastructure for the curated model zoo.

Trainers: BaseTrainer (template method) → CNNTrainer, PINNTrainer,
MixedPrecisionTrainer; Trainer is the backward-compat wrapper.
(Spectrogram/distillation/progressive-resizing trainers were pruned in the
2026-06 convergence; recoverable from tag `pre-convergence-2026-06`.)
"""

# Core trainers
from .base_trainer import BaseTrainer
from .trainer import Trainer, TrainingState
from .cnn_trainer import CNNTrainer
from .pinn_trainer import PINNTrainer
from .mixed_precision import MixedPrecisionTrainer

# Mixins
from .mixins import SpecAugmentMixin, PhysicsLossMixin

# Callbacks
from .callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    TensorBoardLogger,
    MLflowLogger,
    ProgressPrinter,
)

# Losses
from .losses.classification import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
    SupConLoss,
    create_criterion,
)

# Schedulers
from .schedulers import create_scheduler, WarmupScheduler, PolynomialLRScheduler

__all__ = [
    # Base
    "BaseTrainer",
    # Core trainers
    "Trainer",
    "TrainingState",
    "CNNTrainer",
    "PINNTrainer",
    "MixedPrecisionTrainer",
    # Mixins
    "SpecAugmentMixin",
    "PhysicsLossMixin",
    # Callbacks
    "Callback",
    "EarlyStopping",
    "ModelCheckpoint",
    "LearningRateScheduler",
    "TensorBoardLogger",
    "MLflowLogger",
    "ProgressPrinter",
    # Loss Functions
    "FocalLoss",
    "LabelSmoothingCrossEntropy",
    "SupConLoss",
    "create_criterion",
    # Schedulers
    "create_scheduler",
    "WarmupScheduler",
    "PolynomialLRScheduler",
]
