"""Training infrastructure for deep learning models."""

# Core trainers
from .base_trainer import BaseTrainer
from .trainer import Trainer, TrainingState
from .cnn_trainer import CNNTrainer
from .pinn_trainer import PINNTrainer
from .spectrogram_trainer import SpectrogramTrainer, MultiTFRTrainer
from .progressive_resizing import ProgressiveResizingTrainer, ResizableSignalDataset
from .knowledge_distillation import DistillationTrainer
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

# Losses (re-export from consolidated package)
from .losses.classification import (
    FocalLoss,
    LabelSmoothingCrossEntropy,
    SupConLoss,
    create_criterion,
)
from .losses.distillation import DistillationLoss

# Schedulers (re-export from unified module)
from .schedulers import create_scheduler, WarmupScheduler, PolynomialLRScheduler

# Legacy aliases
LabelSmoothingLoss = LabelSmoothingCrossEntropy
EarlyStoppingCallback = EarlyStopping
ModelCheckpointCallback = ModelCheckpoint
CallbackManager = None  # removed

__all__ = [
    # Base
    "BaseTrainer",
    # Core trainers
    "Trainer",
    "TrainingState",
    "CNNTrainer",
    "PINNTrainer",
    "SpectrogramTrainer",
    "MultiTFRTrainer",
    "ProgressiveResizingTrainer",
    "DistillationTrainer",
    "MixedPrecisionTrainer",
    # Mixins
    "SpecAugmentMixin",
    "PhysicsLossMixin",
    # Support classes
    "ResizableSignalDataset",
    "DistillationLoss",
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
    "LabelSmoothingLoss",
    "SupConLoss",
    "create_criterion",
    # Schedulers
    "create_scheduler",
    "WarmupScheduler",
    "PolynomialLRScheduler",
    # Legacy aliases
    "EarlyStoppingCallback",
    "ModelCheckpointCallback",
]
