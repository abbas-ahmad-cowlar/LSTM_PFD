"""Training infrastructure for deep learning models."""

from .base_trainer import BaseTrainer
from .trainer import Trainer, TrainingState
from .cnn_trainer import CNNTrainer
from .pinn_trainer import PINNTrainer
from .spectrogram_trainer import SpectrogramTrainer
from .progressive_resizing import ProgressiveResizingTrainer, ResizableSignalDataset
from .knowledge_distillation import DistillationTrainer, DistillationLoss
from .callbacks import EarlyStopping, ModelCheckpoint
from .losses import FocalLoss, LabelSmoothingCrossEntropy

__all__ = [
    # Base
    'BaseTrainer',
    # Core Trainers
    'Trainer',
    'TrainingState',
    'CNNTrainer',
    'PINNTrainer',
    'SpectrogramTrainer',
    'ProgressiveResizingTrainer',
    'DistillationTrainer',
    # Support Classes
    'ResizableSignalDataset',
    'DistillationLoss',
    # Callbacks
    'EarlyStopping',
    'ModelCheckpoint',
    # Loss Functions
    'FocalLoss',
    'LabelSmoothingCrossEntropy',
]


