"""
Training configuration for model optimization and training loop.

Purpose:
    Configuration for all training aspects:
    - Optimizer settings (Adam, SGD, AdamW)
    - Learning rate schedules
    - Training loop parameters
    - Regularization and callbacks
    - Mixed precision training

Author: LSTM_PFD Team
Date: 2025-11-19
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from config.base_config import BaseConfig


@dataclass
class OptimizerConfig(BaseConfig):
    """
    Optimizer configuration.

    Supports: Adam, AdamW, SGD, RMSprop

    Example:
        >>> config = OptimizerConfig(
        ...     name='adamw',
        ...     lr=0.001,
        ...     weight_decay=0.01
        ... )
    """
    # Optimizer type
    name: str = 'adamw'  # 'adam', 'adamw', 'sgd', 'rmsprop'

    # Learning rate
    lr: float = 0.001

    # Regularization
    weight_decay: float = 0.01

    # Adam/AdamW parameters
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False

    # SGD parameters
    momentum: float = 0.9
    nesterov: bool = True

    # RMSprop parameters
    alpha: float = 0.99

    # Gradient clipping
    clip_grad_norm: Optional[float] = 1.0
    clip_grad_value: Optional[float] = None

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": ["adam", "adamw", "sgd", "rmsprop"]},
                "lr": {"type": "number", "minimum": 0.0},
                "weight_decay": {"type": "number", "minimum": 0.0},
                "momentum": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            }
        }


@dataclass
class SchedulerConfig(BaseConfig):
    """
    Learning rate scheduler configuration.

    Supports:
    - StepLR: Decay by gamma every step_size epochs
    - CosineAnnealingLR: Cosine annealing
    - ReduceLROnPlateau: Reduce on metric plateau
    - OneCycleLR: 1cycle policy for super-convergence

    Example:
        >>> config = SchedulerConfig(
        ...     name='cosine',
        ...     T_max=100,
        ...     eta_min=1e-6
        ... )
    """
    # Scheduler type
    name: str = 'cosine'  # 'step', 'cosine', 'plateau', 'onecycle', 'none'

    # StepLR parameters
    step_size: int = 30
    gamma: float = 0.1

    # CosineAnnealingLR parameters
    T_max: int = 100
    eta_min: float = 1e-6

    # ReduceLROnPlateau parameters
    mode: str = 'min'
    factor: float = 0.1
    patience: int = 10
    threshold: float = 1e-4
    min_lr: float = 1e-6

    # OneCycleLR parameters
    max_lr: float = 0.01
    pct_start: float = 0.3
    anneal_strategy: str = 'cos'
    div_factor: float = 25.0
    final_div_factor: float = 1e4

    # Warmup
    warmup_epochs: int = 0
    warmup_start_lr: float = 1e-5

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string",
                        "enum": ["step", "cosine", "plateau", "onecycle", "none"]},
                "step_size": {"type": "integer", "minimum": 1},
                "gamma": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            }
        }


@dataclass
class CallbackConfig(BaseConfig):
    """
    Training callback configuration.

    Callbacks:
    - EarlyStopping: Stop training on metric plateau
    - ModelCheckpoint: Save best model
    - TensorBoard: Logging to TensorBoard
    - CSVLogger: Log metrics to CSV

    Example:
        >>> config = CallbackConfig(
        ...     early_stopping_patience=20,
        ...     save_best_only=True
        ... )
    """
    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_metric: str = 'val_loss'
    early_stopping_mode: str = 'min'
    early_stopping_min_delta: float = 1e-4

    # Model checkpointing
    use_checkpoint: bool = True
    checkpoint_dir: str = 'checkpoints'
    save_best_only: bool = True
    checkpoint_metric: str = 'val_loss'
    checkpoint_mode: str = 'min'
    save_frequency: int = 1  # Save every N epochs

    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_dir: str = 'runs'
    log_histograms: bool = False

    # CSV logging
    use_csv_logger: bool = True
    csv_log_file: str = 'training_log.csv'

    # Learning rate monitoring
    log_lr: bool = True

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "early_stopping_patience": {"type": "integer", "minimum": 1},
                "save_best_only": {"type": "boolean"},
                "early_stopping_metric": {"type": "string"},
                "checkpoint_metric": {"type": "string"}
            }
        }


@dataclass
class MixedPrecisionConfig(BaseConfig):
    """
    Mixed precision training configuration (FP16/BF16).

    Uses automatic mixed precision (AMP) for faster training with lower memory.

    Example:
        >>> config = MixedPrecisionConfig(
        ...     enabled=True,
        ...     dtype='float16',
        ...     loss_scale='dynamic'
        ... )
    """
    # Enable mixed precision
    enabled: bool = False

    # Precision type
    dtype: str = 'float16'  # 'float16', 'bfloat16'

    # Loss scaling
    loss_scale: str = 'dynamic'  # 'dynamic', or fixed value
    init_scale: float = 2.0**16
    growth_interval: int = 2000

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "dtype": {"type": "string", "enum": ["float16", "bfloat16"]},
                "loss_scale": {"type": "string"}
            }
        }


@dataclass
class RegularizationConfig(BaseConfig):
    """
    Additional regularization techniques.

    Includes:
    - Label smoothing
    - Mixup/CutMix
    - Stochastic depth
    - Data augmentation

    Example:
        >>> config = RegularizationConfig(
        ...     label_smoothing=0.1,
        ...     mixup_alpha=0.2
        ... )
    """
    # Label smoothing
    label_smoothing: float = 0.0

    # Mixup
    use_mixup: bool = False
    mixup_alpha: float = 0.2

    # CutMix
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0

    # Stochastic depth (for ResNets)
    stochastic_depth_prob: float = 0.0

    # Dropout (model-level)
    dropout_prob: float = 0.3

    # Weight averaging
    use_ema: bool = False  # Exponential moving average
    ema_decay: float = 0.999

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "label_smoothing": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "mixup_alpha": {"type": "number", "minimum": 0.0},
                "dropout_prob": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            }
        }


@dataclass
class TrainingConfig(BaseConfig):
    """
    Master training configuration.

    Aggregates all training-related settings.

    Example:
        >>> config = TrainingConfig(
        ...     num_epochs=100,
        ...     batch_size=64,
        ...     optimizer=OptimizerConfig(name='adamw', lr=0.001),
        ...     scheduler=SchedulerConfig(name='cosine', T_max=100)
        ... )
        >>> config.to_yaml('config/training_config.yaml')
    """
    # Training loop
    num_epochs: int = 100
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True

    # Optimizer and scheduler
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Callbacks
    callbacks: CallbackConfig = field(default_factory=CallbackConfig)

    # Mixed precision
    mixed_precision: MixedPrecisionConfig = field(default_factory=MixedPrecisionConfig)

    # Regularization
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)

    # Loss function
    loss_function: str = 'cross_entropy'  # 'cross_entropy', 'focal', 'label_smoothing'

    # Metrics to track
    metrics: List[str] = field(default_factory=lambda: ['accuracy', 'f1', 'precision', 'recall'])

    # Device
    device: str = 'cuda'  # 'cuda', 'cpu', 'auto'
    multi_gpu: bool = False
    gpu_ids: Optional[List[int]] = None

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Logging frequency
    log_interval: int = 10  # Log every N batches
    val_interval: int = 1  # Validate every N epochs

    # Resume training
    resume_from_checkpoint: Optional[str] = None

    # Gradient accumulation
    accumulation_steps: int = 1

    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    dist_backend: str = 'nccl'

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "num_epochs": {"type": "integer", "minimum": 1},
                "batch_size": {"type": "integer", "minimum": 1},
                "num_workers": {"type": "integer", "minimum": 0},
                "loss_function": {"type": "string"},
                "device": {"type": "string", "enum": ["cuda", "cpu", "auto"]},
                "seed": {"type": "integer", "minimum": 0}
            }
        }

    def get_total_training_steps(self, dataset_size: int) -> int:
        """
        Calculate total training steps.

        Args:
            dataset_size: Number of training samples

        Returns:
            Total number of optimization steps
        """
        steps_per_epoch = dataset_size // (self.batch_size * self.accumulation_steps)
        total_steps = steps_per_epoch * self.num_epochs
        return total_steps
