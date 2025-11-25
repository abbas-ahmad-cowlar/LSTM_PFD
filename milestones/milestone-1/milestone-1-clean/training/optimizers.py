"""
Optimizer Wrappers and Learning Rate Schedules

Provides utilities for:
- Creating optimizers from config
- Learning rate schedulers
- Adaptive learning rate strategies
"""

import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW, RMSprop
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
    CosineAnnealingWarmRestarts
)
from typing import Dict, Any, Optional
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH


def create_optimizer(
    model_params,
    optimizer_name: str = 'adam',
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Create optimizer from name and parameters.

    Args:
        model_params: Model parameters to optimize
        optimizer_name: Name of optimizer ('adam', 'sgd', 'adamw', 'rmsprop')
        lr: Learning rate
        weight_decay: Weight decay (L2 regularization)
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Optimizer instance

    Example:
        >>> optimizer = create_optimizer(
        ...     model.parameters(),
        ...     optimizer_name='adam',
        ...     lr=1e-3,
        ...     weight_decay=1e-4
        ... )
    """
    optimizer_name = optimizer_name.lower()

    if optimizer_name == 'adam':
        optimizer = Adam(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

    elif optimizer_name == 'adamw':
        optimizer = AdamW(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

    elif optimizer_name == 'sgd':
        momentum = kwargs.pop('momentum', 0.9)
        nesterov = kwargs.pop('nesterov', True)
        optimizer = SGD(
            model_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            **kwargs
        )

    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(
            model_params,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    return optimizer


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str = 'cosine',
    num_epochs: int = 100,
    **kwargs
):
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer instance
        scheduler_name: Name of scheduler
            ('step', 'cosine', 'plateau', 'onecycle', 'cosine_restarts')
        num_epochs: Total number of training epochs
        **kwargs: Scheduler-specific arguments

    Returns:
        Scheduler instance

    Example:
        >>> scheduler = create_scheduler(
        ...     optimizer,
        ...     scheduler_name='cosine',
        ...     num_epochs=100
        ... )
    """
    scheduler_name = scheduler_name.lower()

    if scheduler_name == 'step':
        step_size = kwargs.pop('step_size', num_epochs // 3)
        gamma = kwargs.pop('gamma', 0.1)
        scheduler = StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
            **kwargs
        )

    elif scheduler_name == 'cosine':
        T_max = kwargs.pop('T_max', num_epochs)
        eta_min = kwargs.pop('eta_min', 1e-6)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
            **kwargs
        )

    elif scheduler_name == 'plateau':
        mode = kwargs.pop('mode', 'min')
        factor = kwargs.pop('factor', 0.1)
        patience = kwargs.pop('patience', 10)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            **kwargs
        )

    elif scheduler_name == 'onecycle':
        max_lr = kwargs.pop('max_lr', 1e-2)
        steps_per_epoch = kwargs.pop('steps_per_epoch', 100)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            **kwargs
        )

    elif scheduler_name == 'cosine_restarts':
        T_0 = kwargs.pop('T_0', num_epochs // 4)
        T_mult = kwargs.pop('T_mult', 2)
        eta_min = kwargs.pop('eta_min', 1e-6)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min,
            **kwargs
        )

    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer.

    Args:
        optimizer: Optimizer instance

    Returns:
        Current learning rate
    """
    return optimizer.param_groups[0]['lr']


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    """
    Set learning rate for optimizer.

    Args:
        optimizer: Optimizer instance
        lr: New learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
