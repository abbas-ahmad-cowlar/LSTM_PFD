"""
Unified Learning Rate Schedulers

Consolidates all LR scheduling strategies from CNN and Transformer trainers:
- Cosine annealing (with/without warm restarts)
- One-cycle policy for super-convergence
- Warmup schedulers (linear, cosine, Noam)
- Step decay and exponential decay
- ReduceLROnPlateau for adaptive scheduling
- Polynomial decay

Merged from: training/cnn_schedulers.py + training/transformer_schedulers.py

Usage:
    from training.schedulers import create_scheduler
    scheduler = create_scheduler('warmup_cosine', optimizer, warmup_epochs=10, total_epochs=100)
"""

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    _LRScheduler,
    LambdaLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
    ExponentialLR,
    ReduceLROnPlateau,
    OneCycleLR,
)
from typing import Optional

from utils.logging import get_logger

logger = get_logger(__name__)


# ======================================================================
# CNN Schedulers (from cnn_schedulers.py)
# ======================================================================


def create_cosine_scheduler(
    optimizer: Optimizer,
    num_epochs: int,
    eta_min: float = 1e-6,
    last_epoch: int = -1,
) -> CosineAnnealingLR:
    """Create cosine annealing LR scheduler."""
    scheduler = CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=eta_min, last_epoch=last_epoch
    )
    logger.info(f"Cosine annealing scheduler: T_max={num_epochs}, eta_min={eta_min}")
    return scheduler


def create_cosine_warmrestarts_scheduler(
    optimizer: Optimizer,
    T_0: int = 10,
    T_mult: int = 2,
    eta_min: float = 1e-6,
    last_epoch: int = -1,
) -> CosineAnnealingWarmRestarts:
    """Create cosine annealing with warm restarts."""
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch
    )
    logger.info(f"Cosine with warm restarts: T_0={T_0}, T_mult={T_mult}")
    return scheduler


def create_onecycle_scheduler(
    optimizer: Optimizer,
    max_lr: float,
    total_steps: int,
    pct_start: float = 0.3,
    anneal_strategy: str = "cos",
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
) -> OneCycleLR:
    """Create one-cycle LR scheduler for super-convergence."""
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        div_factor=div_factor,
        final_div_factor=final_div_factor,
    )
    logger.info(f"OneCycle scheduler: max_lr={max_lr}, total_steps={total_steps}")
    return scheduler


def create_step_scheduler(
    optimizer: Optimizer,
    step_size: int = 10,
    gamma: float = 0.1,
    last_epoch: int = -1,
) -> StepLR:
    """Create step decay LR scheduler."""
    scheduler = StepLR(
        optimizer, step_size=step_size, gamma=gamma, last_epoch=last_epoch
    )
    logger.info(f"Step scheduler: step_size={step_size}, gamma={gamma}")
    return scheduler


def create_exponential_scheduler(
    optimizer: Optimizer,
    gamma: float = 0.95,
    last_epoch: int = -1,
) -> ExponentialLR:
    """Create exponential decay LR scheduler."""
    scheduler = ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)
    logger.info(f"Exponential scheduler: gamma={gamma}")
    return scheduler


def create_plateau_scheduler(
    optimizer: Optimizer,
    mode: str = "max",
    factor: float = 0.1,
    patience: int = 10,
    threshold: float = 1e-4,
    min_lr: float = 1e-6,
) -> ReduceLROnPlateau:
    """Create ReduceLROnPlateau scheduler."""
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        threshold=threshold,
        min_lr=min_lr,
        verbose=True,
    )
    logger.info(f"ReduceLROnPlateau: mode={mode}, patience={patience}, factor={factor}")
    return scheduler


class WarmupScheduler(_LRScheduler):
    """
    Linear warmup then delegates to a base scheduler.

    Args:
        optimizer: Optimizer
        warmup_epochs: Number of warmup epochs
        base_scheduler: Scheduler to use after warmup
        last_epoch: Last epoch index
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        base_scheduler: Optional[_LRScheduler] = None,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        super().__init__(optimizer, last_epoch)
        logger.info(f"Warmup scheduler: warmup_epochs={warmup_epochs}")

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            if self.base_scheduler is not None:
                return self.base_scheduler.get_last_lr()
            return self.base_lrs

    def step(self, epoch=None):
        super().step(epoch)
        if self.base_scheduler is not None and self.last_epoch >= self.warmup_epochs:
            self.base_scheduler.step()


class PolynomialLRScheduler(_LRScheduler):
    """
    Polynomial learning rate decay.

    Args:
        optimizer: Optimizer
        total_epochs: Total training epochs
        power: Polynomial power (1.0 = linear)
        end_lr: Final learning rate
        last_epoch: Last epoch index
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_epochs: int,
        power: float = 1.0,
        end_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.total_epochs = total_epochs
        self.power = power
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch)
        logger.info(f"Polynomial scheduler: total_epochs={total_epochs}, power={power}")

    def get_lr(self):
        if self.last_epoch >= self.total_epochs:
            return [self.end_lr for _ in self.base_lrs]
        factor = (1 - self.last_epoch / self.total_epochs) ** self.power
        return [self.end_lr + (base_lr - self.end_lr) * factor for base_lr in self.base_lrs]


# ======================================================================
# Transformer Schedulers (from transformer_schedulers.py)
# ======================================================================


def create_warmup_cosine_schedule(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Create LR scheduler with linear warmup and cosine annealing."""

    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return (current_epoch + 1) / warmup_epochs
        progress = (current_epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def create_warmup_linear_schedule(
    optimizer: Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Create LR scheduler with linear warmup and linear decay."""

    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            return (current_epoch + 1) / warmup_epochs
        progress = (current_epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

    return LambdaLR(optimizer, lr_lambda)


def create_noam_schedule(
    optimizer: Optimizer,
    d_model: int,
    warmup_steps: int = 4000,
    scale: float = 1.0,
) -> LambdaLR:
    """
    Create Noam scheduler from "Attention Is All You Need".

    Note: This scheduler operates on steps, not epochs.
    Call scheduler.step() after every batch.
    """

    def lr_lambda(current_step: int) -> float:
        current_step = max(1, current_step)
        return scale * (d_model ** -0.5) * min(
            current_step ** -0.5, current_step * (warmup_steps ** -1.5)
        )

    return LambdaLR(optimizer, lr_lambda)


class WarmupCosineScheduler(_LRScheduler):
    """
    Warmup-Cosine scheduler as a stateful class.

    Args:
        optimizer: Optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        min_lr_ratio: Minimum LR as ratio of initial LR
        last_epoch: Last epoch index
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            decay_factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay
            return [base_lr * decay_factor for base_lr in self.base_lrs]


# ======================================================================
# Unified Factory
# ======================================================================


def create_scheduler(
    scheduler_type: str,
    optimizer: Optimizer,
    **kwargs,
) -> _LRScheduler:
    """
    Unified factory for all learning rate schedulers.

    Args:
        scheduler_type: One of:
            CNN schedulers: 'cosine', 'cosine_warmrestarts', 'onecycle',
                           'step', 'exponential', 'plateau', 'polynomial'
            Transformer schedulers: 'warmup_cosine', 'warmup_linear', 'noam'
            Custom: 'warmup' (requires base_scheduler)
        optimizer: PyTorch optimizer
        **kwargs: Scheduler-specific arguments

    Returns:
        Learning rate scheduler
    """
    factories = {
        # CNN schedulers
        "cosine": lambda: create_cosine_scheduler(
            optimizer,
            num_epochs=kwargs.get("num_epochs", 50),
            eta_min=kwargs.get("eta_min", 1e-6),
        ),
        "cosine_warmrestarts": lambda: create_cosine_warmrestarts_scheduler(
            optimizer,
            T_0=kwargs.get("T_0", 10),
            T_mult=kwargs.get("T_mult", 2),
            eta_min=kwargs.get("eta_min", 1e-6),
        ),
        "onecycle": lambda: create_onecycle_scheduler(
            optimizer,
            max_lr=kwargs["max_lr"],
            total_steps=kwargs["total_steps"],
            pct_start=kwargs.get("pct_start", 0.3),
        ),
        "step": lambda: create_step_scheduler(
            optimizer,
            step_size=kwargs.get("step_size", 10),
            gamma=kwargs.get("gamma", 0.1),
        ),
        "exponential": lambda: create_exponential_scheduler(
            optimizer,
            gamma=kwargs.get("gamma", 0.95),
        ),
        "plateau": lambda: create_plateau_scheduler(
            optimizer,
            mode=kwargs.get("mode", "max"),
            factor=kwargs.get("factor", 0.1),
            patience=kwargs.get("patience", 10),
        ),
        "polynomial": lambda: PolynomialLRScheduler(
            optimizer,
            total_epochs=kwargs.get("total_epochs", 50),
            power=kwargs.get("power", 1.0),
            end_lr=kwargs.get("end_lr", 1e-6),
        ),
        # Transformer schedulers
        "warmup_cosine": lambda: create_warmup_cosine_schedule(
            optimizer,
            warmup_epochs=kwargs.get("warmup_epochs", 10),
            total_epochs=kwargs.get("total_epochs", 100),
            min_lr_ratio=kwargs.get("min_lr_ratio", 0.0),
        ),
        "warmup_linear": lambda: create_warmup_linear_schedule(
            optimizer,
            warmup_epochs=kwargs.get("warmup_epochs", 5),
            total_epochs=kwargs.get("total_epochs", 100),
            min_lr_ratio=kwargs.get("min_lr_ratio", 0.0),
        ),
        "noam": lambda: create_noam_schedule(
            optimizer,
            d_model=kwargs.get("d_model", 256),
            warmup_steps=kwargs.get("warmup_steps", 4000),
            scale=kwargs.get("scale", 1.0),
        ),
        # Custom
        "warmup": lambda: WarmupScheduler(
            optimizer,
            warmup_epochs=kwargs.get("warmup_epochs", 5),
            base_scheduler=kwargs.get("base_scheduler"),
        ),
    }

    if scheduler_type not in factories:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Supported: {sorted(factories.keys())}"
        )

    return factories[scheduler_type]()


# Legacy alias
get_scheduler = create_scheduler
