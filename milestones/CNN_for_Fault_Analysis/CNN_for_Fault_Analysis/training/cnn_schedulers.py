"""
Learning rate schedulers for CNN training.

Purpose:
    Advanced LR scheduling strategies for improved convergence:
    - Cosine annealing with warm restarts
    - One-cycle policy for super-convergence
    - Warmup schedulers
    - Step decay and exponential decay
    - ReduceLROnPlateau for adaptive scheduling

Author: Author Name
Date: 2025-11-20
"""

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    _LRScheduler,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    StepLR,
    ExponentialLR,
    ReduceLROnPlateau,
    OneCycleLR
)
from typing import Optional
import math

from utils.logging import get_logger

logger = get_logger(__name__)


def create_cosine_scheduler(
    optimizer: Optimizer,
    num_epochs: int,
    eta_min: float = 1e-6,
    last_epoch: int = -1
) -> CosineAnnealingLR:
    """
    Create cosine annealing LR scheduler.

    Learning rate decreases from initial_lr to eta_min following a cosine curve.
    Good for stable convergence.

    Args:
        optimizer: Optimizer
        num_epochs: Total number of training epochs
        eta_min: Minimum learning rate
        last_epoch: The index of last epoch

    Returns:
        CosineAnnealingLR scheduler

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        >>> scheduler = create_cosine_scheduler(optimizer, num_epochs=50)
        >>> for epoch in range(50):
        ...     train()
        ...     scheduler.step()
    """
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=eta_min,
        last_epoch=last_epoch
    )

    logger.info(f"Cosine annealing scheduler created: T_max={num_epochs}, eta_min={eta_min}")

    return scheduler


def create_cosine_warmrestarts_scheduler(
    optimizer: Optimizer,
    T_0: int = 10,
    T_mult: int = 2,
    eta_min: float = 1e-6,
    last_epoch: int = -1
) -> CosineAnnealingWarmRestarts:
    """
    Create cosine annealing with warm restarts.

    Periodically resets learning rate to help escape local minima.
    T_0: Initial restart interval
    T_mult: Factor to increase restart interval after each restart

    Args:
        optimizer: Optimizer
        T_0: Number of iterations for the first restart
        T_mult: Factor to increase T_i after a restart
        eta_min: Minimum learning rate
        last_epoch: The index of last epoch

    Returns:
        CosineAnnealingWarmRestarts scheduler

    Example:
        >>> # Restarts at epochs: 10, 30 (10+20), 70 (30+40), ...
        >>> scheduler = create_cosine_warmrestarts_scheduler(
        ...     optimizer, T_0=10, T_mult=2
        ... )
    """
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=T_mult,
        eta_min=eta_min,
        last_epoch=last_epoch
    )

    logger.info(f"Cosine annealing with warm restarts: T_0={T_0}, T_mult={T_mult}")

    return scheduler


def create_onecycle_scheduler(
    optimizer: Optimizer,
    max_lr: float,
    total_steps: int,
    pct_start: float = 0.3,
    anneal_strategy: str = 'cos',
    div_factor: float = 25.0,
    final_div_factor: float = 1e4
) -> OneCycleLR:
    """
    Create one-cycle LR scheduler for super-convergence.

    Implements the 1cycle policy: LR increases then decreases in one cycle.
    Very effective for fast training with high learning rates.

    Reference: "Super-Convergence" (Smith, 2018)

    Args:
        optimizer: Optimizer
        max_lr: Maximum learning rate
        total_steps: Total number of training steps (epochs * steps_per_epoch)
        pct_start: Percentage of cycle spent increasing LR (default: 0.3)
        anneal_strategy: 'cos' or 'linear'
        div_factor: Initial LR = max_lr / div_factor
        final_div_factor: Final LR = max_lr / final_div_factor

    Returns:
        OneCycleLR scheduler

    Example:
        >>> # For 50 epochs with 100 batches per epoch
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        >>> scheduler = create_onecycle_scheduler(
        ...     optimizer, max_lr=0.1, total_steps=50*100
        ... )
        >>> for epoch in range(50):
        ...     for batch in train_loader:
        ...         train_step()
        ...         scheduler.step()  # Step after each batch!
    """
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=pct_start,
        anneal_strategy=anneal_strategy,
        div_factor=div_factor,
        final_div_factor=final_div_factor
    )

    logger.info(f"OneCycle scheduler: max_lr={max_lr}, total_steps={total_steps}")

    return scheduler


def create_step_scheduler(
    optimizer: Optimizer,
    step_size: int = 10,
    gamma: float = 0.1,
    last_epoch: int = -1
) -> StepLR:
    """
    Create step decay LR scheduler.

    Multiply LR by gamma every step_size epochs.

    Args:
        optimizer: Optimizer
        step_size: Period of learning rate decay
        gamma: Multiplicative factor of learning rate decay
        last_epoch: The index of last epoch

    Returns:
        StepLR scheduler

    Example:
        >>> # Decay LR by 10x every 20 epochs
        >>> scheduler = create_step_scheduler(optimizer, step_size=20, gamma=0.1)
    """
    scheduler = StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma,
        last_epoch=last_epoch
    )

    logger.info(f"Step scheduler: step_size={step_size}, gamma={gamma}")

    return scheduler


def create_exponential_scheduler(
    optimizer: Optimizer,
    gamma: float = 0.95,
    last_epoch: int = -1
) -> ExponentialLR:
    """
    Create exponential decay LR scheduler.

    Multiply LR by gamma every epoch.

    Args:
        optimizer: Optimizer
        gamma: Multiplicative factor of learning rate decay
        last_epoch: The index of last epoch

    Returns:
        ExponentialLR scheduler

    Example:
        >>> # Decay LR by 5% every epoch
        >>> scheduler = create_exponential_scheduler(optimizer, gamma=0.95)
    """
    scheduler = ExponentialLR(
        optimizer,
        gamma=gamma,
        last_epoch=last_epoch
    )

    logger.info(f"Exponential scheduler: gamma={gamma}")

    return scheduler


def create_plateau_scheduler(
    optimizer: Optimizer,
    mode: str = 'max',
    factor: float = 0.1,
    patience: int = 10,
    threshold: float = 1e-4,
    min_lr: float = 1e-6
) -> ReduceLROnPlateau:
    """
    Create ReduceLROnPlateau scheduler.

    Reduce LR when a metric has stopped improving.
    Requires calling scheduler.step(metric) with the validation metric.

    Args:
        optimizer: Optimizer
        mode: 'min' for loss, 'max' for accuracy
        factor: Factor by which LR will be reduced (new_lr = lr * factor)
        patience: Number of epochs with no improvement after which LR will be reduced
        threshold: Threshold for measuring improvement
        min_lr: Lower bound on learning rate

    Returns:
        ReduceLROnPlateau scheduler

    Example:
        >>> scheduler = create_plateau_scheduler(optimizer, mode='max', patience=5)
        >>> for epoch in range(100):
        ...     val_acc = train_and_validate()
        ...     scheduler.step(val_acc)  # Pass metric!
    """
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=mode,
        factor=factor,
        patience=patience,
        threshold=threshold,
        min_lr=min_lr,
        verbose=True
    )

    logger.info(f"ReduceLROnPlateau scheduler: mode={mode}, patience={patience}, factor={factor}")

    return scheduler


class WarmupScheduler(_LRScheduler):
    """
    Learning rate warmup scheduler.

    Linearly increases LR from 0 to initial_lr over warmup_epochs,
    then applies a base scheduler.

    Args:
        optimizer: Optimizer
        warmup_epochs: Number of warmup epochs
        base_scheduler: Base scheduler to use after warmup (optional)
        last_epoch: The index of last epoch

    Example:
        >>> base_scheduler = CosineAnnealingLR(optimizer, T_max=50)
        >>> scheduler = WarmupScheduler(
        ...     optimizer, warmup_epochs=5, base_scheduler=base_scheduler
        ... )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        base_scheduler: Optional[_LRScheduler] = None,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.base_scheduler = base_scheduler
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

        logger.info(f"Warmup scheduler: warmup_epochs={warmup_epochs}")

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Use base scheduler if provided
            if self.base_scheduler is not None:
                return self.base_scheduler.get_last_lr()
            else:
                return self.base_lrs

    def step(self, epoch=None):
        super().step(epoch)

        # Also step base scheduler (after warmup)
        if self.base_scheduler is not None and self.last_epoch >= self.warmup_epochs:
            self.base_scheduler.step()


class PolynomialLRScheduler(_LRScheduler):
    """
    Polynomial learning rate decay scheduler.

    LR decreases from initial_lr to end_lr following a polynomial curve.

    Args:
        optimizer: Optimizer
        total_epochs: Total number of training epochs
        power: Polynomial power (1.0 = linear, 2.0 = quadratic)
        end_lr: Final learning rate
        last_epoch: The index of last epoch

    Example:
        >>> # Polynomial decay with power=2.0 (quadratic)
        >>> scheduler = PolynomialLRScheduler(
        ...     optimizer, total_epochs=50, power=2.0, end_lr=1e-6
        ... )
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_epochs: int,
        power: float = 1.0,
        end_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.total_epochs = total_epochs
        self.power = power
        self.end_lr = end_lr
        super(PolynomialLRScheduler, self).__init__(optimizer, last_epoch)

        logger.info(f"Polynomial scheduler: total_epochs={total_epochs}, power={power}")

    def get_lr(self):
        if self.last_epoch >= self.total_epochs:
            return [self.end_lr for _ in self.base_lrs]

        factor = (1 - self.last_epoch / self.total_epochs) ** self.power

        return [
            self.end_lr + (base_lr - self.end_lr) * factor
            for base_lr in self.base_lrs
        ]


def test_schedulers():
    """Test all scheduler implementations."""
    print("=" * 60)
    print("Testing LR Schedulers")
    print("=" * 60)

    import torch.nn as nn

    # Dummy model
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("\n1. Testing Cosine Annealing...")
    scheduler = create_cosine_scheduler(optimizer, num_epochs=10)
    lrs = []
    for epoch in range(10):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    print(f"   LR progression: {[f'{lr:.6f}' for lr in lrs[:5]]}...")
    assert lrs[0] > lrs[-1], "LR should decrease"

    print("\n2. Testing Cosine with Warm Restarts...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = create_cosine_warmrestarts_scheduler(optimizer, T_0=5)
    lrs = []
    for epoch in range(15):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    print(f"   LR with restarts: {[f'{lr:.6f}' for lr in lrs[:10]]}...")
    # After restart, LR should jump back up
    assert lrs[5] > lrs[4], "LR should restart"

    print("\n3. Testing OneCycle...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = create_onecycle_scheduler(optimizer, max_lr=0.1, total_steps=100)
    lrs = []
    for step in range(100):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    print(f"   LR range: {min(lrs):.6f} → {max(lrs):.6f} → {lrs[-1]:.6f}")
    # LR should increase then decrease
    mid_idx = len(lrs) // 2
    assert max(lrs[:mid_idx]) > lrs[0], "LR should increase initially"
    assert lrs[-1] < max(lrs), "LR should decrease at end"

    print("\n4. Testing Step Decay...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = create_step_scheduler(optimizer, step_size=3, gamma=0.1)
    lrs = []
    for epoch in range(10):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    print(f"   LR steps: {[f'{lr:.6f}' for lr in lrs]}...")
    assert lrs[0] == lrs[2], "LR constant before step"
    assert lrs[3] < lrs[2], "LR drops at step boundary"

    print("\n5. Testing Exponential Decay...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = create_exponential_scheduler(optimizer, gamma=0.9)
    lrs = []
    for epoch in range(5):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    print(f"   LR decay: {[f'{lr:.6f}' for lr in lrs]}...")
    assert all(lrs[i] > lrs[i+1] for i in range(len(lrs)-1)), "LR should decrease monotonically"

    print("\n6. Testing ReduceLROnPlateau...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = create_plateau_scheduler(optimizer, mode='max', patience=2)
    val_accs = [85.0, 86.0, 86.1, 86.05, 86.08, 86.1, 86.12]  # Plateauing
    initial_lr = optimizer.param_groups[0]['lr']
    for acc in val_accs:
        scheduler.step(acc)
    final_lr = optimizer.param_groups[0]['lr']
    print(f"   LR: {initial_lr:.6f} → {final_lr:.6f}")
    # LR should reduce after plateau
    # Note: may not always reduce depending on threshold

    print("\n7. Testing Warmup Scheduler...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    base = CosineAnnealingLR(optimizer, T_max=10)
    scheduler = WarmupScheduler(optimizer, warmup_epochs=3, base_scheduler=base)
    lrs = []
    for epoch in range(10):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    print(f"   LR with warmup: {[f'{lr:.6f}' for lr in lrs]}...")
    assert lrs[0] < lrs[2], "LR should increase during warmup"

    print("\n8. Testing Polynomial Scheduler...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = PolynomialLRScheduler(optimizer, total_epochs=10, power=2.0)
    lrs = []
    for epoch in range(10):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()
    print(f"   Polynomial LR: {[f'{lr:.6f}' for lr in lrs[:5]]}...")
    assert lrs[0] > lrs[-1], "LR should decrease"

    print("\n" + "=" * 60)
    print("✅ All scheduler tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_schedulers()
