"""
Learning Rate Schedulers for Transformer Training

Implements specialized learning rate schedules that are critical for training transformers:
- Warmup-Cosine: Linear warmup followed by cosine annealing
- Noam Scheduler: Original Transformer scheduler from "Attention Is All You Need"
- Warmup-Linear: Linear warmup followed by linear decay

These schedulers help prevent training divergence which is common in transformers
without proper warmup.
"""

import math
import torch
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler
from typing import Optional


def create_warmup_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.0
) -> LambdaLR:
    """
    Create learning rate scheduler with linear warmup and cosine annealing.

    This is the recommended schedule for Transformer models. It prevents
    training divergence in early epochs and smoothly decays the learning rate.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for linear warmup
        total_epochs: Total number of training epochs
        min_lr_ratio: Minimum learning rate as ratio of initial LR (default: 0.0)

    Returns:
        LambdaLR scheduler

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = create_warmup_cosine_schedule(optimizer, warmup_epochs=10, total_epochs=100)
        >>> for epoch in range(100):
        ...     train_epoch(model, train_loader, optimizer)
        ...     scheduler.step()
    """
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            # Linear warmup
            return (current_epoch + 1) / warmup_epochs
        else:
            # Cosine annealing after warmup
            progress = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            # Interpolate between min_lr_ratio and 1.0
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def create_warmup_linear_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.0
) -> LambdaLR:
    """
    Create learning rate scheduler with linear warmup and linear decay.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for linear warmup
        total_epochs: Total number of training epochs
        min_lr_ratio: Minimum learning rate as ratio of initial LR (default: 0.0)

    Returns:
        LambdaLR scheduler

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = create_warmup_linear_schedule(optimizer, warmup_epochs=5, total_epochs=50)
    """
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < warmup_epochs:
            # Linear warmup
            return (current_epoch + 1) / warmup_epochs
        else:
            # Linear decay after warmup
            progress = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

    return LambdaLR(optimizer, lr_lambda)


def create_noam_schedule(
    optimizer: torch.optim.Optimizer,
    d_model: int,
    warmup_steps: int = 4000,
    scale: float = 1.0
) -> LambdaLR:
    """
    Create Noam learning rate scheduler from "Attention Is All You Need" paper.

    Formula: lr = scale * (d_model^-0.5) * min(step^-0.5, step * warmup_steps^-1.5)

    This scheduler increases the learning rate linearly for the first warmup_steps,
    then decreases it proportionally to the inverse square root of the step number.

    Args:
        optimizer: PyTorch optimizer
        d_model: Model dimension (embedding size)
        warmup_steps: Number of warmup steps (not epochs!)
        scale: Scaling factor for learning rate (default: 1.0)

    Returns:
        LambdaLR scheduler

    Note:
        This scheduler operates on steps (iterations), not epochs.
        You should call scheduler.step() after every batch, not after every epoch.

    Example:
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1.0)  # lr=1.0 as base
        >>> scheduler = create_noam_schedule(optimizer, d_model=256, warmup_steps=4000)
        >>> for epoch in range(num_epochs):
        ...     for batch in train_loader:
        ...         optimizer.zero_grad()
        ...         loss = train_step(batch)
        ...         loss.backward()
        ...         optimizer.step()
        ...         scheduler.step()  # Step after each batch!
    """
    def lr_lambda(current_step: int) -> float:
        current_step = max(1, current_step)  # Avoid division by zero
        return scale * (d_model ** -0.5) * min(
            current_step ** -0.5,
            current_step * (warmup_steps ** -1.5)
        )

    return LambdaLR(optimizer, lr_lambda)


class WarmupCosineScheduler(_LRScheduler):
    """
    Warmup-Cosine scheduler as a stateful scheduler class.

    This provides the same functionality as create_warmup_cosine_schedule()
    but as a proper scheduler class for compatibility with PyTorch training loops.

    Args:
        optimizer: PyTorch optimizer
        warmup_epochs: Number of epochs for linear warmup
        total_epochs: Total number of training epochs
        min_lr_ratio: Minimum learning rate as ratio of initial LR
        last_epoch: The index of last epoch (default: -1)

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=100)
        >>> for epoch in range(100):
        ...     train_epoch(model, train_loader, optimizer)
        ...     val_loss = validate(model, val_loader)
        ...     scheduler.step()
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing after warmup
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            decay_factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay
            return [base_lr * decay_factor for base_lr in self.base_lrs]


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'warmup_cosine',
    **kwargs
) -> _LRScheduler:
    """
    Factory function to create learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('warmup_cosine', 'warmup_linear', 'noam')
        **kwargs: Additional arguments for the scheduler

    Returns:
        Learning rate scheduler

    Example:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = get_scheduler(
        ...     optimizer,
        ...     scheduler_type='warmup_cosine',
        ...     warmup_epochs=10,
        ...     total_epochs=100
        ... )
    """
    if scheduler_type == 'warmup_cosine':
        warmup_epochs = kwargs.get('warmup_epochs', 10)
        total_epochs = kwargs.get('total_epochs', 100)
        min_lr_ratio = kwargs.get('min_lr_ratio', 0.0)
        return create_warmup_cosine_schedule(
            optimizer, warmup_epochs, total_epochs, min_lr_ratio
        )
    elif scheduler_type == 'warmup_linear':
        warmup_epochs = kwargs.get('warmup_epochs', 5)
        total_epochs = kwargs.get('total_epochs', 100)
        min_lr_ratio = kwargs.get('min_lr_ratio', 0.0)
        return create_warmup_linear_schedule(
            optimizer, warmup_epochs, total_epochs, min_lr_ratio
        )
    elif scheduler_type == 'noam':
        d_model = kwargs.get('d_model', 256)
        warmup_steps = kwargs.get('warmup_steps', 4000)
        scale = kwargs.get('scale', 1.0)
        return create_noam_schedule(optimizer, d_model, warmup_steps, scale)
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Supported types: 'warmup_cosine', 'warmup_linear', 'noam'"
        )


if __name__ == '__main__':
    # Example usage and testing
    import matplotlib.pyplot as plt
    import numpy as np

    # Create a dummy optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Test warmup-cosine schedule
    scheduler = create_warmup_cosine_schedule(
        optimizer,
        warmup_epochs=10,
        total_epochs=100,
        min_lr_ratio=0.1
    )

    # Simulate training and collect learning rates
    lrs = []
    for epoch in range(100):
        lrs.append(optimizer.param_groups[0]['lr'])
        scheduler.step()

    # Plot the schedule
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Warmup-Cosine Learning Rate Schedule', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=10, color='r', linestyle='--', label='End of Warmup')
    plt.legend()
    plt.tight_layout()
    plt.savefig('warmup_cosine_schedule.png', dpi=300)
    print("Saved warmup-cosine schedule visualization to 'warmup_cosine_schedule.png'")

    # Test Noam schedule
    optimizer2 = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler2 = create_noam_schedule(optimizer2, d_model=256, warmup_steps=4000)

    lrs2 = []
    for step in range(20000):
        lrs2.append(optimizer2.param_groups[0]['lr'])
        scheduler2.step()

    plt.figure(figsize=(10, 6))
    plt.plot(lrs2, linewidth=2)
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.title('Noam Learning Rate Schedule (d_model=256)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.axvline(x=4000, color='r', linestyle='--', label='End of Warmup (4000 steps)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('noam_schedule.png', dpi=300)
    print("Saved Noam schedule visualization to 'noam_schedule.png'")

    print("\nScheduler test complete!")
