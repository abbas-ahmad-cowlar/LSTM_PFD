"""
Mixed Precision Training Utilities

Provides support for FP16 training to:
- Reduce memory usage
- Speed up training
- Maintain numerical stability

Note: BaseTrainer already has full AMP support via ``mixed_precision=True``.
This module provides a convenience subclass and utility functions.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from typing import Optional, Any

from .base_trainer import BaseTrainer


class MixedPrecisionTrainer(BaseTrainer):
    """
    Convenience trainer that defaults to mixed precision.

    Equivalent to creating any BaseTrainer subclass with
    ``mixed_precision=True``.  Kept for backward compatibility.

    Args:
        Same as BaseTrainer, but ``mixed_precision`` defaults to True.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        device: str = "cuda",
        lr_scheduler=None,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        checkpoint_dir=None,
        callbacks=None,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion or nn.CrossEntropyLoss(),
            device=device,
            lr_scheduler=lr_scheduler,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=True,
            checkpoint_dir=checkpoint_dir,
            callbacks=callbacks,
        )

    def _forward_pass(
        self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        return self.model(inputs)


def enable_mixed_precision():
    """
    Enable mixed precision training globally.

    Sets appropriate backends for optimal FP16 performance.
    """
    # Enable TF32 for matrix multiplications (Ampere GPUs)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Enable TF32 for convolutions (Ampere GPUs)
    torch.backends.cudnn.allow_tf32 = True

    print("Mixed precision training enabled")


def check_mixed_precision_support() -> bool:
    """
    Check if mixed precision training is supported.

    Returns:
        True if supported, False otherwise
    """
    if not torch.cuda.is_available():
        print("CUDA not available - mixed precision not supported")
        return False

    device = torch.device("cuda")
    try:
        x = torch.randn(10, 10, device=device, dtype=torch.float16)
        y = torch.randn(10, 10, device=device, dtype=torch.float16)
        _ = torch.matmul(x, y)

        print("Mixed precision training supported")
        return True

    except Exception as e:
        print(f"Mixed precision not supported: {e}")
        return False
