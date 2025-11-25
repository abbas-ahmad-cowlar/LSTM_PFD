"""
Mixed Precision Training Utilities

Provides support for FP16 training to:
- Reduce memory usage
- Speed up training
- Maintain numerical stability
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Optional
from .trainer import Trainer
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH


class MixedPrecisionTrainer(Trainer):
    """
    Trainer with mixed precision training support.

    Automatically handles:
    - Gradient scaling
    - Autocast contexts
    - Gradient clipping with FP16

    Args:
        Same as Trainer, with additional:
        grad_scaler: Optional GradScaler instance (created if None)
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader=None,
        optimizer=None,
        criterion=None,
        device='cuda',
        callbacks=None,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        grad_scaler: Optional[GradScaler] = None
    ):
        # Initialize parent with mixed_precision=True
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            callbacks=callbacks,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            mixed_precision=True
        )

        # Create or use provided scaler
        if grad_scaler is None:
            self.scaler = GradScaler()
        else:
            self.scaler = grad_scaler

    def _backward_pass(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer):
        """
        Backward pass with gradient scaling.

        Args:
            loss: Loss tensor
            optimizer: Optimizer instance
        """
        # Scale loss and backward
        self.scaler.scale(loss).backward()

    def _optimizer_step(self, optimizer: torch.optim.Optimizer):
        """
        Optimizer step with gradient unscaling and clipping.

        Args:
            optimizer: Optimizer instance
        """
        # Unscale gradients
        self.scaler.unscale_(optimizer)

        # Clip gradients
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm
            )

        # Optimizer step
        self.scaler.step(optimizer)

        # Update scaler
        self.scaler.update()

        # Zero gradients
        optimizer.zero_grad()


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

    # Check for FP16 support
    device = torch.device('cuda')
    try:
        # Test FP16 operations
        x = torch.randn(10, 10, device=device, dtype=torch.float16)
        y = torch.randn(10, 10, device=device, dtype=torch.float16)
        z = torch.matmul(x, y)

        print("Mixed precision training supported")
        return True

    except Exception as e:
        print(f"Mixed precision not supported: {e}")
        return False
