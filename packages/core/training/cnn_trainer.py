"""
CNN Trainer — specialized trainer for CNN models.

Inherits from BaseTrainer; overrides _forward_pass.
Preserves the original CNNTrainer API for backward compatibility.

Author: Phase 2 - CNN Implementation
Date: 2025-11-20
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Any
from pathlib import Path

from utils.logging import get_logger
from .base_trainer import BaseTrainer

logger = get_logger(__name__)


class CNNTrainer(BaseTrainer):
    """
    Trainer specialized for 1-D CNN bearing-fault classifiers.

    Adds CNN-specific defaults (higher grad-clip, sensible checkpoint naming).

    Args:
        model: PyTorch CNN model
        optimizer: Optimizer
        criterion: Loss function
        device: 'cuda' or 'cpu'
        lr_scheduler: Optional learning-rate scheduler
        max_grad_norm: Gradient clipping max norm (default 1.0)
        gradient_accumulation_steps: Gradient accumulation steps
        mixed_precision: Whether to use AMP FP16
        checkpoint_dir: Directory for saving checkpoints
        num_classes: Number of fault classes (for logging only)

    Examples:
        >>> model = CNN1D(num_classes=11)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> criterion = nn.CrossEntropyLoss()
        >>> trainer = CNNTrainer(model, optimizer, criterion, device='cuda')
        >>> history = trainer.fit(train_loader, val_loader, num_epochs=50)
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cpu",
        lr_scheduler=None,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = False,
        checkpoint_dir: Optional[Path] = None,
        num_classes: int = 11,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            lr_scheduler=lr_scheduler,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            checkpoint_dir=checkpoint_dir,
        )
        self.num_classes = num_classes
        logger.info(
            f"CNNTrainer initialized — "
            f"device={device}, mixed_precision={mixed_precision}, "
            f"num_classes={num_classes}"
        )

    # -- Template hook --------------------------------------------------

    def _forward_pass(
        self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Standard forward through the CNN model."""
        return self.model(inputs)
