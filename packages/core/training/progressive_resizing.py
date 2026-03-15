"""
Progressive Resizing Trainer — trains on progressively longer signals.

Inherits from BaseTrainer for the single-epoch loop; adds the multi-stage
``train_progressive()`` method that adjusts signal length across stages.

Author: Phase 2 - CNN Implementation
Date: 2025-11-20
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
import numpy as np
import time

from utils.logging import get_logger
from .base_trainer import BaseTrainer

logger = get_logger(__name__)


class ResizableSignalDataset(Dataset):
    """
    Wrapper that truncates or pads signals to a target length.

    Used by ProgressiveResizingTrainer to progressively increase the
    signal length seen during training.

    Args:
        signals: Original signal array (N, L) or (N, C, L)
        labels: Label array (N,)
        target_length: Desired signal length
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray, target_length: int):
        self.signals = signals
        self.labels = labels
        self.target_length = target_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = self.signals[idx]

        # Ensure shape is (C, L)
        if signal.ndim == 1:
            signal = signal[np.newaxis, :]  # (1, L)

        original_length = signal.shape[-1]

        if original_length >= self.target_length:
            # Truncate
            signal_resized = signal[..., : self.target_length]
        else:
            # Pad with zeros
            pad_width = self.target_length - original_length
            if signal.ndim == 2:
                signal_resized = np.pad(signal, ((0, 0), (0, pad_width)), mode="constant")
            else:
                signal_resized = np.pad(signal, ((0, pad_width),), mode="constant")

        signal_tensor = torch.tensor(signal_resized, dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return signal_tensor, label_tensor


class ProgressiveResizingTrainer(BaseTrainer):
    """
    Trains on progressively longer signals (curriculum-style).

    Stages: start at ``start_length`` and double until reaching ``full_length``.
    Each stage runs ``epochs_per_stage`` epochs before the next.

    Args:
        model: PyTorch model to train
        optimizer: Optimizer
        criterion: Loss function
        device: 'cuda' or 'cpu'
        lr_scheduler: Optional learning-rate scheduler
        max_grad_norm: Gradient clipping max norm
        gradient_accumulation_steps: Gradient accumulation steps
        mixed_precision: Use AMP FP16
        checkpoint_dir: Where to save checkpoints
        start_length: Initial signal length (default 1024)
        full_length: Full signal length (default 102400)
        epochs_per_stage: Epochs per progressive stage (default 5)

    Examples:
        >>> trainer = ProgressiveResizingTrainer(model, optimizer, criterion,
        ...     device='cuda', start_length=1024, full_length=102400)
        >>> history = trainer.train_progressive(train_signals, train_labels,
        ...     val_signals, val_labels, batch_size=32)
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
        start_length: int = 1024,
        full_length: int = 102400,
        epochs_per_stage: int = 5,
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
        self.start_length = start_length
        self.full_length = full_length
        self.epochs_per_stage = epochs_per_stage

        logger.info(
            f"ProgressiveResizingTrainer initialized — "
            f"stages: {start_length} → {full_length}, "
            f"{epochs_per_stage} epochs/stage"
        )

    # -- Template hook --------------------------------------------------

    def _forward_pass(
        self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        return self.model(inputs)

    # -- Progressive training -------------------------------------------

    def train_progressive(
        self,
        train_signals: np.ndarray,
        train_labels: np.ndarray,
        val_signals: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        batch_size: int = 32,
    ) -> Dict[str, List[float]]:
        """
        Run the progressive resizing curriculum.

        Returns the accumulated training history across all stages.
        """
        # Build length schedule: powers of 2 from start_length to full_length
        lengths = []
        length = self.start_length
        while length < self.full_length:
            lengths.append(length)
            length *= 2
        lengths.append(self.full_length)

        logger.info(f"Progressive schedule: {lengths}")

        for stage_idx, current_length in enumerate(lengths):
            logger.info(
                f"\n{'='*60}\n"
                f"Stage {stage_idx + 1}/{len(lengths)} — Signal length: {current_length}\n"
                f"{'='*60}"
            )

            # Build dataloaders for this stage
            train_dataset = ResizableSignalDataset(
                train_signals, train_labels, current_length
            )
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
            )

            val_loader = None
            if val_signals is not None and val_labels is not None:
                val_dataset = ResizableSignalDataset(
                    val_signals, val_labels, current_length
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
                )

            # Use inherited fit() for this stage
            self.fit(
                train_loader,
                val_loader,
                num_epochs=self.epochs_per_stage,
                save_best=True,
                verbose=True,
            )

        logger.info("Progressive resizing training complete!")
        return self.history
