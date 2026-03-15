"""
Main Training Loop Orchestrator (Backward-compat shim)

This module provides the legacy ``Trainer`` class and ``TrainingState``
for backward compatibility.  New code should use ``BaseTrainer`` directly.

The ``Trainer`` class wraps ``BaseTrainer`` while preserving the old API
that accepted ``train_loader`` and ``val_loader`` at init time.
"""

import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Any
from pathlib import Path

from .base_trainer import BaseTrainer


class TrainingState:
    """
    Tracks the state of training across epochs.

    Attributes:
        epoch: Current epoch number
        best_metric: Best validation metric achieved
        best_epoch: Epoch where best metric was achieved
        patience_counter: Counter for early stopping
        history: Dictionary storing training history
    """

    def __init__(self):
        self.epoch = 0
        self.best_metric = float("inf")
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def update(self, metrics: Dict[str, float]):
        """Update history with new metrics."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)


class Trainer(BaseTrainer):
    """
    Legacy trainer — backward-compat wrapper around BaseTrainer.

    New code should use BaseTrainer, CNNTrainer, PINNTrainer, etc. directly.
    This class preserves the old API where train_loader/val_loader are
    provided at ``__init__`` time and ``fit()`` takes only ``num_epochs``.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to train on
        callbacks: Optional list of callback functions
        gradient_accumulation_steps: Gradient accumulation steps
        max_grad_norm: Gradient clipping max norm
        mixed_precision: Use mixed precision training
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: str = "cuda",
        callbacks: Optional[List[Callable]] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        mixed_precision: bool = False,
    ):
        warnings.warn(
            "training.trainer.Trainer is deprecated. "
            "Use training.base_trainer.BaseTrainer (or a concrete subclass) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        super().__init__(
            model=model,
            optimizer=optimizer or torch.optim.Adam(model.parameters()),
            criterion=criterion or nn.CrossEntropyLoss(),
            device=device,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )

        # Store loaders for legacy .fit(num_epochs) API
        self._train_loader = train_loader
        self._val_loader = val_loader

        # Legacy state object (for callback compat)
        self.state = TrainingState()

        # Legacy callbacks (simple callables rather than Callback objects)
        self._legacy_callbacks = callbacks or []

    # -- Template hook ---------------------------------------------------

    def _forward_pass(
        self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        return self.model(inputs)

    # -- Legacy API shim -------------------------------------------------

    def fit(  # type: ignore[override]
        self,
        num_epochs: int = None,
        train_loader: DataLoader = None,
        val_loader: DataLoader = None,
        **kwargs,
    ) -> Dict[str, List[float]]:
        """
        Legacy fit() — accepts either positional num_epochs or
        the new BaseTrainer signature (train_loader, val_loader, num_epochs).
        """
        # Decide which API is being used
        if isinstance(num_epochs, DataLoader):
            # Called with new signature: fit(train_loader, val_loader, ...)
            _train_loader = num_epochs
            _val_loader = train_loader
            _num_epochs = val_loader if isinstance(val_loader, int) else kwargs.pop("num_epochs", 50)
        elif isinstance(num_epochs, int):
            # Called with old signature: fit(num_epochs)
            _train_loader = self._train_loader
            _val_loader = self._val_loader
            _num_epochs = num_epochs
        else:
            _train_loader = train_loader or self._train_loader
            _val_loader = val_loader or self._val_loader
            _num_epochs = kwargs.pop("num_epochs", 50)

        # Delegate to BaseTrainer.fit
        return super().fit(
            train_loader=_train_loader,
            val_loader=_val_loader,
            num_epochs=_num_epochs,
            **kwargs,
        )

    def get_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return self.history
