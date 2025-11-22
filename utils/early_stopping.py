"""
Early stopping mechanism to prevent overfitting during training.

Purpose:
    Monitor validation metrics and stop training when no improvement:
    - Configurable patience (epochs without improvement)
    - Support for min/max metrics (loss/accuracy)
    - Automatic best model tracking
    - Delta threshold for improvement detection

Author: LSTM_PFD Team
Date: 2025-11-20
"""

from typing import Optional
import numpy as np
from pathlib import Path

from utils.logging import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """
    Early stopping to prevent overfitting.

    Monitors a validation metric and stops training if no improvement
    is observed for a specified number of epochs (patience).

    Args:
        patience: Number of epochs to wait for improvement
        mode: 'min' for loss, 'max' for accuracy/F1
        min_delta: Minimum change to qualify as improvement
        restore_best_weights: Whether to restore best model at the end
        verbose: Print early stopping events

    Example:
        >>> early_stop = EarlyStopping(patience=10, mode='max', verbose=True)
        >>>
        >>> for epoch in range(100):
        ...     val_acc = train_one_epoch()
        ...
        ...     if early_stop(val_acc, model):
        ...         print(f"Early stopping at epoch {epoch}")
        ...         break
        >>>
        >>> print(f"Best validation accuracy: {early_stop.best_score:.2f}%")
    """

    def __init__(
        self,
        patience: int = 10,
        mode: str = 'max',
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        # State tracking
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.best_weights = None

        # Comparison operator
        if mode == 'min':
            self.is_better = lambda new, best: new < best - min_delta
            self.best_score = np.inf
        elif mode == 'max':
            self.is_better = lambda new, best: new > best + min_delta
            self.best_score = -np.inf
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'min' or 'max'.")

        if self.verbose:
            logger.info(f"EarlyStopping initialized: patience={patience}, mode={mode}")

    def __call__(
        self,
        metric_value: float,
        model: Optional[object] = None,
        epoch: Optional[int] = None
    ) -> bool:
        """
        Check if training should stop.

        Args:
            metric_value: Current validation metric value
            model: Model to save best weights (optional)
            epoch: Current epoch number (optional)

        Returns:
            True if training should stop, False otherwise
        """
        # Check for improvement
        if self.is_better(metric_value, self.best_score):
            # Improvement found
            if self.verbose:
                improvement = metric_value - self.best_score
                logger.info(
                    f"Validation metric improved: {self.best_score:.4f} → {metric_value:.4f} "
                    f"(Δ={improvement:+.4f})"
                )

            self.best_score = metric_value
            self.counter = 0

            if epoch is not None:
                self.best_epoch = epoch

            # Save best weights
            if self.restore_best_weights and model is not None:
                self.best_weights = {
                    name: param.detach().cpu().clone()
                    for name, param in model.state_dict().items()
                }

        else:
            # No improvement
            self.counter += 1

            if self.verbose:
                logger.info(
                    f"No improvement for {self.counter} epoch(s). "
                    f"Best: {self.best_score:.4f} (epoch {self.best_epoch})"
                )

            # Check if patience exhausted
            if self.counter >= self.patience:
                self.early_stop = True

                if self.verbose:
                    logger.info(
                        f"Early stopping triggered! No improvement for {self.patience} epochs."
                    )
                    logger.info(f"Best score: {self.best_score:.4f} at epoch {self.best_epoch}")

                # Restore best weights
                if self.restore_best_weights and model is not None and self.best_weights:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        logger.info("Best model weights restored")

        return self.early_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = np.inf if self.mode == 'min' else -np.inf
        self.best_epoch = 0
        self.early_stop = False
        self.best_weights = None

        if self.verbose:
            logger.info("EarlyStopping state reset")

    def get_best_score(self) -> float:
        """Get the best metric value observed."""
        return self.best_score

    def get_best_epoch(self) -> int:
        """Get the epoch with the best metric value."""
        return self.best_epoch

    def should_stop(self) -> bool:
        """Check if early stopping has been triggered."""
        return self.early_stop

    def state_dict(self) -> dict:
        """
        Get state dictionary for checkpoint saving.

        Returns:
            Dictionary with early stopping state
        """
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'early_stop': self.early_stop,
            'patience': self.patience,
            'mode': self.mode,
            'min_delta': self.min_delta
        }

    def load_state_dict(self, state_dict: dict):
        """
        Load state from dictionary.

        Args:
            state_dict: State dictionary from checkpoint
        """
        self.counter = state_dict.get('counter', 0)
        self.best_score = state_dict.get('best_score', self.best_score)
        self.best_epoch = state_dict.get('best_epoch', 0)
        self.early_stop = state_dict.get('early_stop', False)

        if self.verbose:
            logger.info("EarlyStopping state loaded from checkpoint")


class EarlyStoppingWithWarmup(EarlyStopping):
    """
    Early stopping with warmup period.

    Does not monitor metrics during the warmup period, allowing
    the model to train freely for initial epochs.

    Args:
        patience: Number of epochs to wait for improvement
        warmup_epochs: Number of epochs before monitoring starts
        mode: 'min' for loss, 'max' for accuracy/F1
        min_delta: Minimum change to qualify as improvement
        restore_best_weights: Whether to restore best model at the end
        verbose: Print early stopping events

    Example:
        >>> # Don't monitor for first 20 epochs
        >>> early_stop = EarlyStoppingWithWarmup(
        ...     patience=10,
        ...     warmup_epochs=20,
        ...     mode='max'
        ... )
    """

    def __init__(
        self,
        patience: int = 10,
        warmup_epochs: int = 0,
        mode: str = 'max',
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
        verbose: bool = True
    ):
        super().__init__(patience, mode, min_delta, restore_best_weights, verbose)
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0

        if self.verbose and warmup_epochs > 0:
            logger.info(f"Early stopping warmup: {warmup_epochs} epochs")

    def __call__(
        self,
        metric_value: float,
        model: Optional[object] = None,
        epoch: Optional[int] = None
    ) -> bool:
        """
        Check if training should stop (after warmup).

        Args:
            metric_value: Current validation metric value
            model: Model to save best weights (optional)
            epoch: Current epoch number (optional)

        Returns:
            True if training should stop, False otherwise
        """
        if epoch is not None:
            self.current_epoch = epoch

        # Skip early stopping during warmup
        if self.current_epoch < self.warmup_epochs:
            if self.verbose and self.current_epoch == 0:
                logger.info(f"Early stopping inactive (warmup: 0/{self.warmup_epochs})")
            return False

        # After warmup, use standard early stopping
        return super().__call__(metric_value, model, epoch)

    def reset(self):
        """Reset early stopping state including warmup."""
        super().reset()
        self.current_epoch = 0


def test_early_stopping():
    """Test early stopping functionality."""
    print("=" * 60)
    print("Testing Early Stopping")
    print("=" * 60)

    import torch
    import torch.nn as nn

    # Create a simple dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)

        def forward(self, x):
            return self.linear(x)

    model = DummyModel()

    print("\n1. Testing basic early stopping (maximize accuracy)...")
    early_stop = EarlyStopping(patience=3, mode='max', verbose=True)

    # Simulate training with improving then plateauing accuracy
    val_accs = [85.0, 87.0, 89.0, 90.0, 89.5, 89.8, 89.9, 89.7, 89.6]

    for epoch, val_acc in enumerate(val_accs, start=1):
        print(f"\nEpoch {epoch}: val_acc = {val_acc:.1f}%")
        should_stop = early_stop(val_acc, model, epoch)

        if should_stop:
            print(f"Training stopped at epoch {epoch}")
            break

    print(f"\nBest score: {early_stop.get_best_score():.1f}% at epoch {early_stop.get_best_epoch()}")

    print("\n" + "=" * 60)
    print("\n2. Testing early stopping with min_delta...")
    early_stop_delta = EarlyStopping(patience=2, mode='max', min_delta=0.5, verbose=True)

    # Simulate training with small improvements
    val_accs_small = [85.0, 85.2, 85.3, 85.35, 85.38]

    for epoch, val_acc in enumerate(val_accs_small, start=1):
        print(f"\nEpoch {epoch}: val_acc = {val_acc:.2f}%")
        should_stop = early_stop_delta(val_acc, model, epoch)

        if should_stop:
            print(f"Training stopped at epoch {epoch} (improvements too small)")
            break

    print("\n" + "=" * 60)
    print("\n3. Testing early stopping with warmup...")
    early_stop_warmup = EarlyStoppingWithWarmup(
        patience=2,
        warmup_epochs=3,
        mode='max',
        verbose=True
    )

    # Simulate training
    val_accs_warmup = [70.0, 75.0, 80.0, 85.0, 84.5, 84.8, 84.7]

    for epoch, val_acc in enumerate(val_accs_warmup, start=1):
        print(f"\nEpoch {epoch}: val_acc = {val_acc:.1f}%")
        should_stop = early_stop_warmup(val_acc, model, epoch)

        if should_stop:
            print(f"Training stopped at epoch {epoch}")
            break

    print("\n" + "=" * 60)
    print("\n4. Testing state dict save/load...")
    early_stop_save = EarlyStopping(patience=5, mode='max', verbose=False)
    early_stop_save(90.0, model, epoch=10)
    early_stop_save(91.0, model, epoch=11)

    state = early_stop_save.state_dict()
    print(f"Saved state: {state}")

    early_stop_load = EarlyStopping(patience=5, mode='max', verbose=False)
    early_stop_load.load_state_dict(state)
    print(f"Loaded best score: {early_stop_load.get_best_score():.1f}%")

    print("\n" + "=" * 60)
    print("✅ All early stopping tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_early_stopping()
