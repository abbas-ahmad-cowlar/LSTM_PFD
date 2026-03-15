"""
Abstract Base Trainer with Template Method Pattern.

Purpose:
    Provides a common training interface for all trainer subclasses.
    Eliminates duplicated train/validate/fit loops across:
      - CNNTrainer
      - ProgressiveResizingTrainer
      - DistillationTrainer

    Subclasses override only the pieces that differ:
      _forward_pass, _compute_loss, _backward_pass, _optimizer_step

Author: Syed Abbas Ahmad
Date: 2025-11-20
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any
from pathlib import Path
import time

from utils.logging import get_logger

logger = get_logger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.

    Implements the Template Method pattern: the ``fit`` / ``train_epoch`` /
    ``validate_epoch`` skeleton is defined here; subclasses plug in their
    own ``_forward_pass`` and ``_compute_loss`` logic.

    Args:
        model: PyTorch model to train
        optimizer: Optimizer
        criterion: Loss function
        device: 'cuda' or 'cpu'
        lr_scheduler: Optional learning-rate scheduler
        max_grad_norm: Max gradient norm for clipping (0 = disabled)
        gradient_accumulation_steps: Effective batch-size multiplier
        mixed_precision: Use AMP FP16 training
        checkpoint_dir: Directory for saving checkpoints
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = "cpu",
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = False,
        checkpoint_dir: Optional[Path] = None,
    ):
        # Guard against requesting AMP without CUDA
        if mixed_precision and not torch.cuda.is_available():
            logger.warning("Mixed precision requested but CUDA not available. Disabling.")
            mixed_precision = False
            device = "cpu"

        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

    # ------------------------------------------------------------------
    # Template hooks — subclasses override these
    # ------------------------------------------------------------------

    @abstractmethod
    def _forward_pass(
        self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """
        Run the model forward and return raw outputs (logits).

        Args:
            inputs: Batch inputs already on device.
            targets: Batch targets already on device.

        Returns:
            Model outputs (logits).
        """

    def _compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Compute loss from model outputs and targets.

        Default implementation uses ``self.criterion``.
        Subclasses (e.g. distillation) override this.
        """
        return self.criterion(outputs, targets)

    # ------------------------------------------------------------------
    # Concrete skeleton methods
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        train_loader: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Run one training epoch and return metrics dict."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self._forward_pass(inputs, targets, **kwargs)
                    loss = self._compute_loss(outputs, targets, **kwargs)
            else:
                outputs = self._forward_pass(inputs, targets, **kwargs)
                loss = self._compute_loss(outputs, targets, **kwargs)

            # Scale for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            # Backward
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step (with grad accumulation)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.max_grad_norm > 0:
                    if self.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )

                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Metrics
            batch_size = targets.size(0)
            running_loss += loss.item() * self.gradient_accumulation_steps * batch_size
            _, predicted = outputs.max(1)
            total += batch_size
            correct += predicted.eq(targets).sum().item()

        return {
            "loss": running_loss / max(total, 1),
            "accuracy": 100.0 * correct / max(total, 1),
        }

    @torch.no_grad()
    def validate_epoch(
        self,
        val_loader: Optional[DataLoader],
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Run one validation epoch and return metrics dict."""
        if val_loader is None:
            return {}

        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self._forward_pass(inputs, targets, **kwargs)
                    loss = self._compute_loss(outputs, targets, **kwargs)
            else:
                outputs = self._forward_pass(inputs, targets, **kwargs)
                loss = self._compute_loss(outputs, targets, **kwargs)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += batch_size
            correct += predicted.eq(targets).sum().item()

        return {
            "loss": running_loss / max(total, 1),
            "accuracy": 100.0 * correct / max(total, 1),
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 50,
        save_best: bool = True,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            train_loader: Training dataloader
            val_loader: Optional validation dataloader
            num_epochs: Epochs to train
            save_best: Save checkpoint on best val accuracy
            verbose: Log per-epoch summaries

        Returns:
            Training history dictionary.
        """
        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate_epoch(val_loader)

            # Scheduler step
            if self.lr_scheduler is not None:
                if isinstance(
                    self.lr_scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    metric = val_metrics.get("loss", train_metrics["loss"])
                    self.lr_scheduler.step(metric)
                else:
                    self.lr_scheduler.step()

            # Record history
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["lr"].append(current_lr)
            if val_metrics:
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_acc"].append(val_metrics["accuracy"])

            # Log
            if verbose:
                elapsed = time.time() - start
                msg = (
                    f"Epoch {epoch + 1}/{num_epochs} ({elapsed:.1f}s) — "
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Train Acc: {train_metrics['accuracy']:.2f}%"
                )
                if val_metrics:
                    msg += (
                        f", Val Loss: {val_metrics['loss']:.4f}, "
                        f"Val Acc: {val_metrics['accuracy']:.2f}%"
                    )
                msg += f", LR: {current_lr:.6f}"
                logger.info(msg)

            # Save best
            if save_best and val_metrics and val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint("best_model.pth")
                logger.info(
                    f"✓ New best model saved (Val Acc: {self.best_val_acc:.2f}%)"
                )

        logger.info("Training complete!")
        return self.history

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint to ``checkpoint_dir / filename``."""
        if self.checkpoint_dir is None:
            logger.warning("No checkpoint_dir specified, skipping save")
            return

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.checkpoint_dir / filename

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

        if self.lr_scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: Path) -> None:
        """Load checkpoint from file."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = checkpoint.get("history", self.history)

        if self.lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        logger.info(f"Checkpoint loaded from {filepath}")

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
