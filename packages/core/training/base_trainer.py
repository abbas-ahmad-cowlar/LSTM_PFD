"""
Abstract Base Trainer with Template Method Pattern.

Purpose:
    Provides a common training interface for all trainer subclasses.
    Eliminates duplicated train/validate/fit loops across:
      - CNNTrainer
      - ProgressiveResizingTrainer
      - DistillationTrainer
      - PINNTrainer
      - SpectrogramTrainer

    Subclasses override only the pieces that differ:
      _forward_pass, _compute_loss, _on_epoch_end

Author: Syed Abbas Ahmad
Date: 2025-11-20
Updated: 2026-03-15 — Unified hierarchy (absorbed Trainer features: tqdm, callbacks)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, TYPE_CHECKING
from pathlib import Path
import time
import warnings

from tqdm import tqdm
from utils.logging import get_logger

if TYPE_CHECKING:
    from .callbacks import Callback

logger = get_logger(__name__)

# Checkpoint format version — bump when changing the schema
_CHECKPOINT_VERSION = 2


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
        callbacks: Optional list of Callback objects
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
        callbacks: Optional[List["Callback"]] = None,
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
        self.callbacks = callbacks or []

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

    def _on_epoch_end(
        self,
        epoch: int,
        num_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        """
        Hook called at end of each epoch, before scheduler step.

        Subclasses can override to update adaptive parameters
        (e.g. PINN physics lambda scheduling).
        """

    # ------------------------------------------------------------------
    # Callback helpers
    # ------------------------------------------------------------------

    def _fire_callbacks(self, hook_name: str, **kwargs: Any) -> None:
        """Call a named hook on all registered callbacks."""
        for cb in self.callbacks:
            method = getattr(cb, hook_name, None)
            if method is not None:
                method(self, **kwargs)

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

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch + 1} [Train]",
            leave=False,
        )

        for batch_idx, (inputs, targets) in enumerate(pbar):
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

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{running_loss / max(total, 1):.4f}",
                "acc": f"{100.0 * correct / max(total, 1):.2f}%",
            })

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

        pbar = tqdm(
            val_loader,
            desc=f"Epoch {self.current_epoch + 1} [Val]",
            leave=False,
        )

        for inputs, targets in pbar:
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

            pbar.set_postfix({
                "loss": f"{running_loss / max(total, 1):.4f}",
                "acc": f"{100.0 * correct / max(total, 1):.2f}%",
            })

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
        self._fire_callbacks("on_train_begin")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            start = time.time()

            self._fire_callbacks("on_epoch_begin", epoch=epoch)

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate_epoch(val_loader)

            # Subclass hook (e.g., PINN lambda scheduling)
            self._on_epoch_end(epoch, num_epochs, train_metrics, val_metrics)

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

            # Fire epoch-end callbacks
            epoch_metrics = {**train_metrics}
            if val_metrics:
                epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            self._fire_callbacks("on_epoch_end", epoch=epoch, metrics=epoch_metrics)

            # Save best
            if save_best and val_metrics and val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint("best_model.pth")
                logger.info(
                    f"✓ New best model saved (Val Acc: {self.best_val_acc:.2f}%)"
                )

        self._fire_callbacks("on_train_end")
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

        checkpoint: Dict[str, Any] = {
            "version": _CHECKPOINT_VERSION,
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "history": self.history,
        }

        # Optional: model config
        if hasattr(self.model, "get_config"):
            checkpoint["model_config"] = self.model.get_config()

        # Scheduler state
        if self.lr_scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()

        # AMP scaler state
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: Path) -> None:
        """Load checkpoint from file."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = checkpoint.get("history", self.history)

        if self.lr_scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        ckpt_version = checkpoint.get("version", 1)
        logger.info(
            f"Checkpoint loaded from {filepath} (format v{ckpt_version})"
        )

    def get_current_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
