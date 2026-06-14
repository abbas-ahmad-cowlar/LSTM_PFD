"""
PINN Trainer

Extends BaseTrainer to support Physics-Informed Neural Networks.
Handles combined loss computation (classification + physics constraints).

Key Features:
- Physics loss integration (frequency consistency, Sommerfeld consistency, etc.)
- Adaptive physics loss weighting (gradual increase during training)
- Metadata handling for operating conditions
- Detailed logging of physics loss components

Usage:
    trainer = PINNTrainer(
        model=hybrid_pinn,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device='cuda',
        lambda_physics=0.5
    )
    history = trainer.fit(train_loader, val_loader, num_epochs=50)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Any, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm

from utils.logging import get_logger
from utils.constants import SAMPLING_RATE
from .base_trainer import BaseTrainer
from .physics_loss_functions import (
    FrequencyConsistencyLoss,
    SommerfeldConsistencyLoss,
    TemporalSmoothnessLoss,
    PhysicalConstraintLoss,
)

logger = get_logger(__name__)


class PINNTrainer(BaseTrainer):
    """
    Trainer for Physics-Informed Neural Networks.

    ⚠ PHYSICS PATH INERT — QUARANTINED (P6 remediation Step 3, 2026-06-14). This
    trainer's physics term is `PhysicalConstraintLoss`, which is non-differentiable
    (argmax) → zero gradient (external audit Finding 5). Training through it does
    NOT learn physics. It produced no committed result (the benchmark used pure CE;
    Phase-5 used the model-method loss in PhysicsConstrainedCNN). Do not use the
    physics mode until the ratified band-energy loss (Step 4) replaces
    `PhysicalConstraintLoss`. The CE/classification training path is unaffected.

    Extends BaseTrainer to support physics-based loss constraints.
    Compatible with both HybridPINN and PhysicsConstrainedCNN models.

    Args:
        model: PINN model
        optimizer: Optimizer
        criterion: Classification loss (default: CrossEntropyLoss)
        device: Device to train on
        lr_scheduler: Optional LR scheduler
        max_grad_norm: Gradient clipping norm
        gradient_accumulation_steps: Gradient accumulation steps
        mixed_precision: Use mixed precision training
        checkpoint_dir: Directory for saving checkpoints
        callbacks: Optional list of callback objects
        lambda_physics: Overall physics loss weight
        lambda_freq: Frequency consistency loss weight
        lambda_sommerfeld: Sommerfeld consistency loss weight
        lambda_temporal: Temporal smoothness loss weight
        adaptive_lambda: Gradually increase physics loss weight
        lambda_schedule: Schedule type ('linear', 'exponential', 'step')
        sample_rate: Signal sampling rate for physics loss
        metadata_keys: Keys to extract from batch dict (e.g., ['rpm', 'load'])
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        device: str = "cpu",
        lr_scheduler=None,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = False,
        checkpoint_dir: Optional[Path] = None,
        callbacks: Optional[List] = None,
        # PINN-specific parameters
        lambda_physics: float = 0.5,
        lambda_freq: float = 1.0,
        lambda_sommerfeld: float = 0.0,
        lambda_temporal: float = 0.0,
        adaptive_lambda: bool = True,
        lambda_schedule: str = "linear",
        sample_rate: int = SAMPLING_RATE,
        metadata_keys: Optional[List[str]] = None,
    ):
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion or nn.CrossEntropyLoss(),
            device=device,
            lr_scheduler=lr_scheduler,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            checkpoint_dir=checkpoint_dir,
            callbacks=callbacks,
        )

        # PINN-specific attributes
        self.lambda_physics = lambda_physics
        self.lambda_freq = lambda_freq
        self.lambda_sommerfeld = lambda_sommerfeld
        self.lambda_temporal = lambda_temporal
        self.adaptive_lambda = adaptive_lambda
        self.lambda_schedule = lambda_schedule
        self.sample_rate = sample_rate
        self.metadata_keys = metadata_keys or ["rpm", "load", "viscosity"]

        # Current lambda (for adaptive scheduling)
        self.current_lambda_physics = 0.0 if adaptive_lambda else lambda_physics

        # Physics loss function
        self.physics_loss_fn = PhysicalConstraintLoss(
            lambda_freq=lambda_freq,
            lambda_sommerfeld=lambda_sommerfeld,
            lambda_temporal=lambda_temporal,
            sample_rate=sample_rate,
        )

        # Extend history with PINN-specific metrics
        self.history.update(
            {
                "train_physics_loss": [],
                "val_physics_loss": [],
                "train_total_loss": [],
                "val_total_loss": [],
            }
        )

        logger.info(
            f"PINNTrainer initialized — "
            f"λ_physics={lambda_physics}, adaptive={adaptive_lambda}, "
            f"schedule={lambda_schedule}"
        )

    # -- Epoch hooks -----------------------------------------------------

    def _on_epoch_end(
        self,
        epoch: int,
        num_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ) -> None:
        """Update adaptive physics loss weight."""
        self._update_lambda_physics(epoch, num_epochs)

    def _update_lambda_physics(self, epoch: int, max_epochs: int) -> None:
        """Update physics loss weight based on training progress."""
        if not self.adaptive_lambda:
            self.current_lambda_physics = self.lambda_physics
            return

        progress = epoch / max(max_epochs, 1)

        if self.lambda_schedule == "linear":
            self.current_lambda_physics = self.lambda_physics * progress
        elif self.lambda_schedule == "exponential":
            self.current_lambda_physics = self.lambda_physics * (
                1 - np.exp(-5 * progress)
            )
        elif self.lambda_schedule == "step":
            if progress < 0.4:
                self.current_lambda_physics = 0.0
            elif progress < 0.7:
                self.current_lambda_physics = self.lambda_physics * 0.5
            else:
                self.current_lambda_physics = self.lambda_physics
        else:
            self.current_lambda_physics = self.lambda_physics

    # -- Metadata extraction ---------------------------------------------

    def _extract_metadata(
        self, batch: tuple
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Extract inputs, targets, and metadata from batch.

        Supports two batch formats:
        1. (inputs, targets) — standard format
        2. (inputs, targets, metadata_dict) — PINN format
        """
        if len(batch) == 2:
            inputs, targets = batch
            metadata = None
        elif len(batch) == 3:
            inputs, targets, metadata = batch
            if metadata is not None and isinstance(metadata, dict):
                metadata = {
                    k: v
                    for k, v in metadata.items()
                    if k in self.metadata_keys and v is not None
                }
                if not metadata:
                    metadata = None
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        return inputs, targets, metadata

    # -- Physics loss computation ----------------------------------------

    def compute_physics_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        signal: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined classification + physics loss.

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of loss components
        """
        ce_loss = self.criterion(outputs, targets)

        if self.current_lambda_physics > 0:
            physics_loss, physics_dict = self.physics_loss_fn(
                signal=signal,
                predictions=outputs,
                metadata=metadata,
                severity_predictions=None,
                predictions_sequence=None,
            )
            total_loss = ce_loss + self.current_lambda_physics * physics_loss
            loss_dict = {
                "ce_loss": ce_loss.item(),
                "physics_loss": physics_loss.item(),
                **physics_dict,
                "total_loss": total_loss.item(),
            }
        else:
            total_loss = ce_loss
            loss_dict = {
                "ce_loss": ce_loss.item(),
                "physics_loss": 0.0,
                "total_loss": total_loss.item(),
            }

        return total_loss, loss_dict

    # -- Template hooks --------------------------------------------------

    def _forward_pass(
        self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs: Any
    ) -> torch.Tensor:
        """Forward pass — routes metadata to model if available."""
        metadata = kwargs.get("metadata")
        if metadata:
            return self.model(inputs, metadata)
        return self.model(inputs)

    # -- Override train_epoch for metadata-aware batching -----------------

    def train_epoch(
        self,
        train_loader: DataLoader,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Training epoch with physics loss and metadata extraction."""
        self.model.train()

        running_loss = 0.0
        running_ce = 0.0
        running_physics = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {self.current_epoch + 1} [Train] λ={self.current_lambda_physics:.3f}",
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            inputs, targets, metadata = self._extract_metadata(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            if metadata is not None:
                metadata = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in metadata.items()
                }

            # Forward + physics loss
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self._forward_pass(inputs, targets, metadata=metadata)
                    loss, loss_dict = self.compute_physics_loss(
                        outputs, targets, inputs, metadata
                    )
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self._forward_pass(inputs, targets, metadata=metadata)
                loss, loss_dict = self.compute_physics_loss(
                    outputs, targets, inputs, metadata
                )
                loss = loss / self.gradient_accumulation_steps

            # Backward
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Optimizer step
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
            running_loss += loss_dict["total_loss"] * batch_size
            running_ce += loss_dict["ce_loss"] * batch_size
            running_physics += loss_dict["physics_loss"] * batch_size

            _, predicted = outputs.max(1)
            total += batch_size
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(
                {
                    "loss": f"{loss_dict['total_loss']:.4f}",
                    "acc": f"{100.0 * correct / total:.2f}%",
                }
            )

        metrics = {
            "loss": running_ce / max(total, 1),
            "accuracy": 100.0 * correct / max(total, 1),
            "physics_loss": running_physics / max(total, 1),
            "total_loss": running_loss / max(total, 1),
        }

        # Record PINN-specific history
        self.history["train_physics_loss"].append(metrics["physics_loss"])
        self.history["train_total_loss"].append(metrics["total_loss"])

        return metrics

    @torch.no_grad()
    def validate_epoch(
        self,
        val_loader: Optional[DataLoader],
        **kwargs: Any,
    ) -> Dict[str, float]:
        """Validation epoch with physics loss."""
        if val_loader is None:
            return {}

        self.model.eval()

        running_loss = 0.0
        running_ce = 0.0
        running_physics = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            val_loader,
            desc=f"Epoch {self.current_epoch + 1} [Val]",
            leave=False,
        )

        for batch in pbar:
            inputs, targets, metadata = self._extract_metadata(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            if metadata is not None:
                metadata = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in metadata.items()
                }

            outputs = self._forward_pass(inputs, targets, metadata=metadata)
            _, loss_dict = self.compute_physics_loss(outputs, targets, inputs, metadata)

            batch_size = targets.size(0)
            running_loss += loss_dict["total_loss"] * batch_size
            running_ce += loss_dict["ce_loss"] * batch_size
            running_physics += loss_dict["physics_loss"] * batch_size

            _, predicted = outputs.max(1)
            total += batch_size
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix(
                {
                    "loss": f"{loss_dict['total_loss']:.4f}",
                    "acc": f"{100.0 * correct / total:.2f}%",
                }
            )

        metrics = {
            "loss": running_ce / max(total, 1),
            "accuracy": 100.0 * correct / max(total, 1),
            "physics_loss": running_physics / max(total, 1),
            "total_loss": running_loss / max(total, 1),
        }

        # Record PINN-specific history
        self.history["val_physics_loss"].append(metrics["physics_loss"])
        self.history["val_total_loss"].append(metrics["total_loss"])

        return metrics

    # Backward-compat alias
    def train(self, num_epochs: int, **kwargs) -> Dict[str, List[float]]:
        """Backward-compatible alias. Delegates to inherited ``fit()``."""
        train_loader = kwargs.pop("train_loader", None)
        val_loader = kwargs.pop("val_loader", None)
        if train_loader is None:
            raise ValueError(
                "PINNTrainer.train() requires train_loader keyword argument. "
                "Preferred API is trainer.fit(train_loader, val_loader, num_epochs)."
            )
        return self.fit(
            train_loader, val_loader, num_epochs=num_epochs, **kwargs
        )
