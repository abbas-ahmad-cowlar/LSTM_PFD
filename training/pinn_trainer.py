"""
PINN Trainer

Extends the base Trainer class to support Physics-Informed Neural Networks.
Handles combined loss computation (classification + physics constraints).

Key Features:
- Physics loss integration (frequency consistency, Sommerfeld consistency, etc.)
- Adaptive physics loss weighting (gradual increase during training)
- Metadata handling for operating conditions
- Detailed logging of physics loss components

Usage:
    trainer = PINNTrainer(
        model=hybrid_pinn,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        lambda_physics=0.5
    )
    trainer.train(num_epochs=50)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Tuple
from pathlib import Path
import time
from tqdm import tqdm
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from training.trainer import Trainer, TrainingState
from training.physics_loss_functions import (
    FrequencyConsistencyLoss,
    SommerfeldConsistencyLoss,
    TemporalSmoothnessLoss,
    PhysicalConstraintLoss
)


class PINNTrainingState(TrainingState):
    """
    Extended training state for PINN models.

    Tracks physics loss components separately for analysis.
    """

    def __init__(self):
        super().__init__()

        # Add physics-specific metrics
        self.history.update({
            'train_physics_loss': [],
            'val_physics_loss': [],
            'train_freq_loss': [],
            'val_freq_loss': [],
            'train_total_loss': [],
            'val_total_loss': []
        })


class PINNTrainer(Trainer):
    """
    Trainer for Physics-Informed Neural Networks.

    Extends base Trainer to support physics-based loss constraints.
    Compatible with both HybridPINN and PhysicsConstrainedCNN models.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        device: str = 'cuda',
        callbacks: Optional[List[Callable]] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        mixed_precision: bool = False,
        # PINN-specific parameters
        lambda_physics: float = 0.5,
        lambda_freq: float = 1.0,
        lambda_sommerfeld: float = 0.0,
        lambda_temporal: float = 0.0,
        adaptive_lambda: bool = True,
        lambda_schedule: str = 'linear',
        sample_rate: int = 20480,
        metadata_keys: List[str] = None
    ):
        """
        Initialize PINN Trainer.

        Args:
            model: PINN model (HybridPINN or PhysicsConstrainedCNN)
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Classification loss (default: CrossEntropyLoss)
            device: Device to train on
            callbacks: List of callback functions
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Gradient clipping norm
            mixed_precision: Use mixed precision training
            lambda_physics: Overall physics loss weight
            lambda_freq: Frequency consistency loss weight
            lambda_sommerfeld: Sommerfeld consistency loss weight
            lambda_temporal: Temporal smoothness loss weight
            adaptive_lambda: Gradually increase physics loss weight
            lambda_schedule: Schedule type ('linear', 'exponential', 'step')
            sample_rate: Signal sampling rate for physics loss
            metadata_keys: Keys to extract from batch dict (e.g., ['rpm', 'load'])
        """
        # Initialize base trainer
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
            mixed_precision=mixed_precision
        )

        # PINN-specific attributes
        self.lambda_physics = lambda_physics
        self.lambda_freq = lambda_freq
        self.lambda_sommerfeld = lambda_sommerfeld
        self.lambda_temporal = lambda_temporal
        self.adaptive_lambda = adaptive_lambda
        self.lambda_schedule = lambda_schedule
        self.sample_rate = sample_rate
        self.metadata_keys = metadata_keys or ['rpm', 'load', 'viscosity']

        # Current lambda (for adaptive scheduling)
        self.current_lambda_physics = 0.0 if adaptive_lambda else lambda_physics

        # Physics loss function
        self.physics_loss_fn = PhysicalConstraintLoss(
            lambda_freq=lambda_freq,
            lambda_sommerfeld=lambda_sommerfeld,
            lambda_temporal=lambda_temporal,
            sample_rate=sample_rate
        )

        # Override state with PINN state
        self.state = PINNTrainingState()

    def _update_lambda_physics(self, epoch: int, max_epochs: int):
        """
        Update physics loss weight based on training progress.

        Args:
            epoch: Current epoch (0-indexed)
            max_epochs: Total epochs
        """
        if not self.adaptive_lambda:
            self.current_lambda_physics = self.lambda_physics
            return

        progress = epoch / max_epochs

        if self.lambda_schedule == 'linear':
            # Linear increase: 0 → lambda_physics
            self.current_lambda_physics = self.lambda_physics * progress

        elif self.lambda_schedule == 'exponential':
            # Exponential increase
            import numpy as np
            self.current_lambda_physics = self.lambda_physics * (1 - np.exp(-5 * progress))

        elif self.lambda_schedule == 'step':
            # Step increase
            if progress < 0.4:
                self.current_lambda_physics = 0.0
            elif progress < 0.7:
                self.current_lambda_physics = self.lambda_physics * 0.5
            else:
                self.current_lambda_physics = self.lambda_physics

        else:
            self.current_lambda_physics = self.lambda_physics

    def _extract_metadata(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]:
        """
        Extract inputs, targets, and metadata from batch.

        Supports two batch formats:
        1. (inputs, targets) - standard format
        2. (inputs, targets, metadata_dict) - PINN format

        Args:
            batch: Batch from dataloader

        Returns:
            inputs: Signal tensor
            targets: Label tensor
            metadata: Optional dict with operating conditions
        """
        if len(batch) == 2:
            inputs, targets = batch
            metadata = None
        elif len(batch) == 3:
            inputs, targets, metadata = batch
            # Filter metadata to only include needed keys
            if metadata is not None and isinstance(metadata, dict):
                metadata = {k: v for k, v in metadata.items() if k in self.metadata_keys}
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        return inputs, targets, metadata

    def compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        signal: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined classification + physics loss.

        Args:
            outputs: Model predictions [B, num_classes]
            targets: Ground truth labels [B]
            signal: Input signal [B, 1, T]
            metadata: Optional operating conditions

        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of loss components
        """
        # Classification loss
        ce_loss = self.criterion(outputs, targets)

        # Physics loss
        if self.current_lambda_physics > 0:
            physics_loss, physics_dict = self.physics_loss_fn(
                signal=signal,
                predictions=outputs,
                metadata=metadata,
                severity_predictions=None,  # Not implemented yet
                predictions_sequence=None  # Not implemented yet
            )

            # Combined loss
            total_loss = ce_loss + self.current_lambda_physics * physics_loss

            loss_dict = {
                'ce_loss': ce_loss.item(),
                'physics_loss': physics_loss.item(),
                **physics_dict,
                'total_loss': total_loss.item()
            }
        else:
            # No physics loss (early training or disabled)
            total_loss = ce_loss
            loss_dict = {
                'ce_loss': ce_loss.item(),
                'physics_loss': 0.0,
                'total_loss': total_loss.item()
            }

        return total_loss, loss_dict

    def train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch with physics loss.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_physics_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.state.epoch+1} [Train] λ={self.current_lambda_physics:.3f}",
            leave=False
        )

        for batch_idx, batch in enumerate(pbar):
            # Extract batch components
            inputs, targets, metadata = self._extract_metadata(batch)
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Move metadata to device
            if metadata is not None:
                metadata = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                           for k, v in metadata.items()}

            # Forward pass
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs, metadata) if metadata else self.model(inputs)
                    loss, loss_dict = self.compute_loss(outputs, targets, inputs, metadata)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(inputs, metadata) if metadata else self.model(inputs)
                loss, loss_dict = self.compute_loss(outputs, targets, inputs, metadata)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    if self.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )
                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Accumulate metrics
            total_loss += loss_dict['total_loss'] * inputs.size(0)
            total_ce_loss += loss_dict['ce_loss'] * inputs.size(0)
            total_physics_loss += loss_dict['physics_loss'] * inputs.size(0)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'acc': f"{100.*correct/total:.2f}%"
            })

        # Calculate epoch metrics
        metrics = {
            'train_loss': total_ce_loss / total,
            'train_physics_loss': total_physics_loss / total,
            'train_total_loss': total_loss / total,
            'train_acc': 100. * correct / total
        }

        return metrics

    def validate_epoch(self) -> Dict[str, float]:
        """
        Run one validation epoch.

        Returns:
            Dictionary of validation metrics
        """
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_loss = 0.0
        total_ce_loss = 0.0
        total_physics_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc=f"Epoch {self.state.epoch+1} [Val]",
                leave=False
            )

            for batch in pbar:
                inputs, targets, metadata = self._extract_metadata(batch)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                if metadata is not None:
                    metadata = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                               for k, v in metadata.items()}

                # Forward pass
                outputs = self.model(inputs, metadata) if metadata else self.model(inputs)
                loss, loss_dict = self.compute_loss(outputs, targets, inputs, metadata)

                # Accumulate metrics
                total_loss += loss_dict['total_loss'] * inputs.size(0)
                total_ce_loss += loss_dict['ce_loss'] * inputs.size(0)
                total_physics_loss += loss_dict['physics_loss'] * inputs.size(0)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({
                    'loss': f"{loss_dict['total_loss']:.4f}",
                    'acc': f"{100.*correct/total:.2f}%"
                })

        metrics = {
            'val_loss': total_ce_loss / total,
            'val_physics_loss': total_physics_loss / total,
            'val_total_loss': total_loss / total,
            'val_acc': 100. * correct / total
        }

        return metrics

    def train(self, num_epochs: int, start_epoch: int = 0) -> Dict[str, List[float]]:
        """
        Train the model for specified number of epochs.

        Args:
            num_epochs: Number of epochs to train
            start_epoch: Starting epoch number

        Returns:
            Training history
        """
        print(f"Starting PINN Training for {num_epochs} epochs")
        print(f"Lambda Physics: {self.lambda_physics} (adaptive: {self.adaptive_lambda})")
        print(f"Device: {self.device}")
        print("=" * 60)

        for epoch in range(start_epoch, num_epochs):
            self.state.epoch = epoch

            # Update physics loss weight
            self._update_lambda_physics(epoch, num_epochs)

            # Train epoch
            train_metrics = self.train_epoch()

            # Validate epoch
            val_metrics = self.validate_epoch()

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            self.state.update(metrics)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train - Loss: {train_metrics['train_loss']:.4f}, "
                  f"Phys: {train_metrics['train_physics_loss']:.4f}, "
                  f"Acc: {train_metrics['train_acc']:.2f}%")
            if val_metrics:
                print(f"  Val   - Loss: {val_metrics['val_loss']:.4f}, "
                      f"Phys: {val_metrics['val_physics_loss']:.4f}, "
                      f"Acc: {val_metrics['val_acc']:.2f}%")

            # Run callbacks
            for callback in self.callbacks:
                callback(self, metrics)

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

        return self.state.history


if __name__ == "__main__":
    # Test PINN trainer
    print("=" * 60)
    print("PINN Trainer - Validation")
    print("=" * 60)

    # This is just a structure test - actual training would require data
    from models.pinn.hybrid_pinn import HybridPINN

    # Create dummy model
    model = HybridPINN(num_classes=NUM_CLASSES, backbone='resnet18')

    # Create dummy data loader
    from torch.utils.data import TensorDataset

    dummy_signals = torch.randn(100, 1, SIGNAL_LENGTH)
    dummy_labels = torch.randint(0, 11, (100,))
    dummy_rpm = torch.tensor([3600.0] * 100)

    dataset = TensorDataset(dummy_signals, dummy_labels, dummy_rpm)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Create trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    trainer = PINNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=train_loader,
        optimizer=optimizer,
        device='cpu',
        lambda_physics=0.5,
        adaptive_lambda=True
    )

    print("\nTrainer Configuration:")
    print(f"  Lambda physics: {trainer.lambda_physics}")
    print(f"  Adaptive: {trainer.adaptive_lambda}")
    print(f"  Current lambda: {trainer.current_lambda_physics}")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
