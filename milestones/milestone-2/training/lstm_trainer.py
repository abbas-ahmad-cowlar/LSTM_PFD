"""
LSTM Trainer for Bearing Fault Diagnosis

Training loop implementation for LSTM models.

Author: Bearing Fault Diagnosis Team
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
import time
from pathlib import Path
import numpy as np

from utils.early_stopping import EarlyStopping
from utils.checkpoint_manager import CheckpointManager
from utils.device_manager import get_device
from .metrics import calculate_metrics


class LSTMTrainer:
    """
    Trainer class for LSTM models.

    Args:
        model: PyTorch LSTM model
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on ('cuda' or 'cpu')
        scheduler: Optional learning rate scheduler
        early_stopping_patience: Patience for early stopping (None to disable)
        checkpoint_dir: Directory to save checkpoints
        mixed_precision: Whether to use mixed precision training
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: Optional[str] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping_patience: Optional[int] = 10,
        checkpoint_dir: Optional[str] = None,
        mixed_precision: bool = False
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device if device is not None else get_device()
        self.scheduler = scheduler
        self.mixed_precision = mixed_precision

        # Move model to device
        self.model = self.model.to(self.device)

        # Early stopping
        self.early_stopping = None
        if early_stopping_patience is not None:
            self.early_stopping = EarlyStopping(
                patience=early_stopping_patience,
                verbose=True
            )

        # Checkpoint manager
        self.checkpoint_manager = None
        if checkpoint_dir is not None:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                max_checkpoints=3
            )

        # Mixed precision scaler
        self.scaler = None
        if mixed_precision and torch.cuda.is_available():
            self.scaler = torch.cuda.amp.GradScaler()

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training DataLoader

        Returns:
            metrics: Dictionary with training metrics
        """
        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (signals, labels) in enumerate(train_loader):
            # Move to device
            signals = signals.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision if enabled
            if self.mixed_precision and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(signals)
                    loss = self.criterion(outputs, labels)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward pass
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

            # Statistics
            running_loss += loss.item() * signals.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate metrics
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc
        }

        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Args:
            val_loader: Validation DataLoader

        Returns:
            metrics: Dictionary with validation metrics
        """
        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for signals, labels in val_loader:
                # Move to device
                signals = signals.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                if self.mixed_precision and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = self.model(signals)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(signals)
                    loss = self.criterion(outputs, labels)

                # Statistics
                running_loss += loss.item() * signals.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Store for detailed metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        epoch_loss = running_loss / total
        epoch_acc = 100.0 * correct / total

        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'predictions': np.array(all_predictions),
            'labels': np.array(all_labels)
        }

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        Full training loop.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of epochs to train
            verbose: Whether to print progress

        Returns:
            history: Training history
        """
        print(f"Training on device: {self.device}")
        print(f"Mixed precision: {self.mixed_precision}")
        print(f"Total epochs: {num_epochs}\n")

        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)

            # Print progress
            if verbose:
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch [{epoch}/{num_epochs}] ({epoch_time:.2f}s)")
                print(f"  Train Loss: {train_metrics['loss']:.4f}, "
                      f"Train Acc: {train_metrics['accuracy']:.2f}%")
                print(f"  Val Loss: {val_metrics['loss']:.4f}, "
                      f"Val Acc: {val_metrics['accuracy']:.2f}%")
                print(f"  LR: {current_lr:.6f}\n")

            # Learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Save checkpoint
            if self.checkpoint_manager is not None:
                is_best = (epoch == 1 or
                          val_metrics['accuracy'] > max(self.history['val_acc'][:-1]))

                self.checkpoint_manager.save_checkpoint(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics=val_metrics,
                    is_best=is_best
                )

            # Early stopping
            if self.early_stopping is not None:
                self.early_stopping(val_metrics['loss'])

                if self.early_stopping.early_stop:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        print("Training completed!")
        return self.history

    def save_model(self, save_path: str):
        """Save model weights."""
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to: {save_path}")

    def load_model(self, load_path: str):
        """Load model weights."""
        self.model.load_state_dict(torch.load(load_path, map_location=self.device))
        print(f"Model loaded from: {load_path}")
