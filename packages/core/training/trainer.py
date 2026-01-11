"""
Main Training Loop Orchestrator

Handles the training and validation loops with support for:
- Gradient accumulation
- Gradient clipping
- Mixed precision training
- Callbacks (early stopping, checkpointing, logging)
- Multi-GPU training (DataParallel/DistributedDataParallel)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable
from pathlib import Path
import time
from tqdm import tqdm
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH


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
        self.best_metric = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }

    def update(self, metrics: Dict[str, float]):
        """Update history with new metrics."""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)


class Trainer:
    """
    Main trainer class for deep learning models.

    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to train on ('cuda' or 'cpu')
        callbacks: Optional list of callback functions
        gradient_accumulation_steps: Number of steps to accumulate gradients (default: 1)
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)
        mixed_precision: Whether to use mixed precision training (default: False)
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
        mixed_precision: bool = False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device
        self.callbacks = callbacks or []
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.mixed_precision = mixed_precision

        # Training state
        self.state = TrainingState()

        # Mixed precision scaler
        if mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

    def train_epoch(self) -> Dict[str, float]:
        """
        Run one training epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.state.epoch+1} [Train]",
            leave=False
        )

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.max_grad_norm
                        )

                    self.optimizer.step()

                self.optimizer.zero_grad()

            # Statistics
            total_loss += loss.item() * self.gradient_accumulation_steps * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / total,
                'acc': 100. * correct / total
            })

        metrics = {
            'train_loss': total_loss / total,
            'train_acc': 100. * correct / total
        }

        return metrics

    @torch.no_grad()
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
        correct = 0
        total = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {self.state.epoch+1} [Val]",
            leave=False
        )

        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            # Statistics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / total,
                'acc': 100. * correct / total
            })

        metrics = {
            'val_loss': total_loss / total,
            'val_acc': 100. * correct / total
        }

        return metrics

    def fit(self, num_epochs: int):
        """
        Main training loop.

        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            self.state.epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate_epoch()

            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}

            # Update state
            self.state.update(epoch_metrics)

            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_acc']:.2f}%")

            if val_metrics:
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}, "
                      f"Val Acc: {val_metrics['val_acc']:.2f}%")

            # Run callbacks
            for callback in self.callbacks:
                callback(self, epoch_metrics)

        print("\nTraining complete!")

    def get_history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return self.state.history
