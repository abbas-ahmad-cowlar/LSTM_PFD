"""
Progressive Resizing for Signal Training

Train models with progressively longer signals for faster convergence.

Strategy:
    Stage 1: Short signals (25600 samples) → Fast initial training
    Stage 2: Medium signals (51200 samples) → Refine features
    Stage 3: Full signals (102400 samples) → Final accuracy

Benefits:
- 2-3× faster initial convergence
- Better regularization (short signals act as augmentation)
- Improved final accuracy
- GPU memory efficient in early stages

Reference:
- Howard et al. (2018). "fastai: A Layered API for Deep Learning"
- Progressive resizing strategy from fast.ai
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple, Optional, Dict
import numpy as np


class ResizableSignalDataset(Dataset):
    """
    Dataset wrapper that resizes signals to specified length.

    Args:
        base_dataset: Original dataset with full-length signals
        target_length: Target signal length
        resize_method: 'interpolate' or 'crop' or 'subsample'
    """

    def __init__(
        self,
        base_dataset: Dataset,
        target_length: int,
        resize_method: str = 'interpolate'
    ):
        self.base_dataset = base_dataset
        self.target_length = target_length
        self.resize_method = resize_method

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        signal, label = self.base_dataset[idx]

        # Ensure signal is tensor
        if not isinstance(signal, torch.Tensor):
            signal = torch.tensor(signal, dtype=torch.float32)

        # Resize signal
        signal = self.resize_signal(signal, self.target_length)

        return signal, label

    def resize_signal(self, signal: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Resize signal to target length.

        Args:
            signal: Input signal [C, T] or [T]
            target_length: Target length

        Returns:
            Resized signal
        """
        # Handle 1D or 2D signals
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)  # [T] → [1, T]
            squeeze_output = True
        else:
            squeeze_output = False

        current_length = signal.shape[1]

        if current_length == target_length:
            result = signal
        elif self.resize_method == 'interpolate':
            # Interpolation (smooth resizing)
            signal = signal.unsqueeze(0)  # [C, T] → [1, C, T]
            result = torch.nn.functional.interpolate(
                signal,
                size=target_length,
                mode='linear',
                align_corners=False
            )
            result = result.squeeze(0)  # [1, C, T] → [C, T]
        elif self.resize_method == 'crop':
            # Center crop
            if current_length > target_length:
                start = (current_length - target_length) // 2
                result = signal[:, start:start + target_length]
            else:
                # Pad if too short
                pad = target_length - current_length
                result = torch.nn.functional.pad(signal, (0, pad))
        elif self.resize_method == 'subsample':
            # Subsample (decimation)
            if current_length > target_length:
                step = current_length // target_length
                result = signal[:, ::step][:, :target_length]
            else:
                # Repeat if too short
                repeats = (target_length // current_length) + 1
                result = signal.repeat(1, repeats)[:, :target_length]
        else:
            raise ValueError(f"Unknown resize method: {self.resize_method}")

        if squeeze_output:
            result = result.squeeze(0)

        return result


class ProgressiveResizingTrainer:
    """
    Trainer with progressive signal length schedule.

    Args:
        model: Model to train
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        schedule: List of (signal_length, epochs) tuples
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str = 'cpu',
        schedule: Optional[List[Tuple[int, int]]] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        # Default 3-stage schedule
        if schedule is None:
            self.schedule = [
                (25600, 30),   # Stage 1: Short signals, 30 epochs
                (51200, 20),   # Stage 2: Medium signals, 20 epochs
                (102400, 50),  # Stage 3: Full signals, 50 epochs
            ]
        else:
            self.schedule = schedule

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        return {
            'train_loss': total_loss / len(train_loader),
            'train_accuracy': 100.0 * correct / total
        }

    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model."""
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': 100.0 * correct / total
        }

    def train_progressive(
        self,
        base_train_dataset: Dataset,
        base_val_dataset: Dataset,
        batch_size: int = 32,
        num_workers: int = 4,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> Dict[str, list]:
        """
        Train with progressive resizing schedule.

        Args:
            base_train_dataset: Training dataset with full-length signals
            base_val_dataset: Validation dataset with full-length signals
            batch_size: Batch size
            num_workers: Number of data loading workers
            scheduler: Optional learning rate scheduler

        Returns:
            Training history
        """
        history = {
            'stage_lengths': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        for stage_idx, (signal_length, stage_epochs) in enumerate(self.schedule):
            print(f"\n{'='*60}")
            print(f"Stage {stage_idx + 1}: Signal length = {signal_length}, Epochs = {stage_epochs}")
            print(f"{'='*60}")

            # Create resized datasets
            train_dataset = ResizableSignalDataset(
                base_train_dataset,
                target_length=signal_length,
                resize_method='interpolate'
            )

            val_dataset = ResizableSignalDataset(
                base_val_dataset,
                target_length=signal_length,
                resize_method='interpolate'
            )

            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )

            # Train for this stage
            for epoch in range(stage_epochs):
                train_metrics = self.train_epoch(train_loader, epoch)
                val_metrics = self.evaluate(val_loader)

                if scheduler is not None:
                    scheduler.step()

                # Track history
                history['stage_lengths'].append(signal_length)
                history['train_loss'].append(train_metrics['train_loss'])
                history['train_accuracy'].append(train_metrics['train_accuracy'])
                history['val_loss'].append(val_metrics['val_loss'])
                history['val_accuracy'].append(val_metrics['val_accuracy'])

                # Print progress
                print(f"Epoch {epoch+1}/{stage_epochs} "
                      f"(Length={signal_length}): "
                      f"Train Loss={train_metrics['train_loss']:.4f}, "
                      f"Train Acc={train_metrics['train_accuracy']:.2f}%, "
                      f"Val Loss={val_metrics['val_loss']:.4f}, "
                      f"Val Acc={val_metrics['val_accuracy']:.2f}%")

        return history


# Example usage
if __name__ == "__main__":
    print("Progressive Resizing Training Framework")
    print("\nExample usage:")
    print("""
    # Create model
    model = create_resnet18_1d(num_classes=11)

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Progressive training schedule
    schedule = [
        (25600, 30),   # Stage 1: Short signals
        (51200, 20),   # Stage 2: Medium signals
        (102400, 50),  # Stage 3: Full signals
    ]

    trainer = ProgressiveResizingTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device='cuda',
        schedule=schedule
    )

    # Train with progressive resizing
    history = trainer.train_progressive(
        base_train_dataset=train_dataset,
        base_val_dataset=val_dataset,
        batch_size=32
    )

    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.2f}%")
    """)
