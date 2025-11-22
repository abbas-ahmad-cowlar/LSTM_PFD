"""
Training callbacks for monitoring and debugging.

Purpose:
    Callback system for tracking training dynamics:
    - Learning rate monitoring
    - Gradient statistics (norm, distribution)
    - Model weight statistics
    - Training time profiling
    - Custom metric logging
    - Integration with tensorboard/MLflow (optional)

Author: LSTM_PFD Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import time
import json
from collections import defaultdict

from utils.logging import get_logger

logger = get_logger(__name__)


class Callback:
    """
    Base callback class.

    Callbacks can hook into different training events:
    - on_train_begin / on_train_end
    - on_epoch_begin / on_epoch_end
    - on_batch_begin / on_batch_end
    """

    def on_train_begin(self, logs: Optional[Dict] = None):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, logs: Optional[Dict] = None):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """Called at the end of each batch."""
        pass


class LearningRateMonitor(Callback):
    """
    Monitor learning rate during training.

    Logs the current learning rate at each epoch.

    Args:
        optimizer: Optimizer to monitor
        log_interval: Log every N epochs (default: 1)
        verbose: Print LR changes

    Example:
        >>> lr_monitor = LearningRateMonitor(optimizer, verbose=True)
        >>> callbacks = [lr_monitor]
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        log_interval: int = 1,
        verbose: bool = True
    ):
        self.optimizer = optimizer
        self.log_interval = log_interval
        self.verbose = verbose
        self.lr_history = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if (epoch + 1) % self.log_interval == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_history.append((epoch, current_lr))

            if self.verbose:
                logger.info(f"Epoch {epoch + 1}: LR = {current_lr:.6f}")

    def get_history(self) -> List:
        """Get LR history."""
        return self.lr_history


class GradientMonitor(Callback):
    """
    Monitor gradient statistics during training.

    Tracks gradient norms and detects gradient issues (vanishing/exploding).

    Args:
        model: Model to monitor
        log_interval: Log every N batches
        norm_type: Norm type for gradient calculation (default: 2.0)
        alert_threshold: Alert if gradient norm exceeds this (default: 10.0)

    Example:
        >>> grad_monitor = GradientMonitor(model, log_interval=100)
    """

    def __init__(
        self,
        model: nn.Module,
        log_interval: int = 100,
        norm_type: float = 2.0,
        alert_threshold: float = 10.0
    ):
        self.model = model
        self.log_interval = log_interval
        self.norm_type = norm_type
        self.alert_threshold = alert_threshold
        self.gradient_norms = []
        self.batch_count = 0

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        self.batch_count += 1

        if self.batch_count % self.log_interval == 0:
            # Compute gradient norm
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(self.norm_type)
                    total_norm += param_norm.item() ** self.norm_type

            total_norm = total_norm ** (1.0 / self.norm_type)
            self.gradient_norms.append(total_norm)

            # Alert for exploding gradients
            if total_norm > self.alert_threshold:
                logger.warning(f"⚠️ Large gradient detected: {total_norm:.2f} (threshold: {self.alert_threshold})")

            # Alert for vanishing gradients
            if total_norm < 1e-7:
                logger.warning(f"⚠️ Very small gradient detected: {total_norm:.2e} (potential vanishing gradients)")

    def get_statistics(self) -> Dict[str, float]:
        """Get gradient statistics."""
        if not self.gradient_norms:
            return {}

        import numpy as np
        norms = np.array(self.gradient_norms)

        return {
            'mean': float(norms.mean()),
            'std': float(norms.std()),
            'min': float(norms.min()),
            'max': float(norms.max()),
            'median': float(np.median(norms))
        }


class ModelCheckpointCallback(Callback):
    """
    Save model checkpoints during training.

    Args:
        checkpoint_dir: Directory to save checkpoints
        monitor: Metric to monitor ('val_loss' or 'val_acc')
        mode: 'min' or 'max'
        save_best_only: Only save when monitored metric improves
        verbose: Print save events

    Example:
        >>> checkpoint_cb = ModelCheckpointCallback(
        ...     checkpoint_dir='./checkpoints',
        ...     monitor='val_acc',
        ...     mode='max',
        ...     save_best_only=True
        ... )
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        monitor: str = 'val_acc',
        mode: str = 'max',
        save_best_only: bool = True,
        verbose: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.optimizer = optimizer
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose

        self.best_metric = float('-inf') if mode == 'max' else float('inf')

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if logs is None or self.monitor not in logs:
            return

        current_metric = logs[self.monitor]

        # Check if improvement
        is_improvement = (
            (self.mode == 'max' and current_metric > self.best_metric) or
            (self.mode == 'min' and current_metric < self.best_metric)
        )

        if not self.save_best_only or is_improvement:
            self.best_metric = current_metric

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                self.monitor: current_metric
            }

            if self.optimizer is not None:
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

            filename = f"checkpoint_epoch{epoch:03d}_{self.monitor}{current_metric:.4f}.pth"
            filepath = self.checkpoint_dir / filename

            torch.save(checkpoint, filepath)

            if self.verbose:
                logger.info(f"✓ Checkpoint saved: {filepath}")


class TimingCallback(Callback):
    """
    Profile training time per epoch and batch.

    Args:
        verbose: Print timing information

    Example:
        >>> timer = TimingCallback(verbose=True)
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.epoch_start_time = None
        self.batch_start_time = None
        self.epoch_times = []
        self.batch_times = []

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if self.epoch_start_time is not None:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_times.append(epoch_time)

            if self.verbose:
                logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        self.batch_start_time = time.time()

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        if self.batch_start_time is not None:
            batch_time = time.time() - self.batch_start_time
            self.batch_times.append(batch_time)

    def get_statistics(self) -> Dict[str, float]:
        """Get timing statistics."""
        import numpy as np

        epoch_times = np.array(self.epoch_times) if self.epoch_times else np.array([])
        batch_times = np.array(self.batch_times) if self.batch_times else np.array([])

        stats = {}

        if len(epoch_times) > 0:
            stats['epoch_mean'] = float(epoch_times.mean())
            stats['epoch_std'] = float(epoch_times.std())
            stats['epoch_total'] = float(epoch_times.sum())

        if len(batch_times) > 0:
            stats['batch_mean'] = float(batch_times.mean())
            stats['batch_std'] = float(batch_times.std())

        return stats


class MetricLogger(Callback):
    """
    Log training metrics to file.

    Args:
        log_file: Path to log file
        log_interval: Log every N epochs

    Example:
        >>> logger_cb = MetricLogger(log_file='./logs/metrics.json')
    """

    def __init__(
        self,
        log_file: Path,
        log_interval: int = 1
    ):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.metrics_history = []

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if (epoch + 1) % self.log_interval == 0 and logs is not None:
            # Add epoch number
            log_entry = {'epoch': epoch + 1, **logs}
            self.metrics_history.append(log_entry)

            # Save to file
            with open(self.log_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)

    def get_history(self) -> List[Dict]:
        """Get metrics history."""
        return self.metrics_history


class EarlyStoppingCallback(Callback):
    """
    Early stopping based on validation metric.

    Args:
        monitor: Metric to monitor
        patience: Number of epochs without improvement
        mode: 'min' or 'max'
        min_delta: Minimum change to qualify as improvement
        verbose: Print early stopping events

    Example:
        >>> early_stop = EarlyStoppingCallback(
        ...     monitor='val_loss',
        ...     patience=10,
        ...     mode='min'
        ... )
    """

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        mode: str = 'min',
        min_delta: float = 0.0,
        verbose: bool = True
    ):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose

        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.should_stop = False

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        if logs is None or self.monitor not in logs:
            return

        current_metric = logs[self.monitor]

        # Check for improvement
        is_improvement = (
            (self.mode == 'min' and current_metric < self.best_metric - self.min_delta) or
            (self.mode == 'max' and current_metric > self.best_metric + self.min_delta)
        )

        if is_improvement:
            self.best_metric = current_metric
            self.counter = 0

            if self.verbose:
                logger.info(f"Validation metric improved to {current_metric:.4f}")
        else:
            self.counter += 1

            if self.verbose:
                logger.info(f"No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                self.should_stop = True

                if self.verbose:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")


class CallbackList:
    """
    Container for managing multiple callbacks.

    Args:
        callbacks: List of callback instances

    Example:
        >>> callbacks = CallbackList([
        ...     LearningRateMonitor(optimizer),
        ...     GradientMonitor(model),
        ...     TimingCallback()
        ... ])
        >>> callbacks.on_epoch_begin(0)
    """

    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks

    def on_train_begin(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)


def test_callbacks():
    """Test callback system."""
    print("=" * 60)
    print("Testing Callbacks")
    print("=" * 60)

    import torch.nn as nn
    import tempfile

    # Dummy model and optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        print("\n1. Testing LearningRateMonitor...")
        lr_monitor = LearningRateMonitor(optimizer, verbose=True)
        for epoch in range(3):
            lr_monitor.on_epoch_end(epoch)
        print(f"   LR history: {lr_monitor.get_history()}")

        print("\n2. Testing GradientMonitor...")
        grad_monitor = GradientMonitor(model, log_interval=1)

        # Simulate forward/backward
        for batch in range(3):
            x = torch.randn(4, 10)
            y = model(x).sum()
            y.backward()
            grad_monitor.on_batch_end(batch)

        stats = grad_monitor.get_statistics()
        print(f"   Gradient stats: {stats}")

        print("\n3. Testing TimingCallback...")
        timer = TimingCallback(verbose=True)
        for epoch in range(3):
            timer.on_epoch_begin(epoch)
            time.sleep(0.01)  # Simulate training
            timer.on_epoch_end(epoch)

        timing_stats = timer.get_statistics()
        print(f"   Timing stats: {timing_stats}")

        print("\n4. Testing MetricLogger...")
        log_file = tmpdir / 'metrics.json'
        metric_logger = MetricLogger(log_file)

        for epoch in range(3):
            logs = {'train_loss': 0.5 - epoch * 0.1, 'val_acc': 85.0 + epoch * 2.0}
            metric_logger.on_epoch_end(epoch, logs)

        print(f"   Logged {len(metric_logger.get_history())} entries")

        print("\n5. Testing EarlyStoppingCallback...")
        early_stop = EarlyStoppingCallback(monitor='val_loss', patience=2, mode='min', verbose=True)

        # Simulate training with plateauing loss
        val_losses = [0.5, 0.4, 0.38, 0.375, 0.374, 0.373]
        for epoch, loss in enumerate(val_losses):
            early_stop.on_epoch_end(epoch, {'val_loss': loss})
            if early_stop.should_stop:
                print(f"   Stopped at epoch {epoch + 1}")
                break

        print("\n6. Testing CallbackList...")
        callbacks = CallbackList([
            LearningRateMonitor(optimizer, verbose=False),
            TimingCallback(verbose=False)
        ])

        callbacks.on_train_begin()
        for epoch in range(2):
            callbacks.on_epoch_begin(epoch)
            time.sleep(0.01)
            callbacks.on_epoch_end(epoch, {'train_loss': 0.5})
        callbacks.on_train_end()
        print("   CallbackList executed successfully")

    print("\n" + "=" * 60)
    print("✅ All callback tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_callbacks()
