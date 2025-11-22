"""
Callback System for Training Extensibility

Provides callbacks for:
- Early stopping
- Model checkpointing
- Learning rate scheduling
- Logging (TensorBoard, MLflow)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Callable
from pathlib import Path
import numpy as np


class Callback:
    """Base callback class."""

    def on_train_begin(self, trainer):
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self, trainer, epoch: int):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        """Called at the end of each epoch."""
        pass

    def __call__(self, trainer, metrics: Dict[str, float]):
        """Make callback callable."""
        self.on_epoch_end(trainer, trainer.state.epoch, metrics)


class EarlyStopping(Callback):
    """
    Stop training when monitored metric stops improving.

    Args:
        monitor: Metric to monitor (e.g., 'val_loss')
        patience: Number of epochs with no improvement before stopping
        mode: 'min' or 'max' (default: 'min')
        min_delta: Minimum change to qualify as improvement (default: 0)
        verbose: Print messages (default: True)
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

        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.counter = 0
        self.stopped_epoch = 0

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        if self.monitor not in metrics:
            return

        current = metrics[self.monitor]

        if self.mode == 'min':
            is_improvement = current < (self.best_value - self.min_delta)
        else:
            is_improvement = current > (self.best_value + self.min_delta)

        if is_improvement:
            self.best_value = current
            self.counter = 0
            if self.verbose:
                print(f"EarlyStopping: {self.monitor} improved to {current:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.monitor} did not improve ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                raise StopIteration("Early stopping")


class ModelCheckpoint(Callback):
    """
    Save model checkpoints based on monitored metric.

    Args:
        filepath: Path to save checkpoint
        monitor: Metric to monitor (e.g., 'val_loss')
        mode: 'min' or 'max' (default: 'min')
        save_best_only: Only save when metric improves (default: True)
        verbose: Print messages (default: True)
    """
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = True,
        verbose: bool = True
    ):
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose

        self.best_value = float('inf') if mode == 'min' else float('-inf')

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        if self.monitor not in metrics:
            return

        current = metrics[self.monitor]

        if self.mode == 'min':
            is_improvement = current < self.best_value
        else:
            is_improvement = current > self.best_value

        if is_improvement or not self.save_best_only:
            if is_improvement:
                self.best_value = current

            # Create checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': trainer.model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'metrics': metrics,
                'best_value': self.best_value
            }

            # Save
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, self.filepath)

            if self.verbose:
                print(f"Checkpoint saved to {self.filepath}")


class LearningRateScheduler(Callback):
    """
    Learning rate scheduler callback.

    Args:
        scheduler: PyTorch learning rate scheduler
        verbose: Print messages (default: True)
    """
    def __init__(self, scheduler, verbose: bool = True):
        self.scheduler = scheduler
        self.verbose = verbose

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        old_lr = trainer.optimizer.param_groups[0]['lr']

        # Step the scheduler
        if hasattr(self.scheduler, 'step'):
            # Some schedulers need metrics (e.g., ReduceLROnPlateau)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(metrics.get('val_loss', 0))
            else:
                self.scheduler.step()

        new_lr = trainer.optimizer.param_groups[0]['lr']

        if self.verbose and old_lr != new_lr:
            print(f"Learning rate changed: {old_lr:.6f} -> {new_lr:.6f}")


class TensorBoardLogger(Callback):
    """
    Log metrics to TensorBoard.

    Args:
        log_dir: Directory to save TensorBoard logs
    """
    def __init__(self, log_dir: str = 'runs'):
        self.log_dir = Path(log_dir)
        self.writer = None

    def on_train_begin(self, trainer):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        if self.writer is None:
            return

        for key, value in metrics.items():
            self.writer.add_scalar(key, value, epoch)

        # Log learning rate
        lr = trainer.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, epoch)

    def on_train_end(self, trainer):
        if self.writer is not None:
            self.writer.close()


class MLflowLogger(Callback):
    """
    Log metrics to MLflow.

    Args:
        experiment_name: Name of MLflow experiment
        run_name: Optional name for this run
    """
    def __init__(self, experiment_name: str = 'bearing_fault_diagnosis', run_name: Optional[str] = None):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.mlflow = None
        self.run = None

    def on_train_begin(self, trainer):
        try:
            import mlflow
            self.mlflow = mlflow

            # Set experiment
            mlflow.set_experiment(self.experiment_name)

            # Start run
            self.run = mlflow.start_run(run_name=self.run_name)

            # Log model config
            if hasattr(trainer.model, 'get_config'):
                mlflow.log_params(trainer.model.get_config())

        except ImportError:
            print("MLflow not available. Install with: pip install mlflow")

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        if self.mlflow is None:
            return

        # Log metrics
        self.mlflow.log_metrics(metrics, step=epoch)

    def on_train_end(self, trainer):
        if self.mlflow is not None and self.run is not None:
            self.mlflow.end_run()


class ProgressPrinter(Callback):
    """
    Simple callback to print training progress.
    """
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, float]):
        metric_str = ' | '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Epoch {epoch+1}: {metric_str}")
