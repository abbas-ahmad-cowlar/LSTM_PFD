"""
Model checkpoint management for training and evaluation.

Purpose:
    Robust checkpoint saving/loading with automatic cleanup:
    - Save best model based on validation metrics
    - Maintain top-k checkpoints
    - Resume training from checkpoints
    - Load pretrained weights
    - Automatic old checkpoint cleanup

Author: Author Name
Date: 2025-11-20
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import shutil
from datetime import datetime

from utils.logging import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """
    Manager for model checkpoints with automatic cleanup.

    Features:
    - Save best model based on monitored metric
    - Keep top-k checkpoints (by metric)
    - Resume training with full state restoration
    - Load weights only (for inference)
    - Automatic backup and cleanup

    Args:
        checkpoint_dir: Directory to save checkpoints
        model: Model to checkpoint
        optimizer: Optimizer (optional, needed for training resumption)
        lr_scheduler: Learning rate scheduler (optional)
        mode: 'min' or 'max' for metric monitoring
        save_top_k: Number of best checkpoints to keep (-1 = keep all)
        filename_prefix: Prefix for checkpoint files

    Example:
        >>> from models.cnn.cnn_1d import CNN1D
        >>> model = CNN1D(num_classes=NUM_CLASSES)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>>
        >>> ckpt_manager = CheckpointManager(
        ...     checkpoint_dir='./checkpoints',
        ...     model=model,
        ...     optimizer=optimizer,
        ...     mode='max',  # Monitor validation accuracy
        ...     save_top_k=3
        ... )
        >>>
        >>> # During training
        >>> ckpt_manager.save_checkpoint(
        ...     epoch=10,
        ...     metric_value=95.5,
        ...     metric_name='val_acc'
        ... )
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        mode: str = 'max',
        save_top_k: int = 3,
        filename_prefix: str = 'checkpoint'
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename_prefix = filename_prefix

        # Tracking
        self.best_metric = float('-inf') if mode == 'max' else float('inf')
        self.best_checkpoint_path = None
        self.checkpoint_history: List[Tuple[float, Path]] = []  # (metric, path) pairs

        logger.info(f"CheckpointManager initialized at {checkpoint_dir}")
        logger.info(f"  Mode: {mode}, Top-K: {save_top_k}")

    def save_checkpoint(
        self,
        epoch: int,
        metric_value: float,
        metric_name: str,
        additional_info: Optional[Dict] = None,
        is_best: bool = False
    ) -> Path:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            metric_value: Value of monitored metric
            metric_name: Name of monitored metric (e.g., 'val_acc')
            additional_info: Additional information to save
            is_best: Force save as best model

        Returns:
            Path to saved checkpoint
        """
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metric_value': metric_value,
            'metric_name': metric_name,
            'timestamp': datetime.now().isoformat(),
            'best_metric': self.best_metric
        }

        # Add optimizer state
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

        # Add scheduler state
        if self.lr_scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.lr_scheduler.state_dict()

        # Add additional info
        if additional_info:
            checkpoint.update(additional_info)

        # Determine if this is the best model
        improved = self._is_improvement(metric_value)

        if improved or is_best:
            self.best_metric = metric_value
            is_best = True

        # Generate filename
        filename = self._generate_filename(epoch, metric_value, metric_name, is_best)
        checkpoint_path = self.checkpoint_dir / filename

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Update best checkpoint path
        if is_best:
            self.best_checkpoint_path = checkpoint_path
            # Also save a copy as 'best_model.pth'
            best_path = self.checkpoint_dir / 'best_model.pth'
            shutil.copy(checkpoint_path, best_path)
            logger.info(f"✓ New best model: {metric_name}={metric_value:.4f}")

        # Track checkpoint
        self.checkpoint_history.append((metric_value, checkpoint_path))

        # Cleanup old checkpoints
        if self.save_top_k > 0:
            self._cleanup_old_checkpoints()

        return checkpoint_path

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        load_optimizer: bool = True,
        load_scheduler: bool = True
    ) -> Dict:
        """
        Load checkpoint and restore model state.

        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to restore optimizer state
            load_scheduler: Whether to restore scheduler state

        Returns:
            Checkpoint dictionary with metadata
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Restore model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {checkpoint_path}")

        # Restore optimizer
        if load_optimizer and self.optimizer is not None:
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Optimizer state restored")
            else:
                logger.warning("Optimizer state not found in checkpoint")

        # Restore scheduler
        if load_scheduler and self.lr_scheduler is not None:
            if 'scheduler_state_dict' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Scheduler state restored")
            else:
                logger.warning("Scheduler state not found in checkpoint")

        # Update best metric
        if 'best_metric' in checkpoint:
            self.best_metric = checkpoint['best_metric']

        return checkpoint

    def load_best_checkpoint(
        self,
        load_optimizer: bool = False,
        load_scheduler: bool = False
    ) -> Optional[Dict]:
        """
        Load the best checkpoint.

        Args:
            load_optimizer: Whether to restore optimizer state
            load_scheduler: Whether to restore scheduler state

        Returns:
            Checkpoint dictionary or None if not found
        """
        # Try explicit best_model.pth first
        best_path = self.checkpoint_dir / 'best_model.pth'

        if best_path.exists():
            return self.load_checkpoint(best_path, load_optimizer, load_scheduler)

        # Fall back to tracked best checkpoint
        if self.best_checkpoint_path and self.best_checkpoint_path.exists():
            return self.load_checkpoint(
                self.best_checkpoint_path,
                load_optimizer,
                load_scheduler
            )

        logger.warning("No best checkpoint found")
        return None

    def load_weights_only(self, checkpoint_path: Path) -> nn.Module:
        """
        Load only model weights (no optimizer/scheduler).

        Useful for inference and transfer learning.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Model with loaded weights
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])

        logger.info(f"Weights loaded from {checkpoint_path}")

        return self.model

    def _is_improvement(self, metric_value: float) -> bool:
        """Check if metric value is an improvement."""
        if self.mode == 'max':
            return metric_value > self.best_metric
        else:
            return metric_value < self.best_metric

    def _generate_filename(
        self,
        epoch: int,
        metric_value: float,
        metric_name: str,
        is_best: bool
    ) -> str:
        """Generate checkpoint filename."""
        best_str = '_best' if is_best else ''
        filename = f"{self.filename_prefix}_epoch{epoch:03d}_{metric_name}{metric_value:.4f}{best_str}.pth"
        return filename

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only top-k."""
        if self.save_top_k < 0:
            return  # Keep all checkpoints

        # Sort by metric value
        if self.mode == 'max':
            self.checkpoint_history.sort(key=lambda x: x[0], reverse=True)
        else:
            self.checkpoint_history.sort(key=lambda x: x[0])

        # Remove checkpoints beyond top-k
        to_remove = self.checkpoint_history[self.save_top_k:]

        for _, checkpoint_path in to_remove:
            if checkpoint_path.exists() and checkpoint_path != self.best_checkpoint_path:
                checkpoint_path.unlink()
                logger.debug(f"Removed old checkpoint: {checkpoint_path}")

        # Update history
        self.checkpoint_history = self.checkpoint_history[:self.save_top_k]

    def list_checkpoints(self) -> List[Tuple[float, Path]]:
        """
        List all tracked checkpoints.

        Returns:
            List of (metric_value, path) tuples sorted by metric
        """
        return sorted(
            self.checkpoint_history,
            key=lambda x: x[0],
            reverse=(self.mode == 'max')
        )

    def get_best_metric(self) -> float:
        """Get the best metric value achieved."""
        return self.best_metric

    def get_checkpoint_info(self, checkpoint_path: Path) -> Dict:
        """
        Get information from a checkpoint without loading model weights.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        return {
            'epoch': checkpoint.get('epoch', -1),
            'metric_value': checkpoint.get('metric_value', None),
            'metric_name': checkpoint.get('metric_name', None),
            'timestamp': checkpoint.get('timestamp', None),
            'best_metric': checkpoint.get('best_metric', None)
        }


def test_checkpoint_manager():
    """Test checkpoint manager functionality."""
    print("=" * 60)
    print("Testing Checkpoint Manager")
    print("=" * 60)

    from models.cnn.cnn_1d import CNN1D
    import tempfile

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / 'checkpoints'

        # Create model and optimizer
        model = CNN1D(num_classes=NUM_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        print("\n1. Creating checkpoint manager...")
        ckpt_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            mode='max',
            save_top_k=3
        )

        # Simulate training epochs
        print("\n2. Saving checkpoints during 'training'...")
        val_accs = [85.5, 88.2, 90.1, 89.5, 91.3, 92.0, 91.8]

        for epoch, val_acc in enumerate(val_accs, start=1):
            ckpt_path = ckpt_manager.save_checkpoint(
                epoch=epoch,
                metric_value=val_acc,
                metric_name='val_acc'
            )
            print(f"   Epoch {epoch}: val_acc={val_acc:.1f}% -> {ckpt_path.name}")

        # Check best metric
        print(f"\n3. Best validation accuracy: {ckpt_manager.get_best_metric():.1f}%")

        # List checkpoints
        print("\n4. Tracked checkpoints:")
        for metric, path in ckpt_manager.list_checkpoints():
            print(f"   {metric:.1f}% - {path.name}")

        # Load best checkpoint
        print("\n5. Loading best checkpoint...")
        checkpoint = ckpt_manager.load_best_checkpoint(load_optimizer=True)
        print(f"   Loaded epoch {checkpoint['epoch']}, val_acc={checkpoint['metric_value']:.1f}%")

        # Test weight-only loading
        print("\n6. Testing weight-only loading...")
        new_model = CNN1D(num_classes=NUM_CLASSES)
        new_ckpt_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            model=new_model,
            mode='max'
        )
        best_path = checkpoint_dir / 'best_model.pth'
        new_ckpt_manager.load_weights_only(best_path)
        print(f"   Weights loaded successfully")

        # Get checkpoint info
        print("\n7. Checkpoint metadata:")
        info = ckpt_manager.get_checkpoint_info(best_path)
        for key, value in info.items():
            print(f"   {key}: {value}")

    print("\n" + "=" * 60)
    print("✅ All checkpoint manager tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_checkpoint_manager()
