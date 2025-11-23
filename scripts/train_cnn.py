#!/usr/bin/env python3
"""
CNN Training Script - Train 1D CNN models for bearing fault diagnosis

This script provides a command-line interface for training CNN models.
Supports multiple architectures, hyperparameter configuration, and experiment tracking.

Usage:
    # Train baseline CNN
    python scripts/train_cnn.py --model cnn1d --epochs 50 --batch-size 32

    # Train attention CNN with custom config
    python scripts/train_cnn.py --model attention --epochs 100 --lr 0.001

    # Train multi-scale CNN with mixed precision
    python scripts/train_cnn.py --model multiscale --mixed-precision --epochs 50

Author: Phase 2 - CNN Implementation
Date: 2025-11-20
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any

# Import project modules
from models.cnn.cnn_1d import CNN1D
from models.cnn.attention_cnn import AttentionCNN1D, LightweightAttentionCNN
from models.cnn.multi_scale_cnn import MultiScaleCNN1D, DilatedMultiScaleCNN
from data.cnn_dataloader import create_cnn_dataloaders
from training.cnn_trainer import CNNTrainer
from training.cnn_optimizer import create_optimizer
from training.cnn_losses import create_criterion
from training.cnn_schedulers import create_cosine_scheduler, create_step_scheduler, create_plateau_scheduler, create_onecycle_scheduler
from utils.reproducibility import set_seed
from utils.device_manager import get_device
from utils.logging import get_logger


# Model registry
MODEL_REGISTRY = {
    'cnn1d': CNN1D,
    'attention': AttentionCNN1D,
    'attention-lite': LightweightAttentionCNN,
    'multiscale': MultiScaleCNN1D,
    'dilated': DilatedMultiScaleCNN
}


def create_scheduler(optimizer, scheduler_name: str, num_epochs: int, steps_per_epoch: int, warmup_epochs: int = 0):
    """Create learning rate scheduler based on name."""
    if scheduler_name == 'none':
        return None
    elif scheduler_name == 'cosine':
        return create_cosine_scheduler(optimizer, num_epochs=num_epochs, eta_min=1e-6)
    elif scheduler_name == 'step':
        return create_step_scheduler(optimizer, step_size=num_epochs // 3, gamma=0.1)
    elif scheduler_name == 'plateau':
        return create_plateau_scheduler(optimizer, mode='min', factor=0.1, patience=10)
    elif scheduler_name == 'onecycle':
        return create_onecycle_scheduler(optimizer, max_lr=0.01, epochs=num_epochs, steps_per_epoch=steps_per_epoch)
    elif scheduler_name == 'warmup_cosine':
        # Simple warmup + cosine
        return create_cosine_scheduler(optimizer, num_epochs=num_epochs, eta_min=1e-6)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Train 1D CNN for bearing fault diagnosis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument('--model', type=str, default='cnn1d',
                       choices=list(MODEL_REGISTRY.keys()),
                       help='CNN architecture to train')

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory with processed data')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=64,
                       help='Batch size for validation')

    # Optimization arguments
    parser.add_argument('--optimizer', type=str, default='adamw',
                       choices=['adam', 'adamw', 'sgd'],
                       help='Optimizer')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                       help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum (for SGD)')

    # Loss function
    parser.add_argument('--loss', type=str, default='cross_entropy',
                       choices=['cross_entropy', 'focal', 'label_smoothing'],
                       help='Loss function')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing factor')

    # Scheduler arguments
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['none', 'step', 'cosine', 'onecycle', 'warmup_cosine'],
                       help='Learning rate scheduler')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                       help='Warmup epochs (for warmup schedulers)')

    # Regularization
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout probability')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                       help='Gradient clipping value (0 = no clipping)')

    # Mixed precision
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Enable mixed precision training (FP16)')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=10,
                       help='Save checkpoint every N epochs')

    # Early stopping
    parser.add_argument('--early-stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')

    # Experiment tracking
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name (default: auto-generated)')
    parser.add_argument('--tags', type=str, nargs='*', default=[],
                       help='Experiment tags')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')

    # Logging
    parser.add_argument('--log-interval', type=int, default=10,
                       help='Log training metrics every N batches')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    return parser.parse_args()


def create_model(args: argparse.Namespace, num_classes: int) -> torch.nn.Module:
    """Create CNN model"""
    model_class = MODEL_REGISTRY[args.model]

    # Different models have different parameter names
    if args.model == 'cnn1d':
        # CNN1D uses: num_classes, input_channels, dropout, use_batch_norm
        model_kwargs = {
            'num_classes': num_classes,
            'input_channels': 1,
            'dropout': args.dropout,
            'use_batch_norm': True
        }
    elif args.model in ['attention', 'attention-lite']:
        # AttentionCNN uses: num_classes, input_length, in_channels, dropout
        model_kwargs = {
            'num_classes': num_classes,
            'input_length': 102400,
            'in_channels': 1,
            'dropout': args.dropout
        }
    elif args.model in ['multiscale', 'dilated']:
        # MultiScaleCNN uses: num_classes, input_length, in_channels, dropout
        model_kwargs = {
            'num_classes': num_classes,
            'input_length': 102400,
            'in_channels': 1,
            'dropout': args.dropout
        }
    else:
        # Default for other models
        model_kwargs = {
            'num_classes': num_classes,
            'input_length': 102400,
            'in_channels': 1
        }

    model = model_class(**model_kwargs)

    return model


def load_data(args: argparse.Namespace, logger):
    """Load and prepare data"""
    logger.info(f"Loading data from {args.data_dir}...")

    # Load .mat files from directory
    from data.matlab_importer import load_mat_dataset
    from data.cnn_dataset import RawSignalDataset
    from sklearn.model_selection import train_test_split

    # Load dataset
    logger.info("Loading .MAT files...")
    signals, labels, label_names = load_mat_dataset(args.data_dir)

    logger.info(f"✓ Loaded {len(signals)} signals")
    logger.info(f"  Signal shape: {signals.shape}")
    logger.info(f"  Classes: {len(np.unique(labels))} ({', '.join(label_names[:5])}...)")

    # Split into train/val/test (70/15/15)
    train_signals, temp_signals, train_labels, temp_labels = train_test_split(
        signals, labels, test_size=0.3, random_state=args.seed, stratify=labels
    )
    val_signals, test_signals, val_labels, test_labels = train_test_split(
        temp_signals, temp_labels, test_size=0.5, random_state=args.seed, stratify=temp_labels
    )

    # Create datasets
    train_dataset = RawSignalDataset(train_signals, train_labels)
    val_dataset = RawSignalDataset(val_signals, val_labels)
    test_dataset = RawSignalDataset(test_signals, test_labels)

    logger.info(f"✓ Created datasets:")
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Val:   {len(val_dataset)} samples")
    logger.info(f"  Test:  {len(test_dataset)} samples")

    # Create dataloaders
    loaders = create_cnn_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    logger.info(f"✓ Created dataloaders:")
    logger.info(f"  Train: {len(train_loader)} batches")
    logger.info(f"  Val:   {len(val_loader)} batches")
    logger.info(f"  Test:  {len(test_loader)} batches")

    return train_loader, val_loader, test_loader


def main():
    """Main training function"""
    args = parse_args()

    # Setup logging
    logger = get_logger(__name__)

    # Print header
    print("=" * 80)
    print("CNN Training Script - Bearing Fault Diagnosis")
    print("=" * 80)
    print(f"Model:       {args.model}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Optimizer:   {args.optimizer}")
    print(f"Scheduler:   {args.scheduler}")
    print("=" * 80)

    # Set seed for reproducibility
    set_seed(args.seed)
    logger.info(f"✓ Set random seed: {args.seed}")

    # Get device
    if args.device == 'auto':
        device = get_device(prefer_gpu=True)
    else:
        device = torch.device(args.device)
    logger.info(f"✓ Using device: {device}")

    # Load data
    train_loader, val_loader, test_loader = load_data(args, logger)
    num_classes = len(torch.unique(torch.tensor(train_loader.dataset.labels)))

    # Create model
    logger.info(f"Creating model: {args.model}...")
    model = create_model(args, num_classes)
    logger.info(f"✓ Model created")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"  Trainable:  {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create optimizer
    optimizer = create_optimizer(
        model.parameters(),
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum
    )
    logger.info(f"✓ Created optimizer: {args.optimizer}")

    # Create loss function
    criterion = create_criterion(
        criterion_name=args.loss,
        num_classes=num_classes,
        label_smoothing=args.label_smoothing if args.loss == 'label_smoothing' else 0.0
    )
    logger.info(f"✓ Created loss function: {args.loss}")

    # Create scheduler
    scheduler = create_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        num_epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        warmup_epochs=args.warmup_epochs
    )
    logger.info(f"✓ Created scheduler: {args.scheduler}")

    # Create trainer
    trainer = CNNTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        mixed_precision=args.mixed_precision,
        grad_clip=args.grad_clip if args.grad_clip > 0 else None
    )
    logger.info(f"✓ Created trainer")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / args.model
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Checkpoint directory: {checkpoint_dir}")

    # Experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"{args.model}_{timestamp}"

    logger.info(f"✓ Experiment name: {args.experiment_name}")

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"✓ Resumed from epoch {start_epoch}")

    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info("-" * 80)

        # Train
        train_metrics = trainer.train_epoch(train_loader, epoch)
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Acc: {train_metrics['accuracy']:.4f}")

        # Validate
        val_metrics = trainer.validate(val_loader)
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Acc: {val_metrics['accuracy']:.4f}")

        # Update scheduler
        if scheduler:
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_val_acc
        if is_best:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0

            # Save best model
            best_path = checkpoint_dir / f"{args.experiment_name}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_acc': best_val_acc,
                'args': vars(args)
            }, best_path)
            logger.info(f"✓ Saved best model: {best_path}")

        else:
            patience_counter += 1

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"{args.experiment_name}_epoch{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_acc': val_metrics['accuracy'],
                'args': vars(args)
            }, checkpoint_path)
            logger.info(f"✓ Saved checkpoint: {checkpoint_path}")

        # Early stopping
        if args.early_stopping and patience_counter >= args.patience:
            logger.info(f"\nEarly stopping triggered (patience={args.patience})")
            break

    # Final evaluation on test set
    logger.info("\n" + "=" * 80)
    logger.info("Final Evaluation on Test Set")
    logger.info("=" * 80)

    # Load best model
    best_checkpoint = torch.load(checkpoint_dir / f"{args.experiment_name}_best.pth")
    model.load_state_dict(best_checkpoint['model_state_dict'])
    logger.info(f"✓ Loaded best model (epoch {best_checkpoint['epoch']+1}, "
               f"val_acc={best_checkpoint['val_acc']:.4f})")

    # Evaluate
    test_metrics = trainer.validate(test_loader)
    logger.info(f"\nTest Results:")
    logger.info(f"  Loss:     {test_metrics['loss']:.4f}")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
