#!/usr/bin/env python3
"""
LSTM Training Script - Train LSTM models for bearing fault diagnosis

This script provides a command-line interface for training LSTM models.

Usage:
    # Train Vanilla LSTM
    python scripts/train_lstm.py --model vanilla_lstm --epochs 50 --batch-size 32

    # Train BiLSTM with custom hidden size
    python scripts/train_lstm.py --model bilstm --hidden-size 256 --epochs 100

    # Train with mixed precision
    python scripts/train_lstm.py --model bilstm --mixed-precision --epochs 75

Author: Bearing Fault Diagnosis Team
Milestone: 2 - LSTM Implementation
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

# Import project modules
from models import create_model
from data.lstm_dataloader import create_lstm_dataloaders
from training.lstm_trainer import LSTMTrainer
from training.optimizers import create_optimizer, create_scheduler
from training.losses import create_loss_function
from utils.reproducibility import set_seed
from utils.device_manager import get_device
from utils.logging import get_logger
from utils.constants import NUM_CLASSES


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train LSTM for bearing fault diagnosis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument('--model', type=str, default='vanilla_lstm',
                       choices=['vanilla_lstm', 'lstm', 'bilstm'],
                       help='LSTM architecture to train')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout probability')

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/raw/bearing_data',
                       help='Directory with .MAT files')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--signal-length', type=int, default=102400,
                       help='Signal length')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=64,
                       help='Batch size for validation')

    # Optimization arguments
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd', 'rmsprop'],
                       help='Optimizer')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                       help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum (for SGD)')

    # Loss function
    parser.add_argument('--loss', type=str, default='cross_entropy',
                       choices=['cross_entropy', 'label_smoothing'],
                       help='Loss function')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                       help='Label smoothing factor (if using label_smoothing loss)')

    # Learning rate scheduling
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='Learning rate scheduler')
    parser.add_argument('--lr-patience', type=int, default=5,
                       help='Patience for ReduceLROnPlateau scheduler')
    parser.add_argument('--lr-factor', type=float, default=0.1,
                       help='LR reduction factor for ReduceLROnPlateau')

    # Regularization
    parser.add_argument('--early-stopping-patience', type=int, default=15,
                       help='Early stopping patience (0 to disable)')
    parser.add_argument('--gradient-clip', type=float, default=0.0,
                       help='Gradient clipping max norm (0 to disable)')

    # Performance
    parser.add_argument('--mixed-precision', action='store_true',
                       help='Enable FP16 mixed precision training')

    # Output
    parser.add_argument('--checkpoint-dir', type=str, default='results/checkpoints',
                       help='Checkpoint base directory (model name will be appended)')
    parser.add_argument('--save-every', type=int, default=0,
                       help='Save checkpoint every N epochs (0 to save only best)')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cuda', 'cpu', 'auto'],
                       help='Device to use')

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Validate arguments
    if not Path(args.data_dir).exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    if args.hidden_size <= 0 or args.num_layers <= 0:
        raise ValueError("hidden_size and num_layers must be positive integers")

    if not 0 <= args.dropout <= 1:
        raise ValueError(f"dropout must be in range [0, 1], got {args.dropout}")

    if args.lr <= 0:
        raise ValueError(f"learning rate must be positive, got {args.lr}")

    if args.weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {args.weight_decay}")

    if args.epochs <= 0:
        raise ValueError(f"epochs must be positive, got {args.epochs}")

    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")

    # Set random seed
    set_seed(args.seed)

    # Get device
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device

    print("="*70)
    print("  LSTM Training - Bearing Fault Diagnosis")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Scheduler: {args.scheduler}")
    print(f"  Device: {device}")
    print(f"  Mixed precision: {args.mixed_precision}")
    print(f"  Random seed: {args.seed}\n")

    # Validate data directory exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {args.data_dir}\n"
            f"Please specify a valid directory containing .MAT files using --data-dir\n"
            f"Expected structure: {args.data_dir}/<fault_type>/*.mat"
        )

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = create_lstm_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        signal_length=args.signal_length,
        random_seed=args.seed
    )
    print("Dataloaders created!\n")

    # Create model
    print(f"Creating {args.model} model...")
    model = create_model(
        model_name=args.model,
        num_classes=NUM_CLASSES,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    # Print model info
    if hasattr(model, 'get_model_info'):
        model_info = model.get_model_info()
        print(f"Model created!")
        print(f"  Parameters: {model_info['total_params']:,}")
        print(f"  Model size: {model_info['model_size_mb']:.2f} MB\n")
    else:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model created!")
        print(f"  Parameters: {total_params:,}\n")

    # Create optimizer
    optimizer = create_optimizer(
        model_params=model.parameters(),
        optimizer_name=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum
    )

    # Create loss function
    criterion = create_loss_function(
        loss_name=args.loss,
        num_classes=NUM_CLASSES,
        label_smoothing=args.label_smoothing
    )

    # Create scheduler
    if args.scheduler != 'none':
        scheduler = create_scheduler(
            optimizer=optimizer,
            scheduler_name=args.scheduler,
            num_epochs=args.epochs,
            patience=args.lr_patience,
            factor=args.lr_factor
        )
    else:
        scheduler = None

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / args.model / datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer
    early_stopping_patience = args.early_stopping_patience if args.early_stopping_patience > 0 else None

    trainer = LSTMTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        early_stopping_patience=early_stopping_patience,
        checkpoint_dir=str(checkpoint_dir),
        mixed_precision=args.mixed_precision
    )

    # Train
    print("="*70)
    print("  Starting Training")
    print("="*70 + "\n")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        verbose=True
    )

    # Final evaluation on test set
    print("\n" + "="*70)
    print("  Final Evaluation on Test Set")
    print("="*70 + "\n")

    test_metrics = trainer.validate(test_loader)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%\n")

    # Save final model
    final_model_path = checkpoint_dir / 'final_model.pth'
    trainer.save_model(str(final_model_path))

    print("="*70)
    print("  Training Complete!")
    print("="*70)
    print(f"\nBest validation accuracy: {max(history['val_acc']):.2f}%")
    print(f"Final test accuracy: {test_metrics['accuracy']:.2f}%")
    print(f"\nCheckpoints saved to: {checkpoint_dir}")
    print(f"Best model: {checkpoint_dir}/best_model.pth")
    print(f"Final model: {final_model_path}\n")


if __name__ == "__main__":
    main()
