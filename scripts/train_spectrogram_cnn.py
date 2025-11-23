"""
Training Script for Spectrogram CNN Models

Train 2D CNN models on spectrograms for bearing fault diagnosis.

Usage:
    # Train ResNet-18 on STFT spectrograms
    python scripts/train_spectrogram_cnn.py --model resnet18_2d --tfr_type stft

    # Train EfficientNet-B0 on CWT scalograms with transfer learning
    python scripts/train_spectrogram_cnn.py --model efficientnet_b0 --tfr_type cwt --pretrained

    # Train with SpecAugment
    python scripts/train_spectrogram_cnn.py --model resnet18_2d --use_specaugment --time_mask 40 --freq_mask 20

Example:
    python scripts/train_spectrogram_cnn.py \\
        --model resnet18_2d \\
        --tfr_type stft \\
        --batch_size 32 \\
        --epochs 100 \\
        --lr 1e-3 \\
        --pretrained \\
        --use_specaugment
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.spectrogram_cnn import get_model
from data.tfr_dataset import load_spectrograms, create_tfr_dataloaders
from training.spectrogram_trainer import SpectrogramTrainer
from evaluation.spectrogram_evaluator import SpectrogramEvaluator
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train 2D CNN models on spectrograms',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument('--model', type=str, default='resnet18_2d',
                        choices=['resnet18_2d', 'resnet34_2d', 'resnet50_2d',
                                 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b3'],
                        help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=11,
                        help='Number of output classes')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use ImageNet pretrained weights')

    # Data arguments
    parser.add_argument('--data_dir', type=Path, default='data/spectrograms',
                        help='Directory containing precomputed spectrograms')
    parser.add_argument('--tfr_type', type=str, default='stft',
                        choices=['stft', 'cwt', 'wvd'],
                        help='Time-frequency representation type')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw', 'sgd'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')

    # Augmentation arguments
    parser.add_argument('--use_specaugment', action='store_true',
                        help='Use SpecAugment for data augmentation')
    parser.add_argument('--time_mask', type=int, default=40,
                        help='Time mask parameter for SpecAugment')
    parser.add_argument('--freq_mask', type=int, default=20,
                        help='Frequency mask parameter for SpecAugment')
    parser.add_argument('--num_time_masks', type=int, default=2,
                        help='Number of time masks')
    parser.add_argument('--num_freq_masks', type=int, default=2,
                        help='Number of frequency masks')

    # Regularization arguments
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='Gradient clipping norm (0 to disable)')
    parser.add_argument('--early_stopping', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training (AMP)')

    # Output arguments
    parser.add_argument('--output_dir', type=Path, default='models/spectrogram_cnn/checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--save_frequency', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (default: auto-generated)')

    # System arguments
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to train on (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_optimizer(model, args):
    """Create optimizer based on arguments."""
    if args.optimizer == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    return optimizer


def create_scheduler(optimizer, args, num_epochs):
    """Create learning rate scheduler."""
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=10,
            verbose=True
        )
    elif args.scheduler == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    return scheduler


def main():
    """Main training loop."""
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Generate experiment name
    if args.experiment_name is None:
        args.experiment_name = f"{args.model}_{args.tfr_type}_lr{args.lr}_bs{args.batch_size}"
        if args.pretrained:
            args.experiment_name += "_pretrained"
        if args.use_specaugment:
            args.experiment_name += "_specaug"

    print(f"Experiment: {args.experiment_name}")

    # Load data
    print(f"\nLoading {args.tfr_type.upper()} spectrograms from {args.data_dir}...")

    try:
        train_loader, val_loader, test_loader = create_tfr_dataloaders(
            data_dir=args.data_dir / args.tfr_type,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Test samples: {len(test_loader.dataset)}")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease run the following command first to precompute spectrograms:")
        print(f"  python scripts/precompute_spectrograms.py --tfr_type {args.tfr_type}")
        sys.exit(1)

    # Create model
    print(f"\nCreating {args.model} model...")
    model = get_model(
        args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        dropout=args.dropout
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args, args.epochs)

    # Create criterion
    criterion = nn.CrossEntropyLoss()

    # Create trainer
    print("\nInitializing trainer...")
    trainer = SpectrogramTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        use_specaugment=args.use_specaugment,
        time_mask_param=args.time_mask,
        freq_mask_param=args.freq_mask,
        num_time_masks=args.num_time_masks,
        num_freq_masks=args.num_freq_masks,
        max_grad_norm=args.gradient_clip,
        mixed_precision=args.mixed_precision
    )

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("="*70)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epochs):
        # Train
        train_metrics = trainer.train_epoch()

        # Validate
        val_metrics = trainer.validate_epoch()

        # Update scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics['val_loss'])
            else:
                scheduler.step()

        # Update state
        trainer.state.epoch = epoch
        trainer.state.update({**train_metrics, **val_metrics})

        # Print metrics
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_metrics['train_loss']:.4f} "
              f"Train Acc: {train_metrics['train_acc']:.2f}% | "
              f"Val Loss: {val_metrics['val_loss']:.4f} "
              f"Val Acc: {val_metrics['val_acc']:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint
        if (epoch + 1) % args.save_frequency == 0:
            checkpoint_path = args.output_dir / f"{args.experiment_name}_epoch{epoch+1}.pth"
            trainer.save_checkpoint(checkpoint_path, val_metrics)

        # Save best model
        if val_metrics['val_acc'] > best_val_acc:
            best_val_acc = val_metrics['val_acc']
            best_checkpoint_path = args.output_dir / f"{args.experiment_name}_best.pth"
            trainer.save_checkpoint(best_checkpoint_path, val_metrics, is_best=True)
            patience_counter = 0
            print(f"✓ New best model saved (Val Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.early_stopping:
            print(f"\n⚠ Early stopping triggered (patience: {args.early_stopping})")
            break

    print("="*70)
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    evaluator = SpectrogramEvaluator(model, device=device)
    test_results = evaluator.evaluate(test_loader)

    print(f"\nTest Set Results:")
    print(f"  Accuracy: {test_results['accuracy']:.2f}%")

    # Save test results
    results_path = args.output_dir / f"{args.experiment_name}_test_results.npz"
    np.savez(
        results_path,
        accuracy=test_results['accuracy'],
        confusion_matrix=test_results['confusion_matrix'],
        predictions=test_results['predictions'],
        targets=test_results['targets'],
        probabilities=test_results['probabilities']
    )
    print(f"\nTest results saved to {results_path}")


if __name__ == '__main__':
    main()
