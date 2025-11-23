#!/usr/bin/env python3
"""
Hybrid CNN-LSTM Training Script

Train configurable hybrid models combining CNN and LSTM.

Usage:
    # Recommended configuration 1 (ResNet34 + BiLSTM)
    python scripts/train_hybrid.py --model recommended_1 --epochs 75

    # Recommended configuration 2 (EfficientNet + BiLSTM)
    python scripts/train_hybrid.py --model recommended_2 --epochs 75

    # Custom hybrid (any CNN + any LSTM)
    python scripts/train_hybrid.py --model custom \
        --cnn-type resnet34 --lstm-type bilstm --epochs 75

Author: Bearing Fault Diagnosis Team
Milestone: 3 - CNN-LSTM Hybrid
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
from datetime import datetime

from models import create_model
from data.matlab_importer import MatlabImporter
from data.cnn_dataset import create_cnn_datasets_from_arrays
from data.cnn_dataloader import create_cnn_dataloaders
from training.cnn_trainer import CNNTrainer
from training.optimizers import create_optimizer
from training.losses import create_loss_function
from utils.reproducibility import set_seed
from utils.device_manager import get_device


def parse_args():
    parser = argparse.ArgumentParser(description='Train Hybrid CNN-LSTM')
    parser.add_argument('--model', type=str, default='recommended_1',
                       choices=['recommended_1', 'recommended_2', 'recommended_3', 'custom'],
                       help='Hybrid model configuration')
    parser.add_argument('--cnn-type', type=str, default='resnet34',
                       help='CNN backbone (for custom hybrid)')
    parser.add_argument('--lstm-type', type=str, default='bilstm',
                       help='LSTM type (for custom hybrid)')
    parser.add_argument('--lstm-hidden-size', type=int, default=256)
    parser.add_argument('--lstm-num-layers', type=int, default=2)
    parser.add_argument('--pooling', type=str, default='mean',
                       choices=['mean', 'max', 'last', 'attention'])
    parser.add_argument('--freeze-cnn', action='store_true',
                       help='Freeze CNN weights')

    parser.add_argument('--data-dir', type=str, default='data/raw/bearing_data',
                       help='Directory containing .mat files')
    parser.add_argument('--epochs', type=int, default=75)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--mixed-precision', action='store_true')
    parser.add_argument('--checkpoint-dir', type=str, default='results/checkpoints/hybrid')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_mat_files(data_dir: str, seed: int = 42):
    """
    Load .mat files from directory and prepare datasets.

    Args:
        data_dir: Directory containing .mat files
        seed: Random seed for reproducibility

    Returns:
        (train_dataset, val_dataset, test_dataset) tuple
    """
    print(f"\nLoading .mat files from: {data_dir}")

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}\n"
            f"Please ensure your .mat files are in this directory."
        )

    # Load all .mat files
    importer = MatlabImporter()
    batch_data = importer.load_batch(data_path, pattern='*.mat')

    if not batch_data:
        raise ValueError(f"No .mat files found in {data_dir}")

    print(f"✓ Loaded {len(batch_data)} signals")

    # Extract signals and labels as numpy arrays
    signals, labels = importer.extract_signals_and_labels(batch_data)
    print(f"✓ Signal array shape: {signals.shape}")
    print(f"✓ Unique labels: {sorted(set(labels))}")

    # Create train/val/test datasets
    train_dataset, val_dataset, test_dataset = create_cnn_datasets_from_arrays(
        signals=signals,
        labels=labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        augment_train=True,
        random_seed=seed
    )

    print(f"✓ Created datasets: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print("="*70)
    print("  Hybrid CNN-LSTM Training - Bearing Fault Diagnosis")
    print("="*70)
    print(f"\nConfiguration: {args.model}")
    if args.model == 'custom':
        print(f"  CNN: {args.cnn_type}")
        print(f"  LSTM: {args.lstm_type}")
    print(f"  LSTM Hidden: {args.lstm_hidden_size}")
    print(f"  Pooling: {args.pooling}")
    print(f"  Freeze CNN: {args.freeze_cnn}")
    print(f"  Device: {device}\n")

    # Load data from .mat files
    train_dataset, val_dataset, test_dataset = load_mat_files(args.data_dir, args.seed)

    # Create dataloaders from datasets
    loaders = create_cnn_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )

    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']

    # Create model
    print(f"\nCreating {args.model} hybrid model...")
    if args.model == 'custom':
        model = create_model(
            'custom',
            cnn_type=args.cnn_type,
            lstm_type=args.lstm_type,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_num_layers=args.lstm_num_layers,
            pooling_method=args.pooling,
            freeze_cnn=args.freeze_cnn
        )
    else:
        model = create_model(
            args.model,
            lstm_hidden_size=args.lstm_hidden_size,
            pooling_method=args.pooling,
            freeze_cnn=args.freeze_cnn
        )

    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        print(f"Model created!")
        print(f"  Total params: {info['total_params']:,}")
        print(f"  CNN params: {info['cnn_params']:,}")
        print(f"  LSTM params: {info['lstm_params']:,}")
        print(f"  Size: {info['model_size_mb']:.2f} MB\n")

    # Create optimizer and loss
    optimizer = create_optimizer(model.parameters(), args.optimizer, lr=args.lr)
    criterion = create_loss_function('cross_entropy', num_classes=11)

    # Create scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / args.model / datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train - FIXED: pass train_loader and val_loader to CNNTrainer
    trainer = CNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        lr_scheduler=scheduler,
        mixed_precision=args.mixed_precision,
        checkpoint_dir=str(checkpoint_dir)
    )

    print("="*70)
    print("  Starting Training")
    print("="*70 + "\n")

    # FIXED: use fit() instead of train()
    history = trainer.fit(num_epochs=args.epochs, save_best=True, verbose=True)

    # Final evaluation on test set
    print("\n" + "="*70)
    print("  Final Evaluation on Test Set")
    print("="*70 + "\n")

    # FIXED: manually evaluate on test set since validate() takes no arguments
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * signals.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = test_loss / total
    test_accuracy = 100.0 * correct / total

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%\n")

    print("="*70)
    print("  Training Complete!")
    print("="*70)
    print(f"\nBest val accuracy: {max(history['val_acc']):.2f}%")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"\nCheckpoints saved to: {checkpoint_dir}\n")


if __name__ == "__main__":
    main()
