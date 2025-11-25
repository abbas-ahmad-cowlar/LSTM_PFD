#!/usr/bin/env python3
"""
Example Usage - Hybrid CNN-LSTM Models

Demonstrates how to use the hybrid models programmatically.

Usage:
    python scripts/example_usage.py

Author: Bearing Fault Diagnosis Team
Milestone: 3 - CNN-LSTM Hybrid
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from models import (
    create_model,
    list_available_cnn_backbones,
    list_available_lstm_types
)
from data.cnn_dataloader import create_cnn_dataloaders
from training.cnn_trainer import CNNTrainer
from training.optimizers import create_optimizer
from training.losses import create_loss_function
from utils.device_manager import get_device
from utils.reproducibility import set_seed


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def example_1_list_available_models():
    """Example 1: List available architectures."""
    print_section("Example 1: List Available Architectures")

    print("Available CNN Backbones:")
    for cnn in list_available_cnn_backbones():
        print(f"  - {cnn}")

    print("\nAvailable LSTM Types:")
    for lstm in list_available_lstm_types():
        print(f"  - {lstm}")

    print("\nPossible Combinations:")
    cnns = list_available_cnn_backbones()
    lstms = list_available_lstm_types()
    print(f"  {len(cnns)} CNNs × {len(lstms)} LSTMs = {len(cnns) * len(lstms)} base combinations")
    print("  (Plus variations in LSTM hidden size, layers, pooling methods)")


def example_2_create_recommended_model():
    """Example 2: Create a recommended model."""
    print_section("Example 2: Create Recommended Model")

    print("Creating 'recommended_1' (ResNet34 + BiLSTM)...")
    model = create_model('recommended_1')

    # Get model info
    info = model.get_model_info()

    print(f"\nModel Information:")
    print(f"  Total parameters: {info['total_params']:,}")
    print(f"  CNN parameters: {info['cnn_params']:,}")
    print(f"  LSTM parameters: {info['lstm_params']:,}")
    print(f"  Other parameters: {info['other_params']:,}")
    print(f"  Model size: {info['model_size_mb']:.2f} MB")

    # Test forward pass
    print("\nTesting forward pass...")
    device = get_device()
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(4, 1, 102400).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input shape: {tuple(dummy_input.shape)}")
    print(f"  Output shape: {tuple(output.shape)}")
    print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")


def example_3_create_custom_model():
    """Example 3: Create a custom hybrid model."""
    print_section("Example 3: Create Custom Hybrid Model")

    print("Creating custom hybrid:")
    print("  CNN: EfficientNet-B2")
    print("  LSTM: BiLSTM")
    print("  LSTM Hidden Size: 256")
    print("  LSTM Layers: 2")
    print("  Pooling: Attention\n")

    model = create_model(
        'custom',
        cnn_type='efficientnet_b2',
        lstm_type='bilstm',
        lstm_hidden_size=256,
        lstm_num_layers=2,
        pooling_method='attention'
    )

    # Get model info
    info = model.get_model_info()

    print(f"Model Information:")
    print(f"  Total parameters: {info['total_params']:,}")
    print(f"  Model size: {info['model_size_mb']:.2f} MB")

    # Test forward pass
    print("\nTesting forward pass...")
    device = get_device()
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(2, 1, 102400).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input shape: {tuple(dummy_input.shape)}")
    print(f"  Output shape: {tuple(output.shape)}")


def example_4_data_loading():
    """Example 4: Load data."""
    print_section("Example 4: Load Data")

    data_dir = project_root / 'data' / 'raw' / 'bearing_data'

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        print("Please create the directory and add .mat files.")
        return

    print(f"Loading data from: {data_dir}")

    train_loader, val_loader, test_loader = create_cnn_dataloaders(
        data_dir=str(data_dir),
        batch_size=32,
        num_workers=4,
        random_seed=42
    )

    print(f"\nDataset sizes:")
    print(f"  Training: {len(train_loader.dataset)} samples")
    print(f"  Validation: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")

    print(f"\nNumber of batches (batch_size=32):")
    print(f"  Training: {len(train_loader)} batches")
    print(f"  Validation: {len(val_loader)} batches")
    print(f"  Test: {len(test_loader)} batches")

    # Get a batch
    print("\nLoading a batch...")
    for signals, labels in train_loader:
        print(f"  Signals shape: {tuple(signals.shape)}")
        print(f"  Labels shape: {tuple(labels.shape)}")
        print(f"  Signal range: [{signals.min().item():.3f}, {signals.max().item():.3f}]")
        print(f"  Unique labels in batch: {torch.unique(labels).tolist()}")
        break


def example_5_training_setup():
    """Example 5: Set up training (without actually training)."""
    print_section("Example 5: Training Setup")

    # Set seed for reproducibility
    set_seed(42)

    # Create model
    print("Creating model...")
    model = create_model('recommended_1')

    # Create optimizer and loss
    print("Creating optimizer and loss function...")
    optimizer = create_optimizer(model.parameters(), 'adam', lr=0.001)
    criterion = create_loss_function('cross_entropy', num_classes=11)

    print(f"  Optimizer: {optimizer.__class__.__name__}")
    print(f"  Loss function: {criterion.__class__.__name__}")

    # Create scheduler
    print("\nCreating learning rate scheduler...")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=75)
    print(f"  Scheduler: {scheduler.__class__.__name__}")

    # Get device
    device = get_device()
    print(f"\nDevice: {device}")

    # Create dummy dataloaders for demonstration
    print("\nCreating dummy dataloaders for demonstration...")
    from torch.utils.data import TensorDataset, DataLoader
    dummy_signals = torch.randn(32, 1, 102400)
    dummy_labels = torch.randint(0, 11, (32,))
    dummy_dataset = TensorDataset(dummy_signals, dummy_labels)
    dummy_train_loader = DataLoader(dummy_dataset, batch_size=8)
    dummy_val_loader = DataLoader(dummy_dataset, batch_size=8)
    print("  Dummy dataloaders created (32 samples)")

    # Create trainer
    print("\nCreating trainer...")
    trainer = CNNTrainer(
        model=model,
        train_loader=dummy_train_loader,
        val_loader=dummy_val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        lr_scheduler=scheduler,
        checkpoint_dir=Path('results/checkpoints/hybrid/example'),
        mixed_precision=True
    )

    print("  Trainer created successfully!")
    print("\nTo start training, you would call:")
    print("  trainer.fit(num_epochs=75, save_best=True, verbose=True)")


def example_6_model_inference():
    """Example 6: Model inference."""
    print_section("Example 6: Model Inference")

    # Create model
    print("Loading model...")
    model = create_model('recommended_1')
    device = get_device()
    model = model.to(device)
    model.eval()

    # Create dummy signal (in practice, load from data)
    print("\nCreating dummy signal...")
    signal = torch.randn(1, 1, 102400).to(device)  # Single sample
    print(f"  Signal shape: {tuple(signal.shape)}")

    # Inference
    print("\nRunning inference...")
    with torch.no_grad():
        logits = model(signal)
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1)

    print(f"  Logits shape: {tuple(logits.shape)}")
    print(f"  Predicted class: {prediction.item()}")
    print(f"  Confidence: {probabilities[0, prediction].item():.4f}")

    # Show top-3 predictions
    top3_probs, top3_classes = torch.topk(probabilities, 3, dim=1)
    print("\n  Top-3 predictions:")
    for i in range(3):
        print(f"    Class {top3_classes[0, i].item()}: {top3_probs[0, i].item():.4f}")


def example_7_compare_models():
    """Example 7: Compare different configurations."""
    print_section("Example 7: Compare Model Configurations")

    configs = [
        ('recommended_1', 'ResNet34 + BiLSTM'),
        ('recommended_2', 'EfficientNet-B2 + BiLSTM'),
        ('recommended_3', 'ResNet18 + LSTM'),
    ]

    print(f"{'Model':<20} {'Description':<30} {'Parameters':>15} {'Size (MB)':>12}")
    print("-" * 80)

    for model_name, description in configs:
        model = create_model(model_name)
        info = model.get_model_info()

        print(f"{model_name:<20} {description:<30} {info['total_params']:>15,} {info['model_size_mb']:>12.2f}")
        del model  # Free memory


def example_8_custom_configurations():
    """Example 8: Try different custom configurations."""
    print_section("Example 8: Custom Configuration Examples")

    custom_configs = [
        {
            'name': 'Lightweight',
            'cnn_type': 'cnn1d',
            'lstm_type': 'lstm',
            'lstm_hidden_size': 128,
        },
        {
            'name': 'High Capacity',
            'cnn_type': 'resnet50',
            'lstm_type': 'bilstm',
            'lstm_hidden_size': 512,
        },
        {
            'name': 'Efficient',
            'cnn_type': 'efficientnet_b0',
            'lstm_type': 'bilstm',
            'lstm_hidden_size': 256,
        },
    ]

    print(f"{'Configuration':<20} {'CNN':<20} {'LSTM':<15} {'Params':>15} {'Size (MB)':>12}")
    print("-" * 85)

    for config in custom_configs:
        model = create_model(
            'custom',
            cnn_type=config['cnn_type'],
            lstm_type=config['lstm_type'],
            lstm_hidden_size=config['lstm_hidden_size']
        )
        info = model.get_model_info()

        lstm_desc = f"{config['lstm_type']} ({config['lstm_hidden_size']})"
        print(f"{config['name']:<20} {config['cnn_type']:<20} {lstm_desc:<15} {info['total_params']:>15,} {info['model_size_mb']:>12.2f}")
        del model


def main():
    """Main function."""
    print("\n" + "=" * 70)
    print("  CNN-LSTM Hybrid Models - Example Usage")
    print("=" * 70)

    try:
        # Example 1: List available models
        example_1_list_available_models()

        # Example 2: Create recommended model
        example_2_create_recommended_model()

        # Example 3: Create custom model
        example_3_create_custom_model()

        # Example 4: Data loading
        example_4_data_loading()

        # Example 5: Training setup
        example_5_training_setup()

        # Example 6: Model inference
        example_6_model_inference()

        # Example 7: Compare models
        example_7_compare_models()

        # Example 8: Custom configurations
        example_8_custom_configurations()

        # Final message
        print_section("Examples Complete")
        print("All examples ran successfully!")
        print("\nNext steps:")
        print("  1. Prepare your data in data/raw/bearing_data/")
        print("  2. Train a model:")
        print("     python scripts/train_hybrid.py --model recommended_1 --epochs 75")
        print("  3. Evaluate the model:")
        print("     python scripts/evaluate_hybrid.py --checkpoint [path] --model recommended_1")
        print("\nFor more details, see README.md and QUICKSTART.md\n")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
