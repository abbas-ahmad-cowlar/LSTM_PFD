#!/usr/bin/env python3
"""
Installation Validation Script

Validates that the Milestone 3 installation is correct and ready to use.

Usage:
    python scripts/validate_installation.py

Author: Bearing Fault Diagnosis Team
Milestone: 3 - CNN-LSTM Hybrid
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 65)
    print(f"  {text}")
    print("=" * 65 + "\n")


def print_check(text, status=True):
    """Print a check item."""
    symbol = "✓" if status else "✗"
    print(f"{symbol} {text}")


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    is_valid = version.major == 3 and version.minor >= 8
    print_check(f"Python version: {version.major}.{version.minor}.{version.micro}", is_valid)
    return is_valid


def check_pytorch():
    """Check PyTorch installation."""
    try:
        version = torch.__version__
        print_check(f"PyTorch version: {version}", True)
        return True
    except Exception as e:
        print_check(f"PyTorch not found: {e}", False)
        return False


def check_cuda():
    """Check CUDA availability."""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        print_check(f"CUDA available: True", True)
        print_check(f"GPU: {gpu_name}", True)
    else:
        print_check("CUDA available: False (CPU mode)", True)
    return True


def check_packages():
    """Check required packages."""
    required_packages = [
        ('numpy', 'np'),
        ('scipy', 'scipy'),
        ('sklearn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'sns'),
        ('tqdm', 'tqdm'),
    ]

    all_ok = True
    for pkg_name, import_name in required_packages:
        try:
            __import__(import_name)
            print_check(f"{pkg_name} installed", True)
        except ImportError:
            print_check(f"{pkg_name} not found", False)
            all_ok = False

    return all_ok


def check_data_directory():
    """Check data directory structure."""
    data_dir = project_root / 'data' / 'raw' / 'bearing_data'

    if not data_dir.exists():
        print_check(f"Data directory not found: {data_dir}", False)
        print("  → Please create data/raw/bearing_data/ and add .mat files")
        return False

    print_check("Data directory exists", True)

    # Count .mat files
    mat_files = list(data_dir.glob('*.mat'))
    num_files = len(mat_files)

    if num_files == 0:
        print_check("No .mat files found", False)
        print("  → Please add .mat files to data/raw/bearing_data/")
        return False

    print_check(f"Found {num_files} .mat files", True)
    return True


def check_model_creation():
    """Test model creation."""
    try:
        from models import create_model, list_available_cnn_backbones, list_available_lstm_types

        print_check("Model imports successful", True)

        # Test recommended models
        models_to_test = [
            ('recommended_1', 'ResNet34+BiLSTM'),
            ('recommended_2', 'EfficientNet-B2+BiLSTM'),
            ('recommended_3', 'ResNet18+LSTM'),
        ]

        all_ok = True
        for model_name, description in models_to_test:
            try:
                model = create_model(model_name)
                print_check(f"  {model_name} ({description}): OK", True)
                del model  # Free memory
            except Exception as e:
                print_check(f"  {model_name} ({description}): FAIL - {e}", False)
                all_ok = False

        # Test custom model
        try:
            model = create_model('custom', cnn_type='resnet34', lstm_type='bilstm')
            print_check(f"  custom (ResNet34+BiLSTM): OK", True)
            del model
        except Exception as e:
            print_check(f"  custom: FAIL - {e}", False)
            all_ok = False

        return all_ok

    except Exception as e:
        print_check(f"Model creation failed: {e}", False)
        return False


def check_data_loading():
    """Test data loading."""
    try:
        from data.cnn_dataloader import create_cnn_dataloaders
        from data.matlab_importer import MatlabImporter
        from data.cnn_dataset import create_cnn_datasets_from_arrays

        data_dir = project_root / 'data' / 'raw' / 'bearing_data'

        if not data_dir.exists() or len(list(data_dir.glob('*.mat'))) == 0:
            print_check("Skipping data loading test (no data)", True)
            return True

        # Load .mat files
        importer = MatlabImporter()
        batch_data = importer.load_batch(data_dir, pattern='*.mat')
        signals, labels = importer.extract_signals_and_labels(batch_data)

        # Create datasets
        train_dataset, val_dataset, test_dataset = create_cnn_datasets_from_arrays(
            signals=signals,
            labels=labels,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            augment_train=False,
            random_seed=42
        )

        # Create dataloaders
        loaders = create_cnn_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            batch_size=4,
            num_workers=0
        )

        train_loader = loaders['train']
        val_loader = loaders['val']
        test_loader = loaders['test']

        print_check("Data loading successful", True)
        print_check(f"  Train loader: {len(train_loader.dataset)} samples", True)
        print_check(f"  Val loader: {len(val_loader.dataset)} samples", True)
        print_check(f"  Test loader: {len(test_loader.dataset)} samples", True)

        return True

    except Exception as e:
        print_check(f"Data loading failed: {e}", False)
        return False


def check_forward_pass():
    """Test forward pass."""
    try:
        from models import create_model

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = create_model('recommended_1')
        model = model.to(device)
        model.eval()

        # Create dummy input
        batch_size = 4
        signal_length = 102400
        dummy_input = torch.randn(batch_size, 1, signal_length).to(device)

        # Forward pass
        with torch.no_grad():
            output = model(dummy_input)

        expected_shape = (batch_size, 11)  # 11 classes
        actual_shape = tuple(output.shape)

        if actual_shape == expected_shape:
            print_check(f"Forward pass successful", True)
            print_check(f"  Input shape: {tuple(dummy_input.shape)}", True)
            print_check(f"  Output shape: {actual_shape}", True)
            return True
        else:
            print_check(f"Forward pass shape mismatch: expected {expected_shape}, got {actual_shape}", False)
            return False

    except Exception as e:
        print_check(f"Forward pass failed: {e}", False)
        return False


def main():
    """Main validation function."""
    print_header("Milestone 3 Installation Validation")

    results = []

    # Check Python version
    results.append(check_python_version())

    # Check PyTorch
    results.append(check_pytorch())

    # Check CUDA
    results.append(check_cuda())

    # Check packages
    print()
    results.append(check_packages())

    # Check data directory
    print()
    results.append(check_data_directory())

    # Check model creation
    print()
    print_check("Testing model creation...", True)
    results.append(check_model_creation())

    # Check data loading
    print()
    print_check("Testing data loading...", True)
    results.append(check_data_loading())

    # Check forward pass
    print()
    print_check("Testing forward pass...", True)
    results.append(check_forward_pass())

    # Summary
    print_header("Validation Summary")

    if all(results):
        print("✓ All checks passed!")
        print("\nYou're ready to train hybrid models!")
        print("\nQuick start:")
        print("  python scripts/train_hybrid.py --model recommended_1 --epochs 75")
        return 0
    else:
        print("✗ Some checks failed. Please review the errors above.")
        print("\nCommon solutions:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Create data directory: mkdir -p data/raw/bearing_data")
        print("  - Add .mat files to data/raw/bearing_data/")
        return 1


if __name__ == "__main__":
    sys.exit(main())
