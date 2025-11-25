#!/usr/bin/env python3
"""
Installation Validation Script

This script verifies that all dependencies are correctly installed
and that the package structure is valid.

Run this after installation to ensure everything is working.

Usage:
    python validate_installation.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def check_python_version():
    """Verify Python version >= 3.8"""
    print("Checking Python version...", end=" ")
    if sys.version_info < (3, 8):
        print(f"❌ FAILED")
        print(f"   Python 3.8+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        return False
    print(f"✓ OK ({sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro})")
    return True


def check_dependencies():
    """Check if all required packages are installed"""
    print("\nChecking dependencies...")

    required_packages = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('h5py', 'h5py'),
        ('pywt', 'PyWavelets'),
        ('tqdm', 'tqdm'),
        ('joblib', 'joblib'),
    ]

    all_ok = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ❌ {name} - NOT FOUND")
            all_ok = False

    return all_ok


def check_pytorch_cuda():
    """Check PyTorch and CUDA availability"""
    print("\nChecking PyTorch configuration...")

    try:
        import torch

        print(f"  ✓ PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: Yes")
            print(f"    - CUDA version: {torch.version.cuda}")
            print(f"    - GPU device: {torch.cuda.get_device_name(0)}")
            print(f"    - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print(f"  ⚠ CUDA available: No (CPU-only mode)")
            print(f"    Training will be slower without GPU")

        return True
    except Exception as e:
        print(f"  ❌ Error checking PyTorch: {e}")
        return False


def check_project_structure():
    """Verify project directory structure"""
    print("\nChecking project structure...")

    required_dirs = [
        'data',
        'models',
        'models/cnn',
        'models/resnet',
        'models/efficientnet',
        'training',
        'utils',
        'scripts',
        'visualization',
        'results',
    ]

    all_ok = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ - NOT FOUND")
            all_ok = False

    return all_ok


def check_module_imports():
    """Test importing key modules"""
    print("\nChecking module imports...")

    modules_to_test = [
        ('utils.constants', 'Constants'),
        ('utils.device_manager', 'Device Manager'),
        ('data.matlab_importer', 'MAT Importer'),
        ('data.cnn_dataset', 'CNN Dataset'),
        ('models.cnn.cnn_1d', 'CNN Models'),
        ('models.resnet.resnet_1d', 'ResNet Models'),
        ('models.efficientnet.efficientnet_1d', 'EfficientNet Models'),
        ('training.cnn_trainer', 'CNN Trainer'),
        ('visualization.performance_plots', 'Visualization'),
    ]

    all_ok = True
    for module_path, name in modules_to_test:
        try:
            __import__(module_path)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ❌ {name} - IMPORT ERROR")
            print(f"     Error: {e}")
            all_ok = False
        except Exception as e:
            print(f"  ⚠ {name} - WARNING: {e}")

    return all_ok


def check_model_creation():
    """Test creating models"""
    print("\nTesting model creation...")

    try:
        from models import create_model

        test_models = ['cnn1d', 'resnet18', 'efficientnet_b0']

        for model_name in test_models:
            try:
                model = create_model(model_name, num_classes=11)
                param_count = sum(p.numel() for p in model.parameters())
                print(f"  ✓ {model_name} ({param_count:,} params)")
            except Exception as e:
                print(f"  ❌ {model_name} - ERROR: {e}")
                return False

        return True

    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        return False


def main():
    """Run all validation checks"""
    print("\n" + "="*70)
    print("  CNN Bearing Fault Diagnosis - Installation Validation")
    print("="*70 + "\n")

    results = []

    # Run all checks
    results.append(("Python Version", check_python_version()))
    results.append(("Dependencies", check_dependencies()))
    results.append(("PyTorch & CUDA", check_pytorch_cuda()))
    results.append(("Project Structure", check_project_structure()))
    results.append(("Module Imports", check_module_imports()))
    results.append(("Model Creation", check_model_creation()))

    # Summary
    print("\n" + "="*70)
    print("  Validation Summary")
    print("="*70 + "\n")

    all_passed = True
    for check_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {check_name:<25} {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*70)

    if all_passed:
        print("\n✓ All checks passed! Installation is successful.\n")
        print("Next steps:")
        print("  1. Prepare your .MAT data files (see README.md)")
        print("  2. Run: python example_usage.py")
        print("  3. Train a model: python scripts/train_cnn.py --help")
        print("\n")
        return 0
    else:
        print("\n❌ Some checks failed. Please fix the issues above.\n")
        print("Common solutions:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Reinstall PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("  - Check Python version: python --version (need 3.8+)")
        print("\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
