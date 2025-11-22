#!/usr/bin/env python3
"""
Installation Validation - LSTM Milestone

Verifies all dependencies are installed correctly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "="*70)
print("  LSTM Milestone - Installation Validation")
print("="*70 + "\n")

# Check Python version
print("1. Checking Python version...")
if sys.version_info >= (3, 8):
    print(f"   ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}\n")
else:
    print(f"   ❌ Python 3.8+ required\n")
    sys.exit(1)

# Check dependencies
print("2. Checking dependencies...")
deps = [
    ('torch', 'PyTorch'),
    ('numpy', 'NumPy'),
    ('scipy', 'SciPy'),
    ('sklearn', 'scikit-learn'),
    ('matplotlib', 'Matplotlib'),
]

for pkg, name in deps:
    try:
        __import__(pkg)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ❌ {name} - NOT FOUND")

# Check PyTorch & CUDA
print("\n3. Checking PyTorch configuration...")
import torch
print(f"   ✓ PyTorch {torch.__version__}")
if torch.cuda.is_available():
    print(f"   ✓ CUDA available: {torch.version.cuda}")
    print(f"   ✓ GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"   ⚠ CUDA not available (CPU-only mode)")

# Check project structure
print("\n4. Checking project structure...")
dirs = ['data', 'models', 'training', 'utils', 'scripts']
for d in dirs:
    if Path(d).exists():
        print(f"   ✓ {d}/")
    else:
        print(f"   ❌ {d}/ - NOT FOUND")

# Test model creation
print("\n5. Testing model creation...")
try:
    from models import create_model
    model = create_model('vanilla_lstm', num_classes=11)
    print(f"   ✓ Vanilla LSTM created")
    model = create_model('bilstm', num_classes=11)
    print(f"   ✓ BiLSTM created")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*70)
print("✓ Installation validation complete!")
print("="*70 + "\n")
print("Next steps:")
print("  1. Prepare .MAT data files")
print("  2. Run: python example_usage.py")
print("  3. Train: python scripts/train_lstm.py --help\n")
