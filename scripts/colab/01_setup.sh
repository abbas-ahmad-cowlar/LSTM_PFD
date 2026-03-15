#!/bin/bash
# ============================================================
# Colab Setup Script
# Run this cell first in your Colab notebook:
#   !bash scripts/colab/01_setup.sh
# ============================================================

set -e

echo "============================================================"
echo "STEP 1: Verifying GPU"
echo "============================================================"
python -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    print(f'  CUDA: {torch.version.cuda}')
else:
    print('  WARNING: No GPU detected! Training will be very slow.')
print(f'  PyTorch: {torch.__version__}')
"

echo ""
echo "============================================================"
echo "STEP 2: Installing dependencies"
echo "============================================================"
pip install -q pywt h5py tqdm scipy scikit-learn 2>/dev/null
echo "  [OK] Dependencies installed"

echo ""
echo "============================================================"
echo "STEP 3: Verifying project structure"
echo "============================================================"
python -c "
from pathlib import Path
root = Path('.')
checks = [
    'packages/core/models/model_factory.py',
    'data/signal_generation/generator.py',
    'utils/constants.py',
    'scripts/colab/_train_utils.py',
]
all_ok = True
for f in checks:
    exists = (root / f).exists()
    status = '[OK]' if exists else '[MISSING]'
    print(f'  {status} {f}')
    if not exists:
        all_ok = False
if all_ok:
    print('\n  All checks passed!')
else:
    print('\n  ERROR: Some files are missing. Did you clone the right branch?')
    exit(1)
"

echo ""
echo "============================================================"
echo "STEP 4: Creating output directories"
echo "============================================================"
mkdir -p data/generated checkpoints results logs
echo "  [OK] Directories ready"

echo ""
echo "============================================================"
echo "SETUP COMPLETE"
echo "============================================================"
echo "Next: Run 02_generate_data.py to generate the training dataset"
