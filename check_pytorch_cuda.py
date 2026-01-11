#!/usr/bin/env python3
"""
Diagnostic script to check PyTorch and CUDA installation status.
"""

import sys

print("=" * 70)
print("PyTorch and CUDA Diagnostic Check")
print("=" * 70)
print()

# Check if PyTorch is installed
try:
    import torch
    print(f"✓ PyTorch is installed")
    print(f"  Version: {torch.__version__}")
    print()
except ImportError:
    print("✗ PyTorch is NOT installed")
    sys.exit(1)

# Check CUDA availability
print("CUDA Status:")
print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
print()

if torch.cuda.is_available():
    print("✓ CUDA is available in PyTorch")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"  Number of GPUs: {torch.cuda.device_count()}")
    print()
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name}")
        print(f"    Memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
        print()
else:
    print("✗ CUDA is NOT available in PyTorch")
    print()
    print("Possible reasons:")
    print("  1. PyTorch was installed with CPU-only version")
    print("  2. CUDA libraries are not properly installed")
    print("  3. CUDA version mismatch between PyTorch and system")
    print()
    
    # Check if PyTorch build has CUDA support
    print("PyTorch build information:")
    print(f"  Version: {torch.__version__}")
    if hasattr(torch.version, 'cuda') and torch.version.cuda:
        print(f"  Built with CUDA: {torch.version.cuda}")
        print("  → PyTorch was built with CUDA support but can't access it")
        print("  → This usually means CUDA runtime libraries are missing or incompatible")
    else:
        print("  Built with CUDA: None (CPU-only build)")
        print("  → PyTorch was installed without CUDA support")
        print("  → Need to reinstall with: pip install torch --index-url https://download.pytorch.org/whl/cu118")
    print()

# Check system GPU
print("System GPU Detection:")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("✓ nvidia-smi is available")
        # Extract GPU name from nvidia-smi output
        lines = result.stdout.split('\n')
        for line in lines:
            if 'NVIDIA' in line and ('GeForce' in line or 'Quadro' in line or 'Tesla' in line or 'RTX' in line or 'GTX' in line):
                print(f"  GPU detected: {line.strip()}")
                break
    else:
        print("✗ nvidia-smi returned error")
except FileNotFoundError:
    print("✗ nvidia-smi not found (NVIDIA drivers may not be installed)")
except subprocess.TimeoutExpired:
    print("✗ nvidia-smi timed out")
except Exception as e:
    print(f"✗ Error running nvidia-smi: {e}")

print()
print("=" * 70)
print("Recommendations:")
print("=" * 70)

if not torch.cuda.is_available():
    if hasattr(torch.version, 'cuda') and torch.version.cuda:
        print("1. PyTorch has CUDA support but can't access GPU")
        print("   → Check NVIDIA drivers: nvidia-smi should work")
        print("   → Verify CUDA toolkit is installed")
        print("   → Try reinstalling PyTorch: pip uninstall torch torchvision torchaudio -y")
        print("   → Then: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    else:
        print("1. PyTorch was installed without CUDA support")
        print("   → Uninstall: pip uninstall torch torchvision torchaudio -y")
        print("   → Reinstall with CUDA: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
else:
    print("✓ Everything looks good! GPU should be available for training.")

print()

