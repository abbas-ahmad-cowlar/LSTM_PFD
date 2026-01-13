"""
Seed management for reproducibility.

Usage:
    from reproducibility.scripts.set_seeds import set_all_seeds
    set_all_seeds(42)

Reference: Master Roadmap Chapter 3.4
"""

import random
import os

import numpy as np
import torch

MASTER_SEED = 42


def set_all_seeds(seed: int = MASTER_SEED) -> None:
    """
    Set all random seeds for full reproducibility.
    
    Warning: Setting cudnn.deterministic=True may reduce training speed by 10-20%.
    
    Args:
        seed: Random seed to use (default: 42)
    """
    # Python
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # cuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # PyTorch 2.0+ deterministic algorithms
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            # Some operations don't have deterministic implementations
            pass


def get_seed() -> int:
    """Return the master seed value."""
    return MASTER_SEED


if __name__ == '__main__':
    set_all_seeds()
    print(f"All seeds set to {MASTER_SEED}")
    print(f"PyTorch: {torch.initial_seed()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"cuDNN deterministic: {torch.backends.cudnn.deterministic}")
