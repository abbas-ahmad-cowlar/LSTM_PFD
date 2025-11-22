"""
Reproducibility utilities for deterministic experiments.

Purpose:
    Ensure reproducible results by controlling random seeds across:
    - Python random module
    - NumPy random number generation
    - PyTorch CPU and CUDA operations

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import random
import numpy as np
import os
from typing import Optional


def set_seed(seed: int) -> None:
    """
    Set random seed for all libraries.

    Args:
        seed: Random seed value (typically 42 or from config)

    Example:
        >>> set_seed(42)
        >>> # All subsequent random operations are deterministic
    """
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (import only if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
    except ImportError:
        pass  # PyTorch not installed yet in Phase 0

    # Environment variable for Python hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)


def make_deterministic(warn: bool = True) -> None:
    """
    Enable deterministic mode for PyTorch operations.

    This disables non-deterministic algorithms (e.g., CuDNN autotuner)
    for fully reproducible results at potential performance cost.

    Args:
        warn: Whether to warn about performance implications

    Warning:
        This may reduce training performance by 10-20% due to
        disabling optimized non-deterministic algorithms.

    Example:
        >>> set_seed(42)
        >>> make_deterministic()
        >>> # All PyTorch operations are now deterministic
    """
    try:
        import torch

        # Disable CuDNN non-deterministic algorithms
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Use deterministic algorithms where available
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)

        if warn:
            print("⚠️  Deterministic mode enabled. Performance may be reduced by 10-20%.")

    except ImportError:
        if warn:
            print("⚠️  PyTorch not available. Deterministic mode only applies to NumPy.")


def get_random_state() -> dict:
    """
    Capture current random state for all RNGs.

    Returns:
        Dictionary containing random states for reproducibility

    Example:
        >>> state = get_random_state()
        >>> # ... do some random operations ...
        >>> restore_random_state(state)  # Restore to previous state
    """
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
    }

    try:
        import torch
        state['torch'] = torch.random.get_rng_state()
        if torch.cuda.is_available():
            state['torch_cuda'] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass

    return state


def restore_random_state(state: dict) -> None:
    """
    Restore random state from saved state dictionary.

    Args:
        state: Dictionary from get_random_state()

    Example:
        >>> state = get_random_state()
        >>> # ... random operations ...
        >>> restore_random_state(state)
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])

    if 'torch' in state:
        try:
            import torch
            torch.random.set_rng_state(state['torch'])
            if 'torch_cuda' in state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(state['torch_cuda'])
        except ImportError:
            pass
