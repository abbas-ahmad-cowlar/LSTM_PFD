"""
Resource Profiling Utilities

Profile computational resources (GPU, CPU, memory) during training and inference.

Author: Syed Abbas Ahmad
Date: 2025-11-23
"""

import time
import psutil
import numpy as np
from typing import Dict, List, Optional, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def profile_gpu_utilization(
    training_fn: Callable,
    duration_seconds: int = 60
) -> Dict[str, float]:
    """
    Profile GPU utilization during training.

    Monitors GPU usage over time to ensure efficient hardware utilization.
    Target: >80% GPU utilization for efficient training.

    Args:
        training_fn: Function that performs training
        duration_seconds: How long to monitor (seconds)

    Returns:
        Dictionary with GPU utilization statistics

    Example:
        >>> def train():
        ...     # Training loop
        ...     for epoch in range(10):
        ...         train_epoch(model, dataloader)
        >>>
        >>> stats = profile_gpu_utilization(train, duration_seconds=30)
        >>> print(f"Average GPU utilization: {stats['avg_utilization']:.1f}%")

    Requires:
        - pynvml (pip install nvidia-ml-py3)
        - NVIDIA GPU with CUDA
    """
    logger.info("="*60)
    logger.info("GPU Utilization Profiling")
    logger.info("="*60)

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        logger.info(f"Monitoring GPU for {duration_seconds} seconds...")

        gpu_utils = []
        gpu_memory_used = []
        gpu_memory_total = []

        start_time = time.time()

        # Start training in background
        import threading
        training_thread = threading.Thread(target=training_fn)
        training_thread.daemon = True
        training_thread.start()

        # Monitor GPU
        while time.time() - start_time < duration_seconds:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            gpu_utils.append(util.gpu)
            gpu_memory_used.append(mem_info.used / (1024**3))  # GB
            gpu_memory_total.append(mem_info.total / (1024**3))  # GB

            time.sleep(0.5)  # Sample every 0.5s

        pynvml.nvmlShutdown()

        # Calculate statistics
        avg_utilization = np.mean(gpu_utils)
        min_utilization = np.min(gpu_utils)
        max_utilization = np.max(gpu_utils)
        avg_memory = np.mean(gpu_memory_used)
        peak_memory = np.max(gpu_memory_used)

        logger.info(f"\nGPU Utilization Statistics:")
        logger.info(f"  Average: {avg_utilization:.1f}%")
        logger.info(f"  Minimum: {min_utilization:.1f}%")
        logger.info(f"  Maximum: {max_utilization:.1f}%")
        logger.info(f"\nGPU Memory Statistics:")
        logger.info(f"  Average: {avg_memory:.2f} GB")
        logger.info(f"  Peak: {peak_memory:.2f} GB")
        logger.info(f"  Total: {gpu_memory_total[0]:.2f} GB")

        # Assessment
        if avg_utilization > 80:
            logger.info("\n✓ Excellent GPU utilization (>80%)")
        elif avg_utilization > 60:
            logger.info("\n✓ Good GPU utilization (>60%)")
        elif avg_utilization > 40:
            logger.warning("\n⚠ Moderate GPU utilization (40-60%)")
            logger.warning("  Consider increasing batch size or reducing CPU bottlenecks")
        else:
            logger.warning("\n✗ Poor GPU utilization (<40%)")
            logger.warning("  GPU is underutilized. Check for:")
            logger.warning("    - Data loading bottleneck (increase num_workers)")
            logger.warning("    - Small batch size (increase if memory allows)")
            logger.warning("    - CPU-bound preprocessing")

        return {
            'avg_utilization': avg_utilization,
            'min_utilization': min_utilization,
            'max_utilization': max_utilization,
            'avg_memory_gb': avg_memory,
            'peak_memory_gb': peak_memory,
            'total_memory_gb': gpu_memory_total[0],
            'utilization_samples': gpu_utils
        }

    except ImportError:
        logger.error("pynvml not installed. Install with: pip install nvidia-ml-py3")
        return {'error': 'pynvml not available'}
    except Exception as e:
        logger.error(f"GPU profiling failed: {e}")
        return {'error': str(e)}


def profile_memory_footprint(
    model,
    input_shape: tuple = (32, 1, 102400)
) -> Dict[str, float]:
    """
    Profile model memory footprint.

    Measures memory used by:
    - Model parameters
    - Optimizer state
    - Activations/gradients during forward/backward pass

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch_size, channels, length)

    Returns:
        Dictionary with memory statistics (MB)

    Example:
        >>> import torch
        >>> model = create_model('resnet34')
        >>> stats = profile_memory_footprint(model)
        >>> print(f"Total memory: {stats['total_mb']:.1f} MB")
    """
    import torch
    import os

    logger.info("="*60)
    logger.info("Memory Footprint Profiling")
    logger.info("="*60)

    process = psutil.Process(os.getpid())

    # Measure initial memory
    initial_memory = process.memory_info().rss / (1024**2)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    after_load_memory = process.memory_info().rss / (1024**2)
    model_memory = after_load_memory - initial_memory

    # Run forward pass
    dummy_input = torch.randn(*input_shape, device=device)
    output = model(dummy_input)

    after_forward_memory = process.memory_info().rss / (1024**2)
    forward_memory = after_forward_memory - after_load_memory

    # Run backward pass (if model has requires_grad)
    if any(p.requires_grad for p in model.parameters()):
        loss = output.sum()
        loss.backward()

        after_backward_memory = process.memory_info().rss / (1024**2)
        backward_memory = after_backward_memory - after_forward_memory
    else:
        backward_memory = 0

    total_memory = after_backward_memory if backward_memory > 0 else after_forward_memory

    logger.info(f"\nMemory Breakdown:")
    logger.info(f"  Model parameters: {model_memory:.1f} MB")
    logger.info(f"  Forward pass activations: {forward_memory:.1f} MB")
    logger.info(f"  Backward pass gradients: {backward_memory:.1f} MB")
    logger.info(f"  Total: {total_memory:.1f} MB")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"\nParameter Count:")
    logger.info(f"  Total: {total_params/1e6:.2f}M")
    logger.info(f"  Trainable: {trainable_params/1e6:.2f}M")

    return {
        'model_memory_mb': model_memory,
        'forward_memory_mb': forward_memory,
        'backward_memory_mb': backward_memory,
        'total_memory_mb': total_memory,
        'total_params': total_params,
        'trainable_params': trainable_params
    }


def profile_cpu_efficiency(
    data_loading_fn: Callable,
    duration_seconds: int = 30
) -> Dict[str, float]:
    """
    Profile CPU efficiency during data loading.

    Identifies CPU bottlenecks that may slow down training.

    Args:
        data_loading_fn: Function that loads data
        duration_seconds: How long to monitor

    Returns:
        Dictionary with CPU utilization statistics

    Example:
        >>> def load_data():
        ...     for batch in dataloader:
        ...         pass
        >>>
        >>> stats = profile_cpu_efficiency(load_data, duration_seconds=20)
        >>> print(f"CPU utilization: {stats['avg_cpu_percent']:.1f}%")
    """
    logger.info("="*60)
    logger.info("CPU Efficiency Profiling")
    logger.info("="*60)

    import threading

    cpu_percents = []

    def monitor_cpu():
        start = time.time()
        while time.time() - start < duration_seconds:
            cpu_percents.append(psutil.cpu_percent(interval=0.5))

    # Start monitoring
    monitor_thread = threading.Thread(target=monitor_cpu)
    monitor_thread.daemon = True
    monitor_thread.start()

    # Run data loading
    start_time = time.time()
    data_loading_fn()
    actual_duration = time.time() - start_time

    # Wait for monitoring to finish
    monitor_thread.join()

    avg_cpu = np.mean(cpu_percents)
    max_cpu = np.max(cpu_percents)

    logger.info(f"\nCPU Utilization:")
    logger.info(f"  Average: {avg_cpu:.1f}%")
    logger.info(f"  Maximum: {max_cpu:.1f}%")
    logger.info(f"  Duration: {actual_duration:.1f}s")

    # Get per-core utilization
    per_core = psutil.cpu_percent(interval=1, percpu=True)
    logger.info(f"  Cores: {len(per_core)}")
    logger.info(f"  Per-core avg: {np.mean(per_core):.1f}%")

    return {
        'avg_cpu_percent': avg_cpu,
        'max_cpu_percent': max_cpu,
        'num_cores': len(per_core),
        'per_core_avg': np.mean(per_core)
    }
