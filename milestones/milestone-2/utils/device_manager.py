"""
GPU/CPU device management utilities.

Purpose:
    Centralized device management for PyTorch:
    - Automatic GPU detection
    - Device transfer utilities
    - Memory monitoring
    - Multi-GPU support

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import torch
from typing import Any, Union, List, Optional, Dict
import subprocess

from utils.logging import get_logger

logger = get_logger(__name__)


def get_device(prefer_gpu: bool = True, gpu_id: Optional[int] = None) -> torch.device:
    """
    Get available compute device (GPU or CPU).

    Args:
        prefer_gpu: Whether to use GPU if available
        gpu_id: Specific GPU ID to use (None = auto-select)

    Returns:
        torch.device object

    Example:
        >>> device = get_device(prefer_gpu=True)
        >>> print(device)  # cuda:0 or cpu
    """
    if prefer_gpu and torch.cuda.is_available():
        if gpu_id is not None:
            if gpu_id >= torch.cuda.device_count():
                logger.warning(
                    f"GPU {gpu_id} not available (only {torch.cuda.device_count()} GPUs). "
                    f"Using GPU 0."
                )
                gpu_id = 0
            device = torch.device(f'cuda:{gpu_id}')
        else:
            device = torch.device('cuda')

        gpu_name = torch.cuda.get_device_name(device)
        logger.info(f"Using device: {device} ({gpu_name})")
    else:
        device = torch.device('cpu')
        if prefer_gpu:
            logger.warning("GPU requested but not available. Using CPU.")
        else:
            logger.info("Using device: CPU")

    return device


def get_available_gpus() -> List[int]:
    """
    Get list of available GPU IDs.

    Returns:
        List of GPU IDs

    Example:
        >>> gpus = get_available_gpus()
        >>> print(f"Available GPUs: {gpus}")
    """
    if not torch.cuda.is_available():
        return []

    return list(range(torch.cuda.device_count()))


def get_gpu_info() -> Dict[str, Any]:
    """
    Get detailed GPU information.

    Returns:
        Dictionary with GPU details

    Example:
        >>> info = get_gpu_info()
        >>> print(f"GPU count: {info['gpu_count']}")
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'gpu_count': 0,
        'gpus': []
    }

    if torch.cuda.is_available():
        info['gpu_count'] = torch.cuda.device_count()
        info['cuda_version'] = torch.version.cuda
        info['cudnn_version'] = torch.backends.cudnn.version()

        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_info = {
                'id': i,
                'name': gpu_props.name,
                'total_memory_gb': gpu_props.total_memory / (1024**3),
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
            }
            info['gpus'].append(gpu_info)

    return info


def move_to_device(
    data: Any,
    device: torch.device,
    non_blocking: bool = False
) -> Any:
    """
    Recursively move data to device.

    Handles tensors, lists, tuples, and dictionaries.

    Args:
        data: Data to move (tensor, list, tuple, dict)
        device: Target device
        non_blocking: Async transfer (requires pinned memory)

    Returns:
        Data on target device

    Example:
        >>> batch = {'signals': tensor1, 'labels': tensor2}
        >>> batch = move_to_device(batch, device)
    """
    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device, non_blocking) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device, non_blocking) for item in data)
    else:
        # Non-tensor data (ints, strings, etc.)
        return data


def get_gpu_memory_usage(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Get current GPU memory usage.

    Args:
        device: GPU device (None = current device)

    Returns:
        Dictionary with memory stats in GB

    Example:
        >>> memory = get_gpu_memory_usage()
        >>> print(f"Allocated: {memory['allocated_gb']:.2f} GB")
    """
    if not torch.cuda.is_available():
        return {
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'free_gb': 0.0,
            'total_gb': 0.0
        }

    if device is None:
        device = torch.cuda.current_device()
    elif isinstance(device, torch.device):
        device = device.index if device.index is not None else 0

    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    free = total - allocated

    return {
        'allocated_gb': allocated,
        'reserved_gb': reserved,
        'free_gb': free,
        'total_gb': total,
        'utilization_pct': (allocated / total) * 100 if total > 0 else 0
    }


def clear_gpu_memory(device: Optional[torch.device] = None) -> None:
    """
    Clear GPU memory cache.

    Args:
        device: GPU device (None = all devices)

    Example:
        >>> clear_gpu_memory()
        >>> # Frees unused cached memory
    """
    if torch.cuda.is_available():
        if device is not None:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        else:
            torch.cuda.empty_cache()

        logger.debug("Cleared GPU memory cache")


def set_cuda_device(gpu_id: int) -> None:
    """
    Set active CUDA device.

    Args:
        gpu_id: GPU ID to use

    Example:
        >>> set_cuda_device(1)
        >>> # All subsequent operations use GPU 1
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, cannot set device")
        return

    if gpu_id >= torch.cuda.device_count():
        raise ValueError(
            f"GPU {gpu_id} not available. "
            f"Only {torch.cuda.device_count()} GPUs detected."
        )

    torch.cuda.set_device(gpu_id)
    logger.info(f"Set CUDA device to GPU {gpu_id}")


def enable_cudnn_benchmark(enable: bool = True) -> None:
    """
    Enable/disable cuDNN benchmark mode for faster training.

    Benchmark mode finds fastest algorithms for your specific input size.
    Use when input sizes are consistent.

    Args:
        enable: Whether to enable benchmark mode

    Example:
        >>> enable_cudnn_benchmark(True)
        >>> # Training will be faster with fixed input sizes
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = enable
        logger.info(f"cuDNN benchmark mode: {'enabled' if enable else 'disabled'}")
    else:
        logger.warning("CUDA not available, cuDNN settings ignored")


def enable_cudnn_deterministic(enable: bool = True) -> None:
    """
    Enable/disable deterministic cuDNN operations.

    Deterministic mode ensures reproducibility but may reduce performance.

    Args:
        enable: Whether to enable deterministic mode

    Example:
        >>> enable_cudnn_deterministic(True)
        >>> # Results will be reproducible but slower
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = enable
        logger.info(f"cuDNN deterministic mode: {'enabled' if enable else 'disabled'}")
    else:
        logger.warning("CUDA not available, cuDNN settings ignored")


def log_device_info() -> None:
    """
    Log comprehensive device information.

    Example:
        >>> log_device_info()
        # Logs GPU/CPU info to logger
    """
    logger.info("=" * 60)
    logger.info("DEVICE INFORMATION")
    logger.info("=" * 60)

    info = get_gpu_info()

    if info['cuda_available']:
        logger.info(f"CUDA Available: Yes")
        logger.info(f"CUDA Version: {info.get('cuda_version', 'N/A')}")
        logger.info(f"cuDNN Version: {info.get('cudnn_version', 'N/A')}")
        logger.info(f"GPU Count: {info['gpu_count']}")

        for gpu in info['gpus']:
            logger.info(f"\nGPU {gpu['id']}:")
            logger.info(f"  Name: {gpu['name']}")
            logger.info(f"  Memory: {gpu['total_memory_gb']:.2f} GB")
            logger.info(f"  Compute Capability: {gpu['compute_capability']}")

            # Get current memory usage
            mem = get_gpu_memory_usage(gpu['id'])
            logger.info(f"  Current Usage: {mem['allocated_gb']:.2f} GB "
                       f"({mem['utilization_pct']:.1f}%)")
    else:
        logger.info("CUDA Available: No")
        logger.info("Using CPU for computation")

    logger.info("=" * 60)


def get_optimal_num_workers() -> int:
    """
    Get optimal number of DataLoader workers based on CPU count.

    Returns:
        Recommended num_workers

    Example:
        >>> num_workers = get_optimal_num_workers()
        >>> loader = DataLoader(dataset, num_workers=num_workers)
    """
    import multiprocessing as mp

    cpu_count = mp.cpu_count()

    # Rule of thumb: use 4 workers, but don't exceed CPU count
    optimal = min(4, max(1, cpu_count // 2))

    logger.debug(f"Recommended num_workers: {optimal} (CPU count: {cpu_count})")

    return optimal


def synchronize_device(device: Optional[torch.device] = None) -> None:
    """
    Synchronize CUDA device (wait for all operations to complete).

    Useful for accurate timing measurements.

    Args:
        device: Device to synchronize (None = current device)

    Example:
        >>> synchronize_device()
        >>> # All GPU operations are now complete
    """
    if torch.cuda.is_available():
        if device is not None and device.type == 'cuda':
            torch.cuda.synchronize(device)
        else:
            torch.cuda.synchronize()


def get_nvidia_smi_info() -> Optional[str]:
    """
    Get nvidia-smi output (if available).

    Returns:
        nvidia-smi output string or None

    Example:
        >>> info = get_nvidia_smi_info()
        >>> if info:
        ...     print(info)
    """
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout
        else:
            return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


class DeviceManager:
    """
    Context manager for device operations.

    Automatically handles device placement and cleanup.

    Example:
        >>> with DeviceManager(prefer_gpu=True) as device:
        ...     model = MyModel().to(device)
        ...     # Model automatically on correct device
    """

    def __init__(self, prefer_gpu: bool = True, gpu_id: Optional[int] = None):
        """
        Initialize device manager.

        Args:
            prefer_gpu: Whether to prefer GPU
            gpu_id: Specific GPU ID to use
        """
        self.prefer_gpu = prefer_gpu
        self.gpu_id = gpu_id
        self.device = None

    def __enter__(self) -> torch.device:
        """Enter context, return device."""
        self.device = get_device(self.prefer_gpu, self.gpu_id)
        return self.device

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context, cleanup if needed."""
        if self.device is not None and self.device.type == 'cuda':
            clear_gpu_memory(self.device)
        return False
