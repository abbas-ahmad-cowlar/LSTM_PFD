"""
Structured logging setup for LSTM_PFD pipeline.

Purpose:
    Centralized logging configuration with:
    - Consistent formatting across modules
    - File and console output
    - Log levels (DEBUG, INFO, WARNING, ERROR)
    - System information logging

Author: Author Name
Date: 2025-11-19
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# Global logger registry
_loggers = {}


def setup_logging(
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True
) -> None:
    """
    Setup global logging configuration.

    Args:
        log_dir: Directory for log files (default: ./logs)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        console: Enable console output
        file: Enable file output

    Example:
        >>> setup_logging(log_dir=Path('./logs'), level=logging.DEBUG)
        >>> logger = get_logger('my_module')
        >>> logger.info("This goes to console and file")
    """
    # Create log directory
    if log_dir is None:
        log_dir = Path('./logs')
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if file:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'lstm_pfd_{timestamp}.log'

        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Also create a "latest" symlink
        latest_link = log_dir / 'latest.log'
        if latest_link.exists():
            latest_link.unlink()
        try:
            latest_link.symlink_to(log_file.name)
        except (OSError, NotImplementedError):
            # Windows may not support symlinks
            pass


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with given name.

    Args:
        name: Logger name (typically __name__ from calling module)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module started")
        >>> logger.warning("Low memory detected")
        >>> logger.error("Failed to load file")
    """
    if name not in _loggers:
        logger = logging.getLogger(name)
        _loggers[name] = logger
    return _loggers[name]


def log_system_info() -> None:
    """
    Log system information for reproducibility.

    Logs:
        - Python version
        - NumPy version
        - PyTorch version (if available)
        - CUDA availability
        - GPU devices
        - CPU count

    Example:
        >>> setup_logging()
        >>> log_system_info()
        # Logs comprehensive system information
    """
    logger = get_logger('system_info')

    import platform
    import numpy as np

    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)

    # Python
    logger.info(f"Python: {platform.python_version()}")
    logger.info(f"Platform: {platform.platform()}")

    # NumPy
    logger.info(f"NumPy: {np.__version__}")

    # PyTorch
    try:
        import torch
        logger.info(f"PyTorch: {torch.__version__}")
        logger.info(f"CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"GPU Count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            logger.info("CUDA: Not available (CPU only)")

    except ImportError:
        logger.info("PyTorch: Not installed")

    # CPU
    import multiprocessing
    logger.info(f"CPU Count: {multiprocessing.cpu_count()}")

    logger.info("=" * 60)
