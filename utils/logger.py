"""
Logger compatibility module.

This module provides the setup_logger function expected by the dashboard components.
It wraps the project's central logging configuration.
"""
import logging
import sys
from pathlib import Path

# Global storage for initialized loggers
_loggers = {}
_logging_initialized = False


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.

    This is a compatibility function that provides logging API expected
    by dashboard components.

    Args:
        name: Logger name (usually __name__)
        level: Logging level (default: INFO)

    Returns:
        Configured logger instance
    """
    global _logging_initialized
    
    if name in _loggers:
        return _loggers[name]
    
    if not _logging_initialized:
        # Initialize logging system on first call
        log_dir = Path('./logs')
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
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        _logging_initialized = True
    
    logger = logging.getLogger(name)
    _loggers[name] = logger
    return logger

