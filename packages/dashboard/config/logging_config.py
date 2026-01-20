"""
Structured JSON Logging Configuration.

Reference: Master Roadmap Chapter 4.8.B

Configures structlog for enterprise-grade JSON logging.
Enables easy parsing by log aggregation systems (ELK, Loki, Datadog).

Output format:
    {"timestamp": "2026-01-20T06:50:00Z", "level": "info", "event": "user_login", "user_id": 123}

Usage:
    from packages.dashboard.config.logging_config import configure_logging, get_logger
    
    # Initialize at app startup
    configure_logging()
    
    # Get logger in any module
    logger = get_logger(__name__)
    
    # Log structured events
    logger.info("user_login", user_id=123, ip_address="192.168.1.1")
"""

import os
import sys
import logging
from typing import Optional

# Try to import structlog
try:
    import structlog
    from structlog.types import Processor
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False
    structlog = None


# Environment configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()
LOG_FORMAT = os.getenv('LOG_FORMAT', 'json')  # 'json' or 'console'
LOG_FILE = os.getenv('LOG_FILE', '')  # Optional file output


def configure_logging(
    level: str = None,
    format: str = None,
    log_file: str = None
) -> None:
    """
    Configure application logging with structlog.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        format: Output format ('json' or 'console')
        log_file: Optional file path for log output
    """
    level = level or LOG_LEVEL
    format = format or LOG_FORMAT
    log_file = log_file or LOG_FILE
    
    # Convert level string to logging constant
    numeric_level = getattr(logging, level, logging.INFO)
    
    if STRUCTLOG_AVAILABLE:
        _configure_structlog(numeric_level, format, log_file)
    else:
        _configure_standard_logging(numeric_level, format, log_file)


def _configure_structlog(level: int, format: str, log_file: str) -> None:
    """Configure structlog for JSON logging."""
    
    # Shared processors
    shared_processors: list[Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add context processors
    shared_processors.extend([
        _add_app_context,
    ])
    
    # Renderer based on format
    if format == 'json':
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    ))
    
    handlers = [handler]
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),  # Always JSON for files
            foreign_pre_chain=shared_processors,
        ))
        handlers.append(file_handler)
    
    logging.basicConfig(
        format="%(message)s",
        level=level,
        handlers=handlers,
        force=True,
    )


def _configure_standard_logging(level: int, format: str, log_file: str) -> None:
    """Fallback to standard logging when structlog not available."""
    
    if format == 'json':
        # Simple JSON-like format
        log_format = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
    else:
        log_format = '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        format=log_format,
        level=level,
        handlers=handlers,
        force=True,
    )


def _add_app_context(logger, method_name, event_dict):
    """Add application context to log entries."""
    event_dict['app'] = 'lstm-pfd'
    event_dict['version'] = os.getenv('APP_VERSION', '1.0.0')
    return event_dict


def get_logger(name: str = None):
    """
    Get a logger instance.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance (structlog or standard logging)
    """
    if STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    else:
        return logging.getLogger(name)


# Context manager for request logging
class RequestContext:
    """
    Context manager to add request-specific data to logs.
    
    Usage:
        with RequestContext(request_id="abc123", user_id=42):
            logger.info("processing_request")
    """
    
    def __init__(self, **context):
        self.context = context
        self._token = None
    
    def __enter__(self):
        if STRUCTLOG_AVAILABLE:
            self._token = structlog.contextvars.bind_contextvars(**self.context)
        return self
    
    def __exit__(self, *args):
        if STRUCTLOG_AVAILABLE and self._token:
            structlog.contextvars.unbind_contextvars(*self.context.keys())


# Convenience functions for common log events
def log_request(logger, method: str, path: str, status: int, duration_ms: float, **extra):
    """Log an HTTP request."""
    logger.info(
        "http_request",
        method=method,
        path=path,
        status=status,
        duration_ms=round(duration_ms, 2),
        **extra
    )


def log_inference(logger, model: str, fault_class: str, confidence: float, duration_ms: float, **extra):
    """Log a model inference."""
    logger.info(
        "model_inference",
        model=model,
        fault_class=fault_class,
        confidence=round(confidence, 4),
        duration_ms=round(duration_ms, 2),
        **extra
    )


def log_security_event(logger, event_type: str, success: bool, **extra):
    """Log a security-related event."""
    level = "info" if success else "warning"
    getattr(logger, level)(
        "security_event",
        event_type=event_type,
        success=success,
        **extra
    )


if __name__ == '__main__':
    # Test logging configuration
    configure_logging(level='DEBUG', format='json')
    
    logger = get_logger(__name__)
    
    print("Testing structured logging:\n")
    
    # Basic log
    logger.info("app_started", version="1.0.0")
    
    # Request log
    log_request(logger, method="POST", path="/predict", status=200, duration_ms=45.2)
    
    # Inference log
    log_inference(logger, model="pinn", fault_class="inner_race", confidence=0.9823, duration_ms=32.1)
    
    # Security event
    log_security_event(logger, event_type="login_attempt", success=True, user_id=123)
    log_security_event(logger, event_type="login_attempt", success=False, ip_address="1.2.3.4")
    
    # With context
    if STRUCTLOG_AVAILABLE:
        with RequestContext(request_id="req-abc123", user_id=42):
            logger.info("processing_started")
            logger.info("processing_complete")
