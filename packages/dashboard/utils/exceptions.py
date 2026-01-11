"""
Custom exception classes for the Dash application.
"""


class DashAppException(Exception):
    """Base exception for Dash app errors."""
    pass


class DataValidationError(DashAppException):
    """Raised when data validation fails."""
    pass


class ModelNotFoundError(DashAppException):
    """Raised when a requested model is not found."""
    pass


class DatasetNotFoundError(DashAppException):
    """Raised when a requested dataset is not found."""
    pass


class ExperimentNotFoundError(DashAppException):
    """Raised when a requested experiment is not found."""
    pass


class TrainingError(DashAppException):
    """Raised when training fails."""
    pass


class CacheError(DashAppException):
    """Raised when cache operations fail."""
    pass


class IntegrationError(DashAppException):
    """Raised when Phase 0-10 integration fails."""
    pass
