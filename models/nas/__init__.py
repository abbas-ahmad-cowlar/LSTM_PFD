"""
Neural Architecture Search (NAS) for 1D signal processing.

This package provides tools for automated architecture discovery:
- Search space definition
- DARTS algorithm (differentiable architecture search)
- Random search and evolutionary algorithms
- Architecture evaluation framework

Note: NAS is computationally expensive. Consider using pre-designed
architectures (ResNet, EfficientNet) for most applications.
"""

from .search_space import (
    SearchSpaceConfig,
    ArchitectureSpec,
    OperationType
)

__all__ = [
    'SearchSpaceConfig',
    'ArchitectureSpec',
    'OperationType',
]
