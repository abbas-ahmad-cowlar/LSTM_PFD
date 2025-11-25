"""
Benchmarking Module for Phase 10

Provides benchmarking utilities for comparing with literature,
industrial validation, scalability testing, and resource profiling.

Author: Syed Abbas Ahmad
Date: 2025-11-23
"""

from .literature_comparison import compare_with_cwru_benchmark
from .industrial_validation import validate_on_real_bearings
from .scalability_benchmark import benchmark_training_scalability
from .resource_profiling import profile_gpu_utilization

__all__ = [
    'compare_with_cwru_benchmark',
    'validate_on_real_bearings',
    'benchmark_training_scalability',
    'profile_gpu_utilization'
]
