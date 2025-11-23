"""
Integration Module for Phase 10

Provides unified pipeline, model registry, and validation tools for
integrating all phases of the LSTM_PFD project.

Author: Syed Abbas Ahmad
Date: 2025-11-23
"""

from .unified_pipeline import UnifiedMLPipeline
from .model_registry import ModelRegistry
from .data_pipeline_validator import validate_data_compatibility
from .configuration_validator import validate_config

__all__ = [
    'UnifiedMLPipeline',
    'ModelRegistry',
    'validate_data_compatibility',
    'validate_config'
]
