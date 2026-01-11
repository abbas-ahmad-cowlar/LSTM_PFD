"""
Pipeline integration for classical ML and feature extraction.

This module provides end-to-end pipelines for feature extraction,
model training, and evaluation.

Author: Syed Abbas Ahmad
Date: 2025-11-19
"""

from .classical_ml_pipeline import ClassicalMLPipeline
from .feature_pipeline import FeaturePipeline
from .matlab_compat import MatlabCompatibility
from .pipeline_validator import PipelineValidator
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

__all__ = [
    'ClassicalMLPipeline',
    'FeaturePipeline',
    'MatlabCompatibility',
    'PipelineValidator'
]
