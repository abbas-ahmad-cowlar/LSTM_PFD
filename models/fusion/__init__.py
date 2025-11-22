"""
Multi-Modal Fusion for Bearing Fault Diagnosis

This module contains various fusion strategies:
- Early Fusion: Concatenate features from multiple domains before classification
- Late Fusion: Combine final predictions from multiple models
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
from .early_fusion import (
    EarlyFusion,
    MultiModalFeatureExtractor,
    create_early_fusion
)
from .late_fusion import (
    LateFusion,
    late_fusion_weighted_average,
    late_fusion_max,
    late_fusion_product,
    create_late_fusion
)

__all__ = [
    # Early Fusion
    'EarlyFusion',
    'MultiModalFeatureExtractor',
    'create_early_fusion',

    # Late Fusion
    'LateFusion',
    'late_fusion_weighted_average',
    'late_fusion_max',
    'late_fusion_product',
    'create_late_fusion',
]
