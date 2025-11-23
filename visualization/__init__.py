"""
Visualization module for classical ML results and XAI methods.

Author: Syed Abbas Ahmad
Date: 2025-11-19
Updated: 2025-11-20 (Phase 7: XAI)
"""

from .feature_visualization import FeatureVisualizer
from .performance_plots import PerformancePlotter
from .signal_plots import SignalPlotter

# Phase 7: XAI Visualization Tools
from .saliency_maps import (
    SaliencyMapGenerator,
    plot_saliency_map,
    compare_saliency_methods
)
from .counterfactual_explanations import (
    CounterfactualGenerator,
    plot_counterfactual_explanation,
    plot_optimization_history
)

__all__ = [
    'FeatureVisualizer',
    'PerformancePlotter',
    'SignalPlotter',
    # XAI Tools
    'SaliencyMapGenerator',
    'plot_saliency_map',
    'compare_saliency_methods',
    'CounterfactualGenerator',
    'plot_counterfactual_explanation',
    'plot_optimization_history'
]
