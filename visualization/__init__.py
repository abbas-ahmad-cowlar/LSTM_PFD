"""
Visualization module for results, signals, and XAI methods.
"""

from .feature_visualization import FeatureVisualizer
from .performance_plots import PerformancePlotter
from .signal_plots import SignalPlotter

# XAI visualization
from .saliency_maps import (
    SaliencyMapGenerator,
    plot_saliency_map,
    compare_saliency_methods
)

__all__ = [
    'FeatureVisualizer',
    'PerformancePlotter',
    'SignalPlotter',
    # XAI Tools
    'SaliencyMapGenerator',
    'plot_saliency_map',
    'compare_saliency_methods',
]
