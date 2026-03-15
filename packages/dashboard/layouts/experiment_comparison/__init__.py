"""
Experiment Comparison Dashboard package.
Split from monolithic experiment_comparison.py (805 lines) into focused modules.
"""
from layouts.experiment_comparison.layout import (
    create_experiment_comparison_layout,
    format_duration,
)
from layouts.experiment_comparison.tabs import (
    create_overview_tab,
    create_metrics_tab,
    create_visualizations_tab,
    create_statistical_tab,
    create_configuration_tab,
)

__all__ = [
    'create_experiment_comparison_layout',
    'create_overview_tab',
    'create_metrics_tab',
    'create_visualizations_tab',
    'create_statistical_tab',
    'create_configuration_tab',
    'format_duration',
]
