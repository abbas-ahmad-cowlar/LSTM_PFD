"""Evaluation suite for model assessment."""

from .cross_validation import CrossValidationTrainer
from .temporal_cv import TimeSeriesSplit, BlockingTimeSeriesSplit, TemporalCrossValidator
from .statistical_analysis import (
    MultiSeedExperiment,
    compute_confidence_interval,
    paired_ttest,
    wilcoxon_test,
    compare_models
)
from .check_data_leakage import LeakageChecker

__all__ = [
    'CrossValidationTrainer',
    'TimeSeriesSplit',
    'BlockingTimeSeriesSplit',
    'TemporalCrossValidator',
    'MultiSeedExperiment',
    'compute_confidence_interval',
    'paired_ttest',
    'wilcoxon_test',
    'compare_models',
    'LeakageChecker',
]
