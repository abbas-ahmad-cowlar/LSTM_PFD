"""Evaluation suite for model assessment."""

from .cross_validation import CrossValidationTrainer
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
    'MultiSeedExperiment',
    'compute_confidence_interval',
    'paired_ttest',
    'wilcoxon_test',
    'compare_models',
    'LeakageChecker',
]
