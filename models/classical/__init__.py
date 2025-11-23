"""
Classical machine learning models for bearing fault classification.

This module contains implementations of SVM, Random Forest, Neural Network,
Gradient Boosting, and ensemble methods.

Author: Syed Abbas Ahmad
Date: 2025-11-19
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
from .svm_classifier import SVMClassifier
from .random_forest import RandomForestClassifier
from .neural_network import MLPClassifier
from .gradient_boosting import GradientBoostingClassifier
from .stacked_ensemble import StackedEnsemble
from .model_selector import ModelSelector

__all__ = [
    'SVMClassifier',
    'RandomForestClassifier',
    'MLPClassifier',
    'GradientBoostingClassifier',
    'StackedEnsemble',
    'ModelSelector'
]
