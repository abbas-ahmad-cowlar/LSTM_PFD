"""
Classical machine learning models for bearing fault classification.

Tier-1 baselines: SVM, Random Forest, Gradient Boosting — trained on the
36 hand-crafted features. ModelSelector compares them on a validation set.
"""

from .svm_classifier import SVMClassifier
from .random_forest import RandomForestClassifier
from .gradient_boosting import GradientBoostingClassifier
from .model_selector import ModelSelector

__all__ = [
    'SVMClassifier',
    'RandomForestClassifier',
    'GradientBoostingClassifier',
    'ModelSelector'
]
