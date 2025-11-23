"""
Model Selection for Ensemble Building

Selects diverse models for ensemble to maximize performance.
Uses diversity metrics to choose complementary models.

Author: Syed Abbas Ahmad
Date: 2025-11-23
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import combinations


class DiversityBasedSelector:
    """
    Select models for ensemble based on diversity metrics.

    Balances individual model accuracy with ensemble diversity to select
    the best combination of models for ensemble learning.

    Metrics supported:
    - disagreement: Fraction of samples where models disagree
    - kappa: Cohen's kappa statistic
    - q_statistic: Q-statistic correlation

    Args:
        metric: Diversity metric to use ('disagreement', 'kappa', 'q_statistic')

    Example:
        >>> selector = DiversityBasedSelector(metric='disagreement')
        >>> selected = selector.select(predictions, accuracies, num_models=5)
    """

    def __init__(self, metric: str = 'disagreement'):
        """
        Initialize diversity-based model selector.

        Args:
            metric: Diversity metric ('disagreement', 'kappa', 'q_statistic')
        """
        if metric not in ['disagreement', 'kappa', 'q_statistic']:
            raise ValueError(
                f"Unknown metric: {metric}. "
                f"Must be one of: 'disagreement', 'kappa', 'q_statistic'"
            )

        self.metric = metric

    def select(
        self,
        predictions: Dict[str, np.ndarray],
        accuracies: Dict[str, float],
        num_models: int = 5,
        diversity_weight: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Select diverse models for ensemble.

        Args:
            predictions: Dictionary mapping model names to predictions [N]
            accuracies: Dictionary mapping model names to accuracies
            num_models: Number of models to select
            diversity_weight: Weight for diversity (vs accuracy) in [0, 1]
                             0 = pure accuracy, 1 = pure diversity

        Returns:
            List of (model_name, score) tuples, sorted by score (descending)

        Example:
            >>> predictions = {
            ...     'CNN': np.array([0, 1, 2, ...]),
            ...     'ResNet': np.array([0, 1, 1, ...]),
            ...     'Transformer': np.array([0, 2, 2, ...])
            ... }
            >>> accuracies = {'CNN': 0.95, 'ResNet': 0.96, 'Transformer': 0.94}
            >>> selected = selector.select(predictions, accuracies, num_models=2)
        """
        if not 0 <= diversity_weight <= 1:
            raise ValueError("diversity_weight must be in [0, 1]")

        if num_models > len(predictions):
            raise ValueError(
                f"Cannot select {num_models} models from {len(predictions)} available"
            )

        model_names = list(predictions.keys())

        # Compute pairwise diversity matrix
        diversity_matrix = self._compute_diversity_matrix(predictions, model_names)

        # Score each model
        model_scores = {}

        for i, name in enumerate(model_names):
            # Accuracy component
            accuracy = accuracies[name]

            # Diversity component: average diversity with all other models
            diversity = np.mean([
                diversity_matrix[i, j]
                for j in range(len(model_names))
                if i != j
            ])

            # Combined score
            score = (1 - diversity_weight) * accuracy + diversity_weight * diversity
            model_scores[name] = score

        # Sort by score (descending)
        sorted_models = sorted(
            model_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top num_models
        return sorted_models[:num_models]

    def _compute_diversity_matrix(
        self,
        predictions: Dict[str, np.ndarray],
        model_names: List[str]
    ) -> np.ndarray:
        """
        Compute pairwise diversity between all models.

        Args:
            predictions: Model predictions
            model_names: List of model names

        Returns:
            Diversity matrix [n_models, n_models]
        """
        n_models = len(model_names)
        diversity_matrix = np.zeros((n_models, n_models))

        for i, j in combinations(range(n_models), 2):
            pred_i = predictions[model_names[i]]
            pred_j = predictions[model_names[j]]

            if self.metric == 'disagreement':
                diversity = self._disagreement(pred_i, pred_j)
            elif self.metric == 'kappa':
                diversity = 1 - self._kappa_statistic(pred_i, pred_j)
            elif self.metric == 'q_statistic':
                diversity = 1 - abs(self._q_statistic(pred_i, pred_j))
            else:
                diversity = 0

            diversity_matrix[i, j] = diversity
            diversity_matrix[j, i] = diversity

        return diversity_matrix

    def _disagreement(self, pred_i: np.ndarray, pred_j: np.ndarray) -> float:
        """
        Compute disagreement between two models.

        Args:
            pred_i: Predictions from model i [N]
            pred_j: Predictions from model j [N]

        Returns:
            Disagreement fraction (0 = always agree, 1 = always disagree)
        """
        return (pred_i != pred_j).mean()

    def _kappa_statistic(self, pred_i: np.ndarray, pred_j: np.ndarray) -> float:
        """
        Compute Cohen's kappa statistic.

        Measures agreement beyond chance.

        Args:
            pred_i: Predictions from model i [N]
            pred_j: Predictions from model j [N]

        Returns:
            Kappa statistic (-1 to 1, where 1 = perfect agreement)
        """
        # Observed agreement
        p_o = (pred_i == pred_j).mean()

        # Expected agreement by chance
        n_samples = len(pred_i)
        classes = np.union1d(pred_i, pred_j)

        p_e = 0
        for c in classes:
            p_i = (pred_i == c).sum() / n_samples
            p_j = (pred_j == c).sum() / n_samples
            p_e += p_i * p_j

        # Kappa
        if p_e == 1:
            return 0  # Avoid division by zero
        else:
            kappa = (p_o - p_e) / (1 - p_e)
            return kappa

    def _q_statistic(self, pred_i: np.ndarray, pred_j: np.ndarray) -> float:
        """
        Compute Q-statistic.

        Measures correlation between model errors.

        Args:
            pred_i: Predictions from model i [N]
            pred_j: Predictions from model j [N]

        Returns:
            Q-statistic (-1 to 1)
        """
        # For Q-statistic we need true labels, but we approximate
        # using agreement/disagreement
        agree = (pred_i == pred_j).sum()
        disagree = (pred_i != pred_j).sum()

        if agree + disagree == 0:
            return 0

        q = (agree - disagree) / (agree + disagree)
        return q


def select_diverse_models(
    predictions: Dict[str, np.ndarray],
    accuracies: Dict[str, float],
    num_models: int = 5,
    diversity_weight: float = 0.3,
    metric: str = 'disagreement'
) -> List[Tuple[str, float]]:
    """
    Helper function to select diverse models.

    Args:
        predictions: Dictionary mapping model names to predictions
        accuracies: Dictionary mapping model names to accuracies
        num_models: Number of models to select
        diversity_weight: Weight for diversity (vs accuracy)
        metric: Diversity metric to use

    Returns:
        List of (model_name, score) tuples

    Example:
        >>> selected = select_diverse_models(predictions, accuracies, num_models=3)
    """
    selector = DiversityBasedSelector(metric=metric)
    return selector.select(predictions, accuracies, num_models, diversity_weight)
