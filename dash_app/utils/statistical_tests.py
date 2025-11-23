"""
Statistical Testing Utilities for Model Comparison (Phase 11C).
Implements McNemar's test, Friedman test, and bootstrap confidence intervals.
"""
import numpy as np
from typing import Tuple, Dict, Any, List, Optional
from scipy import stats
from scipy.stats import chi2, friedmanchisquare
from utils.logger import setup_logger

logger = setup_logger(__name__)


def mcnemar_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray
) -> Dict[str, Any]:
    """
    Perform McNemar's test for paired comparison of two classifiers.

    McNemar's test determines if two classifiers have significantly different error rates
    when tested on the same dataset.

    Contingency table:
                    Model 2 Correct  Model 2 Wrong
    Model 1 Correct      a              b
    Model 1 Wrong        c              d

    Test statistic: χ² = (b - c)² / (b + c)
    p-value: From chi-square distribution with 1 degree of freedom

    Args:
        y_true: True labels [N]
        y_pred1: Predictions from model 1 [N]
        y_pred2: Predictions from model 2 [N]

    Returns:
        Dictionary with test results:
        {
            'test_statistic': float,
            'p_value': float,
            'contingency_table': [[a, b], [c, d]],
            'interpretation': str,
            'significant': bool (p < 0.05)
        }
    """
    # Validate inputs
    if len(y_true) != len(y_pred1) or len(y_true) != len(y_pred2):
        raise ValueError("All arrays must have the same length")

    # Build contingency table
    correct1 = (y_pred1 == y_true)
    correct2 = (y_pred2 == y_true)

    a = np.sum(correct1 & correct2)  # Both correct
    b = np.sum(correct1 & ~correct2)  # Model 1 correct, Model 2 wrong
    c = np.sum(~correct1 & correct2)  # Model 1 wrong, Model 2 correct
    d = np.sum(~correct1 & ~correct2)  # Both wrong

    contingency_table = [[int(a), int(b)], [int(c), int(d)]]

    # McNemar's test statistic
    if b + c == 0:
        # No disagreements, models are identical
        return {
            'test_statistic': 0.0,
            'p_value': 1.0,
            'contingency_table': contingency_table,
            'interpretation': 'Models make identical predictions (no disagreements).',
            'significant': False,
            'effect_size': 0.0
        }

    test_statistic = (b - c) ** 2 / (b + c)

    # p-value from chi-square distribution (1 degree of freedom)
    p_value = 1 - chi2.cdf(test_statistic, df=1)

    # Interpretation
    if p_value < 0.05:
        if b > c:
            winner = "Model 1"
        else:
            winner = "Model 2"
        interpretation = f"{winner} performs significantly better (p = {p_value:.4f})."
    else:
        interpretation = f"No significant difference between models (p = {p_value:.4f})."

    # Effect size (odds ratio)
    if c > 0:
        odds_ratio = b / c
    else:
        odds_ratio = float('inf') if b > 0 else 1.0

    return {
        'test_statistic': float(test_statistic),
        'p_value': float(p_value),
        'contingency_table': contingency_table,
        'interpretation': interpretation,
        'significant': p_value < 0.05,
        'effect_size': float(odds_ratio),
        'disagreements': {'model1_better': int(b), 'model2_better': int(c)}
    }


def friedman_test(
    y_true: np.ndarray,
    y_preds: List[np.ndarray]
) -> Dict[str, Any]:
    """
    Perform Friedman test for comparing 3+ classifiers.

    Friedman test is a non-parametric test for repeated measures, like ANOVA but for ranks.
    It's used when comparing multiple classifiers on the same dataset.

    Args:
        y_true: True labels [N]
        y_preds: List of prediction arrays from different models, each [N]

    Returns:
        Dictionary with test results:
        {
            'test_statistic': float,
            'p_value': float,
            'rankings': List[float],  # Average rank per model (1=best)
            'interpretation': str,
            'significant': bool
        }
    """
    # Validate inputs
    if len(y_preds) < 3:
        raise ValueError("Friedman test requires at least 3 models")

    for y_pred in y_preds:
        if len(y_pred) != len(y_true):
            raise ValueError("All prediction arrays must match true labels length")

    n_samples = len(y_true)
    n_models = len(y_preds)

    # Compute correctness for each model on each sample
    correctness = []
    for y_pred in y_preds:
        correctness.append((y_pred == y_true).astype(int))

    # Run Friedman test
    statistic, p_value = friedmanchisquare(*correctness)

    # Compute average rankings
    sample_ranks = []
    for i in range(n_samples):
        sample_correctness = [correctness[m][i] for m in range(n_models)]
        # Rank: Lower rank for correct (higher correctness), higher rank for incorrect
        # We negate to rank correctly (1 is best)
        ranks = stats.rankdata([-c for c in sample_correctness], method='average')
        sample_ranks.append(ranks)

    # Average rank per model
    avg_ranks = np.mean(sample_ranks, axis=0).tolist()

    # Interpretation
    if p_value < 0.05:
        best_model_idx = np.argmin(avg_ranks)
        interpretation = (
            f"Significant difference exists (p = {p_value:.4f}). "
            f"Model {best_model_idx + 1} ranks best (avg rank: {avg_ranks[best_model_idx]:.2f})."
        )
    else:
        interpretation = f"No significant difference among models (p = {p_value:.4f})."

    return {
        'test_statistic': float(statistic),
        'p_value': float(p_value),
        'rankings': avg_ranks,
        'interpretation': interpretation,
        'significant': p_value < 0.05,
        'n_models': n_models,
        'n_samples': n_samples
    }


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = 'accuracy',
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute bootstrap confidence interval for a classification metric.

    Uses bootstrap resampling to estimate the confidence interval of a metric
    (accuracy, precision, recall, f1) on a test set.

    Args:
        y_true: True labels [N]
        y_pred: Predicted labels [N]
        metric: Metric to compute ('accuracy', 'precision', 'recall', 'f1')
        n_iterations: Number of bootstrap iterations
        confidence_level: Confidence level (default: 0.95 for 95% CI)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with CI results:
        {
            'mean': float,
            'std': float,
            'ci_lower': float,
            'ci_upper': float,
            'confidence_level': float,
            'n_iterations': int
        }
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    if seed is not None:
        np.random.seed(seed)

    # Select metric function
    metric_funcs = {
        'accuracy': accuracy_score,
        'precision': lambda y_t, y_p: precision_score(y_t, y_p, average='weighted', zero_division=0),
        'recall': lambda y_t, y_p: recall_score(y_t, y_p, average='weighted', zero_division=0),
        'f1': lambda y_t, y_p: f1_score(y_t, y_p, average='weighted', zero_division=0)
    }

    if metric not in metric_funcs:
        raise ValueError(f"Unknown metric: {metric}. Choose from {list(metric_funcs.keys())}")

    metric_func = metric_funcs[metric]

    # Bootstrap resampling
    n_samples = len(y_true)
    scores = []

    for _ in range(n_iterations):
        # Resample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Compute metric
        score = metric_func(y_true_boot, y_pred_boot)
        scores.append(score)

    scores = np.array(scores)

    # Compute confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(scores, alpha / 2 * 100)
    ci_upper = np.percentile(scores, (1 - alpha / 2) * 100)

    return {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'confidence_level': confidence_level,
        'n_iterations': n_iterations,
        'metric': metric
    }


def paired_t_test(
    scores1: np.ndarray,
    scores2: np.ndarray,
    alternative: str = 'two-sided'
) -> Dict[str, Any]:
    """
    Perform paired t-test for comparing two models.

    Used when you have per-sample scores (e.g., from cross-validation)
    rather than just predictions.

    Args:
        scores1: Scores from model 1 [N]
        scores2: Scores from model 2 [N]
        alternative: Alternative hypothesis ('two-sided', 'less', 'greater')

    Returns:
        Dictionary with test results
    """
    if len(scores1) != len(scores2):
        raise ValueError("Score arrays must have the same length")

    statistic, p_value = stats.ttest_rel(scores1, scores2, alternative=alternative)

    # Effect size (Cohen's d for paired samples)
    differences = scores1 - scores2
    cohens_d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences) > 0 else 0

    # Interpretation
    if p_value < 0.05:
        if alternative == 'two-sided':
            if np.mean(scores1) > np.mean(scores2):
                winner = "Model 1"
            else:
                winner = "Model 2"
            interpretation = f"{winner} performs significantly better (p = {p_value:.4f})."
        elif alternative == 'greater':
            interpretation = f"Model 1 performs significantly better (p = {p_value:.4f})."
        else:  # less
            interpretation = f"Model 2 performs significantly better (p = {p_value:.4f})."
    else:
        interpretation = f"No significant difference between models (p = {p_value:.4f})."

    return {
        'test_statistic': float(statistic),
        'p_value': float(p_value),
        'interpretation': interpretation,
        'significant': p_value < 0.05,
        'mean_difference': float(np.mean(differences)),
        'effect_size': float(cohens_d),
        'alternative': alternative
    }


def compute_effect_size(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray
) -> Dict[str, float]:
    """
    Compute various effect size measures for model comparison.

    Args:
        y_true: True labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2

    Returns:
        Dictionary with effect size metrics
    """
    from sklearn.metrics import accuracy_score

    acc1 = accuracy_score(y_true, y_pred1)
    acc2 = accuracy_score(y_true, y_pred2)

    # Absolute difference
    abs_diff = acc1 - acc2

    # Relative improvement
    if acc2 > 0:
        rel_improvement = (acc1 - acc2) / acc2 * 100
    else:
        rel_improvement = 0.0

    # Cohen's h (for proportions)
    p1 = acc1
    p2 = acc2
    cohens_h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

    return {
        'absolute_difference': float(abs_diff),
        'relative_improvement_percent': float(rel_improvement),
        'cohens_h': float(cohens_h)
    }
