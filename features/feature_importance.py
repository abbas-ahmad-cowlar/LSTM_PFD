"""
Feature importance analysis tools.

Purpose:
    Analyze and visualize feature importance from trained models.
    Supports Random Forest Gini importance and permutation importance.

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns


def get_random_forest_importances(rf_model,
                                  feature_names: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Get feature importances from trained Random Forest.

    Uses Gini importance (mean decrease in impurity).

    Args:
        rf_model: Trained RandomForestClassifier
        feature_names: Optional list of feature names

    Returns:
        Dictionary mapping feature names to importance scores

    Example:
        >>> importances = get_random_forest_importances(rf, feature_names)
        >>> sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    """
    importances = rf_model.feature_importances_

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(importances))]

    importance_dict = {
        name: importance for name, importance in zip(feature_names, importances)
    }

    return importance_dict


def get_permutation_importances(model, X_val: np.ndarray, y_val: np.ndarray,
                                feature_names: Optional[List[str]] = None,
                                n_repeats: int = 10,
                                random_state: int = 42) -> Dict[str, Tuple[float, float]]:
    """
    Compute permutation importances.

    Measures importance by randomly shuffling each feature and measuring
    the decrease in model performance.

    Args:
        model: Trained classifier with predict method
        X_val: Validation features
        y_val: Validation labels
        feature_names: Optional feature names
        n_repeats: Number of permutation repeats
        random_state: Random seed

    Returns:
        Dictionary mapping feature names to (mean_importance, std_importance)

    Example:
        >>> importances = get_permutation_importances(model, X_val, y_val, feature_names)
        >>> for name, (mean, std) in importances.items():
        >>>     print(f"{name}: {mean:.4f} +/- {std:.4f}")
    """
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X_val.shape[1])]

    importance_dict = {
        name: (result.importances_mean[i], result.importances_std[i])
        for i, name in enumerate(feature_names)
    }

    return importance_dict


def plot_feature_importances(importances: Dict[str, float],
                             top_n: int = 20,
                             figsize: Tuple[int, int] = (10, 8),
                             title: str = "Feature Importances") -> plt.Figure:
    """
    Plot feature importances as horizontal bar chart.

    Args:
        importances: Dictionary of feature importances
        top_n: Number of top features to plot
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure

    Example:
        >>> importances = get_random_forest_importances(rf, feature_names)
        >>> fig = plot_feature_importances(importances, top_n=15)
        >>> fig.savefig('feature_importances.png')
    """
    # Sort by importance
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    sorted_features = sorted_features[:top_n]

    # Extract names and values
    names = [item[0] for item in sorted_features]
    values = [item[1] for item in sorted_features]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(names))

    ax.barh(y_pos, values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # Top feature at top
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_permutation_importances(importances: Dict[str, Tuple[float, float]],
                                 top_n: int = 20,
                                 figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot permutation importances with error bars.

    Args:
        importances: Dictionary mapping features to (mean, std) tuples
        top_n: Number of top features to plot
        figsize: Figure size

    Returns:
        Matplotlib figure

    Example:
        >>> perm_imp = get_permutation_importances(model, X_val, y_val, feature_names)
        >>> fig = plot_permutation_importances(perm_imp, top_n=15)
    """
    # Sort by mean importance
    sorted_features = sorted(importances.items(),
                           key=lambda x: x[1][0], reverse=True)
    sorted_features = sorted_features[:top_n]

    # Extract data
    names = [item[0] for item in sorted_features]
    means = [item[1][0] for item in sorted_features]
    stds = [item[1][1] for item in sorted_features]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    y_pos = np.arange(len(names))

    ax.barh(y_pos, means, xerr=stds, align='center', alpha=0.7, capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Permutation Importance')
    ax.set_title('Permutation Feature Importances')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def compare_importances(rf_importances: Dict[str, float],
                       perm_importances: Dict[str, Tuple[float, float]],
                       top_n: int = 15) -> plt.Figure:
    """
    Compare Random Forest and permutation importances side by side.

    Args:
        rf_importances: Random Forest Gini importances
        perm_importances: Permutation importances
        top_n: Number of features to compare

    Returns:
        Matplotlib figure with two subplots

    Example:
        >>> rf_imp = get_random_forest_importances(rf, feature_names)
        >>> perm_imp = get_permutation_importances(rf, X_val, y_val, feature_names)
        >>> fig = compare_importances(rf_imp, perm_imp, top_n=15)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Random Forest importances
    rf_sorted = sorted(rf_importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    rf_names = [item[0] for item in rf_sorted]
    rf_values = [item[1] for item in rf_sorted]

    y_pos = np.arange(len(rf_names))
    ax1.barh(y_pos, rf_values, align='center')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(rf_names)
    ax1.invert_yaxis()
    ax1.set_xlabel('Gini Importance')
    ax1.set_title('Random Forest Feature Importances')
    ax1.grid(axis='x', alpha=0.3)

    # Permutation importances
    perm_sorted = sorted(perm_importances.items(),
                        key=lambda x: x[1][0], reverse=True)[:top_n]
    perm_names = [item[0] for item in perm_sorted]
    perm_means = [item[1][0] for item in perm_sorted]
    perm_stds = [item[1][1] for item in perm_sorted]

    y_pos = np.arange(len(perm_names))
    ax2.barh(y_pos, perm_means, xerr=perm_stds, align='center', alpha=0.7, capsize=5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(perm_names)
    ax2.invert_yaxis()
    ax2.set_xlabel('Permutation Importance')
    ax2.set_title('Permutation Feature Importances')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig
