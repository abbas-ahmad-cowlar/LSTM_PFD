"""
Grid search for exhaustive hyperparameter search.

Purpose:
    Systematic search over specified hyperparameter grid.
    Use when search space is small and discrete.

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import numpy as np
from typing import Dict, List, Any
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator


class GridSearchOptimizer:
    """
    Grid search hyperparameter optimizer.

    Example:
        >>> from models.classical import RandomForestClassifier
        >>> param_grid = {
        ...     'n_estimators': [50, 100, 200],
        ...     'max_depth': [10, 20, 30]
        ... }
        >>> optimizer = GridSearchOptimizer()
        >>> best_params = optimizer.search(
        ...     RandomForestClassifier(),
        ...     X_train, y_train,
        ...     param_grid,
        ...     cv=5
        ... )
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize grid search optimizer.

        Args:
            random_state: Random seed
        """
        self.random_state = random_state
        self.grid_search = None

    def search(self, model, X_train: np.ndarray, y_train: np.ndarray,
              param_grid: Dict[str, List[Any]],
              cv: int = 5,
              scoring: str = 'accuracy',
              n_jobs: int = -1) -> Dict[str, Any]:
        """
        Perform grid search.

        Args:
            model: Model instance with sklearn-compatible interface
            X_train: Training features
            y_train: Training labels
            param_grid: Dictionary mapping param names to lists of values
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs

        Returns:
            Dictionary with best hyperparameters
        """
        # For our custom models, we need to access the underlying sklearn model
        if hasattr(model, 'model'):
            sklearn_model = model.model
        else:
            sklearn_model = model

        # Perform grid search
        self.grid_search = GridSearchCV(
            estimator=sklearn_model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            return_train_score=True
        )

        self.grid_search.fit(X_train, y_train)

        return self.grid_search.best_params_

    def get_results(self) -> Dict:
        """
        Get detailed grid search results.

        Returns:
            Dictionary with cv results
        """
        if self.grid_search is None:
            raise ValueError("Grid search has not been run")

        return {
            'best_params': self.grid_search.best_params_,
            'best_score': self.grid_search.best_score_,
            'cv_results': self.grid_search.cv_results_
        }

    def get_best_score(self) -> float:
        """Get best cross-validation score."""
        if self.grid_search is None:
            raise ValueError("Grid search has not been run")

        return self.grid_search.best_score_

    def plot_grid_results(self, param_x: str, param_y: str, save_path=None):
        """
        Plot grid search results for 2D parameter grid.

        Args:
            param_x: Parameter for x-axis
            param_y: Parameter for y-axis
            save_path: Optional path to save figure
        """
        if self.grid_search is None:
            raise ValueError("Grid search has not been run")

        import matplotlib.pyplot as plt
        import pandas as pd

        # Extract results
        results = pd.DataFrame(self.grid_search.cv_results_)

        # Get unique param values
        x_values = results[f'param_{param_x}'].unique()
        y_values = results[f'param_{param_y}'].unique()

        # Create grid of scores
        scores = np.zeros((len(y_values), len(x_values)))
        for i, y_val in enumerate(y_values):
            for j, x_val in enumerate(x_values):
                mask = (results[f'param_{param_x}'] == x_val) & \
                       (results[f'param_{param_y}'] == y_val)
                if mask.any():
                    scores[i, j] = results.loc[mask, 'mean_test_score'].values[0]

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(scores, aspect='auto', cmap='viridis')

        # Set ticks
        ax.set_xticks(np.arange(len(x_values)))
        ax.set_yticks(np.arange(len(y_values)))
        ax.set_xticklabels(x_values)
        ax.set_yticklabels(y_values)

        # Labels
        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        ax.set_title('Grid Search Results')

        # Colorbar
        plt.colorbar(im, ax=ax, label='Mean CV Score')

        # Annotate cells with scores
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                text = ax.text(j, i, f'{scores[i, j]:.3f}',
                             ha="center", va="center", color="w", fontsize=8)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


def get_default_param_grid(model_name: str) -> Dict[str, List[Any]]:
    """
    Get default parameter grid for common models.

    Args:
        model_name: Name of model ('SVM', 'RandomForest', 'MLP', 'GradientBoosting')

    Returns:
        Parameter grid dictionary
    """
    if model_name == 'SVM':
        return {
            'C': [0.1, 1.0, 10.0, 100.0],
            'gamma': [0.01, 0.1, 1.0, 10.0],
            'kernel': ['rbf', 'poly']
        }

    elif model_name == 'RandomForest':
        return {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': ['sqrt', 'log2']
        }

    elif model_name == 'MLP':
        return {
            'hidden_layer_sizes': [(20, 10), (30, 15), (40, 20)],
            'learning_rate_init': [0.0001, 0.001, 0.01],
            'alpha': [0.0001, 0.001, 0.01],
            'activation': ['relu', 'tanh']
        }

    elif model_name == 'GradientBoosting':
        return {
            'n_estimators': [50, 100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 10],
            'subsample': [0.6, 0.8, 1.0]
        }

    else:
        raise ValueError(f"Unknown model name: {model_name}")
