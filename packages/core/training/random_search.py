"""
Random search for hyperparameter optimization.

Purpose:
    Random sampling from hyperparameter distributions.
    More efficient than grid search for large search spaces.

Author: Syed Abbas Ahmad
Date: 2025-11-19
"""

import numpy as np
from typing import Dict, Any
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint


class RandomSearchOptimizer:
    """
    Random search hyperparameter optimizer.

    Example:
        >>> from models.classical import RandomForestClassifier
        >>> param_dist = {
        ...     'n_estimators': randint(50, 200),
        ...     'max_depth': randint(5, 30)
        ... }
        >>> optimizer = RandomSearchOptimizer()
        >>> best_params = optimizer.search(
        ...     RandomForestClassifier(),
        ...     X_train, y_train,
        ...     param_dist,
        ...     n_iter=50
        ... )
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize random search optimizer.

        Args:
            random_state: Random seed
        """
        self.random_state = random_state
        self.random_search = None

    def search(self, model, X_train: np.ndarray, y_train: np.ndarray,
              param_distributions: Dict[str, Any],
              n_iter: int = 50,
              cv: int = 5,
              scoring: str = 'accuracy',
              n_jobs: int = -1) -> Dict[str, Any]:
        """
        Perform random search.

        Args:
            model: Model instance with sklearn-compatible interface
            X_train: Training features
            y_train: Training labels
            param_distributions: Dictionary mapping param names to distributions
            n_iter: Number of random samples
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs

        Returns:
            Dictionary with best hyperparameters
        """
        # For our custom models, we need to access the underlying sklearn model
        if hasattr(model, 'model') and model.model is not None:
            sklearn_model = model.model
        else:
            sklearn_model = model

        # Perform random search
        self.random_search = RandomizedSearchCV(
            estimator=sklearn_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            random_state=self.random_state,
            return_train_score=True
        )

        self.random_search.fit(X_train, y_train)

        return self.random_search.best_params_

    def get_results(self) -> Dict:
        """
        Get detailed random search results.

        Returns:
            Dictionary with cv results
        """
        if self.random_search is None:
            raise ValueError("Random search has not been run")

        return {
            'best_params': self.random_search.best_params_,
            'best_score': self.random_search.best_score_,
            'cv_results': self.random_search.cv_results_
        }

    def get_best_score(self) -> float:
        """Get best cross-validation score."""
        if self.random_search is None:
            raise ValueError("Random search has not been run")

        return self.random_search.best_score_

    def plot_search_results(self, save_path=None):
        """
        Plot random search iteration vs score.

        Args:
            save_path: Optional path to save figure
        """
        if self.random_search is None:
            raise ValueError("Random search has not been run")

        import matplotlib.pyplot as plt

        # Extract results
        results = self.random_search.cv_results_
        scores = results['mean_test_score']
        iterations = np.arange(len(scores))

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(iterations, scores, 'o-', alpha=0.7)
        ax.axhline(y=self.random_search.best_score_, color='r', linestyle='--',
                  label=f'Best: {self.random_search.best_score_:.4f}')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Mean CV Score')
        ax.set_title('Random Search Results')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


def get_default_param_distributions(model_name: str) -> Dict[str, Any]:
    """
    Get default parameter distributions for common models.

    Args:
        model_name: Name of model ('SVM', 'RandomForest', 'MLP', 'GradientBoosting')

    Returns:
        Parameter distribution dictionary
    """
    if model_name == 'SVM':
        return {
            'C': uniform(0.1, 100),  # Uniform from 0.1 to 100.1
            'gamma': uniform(0.01, 10),  # Uniform from 0.01 to 10.01
            'kernel': ['rbf', 'poly']
        }

    elif model_name == 'RandomForest':
        return {
            'n_estimators': randint(50, 200),
            'max_depth': randint(5, 30),
            'min_samples_leaf': randint(1, 20),
            'max_features': ['sqrt', 'log2']
        }

    elif model_name == 'MLP':
        return {
            'hidden_layer_sizes': [(20, 10), (30, 15), (40, 20), (50, 25)],
            'learning_rate_init': uniform(0.0001, 0.01),
            'alpha': uniform(0.0001, 0.01),
            'activation': ['relu', 'tanh']
        }

    elif model_name == 'GradientBoosting':
        return {
            'n_estimators': randint(50, 200),
            'learning_rate': uniform(0.01, 0.3),
            'max_depth': randint(2, 10),
            'subsample': uniform(0.5, 0.5)  # 0.5 to 1.0
        }

    else:
        raise ValueError(f"Unknown model name: {model_name}")
