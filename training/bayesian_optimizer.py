"""
Bayesian hyperparameter optimization using Optuna.

Purpose:
    Efficient hyperparameter search using Bayesian optimization.
    Much faster than grid search (50 trials vs thousands).

Reference: Section 9.5 of technical report

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import numpy as np
from typing import Dict, Any, Optional, Callable
import optuna
from optuna.samplers import TPESampler
from pathlib import Path


class BayesianOptimizer:
    """
    Bayesian hyperparameter optimizer using Optuna's TPE sampler.

    Example:
        >>> from models.classical import RandomForestClassifier
        >>> optimizer = BayesianOptimizer()
        >>> best_params = optimizer.optimize(
        ...     RandomForestClassifier,
        ...     X_train, y_train,
        ...     X_val, y_val,
        ...     n_trials=50
        ... )
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize Bayesian optimizer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.study = None

    def optimize(self, model_class, X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                n_trials: int = 50,
                timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Run Bayesian optimization to find best hyperparameters.

        Args:
            model_class: Model class to optimize (e.g., RandomForestClassifier)
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of optimization trials
            timeout: Time limit in seconds

        Returns:
            Dictionary with best hyperparameters
        """
        # Determine model type
        model_name = model_class.__name__

        # Create objective function
        def objective(trial: optuna.Trial) -> float:
            # Suggest hyperparameters based on model type
            hyperparams = self._suggest_hyperparameters(trial, model_name)

            # Create and train model
            model = model_class(random_state=self.random_state)
            model.train(X_train, y_train, hyperparams=hyperparams)

            # Evaluate on validation set
            accuracy = model.score(X_val, y_val)

            return accuracy

        # Create study
        sampler = TPESampler(seed=self.random_state)
        self.study = optuna.create_study(
            direction='maximize',
            sampler=sampler
        )

        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        return self.study.best_params

    def _suggest_hyperparameters(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """
        Suggest hyperparameters based on model type.

        Args:
            trial: Optuna trial object
            model_name: Name of model class

        Returns:
            Dictionary of suggested hyperparameters
        """
        if model_name == 'SVMClassifier':
            return {
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'gamma': trial.suggest_float('gamma', 0.01, 10.0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly'])
            }

        elif model_name == 'RandomForestClassifier':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2'])
            }

        elif model_name == 'MLPClassifier':
            # Hidden layer sizes
            n_units_1 = trial.suggest_int('n_units_1', 10, 50)
            n_units_2 = trial.suggest_int('n_units_2', 5, 30)

            return {
                'hidden_layer_sizes': (n_units_1, n_units_2),
                'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
                'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
                'max_iter': 500,
                'activation': trial.suggest_categorical('activation', ['relu', 'tanh'])
            }

        elif model_name == 'GradientBoostingClassifier':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 2, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0)
            }

        else:
            raise ValueError(f"Unknown model type: {model_name}")

    def get_optimization_history(self) -> Dict:
        """
        Get optimization history.

        Returns:
            Dictionary with trial numbers and values
        """
        if self.study is None:
            raise ValueError("No optimization has been run")

        trials = self.study.trials
        history = {
            'trial_numbers': [t.number for t in trials],
            'values': [t.value for t in trials],
            'params': [t.params for t in trials]
        }

        return history

    def get_best_trial_info(self) -> Dict:
        """
        Get information about best trial.

        Returns:
            Dictionary with best trial information
        """
        if self.study is None:
            raise ValueError("No optimization has been run")

        best_trial = self.study.best_trial

        return {
            'number': best_trial.number,
            'value': best_trial.value,
            'params': best_trial.params
        }

    def plot_optimization_history(self, save_path: Optional[Path] = None):
        """
        Plot optimization history.

        Args:
            save_path: Path to save plot (optional)
        """
        if self.study is None:
            raise ValueError("No optimization has been run")

        import matplotlib.pyplot as plt

        trials = self.study.trials
        trial_numbers = [t.number for t in trials]
        values = [t.value for t in trials]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(trial_numbers, values, 'o-', alpha=0.7)
        ax.axhline(y=self.study.best_value, color='r', linestyle='--',
                  label=f'Best: {self.study.best_value:.4f}')
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Validation Accuracy')
        ax.set_title('Bayesian Optimization History')
        ax.legend()
        ax.grid(alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_param_importances(self, save_path: Optional[Path] = None):
        """
        Plot hyperparameter importances.

        Args:
            save_path: Path to save plot (optional)
        """
        if self.study is None:
            raise ValueError("No optimization has been run")

        import matplotlib.pyplot as plt
        from optuna.importance import get_param_importances
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH

        importances = get_param_importances(self.study)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        params = list(importances.keys())
        values = list(importances.values())

        ax.barh(params, values)
        ax.set_xlabel('Importance')
        ax.set_title('Hyperparameter Importances')
        ax.grid(axis='x', alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
