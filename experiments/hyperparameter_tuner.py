"""
Hyperparameter Tuning with Optuna

Provides Bayesian optimization for:
- Learning rate
- Batch size
- Model architecture parameters
- Regularization parameters
"""

from typing import Dict, Any, Callable, Optional
import torch


class HyperparameterTuner:
    """
    Tune hyperparameters using Optuna.

    Args:
        objective_fn: Function that takes trial and returns metric to optimize
        direction: 'minimize' or 'maximize' (default: 'maximize')
        n_trials: Number of trials (default: 50)
        study_name: Optional name for the study
    """
    def __init__(
        self,
        objective_fn: Callable,
        direction: str = 'maximize',
        n_trials: int = 50,
        study_name: Optional[str] = None
    ):
        self.objective_fn = objective_fn
        self.direction = direction
        self.n_trials = n_trials
        self.study_name = study_name or 'hyperparameter_tuning'
        self.optuna = None
        self.study = None

        self._init_optuna()

    def _init_optuna(self):
        """Initialize Optuna."""
        try:
            import optuna
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
            self.optuna = optuna

            self.study = optuna.create_study(
                direction=self.direction,
                study_name=self.study_name
            )

        except ImportError:
            print("Optuna not available. Install with: pip install optuna")

    def tune(self) -> Dict[str, Any]:
        """
        Run hyperparameter tuning.

        Returns:
            Dictionary of best hyperparameters
        """
        if self.optuna is None:
            return {}

        # Run optimization
        self.study.optimize(self.objective_fn, n_trials=self.n_trials)

        # Get best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value

        print(f"\nBest value: {best_value}")
        print("Best parameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")

        return best_params

    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters found."""
        if self.study is None:
            return {}

        return self.study.best_params

    def get_best_value(self) -> float:
        """Get best metric value achieved."""
        if self.study is None:
            return 0.0

        return self.study.best_value


def suggest_hyperparameters(trial) -> Dict[str, Any]:
    """
    Suggest hyperparameters for a trial.

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary of suggested hyperparameters
    """
    params = {
        # Learning rate
        'lr': trial.suggest_float('lr', 1e-5, 1e-2, log=True),

        # Batch size
        'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),

        # Optimizer
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),

        # Dropout
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),

        # Weight decay
        'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),

        # Model-specific parameters
        'd_model': trial.suggest_categorical('d_model', [128, 256, 512]),
        'num_layers': trial.suggest_int('num_layers', 2, 8),
    }

    return params
