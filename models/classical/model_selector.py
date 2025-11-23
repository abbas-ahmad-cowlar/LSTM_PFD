"""
Automated model selection based on validation performance.

Purpose:
    Train all classical models and select the best performer based on
    validation accuracy or other metrics.

Author: Syed Abbas Ahmad
Date: 2025-11-19
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import cross_val_score

from .svm_classifier import SVMClassifier
from .random_forest import RandomForestClassifier
from .neural_network import MLPClassifier
from .gradient_boosting import GradientBoostingClassifier


class ModelSelector:
    """
    Automated model selection and comparison.

    Trains all classical models and selects best based on validation performance.

    Example:
        >>> selector = ModelSelector()
        >>> results = selector.train_all_models(X_train, y_train, X_val, y_val)
        >>> best_model = selector.select_best_model(results)
        >>> print(f"Best model: {best_model['name']}, accuracy: {best_model['accuracy']:.4f}")
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize model selector.

        Args:
            random_state: Random seed
        """
        self.random_state = random_state

    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        hyperparams: Optional[Dict] = None) -> Dict:
        """
        Train all classical models with default or provided hyperparameters.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            hyperparams: Optional dict of hyperparams for each model

        Returns:
            Dictionary with model names as keys and results as values
        """
        if hyperparams is None:
            hyperparams = {}

        results = {}

        # SVM
        print("Training SVM...")
        svm = SVMClassifier(random_state=self.random_state)
        svm.train(X_train, y_train, hyperparams=hyperparams.get('svm', None))
        svm_acc = svm.score(X_val, y_val)
        results['SVM'] = {
            'model': svm,
            'accuracy': svm_acc,
            'hyperparams': svm.get_params()
        }
        print(f"  SVM validation accuracy: {svm_acc:.4f}")

        # Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(random_state=self.random_state)
        rf.train(X_train, y_train, hyperparams=hyperparams.get('rf', None))
        rf_acc = rf.score(X_val, y_val)
        results['RandomForest'] = {
            'model': rf,
            'accuracy': rf_acc,
            'hyperparams': rf.get_params()
        }
        print(f"  Random Forest validation accuracy: {rf_acc:.4f}")

        # Neural Network
        print("Training Neural Network...")
        nn = MLPClassifier(random_state=self.random_state)
        nn.train(X_train, y_train, hyperparams=hyperparams.get('nn', None))
        nn_acc = nn.score(X_val, y_val)
        results['NeuralNetwork'] = {
            'model': nn,
            'accuracy': nn_acc,
            'hyperparams': nn.get_params()
        }
        print(f"  Neural Network validation accuracy: {nn_acc:.4f}")

        # Gradient Boosting
        print("Training Gradient Boosting...")
        gbm = GradientBoostingClassifier(random_state=self.random_state)
        gbm.train(X_train, y_train, hyperparams=hyperparams.get('gbm', None))
        gbm_acc = gbm.score(X_val, y_val)
        results['GradientBoosting'] = {
            'model': gbm,
            'accuracy': gbm_acc,
            'hyperparams': gbm.get_params()
        }
        print(f"  Gradient Boosting validation accuracy: {gbm_acc:.4f}")

        return results

    def select_best_model(self, results: Dict, metric: str = 'accuracy'):
        """
        Select best model based on specified metric.

        Args:
            results: Dictionary from train_all_models
            metric: Metric to use for selection (default: 'accuracy')

        Returns:
            Dictionary with best model info
        """
        best_name = None
        best_score = -np.inf

        for name, result in results.items():
            score = result[metric]
            if score > best_score:
                best_score = score
                best_name = name

        return {
            'name': best_name,
            'model': results[best_name]['model'],
            'accuracy': results[best_name]['accuracy'],
            'hyperparams': results[best_name]['hyperparams']
        }

    def cross_validate_models(self, X: np.ndarray, y: np.ndarray,
                             cv: int = 5,
                             hyperparams: Optional[Dict] = None) -> Dict:
        """
        Perform k-fold cross-validation for all models.

        Args:
            X: Feature matrix
            y: Labels
            cv: Number of folds
            hyperparams: Optional hyperparameters for each model

        Returns:
            Dictionary with model names and CV scores
        """
        if hyperparams is None:
            hyperparams = {}

        results = {}

        # SVM
        print(f"Cross-validating SVM ({cv} folds)...")
        svm = SVMClassifier(random_state=self.random_state)
        svm.train(X, y, hyperparams=hyperparams.get('svm', None))
        svm_scores = cross_val_score(svm.model, X, y, cv=cv, n_jobs=-1)
        results['SVM'] = {
            'mean_accuracy': np.mean(svm_scores),
            'std_accuracy': np.std(svm_scores),
            'scores': svm_scores
        }
        print(f"  SVM: {np.mean(svm_scores):.4f} +/- {np.std(svm_scores):.4f}")

        # Random Forest
        print(f"Cross-validating Random Forest ({cv} folds)...")
        rf = RandomForestClassifier(random_state=self.random_state)
        rf.train(X, y, hyperparams=hyperparams.get('rf', None))
        rf_scores = cross_val_score(rf.model, X, y, cv=cv, n_jobs=-1)
        results['RandomForest'] = {
            'mean_accuracy': np.mean(rf_scores),
            'std_accuracy': np.std(rf_scores),
            'scores': rf_scores
        }
        print(f"  Random Forest: {np.mean(rf_scores):.4f} +/- {np.std(rf_scores):.4f}")

        # Neural Network
        print(f"Cross-validating Neural Network ({cv} folds)...")
        nn = MLPClassifier(random_state=self.random_state)
        nn.train(X, y, hyperparams=hyperparams.get('nn', None))
        nn_scores = cross_val_score(nn.model, X, y, cv=cv, n_jobs=-1)
        results['NeuralNetwork'] = {
            'mean_accuracy': np.mean(nn_scores),
            'std_accuracy': np.std(nn_scores),
            'scores': nn_scores
        }
        print(f"  Neural Network: {np.mean(nn_scores):.4f} +/- {np.std(nn_scores):.4f}")

        # Gradient Boosting
        print(f"Cross-validating Gradient Boosting ({cv} folds)...")
        gbm = GradientBoostingClassifier(random_state=self.random_state)
        gbm.train(X, y, hyperparams=hyperparams.get('gbm', None))
        gbm_scores = cross_val_score(gbm.model, X, y, cv=cv, n_jobs=-1)
        results['GradientBoosting'] = {
            'mean_accuracy': np.mean(gbm_scores),
            'std_accuracy': np.std(gbm_scores),
            'scores': gbm_scores
        }
        print(f"  Gradient Boosting: {np.mean(gbm_scores):.4f} +/- {np.std(gbm_scores):.4f}")

        return results

    def compare_models_summary(self, results: Dict) -> str:
        """
        Generate summary comparison of models.

        Args:
            results: Results from train_all_models or cross_validate_models

        Returns:
            String summary
        """
        summary = "\n=== Model Comparison Summary ===\n\n"

        # Sort by accuracy
        sorted_results = sorted(results.items(),
                              key=lambda x: x[1].get('accuracy', x[1].get('mean_accuracy', 0)),
                              reverse=True)

        for i, (name, result) in enumerate(sorted_results, 1):
            if 'accuracy' in result:
                summary += f"{i}. {name:20s}: {result['accuracy']:.4f}\n"
            else:
                summary += f"{i}. {name:20s}: {result['mean_accuracy']:.4f} +/- {result['std_accuracy']:.4f}\n"

        return summary
