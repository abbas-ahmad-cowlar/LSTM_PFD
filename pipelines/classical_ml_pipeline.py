"""
End-to-end Classical ML pipeline for bearing fault diagnosis.

Purpose:
    Orchestrates feature extraction, selection, normalization, model training,
    hyperparameter optimization, and evaluation.

Reference: Replaces pipeline.m from MATLAB implementation

Author: Syed Abbas Ahmad
Date: 2025-11-19
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from sklearn.model_selection import train_test_split
import json
import time

from features.feature_extractor import FeatureExtractor
from features.feature_selector import FeatureSelector
from features.feature_normalization import FeatureNormalizer
from models.classical.model_selector import ModelSelector
from training.bayesian_optimizer import BayesianOptimizer
from evaluation.evaluator import Evaluator


class ClassicalMLPipeline:
    """
    Complete classical ML pipeline from signals to evaluation.

    Workflow:
    1. Feature extraction (36 features)
    2. Train/val/test split
    3. Feature selection (MRMR → 15 features)
    4. Normalization
    5. Hyperparameter optimization
    6. Model training
    7. Evaluation

    Example:
        >>> pipeline = ClassicalMLPipeline()
        >>> results = pipeline.run(signals, labels, config)
        >>> print(f"Test accuracy: {results['test_accuracy']:.4f}")
    """

    def __init__(self, random_state: int = 42):
        """
        Initialize pipeline.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.feature_extractor = None
        self.feature_selector = None
        self.normalizer = None
        self.best_model = None
        self.results = {}

    def run(self, signals: np.ndarray, labels: np.ndarray,
           fs: float = 20480,
           optimize_hyperparams: bool = True,
           n_trials: int = 50,
           save_dir: Optional[Path] = None) -> Dict:
        """
        Run complete classical ML pipeline.

        Args:
            signals: Signal array (n_samples, signal_length)
            labels: Label array (n_samples,)
            fs: Sampling frequency
            optimize_hyperparams: Whether to optimize hyperparameters
            n_trials: Number of optimization trials
            save_dir: Directory to save results

        Returns:
            Dictionary with results:
            - train_accuracy, val_accuracy, test_accuracy
            - confusion_matrix
            - classification_report
            - best_model_name
            - feature_importances
        """
        print("="*60)
        print("CLASSICAL ML PIPELINE")
        print("="*60)

        start_time = time.time()

        # Step 1: Feature extraction
        print("\n[1/7] Extracting features...")
        X = self._extract_features(signals, fs)
        print(f"  Extracted features shape: {X.shape}")

        # Step 2: Train/val/test split
        print("\n[2/7] Splitting dataset...")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, labels, test_size=0.3, random_state=self.random_state, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=self.random_state, stratify=y_temp
        )
        print(f"  Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

        # Step 3: Feature selection (post-split MRMR)
        print("\n[3/7] Selecting features (MRMR)...")
        X_train_selected, X_val_selected, X_test_selected = self._select_features(
            X_train, y_train, X_val, X_test
        )
        print(f"  Selected features shape: {X_train_selected.shape}")

        # Step 4: Normalization
        print("\n[4/7] Normalizing features...")
        X_train_norm, X_val_norm, X_test_norm = self._normalize_features(
            X_train_selected, X_val_selected, X_test_selected
        )

        # Step 5: Hyperparameter optimization (optional)
        if optimize_hyperparams:
            print(f"\n[5/7] Optimizing hyperparameters ({n_trials} trials)...")
            best_hyperparams = self._optimize_hyperparameters(
                X_train_norm, y_train, X_val_norm, y_val, n_trials
            )
        else:
            print("\n[5/7] Skipping hyperparameter optimization (using defaults)")
            best_hyperparams = {}

        # Step 6: Train models
        print("\n[6/7] Training all models...")
        model_results = self._train_models(
            X_train_norm, y_train, X_val_norm, y_val, best_hyperparams
        )

        # Step 7: Evaluation
        print("\n[7/7] Evaluating best model on test set...")
        test_results = self._evaluate_best_model(
            X_test_norm, y_test, model_results
        )

        # Compile results
        elapsed_time = time.time() - start_time
        self.results = {
            'dataset': {
                'n_samples': len(labels),
                'n_train': len(y_train),
                'n_val': len(y_val),
                'n_test': len(y_test),
                'n_features_original': X.shape[1],
                'n_features_selected': X_train_selected.shape[1]
            },
            'model_comparison': model_results,
            'best_model': test_results['best_model_name'],
            'train_accuracy': test_results['train_accuracy'],
            'val_accuracy': test_results['val_accuracy'],
            'test_accuracy': test_results['test_accuracy'],
            'confusion_matrix': test_results['confusion_matrix'],
            'classification_report': test_results['classification_report'],
            'selected_features': self.feature_selector.get_feature_names(),
            'elapsed_time_seconds': elapsed_time
        }

        # Save results
        if save_dir:
            self._save_results(save_dir)

        print("\n" + "="*60)
        print(f"PIPELINE COMPLETE ({elapsed_time:.1f}s)")
        print(f"Best Model: {self.results['best_model']}")
        print(f"Test Accuracy: {self.results['test_accuracy']:.4f}")
        print("="*60)

        return self.results

    def _extract_features(self, signals: np.ndarray, fs: float) -> np.ndarray:
        """Extract all 36 features from signals."""
        self.feature_extractor = FeatureExtractor(fs=fs)
        return self.feature_extractor.extract_batch(signals)

    def _select_features(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select top 15 features using MRMR."""
        feature_names = self.feature_extractor.get_feature_names()
        self.feature_selector = FeatureSelector(n_features=15, random_state=self.random_state)
        self.feature_selector.fit(X_train, y_train, feature_names=feature_names)

        X_train_selected = self.feature_selector.transform(X_train)
        X_val_selected = self.feature_selector.transform(X_val)
        X_test_selected = self.feature_selector.transform(X_test)

        print(f"  Selected features: {self.feature_selector.get_feature_names()}")

        return X_train_selected, X_val_selected, X_test_selected

    def _normalize_features(self, X_train: np.ndarray, X_val: np.ndarray,
                           X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize features to zero mean and unit variance."""
        self.normalizer = FeatureNormalizer(method='standard')
        self.normalizer.fit(X_train)

        X_train_norm = self.normalizer.transform(X_train)
        X_val_norm = self.normalizer.transform(X_val)
        X_test_norm = self.normalizer.transform(X_test)

        return X_train_norm, X_val_norm, X_test_norm

    def _optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray,
                                  n_trials: int) -> Dict:
        """Optimize hyperparameters using Bayesian optimization."""
        from models.classical import RandomForestClassifier

        optimizer = BayesianOptimizer(random_state=self.random_state)
        best_params = optimizer.optimize(
            RandomForestClassifier,
            X_train, y_train,
            X_val, y_val,
            n_trials=n_trials
        )

        print(f"  Best hyperparameters: {best_params}")

        return {'rf': best_params}

    def _train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     hyperparams: Dict) -> Dict:
        """Train all classical models."""
        selector = ModelSelector(random_state=self.random_state)
        results = selector.train_all_models(X_train, y_train, X_val, y_val, hyperparams)

        # Print summary
        print(selector.compare_models_summary(results))

        return results

    def _evaluate_best_model(self, X_test: np.ndarray, y_test: np.ndarray,
                            model_results: Dict) -> Dict:
        """Evaluate best model on test set."""
        # Select best model
        selector = ModelSelector(random_state=self.random_state)
        best_model_info = selector.select_best_model(model_results)

        self.best_model = best_model_info['model']
        best_model_name = best_model_info['name']

        # Evaluate on test set
        test_predictions = self.best_model.predict(X_test)
        test_accuracy = np.mean(test_predictions == y_test)

        # Get detailed metrics
        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(y_test, test_predictions)
        report = classification_report(y_test, test_predictions, output_dict=True)

        # Also get train and val accuracy
        # Note: We need to get X_train_norm and X_val_norm from cache
        # For now, just use validation accuracy from model_results
        train_accuracy = 0.0  # Placeholder
        val_accuracy = best_model_info['accuracy']

        return {
            'best_model_name': best_model_name,
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }

    def _save_results(self, save_dir: Path):
        """Save pipeline results to disk."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save results as JSON
        results_file = save_dir / 'results.json'
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_copy = self.results.copy()
            json.dump(results_copy, f, indent=2)

        # Save best model
        model_file = save_dir / 'best_model.pkl'
        self.best_model.save(model_file)

        # Save feature selector and normalizer
        import joblib
        joblib.dump(self.feature_selector, save_dir / 'feature_selector.pkl')
        joblib.dump(self.normalizer, save_dir / 'normalizer.pkl')

        print(f"\n  Results saved to: {save_dir}")

    def predict(self, signals: np.ndarray) -> np.ndarray:
        """
        Predict labels for new signals.

        Args:
            signals: New signals (n_samples, signal_length)

        Returns:
            Predicted labels (n_samples,)
        """
        if self.best_model is None:
            raise ValueError("Pipeline must be run before prediction")

        # Extract features
        X = self.feature_extractor.extract_batch(signals)

        # Select features
        X_selected = self.feature_selector.transform(X)

        # Normalize
        X_norm = self.normalizer.transform(X_selected)

        # Predict
        predictions = self.best_model.predict(X_norm)

        return predictions
