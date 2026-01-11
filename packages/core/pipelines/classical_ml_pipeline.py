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
from packages.core.models.classical.model_selector import ModelSelector
from packages.core.models.classical import (
    RandomForestClassifier,
    SVMClassifier,
    MLPClassifier,
    GradientBoostingClassifier
)
from training.bayesian_optimizer import BayesianOptimizer


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
           save_dir: Optional[Path] = None,
           use_existing_splits: bool = False,
           X_train: Optional[np.ndarray] = None,
           X_val: Optional[np.ndarray] = None,
           X_test: Optional[np.ndarray] = None,
           y_train: Optional[np.ndarray] = None,
           y_val: Optional[np.ndarray] = None,
           y_test: Optional[np.ndarray] = None) -> Dict:
        """
        Run complete classical ML pipeline.

        Args:
            signals: Signal array (n_samples, signal_length)
            labels: Label array (n_samples,)
            fs: Sampling frequency
            optimize_hyperparams: Whether to optimize hyperparameters
            n_trials: Number of optimization trials per model
            save_dir: Directory to save results
            use_existing_splits: If True, use provided X_train/X_val/X_test/y_train/y_val/y_test
            X_train, X_val, X_test: Optional pre-split feature arrays (if use_existing_splits=True)
            y_train, y_val, y_test: Optional pre-split label arrays (if use_existing_splits=True)

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

        # Step 1: Feature extraction or use existing feature splits
        if use_existing_splits and all(arr is not None for arr in [X_train, X_val, X_test, y_train, y_val, y_test]):
            print("\n[1/7] Using existing feature splits (skipping extraction and re-splitting)...")
            # Validate existing splits
            train_count = len(y_train)
            val_count = len(y_val)
            test_count = len(y_test)
            
            if X_train.shape[0] != train_count or X_val.shape[0] != val_count or X_test.shape[0] != test_count:
                raise ValueError("Mismatch between feature and label array sizes in existing splits")
            
            if X_train.shape[1] != X_val.shape[1] or X_train.shape[1] != X_test.shape[1]:
                raise ValueError("Feature dimensions must match across splits")
            
            print(f"  Train: {train_count} samples, Val: {val_count} samples, Test: {test_count} samples")
            print(f"  Feature dimensions: {X_train.shape[1]}")
            print(f"  Note: Using provided splits (no feature extraction or re-splitting)")
            # Store original feature count for results
            n_features_original = X_train.shape[1]
        else:
            print("\n[1/7] Extracting features...")
            X = self._extract_features(signals, fs)
            print(f"  Extracted features shape: {X.shape}")
            n_features_original = X.shape[1]

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
        if use_existing_splits and all(arr is not None for arr in [X_train, X_val, X_test, y_train, y_val, y_test]):
            # Skip step numbering adjustment for existing splits flow
            step_num = 2
        else:
            step_num = 3
        
        print(f"\n[{step_num}/7] Selecting features (MRMR)...")
        X_train_selected, X_val_selected, X_test_selected = self._select_features(
            X_train, y_train, X_val, X_test
        )
        print(f"  Selected features shape: {X_train_selected.shape}")

        # Step 4: Normalization
        print(f"\n[{step_num + 1}/7] Normalizing features...")
        X_train_norm, X_val_norm, X_test_norm = self._normalize_features(
            X_train_selected, X_val_selected, X_test_selected
        )

        # Step 5: Hyperparameter optimization (optional)
        if optimize_hyperparams:
            print(f"\n[{step_num + 2}/7] Optimizing hyperparameters ({n_trials} trials per model)...")
            best_hyperparams = self._optimize_hyperparameters(
                X_train_norm, y_train, X_val_norm, y_val, n_trials,
                optimize_all_models=True  # Optimize all models, not just RF
            )
        else:
            print(f"\n[{step_num + 2}/7] Skipping hyperparameter optimization (using defaults)")
            best_hyperparams = {}

        # Step 6: Train models
        print(f"\n[{step_num + 3}/7] Training all models...")
        model_results = self._train_models(
            X_train_norm, y_train, X_val_norm, y_val, best_hyperparams
        )

        # Step 7: Evaluation
        print(f"\n[{step_num + 4}/7] Evaluating best model on test set...")
        test_results = self._evaluate_best_model(
            X_train_norm, y_train, X_test_norm, y_test, model_results
        )

        # Compile results
        elapsed_time = time.time() - start_time
        
        # Get selected features safely
        selected_features = None
        if self.feature_selector is not None:
            selected_features = self.feature_selector.get_feature_names()
        
        self.results = {
            'dataset': {
                'n_samples': len(labels),
                'n_train': len(y_train),
                'n_val': len(y_val),
                'n_test': len(y_test),
                'n_features_original': n_features_original,
                'n_features_selected': X_train_selected.shape[1]
            },
            'model_comparison': model_results,
            'best_model': test_results['best_model_name'],
            'train_accuracy': float(test_results['train_accuracy']),
            'val_accuracy': float(test_results['val_accuracy']),
            'test_accuracy': float(test_results['test_accuracy']),
            'confusion_matrix': test_results['confusion_matrix'],
            'classification_report': test_results['classification_report'],
            'selected_features': selected_features,
            'elapsed_time_seconds': float(elapsed_time)
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
        # Get feature names if feature_extractor is available, otherwise use generic names
        if self.feature_extractor is not None:
            feature_names = self.feature_extractor.get_feature_names()
        else:
            # Generate generic feature names when using existing splits
            feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
        
        self.feature_selector = FeatureSelector(n_features=15, random_state=self.random_state)
        self.feature_selector.fit(X_train, y_train, feature_names=feature_names)

        X_train_selected = self.feature_selector.transform(X_train)
        X_val_selected = self.feature_selector.transform(X_val)
        X_test_selected = self.feature_selector.transform(X_test)

        selected_names = self.feature_selector.get_feature_names()
        if selected_names:
            print(f"  Selected features: {selected_names}")

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
                                  n_trials: int,
                                  optimize_all_models: bool = True) -> Dict:
        """
        Optimize hyperparameters using Bayesian optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            n_trials: Number of optimization trials per model
            optimize_all_models: If True, optimize all models; if False, only RF
        
        Returns:
            Dictionary with optimized hyperparameters for each model
        """
        optimizer = BayesianOptimizer(random_state=self.random_state)
        all_best_params = {}
        
        if optimize_all_models:
            # Optimize all models
            model_classes = {
                'svm': SVMClassifier,
                'rf': RandomForestClassifier,
                'nn': MLPClassifier,
                'gbm': GradientBoostingClassifier
            }
            
            print(f"  Optimizing hyperparameters for {len(model_classes)} models...")
            
            for model_key, model_class in model_classes.items():
                model_name = model_class.__name__
                print(f"    Optimizing {model_name} ({n_trials} trials)...")
                
                try:
                    best_params = optimizer.optimize(
                        model_class,
                        X_train, y_train,
                        X_val, y_val,
                        n_trials=n_trials
                    )
                    all_best_params[model_key] = best_params
                    print(f"      ✓ {model_name} optimization complete")
                except Exception as e:
                    print(f"      ⚠ {model_name} optimization failed: {e}")
                    print(f"      Using default hyperparameters for {model_name}")
                    all_best_params[model_key] = None
        else:
            # Legacy: Only optimize RF
            print(f"  Optimizing Random Forest only ({n_trials} trials)...")
            best_params = optimizer.optimize(
                RandomForestClassifier,
                X_train, y_train,
                X_val, y_val,
                n_trials=n_trials
            )
            all_best_params['rf'] = best_params

        print(f"  Optimization summary:")
        for model_key, params in all_best_params.items():
            if params is not None:
                print(f"    {model_key.upper()}: {params}")

        return all_best_params

    def _train_models(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_val: np.ndarray, y_val: np.ndarray,
                     hyperparams: Dict) -> Dict:
        """Train all classical models."""
        selector = ModelSelector(random_state=self.random_state)
        results = selector.train_all_models(X_train, y_train, X_val, y_val, hyperparams)

        # Print summary
        print(selector.compare_models_summary(results))

        return results

    def _evaluate_best_model(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            model_results: Dict) -> Dict:
        """Evaluate best model on train and test sets."""
        # Select best model
        selector = ModelSelector(random_state=self.random_state)
        best_model_info = selector.select_best_model(model_results)

        self.best_model = best_model_info['model']
        best_model_name = best_model_info['name']

        # Evaluate on train set
        train_predictions = self.best_model.predict(X_train)
        train_accuracy = np.mean(train_predictions == y_train)

        # Evaluate on test set
        test_predictions = self.best_model.predict(X_test)
        test_accuracy = np.mean(test_predictions == y_test)

        # Get detailed metrics
        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(y_test, test_predictions)
        report = classification_report(y_test, test_predictions, output_dict=True)

        # Val accuracy from validation during training
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
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError) as e:
            raise RuntimeError(f"Failed to create save directory {save_dir}: {e}") from e

        # Create JSON-serializable results dict explicitly
        json_results = self._create_serializable_results()

        # Save results as JSON
        results_file = save_dir / 'results.json'
        try:
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2)
        except (IOError, OSError) as e:
            raise RuntimeError(f"Failed to save results.json to {results_file}: {e}") from e

        # Save best model (if available)
        if self.best_model is not None:
            model_file = save_dir / 'best_model.pkl'
            try:
                self.best_model.save(model_file)
            except Exception as e:
                raise RuntimeError(f"Failed to save best model to {model_file}: {e}") from e
        else:
            print("  Warning: best_model is None, skipping model save")

        # Save feature selector and normalizer (if available)
        import joblib
        if self.feature_selector is not None:
            try:
                joblib.dump(self.feature_selector, save_dir / 'feature_selector.pkl')
            except Exception as e:
                print(f"  Warning: Failed to save feature_selector: {e}")
        else:
            print("  Warning: feature_selector is None, skipping save")
            
        if self.normalizer is not None:
            try:
                joblib.dump(self.normalizer, save_dir / 'normalizer.pkl')
            except Exception as e:
                print(f"  Warning: Failed to save normalizer: {e}")
        else:
            print("  Warning: normalizer is None, skipping save")

        print(f"\n  Results saved to: {save_dir}")

    def _create_serializable_results(self) -> Dict:
        """
        Create a JSON-serializable version of results.
        
        Explicitly extracts only serializable data, avoiding recursive cleaning
        which can miss edge cases.
        
        Returns:
            Dictionary with only JSON-serializable data
        """
        # Start with base results
        json_results = {
            'dataset': self.results.get('dataset', {}).copy(),
            'best_model': self.results.get('best_model'),
            'train_accuracy': float(self.results.get('train_accuracy', 0.0)),
            'val_accuracy': float(self.results.get('val_accuracy', 0.0)),
            'test_accuracy': float(self.results.get('test_accuracy', 0.0)),
            'elapsed_time_seconds': float(self.results.get('elapsed_time_seconds', 0.0)),
            'selected_features': self.results.get('selected_features'),
        }
        
        # Handle confusion matrix (convert numpy array to list)
        cm = self.results.get('confusion_matrix')
        if cm is not None:
            if isinstance(cm, np.ndarray):
                json_results['confusion_matrix'] = cm.tolist()
            elif isinstance(cm, list):
                json_results['confusion_matrix'] = cm
            else:
                json_results['confusion_matrix'] = None
        else:
            json_results['confusion_matrix'] = None
        
        # Handle classification report (recursively convert numpy types)
        report = self.results.get('classification_report')
        if report is not None:
            json_results['classification_report'] = self._convert_numpy_types(report)
        else:
            json_results['classification_report'] = None
        
        # Handle model_comparison - extract only serializable metadata
        model_comparison = self.results.get('model_comparison', {})
        json_model_comparison = {}
        for model_name, model_data in model_comparison.items():
            if isinstance(model_data, dict):
                json_model_comparison[model_name] = {
                    'accuracy': float(model_data.get('accuracy', 0.0)),
                    'hyperparams': self._convert_numpy_types(model_data.get('hyperparams', {}))
                }
            else:
                # Fallback if structure is unexpected
                json_model_comparison[model_name] = str(model_data)
        
        json_results['model_comparison'] = json_model_comparison
        
        return json_results
    
    def _convert_numpy_types(self, obj):
        """
        Recursively convert numpy types to Python native types.
        
        Compatible with both NumPy 1.x and 2.x.
        
        Args:
            obj: Object that may contain numpy types
            
        Returns:
            Object with all numpy types converted to Python types
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Check for numpy integer types (base class works in both NumPy 1.x and 2.x)
        elif isinstance(obj, np.integer):
            return int(obj)
        # Check for numpy floating types (base class works in both NumPy 1.x and 2.x)
        elif isinstance(obj, np.floating):
            return float(obj)
        # Check for numpy boolean
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # For unknown types, try to convert to string
            try:
                # Test if it's JSON serializable
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

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
        
        if self.feature_extractor is None:
            raise ValueError(
                "Feature extractor not available. This occurs when pipeline was run "
                "with use_existing_splits=True. To use predict(), run the pipeline "
                "normally (without use_existing_splits) or manually extract features first."
            )
        
        if self.feature_selector is None:
            raise ValueError("Feature selector not available. Pipeline must be run before prediction.")
        
        if self.normalizer is None:
            raise ValueError("Normalizer not available. Pipeline must be run before prediction.")

        # Extract features
        X = self.feature_extractor.extract_batch(signals)

        # Select features
        X_selected = self.feature_selector.transform(X)

        # Normalize
        X_norm = self.normalizer.transform(X_selected)

        # Predict
        predictions = self.best_model.predict(X_norm)

        return predictions
