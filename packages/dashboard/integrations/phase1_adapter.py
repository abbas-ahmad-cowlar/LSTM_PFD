"""
Phase 1 (Classical ML) integration adapter.
Wraps Phase 1 classical ML training functionality.
"""
import sys
from pathlib import Path
import json

# Add parent directory to path to import Phase 1 modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.logger import setup_logger
from utils.constants import SAMPLING_RATE

logger = setup_logger(__name__)


class Phase1Adapter:
    """Adapter for Phase 1 classical ML training."""

    @staticmethod
    def train(config: dict, progress_callback=None):
        """
        Train classical ML model using Phase 1 pipeline.

        Args:
            config: Training configuration
                - model_type: 'rf', 'svm', or 'gbm'
                - dataset_id: Dataset ID from dashboard database
                - hyperparameters: Model-specific hyperparameters
                - num_epochs: Not used for classical ML (instant training)
                - random_state: Random seed

            progress_callback: Optional callback(epoch, metrics) for progress updates

        Returns:
            Training results dictionary
        """
        try:
            from pipelines.classical_ml_pipeline import ClassicalMLPipeline
            import h5py
            import numpy as np

            logger.info(f"Starting Phase 1 training with config: {config}")

            # Load dataset from HDF5 cache
            cache_path = config.get("cache_path", "data/processed/signals_cache.h5")
            with h5py.File(cache_path, 'r') as f:
                # Load train/val/test splits and concatenate them
                # ClassicalMLPipeline will split internally
                X_train = f['train']['signals'][:]
                y_train = f['train']['labels'][:]
                X_val = f['val']['signals'][:]
                y_val = f['val']['labels'][:]
                X_test = f['test']['signals'][:]
                y_test = f['test']['labels'][:]
                fs = f.attrs.get('sampling_rate', SAMPLING_RATE)

            # Concatenate all signals for pipeline (it will split internally)
            all_signals = np.concatenate([X_train, X_val, X_test], axis=0)
            all_labels = np.concatenate([y_train, y_val, y_test], axis=0)

            # Initialize pipeline (model_type not needed - auto-selected)
            model_type = config.get("model_type", "rf")  # For reporting only
            pipeline = ClassicalMLPipeline(
                random_state=config.get("random_state", 42)
            )

            # Update progress - starting feature extraction
            if progress_callback:
                progress_callback(0, {"status": "Extracting features..."})

            # Run training pipeline
            results = pipeline.run(
                signals=all_signals,
                labels=all_labels,
                fs=fs,
                optimize_hyperparams=config.get("optimize_hyperparams", False),
                n_trials=config.get("n_trials", 50)
            )

            # Update progress - training complete
            if progress_callback:
                progress_callback(1, {
                    "status": "Training complete",
                    "accuracy": results["test_accuracy"]
                })

            logger.info(f"Phase 1 training complete. Test accuracy: {results['test_accuracy']:.4f}")

            # Calculate overall metrics from classification_report
            classification_report = results.get("classification_report", {})
            weighted_avg = classification_report.get("weighted avg", {})

            return {
                "success": True,
                "model_type": model_type,
                "test_accuracy": results["test_accuracy"],
                "test_f1": weighted_avg.get("f1-score", 0),
                "test_precision": weighted_avg.get("precision", 0),
                "test_recall": weighted_avg.get("recall", 0),
                "confusion_matrix": results.get("confusion_matrix", []),
                "selected_features": results.get("selected_features", []),
                "training_time": results.get("elapsed_time_seconds", 0),
                "best_model": results.get("best_model", model_type),
            }

        except Exception as e:
            logger.error(f"Phase 1 training failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    def get_model_params(model_type: str) -> dict:
        """Get default hyperparameters for a classical ML model."""
        defaults = {
            "rf": {
                "n_estimators": 100,
                "max_depth": 20,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
            },
            "svm": {
                "C": 1.0,
                "gamma": "scale",
                "kernel": "rbf",
            },
            "gbm": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 5,
            }
        }
        return defaults.get(model_type, {})
