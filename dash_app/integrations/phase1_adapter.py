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
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

            logger.info(f"Starting Phase 1 training with config: {config}")

            # Load dataset from HDF5 cache
            cache_path = config.get("cache_path", "data/processed/signals_cache.h5")
            with h5py.File(cache_path, 'r') as f:
                # Load train/val/test splits
                X_train = f['train']['signals'][:]
                y_train = f['train']['labels'][:]
                X_val = f['val']['signals'][:]
                y_val = f['val']['labels'][:]
                X_test = f['test']['signals'][:]
                y_test = f['test']['labels'][:]
                fs = f.attrs.get('sampling_rate', SAMPLING_RATE)

            # Initialize pipeline
            model_type = config["model_type"]
            pipeline = ClassicalMLPipeline(
                model_type=model_type,
                random_state=config.get("random_state", 42)
            )

            # Update progress - starting feature extraction
            if progress_callback:
                progress_callback(0, {"status": "Extracting features..."})

            # Run training pipeline
            results = pipeline.run(
                signals=X_train,
                labels=y_train,
                fs=fs,
                val_signals=X_val,
                val_labels=y_val,
                test_signals=X_test,
                test_labels=y_test,
                optimize_hyperparams=config.get("optimize_hyperparams", False),
                n_trials=config.get("n_trials", 50)
            )

            # Update progress - training complete
            if progress_callback:
                progress_callback(1, {
                    "status": "Training complete",
                    "accuracy": results["test_accuracy"],
                    "f1_score": results.get("test_f1", 0)
                })

            logger.info(f"Phase 1 training complete. Test accuracy: {results['test_accuracy']:.4f}")

            return {
                "success": True,
                "model_type": model_type,
                "test_accuracy": results["test_accuracy"],
                "test_f1": results.get("test_f1", 0),
                "test_precision": results.get("test_precision", 0),
                "test_recall": results.get("test_recall", 0),
                "confusion_matrix": results.get("confusion_matrix", []),
                "feature_importance": results.get("feature_importance", {}),
                "training_time": results.get("training_time", 0),
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
