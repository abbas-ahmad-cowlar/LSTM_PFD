"""
Phase 1 Adapter – Classical ML Training.

Bridges dashboard Celery tasks to the core classical ML pipeline:
  - ClassicalMLPipeline  (packages/core/pipelines/classical_ml_pipeline.py)

Used by:
  - tasks/training_tasks.py  →  Phase1Adapter.train(config, progress_callback)
    (when model_type is one of: rf, svm, gbm)
"""

import logging
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class Phase1Adapter:
    """
    Static adapter that wraps ClassicalMLPipeline behind the same interface
    the dashboard Celery tasks expect (matching DeepLearningAdapter.train).
    """

    @staticmethod
    def train(
        config: dict,
        progress_callback: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Train a classical ML model.

        Args:
            config: Training configuration from the dashboard.
                Expected keys:
                  - model_type (str): 'rf' | 'svm' | 'gbm'
                  - dataset_path (str): path to HDF5 cache
                  - hyperparameters (dict): optional hyper-parameter overrides
                  - num_epochs (int): treated as n_trials for Bayesian opt
                  - random_seed (int | None): RNG seed
            progress_callback: Optional ``callback(epoch, metrics)``

        Returns:
            dict with keys:
              - success (bool)
              - test_accuracy (float)
              - test_loss (float)           always 0 for classical ML
              - best_val_loss (float)       always 0 for classical ML
              - total_epochs (int)          number of optimization trials
              - best_epoch (int)
              - training_time (float)       seconds
              - precision, recall, f1_score
              - best_model_name (str)
              - model_path (str)
              - error (str)                 only on failure
        """
        try:
            import sys
            project_root = str(Path(__file__).resolve().parents[3])
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from packages.core.pipelines.classical_ml_pipeline import ClassicalMLPipeline
            import h5py

            logger.info(
                f"Phase1Adapter: starting classical ML training "
                f"(model_type={config.get('model_type')})"
            )

            # ---- validate config (optional, non-blocking) ---- #
            try:
                from integration.configuration_validator import validate_model_config
                validate_model_config(
                    config.get('model_type', 'rf'),
                    config.get('hyperparameters', {}),
                )
            except ImportError:
                pass  # validator not available
            except ValueError as val_err:
                return {"success": False, "error": f"Config validation failed: {val_err}"}

            start = time.time()

            # ---- load data ---- #
            cache_path = config.get(
                "dataset_path",
                config.get("cache_path", "data/processed/signals_cache.h5"),
            )

            with h5py.File(cache_path, "r") as f:
                # Try split format first (train/val/test groups)
                if "train" in f:
                    X_train = np.array(f["train"]["signals"], dtype=np.float32)
                    y_train = np.array(f["train"]["labels"], dtype=np.int64)
                    X_val = np.array(f["val"]["signals"], dtype=np.float32)
                    y_val = np.array(f["val"]["labels"], dtype=np.int64)
                    X_test = np.array(f["test"]["signals"], dtype=np.float32)
                    y_test = np.array(f["test"]["labels"], dtype=np.int64)
                    use_existing = True
                else:
                    # Flat format: signals + labels at root
                    X_all = np.array(f["signals"], dtype=np.float32)
                    y_all = np.array(f["labels"], dtype=np.int64)
                    use_existing = False

                fs = float(f.attrs.get("sampling_rate", 20480))

            # ---- configure pipeline ---- #
            seed = config.get("random_seed", 42)
            pipeline = ClassicalMLPipeline(random_state=seed)

            n_trials = config.get("num_epochs", 50)  # map epochs → trials
            hyperparams = config.get("hyperparameters", {})
            optimize = hyperparams.get("optimize", True)

            save_dir = Path(config.get("save_dir", "models/classical"))
            save_dir.mkdir(parents=True, exist_ok=True)

            # Fire initial progress
            if progress_callback:
                try:
                    progress_callback(0, {
                        "train_loss": 0, "val_loss": 0,
                        "train_accuracy": 0, "val_accuracy": 0,
                        "status": "Feature extraction & training...",
                    })
                except Exception:
                    pass

            # ---- run pipeline ---- #
            if use_existing:
                results = pipeline.run(
                    signals=X_train,  # not used when use_existing_splits=True
                    labels=y_train,
                    fs=fs,
                    optimize_hyperparams=optimize,
                    n_trials=n_trials,
                    save_dir=save_dir,
                    use_existing_splits=True,
                    X_train=X_train,
                    X_val=X_val,
                    X_test=X_test,
                    y_train=y_train,
                    y_val=y_val,
                    y_test=y_test,
                )
            else:
                results = pipeline.run(
                    signals=X_all,
                    labels=y_all,
                    fs=fs,
                    optimize_hyperparams=optimize,
                    n_trials=n_trials,
                    save_dir=save_dir,
                )

            elapsed = time.time() - start

            # ---- extract metrics ---- #
            test_acc = float(results.get("test_accuracy", 0))

            # Classification report per-class metrics → macro averages
            report = results.get("classification_report", {})
            if isinstance(report, dict) and "macro avg" in report:
                macro = report["macro avg"]
                precision = float(macro.get("precision", 0))
                recall = float(macro.get("recall", 0))
                f1 = float(macro.get("f1-score", 0))
            else:
                precision, recall, f1 = 0.0, 0.0, 0.0

            # Fire final progress
            if progress_callback:
                try:
                    progress_callback(1, {
                        "train_loss": 0,
                        "val_loss": 0,
                        "train_accuracy": float(results.get("train_accuracy", 0)),
                        "val_accuracy": float(results.get("val_accuracy", 0)),
                        "status": "Training complete",
                    })
                except Exception:
                    pass

            logger.info(
                f"Phase1Adapter: training complete – "
                f"test_accuracy={test_acc:.4f} in {elapsed:.1f}s"
            )

            results = {
                "success": True,
                "test_accuracy": test_acc,
                "test_loss": 0.0,
                "best_val_loss": 0.0,
                "total_epochs": n_trials,
                "best_epoch": 1,
                "training_time": elapsed,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "best_model_name": results.get("best_model_name", config.get("model_type", "unknown")),
                "model_path": str(save_dir),
            }

            # ---- register in ModelRegistry (optional, non-blocking) ---- #
            try:
                import sys as _sys
                _project_root = str(Path(__file__).resolve().parents[3])
                if _project_root not in _sys.path:
                    _sys.path.insert(0, _project_root)
                from integration.model_registry import ModelRegistry
                registry = ModelRegistry()
                registry.auto_register(results, phase='classical')
            except Exception as reg_err:
                logger.debug(f"ModelRegistry registration skipped: {reg_err}")

            return results

        except Exception as exc:
            logger.error(f"Phase1Adapter.train failed: {exc}")
            logger.debug(traceback.format_exc())
            return {
                "success": False,
                "error": str(exc),
                "test_accuracy": 0.0,
                "test_loss": 0.0,
                "best_val_loss": 0.0,
                "total_epochs": 0,
                "best_epoch": 0,
                "training_time": 0.0,
            }
