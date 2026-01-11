"""
Utility for saving evaluation results in dashboard-compatible format.
Saves predictions, confusion matrices, and metrics for experiment comparison.
"""
import numpy as np
import json
from pathlib import Path
from typing import Dict, Optional
from dashboard_config import STORAGE_RESULTS_DIR


class EvaluationSaver:
    """Save evaluation results for dashboard consumption."""

    @staticmethod
    def save_experiment_results(
        experiment_id: int,
        predictions: np.ndarray,
        labels: np.ndarray,
        probabilities: Optional[np.ndarray] = None,
        confusion_matrix: Optional[np.ndarray] = None,
        metrics: Optional[Dict] = None
    ) -> None:
        """
        Save evaluation results for an experiment.

        This saves results in the format expected by the ComparisonService:
        - predictions.npz: Contains predictions, labels, and probabilities
        - confusion_matrix.npy: Confusion matrix
        - metrics.json: Detailed metrics

        Args:
            experiment_id: Experiment ID
            predictions: Model predictions (1D array of class indices)
            labels: Ground truth labels (1D array of class indices)
            probabilities: Prediction probabilities (2D array, shape: [n_samples, n_classes])
            confusion_matrix: Confusion matrix (2D array)
            metrics: Dictionary of metrics (accuracy, precision, recall, etc.)
        """
        # Create results directory
        results_dir = STORAGE_RESULTS_DIR / f"experiment_{experiment_id}"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        pred_path = results_dir / "predictions.npz"
        if probabilities is not None:
            np.savez(
                str(pred_path),
                predictions=predictions,
                labels=labels,
                probabilities=probabilities
            )
        else:
            np.savez(
                str(pred_path),
                predictions=predictions,
                labels=labels
            )

        print(f"Saved predictions to: {pred_path}")

        # Save confusion matrix
        if confusion_matrix is not None:
            cm_path = results_dir / "confusion_matrix.npy"
            np.save(str(cm_path), confusion_matrix)
            print(f"Saved confusion matrix to: {cm_path}")

        # Save metrics
        if metrics is not None:
            metrics_path = results_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_metrics = {}
                for key, value in metrics.items():
                    if isinstance(value, np.ndarray):
                        json_metrics[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        json_metrics[key] = value.item()
                    elif isinstance(value, dict):
                        # Recursively convert nested dicts
                        json_metrics[key] = EvaluationSaver._convert_dict(value)
                    else:
                        json_metrics[key] = value

                json.dump(json_metrics, f, indent=2)
            print(f"Saved metrics to: {metrics_path}")

    @staticmethod
    def _convert_dict(d: Dict) -> Dict:
        """Convert numpy types in nested dictionary to native Python types."""
        result = {}
        for key, value in d.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                result[key] = value.item()
            elif isinstance(value, dict):
                result[key] = EvaluationSaver._convert_dict(value)
            else:
                result[key] = value
        return result

    @staticmethod
    def save_from_evaluator_results(experiment_id: int, eval_results: Dict) -> None:
        """
        Save results from ModelEvaluator.evaluate() output.

        Args:
            experiment_id: Experiment ID
            eval_results: Dictionary from ModelEvaluator.evaluate()
                Expected keys: predictions, targets, probabilities,
                               confusion_matrix, accuracy, per_class_metrics
        """
        EvaluationSaver.save_experiment_results(
            experiment_id=experiment_id,
            predictions=eval_results['predictions'],
            labels=eval_results['targets'],
            probabilities=eval_results.get('probabilities'),
            confusion_matrix=eval_results.get('confusion_matrix'),
            metrics={
                'accuracy': eval_results.get('accuracy'),
                'per_class': eval_results.get('per_class_metrics', {})
            }
        )


# Example usage in training script:
"""
from evaluation.evaluator import ModelEvaluator
from dash_app.utils.evaluation_saver import EvaluationSaver
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

# After training, evaluate the model
evaluator = ModelEvaluator(model, device='cuda')
eval_results = evaluator.evaluate(test_loader, class_names=FAULT_CLASSES)

# Save results for dashboard
EvaluationSaver.save_from_evaluator_results(
    experiment_id=experiment.id,
    eval_results=eval_results
)
"""
