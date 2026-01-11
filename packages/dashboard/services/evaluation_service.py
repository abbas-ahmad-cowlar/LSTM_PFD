"""
Enhanced Evaluation Service.
Business logic for advanced model evaluation (ROC, error analysis, architecture comparison).
"""
from typing import Dict, List, Optional, Any, Tuple
from database.connection import get_db_session
from models.experiment import Experiment
from utils.logger import setup_logger
import numpy as np
import torch
from pathlib import Path

logger = setup_logger(__name__)


class EvaluationService:
    """Service for advanced model evaluation."""

    @staticmethod
    def load_experiment_predictions(experiment_id: int) -> Optional[Dict]:
        """
        Load predictions and ground truth for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Dictionary with predictions, probabilities, targets, class_names
        """
        try:
            with get_db_session() as session:
                experiment = session.query(Experiment).filter_by(id=experiment_id).first()
                if not experiment:
                    logger.error(f"Experiment {experiment_id} not found")
                    return None

                # Get predictions from experiment results
                config = experiment.config or {}
                results_path = config.get('results_path')

                if not results_path:
                    logger.error(f"No results path for experiment {experiment_id}")
                    return None

                results_path = Path(results_path)
                if not results_path.exists():
                    logger.error(f"Results file not found: {results_path}")
                    return None

                # Load results
                results = torch.load(results_path, map_location='cpu')

                return {
                    'predictions': results.get('predictions'),
                    'probabilities': results.get('probabilities'),
                    'targets': results.get('targets'),
                    'class_names': results.get('class_names', []),
                    'experiment_id': experiment_id,
                    'experiment_name': experiment.name
                }

        except Exception as e:
            logger.error(f"Failed to load predictions: {e}", exc_info=True)
            return None

    @staticmethod
    def generate_roc_data(
        probabilities: np.ndarray,
        targets: np.ndarray,
        class_names: List[str]
    ) -> Dict[str, Any]:
        """
        Generate ROC curve data for all classes.

        Args:
            probabilities: Predicted probabilities [N, num_classes]
            targets: Ground truth labels [N]
            class_names: List of class names

        Returns:
            Dictionary with ROC data per class
        """
        try:
            from evaluation.roc_analyzer import ROCAnalyzer

            analyzer = ROCAnalyzer(probabilities, targets, class_names)

            # Compute ROC curves
            roc_curves = analyzer.compute_roc_curves()

            # Compute AUC scores
            auc_scores = analyzer.compute_auc_scores()

            # Compute macro AUC
            macro_auc = analyzer.compute_macro_auc()

            return {
                'roc_curves': roc_curves,  # {class_name: (fpr, tpr)}
                'auc_scores': auc_scores,  # {class_name: auc}
                'macro_auc': macro_auc
            }

        except Exception as e:
            logger.error(f"Failed to generate ROC data: {e}", exc_info=True)
            return {}

    @staticmethod
    def analyze_errors(
        predictions: np.ndarray,
        probabilities: np.ndarray,
        targets: np.ndarray,
        class_names: List[str]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive error analysis.

        Args:
            predictions: Predicted classes [N]
            probabilities: Predicted probabilities [N, num_classes]
            targets: Ground truth labels [N]
            class_names: List of class names

        Returns:
            Dictionary with error analysis results
        """
        try:
            from sklearn.metrics import confusion_matrix

            # Compute confusion matrix
            cm = confusion_matrix(targets, predictions)

            # Find top confused pairs
            confused_pairs = []
            num_classes = len(class_names)

            for i in range(num_classes):
                for j in range(num_classes):
                    if i != j and cm[i, j] > 0:
                        confused_pairs.append({
                            'true_class': class_names[i],
                            'predicted_class': class_names[j],
                            'count': int(cm[i, j]),
                            'true_idx': i,
                            'pred_idx': j
                        })

            # Sort by count
            confused_pairs = sorted(confused_pairs, key=lambda x: -x['count'])

            # Find misclassified samples
            misclassified_idx = np.where(predictions != targets)[0]
            misclassified_samples = []

            for idx in misclassified_idx[:50]:  # Limit to 50 samples
                misclassified_samples.append({
                    'index': int(idx),
                    'true_class': class_names[targets[idx]],
                    'predicted_class': class_names[predictions[idx]],
                    'confidence': float(probabilities[idx, predictions[idx]]),
                    'true_probability': float(probabilities[idx, targets[idx]])
                })

            return {
                'confusion_matrix': cm.tolist(),
                'confused_pairs': confused_pairs[:20],  # Top 20 pairs
                'misclassified_samples': misclassified_samples,
                'total_errors': len(misclassified_idx),
                'error_rate': len(misclassified_idx) / len(targets) * 100
            }

        except Exception as e:
            logger.error(f"Failed to analyze errors: {e}", exc_info=True)
            return {}

    @staticmethod
    def compare_architectures(experiment_ids: List[int]) -> Dict[str, Any]:
        """
        Compare multiple experiments (architectures).

        Args:
            experiment_ids: List of experiment IDs to compare

        Returns:
            Dictionary with comparison data
        """
        try:
            from evaluation.architecture_comparison import count_parameters

            comparison_data = []

            with get_db_session() as session:
                for exp_id in experiment_ids:
                    experiment = session.query(Experiment).filter_by(id=exp_id).first()
                    if not experiment:
                        continue

                    metrics = experiment.metrics or {}

                    # Get model complexity (if available)
                    config = experiment.config or {}
                    params = config.get('num_parameters', 0)
                    flops = config.get('flops', 0)

                    comparison_data.append({
                        'experiment_id': exp_id,
                        'name': experiment.name,
                        'model_type': experiment.model_type,
                        'test_accuracy': metrics.get('test_accuracy', 0),
                        'test_loss': metrics.get('test_loss', 0),
                        'num_parameters': params,
                        'flops': flops,
                        'training_time': experiment.duration_seconds or 0
                    })

            return {
                'experiments': comparison_data,
                'count': len(comparison_data)
            }

        except Exception as e:
            logger.error(f"Failed to compare architectures: {e}", exc_info=True)
            return {}

    @staticmethod
    def test_robustness(
        experiment_id: int,
        noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2]
    ) -> Dict[str, Any]:
        """
        Test model robustness to noise.

        Args:
            experiment_id: Experiment ID
            noise_levels: List of noise levels (SNR)

        Returns:
            Dictionary with robustness results
        """
        try:
            # This is a simplified implementation
            # In production, you'd load the model and test set, add noise, and evaluate

            results = {
                'experiment_id': experiment_id,
                'noise_levels': noise_levels,
                'accuracies': [],
                'degradation': []
            }

            # Placeholder for actual robustness testing
            # Would use evaluation/robustness_tester.py

            logger.info(f"Robustness testing for experiment {experiment_id}")

            return results

        except Exception as e:
            logger.error(f"Failed to test robustness: {e}", exc_info=True)
            return {}

    @staticmethod
    def cache_evaluation_results(experiment_id: int, evaluation_type: str, results: Dict):
        """
        Cache evaluation results for faster loading.

        Args:
            experiment_id: Experiment ID
            evaluation_type: Type of evaluation (roc, error_analysis, etc.)
            results: Results to cache
        """
        try:
            # Store in Redis or database for caching
            # This is a placeholder
            logger.info(f"Caching {evaluation_type} for experiment {experiment_id}")

        except Exception as e:
            logger.error(f"Failed to cache results: {e}", exc_info=True)

    @staticmethod
    def get_cached_evaluation(experiment_id: int, evaluation_type: str) -> Optional[Dict]:
        """
        Get cached evaluation results.

        Args:
            experiment_id: Experiment ID
            evaluation_type: Type of evaluation

        Returns:
            Cached results or None
        """
        try:
            # Retrieve from cache
            # This is a placeholder
            return None

        except Exception as e:
            logger.error(f"Failed to get cached results: {e}", exc_info=True)
            return None
