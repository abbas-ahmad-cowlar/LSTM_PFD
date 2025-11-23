"""
Comparison Service for comparing multiple experiments.
Provides data loading, statistical tests, and comparison utilities.
"""
import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from scipy import stats
from scipy.stats import chi2, friedmanchisquare

from database.connection import get_db_session
from models.experiment import Experiment, ExperimentStatus
from models.training_run import TrainingRun
from config import STORAGE_RESULTS_DIR, FAULT_CLASSES
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


class ComparisonService:
    """Service for comparing multiple experiments."""

    @staticmethod
    def validate_comparison_request(experiment_ids: List[int], user_id: Optional[int] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate that comparison request is valid.

        Args:
            experiment_ids: List of experiment IDs to compare
            user_id: ID of requesting user (for authorization, optional for now)

        Returns:
            (valid: bool, error_message: str or None)
        """
        # Validate count
        if len(experiment_ids) < 2:
            return False, "At least 2 experiments required for comparison"

        if len(experiment_ids) > 3:
            return False, "Maximum 3 experiments can be compared. For more, use HPO Campaign analysis."

        # Check for duplicates
        if len(experiment_ids) != len(set(experiment_ids)):
            return False, "Duplicate experiment IDs not allowed"

        # Validate existence
        with get_db_session() as session:
            experiments = session.query(Experiment).filter(
                Experiment.id.in_(experiment_ids)
            ).all()

        if len(experiments) != len(experiment_ids):
            found_ids = {e.id for e in experiments}
            missing = set(experiment_ids) - found_ids
            return False, f"Experiments not found: {list(missing)}"

        # Check if experiments are completed
        incomplete = [e.id for e in experiments if e.status != ExperimentStatus.COMPLETED]
        if incomplete:
            return False, f"Experiments not completed: {incomplete}. Only completed experiments can be compared."

        # Authorization check (if user_id provided)
        if user_id is not None:
            unauthorized = [e.id for e in experiments if e.created_by and e.created_by != user_id]
            if unauthorized:
                return False, f"Unauthorized access to experiments: {unauthorized}"

        return True, None

    @staticmethod
    def get_comparison_data(experiment_ids: List[int]) -> Dict:
        """
        Load all data needed for comparison.

        Args:
            experiment_ids: List of experiment IDs (2-3)

        Returns:
            Dictionary with structure:
            {
                'experiments': [
                    {
                        'id': 1234,
                        'name': 'ResNet34_Standard',
                        'model_type': 'resnet',
                        'created_at': '2025-06-15T14:32:11Z',
                        'status': 'completed',
                        'duration_seconds': 872,
                        'metrics': {...},
                        'per_class_metrics': {...},
                        'confusion_matrix': [[...], [...], ...],
                        'training_history': {...},
                        'config': {...}
                    },
                    ...
                ],
                'statistical_tests': {...}
            }
        """
        from sqlalchemy.orm import selectinload

        with get_db_session() as session:
            # Load experiments with eager loading to prevent N+1 queries
            experiments = session.query(Experiment).options(
                selectinload(Experiment.training_runs)
            ).filter(
                Experiment.id.in_(experiment_ids)
            ).order_by(Experiment.id).all()

            comparison_data = {
                'experiments': [],
                'statistical_tests': {}
            }

            # Load detailed data for each experiment
            for exp in experiments:
                experiment_data = ComparisonService._load_experiment_data(exp, session)
                comparison_data['experiments'].append(experiment_data)

        # Run statistical tests
        if len(experiments) == 2:
            # McNemar's test for pairwise comparison
            comparison_data['statistical_tests']['mcnemar'] = ComparisonService._run_mcnemar_test(
                experiment_ids[0], experiment_ids[1]
            )
        elif len(experiments) == 3:
            # Friedman test for 3 models
            comparison_data['statistical_tests']['friedman'] = ComparisonService._run_friedman_test(
                experiment_ids
            )

        return comparison_data

    @staticmethod
    def _load_experiment_data(exp: Experiment, session) -> Dict:
        """
        Load detailed data for a single experiment.

        Args:
            exp: Experiment model instance
            session: Database session

        Returns:
            Dictionary with experiment data
        """
        # Parse metrics (may be stored as JSON string or dict)
        if isinstance(exp.metrics, str):
            metrics = json.loads(exp.metrics)
        else:
            metrics = exp.metrics or {}

        # Parse config
        if isinstance(exp.config, str):
            config = json.loads(exp.config)
        else:
            config = exp.config or {}

        # Load training history (using eager-loaded relationship to avoid N+1 query)
        training_runs = sorted(exp.training_runs, key=lambda r: r.epoch)

        training_history = {
            'epochs': [run.epoch for run in training_runs],
            'train_loss': [run.train_loss for run in training_runs],
            'val_loss': [run.val_loss for run in training_runs],
            'val_accuracy': [run.val_accuracy for run in training_runs]
        }

        # Load confusion matrix
        confusion_matrix = ComparisonService._load_confusion_matrix(exp.id)

        # Load predictions (for statistical tests)
        predictions_data = ComparisonService._load_predictions(exp.id)

        # Extract per-class metrics
        per_class_metrics = metrics.get('per_class', {})

        # Build experiment data
        experiment_data = {
            'id': exp.id,
            'name': exp.name,
            'model_type': exp.model_type,
            'created_at': exp.created_at.isoformat() if exp.created_at else None,
            'status': exp.status.value,
            'duration_seconds': exp.duration_seconds,
            'metrics': {
                'accuracy': metrics.get('test_accuracy', metrics.get('accuracy', 0)),
                'precision': metrics.get('test_precision', metrics.get('precision', 0)),
                'recall': metrics.get('test_recall', metrics.get('recall', 0)),
                'f1_score': metrics.get('test_f1', metrics.get('f1_score', 0))
            },
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': confusion_matrix,
            'training_history': training_history,
            'config': config,
            'predictions': predictions_data
        }

        return experiment_data

    @staticmethod
    def _load_confusion_matrix(experiment_id: int) -> List[List[int]]:
        """
        Load confusion matrix for an experiment.

        Confusion matrix is stored in:
          storage/results/experiment_{id}/confusion_matrix.npy

        Returns:
            NxN matrix (list of lists) where N is number of classes
        """
        matrix_path = STORAGE_RESULTS_DIR / f"experiment_{experiment_id}" / "confusion_matrix.npy"

        if not matrix_path.exists():
            # Return empty matrix if not found
            n_classes = len(FAULT_CLASSES)
            return [[0] * n_classes for _ in range(n_classes)]

        try:
            matrix = np.load(str(matrix_path))
            return matrix.tolist()
        except Exception as e:
            print(f"Error loading confusion matrix for experiment {experiment_id}: {e}")
            n_classes = len(FAULT_CLASSES)
            return [[0] * n_classes for _ in range(n_classes)]

    @staticmethod
    def _load_predictions(experiment_id: int) -> Optional[Dict]:
        """
        Load predictions for an experiment.

        Predictions are stored in:
          storage/results/experiment_{id}/predictions.npz

        Returns:
            {
                'predictions': array,
                'labels': array,
                'probabilities': array (optional)
            }
        """
        pred_path = STORAGE_RESULTS_DIR / f"experiment_{experiment_id}" / "predictions.npz"

        if not pred_path.exists():
            return None

        try:
            data = np.load(str(pred_path))
            return {
                'predictions': data['predictions'],
                'labels': data['labels'],
                'probabilities': data.get('probabilities', None)
            }
        except Exception as e:
            print(f"Error loading predictions for experiment {experiment_id}: {e}")
            return None

    @staticmethod
    def _run_mcnemar_test(exp1_id: int, exp2_id: int) -> Dict:
        """
        Run McNemar's test for paired comparison of two models.

        McNemar's test: Tests if two models have significantly different error rates.

        Contingency table:
                    Model 2 Correct  Model 2 Wrong
        Model 1 Correct      a              b
        Model 1 Wrong        c              d

        Test statistic: χ² = (b - c)² / (b + c)
        p-value: From chi-square distribution with 1 degree of freedom

        Args:
            exp1_id: First experiment ID
            exp2_id: Second experiment ID

        Returns:
            {
                'test_statistic': float,
                'p_value': float,
                'contingency_table': [[a, b], [c, d]],
                'interpretation': str,
                'significant': bool (p < 0.05)
            }
        """
        # Load predictions for both experiments
        pred1_data = ComparisonService._load_predictions(exp1_id)
        pred2_data = ComparisonService._load_predictions(exp2_id)

        if pred1_data is None or pred2_data is None:
            return {
                'error': 'Prediction files not found for one or both experiments',
                'test_statistic': None,
                'p_value': None,
                'contingency_table': None,
                'interpretation': 'Cannot perform statistical test without prediction data.',
                'significant': False
            }

        y_true = pred1_data['labels']
        y_pred1 = pred1_data['predictions']
        y_pred2 = pred2_data['predictions']

        # Verify same test set
        if not np.array_equal(pred1_data['labels'], pred2_data['labels']):
            return {
                'error': 'Experiments were evaluated on different test sets',
                'test_statistic': None,
                'p_value': None,
                'contingency_table': None,
                'interpretation': 'Cannot compare models trained on different test sets.',
                'significant': False
            }

        # Build contingency table
        correct1 = (y_pred1 == y_true)
        correct2 = (y_pred2 == y_true)

        a = np.sum(correct1 & correct2)  # Both correct
        b = np.sum(correct1 & ~correct2)  # Model 1 correct, Model 2 wrong
        c = np.sum(~correct1 & correct2)  # Model 1 wrong, Model 2 correct
        d = np.sum(~correct1 & ~correct2)  # Both wrong

        contingency_table = [[int(a), int(b)], [int(c), int(d)]]

        # McNemar's test statistic
        if b + c == 0:
            # No disagreements, models are identical
            return {
                'test_statistic': 0.0,
                'p_value': 1.0,
                'contingency_table': contingency_table,
                'interpretation': 'Models make identical predictions (no disagreements).',
                'significant': False
            }

        test_statistic = (b - c) ** 2 / (b + c)

        # p-value from chi-square distribution (1 degree of freedom)
        p_value = 1 - chi2.cdf(test_statistic, df=1)

        # Interpretation
        if p_value < 0.05:
            if b > c:
                winner = f"Experiment {exp1_id}"
            else:
                winner = f"Experiment {exp2_id}"
            interpretation = f"{winner} performs significantly better (p = {p_value:.4f})."
        else:
            interpretation = f"No significant difference between models (p = {p_value:.4f})."

        return {
            'test_statistic': float(test_statistic),
            'p_value': float(p_value),
            'contingency_table': contingency_table,
            'interpretation': interpretation,
            'significant': p_value < 0.05
        }

    @staticmethod
    def _run_friedman_test(experiment_ids: List[int]) -> Dict:
        """
        Run Friedman test for comparing 3 models.

        Friedman test: Non-parametric test for repeated measures (like ANOVA but for ranks).

        Args:
            experiment_ids: List of 3 experiment IDs

        Returns:
            {
                'test_statistic': float,
                'p_value': float,
                'rankings': [1.2, 2.8, 2.0],  # Average rank per model (1=best)
                'interpretation': str,
                'significant': bool
            }
        """
        # Load predictions for all experiments
        all_predictions = []
        y_true = None

        for exp_id in experiment_ids:
            pred_data = ComparisonService._load_predictions(exp_id)

            if pred_data is None:
                return {
                    'error': f'Predictions not found for experiment {exp_id}',
                    'test_statistic': None,
                    'p_value': None,
                    'rankings': None,
                    'interpretation': 'Cannot perform statistical test without prediction data.',
                    'significant': False
                }

            all_predictions.append(pred_data['predictions'])

            if y_true is None:
                y_true = pred_data['labels']
            elif not np.array_equal(y_true, pred_data['labels']):
                return {
                    'error': 'Experiments were evaluated on different test sets',
                    'test_statistic': None,
                    'p_value': None,
                    'rankings': None,
                    'interpretation': 'Cannot compare models trained on different test sets.',
                    'significant': False
                }

        # Compute correctness for each model on each sample
        correctness = []
        for preds in all_predictions:
            correctness.append((preds == y_true).astype(int))

        # Run Friedman test
        statistic, p_value = friedmanchisquare(*correctness)

        # Compute average rankings
        n_samples = len(y_true)
        n_models = len(experiment_ids)

        sample_ranks = []
        for i in range(n_samples):
            sample_correctness = [correctness[m][i] for m in range(n_models)]
            # Rank: Lower rank for correct (higher correctness), higher rank for incorrect
            # We negate to rank correctly (1 is best)
            ranks = stats.rankdata([-c for c in sample_correctness], method='average')
            sample_ranks.append(ranks)

        # Average rank per model
        avg_ranks = np.mean(sample_ranks, axis=0).tolist()

        # Interpretation
        if p_value < 0.05:
            best_model_idx = np.argmin(avg_ranks)
            interpretation = f"Significant difference exists (p = {p_value:.4f}). Experiment {experiment_ids[best_model_idx]} ranks best (avg rank: {avg_ranks[best_model_idx]:.2f})."
        else:
            interpretation = f"No significant difference among models (p = {p_value:.4f})."

        return {
            'test_statistic': float(statistic),
            'p_value': float(p_value),
            'rankings': avg_ranks,
            'interpretation': interpretation,
            'significant': p_value < 0.05
        }

    @staticmethod
    def identify_key_differences(comparison_data: Dict) -> List[str]:
        """
        Automatically identify and highlight key differences between experiments.

        Args:
            comparison_data: Output from get_comparison_data()

        Returns:
            List of human-readable difference descriptions
        """
        experiments = comparison_data['experiments']
        differences = []

        if len(experiments) < 2:
            return differences

        # Accuracy difference
        accuracies = [(exp['id'], exp['name'], exp['metrics']['accuracy'])
                      for exp in experiments]
        accuracies_sorted = sorted(accuracies, key=lambda x: x[2], reverse=True)

        best = accuracies_sorted[0]
        second = accuracies_sorted[1]
        diff_pct = (best[2] - second[2]) * 100

        if diff_pct > 0.5:
            differences.append(
                f"Exp {best[0]} ({best[1]}) is {diff_pct:.2f}% more accurate than Exp {second[0]} ({second[1]})"
            )
        else:
            differences.append(
                f"Models have very similar accuracy (within {diff_pct:.2f}%)"
            )

        # Training time difference
        durations = [(exp['id'], exp['name'], exp['duration_seconds'])
                     for exp in experiments if exp['duration_seconds']]

        if durations:
            fastest = min(durations, key=lambda x: x[2])
            slowest = max(durations, key=lambda x: x[2])
            time_diff_min = (slowest[2] - fastest[2]) / 60

            if time_diff_min > 5:
                differences.append(
                    f"Exp {fastest[0]} trains {time_diff_min:.1f} minutes faster than Exp {slowest[0]}"
                )

        # Per-class performance differences
        for fault_class in FAULT_CLASSES:
            recalls = []
            for exp in experiments:
                class_metrics = exp['per_class_metrics'].get(fault_class, {})
                recall = class_metrics.get('recall', 0)
                recalls.append((exp['id'], exp['name'], recall))

            if recalls:
                best_recall = max(recalls, key=lambda x: x[2])
                worst_recall = min(recalls, key=lambda x: x[2])
                recall_diff = (best_recall[2] - worst_recall[2]) * 100

                if recall_diff > 10:  # >10% difference
                    differences.append(
                        f"Exp {best_recall[0]} excels at {fault_class} detection (+{recall_diff:.1f}% recall vs Exp {worst_recall[0]})"
                    )

        return differences[:5]  # Return top 5 differences
