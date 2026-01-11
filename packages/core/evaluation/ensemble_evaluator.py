"""
Ensemble Evaluator for Bearing Fault Diagnosis

Comprehensive evaluation of ensemble methods including:
- Accuracy and performance metrics
- Diversity metrics (disagreement, Q-statistic, correlation)
- Per-class performance analysis
- Ensemble vs individual model comparison

Author: Syed Abbas Ahmad
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from itertools import combinations
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH


class EnsembleEvaluator:
    """
    Comprehensive ensemble evaluation.

    Features:
    - Performance metrics (accuracy, F1, precision, recall)
    - Diversity metrics (disagreement, Q-statistic, correlation)
    - Confusion matrix analysis
    - Per-class performance
    - Model contribution analysis

    Args:
        num_classes: Number of classes (default: 11)
        class_names: Optional class names for visualization

    Example:
        >>> evaluator = EnsembleEvaluator(num_classes=NUM_CLASSES)
        >>> results = evaluator.evaluate_ensemble(ensemble, test_loader)
        >>> diversity = evaluator.evaluate_ensemble_diversity(models, test_loader)
    """
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        class_names: Optional[List[str]] = None
    ):
        self.num_classes = num_classes

        if class_names is None:
            class_names = [
                'Normal', 'Ball Fault', 'Inner Race', 'Outer Race',
                'Combined', 'Imbalance', 'Misalignment', 'Oil Whirl',
                'Cavitation', 'Looseness', 'Oil Deficiency'
            ]

        self.class_names = class_names

    def evaluate_ensemble(
        self,
        ensemble: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        device: str = 'cuda',
        save_dir: Optional[Path] = None
    ) -> Dict:
        """
        Evaluate ensemble performance.

        Args:
            ensemble: Ensemble model
            test_loader: Test data loader
            device: Device to use
            save_dir: Optional directory to save results

        Returns:
            Dictionary with evaluation metrics
        """
        ensemble.eval()
        ensemble = ensemble.to(device)

        all_predictions = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)

                logits = ensemble(x)
                probs = F.softmax(logits, dim=1)
                _, predicted = logits.max(1)

                all_predictions.append(predicted.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)
        probs = np.concatenate(all_probs)

        # Compute metrics
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

        # Per-class metrics
        per_class_f1 = f1_score(labels, predictions, average=None, zero_division=0)
        per_class_precision = precision_score(labels, predictions, average=None, zero_division=0)
        per_class_recall = recall_score(labels, predictions, average=None, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)

        results = {
            'accuracy': accuracy * 100,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1_score': f1 * 100,
            'per_class_f1': per_class_f1,
            'per_class_precision': per_class_precision,
            'per_class_recall': per_class_recall,
            'confusion_matrix': cm,
            'predictions': predictions,
            'labels': labels,
            'probabilities': probs
        }

        # Print results
        print("\n" + "="*60)
        print("ENSEMBLE EVALUATION RESULTS")
        print("="*60)
        print(f"Accuracy:  {accuracy*100:.2f}%")
        print(f"Precision: {precision*100:.2f}%")
        print(f"Recall:    {recall*100:.2f}%")
        print(f"F1 Score:  {f1*100:.2f}%")
        print("="*60)

        # Per-class performance
        print("\nPer-Class Performance:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name:20s}: F1={per_class_f1[i]*100:5.2f}% "
                  f"Prec={per_class_precision[i]*100:5.2f}% "
                  f"Rec={per_class_recall[i]*100:5.2f}%")

        # Save results
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            self._plot_confusion_matrix(cm, save_dir / 'confusion_matrix.png')
            self._plot_per_class_metrics(per_class_f1, per_class_precision, per_class_recall,
                                          save_dir / 'per_class_metrics.png')

        return results

    def evaluate_ensemble_diversity(
        self,
        models: List[nn.Module],
        test_loader: torch.utils.data.DataLoader,
        device: str = 'cuda'
    ) -> Dict:
        """
        Measure agreement/disagreement between models (ensemble diversity).

        Higher diversity = better ensemble potential

        Metrics:
        - Pairwise disagreement: Fraction of samples where models disagree
        - Q-statistic: Correlation between model predictions
        - Correlation coefficient: Pearson correlation of predictions

        Args:
            models: List of models
            test_loader: Test data loader
            device: Device to use

        Returns:
            Dictionary with diversity metrics

        Example:
            >>> models = [cnn_model, resnet_model, transformer_model]
            >>> diversity = evaluator.evaluate_ensemble_diversity(models, test_loader)
            >>> print(f"Diversity: {diversity['mean_disagreement']:.2f}")
        """
        num_models = len(models)

        # Get predictions from all models
        all_predictions = []

        for model in models:
            model.eval()
            model = model.to(device)
            model_preds = []

            with torch.no_grad():
                for batch in test_loader:
                    x, _ = batch
                    x = x.to(device)

                    logits = model(x)
                    _, predicted = logits.max(1)
                    model_preds.append(predicted.cpu().numpy())

            all_predictions.append(np.concatenate(model_preds))

        all_predictions = np.array(all_predictions)  # [num_models, num_samples]

        # Pairwise disagreement
        disagreement_matrix = np.zeros((num_models, num_models))

        for i, j in combinations(range(num_models), 2):
            disagreement = (all_predictions[i] != all_predictions[j]).mean()
            disagreement_matrix[i, j] = disagreement
            disagreement_matrix[j, i] = disagreement

        mean_disagreement = disagreement_matrix[np.triu_indices(num_models, k=1)].mean()

        # Q-statistic (pairwise)
        q_statistics = []

        for i, j in combinations(range(num_models), 2):
            pred_i = all_predictions[i]
            pred_j = all_predictions[j]

            # N11: both correct, N00: both wrong, N10: i correct j wrong, N01: i wrong j correct
            # For Q-statistic calculation, we need true labels
            # Here we approximate with agreement/disagreement

            agree = (pred_i == pred_j).sum()
            disagree = (pred_i != pred_j).sum()

            # Simplified Q-statistic based on agreement
            if agree + disagree > 0:
                q = (agree - disagree) / (agree + disagree)
            else:
                q = 0

            q_statistics.append(q)

        mean_q_statistic = np.mean(q_statistics) if q_statistics else 0

        # Correlation coefficient
        correlation_matrix = np.zeros((num_models, num_models))

        for i in range(num_models):
            for j in range(num_models):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    # Pearson correlation
                    corr = np.corrcoef(all_predictions[i], all_predictions[j])[0, 1]
                    correlation_matrix[i, j] = corr

        mean_correlation = correlation_matrix[np.triu_indices(num_models, k=1)].mean()

        results = {
            'num_models': num_models,
            'disagreement_matrix': disagreement_matrix,
            'mean_disagreement': mean_disagreement,
            'q_statistics': q_statistics,
            'mean_q_statistic': mean_q_statistic,
            'correlation_matrix': correlation_matrix,
            'mean_correlation': mean_correlation
        }

        # Print results
        print("\n" + "="*60)
        print("ENSEMBLE DIVERSITY METRICS")
        print("="*60)
        print(f"Number of models: {num_models}")
        print(f"Mean disagreement: {mean_disagreement:.4f} (higher = more diverse)")
        print(f"Mean Q-statistic:  {mean_q_statistic:.4f} (range: [-1, 1])")
        print(f"Mean correlation:  {mean_correlation:.4f} (lower = more diverse)")
        print("="*60)

        print("\nPairwise Disagreement Matrix:")
        for i in range(num_models):
            row = " ".join([f"{disagreement_matrix[i, j]:.3f}" for j in range(num_models)])
            print(f"  Model {i}: {row}")

        return results

    def compare_ensemble_vs_individuals(
        self,
        ensemble: nn.Module,
        individual_models: List[nn.Module],
        test_loader: torch.utils.data.DataLoader,
        device: str = 'cuda',
        model_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Compare ensemble performance vs individual models.

        Args:
            ensemble: Ensemble model
            individual_models: List of individual models
            test_loader: Test data loader
            device: Device to use
            model_names: Optional names for models

        Returns:
            Dictionary with comparison results
        """
        if model_names is None:
            model_names = [f"Model {i+1}" for i in range(len(individual_models))]

        # Evaluate ensemble
        ensemble_results = self.evaluate_ensemble(ensemble, test_loader, device)

        # Evaluate individual models
        individual_results = []

        for i, model in enumerate(individual_models):
            print(f"\nEvaluating {model_names[i]}...")

            model.eval()
            model = model.to(device)

            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for batch in test_loader:
                    x, y = batch
                    x, y = x.to(device), y.to(device)

                    logits = model(x)
                    _, predicted = logits.max(1)

                    all_predictions.append(predicted.cpu().numpy())
                    all_labels.append(y.cpu().numpy())

            predictions = np.concatenate(all_predictions)
            labels = np.concatenate(all_labels)

            accuracy = accuracy_score(labels, predictions)
            f1 = f1_score(labels, predictions, average='weighted', zero_division=0)

            individual_results.append({
                'model_name': model_names[i],
                'accuracy': accuracy * 100,
                'f1_score': f1 * 100
            })

        # Print comparison
        print("\n" + "="*60)
        print("ENSEMBLE VS INDIVIDUAL MODELS")
        print("="*60)
        print(f"{'Model':<25s} {'Accuracy':>10s} {'F1 Score':>10s}")
        print("-"*60)

        for result in individual_results:
            print(f"{result['model_name']:<25s} {result['accuracy']:>9.2f}% {result['f1_score']:>9.2f}%")

        print("-"*60)
        print(f"{'ENSEMBLE':<25s} {ensemble_results['accuracy']:>9.2f}% {ensemble_results['f1_score']:>9.2f}%")
        print("="*60)

        # Calculate improvement
        best_individual_acc = max([r['accuracy'] for r in individual_results])
        improvement = ensemble_results['accuracy'] - best_individual_acc

        print(f"\nEnsemble improvement over best individual: {improvement:+.2f}%")

        return {
            'ensemble': ensemble_results,
            'individuals': individual_results,
            'improvement': improvement
        }

    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: Path):
        """Plot confusion matrix."""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_per_class_metrics(self, f1: np.ndarray, precision: np.ndarray,
                                 recall: np.ndarray, save_path: Path):
        """Plot per-class metrics."""
        x = np.arange(len(self.class_names))
        width = 0.25

        plt.figure(figsize=(14, 6))
        plt.bar(x - width, f1 * 100, width, label='F1 Score')
        plt.bar(x, precision * 100, width, label='Precision')
        plt.bar(x + width, recall * 100, width, label='Recall')

        plt.xlabel('Fault Class')
        plt.ylabel('Score (%)')
        plt.title('Per-Class Performance Metrics')
        plt.xticks(x, self.class_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def evaluate_ensemble_performance(
    ensemble: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> Dict:
    """
    Quick evaluation of ensemble performance.

    Args:
        ensemble: Ensemble model
        test_loader: Test data loader
        device: Device to use

    Returns:
        Dictionary with accuracy and F1 score

    Example:
        >>> results = evaluate_ensemble_performance(ensemble, test_loader)
        >>> print(f"Accuracy: {results['accuracy']:.2f}%")
    """
    evaluator = EnsembleEvaluator()
    results = evaluator.evaluate_ensemble(ensemble, test_loader, device)

    return {
        'accuracy': results['accuracy'],
        'f1_score': results['f1_score']
    }
