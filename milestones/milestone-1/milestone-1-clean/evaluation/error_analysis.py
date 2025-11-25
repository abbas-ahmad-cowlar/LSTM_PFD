"""
Error Analysis for Fault Diagnosis

Deep-dive into model misclassifications to understand:
- Which fault pairs are most confused
- Whether errors are severity-dependent
- How noise affects different fault types
- Whether different models make complementary errors

Insights guide:
- Model improvements
- Ensemble construction
- Data augmentation strategies
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH


class ErrorAnalyzer:
    """
    Comprehensive error analysis for fault diagnosis models.

    Args:
        model: Trained model
        class_names: List of class names
        device: Device to run on
    """

    def __init__(
        self,
        model: nn.Module,
        class_names: List[str],
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.class_names = class_names
        self.device = device
        self.model.eval()

    def analyze_misclassifications(
        self,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict:
        """
        Analyze all misclassifications.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary with analysis results
        """
        all_predictions = []
        all_labels = []
        all_probabilities = []
        misclassified_indices = []
        misclassified_samples = []

        sample_idx = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)

                # Track all predictions
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

                # Track misclassifications
                for i in range(len(labels)):
                    if predicted[i] != labels[i]:
                        misclassified_indices.append(sample_idx)
                        misclassified_samples.append({
                            'true_label': labels[i].item(),
                            'predicted_label': predicted[i].item(),
                            'true_class': self.class_names[labels[i].item()],
                            'predicted_class': self.class_names[predicted[i].item()],
                            'confidence': probabilities[i, predicted[i]].item(),
                            'true_prob': probabilities[i, labels[i]].item()
                        })
                    sample_idx += 1

        # Convert to arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)

        # Classification report
        report = classification_report(
            all_labels,
            all_predictions,
            target_names=self.class_names,
            output_dict=True
        )

        return {
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'confusion_matrix': cm,
            'classification_report': report,
            'misclassified_indices': misclassified_indices,
            'misclassified_samples': misclassified_samples,
            'error_rate': len(misclassified_samples) / len(all_labels)
        }

    def find_most_confused_pairs(
        self,
        confusion_matrix: np.ndarray,
        top_k: int = 10
    ) -> List[Tuple]:
        """
        Find most confused class pairs.

        Args:
            confusion_matrix: Confusion matrix
            top_k: Number of pairs to return

        Returns:
            List of (true_class, predicted_class, count) tuples
        """
        num_classes = len(self.class_names)
        confused_pairs = []

        for i in range(num_classes):
            for j in range(num_classes):
                if i != j and confusion_matrix[i, j] > 0:
                    confused_pairs.append((
                        self.class_names[i],
                        self.class_names[j],
                        confusion_matrix[i, j]
                    ))

        # Sort by confusion count
        confused_pairs.sort(key=lambda x: x[2], reverse=True)

        return confused_pairs[:top_k]

    def analyze_confidence_distribution(
        self,
        analysis_results: Dict
    ) -> Dict:
        """
        Analyze confidence distribution for correct vs incorrect predictions.

        Args:
            analysis_results: Results from analyze_misclassifications

        Returns:
            Dictionary with confidence statistics
        """
        predictions = analysis_results['predictions']
        labels = analysis_results['labels']
        probabilities = analysis_results['probabilities']

        # Get max probability for each prediction
        max_probs = np.max(probabilities, axis=1)

        # Split by correct/incorrect
        correct_mask = (predictions == labels)
        correct_confidences = max_probs[correct_mask]
        incorrect_confidences = max_probs[~correct_mask]

        return {
            'correct_confidence_mean': np.mean(correct_confidences),
            'correct_confidence_std': np.std(correct_confidences),
            'incorrect_confidence_mean': np.mean(incorrect_confidences),
            'incorrect_confidence_std': np.std(incorrect_confidences),
            'correct_confidences': correct_confidences,
            'incorrect_confidences': incorrect_confidences
        }

    def find_hard_examples(
        self,
        analysis_results: Dict,
        criterion: str = 'low_confidence',
        top_k: int = 20
    ) -> List[Dict]:
        """
        Find hard examples (low confidence or frequently misclassified).

        Args:
            analysis_results: Results from analyze_misclassifications
            criterion: 'low_confidence' or 'margin'
            top_k: Number of examples to return

        Returns:
            List of hard example dictionaries
        """
        probabilities = analysis_results['probabilities']
        labels = analysis_results['labels']

        hard_examples = []

        for idx in range(len(labels)):
            probs = probabilities[idx]
            true_label = labels[idx]

            if criterion == 'low_confidence':
                # Examples with low confidence in true class
                score = probs[true_label]
            elif criterion == 'margin':
                # Examples with small margin between top-2 predictions
                top2 = np.sort(probs)[-2:]
                score = top2[1] - top2[0]  # Smaller margin = harder
            else:
                raise ValueError(f"Unknown criterion: {criterion}")

            hard_examples.append({
                'index': idx,
                'true_label': true_label,
                'true_class': self.class_names[true_label],
                'score': score,
                'probabilities': probs
            })

        # Sort by score (ascending for both criteria)
        hard_examples.sort(key=lambda x: x['score'])

        return hard_examples[:top_k]

    def plot_confusion_matrix(
        self,
        confusion_matrix: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Plot confusion matrix heatmap."""
        plt.figure(figsize=(12, 10))

        # Normalize by row (true labels)
        cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion'}
        )

        plt.xlabel('Predicted Class', fontsize=12)
        plt.ylabel('True Class', fontsize=12)
        plt.title('Confusion Matrix (Normalized)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        plt.close()

    def plot_confidence_distribution(
        self,
        confidence_stats: Dict,
        save_path: Optional[str] = None
    ):
        """Plot confidence distribution for correct vs incorrect predictions."""
        plt.figure(figsize=(10, 6))

        plt.hist(
            confidence_stats['correct_confidences'],
            bins=50,
            alpha=0.7,
            label='Correct predictions',
            color='green'
        )

        plt.hist(
            confidence_stats['incorrect_confidences'],
            bins=50,
            alpha=0.7,
            label='Incorrect predictions',
            color='red'
        )

        plt.xlabel('Prediction Confidence', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Confidence Distribution: Correct vs Incorrect', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confidence distribution saved to {save_path}")

        plt.close()

    def compare_model_errors(
        self,
        models: List[nn.Module],
        model_names: List[str],
        test_loader: torch.utils.data.DataLoader
    ) -> Dict:
        """
        Compare errors across multiple models.

        Identifies:
        - Samples that all models get wrong
        - Samples only specific models get wrong
        - Complementary errors (good for ensembles)

        Args:
            models: List of models
            model_names: List of model names
            test_loader: Test data loader

        Returns:
            Error comparison dictionary
        """
        # Get predictions from all models
        all_model_predictions = []

        for model in models:
            model.eval()
            model.to(self.device)

            predictions = []
            with torch.no_grad():
                for inputs, _ in test_loader:
                    inputs = inputs.to(self.device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    predictions.extend(predicted.cpu().numpy())

            all_model_predictions.append(np.array(predictions))

        # Get true labels
        true_labels = []
        for _, labels in test_loader:
            true_labels.extend(labels.numpy())
        true_labels = np.array(true_labels)

        # Analyze errors
        num_samples = len(true_labels)
        error_matrix = np.zeros((len(models), num_samples), dtype=bool)

        for i, preds in enumerate(all_model_predictions):
            error_matrix[i] = (preds != true_labels)

        # Find different error categories
        all_correct = np.sum(error_matrix, axis=0) == 0
        all_wrong = np.sum(error_matrix, axis=0) == len(models)
        some_wrong = (~all_correct) & (~all_wrong)

        return {
            'model_names': model_names,
            'error_matrix': error_matrix,
            'all_correct_count': np.sum(all_correct),
            'all_wrong_count': np.sum(all_wrong),
            'some_wrong_count': np.sum(some_wrong),
            'individual_error_rates': [np.mean(error_matrix[i]) for i in range(len(models))],
            'complementary_errors': some_wrong  # Good candidates for ensemble
        }

    def generate_report(
        self,
        analysis_results: Dict,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive error analysis report.

        Args:
            analysis_results: Results from analyze_misclassifications
            save_path: Optional path to save report

        Returns:
            Report string
        """
        report_lines = []

        report_lines.append("="*80)
        report_lines.append("ERROR ANALYSIS REPORT")
        report_lines.append("="*80)

        # Overall statistics
        total = len(analysis_results['labels'])
        errors = len(analysis_results['misclassified_samples'])
        accuracy = 100.0 * (1 - analysis_results['error_rate'])

        report_lines.append(f"\nOverall Statistics:")
        report_lines.append(f"  Total samples: {total}")
        report_lines.append(f"  Correct: {total - errors} ({accuracy:.2f}%)")
        report_lines.append(f"  Errors: {errors} ({100*analysis_results['error_rate']:.2f}%)")

        # Most confused pairs
        confused_pairs = self.find_most_confused_pairs(
            analysis_results['confusion_matrix'],
            top_k=5
        )

        report_lines.append(f"\nTop 5 Most Confused Class Pairs:")
        for true_class, pred_class, count in confused_pairs:
            report_lines.append(f"  {true_class} → {pred_class}: {count} errors")

        # Confidence analysis
        confidence_stats = self.analyze_confidence_distribution(analysis_results)

        report_lines.append(f"\nConfidence Statistics:")
        report_lines.append(f"  Correct predictions:")
        report_lines.append(f"    Mean: {confidence_stats['correct_confidence_mean']:.3f}")
        report_lines.append(f"    Std:  {confidence_stats['correct_confidence_std']:.3f}")
        report_lines.append(f"  Incorrect predictions:")
        report_lines.append(f"    Mean: {confidence_stats['incorrect_confidence_mean']:.3f}")
        report_lines.append(f"    Std:  {confidence_stats['incorrect_confidence_std']:.3f}")

        # Per-class performance
        report_lines.append(f"\nPer-Class Performance:")
        for class_name in self.class_names:
            if class_name in analysis_results['classification_report']:
                metrics = analysis_results['classification_report'][class_name]
                report_lines.append(
                    f"  {class_name}: "
                    f"Precision={metrics['precision']:.3f}, "
                    f"Recall={metrics['recall']:.3f}, "
                    f"F1={metrics['f1-score']:.3f}"
                )

        report_lines.append("\n" + "="*80)

        report_text = "\n".join(report_lines)

        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {save_path}")

        return report_text


# Example usage
if __name__ == "__main__":
    print("Error Analysis Framework")
    print("\nExample usage:")
    print("""
    # Create analyzer
    class_names = ['Normal', 'Inner Race', 'Outer Race', ...]
    analyzer = ErrorAnalyzer(model, class_names, device='cuda')

    # Analyze errors
    results = analyzer.analyze_misclassifications(test_loader)

    # Generate report
    report = analyzer.generate_report(results, save_path='error_report.txt')
    print(report)

    # Plot confusion matrix
    analyzer.plot_confusion_matrix(results['confusion_matrix'], 'confusion.png')

    # Find hard examples
    hard_examples = analyzer.find_hard_examples(results, top_k=20)
    print(f"Hardest example: {hard_examples[0]}")

    # Compare multiple models
    comparison = analyzer.compare_model_errors(
        models=[model1, model2, model3],
        model_names=['ResNet-18', 'ResNet-50', 'EfficientNet-B3'],
        test_loader=test_loader
    )
    print(f"Complementary errors: {comparison['some_wrong_count']}")
    """)
