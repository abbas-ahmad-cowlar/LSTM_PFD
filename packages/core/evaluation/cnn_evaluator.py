"""
CNN model evaluation and testing utilities.

Purpose:
    Comprehensive evaluation of trained CNN models:
    - Test set metrics (accuracy, F1, precision, recall)
    - Confusion matrix analysis
    - Per-class performance breakdown
    - Model inference and predictions
    - Support for both single models and ensembles

Author: Syed Abbas Ahmad
Date: 2025-11-20
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from utils.logging import get_logger

logger = get_logger(__name__)


class CNNEvaluator:
    """
    Evaluator for CNN fault diagnosis models.

    Features:
    - Multi-metric evaluation (accuracy, precision, recall, F1)
    - Confusion matrix computation
    - Per-class performance analysis
    - Batch prediction with progress tracking
    - Result export to JSON

    Args:
        model: Trained CNN model
        device: Device to run evaluation on
        class_names: Optional list of class names for reports

    Example:
        >>> from models.cnn.cnn_1d import CNN1D
        >>> model = CNN1D(num_classes=NUM_CLASSES)
        >>> model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])
        >>>
        >>> evaluator = CNNEvaluator(model, device='cuda')
        >>> metrics = evaluator.evaluate(test_loader)
        >>> print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        class_names: Optional[List[str]] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.class_names = class_names
        self.model.eval()  # Set to evaluation mode

        logger.info(f"CNNEvaluator initialized on {device}")

    def evaluate(
        self,
        test_loader: DataLoader,
        criterion: Optional[nn.Module] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            test_loader: Test data loader
            criterion: Loss function (optional)
            verbose: Show progress bar

        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            iterator = tqdm(test_loader, desc="Evaluating") if verbose else test_loader

            for signals, labels in iterator:
                signals = signals.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(signals)
                _, predicted = outputs.max(1)

                # Collect predictions
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Compute loss if criterion provided
                if criterion is not None:
                    loss = criterion(outputs, labels)
                    total_loss += loss.item() * signals.size(0)
                    total_samples += signals.size(0)

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_labels)

        # Add loss if available
        if criterion is not None:
            metrics['loss'] = total_loss / total_samples

        if verbose:
            self._print_metrics(metrics)

        return metrics

    def _compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            predictions: Predicted labels
            labels: Ground truth labels

        Returns:
            Dictionary with metrics
        """
        # Overall accuracy
        accuracy = accuracy_score(labels, predictions) * 100

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels,
            predictions,
            average=None,
            zero_division=0
        )

        # Macro averages
        precision_macro = precision.mean() * 100
        recall_macro = recall.mean() * 100
        f1_macro = f1.mean() * 100

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)

        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_per_class': precision,
            'recall_per_class': recall,
            'f1_per_class': f1,
            'support_per_class': support,
            'confusion_matrix': cm
        }

    def _print_metrics(self, metrics: Dict[str, float]):
        """Print formatted metrics."""
        logger.info("=" * 60)
        logger.info("Evaluation Results")
        logger.info("=" * 60)
        logger.info(f"Overall Accuracy: {metrics['accuracy']:.2f}%")
        logger.info(f"Macro Precision:  {metrics['precision_macro']:.2f}%")
        logger.info(f"Macro Recall:     {metrics['recall_macro']:.2f}%")
        logger.info(f"Macro F1-Score:   {metrics['f1_macro']:.2f}%")

        if 'loss' in metrics:
            logger.info(f"Test Loss:        {metrics['loss']:.4f}")

        logger.info("=" * 60)

    def get_classification_report(
        self,
        test_loader: DataLoader,
        output_dict: bool = False
    ) -> str:
        """
        Generate sklearn classification report.

        Args:
            test_loader: Test data loader
            output_dict: Return as dictionary instead of string

        Returns:
            Classification report
        """
        all_preds, all_labels = self.predict(test_loader, return_labels=True)

        report = classification_report(
            all_labels,
            all_preds,
            target_names=self.class_names,
            output_dict=output_dict,
            zero_division=0
        )

        return report

    def get_confusion_matrix(
        self,
        test_loader: DataLoader,
        normalize: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute confusion matrix.

        Args:
            test_loader: Test data loader
            normalize: Normalization mode ('true', 'pred', 'all', or None)

        Returns:
            Confusion matrix [num_classes, num_classes]
        """
        all_preds, all_labels = self.predict(test_loader, return_labels=True)

        cm = confusion_matrix(all_labels, all_preds)

        # Normalize if requested
        if normalize == 'true':
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        elif normalize == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
        elif normalize == 'all':
            cm = cm.astype('float') / cm.sum()

        return cm

    def predict(
        self,
        test_loader: DataLoader,
        return_labels: bool = False,
        return_probs: bool = False
    ) -> Tuple[np.ndarray, ...]:
        """
        Generate predictions for test set.

        Args:
            test_loader: Test data loader
            return_labels: Also return ground truth labels
            return_probs: Also return class probabilities

        Returns:
            Predictions (and optionally labels and probabilities)
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for signals, labels in tqdm(test_loader, desc="Predicting"):
                signals = signals.to(self.device)

                # Forward pass
                outputs = self.model(signals)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())

                if return_probs:
                    all_probs.extend(probs.cpu().numpy())

        results = [np.array(all_preds)]

        if return_labels:
            results.append(np.array(all_labels))

        if return_probs:
            results.append(np.array(all_probs))

        return tuple(results) if len(results) > 1 else results[0]

    def predict_single(
        self,
        signal: torch.Tensor,
        return_prob: bool = False
    ) -> Tuple[int, Optional[float]]:
        """
        Predict fault class for a single signal.

        Args:
            signal: Input signal [1, signal_length] or [signal_length]
            return_prob: Also return confidence probability

        Returns:
            Predicted class (and optionally probability)
        """
        self.model.eval()

        # Prepare input
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)  # Add batch dimension

        signal = signal.to(self.device)

        with torch.no_grad():
            output = self.model(signal)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = probs.max(1)

        pred_class = predicted.item()

        if return_prob:
            return pred_class, confidence.item()
        else:
            return pred_class

    def save_results(
        self,
        metrics: Dict,
        save_path: Path,
        include_cm: bool = True
    ):
        """
        Save evaluation results to JSON file.

        Args:
            metrics: Evaluation metrics dictionary
            save_path: Path to save JSON file
            include_cm: Include confusion matrix in output
        """
        # Convert numpy arrays to lists for JSON serialization
        results = {}

        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                if include_cm or key != 'confusion_matrix':
                    results[key] = value.tolist()
            else:
                results[key] = value

        # Save to file
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {save_path}")

    def get_per_class_accuracy(
        self,
        test_loader: DataLoader
    ) -> Dict[int, float]:
        """
        Compute per-class accuracy.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary mapping class index to accuracy
        """
        all_preds, all_labels = self.predict(test_loader, return_labels=True)

        per_class_acc = {}
        for class_idx in np.unique(all_labels):
            mask = all_labels == class_idx
            class_preds = all_preds[mask]
            class_labels = all_labels[mask]
            accuracy = accuracy_score(class_labels, class_preds) * 100
            per_class_acc[int(class_idx)] = accuracy

        return per_class_acc


def test_cnn_evaluator():
    """Test CNN evaluator with dummy data."""
    print("=" * 60)
    print("Testing CNN Evaluator")
    print("=" * 60)

    from data.cnn_dataset import create_cnn_datasets_from_arrays
    from data.cnn_dataloader import create_cnn_dataloaders
    from models.cnn.cnn_1d import CNN1D
    import torch.nn.functional as F

    # Create dummy data
    num_samples = 100
    signal_length = SIGNAL_LENGTH
    num_classes=NUM_CLASSES

    signals = np.random.randn(num_samples, signal_length).astype(np.float32)
    labels = np.random.randint(0, num_classes, num_samples)

    # Create datasets and loaders
    _, _, test_ds = create_cnn_datasets_from_arrays(signals, labels)
    loaders = create_cnn_dataloaders(test_dataset=test_ds, batch_size=8, num_workers=0)

    # Create and train a simple model
    model = CNN1D(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()

    print("\n1. Creating evaluator...")
    class_names = [f"Class_{i}" for i in range(num_classes)]
    evaluator = CNNEvaluator(model, device='cpu', class_names=class_names)

    print("\n2. Evaluating model...")
    metrics = evaluator.evaluate(loaders['test'], criterion=criterion, verbose=True)

    print("\n3. Getting classification report...")
    report = evaluator.get_classification_report(loaders['test'])
    print(report)

    print("\n4. Getting confusion matrix...")
    cm = evaluator.get_confusion_matrix(loaders['test'])
    print(f"   Confusion matrix shape: {cm.shape}")

    print("\n5. Per-class accuracy...")
    per_class_acc = evaluator.get_per_class_accuracy(loaders['test'])
    for class_idx, acc in per_class_acc.items():
        print(f"   Class {class_idx}: {acc:.2f}%")

    print("\n6. Single prediction test...")
    test_signal = torch.randn(102400)
    pred_class, confidence = evaluator.predict_single(test_signal, return_prob=True)
    print(f"   Predicted class: {pred_class}, Confidence: {confidence:.2%}")

    print("\n7. Saving results...")
    save_path = Path('./test_results/evaluation_results.json')
    evaluator.save_results(metrics, save_path)

    # Cleanup
    import shutil
    if Path('./test_results').exists():
        shutil.rmtree('./test_results')
        print("   Test results cleaned up")

    print("\n" + "=" * 60)
    print("✅ All CNN evaluator tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_cnn_evaluator()
