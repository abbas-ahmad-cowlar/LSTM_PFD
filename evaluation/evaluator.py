"""
Comprehensive Model Evaluation

Provides tools for:
- Test set evaluation
- Per-class metrics
- Classification reports
- Metric aggregation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import numpy as np
from typing import Dict, Optional, Tuple
from tqdm import tqdm


class ModelEvaluator:
    """
    Comprehensive model evaluation on test set.

    Args:
        model: Trained model to evaluate
        device: Device to run evaluation on
    """
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        class_names: Optional[list] = None
    ) -> Dict[str, any]:
        """
        Evaluate model on dataloader.

        Args:
            dataloader: DataLoader for test data
            class_names: Optional list of class names

        Returns:
            Dictionary containing:
            - accuracy: Overall accuracy
            - per_class_metrics: Metrics for each class
            - confusion_matrix: Confusion matrix
            - predictions: All predictions
            - targets: All ground truth labels
        """
        all_predictions = []
        all_targets = []
        all_probs = []

        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            probs = torch.softmax(outputs, dim=1)

            # Get predictions
            _, predicted = outputs.max(1)

            all_predictions.append(predicted.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

        # Concatenate all batches
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        probs = np.concatenate(all_probs)

        # Compute metrics
        accuracy = (predictions == targets).mean() * 100

        # Confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(targets, predictions)

        # Per-class metrics
        per_class_metrics = self.compute_per_class_metrics(
            predictions, targets, probs, class_names
        )

        results = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'per_class_metrics': per_class_metrics,
            'predictions': predictions,
            'targets': targets,
            'probabilities': probs
        }

        return results

    def compute_per_class_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        probs: np.ndarray,
        class_names: Optional[list] = None
    ) -> Dict:
        """
        Compute precision, recall, F1 for each class.

        Args:
            predictions: Predicted labels
            targets: Ground truth labels
            probs: Prediction probabilities
            class_names: Optional class names

        Returns:
            Dictionary of per-class metrics
        """
        from sklearn.metrics import precision_recall_fscore_support
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH

        precision, recall, f1, support = precision_recall_fscore_support(
            targets,
            predictions,
            average=None,
            zero_division=0
        )

        num_classes = len(precision)
        if class_names is None:
            class_names = [f"Class {i}" for i in range(num_classes)]

        per_class = {}
        for i in range(num_classes):
            per_class[class_names[i]] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': int(support[i])
            }

        return per_class

    def generate_classification_report(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        class_names: Optional[list] = None
    ) -> str:
        """
        Generate detailed classification report.

        Args:
            predictions: Predicted labels
            targets: Ground truth labels
            class_names: Optional class names

        Returns:
            Classification report string
        """
        return classification_report(
            targets,
            predictions,
            target_names=class_names,
            zero_division=0
        )
