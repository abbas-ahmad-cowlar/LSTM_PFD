"""
ROC Curve Analysis

Computes ROC curves and AUC scores for multi-class classification:
- One-vs-rest ROC curves
- AUC scores per class
- Calibration analysis
"""

import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
from typing import Dict, Tuple, List
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH


class ROCAnalyzer:
    """
    Analyze ROC curves for multi-class classification.

    Args:
        probabilities: Predicted probabilities [N, num_classes]
        targets: Ground truth labels [N]
        class_names: List of class names
    """
    def __init__(
        self,
        probabilities: np.ndarray,
        targets: np.ndarray,
        class_names: List[str]
    ):
        self.probs = probabilities
        self.targets = targets
        self.class_names = class_names
        self.num_classes = len(class_names)

    def compute_roc_curves(self) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Compute ROC curve for each class (one-vs-rest).

        Returns:
            Dictionary mapping class name to (fpr, tpr) arrays
        """
        roc_curves = {}

        for i in range(self.num_classes):
            # Binary labels: class i vs rest
            binary_targets = (self.targets == i).astype(int)
            class_probs = self.probs[:, i]

            # Compute ROC curve
            fpr, tpr, _ = roc_curve(binary_targets, class_probs)

            roc_curves[self.class_names[i]] = (fpr, tpr)

        return roc_curves

    def compute_auc_scores(self) -> Dict[str, float]:
        """
        Compute AUC score for each class.

        Returns:
            Dictionary mapping class name to AUC score
        """
        auc_scores = {}

        for i in range(self.num_classes):
            binary_targets = (self.targets == i).astype(int)
            class_probs = self.probs[:, i]

            # Compute AUC
            try:
                auc_score = roc_auc_score(binary_targets, class_probs)
            except ValueError:
                # Handle case where class not present in targets
                auc_score = 0.0

            auc_scores[self.class_names[i]] = auc_score

        return auc_scores

    def compute_macro_auc(self) -> float:
        """Compute macro-averaged AUC."""
        auc_scores = self.compute_auc_scores()
        return np.mean(list(auc_scores.values()))
