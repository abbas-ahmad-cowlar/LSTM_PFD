"""
Confusion Matrix Analysis

Deep analysis of confusion patterns to identify:
- Most confused class pairs
- Error concentration
- Misclassification patterns
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH


class ConfusionAnalyzer:
    """
    Analyze confusion matrix for insights.

    Args:
        confusion_matrix: Confusion matrix [num_classes, num_classes]
        class_names: List of class names
    """
    def __init__(self, confusion_matrix: np.ndarray, class_names: List[str]):
        self.cm = confusion_matrix
        self.class_names = class_names
        self.num_classes = len(class_names)

    def find_most_confused_pairs(self, top_k: int = 5) -> List[Tuple[str, str, int]]:
        """
        Find pairs of classes that are most frequently confused.

        Args:
            top_k: Number of top confused pairs to return

        Returns:
            List of (class_i, class_j, count) tuples
        """
        confused_pairs = []

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                if i != j:  # Exclude diagonal
                    count = self.cm[i, j]
                    confused_pairs.append((
                        self.class_names[i],
                        self.class_names[j],
                        int(count)
                    ))

        # Sort by count
        confused_pairs.sort(key=lambda x: x[2], reverse=True)

        return confused_pairs[:top_k]

    def compute_error_concentration(self) -> float:
        """
        Compute percentage of errors concentrated in specific classes.

        Returns:
            Error concentration percentage
        """
        total_errors = self.cm.sum() - np.diag(self.cm).sum()

        if total_errors == 0:
            return 0.0

        # Find class with most errors
        errors_per_class = self.cm.sum(axis=1) - np.diag(self.cm)
        max_errors = errors_per_class.max()

        concentration = 100.0 * max_errors / total_errors

        return concentration

    def analyze_per_class_errors(self) -> pd.DataFrame:
        """
        Analyze errors for each class.

        Returns:
            DataFrame with per-class error analysis
        """
        data = []

        for i in range(self.num_classes):
            total = self.cm[i, :].sum()
            correct = self.cm[i, i]
            errors = total - correct

            data.append({
                'class': self.class_names[i],
                'total': int(total),
                'correct': int(correct),
                'errors': int(errors),
                'accuracy': 100.0 * correct / total if total > 0 else 0
            })

        return pd.DataFrame(data)
