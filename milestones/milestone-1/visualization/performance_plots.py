"""
Performance visualization tools.

Purpose:
    Visualize model performance, confusion matrices, ROC curves.
    Reproduces Figures 7, 8, 9 from technical report.

Author: Author Name
Date: 2025-11-19
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional


class PerformancePlotter:
    """Model performance visualization utilities."""

    @staticmethod
    def plot_model_comparison(results: Dict, save_path: Optional[Path] = None):
        """
        Plot model accuracy comparison (Figure 7).

        Args:
            results: Dictionary from ModelSelector with model results
            save_path: Optional path to save figure
        """
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(model_names, accuracies, alpha=0.7, color='steelblue')

        # Annotate bars with values
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{acc:.4f}', ha='center', va='bottom')

        ax.set_ylabel('Validation Accuracy')
        ax.set_title('Model Performance Comparison')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: Optional[List[str]] = None,
                             normalize: bool = True, save_path: Optional[Path] = None):
        """
        Plot confusion matrix (Figure 8).

        Args:
            cm: Confusion matrix
            class_names: Optional class names
            normalize: Normalize by row (true labels)
            save_path: Optional path to save figure
        """
        if normalize:
            cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

        if class_names is None:
            class_names = [str(i) for i in range(len(cm))]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', xticklabels=class_names, yticklabels=class_names,
                   ax=ax, cbar_kws={'label': 'Proportion' if normalize else 'Count'})
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray,
                       class_names: Optional[List[str]] = None,
                       save_path: Optional[Path] = None):
        """
        Plot One-vs-Rest ROC curves (Figure 9).

        Args:
            y_true: True labels
            y_proba: Predicted probabilities (n_samples, n_classes)
            class_names: Optional class names
            save_path: Optional path to save figure
        """
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        n_classes = y_proba.shape[1]
        y_true_bin = label_binarize(y_true, classes=range(n_classes))

        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]

        # Compute ROC curve for each class
        fig, ax = plt.subplots(figsize=(10, 8))

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{class_names[i]} (AUC = {roc_auc:.2f})')

        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves (One-vs-Rest)')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
