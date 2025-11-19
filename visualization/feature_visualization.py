"""
Feature visualization tools.

Purpose:
    Visualize feature distributions, correlations, and t-SNE clustering.
    Reproduces Figures 4, 5, 6 from technical report.

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional
from sklearn.manifold import TSNE


class FeatureVisualizer:
    """Feature visualization utilities."""

    @staticmethod
    def plot_correlation_matrix(features: np.ndarray,
                                feature_names: List[str],
                                save_path: Optional[Path] = None):
        """
        Plot feature correlation heatmap (Figure 4).

        Args:
            features: Feature matrix
            feature_names: List of feature names
            save_path: Optional path to save figure
        """
        # Compute correlation matrix
        corr = np.corrcoef(features.T)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, xticklabels=feature_names, yticklabels=feature_names,
                   cmap='coolwarm', center=0, ax=ax, cbar_kws={'label': 'Correlation'})
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_feature_distributions(features: np.ndarray, labels: np.ndarray,
                                   feature_names: List[str],
                                   save_path: Optional[Path] = None):
        """
        Plot feature distributions by class (Figure 5).

        Args:
            features: Feature matrix
            labels: Label array
            feature_names: List of feature names
            save_path: Optional path to save figure
        """
        n_features = min(9, features.shape[1])  # Plot top 9 features
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()

        for i in range(n_features):
            ax = axes[i]
            for label in np.unique(labels):
                mask = labels == label
                ax.hist(features[mask, i], alpha=0.6, bins=30, label=f'Class {label}')

            ax.set_title(feature_names[i])
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Count')
            if i == 0:
                ax.legend()

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_tsne_clusters(features: np.ndarray, labels: np.ndarray,
                          save_path: Optional[Path] = None):
        """
        Plot t-SNE visualization of feature clusters (Figure 6).

        Args:
            features: Feature matrix
            labels: Label array
            save_path: Optional path to save figure
        """
        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                           c=labels, cmap='tab10', alpha=0.7, s=50)
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title('t-SNE Feature Clustering')
        plt.colorbar(scatter, ax=ax, label='Class')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig
