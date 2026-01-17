#!/usr/bin/env python3
"""
Feature visualization tools with t-SNE and UMAP.

Purpose:
    Visualize feature distributions, correlations, and clustering.
    Compare physics branch vs CNN branch feature representations.
    Reproduces Figures 4, 5, 6 from technical report.

Features:
    - t-SNE projections for feature clustering
    - UMAP projections for comparison (faster, better global structure)
    - Physics branch vs CNN branch comparison
    - Publication-quality scatter plots

Usage:
    # Compare embeddings from two feature sets
    python visualization/feature_visualization.py \
        --physics-features features_physics.npy \
        --cnn-features features_cnn.npy \
        --labels labels.npy

Author: Syed Abbas Ahmad (enhanced 2026-01-18)
Date: 2025-11-19
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict, Any

from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from utils.constants import NUM_CLASSES


# Publication-quality matplotlib settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (10, 8),
    'figure.dpi': 150
})

# Color palette for fault classes
CLASS_COLORS = [
    '#2ecc71',  # Normal - green
    '#e74c3c',  # Ball fault 1 - red
    '#e74c3c',  # Ball fault 2
    '#e74c3c',  # Ball fault 3
    '#3498db',  # Inner race 1 - blue
    '#3498db',  # Inner race 2
    '#3498db',  # Inner race 3
    '#f39c12',  # Outer race 1 - orange
    '#f39c12',  # Outer race 2
    '#f39c12',  # Outer race 3
    '#9b59b6'   # Combined - purple
]

CLASS_NAMES = [
    'Normal',
    'Ball (0.007)',
    'Ball (0.014)',
    'Ball (0.021)',
    'Inner (0.007)',
    'Inner (0.014)',
    'Inner (0.021)',
    'Outer (0.007)',
    'Outer (0.014)',
    'Outer (0.021)',
    'Combined'
]


class FeatureVisualizer:
    """Feature visualization utilities with t-SNE and UMAP support."""

    @staticmethod
    def plot_correlation_matrix(features: np.ndarray,
                                feature_names: List[str],
                                save_path: Optional[Path] = None):
        """
        Plot feature correlation heatmap (Figure 4).

        Args:
            features: Feature matrix [N, D]
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
            features: Feature matrix [N, D]
            labels: Label array [N]
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
                          save_path: Optional[Path] = None,
                          perplexity: int = 30,
                          title: str = 't-SNE Feature Clustering'):
        """
        Plot t-SNE visualization of feature clusters (Figure 6).

        Args:
            features: Feature matrix [N, D]
            labels: Label array [N]
            save_path: Optional path to save figure
            perplexity: t-SNE perplexity parameter
            title: Plot title
        """
        # Compute t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        features_2d = tsne.fit_transform(features)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            color = CLASS_COLORS[label] if label < len(CLASS_COLORS) else f'C{label}'
            name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f'Class {label}'
            
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c=color, alpha=0.7, s=50, label=name, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, features_2d

    @staticmethod
    def plot_umap_clusters(features: np.ndarray, labels: np.ndarray,
                          save_path: Optional[Path] = None,
                          n_neighbors: int = 15,
                          min_dist: float = 0.1,
                          title: str = 'UMAP Feature Clustering'):
        """
        Plot UMAP visualization of feature clusters.
        
        UMAP advantages over t-SNE:
        - Faster computation
        - Better preservation of global structure
        - More interpretable distances

        Args:
            features: Feature matrix [N, D]
            labels: Label array [N]
            save_path: Optional path to save figure
            n_neighbors: UMAP neighborhood size
            min_dist: Minimum distance between points
            title: Plot title
        """
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not installed. Run: pip install umap-learn")
        
        # Compute UMAP
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, 
                           min_dist=min_dist, random_state=42)
        features_2d = reducer.fit_transform(features)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        for label in unique_labels:
            mask = labels == label
            color = CLASS_COLORS[label] if label < len(CLASS_COLORS) else f'C{label}'
            name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f'Class {label}'
            
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c=color, alpha=0.7, s=50, label=name, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel('UMAP Component 1')
        ax.set_ylabel('UMAP Component 2')
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig, features_2d

    @staticmethod
    def compare_embeddings(
        features_a: np.ndarray,
        features_b: np.ndarray,
        labels: np.ndarray,
        name_a: str = 'Physics Branch',
        name_b: str = 'CNN Branch',
        method: str = 'tsne',
        save_path: Optional[Path] = None
    ):
        """
        Compare two feature embeddings side-by-side.
        
        This is the key visualization to show physics features
        cluster faults better than raw CNN features.

        Args:
            features_a: First feature set [N, D1]
            features_b: Second feature set [N, D2]
            labels: Label array [N]
            name_a: Name for first set
            name_b: Name for second set
            method: 'tsne' or 'umap'
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Compute embeddings
        if method == 'umap' and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, random_state=42)
            embed_a = reducer.fit_transform(features_a)
            embed_b = reducer.fit_transform(features_b)
            method_name = 'UMAP'
        else:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            embed_a = tsne.fit_transform(features_a)
            embed_b = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(features_b)
            method_name = 't-SNE'
        
        # Plot both
        unique_labels = np.unique(labels)
        
        for ax, embed, name in [(axes[0], embed_a, name_a), (axes[1], embed_b, name_b)]:
            for label in unique_labels:
                mask = labels == label
                color = CLASS_COLORS[label] if label < len(CLASS_COLORS) else f'C{label}'
                label_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f'Class {label}'
                
                ax.scatter(embed[mask, 0], embed[mask, 1],
                          c=color, alpha=0.7, s=50, label=label_name,
                          edgecolors='white', linewidth=0.5)
            
            ax.set_xlabel(f'{method_name} Component 1')
            ax.set_ylabel(f'{method_name} Component 2')
            ax.set_title(f'{name}')
        
        # Single legend for both
        handles, labels_legend = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_legend, loc='center right', bbox_to_anchor=(0.99, 0.5))
        
        fig.suptitle(f'{method_name} Comparison: {name_a} vs {name_b}', fontsize=18, y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @staticmethod
    def compute_clustering_metrics(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Compute clustering quality metrics.
        
        Returns:
            Dictionary with silhouette score, Davies-Bouldin index, etc.
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        
        metrics = {}
        
        try:
            metrics['silhouette'] = silhouette_score(features, labels)
        except:
            metrics['silhouette'] = None
            
        try:
            metrics['davies_bouldin'] = davies_bouldin_score(features, labels)
        except:
            metrics['davies_bouldin'] = None
            
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(features, labels)
        except:
            metrics['calinski_harabasz'] = None
        
        return metrics


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Feature visualization with t-SNE and UMAP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--physics-features', type=str, default=None,
                       help='Path to physics branch features (.npy)')
    parser.add_argument('--cnn-features', type=str, default=None,
                       help='Path to CNN branch features (.npy)')
    parser.add_argument('--labels', type=str, default=None,
                       help='Path to labels (.npy)')
    parser.add_argument('--method', type=str, default='tsne',
                       choices=['tsne', 'umap', 'both'],
                       help='Dimensionality reduction method')
    parser.add_argument('--output-dir', type=str, default='results/visualizations',
                       help='Output directory for plots')
    parser.add_argument('--demo', action='store_true',
                       help='Run with synthetic demo data')
    
    return parser.parse_args()


def demo_visualization():
    """Run demonstration with synthetic data."""
    print("=" * 60)
    print("FEATURE VISUALIZATION DEMO")
    print("=" * 60)
    
    # Generate synthetic features
    np.random.seed(42)
    n_samples = 500
    n_features = 64
    n_classes = 5
    
    # Create clustered features (simulating physics features)
    physics_features = []
    labels = []
    
    for i in range(n_classes):
        cluster_center = np.random.randn(n_features) * 3
        cluster_samples = cluster_center + np.random.randn(n_samples // n_classes, n_features) * 0.5
        physics_features.append(cluster_samples)
        labels.extend([i] * (n_samples // n_classes))
    
    physics_features = np.vstack(physics_features)
    labels = np.array(labels)
    
    # Create less clustered features (simulating raw CNN)
    cnn_features = np.random.randn(len(labels), n_features)
    for i in range(n_classes):
        mask = labels == i
        cnn_features[mask] += np.random.randn(n_features) * 1.5
    
    print(f"\nGenerated {len(labels)} samples, {n_classes} classes")
    print(f"Physics features shape: {physics_features.shape}")
    print(f"CNN features shape: {cnn_features.shape}")
    
    # Compute clustering metrics
    viz = FeatureVisualizer()
    
    print("\nClustering Metrics:")
    print("-" * 40)
    
    physics_metrics = viz.compute_clustering_metrics(physics_features, labels)
    cnn_metrics = viz.compute_clustering_metrics(cnn_features, labels)
    
    print(f"  Physics Branch - Silhouette: {physics_metrics['silhouette']:.3f}")
    print(f"  CNN Branch     - Silhouette: {cnn_metrics['silhouette']:.3f}")
    
    # Create output directory
    output_dir = Path('results/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate t-SNE plots
    print("\nGenerating t-SNE visualizations...")
    viz.plot_tsne_clusters(physics_features, labels, 
                          output_dir / 'demo_tsne_physics.png',
                          title='Physics Branch Features (t-SNE)')
    viz.plot_tsne_clusters(cnn_features, labels,
                          output_dir / 'demo_tsne_cnn.png',
                          title='CNN Branch Features (t-SNE)')
    
    # Generate comparison
    print("Generating comparison plot...")
    viz.compare_embeddings(physics_features, cnn_features, labels,
                          name_a='Physics Branch', name_b='CNN Branch',
                          method='tsne',
                          save_path=output_dir / 'demo_comparison_tsne.png')
    
    # UMAP if available
    if UMAP_AVAILABLE:
        print("Generating UMAP visualizations...")
        viz.plot_umap_clusters(physics_features, labels,
                              output_dir / 'demo_umap_physics.png',
                              title='Physics Branch Features (UMAP)')
        viz.compare_embeddings(physics_features, cnn_features, labels,
                              method='umap',
                              save_path=output_dir / 'demo_comparison_umap.png')
    else:
        print("⚠ UMAP not installed. Run: pip install umap-learn")
    
    print(f"\n✓ Visualizations saved to {output_dir}")
    print("=" * 60)


def main():
    args = parse_args()
    
    if args.demo:
        demo_visualization()
        return
    
    if args.physics_features and args.cnn_features and args.labels:
        # Load data
        physics_features = np.load(args.physics_features)
        cnn_features = np.load(args.cnn_features)
        labels = np.load(args.labels)
        
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        viz = FeatureVisualizer()
        
        # Generate visualizations
        viz.compare_embeddings(
            physics_features, cnn_features, labels,
            method=args.method,
            save_path=output_dir / f'comparison_{args.method}.png'
        )
        
        print(f"✓ Saved to {output_dir}")
    else:
        print("Run with --demo for demonstration, or provide --physics-features, --cnn-features, --labels")


if __name__ == '__main__':
    main()

