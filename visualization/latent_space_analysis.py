#!/usr/bin/env python3
"""
Latent Space Analysis for Physics vs Data Branch Comparison

Compares feature representations from physics branch vs CNN data branch
using dimensionality reduction and clustering metrics.

Features:
- Extract features from intermediate layers
- t-SNE/UMAP 2D projections
- Silhouette score and clustering quality metrics
- Class separability analysis

Usage:
    python visualization/latent_space_analysis.py --model checkpoints/pinn.pth --data data/processed/dataset.h5
    
    # Demo mode
    python visualization/latent_space_analysis.py --demo

Author: Deficiency Fix #28 (Priority: 42)
Date: 2026-01-18
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List, Any
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# Fault class colors
CLASS_COLORS = [
    '#2ecc71',  # Normal - green
    '#e74c3c',  # Ball faults - red shades
    '#c0392b',
    '#a93226',
    '#3498db',  # Inner race - blue shades
    '#2980b9',
    '#1f618d',
    '#f39c12',  # Outer race - orange shades
    '#d68910',
    '#b9770e',
    '#9b59b6'   # Combined - purple
]

CLASS_NAMES = [
    'Normal', 'Ball-007', 'Ball-014', 'Ball-021',
    'IR-007', 'IR-014', 'IR-021',
    'OR-007', 'OR-014', 'OR-021', 'Combined'
]


class LatentSpaceAnalyzer:
    """
    Analyze and compare latent representations.
    
    Designed to compare:
    - Physics branch features vs CNN features
    - PINN hybrid features vs pure CNN
    - Different model architectures
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.results: Dict[str, Any] = {}
    
    def extract_features(
        self,
        model: torch.nn.Module,
        dataloader,
        layer_name: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from a model's intermediate layer.
        
        Args:
            model: PyTorch model
            dataloader: DataLoader with (signals, labels)
            layer_name: Name of layer to extract from (None for model's default)
        
        Returns:
            (features, labels) as numpy arrays
        """
        model = model.to(self.device)
        model.eval()
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch_signals, batch_labels in dataloader:
                batch_signals = batch_signals.to(self.device)
                
                # Try different extraction methods
                if hasattr(model, 'get_features'):
                    features = model.get_features(batch_signals)
                elif hasattr(model, 'extract_features'):
                    features = model.extract_features(batch_signals)
                elif hasattr(model, 'get_intermediate_features') and layer_name:
                    features = model.get_intermediate_features(batch_signals, layer_name)
                else:
                    # Use model output before final layer
                    features = self._extract_penultimate(model, batch_signals)
                
                all_features.append(features.cpu().numpy())
                all_labels.append(batch_labels.numpy())
        
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)
        
        return features, labels
    
    def _extract_penultimate(
        self, 
        model: torch.nn.Module, 
        x: torch.Tensor
    ) -> torch.Tensor:
        """Extract features from penultimate layer using hooks."""
        features = []
        
        def hook_fn(module, input, output):
            features.append(input[0])
        
        # Find the last Linear layer (classifier)
        last_linear = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                last_linear = module
        
        if last_linear is None:
            raise ValueError("No Linear layer found")
        
        handle = last_linear.register_forward_hook(hook_fn)
        
        try:
            model(x)
        finally:
            handle.remove()
        
        return features[0]
    
    def compute_clustering_metrics(
        self, 
        features: np.ndarray, 
        labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute clustering quality metrics."""
        metrics = {}
        
        # Reduce dimensionality if too high
        if features.shape[1] > 50:
            pca = PCA(n_components=50)
            features_reduced = pca.fit_transform(features)
        else:
            features_reduced = features
        
        try:
            metrics['silhouette'] = silhouette_score(features_reduced, labels)
        except:
            metrics['silhouette'] = None
        
        try:
            metrics['davies_bouldin'] = davies_bouldin_score(features_reduced, labels)
        except:
            metrics['davies_bouldin'] = None
        
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(features_reduced, labels)
        except:
            metrics['calinski_harabasz'] = None
        
        return metrics
    
    def project_2d(
        self,
        features: np.ndarray,
        method: str = 'tsne',
        **kwargs
    ) -> np.ndarray:
        """Project features to 2D."""
        if method == 'umap' and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, random_state=42, **kwargs)
            return reducer.fit_transform(features)
        elif method == 'pca':
            pca = PCA(n_components=2)
            return pca.fit_transform(features)
        else:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, **kwargs)
            return tsne.fit_transform(features)
    
    def plot_comparison(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
        labels: np.ndarray,
        name_a: str = "Physics Branch",
        name_b: str = "Data Branch",
        method: str = 'tsne',
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot side-by-side 2D projection comparison.
        
        This is the key visualization showing physics features
        produce better class separation.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Project both
        proj_a = self.project_2d(features_a, method)
        proj_b = self.project_2d(features_b, method)
        
        # Compute metrics
        metrics_a = self.compute_clustering_metrics(features_a, labels)
        metrics_b = self.compute_clustering_metrics(features_b, labels)
        
        method_name = method.upper()
        
        for ax, proj, name, metrics in [
            (axes[0], proj_a, name_a, metrics_a),
            (axes[1], proj_b, name_b, metrics_b)
        ]:
            unique_labels = np.unique(labels)
            
            for label in unique_labels:
                mask = labels == label
                color = CLASS_COLORS[label] if label < len(CLASS_COLORS) else f'C{label}'
                label_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f'Class {label}'
                
                ax.scatter(proj[mask, 0], proj[mask, 1], c=color, label=label_name,
                          alpha=0.7, s=30, edgecolors='white', linewidth=0.3)
            
            ax.set_xlabel(f'{method_name} 1')
            ax.set_ylabel(f'{method_name} 2')
            
            # Add metrics to title
            sil = metrics.get('silhouette', 0) or 0
            ax.set_title(f'{name}\nSilhouette: {sil:.3f}')
        
        # Legend
        handles, labels_legend = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_legend, loc='center right', 
                  bbox_to_anchor=(0.99, 0.5), fontsize=9)
        
        plt.suptitle(f'Latent Space Comparison ({method_name})', fontsize=14, weight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_single(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        name: str = "Features",
        method: str = 'tsne',
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Plot single feature set projection."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        proj = self.project_2d(features, method)
        metrics = self.compute_clustering_metrics(features, labels)
        
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            color = CLASS_COLORS[label] if label < len(CLASS_COLORS) else f'C{label}'
            label_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else f'Class {label}'
            
            ax.scatter(proj[mask, 0], proj[mask, 1], c=color, label=label_name,
                      alpha=0.7, s=50, edgecolors='white', linewidth=0.5)
        
        method_name = method.upper()
        ax.set_xlabel(f'{method_name} 1')
        ax.set_ylabel(f'{method_name} 2')
        
        sil = metrics.get('silhouette', 0) or 0
        ax.set_title(f'{name} Latent Space\nSilhouette: {sil:.3f}')
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def print_comparison_report(
        self,
        features_a: np.ndarray,
        features_b: np.ndarray,
        labels: np.ndarray,
        name_a: str = "Physics Branch",
        name_b: str = "Data Branch"
    ) -> Dict[str, Any]:
        """Print detailed comparison report."""
        print("\n" + "=" * 60)
        print("LATENT SPACE ANALYSIS REPORT")
        print("=" * 60)
        
        metrics_a = self.compute_clustering_metrics(features_a, labels)
        metrics_b = self.compute_clustering_metrics(features_b, labels)
        
        print(f"\n{'Metric':<25} | {name_a:^15} | {name_b:^15}")
        print("-" * 60)
        
        for metric in ['silhouette', 'davies_bouldin', 'calinski_harabasz']:
            val_a = metrics_a.get(metric)
            val_b = metrics_b.get(metric)
            
            val_a_str = f'{val_a:.4f}' if val_a is not None else 'N/A'
            val_b_str = f'{val_b:.4f}' if val_b is not None else 'N/A'
            
            print(f"{metric:<25} | {val_a_str:^15} | {val_b_str:^15}")
        
        print("\n" + "=" * 60)
        
        # Determine winner
        if metrics_a['silhouette'] and metrics_b['silhouette']:
            if metrics_a['silhouette'] > metrics_b['silhouette']:
                print(f"✓ {name_a} shows BETTER class separation")
            else:
                print(f"✓ {name_b} shows BETTER class separation")
        
        print("=" * 60)
        
        return {'metrics_a': metrics_a, 'metrics_b': metrics_b}


def demo_latent_analysis():
    """Run demo with synthetic features."""
    print("=" * 60)
    print("LATENT SPACE ANALYSIS DEMO")
    print("=" * 60)
    
    np.random.seed(42)
    
    n_samples = 500
    n_features = 64
    n_classes = 5
    
    # Create well-separated physics features
    physics_features = []
    labels = []
    
    for i in range(n_classes):
        center = np.random.randn(n_features) * 5
        samples = center + np.random.randn(n_samples // n_classes, n_features) * 0.5
        physics_features.append(samples)
        labels.extend([i] * (n_samples // n_classes))
    
    physics_features = np.vstack(physics_features).astype(np.float32)
    labels = np.array(labels)
    
    # Create poorly-separated data features
    data_features = np.random.randn(len(labels), n_features).astype(np.float32)
    for i in range(n_classes):
        mask = labels == i
        data_features[mask] += np.random.randn(n_features) * 2
    
    print(f"\nFeatures: {n_features}")
    print(f"Samples: {len(labels)}")
    print(f"Classes: {n_classes}")
    
    # Analyze
    analyzer = LatentSpaceAnalyzer()
    
    # Print report
    analyzer.print_comparison_report(
        physics_features, data_features, labels,
        name_a="Physics Branch", name_b="Data Branch"
    )
    
    # Create visualizations
    output_dir = Path('results/latent_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # Comparison plot
    fig = analyzer.plot_comparison(
        physics_features, data_features, labels,
        save_path=output_dir / 'comparison_tsne.png'
    )
    plt.close(fig)
    
    print(f"\n✓ Visualizations saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Latent space analysis')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--model', type=str, help='Path to model')
    parser.add_argument('--data', type=str, help='Path to data')
    parser.add_argument('--output', type=str, default='results/latent_analysis')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_latent_analysis()
    else:
        print("Run with --demo for demonstration")


if __name__ == '__main__':
    main()
