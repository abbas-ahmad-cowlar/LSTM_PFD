#!/usr/bin/env python3
"""
PDF Report Generation for Experiment Results

Generates publication-ready PDF reports containing:
- Model configuration and architecture summary
- Training metrics and learning curves
- Confusion matrix visualization
- Per-class performance metrics
- Feature importance (if available)

Usage:
    python scripts/utilities/pdf_report.py --results results/experiment.json --output report.pdf
    
    # With training history
    python scripts/utilities/pdf_report.py --results results/cv_results.json --output cv_report.pdf

Author: Deficiency Fix #21 (Priority: 56)
Date: 2026-01-18
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, List

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PDF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from utils.logging import get_logger


logger = get_logger(__name__)


class PDFReportGenerator:
    """
    Generate PDF reports for experiment results.
    
    Supports:
    - Cross-validation results
    - Statistical analysis results
    - Transformer benchmark results
    - General training results
    """
    
    def __init__(self, title: str = "Experiment Report"):
        self.title = title
        self.figures: List[plt.Figure] = []
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'figure.figsize': (8, 6),
            'figure.dpi': 150
        })
    
    def add_title_page(self, results: Dict[str, Any]) -> None:
        """Add title page with experiment summary."""
        fig = plt.figure(figsize=(8.5, 11))
        
        # Title
        fig.text(0.5, 0.85, self.title, fontsize=24, ha='center', weight='bold')
        
        # Timestamp
        timestamp = results.get('timestamp', datetime.now().isoformat())
        fig.text(0.5, 0.78, f"Generated: {timestamp[:19]}", fontsize=10, ha='center', color='gray')
        
        # Config summary
        config = results.get('config', {})
        dataset = results.get('dataset', {})
        
        summary_text = "Experiment Configuration\n" + "=" * 40 + "\n\n"
        
        if config:
            for key, value in config.items():
                summary_text += f"{key}: {value}\n"
        
        summary_text += "\n\nDataset Information\n" + "=" * 40 + "\n\n"
        
        if dataset:
            for key, value in dataset.items():
                if key != 'path':
                    summary_text += f"{key}: {value}\n"
        
        fig.text(0.1, 0.65, summary_text, fontsize=10, family='monospace', va='top')
        
        plt.axis('off')
        self.figures.append(fig)
    
    def add_metrics_summary(self, results: Dict[str, Any]) -> None:
        """Add metrics summary page."""
        fig, ax = plt.subplots(figsize=(8.5, 6))
        
        # Extract metrics based on result type
        if 'aggregated' in results:
            # Cross-validation results
            agg = results['aggregated']
            metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
            means = [agg[m]['mean'] for m in metrics if m in agg]
            stds = [agg[m]['std'] for m in metrics if m in agg]
            labels = [m.replace('_', '\n') for m in metrics if m in agg]
            
            x = np.arange(len(labels))
            bars = ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel('Score')
            ax.set_title('Cross-Validation Results (Mean ± Std)')
            ax.set_ylim(0, 1.1)
            
            # Add value labels
            for bar, mean, std in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                       f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        elif 'statistics' in results:
            # Multi-seed experiment results
            stats = results['statistics']
            metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
            means = [stats[m]['mean'] for m in metrics if m in stats]
            cis = [(stats[m]['ci_95_high'] - stats[m]['ci_95_low'])/2 for m in metrics if m in stats]
            labels = [m.replace('_', '\n') for m in metrics if m in stats]
            
            x = np.arange(len(labels))
            bars = ax.bar(x, means, yerr=cis, capsize=5, color='darkorange', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.set_ylabel('Score')
            ax.set_title(f'Multi-Seed Results (Mean ± 95% CI, n={results.get("num_seeds", "?")})')
            ax.set_ylim(0, 1.1)
        
        elif 'results' in results:
            # Benchmark comparison
            model_names = list(results['results'].keys())
            accuracies = []
            
            for name in model_names:
                r = results['results'][name]
                if 'accuracy' in r:
                    accuracies.append(r['accuracy'])
                else:
                    accuracies.append(0)
            
            x = np.arange(len(model_names))
            bars = ax.bar(x, accuracies, color='seagreen', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.set_ylabel('Accuracy')
            ax.set_title('Model Comparison')
            ax.set_ylim(0, 1.1)
            
            for bar, acc in zip(bars, accuracies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        self.figures.append(fig)
    
    def add_confusion_matrix(self, cm: np.ndarray, labels: Optional[List[str]] = None) -> None:
        """Add confusion matrix visualization."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is None:
            labels = [f'Class {i}' for i in range(len(cm))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=labels, yticklabels=labels)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        self.figures.append(fig)
    
    def add_fold_results(self, results: Dict[str, Any]) -> None:
        """Add per-fold results for cross-validation."""
        if 'fold_results' not in results:
            return
        
        fold_results = results['fold_results']
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        
        metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        colors = ['steelblue', 'darkorange', 'seagreen', 'crimson']
        
        for ax, metric, color in zip(axes.flatten(), metrics, colors):
            values = [f[metric] for f in fold_results]
            folds = [f['fold'] for f in fold_results]
            
            ax.bar(folds, values, color=color, alpha=0.8)
            ax.axhline(np.mean(values), color='black', linestyle='--', label=f'Mean: {np.mean(values):.3f}')
            ax.set_xlabel('Fold')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.legend()
            ax.set_ylim(0, 1.1)
        
        plt.suptitle('Per-Fold Results', fontsize=14, weight='bold')
        plt.tight_layout()
        self.figures.append(fig)
    
    def add_training_curves(self, history: List[Dict[str, float]]) -> None:
        """Add training/validation curves."""
        if not history:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = [h['epoch'] for h in history]
        
        # Accuracy
        if 'train_acc' in history[0]:
            train_acc = [h['train_acc'] for h in history]
            axes[0].plot(epochs, train_acc, label='Train', color='steelblue')
        if 'val_acc' in history[0]:
            val_acc = [h['val_acc'] for h in history]
            axes[0].plot(epochs, val_acc, label='Validation', color='darkorange')
        
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Training Progress')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss (if available)
        if 'train_loss' in history[0]:
            train_loss = [h['train_loss'] for h in history]
            axes[1].plot(epochs, train_loss, label='Train', color='steelblue')
        if 'val_loss' in history[0]:
            val_loss = [h['val_loss'] for h in history]
            axes[1].plot(epochs, val_loss, label='Validation', color='darkorange')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Loss Curves')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].axis('off')
        
        plt.tight_layout()
        self.figures.append(fig)
    
    def save_pdf(self, output_path: Path) -> None:
        """Save all figures to PDF."""
        with PdfPages(output_path) as pdf:
            for fig in self.figures:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        logger.info(f"✓ PDF report saved to: {output_path}")
    
    def generate_from_json(self, json_path: Path, output_path: Path) -> None:
        """Generate complete report from JSON results file."""
        with open(json_path) as f:
            results = json.load(f)
        
        # Title page
        self.add_title_page(results)
        
        # Metrics summary
        self.add_metrics_summary(results)
        
        # Fold results (if CV)
        if 'fold_results' in results:
            self.add_fold_results(results)
            
            # Aggregate confusion matrix
            if results['fold_results'] and 'confusion_matrix' in results['fold_results'][0]:
                cm = np.array(results['fold_results'][-1]['confusion_matrix'])
                self.add_confusion_matrix(cm)
        
        # Training history (if available)
        if 'train_history' in results:
            self.add_training_curves(results['train_history'])
        
        # Save
        self.save_pdf(output_path)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate PDF reports from experiment results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--results', type=str, required=True,
                       help='Path to JSON results file')
    parser.add_argument('--output', type=str, default='report.pdf',
                       help='Output PDF path')
    parser.add_argument('--title', type=str, default='Experiment Report',
                       help='Report title')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    results_path = Path(args.results)
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        sys.exit(1)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generator = PDFReportGenerator(title=args.title)
    generator.generate_from_json(results_path, output_path)


if __name__ == '__main__':
    main()
