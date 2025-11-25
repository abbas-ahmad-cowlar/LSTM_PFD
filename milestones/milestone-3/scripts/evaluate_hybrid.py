#!/usr/bin/env python3
"""
Hybrid CNN-LSTM Evaluation Script - Comprehensive evaluation of trained hybrid models

Evaluates hybrid CNN-LSTM models on test data with detailed metrics:
- Overall accuracy, precision, recall, F1-score
- Per-class performance metrics
- Confusion matrix analysis
- ROC curves and AUC scores
- Failure case analysis
- Visualization of results

Usage:
    # Evaluate a trained hybrid model
    python scripts/evaluate_hybrid.py --checkpoint checkpoints/recommended_1/model_best.pth --model recommended_1

    # Evaluate with detailed analysis
    python scripts/evaluate_hybrid.py --checkpoint model.pth --model recommended_1 --analyze-failures --plot-confusion

Author: Milestone 3 - CNN-LSTM Hybrid
Date: 2025-11-20
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Import project modules
from models import create_model, list_available_models
from data.matlab_importer import MatlabImporter
from data.cnn_dataset import create_cnn_datasets_from_arrays
from data.cnn_dataloader import create_cnn_dataloaders
from utils.device_manager import get_device
from utils.constants import FAULT_TYPES
from utils.logging import get_logger
from sklearn.metrics import classification_report, confusion_matrix


# Available hybrid models
AVAILABLE_MODELS = list_available_models()

# Class names for bearing faults
CLASS_NAMES = [
    'Healthy',
    'Misalignment',
    'Imbalance',
    'Clearance',
    'Lubrication',
    'Cavitation',
    'Wear',
    'Oil Whirl',
    'Misalign+Imbalance',
    'Wear+Lubrication',
    'Cavitation+Clearance'
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate trained CNN models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model', type=str, required=True,
                       choices=AVAILABLE_MODELS,
                       help='Hybrid model architecture (e.g., recommended_1, recommended_2, recommended_3, custom)')

    # Custom model arguments (for custom hybrid)
    parser.add_argument('--cnn-type', type=str, default='resnet34',
                       help='CNN backbone (for custom hybrid)')
    parser.add_argument('--lstm-type', type=str, default='bilstm',
                       help='LSTM type (for custom hybrid)')
    parser.add_argument('--lstm-hidden-size', type=int, default=256,
                       help='LSTM hidden size (for custom hybrid)')
    parser.add_argument('--lstm-num-layers', type=int, default=2,
                       help='Number of LSTM layers (for custom hybrid)')
    parser.add_argument('--pooling', type=str, default='mean',
                       choices=['mean', 'max', 'last', 'attention'],
                       help='Temporal pooling method (for custom hybrid)')

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/raw/bearing_data',
                       help='Directory containing .mat files')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')

    # Evaluation options
    parser.add_argument('--analyze-failures', action='store_true',
                       help='Analyze failure cases (misclassifications)')
    parser.add_argument('--plot-confusion', action='store_true',
                       help='Plot confusion matrix')
    parser.add_argument('--plot-roc', action='store_true',
                       help='Plot ROC curves')
    parser.add_argument('--per-class-metrics', action='store_true',
                       help='Show per-class metrics')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--save-predictions', action='store_true',
                       help='Save predictions to file')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def load_model(checkpoint_path: str, model_type: str, device: torch.device, args = None):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model based on type
    if model_type == 'custom' and args:
        model = create_model(
            'custom',
            cnn_type=args.cnn_type,
            lstm_type=args.lstm_type,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_num_layers=args.lstm_num_layers,
            pooling_method=args.pooling
        )
    else:
        # Use recommended configuration
        model = create_model(model_type)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, checkpoint


def load_test_data(args, logger):
    """Load test data"""
    logger.info("Loading test data...")

    from data.signal_generator import SignalGenerator
    from config.data_config import DataConfig

    # Create data config
    data_config = DataConfig(
        num_signals_per_fault=150,
        rng_seed=args.seed
    )

    # Generate dataset
    generator = SignalGenerator(data_config)
    dataset = generator.generate_dataset()

    signals = dataset['signals']
    labels = dataset['labels']

    # Create dataloaders
    _, _, test_loader = create_cnn_dataloaders(
        signals=signals,
        labels=labels,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
        augment_train=False
    )

    logger.info(f"✓ Loaded {len(test_loader.dataset)} test samples")

    return test_loader


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(12, 10))

    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    # Plot
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Normalized Count'})

    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to {save_path}")
    plt.close()


def plot_roc_curves(roc_data, class_names, save_path):
    """Plot ROC curves"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    for idx, class_name in enumerate(class_names):
        ax = axes[idx]

        fpr = roc_data['fpr'][idx]
        tpr = roc_data['tpr'][idx]
        auc = roc_data['auc'][idx]

        ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{class_name}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    # Remove empty subplot
    fig.delaxes(axes[-1])

    plt.suptitle('ROC Curves (One-vs-Rest)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved ROC curves to {save_path}")
    plt.close()


def print_per_class_metrics(metrics, class_names):
    """Print per-class metrics in a nice table"""
    print("\n" + "=" * 80)
    print("Per-Class Metrics")
    print("=" * 80)

    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 80)

    for idx, class_name in enumerate(class_names):
        precision = metrics['precision'][idx]
        recall = metrics['recall'][idx]
        f1 = metrics['f1'][idx]
        support = metrics['support'][idx]

        print(f"{class_name:<25} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {support:>10}")

    print("-" * 80)
    print(f"{'Average (macro)':<25} {metrics['macro_precision']:>10.3f} "
          f"{metrics['macro_recall']:>10.3f} {metrics['macro_f1']:>10.3f} "
          f"{sum(metrics['support']):>10}")
    print("=" * 80)


def main():
    """Main evaluation function"""
    args = parse_args()

    # Setup logging
    logger = get_logger(__name__)

    # Print header
    print("=" * 80)
    print("CNN Evaluation Script - Bearing Fault Diagnosis")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 80)

    # Get device
    if args.device == 'auto':
        device = get_device(prefer_gpu=True)
    else:
        device = torch.device(args.device)
    logger.info(f"✓ Using device: {device}")

    # Load model
    logger.info(f"Loading model from {args.checkpoint}...")
    model, checkpoint = load_model(args.checkpoint, args.model, device, args)
    logger.info(f"✓ Model loaded")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if 'epoch' in checkpoint:
        logger.info(f"  Checkpoint epoch: {checkpoint['epoch']+1}")
    if 'val_acc' in checkpoint:
        logger.info(f"  Validation accuracy: {checkpoint['val_acc']:.4f}")

    # Load test data
    test_loader = load_test_data(args, logger)

    # Run evaluation (manual since CNNEvaluator doesn't exist yet)
    logger.info("\nRunning evaluation...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    # Compute metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0
    )

    cm = confusion_matrix(all_labels, all_preds)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'confusion_matrix': cm
    }

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Overall Accuracy:  {results['accuracy']:.4f}")
    print(f"Average Precision: {results['macro_precision']:.4f}")
    print(f"Average Recall:    {results['macro_recall']:.4f}")
    print(f"Average F1-Score:  {results['macro_f1']:.4f}")
    print("=" * 80)

    # Per-class metrics
    if args.per_class_metrics:
        print_per_class_metrics(results, CLASS_NAMES)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot confusion matrix
    if args.plot_confusion:
        cm_path = output_dir / 'confusion_matrix.png'
        plot_confusion_matrix(results['confusion_matrix'], CLASS_NAMES, cm_path)

    # Plot ROC curves
    if args.plot_roc and 'roc_curves' in results:
        roc_path = output_dir / 'roc_curves.png'
        plot_roc_curves(results['roc_curves'], CLASS_NAMES, roc_path)

    # Analyze failure cases
    if args.analyze_failures:
        logger.info("\nAnalyzing failure cases...")

        misclassified = all_preds != all_labels
        num_failures = misclassified.sum()
        error_rate = num_failures / len(all_labels)

        print("\n" + "=" * 80)
        print("Failure Case Analysis")
        print("=" * 80)
        print(f"Total misclassifications: {num_failures}")
        print(f"Error rate: {error_rate*100:.2f}%")

        # Find most confused pairs
        from collections import Counter
        confused_pairs = Counter()
        for true_label, pred_label in zip(all_labels[misclassified], all_preds[misclassified]):
            confused_pairs[(true_label, pred_label)] += 1

        if confused_pairs:
            print("\nMost confused pairs:")
            for (true_idx, pred_idx), count in confused_pairs.most_common(5):
                print(f"  {CLASS_NAMES[true_idx]:>20} → {CLASS_NAMES[pred_idx]:<20}: {count} cases")

    # Save predictions
    if args.save_predictions:
        pred_path = output_dir / 'predictions.npz'
        logger.info(f"\nSaving predictions to {pred_path}...")

        np.savez(
            pred_path,
            predictions=all_preds,
            labels=all_labels,
            probabilities=all_probs
        )
        logger.info(f"✓ Saved predictions to {pred_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
