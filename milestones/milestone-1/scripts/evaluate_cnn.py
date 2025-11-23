#!/usr/bin/env python3
"""
CNN Evaluation Script - Comprehensive evaluation of trained CNN models

Evaluates CNN models on test data with detailed metrics:
- Overall accuracy, precision, recall, F1-score
- Per-class performance metrics
- Confusion matrix analysis
- ROC curves and AUC scores
- Failure case analysis
- Visualization of results

Usage:
    # Evaluate a trained model
    python scripts/evaluate_cnn.py --checkpoint checkpoints/cnn1d/model_best.pth

    # Evaluate with detailed analysis
    python scripts/evaluate_cnn.py --checkpoint model.pth --analyze-failures --plot-confusion

    # Compare multiple models
    python scripts/evaluate_cnn.py --checkpoints model1.pth model2.pth model3.pth --compare

Author: Phase 2 - CNN Implementation
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
from models.cnn.cnn_1d import CNN1D
from models.cnn.attention_cnn import AttentionCNN1D, LightweightAttentionCNN
from models.cnn.multi_scale_cnn import MultiScaleCNN1D, DilatedMultiScaleCNN
from evaluation.cnn_evaluator import CNNEvaluator
from data.cnn_dataloader import create_cnn_dataloaders
from utils.device_manager import get_device
from utils.logging import get_logger
from utils.constants import NUM_CLASSES


# Model registry
MODEL_REGISTRY = {
    'cnn1d': CNN1D,
    'attention': AttentionCNN1D,
    'attention-lite': LightweightAttentionCNN,
    'multiscale': MultiScaleCNN1D,
    'dilated': DilatedMultiScaleCNN
}

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
    parser.add_argument('--model', type=str, default=None,
                       choices=list(MODEL_REGISTRY.keys()),
                       help='Model architecture (auto-detected if None)')

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/processed',
                       help='Directory with processed data')
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


def load_model(checkpoint_path: str, model_type: str = None, device: torch.device = None):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Try to detect model type from checkpoint
    if model_type is None:
        if 'args' in checkpoint and 'model' in checkpoint['args']:
            model_type = checkpoint['args']['model']
        else:
            raise ValueError("Model type not specified and cannot be auto-detected. "
                           "Please specify --model argument.")

    # Get model class
    model_class = MODEL_REGISTRY[model_type]

    # Get number of classes from checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Find classifier layer
        for key in state_dict.keys():
            if 'classifier' in key and 'weight' in key and key.endswith('.weight'):
                num_classes = state_dict[key].shape[0]
                break
        else:
            num_classes = NUM_CLASSES  # Default if not found
    else:
        num_classes = NUM_CLASSES  # Default

    # Create model with correct parameters for each architecture
    if model_type == 'cnn1d':
        # CNN1D uses: num_classes, input_channels, dropout, use_batch_norm
        model = model_class(
            num_classes=num_classes,
            input_channels=1,
            dropout=0.3,
            use_batch_norm=True
        )
    elif model_type in ['attention', 'attention-lite']:
        # AttentionCNN uses: num_classes, input_length, in_channels, dropout
        model = model_class(
            num_classes=num_classes,
            input_length=102400,
            in_channels=1,
            dropout=0.3
        )
    elif model_type in ['multiscale', 'dilated']:
        # MultiScaleCNN uses: num_classes, input_length, in_channels, dropout
        model = model_class(
            num_classes=num_classes,
            input_length=102400,
            in_channels=1,
            dropout=0.3
        )
    else:
        # Default for other models
        model = model_class(
            num_classes=num_classes,
            input_length=102400,
            in_channels=1
        )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint


def load_test_data(args, logger):
    """Load test data"""
    logger.info("Loading test data...")

    from data.matlab_importer import load_mat_dataset
    from data.cnn_dataset import RawSignalDataset
    from sklearn.model_selection import train_test_split

    # Load .mat files
    logger.info("Loading .MAT files...")
    signals, labels, label_names = load_mat_dataset(args.data_dir)

    logger.info(f"✓ Loaded {len(signals)} signals")
    logger.info(f"  Signal shape: {signals.shape}")
    logger.info(f"  Classes: {len(np.unique(labels))}")

    # Split into train/val/test (70/15/15) - we only need test
    train_signals, temp_signals, train_labels, temp_labels = train_test_split(
        signals, labels, test_size=0.3, random_state=args.seed, stratify=labels
    )
    _, test_signals, _, test_labels = train_test_split(
        temp_signals, temp_labels, test_size=0.5, random_state=args.seed, stratify=temp_labels
    )

    # Create test dataset
    test_dataset = RawSignalDataset(test_signals, test_labels)

    logger.info(f"✓ Created test dataset: {len(test_dataset)} samples")

    # Create test dataloader
    loaders = create_cnn_dataloaders(
        train_dataset=test_dataset,  # Using test as train to get a dataloader
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    test_loader = loaders['train']

    logger.info(f"✓ Test loader ready: {len(test_loader)} batches")

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
    model, checkpoint = load_model(args.checkpoint, args.model, device)
    logger.info(f"✓ Model loaded")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if 'epoch' in checkpoint:
        logger.info(f"  Checkpoint epoch: {checkpoint['epoch']+1}")
    if 'val_acc' in checkpoint:
        logger.info(f"  Validation accuracy: {checkpoint['val_acc']:.4f}")

    # Load test data
    test_loader = load_test_data(args, logger)

    # Create evaluator
    evaluator = CNNEvaluator(model, device=device)

    # Run evaluation
    logger.info("\nRunning evaluation...")
    results = evaluator.evaluate(
        test_loader,
        class_names=CLASS_NAMES,
        compute_roc=args.plot_roc
    )

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
        failure_analysis = evaluator.analyze_failures(
            test_loader,
            class_names=CLASS_NAMES,
            num_cases=20
        )

        print("\n" + "=" * 80)
        print("Failure Case Analysis")
        print("=" * 80)
        print(f"Total misclassifications: {failure_analysis['num_failures']}")
        print(f"Error rate: {failure_analysis['error_rate']*100:.2f}%")

        if failure_analysis['most_confused_pairs']:
            print("\nMost confused pairs:")
            for (true_idx, pred_idx), count in failure_analysis['most_confused_pairs'][:5]:
                print(f"  {CLASS_NAMES[true_idx]:>20} → {CLASS_NAMES[pred_idx]:<20}: {count} cases")

    # Save predictions
    if args.save_predictions:
        pred_path = output_dir / 'predictions.npz'
        logger.info(f"\nSaving predictions to {pred_path}...")

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

        np.savez(
            pred_path,
            predictions=np.concatenate(all_preds),
            labels=np.concatenate(all_labels),
            probabilities=np.concatenate(all_probs)
        )
        logger.info(f"✓ Saved predictions to {pred_path}")

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Complete!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
