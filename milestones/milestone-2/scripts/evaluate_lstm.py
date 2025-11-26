#!/usr/bin/env python3
"""
LSTM Evaluation Script - Comprehensive evaluation of trained LSTM models

Evaluates LSTM models on test data with detailed metrics:
- Overall accuracy, precision, recall, F1-score
- Per-class performance metrics
- Confusion matrix analysis
- Failure case analysis
- Visualization of results

Usage:
    # Evaluate a trained model
    python scripts/evaluate_lstm.py --checkpoint checkpoints/lstm/model_best.pth

    # Evaluate with detailed analysis
    python scripts/evaluate_lstm.py --checkpoint model.pth --analyze-failures --plot-confusion

Author: Bearing Fault Diagnosis Team
Milestone: 2 - LSTM Implementation
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
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Import project modules
from models import create_model
from data.lstm_dataloader import create_lstm_dataloaders
from training.lstm_trainer import LSTMTrainer
from utils.device_manager import get_device
from utils.constants import NUM_CLASSES, FAULT_TYPES


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate trained LSTM models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='bilstm',
                       choices=['vanilla_lstm', 'lstm', 'bilstm'],
                       help='Model architecture')

    # Data arguments
    parser.add_argument('--data-dir', type=str, default='data/raw/bearing_data',
                       help='Directory with test data')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--signal-length', type=int, default=102400,
                       help='Signal length')

    # Evaluation options
    parser.add_argument('--analyze-failures', action='store_true',
                       help='Analyze failure cases (misclassifications)')
    parser.add_argument('--plot-confusion', action='store_true',
                       help='Plot confusion matrix')
    parser.add_argument('--per-class-metrics', action='store_true',
                       help='Show per-class metrics')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='results/evaluation/lstm',
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


def load_model(checkpoint_path: str, model_type: str, device: str) -> torch.nn.Module:
    """Load model from checkpoint"""
    print(f"\nLoading model from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = create_model(
        model_name=model_type,
        num_classes=NUM_CLASSES,
        hidden_size=128,  # Default, will be overwritten by checkpoint
        num_layers=2,
        dropout=0.3
    )

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")

    # Print checkpoint info if available
    if isinstance(checkpoint, dict):
        if 'epoch' in checkpoint:
            print(f"  Checkpoint epoch: {checkpoint['epoch']}")
        if 'val_acc' in checkpoint or 'accuracy' in checkpoint:
            acc = checkpoint.get('val_acc', checkpoint.get('accuracy', 'N/A'))
            print(f"  Validation accuracy: {acc:.2f}%" if isinstance(acc, (int, float)) else f"  Validation accuracy: {acc}")

    return model


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str
) -> Dict:
    """
    Evaluate model on test set.

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_probs = []

    print("\nRunning evaluation...")
    with torch.no_grad():
        for batch_idx, (signals, labels) in enumerate(test_loader):
            signals = signals.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {(batch_idx + 1) * len(signals)} samples...")

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro', zero_division=0
    )

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    results = {
        'accuracy': accuracy * 100,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probs,
        'confusion_matrix': cm,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support_per_class
    }

    return results


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: Path):
    """Plot confusion matrix"""
    plt.figure(figsize=(12, 10))

    # Normalize
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

    # Plot
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Normalized Count'})

    plt.title('Confusion Matrix - LSTM Evaluation', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved confusion matrix to {save_path}")
    plt.close()


def print_per_class_metrics(results: Dict, class_names: list):
    """Print per-class metrics in a nice table"""
    print("\n" + "=" * 90)
    print("Per-Class Metrics")
    print("=" * 90)

    print(f"{'Class':<30} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>12}")
    print("-" * 90)

    for idx, class_name in enumerate(class_names):
        precision = results['precision_per_class'][idx]
        recall = results['recall_per_class'][idx]
        f1 = results['f1_per_class'][idx]
        support = results['support_per_class'][idx]

        print(f"{class_name:<30} {precision:>12.3f} {recall:>12.3f} {f1:>12.3f} {support:>12}")

    print("-" * 90)
    print(f"{'Average (macro)':<30} {results['precision']:>12.3f} "
          f"{results['recall']:>12.3f} {results['f1']:>12.3f} "
          f"{sum(results['support_per_class']):>12}")
    print("=" * 90)


def analyze_failures(results: Dict, class_names: list):
    """Analyze failure cases"""
    predictions = results['predictions']
    labels = results['labels']

    # Find misclassified samples
    misclassified = predictions != labels
    num_failures = misclassified.sum()
    total = len(labels)
    error_rate = num_failures / total

    print("\n" + "=" * 90)
    print("Failure Case Analysis")
    print("=" * 90)
    print(f"Total misclassifications: {num_failures} / {total}")
    print(f"Error rate: {error_rate*100:.2f}%")

    if num_failures > 0:
        # Find most confused pairs
        confusion_pairs = {}
        for true_label, pred_label in zip(labels[misclassified], predictions[misclassified]):
            pair = (int(true_label), int(pred_label))
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

        # Sort by frequency
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)

        print("\nMost confused pairs:")
        for (true_idx, pred_idx), count in sorted_pairs[:5]:
            print(f"  {class_names[true_idx]:>25} → {class_names[pred_idx]:<25}: {count} cases ({count/num_failures*100:.1f}%)")

    print("=" * 90)


def main():
    """Main evaluation function"""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Print header
    print("=" * 90)
    print("LSTM Evaluation Script - Bearing Fault Diagnosis")
    print("=" * 90)
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 90)

    # Get device
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
    print(f"\n✓ Using device: {device}")

    # Validate data directory
    data_path = Path(args.data_dir)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory not found: {args.data_dir}\n"
            f"Please specify a valid directory using --data-dir"
        )

    # Load test data
    print(f"\nLoading test data from {args.data_dir}...")
    try:
        _, _, test_loader = create_lstm_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            val_batch_size=args.batch_size,
            num_workers=args.num_workers,
            signal_length=args.signal_length,
            random_seed=args.seed
        )
        print(f"✓ Loaded {len(test_loader.dataset)} test samples")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        sys.exit(1)

    # Load model
    model = load_model(args.checkpoint, args.model, device)

    # Evaluate
    results = evaluate_model(model, test_loader, device)

    # Print results
    print("\n" + "=" * 90)
    print("Evaluation Results")
    print("=" * 90)
    print(f"Overall Accuracy:  {results['accuracy']:.2f}%")
    print(f"Average Precision: {results['precision']:.4f}")
    print(f"Average Recall:    {results['recall']:.4f}")
    print(f"Average F1-Score:  {results['f1']:.4f}")
    print("=" * 90)

    # Per-class metrics
    if args.per_class_metrics:
        print_per_class_metrics(results, FAULT_TYPES)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot confusion matrix
    if args.plot_confusion:
        cm_path = output_dir / 'confusion_matrix.png'
        plot_confusion_matrix(results['confusion_matrix'], FAULT_TYPES, cm_path)

    # Analyze failures
    if args.analyze_failures:
        analyze_failures(results, FAULT_TYPES)

    # Save predictions
    if args.save_predictions:
        pred_path = output_dir / 'predictions.npz'
        print(f"\nSaving predictions to {pred_path}...")
        np.savez(
            pred_path,
            predictions=results['predictions'],
            labels=results['labels'],
            probabilities=results['probabilities']
        )
        print(f"✓ Saved predictions to {pred_path}")

    print("\n" + "=" * 90)
    print("✓ Evaluation Complete!")
    print("=" * 90)


if __name__ == '__main__':
    main()
