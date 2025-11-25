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
    python scripts/evaluate_lstm.py --checkpoint checkpoints/bilstm/best_model.pth

    # Evaluate with detailed analysis
    python scripts/evaluate_lstm.py --checkpoint model.pth --analyze-failures --plot-confusion

Author: Milestone 2 - LSTM Implementation
Date: 2025-11-23
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Import project modules
from models import create_model, list_available_models
from data.lstm_dataloader import create_lstm_dataloaders
from utils.device_manager import get_device
from utils.constants import NUM_CLASSES, FAULT_TYPES, FAULT_TYPE_DISPLAY_NAMES


# Class names for bearing faults
CLASS_NAMES = [FAULT_TYPE_DISPLAY_NAMES.get(ft, ft) for ft in FAULT_TYPES]


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
                       help='Directory with .MAT files')
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


def load_model(checkpoint_path: str, model_type: str = 'bilstm', device: torch.device = None):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model hyperparameters from checkpoint if available
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Try to infer hidden size from state dict
    hidden_size = 128  # default
    for key in state_dict.keys():
        if 'lstm.weight_ih_l0' in key:
            # weight_ih has shape (4*hidden_size, input_size)
            hidden_size = state_dict[key].shape[0] // 4
            break

    print(f"Detected hidden_size: {hidden_size}")

    # Create model
    model = create_model(
        model_name=model_type,
        num_classes=NUM_CLASSES,
        hidden_size=hidden_size,
        num_layers=2,  # default
        dropout=0.3
    )

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, checkpoint


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()

    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)

    # Classification report
    report = classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    results = {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probs
    }

    return results


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix"""
    plt.figure(figsize=(12, 10))

    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    # Plot
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Normalized Count'})

    plt.title('LSTM - Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to {save_path}")
    plt.close()


def print_per_class_metrics(report, class_names):
    """Print per-class metrics in a nice table"""
    print("\n" + "=" * 80)
    print("Per-Class Metrics")
    print("=" * 80)

    print(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 80)

    for class_name in class_names:
        if class_name in report:
            metrics = report[class_name]
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1-score']
            support = metrics['support']

            print(f"{class_name:<25} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {support:>10.0f}")

    print("-" * 80)
    if 'macro avg' in report:
        macro = report['macro avg']
        print(f"{'Average (macro)':<25} {macro['precision']:>10.3f} "
              f"{macro['recall']:>10.3f} {macro['f1-score']:>10.3f} "
              f"{report['weighted avg']['support']:>10.0f}")
    print("=" * 80)


def analyze_failures(predictions, labels, class_names):
    """Analyze misclassification patterns"""
    misclassified = predictions != labels
    num_failures = misclassified.sum()
    total = len(labels)

    print("\n" + "=" * 80)
    print("Failure Case Analysis")
    print("=" * 80)
    print(f"Total misclassifications: {num_failures} / {total}")
    print(f"Error rate: {100 * num_failures / total:.2f}%")

    if num_failures > 0:
        # Find most confused pairs
        from collections import Counter
        confused_pairs = []

        for true_label, pred_label in zip(labels[misclassified], predictions[misclassified]):
            confused_pairs.append((true_label, pred_label))

        most_common = Counter(confused_pairs).most_common(5)

        print("\nMost confused pairs:")
        for (true_idx, pred_idx), count in most_common:
            print(f"  {class_names[true_idx]:>20} → {class_names[pred_idx]:<20}: {count} cases")

    print("=" * 80)


def main():
    """Main evaluation function"""
    args = parse_args()

    # Print header
    print("=" * 80)
    print("LSTM Evaluation Script - Bearing Fault Diagnosis")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 80)

    # Get device
    if args.device == 'auto':
        device = get_device(prefer_gpu=True)
    else:
        device = torch.device(args.device)
    print(f"✓ Using device: {device}")

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model, checkpoint = load_model(args.checkpoint, args.model, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model loaded")
    print(f"  Parameters: {total_params:,}")

    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch']+1}")
    if 'metric_value' in checkpoint:
        print(f"  Validation accuracy: {checkpoint['metric_value']:.4f}")

    # Load test data
    print(f"\nLoading test data from {args.data_dir}...")
    _, _, test_loader = create_lstm_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        signal_length=args.signal_length,
        random_seed=args.seed
    )
    print(f"✓ Loaded {len(test_loader.dataset)} test samples")

    # Run evaluation
    print("\nRunning evaluation...")
    results = evaluate_model(model, test_loader, device)

    # Print results
    print("\n" + "=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    print(f"Overall Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")

    report = results['classification_report']
    if 'macro avg' in report:
        print(f"Macro Precision:   {report['macro avg']['precision']:.4f}")
        print(f"Macro Recall:      {report['macro avg']['recall']:.4f}")
        print(f"Macro F1-Score:    {report['macro avg']['f1-score']:.4f}")
    print("=" * 80)

    # Per-class metrics
    if args.per_class_metrics:
        print_per_class_metrics(report, CLASS_NAMES)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot confusion matrix
    if args.plot_confusion:
        cm_path = output_dir / 'confusion_matrix.png'
        plot_confusion_matrix(results['confusion_matrix'], CLASS_NAMES, cm_path)

    # Analyze failure cases
    if args.analyze_failures:
        analyze_failures(
            results['predictions'],
            results['labels'],
            CLASS_NAMES
        )

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

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
