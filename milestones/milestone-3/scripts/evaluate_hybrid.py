#!/usr/bin/env python3
"""
Hybrid CNN-LSTM Evaluation Script

Comprehensive evaluation of trained hybrid CNN-LSTM models.

Usage:
    python scripts/evaluate_hybrid.py --checkpoint path/to/checkpoint.pth --model recommended_1

Author: Bearing Fault Diagnosis Team
Milestone: 3 - CNN-LSTM Hybrid
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models import create_model
from data.matlab_importer import MatlabImporter
from data.cnn_dataset import create_cnn_datasets_from_arrays
from data.cnn_dataloader import create_cnn_dataloaders
from utils.device_manager import get_device
from utils.constants import FAULT_TYPES


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Hybrid CNN-LSTM')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--model', type=str, default='recommended_1',
                       choices=['recommended_1', 'recommended_2', 'recommended_3', 'custom'],
                       help='Hybrid model configuration')
    parser.add_argument('--cnn-type', type=str, default='resnet34',
                       help='CNN backbone (for custom hybrid)')
    parser.add_argument('--lstm-type', type=str, default='bilstm',
                       help='LSTM type (for custom hybrid)')
    parser.add_argument('--lstm-hidden-size', type=int, default=256)
    parser.add_argument('--lstm-num-layers', type=int, default=2)
    parser.add_argument('--pooling', type=str, default='mean',
                       choices=['mean', 'max', 'last', 'attention'])

    parser.add_argument('--data-dir', type=str, default='data/raw/bearing_data',
                       help='Directory containing .mat files')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output-dir', type=str, default='results/evaluation',
                       help='Directory to save evaluation results')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_checkpoint_and_model(checkpoint_path: str, args, device):
    """Load model from checkpoint."""
    print(f"\nLoading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model with same configuration
    if args.model == 'custom':
        model = create_model(
            'custom',
            cnn_type=args.cnn_type,
            lstm_type=args.lstm_type,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_num_layers=args.lstm_num_layers,
            pooling_method=args.pooling
        )
    else:
        model = create_model(
            args.model,
            lstm_hidden_size=args.lstm_hidden_size,
            pooling_method=args.pooling
        )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded from checkpoint")
    if 'epoch' in checkpoint:
        print(f"  Checkpoint epoch: {checkpoint['epoch'] + 1}")
    if 'best_val_acc' in checkpoint:
        print(f"  Best validation accuracy: {checkpoint['best_val_acc']:.2f}%")

    return model, checkpoint


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()

    all_predictions = []
    all_labels = []
    all_probs = []

    print("\nRunning evaluation...")

    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            outputs = model(signals)
            probs = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate accuracy
    accuracy = 100.0 * np.mean(all_predictions == all_labels)

    print(f"\n{'='*70}")
    print(f"  Evaluation Results")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Total samples: {len(all_labels)}")

    return all_predictions, all_labels, all_probs, accuracy


def plot_confusion_matrix(cm, class_names, output_path):
    """Plot confusion matrix."""
    plt.figure(figsize=(12, 10))

    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    # Plot
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
               xticklabels=class_names, yticklabels=class_names,
               cbar_kws={'label': 'Normalized Count'})

    plt.title('Confusion Matrix - Hybrid CNN-LSTM', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to {output_path}")
    plt.close()


def save_classification_report(predictions, labels, class_names, output_path):
    """Save detailed classification report."""
    report = classification_report(labels, predictions, target_names=class_names, digits=4)

    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Hybrid CNN-LSTM Classification Report\n")
        f.write("="*70 + "\n\n")
        f.write(report)
        f.write("\n")

    print(f"✓ Saved classification report to {output_path}")

    # Also print to console
    print(f"\n{'='*70}")
    print("Per-Class Performance")
    print(f"{'='*70}")
    print(report)


def main():
    args = parse_args()
    device = get_device()

    print("="*70)
    print("  Hybrid CNN-LSTM Evaluation - Bearing Fault Diagnosis")
    print("="*70)
    print(f"\nModel: {args.model}")
    print(f"Device: {device}")

    # Load checkpoint and create model
    model, checkpoint = load_checkpoint_and_model(args.checkpoint, args, device)

    # Load test data
    print(f"\nLoading test data from: {args.data_dir}")
    data_path = Path(args.data_dir)

    importer = MatlabImporter()
    batch_data = importer.load_batch(data_path, pattern='*.mat')

    if not batch_data:
        raise ValueError(f"No .mat files found in {args.data_dir}")

    print(f"✓ Loaded {len(batch_data)} signals")

    signals, labels = importer.extract_signals_and_labels(batch_data)

    # Create test dataset (use only test split)
    _, _, test_dataset = create_cnn_datasets_from_arrays(
        signals=signals,
        labels=labels,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        augment_train=False,
        random_seed=args.seed
    )

    # Create test dataloader
    loaders = create_cnn_dataloaders(
        train_dataset=test_dataset,  # Reuse for consistency
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )

    test_loader = loaders['test']
    print(f"✓ Created test dataloader: {len(test_dataset)} samples")

    # Evaluate
    predictions, true_labels, probs, accuracy = evaluate_model(model, test_loader, device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get class names
    class_names = test_dataset.idx_to_label
    class_names = [class_names[i] for i in sorted(class_names.keys())]

    # Save classification report
    report_path = output_dir / 'classification_report.txt'
    save_classification_report(predictions, true_labels, class_names, report_path)

    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(cm, class_names, cm_path)

    # Save predictions
    pred_path = output_dir / 'predictions.npz'
    np.savez(
        pred_path,
        predictions=predictions,
        labels=true_labels,
        probabilities=probs
    )
    print(f"✓ Saved predictions to {pred_path}")

    print(f"\n{'='*70}")
    print("  Evaluation Complete!")
    print(f"{'='*70}")
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    print(f"Results saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
