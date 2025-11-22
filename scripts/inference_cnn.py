#!/usr/bin/env python3
"""
CNN Inference Script - Real-time fault diagnosis using trained CNN models

This script provides inference capabilities for trained CNN models:
- Single signal prediction
- Batch prediction from file
- Real-time streaming prediction (simulated)
- Confidence scores and top-k predictions

Usage:
    # Predict fault for a single signal
    python scripts/inference_cnn.py --checkpoint model.pth --signal-file signal.npy

    # Batch prediction from directory
    python scripts/inference_cnn.py --checkpoint model.pth --batch-dir signals/

    # Interactive demo mode
    python scripts/inference_cnn.py --checkpoint model.pth --demo

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
import time
from typing import Dict, List, Tuple

# Import project modules
from models.cnn.cnn_1d import CNN1D
from models.cnn.attention_cnn import AttentionCNN1D, LightweightAttentionCNN
from models.cnn.multi_scale_cnn import MultiScaleCNN1D, DilatedMultiScaleCNN
from data.cnn_transforms import normalize_signal
from utils.device_manager import get_device
from utils.logging import get_logger


# Model registry
MODEL_REGISTRY = {
    'cnn1d': CNN1D,
    'attention': AttentionCNN1D,
    'attention-lite': LightweightAttentionCNN,
    'multiscale': MultiScaleCNN1D,
    'dilated': DilatedMultiScaleCNN
}

# Class names
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
        description='CNN Inference for Bearing Fault Diagnosis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model', type=str, default=None,
                       choices=list(MODEL_REGISTRY.keys()),
                       help='Model architecture (auto-detected if None)')

    # Input arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--signal-file', type=str,
                      help='Path to signal file (.npy)')
    group.add_argument('--batch-dir', type=str,
                      help='Directory with multiple signal files')
    group.add_argument('--demo', action='store_true',
                      help='Run interactive demo mode')

    # Inference options
    parser.add_argument('--top-k', type=int, default=3,
                       help='Show top-k predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold for prediction')
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize input signals')

    # Output arguments
    parser.add_argument('--output-file', type=str, default=None,
                       help='Save predictions to file (CSV format)')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')

    # Device
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')

    return parser.parse_args()


def load_model(checkpoint_path: str, model_type: str = None, device: torch.device = None):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Auto-detect model type
    if model_type is None:
        if 'args' in checkpoint and 'model' in checkpoint['args']:
            model_type = checkpoint['args']['model']
        else:
            model_type = 'cnn1d'  # Default

    # Get model class
    model_class = MODEL_REGISTRY[model_type]

    # Determine number of classes
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        for key in state_dict.keys():
            if 'classifier' in key and 'weight' in key and key.endswith('.weight'):
                num_classes = state_dict[key].shape[0]
                break
    else:
        num_classes=NUM_CLASSES

    # Create model
    model = model_class(num_classes=num_classes, input_length=102400, in_channels=1)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, model_type


def load_signal(file_path: str, normalize: bool = False) -> np.ndarray:
    """Load signal from file"""
    signal = np.load(file_path)

    # Ensure 1D
    if signal.ndim > 1:
        signal = signal.flatten()

    # Normalize if requested
    if normalize:
        signal = normalize_signal(signal)

    return signal


def predict(
    model: torch.nn.Module,
    signal: np.ndarray,
    device: torch.device,
    top_k: int = 3
) -> Dict:
    """
    Predict fault class for a signal

    Args:
        model: Trained CNN model
        signal: Input signal (1D numpy array)
        device: Device to run on
        top_k: Number of top predictions to return

    Returns:
        Dictionary with prediction results
    """
    # Prepare input
    signal_tensor = torch.from_numpy(signal).float().unsqueeze(0).unsqueeze(0)  # [1, 1, length]
    signal_tensor = signal_tensor.to(device)

    # Inference
    start_time = time.time()
    with torch.no_grad():
        output = model(signal_tensor)
        probs = torch.softmax(output, dim=1)[0]
    inference_time = time.time() - start_time

    # Get top-k predictions
    top_probs, top_indices = torch.topk(probs, k=min(top_k, len(probs)))

    results = {
        'predicted_class': top_indices[0].item(),
        'predicted_label': CLASS_NAMES[top_indices[0].item()],
        'confidence': top_probs[0].item(),
        'top_k_classes': [idx.item() for idx in top_indices],
        'top_k_labels': [CLASS_NAMES[idx.item()] for idx in top_indices],
        'top_k_probs': [prob.item() for prob in top_probs],
        'all_probs': probs.cpu().numpy(),
        'inference_time': inference_time
    }

    return results


def print_prediction(results: Dict, verbose: bool = False):
    """Print prediction results in a nice format"""
    print("\n" + "=" * 60)
    print("Prediction Results")
    print("=" * 60)
    print(f"Predicted Fault: {results['predicted_label']}")
    print(f"Confidence:      {results['confidence']*100:.2f}%")
    print(f"Inference Time:  {results['inference_time']*1000:.2f} ms")

    if verbose and len(results['top_k_labels']) > 1:
        print(f"\nTop-{len(results['top_k_labels'])} Predictions:")
        for idx, (label, prob) in enumerate(zip(results['top_k_labels'], results['top_k_probs']), 1):
            print(f"  {idx}. {label:<25} {prob*100:>6.2f}%")

    print("=" * 60)


def predict_single_signal(args, model, device, logger):
    """Predict fault for a single signal"""
    logger.info(f"Loading signal from {args.signal_file}...")
    signal = load_signal(args.signal_file, normalize=args.normalize)
    logger.info(f"✓ Loaded signal: shape={signal.shape}")

    # Pad or trim to correct length
    target_length = 102400
    if len(signal) < target_length:
        signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')
    elif len(signal) > target_length:
        signal = signal[:target_length]

    logger.info("Running inference...")
    results = predict(model, signal, device, top_k=args.top_k)

    print_prediction(results, verbose=args.verbose)

    # Check threshold
    if results['confidence'] < args.threshold:
        print(f"\n⚠ Warning: Confidence ({results['confidence']:.2f}) below threshold ({args.threshold})")

    return results


def predict_batch(args, model, device, logger):
    """Predict faults for multiple signals"""
    batch_dir = Path(args.batch_dir)
    signal_files = sorted(batch_dir.glob('*.npy'))

    if not signal_files:
        logger.error(f"No .npy files found in {batch_dir}")
        return

    logger.info(f"Found {len(signal_files)} signal files")

    results_list = []

    for idx, file_path in enumerate(signal_files, 1):
        logger.info(f"\n[{idx}/{len(signal_files)}] Processing {file_path.name}...")

        signal = load_signal(str(file_path), normalize=args.normalize)

        # Pad or trim
        target_length = 102400
        if len(signal) < target_length:
            signal = np.pad(signal, (0, target_length - len(signal)), mode='constant')
        elif len(signal) > target_length:
            signal = signal[:target_length]

        results = predict(model, signal, device, top_k=args.top_k)
        results['file_name'] = file_path.name
        results_list.append(results)

        print(f"  {file_path.name}: {results['predicted_label']} ({results['confidence']*100:.1f}%)")

    # Save results if requested
    if args.output_file:
        import csv
        with open(args.output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file_name', 'predicted_class', 'predicted_label', 'confidence', 'inference_time'])
            for res in results_list:
                writer.writerow([
                    res['file_name'],
                    res['predicted_class'],
                    res['predicted_label'],
                    f"{res['confidence']:.4f}",
                    f"{res['inference_time']:.6f}"
                ])
        logger.info(f"\n✓ Saved predictions to {args.output_file}")

    return results_list


def run_demo(model, device, logger):
    """Run interactive demo mode"""
    print("\n" + "=" * 80)
    print("CNN Inference Demo - Bearing Fault Diagnosis")
    print("=" * 80)
    print("This demo generates synthetic signals and classifies them in real-time.")
    print("Press Ctrl+C to exit.")
    print("=" * 80)

    from data.signal_generator import SignalGenerator
    from config.data_config import DataConfig
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

    # Create generator
    config = DataConfig(num_signals_per_fault=1, rng_seed=None)
    generator = SignalGenerator(config)

    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"\n{'='*60}")
            print(f"Demo Iteration {iteration}")
            print(f"{'='*60}")

            # Generate random signal
            fault_type = np.random.randint(0, 11)
            print(f"Generating signal: {CLASS_NAMES[fault_type]}...")

            signal = generator.generate_signal(
                fault_type=fault_type,
                severity=np.random.uniform(0.3, 0.9),
                temporal_evolution=False
            )

            # Predict
            results = predict(model, signal, device, top_k=3)

            # Print results
            print(f"\nTrue Label:      {CLASS_NAMES[fault_type]}")
            print(f"Predicted Label: {results['predicted_label']}")
            print(f"Confidence:      {results['confidence']*100:.2f}%")
            print(f"Inference Time:  {results['inference_time']*1000:.2f} ms")

            correct = (results['predicted_class'] == fault_type)
            print(f"Result:          {'✓ CORRECT' if correct else '✗ INCORRECT'}")

            if not correct:
                print(f"\nTop-3 Predictions:")
                for idx, (label, prob) in enumerate(zip(results['top_k_labels'][:3],
                                                        results['top_k_probs'][:3]), 1):
                    marker = "←" if idx == 1 else ""
                    print(f"  {idx}. {label:<25} {prob*100:>6.2f}% {marker}")

            # Wait for user input
            input("\nPress Enter for next signal (Ctrl+C to exit)...")

    except KeyboardInterrupt:
        print("\n\n✓ Demo ended.")


def main():
    """Main inference function"""
    args = parse_args()

    # Setup logging
    logger = get_logger(__name__)

    # Print header
    print("=" * 80)
    print("CNN Inference Script - Bearing Fault Diagnosis")
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
    model, model_type = load_model(args.checkpoint, args.model, device)
    logger.info(f"✓ Model loaded: {model_type}")
    logger.info(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run inference based on mode
    if args.demo:
        run_demo(model, device, logger)
    elif args.signal_file:
        predict_single_signal(args, model, device, logger)
    elif args.batch_dir:
        predict_batch(args, model, device, logger)

    logger.info("\n✓ Inference complete!")


if __name__ == '__main__':
    main()
