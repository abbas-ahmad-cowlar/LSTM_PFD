#!/usr/bin/env python3
"""
Example Usage Script - CNN Bearing Fault Diagnosis

This script demonstrates how to use the CNN models for bearing fault diagnosis.
Run this to verify your installation and understand the API.

Usage:
    python example_usage.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from models import create_model, list_available_models
from utils.constants import FAULT_TYPES, NUM_CLASSES, SIGNAL_LENGTH
from utils.device_manager import get_device


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def example_1_list_available_models():
    """Example 1: List all available CNN models"""
    print_section("Example 1: Available CNN Models")

    models = list_available_models()
    print(f"Total models available: {len(models)}\n")

    print("Basic CNNs:")
    for model_name in ['cnn1d', 'attention_cnn', 'multiscale_cnn']:
        print(f"  - {model_name}")

    print("\nResNet Variants:")
    for model_name in ['resnet18', 'resnet34', 'resnet50', 'se_resnet18', 'se_resnet34']:
        print(f"  - {model_name}")

    print("\nEfficientNet Variants:")
    for model_name in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4']:
        print(f"  - {model_name}")


def example_2_create_model():
    """Example 2: Create a CNN model"""
    print_section("Example 2: Creating a CNN Model")

    # Create ResNet-34 model
    print("Creating ResNet-34 model...")
    model = create_model('resnet34', num_classes=NUM_CLASSES)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Model created successfully!")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / (1024**2):.1f} MB (FP32)")

    return model


def example_3_model_inference():
    """Example 3: Perform inference with dummy data"""
    print_section("Example 3: Model Inference (Dummy Data)")

    # Create model
    model = create_model('resnet34', num_classes=NUM_CLASSES)
    model.eval()

    # Get device
    device = get_device()
    model = model.to(device)
    print(f"Using device: {device}")

    # Create dummy input (batch_size=4, signal_length=102400)
    print(f"\nCreating dummy vibration signals...")
    print(f"  Shape: (batch_size=4, channels=1, length={SIGNAL_LENGTH})")
    dummy_signals = torch.randn(4, 1, SIGNAL_LENGTH).to(device)

    # Perform inference
    print("\nPerforming forward pass...")
    with torch.no_grad():
        outputs = model(dummy_signals)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

    print(f"✓ Inference completed!")
    print(f"  Output shape: {outputs.shape}  # (batch_size=4, num_classes={NUM_CLASSES})")
    print(f"  Predictions shape: {predictions.shape}  # (batch_size=4,)")

    # Print predictions for each sample
    print("\nPredictions:")
    for i in range(4):
        pred_class = predictions[i].item()
        confidence = probabilities[i, pred_class].item()
        fault_type = FAULT_TYPES[pred_class]
        print(f"  Sample {i+1}: Class {pred_class} ({fault_type}) - Confidence: {confidence:.2%}")


def example_4_fault_types():
    """Example 4: Display fault type information"""
    print_section("Example 4: Fault Type Classification")

    print(f"Total fault types: {NUM_CLASSES}\n")

    print("Fault Type Mapping:")
    for i, fault_type in enumerate(FAULT_TYPES):
        print(f"  Class {i:2d}: {fault_type}")


def example_5_compare_models():
    """Example 5: Compare model sizes"""
    print_section("Example 5: Model Size Comparison")

    model_names = ['cnn1d', 'resnet18', 'resnet34', 'efficientnet_b0', 'efficientnet_b2']

    print(f"{'Model':<20} {'Parameters':>15} {'Size (MB)':>12}")
    print("-" * 50)

    for model_name in model_names:
        model = create_model(model_name, num_classes=NUM_CLASSES)
        total_params = sum(p.numel() for p in model.parameters())
        size_mb = total_params * 4 / (1024**2)

        print(f"{model_name:<20} {total_params:>15,} {size_mb:>11.1f}")


def main():
    """Run all examples"""
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "  CNN Bearing Fault Diagnosis - Example Usage".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)

    try:
        # Run examples
        example_1_list_available_models()
        example_2_create_model()
        example_3_model_inference()
        example_4_fault_types()
        example_5_compare_models()

        # Final message
        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70)

        print("\nNext steps:")
        print("  1. Prepare your .MAT data files (see README.md)")
        print("  2. Train a model: python scripts/train_cnn.py --model resnet34 --data-dir data/raw/bearing_data")
        print("  3. Evaluate: python scripts/evaluate_cnn.py --model-checkpoint results/checkpoints/best_model.pth")
        print("\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
