#!/usr/bin/env python3
"""
Example Usage - LSTM Bearing Fault Diagnosis

Demonstrates how to use LSTM models for fault diagnosis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from models import create_model, list_available_models
from utils.constants import FAULT_TYPES, NUM_CLASSES

print("\n" + "="*70)
print("  LSTM Bearing Fault Diagnosis - Example Usage")
print("="*70 + "\n")

# Example 1: List available models
print("Example 1: Available LSTM Models")
print("-" * 50)
models = list_available_models()
for model_name in models:
    print(f"  - {model_name}")
print()

# Example 2: Create Vanilla LSTM
print("Example 2: Creating Vanilla LSTM")
print("-" * 50)
model = create_model('vanilla_lstm', num_classes=11, hidden_size=128)
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model created!")
print(f"  Parameters: {total_params:,}")
print(f"  Size: ~{total_params * 4 / (1024**2):.1f} MB\n")

# Example 3: Create BiLSTM
print("Example 3: Creating BiLSTM")
print("-" * 50)
model = create_model('bilstm', num_classes=11, hidden_size=256)
total_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model created!")
print(f"  Parameters: {total_params:,}")
print(f"  Size: ~{total_params * 4 / (1024**2):.1f} MB\n")

# Example 4: Dummy inference
print("Example 4: Model Inference (Dummy Data)")
print("-" * 50)
model.eval()
dummy_signal = torch.randn(4, 1, 102400)  # Batch of 4
with torch.no_grad():
    outputs = model(dummy_signal)
    predictions = torch.argmax(outputs, dim=1)

print(f"Input shape: {dummy_signal.shape}")
print(f"Output shape: {outputs.shape}")
print(f"Predictions: {predictions.tolist()}\n")

# Example 5: Fault types
print("Example 5: Fault Type Classification")
print("-" * 50)
for i, fault in enumerate(FAULT_TYPES):
    print(f"  Class {i:2d}: {fault}")
print()

print("="*70)
print("✓ All examples completed!")
print("="*70)
print("\nNext steps:")
print("  1. Prepare .MAT data files")
print("  2. Train: python scripts/train_lstm.py --model bilstm")
print("  3. Evaluate: python scripts/evaluate_lstm.py\n")
