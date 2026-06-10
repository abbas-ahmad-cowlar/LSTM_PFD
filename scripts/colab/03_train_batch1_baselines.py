#!/usr/bin/env python3
"""
Batch 1: Deep baselines (Tier 1 data-driven models).

Models: CNN1D, AttentionCNN1D, CNN-LSTM, ResNet18-1D, PatchTST

Usage:
    !python scripts/colab/03_train_batch1_baselines.py
    !python scripts/colab/03_train_batch1_baselines.py --epochs 50
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.colab._train_utils import train_model_batch

MODELS = [
    "cnn1d",
    "attention_cnn",
    "cnn_lstm",
    "resnet18",
    "patchtst",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_model_batch(
        batch_name="Deep Baselines (Tier 1)",
        model_keys=MODELS,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
