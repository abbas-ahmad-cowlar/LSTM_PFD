#!/usr/bin/env python3
"""
Batch 1: CNN Models (5 models)

Models: CNN1D, AttentionCNN1D, LightweightAttentionCNN,
        MultiScaleCNN1D, DilatedMultiScaleCNN

Est. time on T4: ~15 min

Usage:
    !python scripts/colab/03_train_batch1_cnn.py
    !python scripts/colab/03_train_batch1_cnn.py --epochs 50  # shorter
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.colab._train_utils import train_model_batch

MODELS = [
    "cnn1d",
    "attention_cnn",
    "lightweight_attention_cnn",
    "multi_scale_cnn",
    "dilated_multi_scale_cnn",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_model_batch(
        batch_name="CNN Models",
        model_keys=MODELS,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
