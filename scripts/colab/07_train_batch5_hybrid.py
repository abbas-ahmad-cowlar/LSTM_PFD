#!/usr/bin/env python3
"""
Batch 5: Hybrid Models (4 models)

Models: CNN-LSTM, CNN-TCN, CNN-Transformer (Small), MultiScale CNN Hybrid

Est. time on T4: ~20 min

Usage:
    !python scripts/colab/07_train_batch5_hybrid.py
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.colab._train_utils import train_model_batch

MODELS = [
    "cnn_lstm",
    "cnn_tcn",
    "cnn_transformer_small",
    "multiscale_cnn",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_model_batch(
        batch_name="Hybrid Models",
        model_keys=MODELS,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
