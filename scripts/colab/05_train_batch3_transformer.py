#!/usr/bin/env python3
"""
Batch 3: Transformer Models (6 models)

Models: SignalTransformer, ViT-Tiny-1D, ViT-Small-1D, ViT-Base-1D,
        PatchTST, TSMixer

Est. time on T4: ~30 min

Usage:
    !python scripts/colab/05_train_batch3_transformer.py
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from scripts.colab._train_utils import train_model_batch

MODELS = [
    "transformer",
    "vit_tiny_1d",
    "vit_small_1d",
    "patchtst",
    "tsmixer",
    "vit_base_1d",  # largest — last so if it OOMs, others are done
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_model_batch(
        batch_name="Transformer Models",
        model_keys=MODELS,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
