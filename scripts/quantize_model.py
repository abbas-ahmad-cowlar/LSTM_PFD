#!/usr/bin/env python3
"""
Model Quantization Script

Quantize trained models to INT8 or FP16 for deployment.

Usage:
    # Dynamic quantization (INT8)
    python scripts/quantize_model.py \\
        --model checkpoints/phase6/best_model.pth \\
        --output checkpoints/phase9/model_int8.pth \\
        --quantization-type dynamic

    # FP16 conversion
    python scripts/quantize_model.py \\
        --model checkpoints/phase6/best_model.pth \\
        --output checkpoints/phase9/model_fp16.pth \\
        --quantization-type fp16

    # Static quantization (requires calibration data)
    python scripts/quantize_model.py \\
        --model checkpoints/phase6/best_model.pth \\
        --output checkpoints/phase9/model_int8_static.pth \\
        --quantization-type static \\
        --calibration-data data/processed/signals_cache.h5

Author: Syed Abbas Ahmad
Date: 2025-11-20
"""

import argparse
import sys
import logging
from pathlib import Path
import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE
from deployment.quantization import (
    quantize_model_dynamic,
    quantize_model_static,
    quantize_to_fp16,
    compare_model_sizes,
    benchmark_quantized_model,
    save_quantized_model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_calibration_data(data_path: str, num_samples: int = 100):
    """Load calibration data for static quantization."""
    logger.info(f"Loading calibration data from {data_path}")

    with h5py.File(data_path, 'r') as f:
        # Load training data (first num_samples)
        signals = f['signals'][:num_samples]
        labels = f['labels'][:num_samples]

    # Convert to PyTorch tensors
    signals = torch.from_numpy(signals).float()
    if signals.ndim == 2:
        signals = signals.unsqueeze(1)  # Add channel dimension

    labels = torch.from_numpy(labels).long()

    # Create DataLoader
    dataset = TensorDataset(signals, labels)
    calibration_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    logger.info(f"Loaded {len(signals)} calibration samples")
    return calibration_loader


def main():
    parser = argparse.ArgumentParser(description="Quantize PyTorch model")

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save quantized model'
    )

    parser.add_argument(
        '--quantization-type',
        type=str,
        choices=['dynamic', 'static', 'fp16'],
        default='dynamic',
        help='Type of quantization'
    )

    parser.add_argument(
        '--calibration-data',
        type=str,
        default=None,
        help='Path to calibration data (for static quantization)'
    )

    parser.add_argument(
        '--calibration-samples',
        type=int,
        default=100,
        help='Number of calibration samples'
    )

    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark original vs quantized model'
    )

    parser.add_argument(
        '--backend',
        type=str,
        choices=['fbgemm', 'qnnpack'],
        default='fbgemm',
        help='Quantization backend (fbgemm for x86, qnnpack for ARM)'
    )

    args = parser.parse_args()

    # Load original model
    logger.info(f"Loading model from {args.model}")
    checkpoint = torch.load(args.model, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Need to instantiate model class
        logger.error(
            "ERROR: Checkpoint contains state_dict, not the full model.\n"
            "This script requires a checkpoint saved with the full model object.\n"
            "\n"
            "To fix this, save your checkpoint with:\n"
            "  torch.save(model, 'model.pth')  # Save full model\n"
            "\n"
            "Instead of:\n"
            "  torch.save({'model_state_dict': model.state_dict()}, 'model.pth')  # Don't use this format\n"
        )
        return

    model = checkpoint
    model.eval()

    logger.info(f"Model loaded: {model.__class__.__name__}")

    # Apply quantization
    logger.info(f"\n{'='*60}")
    logger.info(f"Applying {args.quantization_type} quantization")
    logger.info(f"{'='*60}\n")

    if args.quantization_type == 'dynamic':
        quantized_model = quantize_model_dynamic(model, inplace=False)

    elif args.quantization_type == 'static':
        if not args.calibration_data:
            logger.error("Static quantization requires --calibration-data")
            return

        # Load calibration data
        calibration_loader = load_calibration_data(
            args.calibration_data,
            args.calibration_samples
        )

        quantized_model = quantize_model_static(
            model,
            calibration_loader,
            backend=args.backend,
            inplace=False
        )

    elif args.quantization_type == 'fp16':
        quantized_model = quantize_to_fp16(model, inplace=False)

    else:
        logger.error(f"Unknown quantization type: {args.quantization_type}")
        return

    # Compare sizes
    logger.info(f"\n{'='*60}")
    logger.info("Model Size Comparison")
    logger.info(f"{'='*60}\n")

    size_stats = compare_model_sizes(model, quantized_model)

    # Benchmark if requested
    if args.benchmark:
        logger.info(f"\n{'='*60}")
        logger.info("Performance Benchmark")
        logger.info(f"{'='*60}\n")

        # Create dummy input
        dummy_input = torch.randn(1, 1, SIGNAL_LENGTH)

        try:
            perf_stats = benchmark_quantized_model(
                model,
                quantized_model,
                dummy_input,
                num_runs=100
            )
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")

    # Save quantized model
    logger.info(f"\n{'='*60}")
    logger.info("Saving Quantized Model")
    logger.info(f"{'='*60}\n")

    metadata = {
        'quantization_type': args.quantization_type,
        'backend': args.backend if args.quantization_type == 'static' else None,
        'original_model': args.model,
        'compression_ratio': size_stats['compression_ratio'],
        'size_reduction_percent': size_stats['size_reduction_percent']
    }

    save_quantized_model(quantized_model, args.output, metadata)

    logger.info(f"\n{'='*60}")
    logger.info("✓ Quantization Complete")
    logger.info(f"{'='*60}")
    logger.info(f"Quantized model saved to: {args.output}")
    logger.info(f"Compression ratio: {size_stats['compression_ratio']:.2f}x")
    logger.info(f"Size reduction: {size_stats['size_reduction_percent']:.1f}%")
    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    main()
