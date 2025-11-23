#!/usr/bin/env python3
"""
ONNX Export Script

Export PyTorch models to ONNX format for cross-platform deployment.

Usage:
    # Basic export
    python scripts/export_onnx.py \\
        --model checkpoints/phase6/best_model.pth \\
        --output models/model.onnx

    # With validation and optimization
    python scripts/export_onnx.py \\
        --model checkpoints/phase6/best_model.pth \\
        --output models/model.onnx \\
        --validate \\
        --optimize \\
        --optimization-level all

    # With quantization
    python scripts/export_onnx.py \\
        --model checkpoints/phase6/best_model.pth \\
        --output models/model_int8.onnx \\
        --quantize

Author: Syed Abbas Ahmad
Date: 2025-11-20
"""

import argparse
import sys
import logging
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE
from deployment.onnx_export import (
    export_to_onnx,
    validate_onnx_export,
    optimize_onnx_model,
    convert_and_quantize_onnx,
    benchmark_onnx_inference,
    ONNXExportConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to PyTorch model checkpoint'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save ONNX model'
    )

    parser.add_argument(
        '--input-shape',
        type=int,
        nargs='+',
        default=[1, 1, SIGNAL_LENGTH],
        help='Input tensor shape (default: 1 1 102400)'
    )

    parser.add_argument(
        '--opset-version',
        type=int,
        default=14,
        help='ONNX opset version (default: 14)'
    )

    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate ONNX export by comparing outputs'
    )

    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Optimize ONNX model'
    )

    parser.add_argument(
        '--optimization-level',
        type=str,
        choices=['basic', 'extended', 'all'],
        default='basic',
        help='Optimization level'
    )

    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Apply quantization to ONNX model'
    )

    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark ONNX inference speed'
    )

    parser.add_argument(
        '--benchmark-runs',
        type=int,
        default=100,
        help='Number of benchmark runs (default: 100)'
    )

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load PyTorch model
    logger.info(f"Loading PyTorch model from {args.model}")
    checkpoint = torch.load(args.model, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        logger.error("Checkpoint format requires model class. Please provide model architecture.")
        return

    model = checkpoint
    model.eval()

    logger.info(f"Model loaded: {model.__class__.__name__}")

    # Create dummy input
    input_shape = tuple(args.input_shape)
    dummy_input = torch.randn(*input_shape)

    logger.info(f"Input shape: {input_shape}")

    # Export configuration
    config = ONNXExportConfig(
        opset_version=args.opset_version,
        do_constant_folding=True,
        verbose=False
    )

    # Export to ONNX
    logger.info(f"\n{'='*60}")
    logger.info("Exporting to ONNX")
    logger.info(f"{'='*60}\n")

    if args.quantize:
        # Export and quantize
        onnx_path = convert_and_quantize_onnx(
            model,
            dummy_input,
            str(output_path),
            quantization_type='dynamic'
        )
    else:
        # Regular export
        onnx_path = export_to_onnx(
            model,
            dummy_input,
            str(output_path),
            config
        )

    # Validate export
    if args.validate:
        logger.info(f"\n{'='*60}")
        logger.info("Validating ONNX Export")
        logger.info(f"{'='*60}\n")

        is_valid = validate_onnx_export(
            str(onnx_path),
            model,
            dummy_input
        )

        if not is_valid:
            logger.error("✗ ONNX validation failed!")
            return

    # Optimize ONNX model
    if args.optimize:
        logger.info(f"\n{'='*60}")
        logger.info("Optimizing ONNX Model")
        logger.info(f"{'='*60}\n")

        optimized_path = str(output_path).replace('.onnx', '_optimized.onnx')
        optimize_onnx_model(
            str(onnx_path),
            optimized_path,
            optimization_level=args.optimization_level
        )

        # Update onnx_path to optimized version
        onnx_path = Path(optimized_path)

    # Benchmark
    if args.benchmark:
        logger.info(f"\n{'='*60}")
        logger.info("Benchmarking ONNX Inference")
        logger.info(f"{'='*60}\n")

        try:
            stats = benchmark_onnx_inference(
                str(onnx_path),
                input_shape,
                num_runs=args.benchmark_runs
            )
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("✓ ONNX Export Complete")
    logger.info(f"{'='*60}")
    logger.info(f"ONNX model saved to: {onnx_path}")
    logger.info(f"Model size: {onnx_path.stat().st_size / (1024*1024):.2f} MB")

    if args.validate:
        logger.info("✓ Validation: Passed")

    if args.optimize:
        logger.info(f"✓ Optimization: {args.optimization_level}")

    if args.quantize:
        logger.info("✓ Quantization: Applied")

    logger.info(f"{'='*60}\n")

    logger.info("\nNext steps:")
    logger.info(f"  1. Test inference: python -c \"from deployment.onnx_export import ONNXInferenceSession; session = ONNXInferenceSession('{onnx_path}'); print('Model loaded successfully')\"")
    logger.info(f"  2. Deploy with API: Set MODEL_PATH={onnx_path} and MODEL_TYPE=onnx")
    logger.info(f"  3. Use in production: See Phase_9_DEPLOYMENT_GUIDE.md\n")


if __name__ == '__main__':
    main()
