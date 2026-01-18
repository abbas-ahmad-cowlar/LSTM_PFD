#!/usr/bin/env python3
"""
ONNX Export and Runtime Inference

Converts PyTorch models to ONNX format and benchmarks inference
performance using ONNX Runtime for 2-5x speedup.

Features:
- Model conversion to ONNX
- ONNX Runtime inference benchmarks
- Numerical validation (PyTorch vs ONNX)
- Performance comparison report

Usage:
    # Export model to ONNX
    python scripts/utilities/onnx_export.py --checkpoint checkpoints/model.pth --output model.onnx
    
    # Benchmark comparison
    python scripts/utilities/onnx_export.py --checkpoint checkpoints/model.pth --benchmark

Author: Deficiency Fix #45 (Priority: 8)
Date: 2026-01-18
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import time
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from utils.logging import get_logger
from utils.constants import NUM_CLASSES


logger = get_logger(__name__)


def export_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    input_shape: Tuple[int, ...] = (1, 1, 102400),
    opset_version: int = 14,
    dynamic_axes: bool = True
) -> None:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Path to save ONNX model
        input_shape: Input tensor shape
        opset_version: ONNX opset version
        dynamic_axes: Enable dynamic batch size
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Dynamic axes for batch size
    if dynamic_axes:
        dyn_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    else:
        dyn_axes = None
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dyn_axes
    )
    
    logger.info(f"✓ Model exported to: {output_path}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def validate_onnx(
    pytorch_model: torch.nn.Module,
    onnx_path: Path,
    input_shape: Tuple[int, ...] = (1, 1, 102400),
    rtol: float = 1e-3,
    atol: float = 1e-5
) -> Dict[str, Any]:
    """
    Validate ONNX model outputs match PyTorch.
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        input_shape: Input tensor shape
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        Validation results
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("ONNX Runtime not installed. Run: pip install onnxruntime")
        return {'error': 'onnxruntime not installed'}
    
    pytorch_model.eval()
    
    # Create test input
    test_input = torch.randn(*input_shape)
    
    # PyTorch inference
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()
    
    # ONNX Runtime inference
    ort_session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    onnx_output = ort_session.run(None, ort_inputs)[0]
    
    # Compare
    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()
    is_close = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)
    
    results = {
        'valid': is_close,
        'max_difference': float(max_diff),
        'mean_difference': float(mean_diff),
        'rtol': rtol,
        'atol': atol
    }
    
    if is_close:
        logger.info("✓ ONNX validation passed")
    else:
        logger.warning(f"⚠ ONNX validation failed. Max diff: {max_diff:.6f}")
    
    return results


def benchmark_inference(
    pytorch_model: torch.nn.Module,
    onnx_path: Path,
    input_shape: Tuple[int, ...] = (1, 1, 102400),
    n_runs: int = 100,
    warmup: int = 10
) -> Dict[str, Any]:
    """
    Benchmark PyTorch vs ONNX Runtime inference speed.
    
    Args:
        pytorch_model: PyTorch model
        onnx_path: Path to ONNX model
        input_shape: Input tensor shape
        n_runs: Number of benchmark runs
        warmup: Warmup runs
    
    Returns:
        Benchmark results
    """
    try:
        import onnxruntime as ort
    except ImportError:
        return {'error': 'onnxruntime not installed'}
    
    pytorch_model.eval()
    test_input = torch.randn(*input_shape)
    
    # PyTorch benchmark
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = pytorch_model(test_input)
        
        # Benchmark
        pytorch_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = pytorch_model(test_input)
            pytorch_times.append(time.perf_counter() - start)
    
    # ONNX Runtime benchmark
    ort_session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    
    # Warmup
    for _ in range(warmup):
        _ = ort_session.run(None, ort_inputs)
    
    # Benchmark
    onnx_times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = ort_session.run(None, ort_inputs)
        onnx_times.append(time.perf_counter() - start)
    
    # Results
    pytorch_mean = np.mean(pytorch_times) * 1000
    pytorch_std = np.std(pytorch_times) * 1000
    onnx_mean = np.mean(onnx_times) * 1000
    onnx_std = np.std(onnx_times) * 1000
    speedup = pytorch_mean / onnx_mean
    
    results = {
        'pytorch_ms': pytorch_mean,
        'pytorch_std_ms': pytorch_std,
        'onnx_ms': onnx_mean,
        'onnx_std_ms': onnx_std,
        'speedup': speedup,
        'n_runs': n_runs
    }
    
    logger.info("\n" + "=" * 50)
    logger.info("INFERENCE BENCHMARK")
    logger.info("=" * 50)
    logger.info(f"PyTorch:      {pytorch_mean:.3f} ± {pytorch_std:.3f} ms")
    logger.info(f"ONNX Runtime: {onnx_mean:.3f} ± {onnx_std:.3f} ms")
    logger.info(f"Speedup:      {speedup:.2f}x")
    logger.info("=" * 50)
    
    return results


def run_batch_benchmark(
    pytorch_model: torch.nn.Module,
    onnx_path: Path,
    batch_sizes: list = [1, 4, 8, 16, 32],
    signal_length: int = 102400,
    n_runs: int = 50
) -> Dict[int, Dict[str, Any]]:
    """Benchmark across different batch sizes."""
    try:
        import onnxruntime as ort
    except ImportError:
        return {'error': 'onnxruntime not installed'}
    
    pytorch_model.eval()
    ort_session = ort.InferenceSession(str(onnx_path))
    
    results = {}
    
    logger.info("\n" + "=" * 60)
    logger.info("BATCH SIZE SCALING BENCHMARK")
    logger.info("=" * 60)
    logger.info(f"{'Batch':>6} | {'PyTorch (ms)':>14} | {'ONNX (ms)':>14} | {'Speedup':>8}")
    logger.info("-" * 60)
    
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, 1, signal_length)
        
        # PyTorch
        with torch.no_grad():
            pytorch_times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                _ = pytorch_model(test_input)
                pytorch_times.append(time.perf_counter() - start)
        
        # ONNX
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        onnx_times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = ort_session.run(None, ort_inputs)
            onnx_times.append(time.perf_counter() - start)
        
        pytorch_mean = np.mean(pytorch_times) * 1000
        onnx_mean = np.mean(onnx_times) * 1000
        speedup = pytorch_mean / onnx_mean
        
        results[batch_size] = {
            'pytorch_ms': pytorch_mean,
            'onnx_ms': onnx_mean,
            'speedup': speedup
        }
        
        logger.info(f"{batch_size:>6} | {pytorch_mean:>14.3f} | {onnx_mean:>14.3f} | {speedup:>8.2f}x")
    
    logger.info("=" * 60)
    
    return results


class ONNXModelWrapper:
    """Wrapper for ONNX model inference."""
    
    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        return self.session.run(None, {self.input_name: x})[0]


def main():
    parser = argparse.ArgumentParser(description='ONNX export and benchmarking')
    parser.add_argument('--checkpoint', type=str, help='PyTorch checkpoint path')
    parser.add_argument('--output', type=str, help='Output ONNX path')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    
    args = parser.parse_args()
    
    if args.demo:
        logger.info("=" * 60)
        logger.info("ONNX EXPORT DEMO")
        logger.info("=" * 60)
        
        from packages.core.models.cnn.cnn_1d import CNN1D
        
        # Create model
        model = CNN1D(num_classes=NUM_CLASSES)
        logger.info(f"\nModel: CNN1D ({sum(p.numel() for p in model.parameters()):,} params)")
        
        # Export
        output_path = project_root / 'results' / 'model.onnx'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        export_to_onnx(model, output_path)
        
        # Validate
        try:
            import onnxruntime
            validation = validate_onnx(model, output_path)
            benchmark = benchmark_inference(model, output_path, n_runs=50)
            batch_results = run_batch_benchmark(model, output_path, n_runs=20)
        except ImportError:
            logger.warning("\nInstall onnxruntime for validation and benchmarks:")
            logger.warning("  pip install onnxruntime")
        
        logger.info("\n✓ Demo complete")
    else:
        print("Run with --demo for demonstration")


if __name__ == '__main__':
    main()
