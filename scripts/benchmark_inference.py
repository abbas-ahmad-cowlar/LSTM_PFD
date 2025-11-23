#!/usr/bin/env python3
"""
Inference Benchmarking Script

Benchmark inference speed across different backends and optimizations.

Usage:
    # Benchmark single model
    python scripts/benchmark_inference.py \\
        --model checkpoints/phase6/best_model.pth \\
        --backend torch

    # Compare multiple backends
    python scripts/benchmark_inference.py \\
        --model checkpoints/phase6/best_model.pth \\
        --backends torch torch_fp16 onnx \\
        --compare

    # Detailed profiling
    python scripts/benchmark_inference.py \\
        --model checkpoints/phase6/best_model.pth \\
        --backend torch \\
        --profile \\
        --num-runs 1000

Author: Syed Abbas Ahmad
Date: 2025-11-20
"""

import argparse
import sys
import logging
from pathlib import Path
import torch
import numpy as np
import json
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE
from deployment.inference import (
    TorchInferenceEngine,
    ONNXInferenceEngine,
    OptimizedInferenceEngine,
    InferenceConfig,
    benchmark_inference,
    compare_backends
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_benchmark_results(results: dict, save_path: str = None):
    """Plot benchmark results."""
    backends = list(results.keys())
    latencies = [results[b]['mean_latency_ms'] for b in backends]
    std_devs = [results[b]['std_latency_ms'] for b in backends]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Latency comparison
    colors = plt.cm.viridis(np.linspace(0, 1, len(backends)))
    bars = ax1.bar(backends, latencies, yerr=std_devs, color=colors, alpha=0.7)
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Inference Latency by Backend')
    ax1.set_xticklabels(backends, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, latency in zip(bars, latencies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{latency:.2f}ms',
                ha='center', va='bottom')

    # Throughput comparison
    throughputs = [results[b]['throughput_samples_per_sec'] for b in backends]
    bars = ax2.bar(backends, throughputs, color=colors, alpha=0.7)
    ax2.set_ylabel('Throughput (samples/sec)')
    ax2.set_title('Inference Throughput by Backend')
    ax2.set_xticklabels(backends, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, throughput in zip(bars, throughputs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{throughput:.1f}',
                ha='center', va='bottom')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference")

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--backend',
        type=str,
        choices=['torch', 'torch_fp16', 'onnx'],
        default='torch',
        help='Inference backend'
    )

    parser.add_argument(
        '--backends',
        type=str,
        nargs='+',
        default=None,
        help='Multiple backends to compare'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple backends'
    )

    parser.add_argument(
        '--input-shape',
        type=int,
        nargs='+',
        default=[1, 1, SIGNAL_LENGTH],
        help='Input tensor shape'
    )

    parser.add_argument(
        '--num-runs',
        type=int,
        default=100,
        help='Number of benchmark runs'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for inference'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device for inference'
    )

    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable detailed profiling'
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Plot benchmark results'
    )

    parser.add_argument(
        '--save-results',
        type=str,
        default=None,
        help='Save results to JSON file'
    )

    args = parser.parse_args()

    input_shape = tuple(args.input_shape)

    # Compare multiple backends
    if args.compare or args.backends:
        logger.info(f"\n{'='*60}")
        logger.info("Comparing Multiple Backends")
        logger.info(f"{'='*60}\n")

        backends_to_test = args.backends if args.backends else ['torch', 'onnx']

        results = benchmark_inference(
            args.model,
            input_shape,
            backends=backends_to_test,
            num_runs=args.num_runs
        )

        # Print comparison
        compare_backends(results)

        # Save results
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\n✓ Results saved to {args.save_results}")

        # Plot results
        if args.plot:
            plot_path = args.save_results.replace('.json', '.png') if args.save_results else 'benchmark_results.png'
            plot_benchmark_results(results, plot_path)

    else:
        # Benchmark single backend
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {args.backend}")
        logger.info(f"{'='*60}\n")

        # Initialize engine
        config = InferenceConfig(
            device=args.device,
            batch_size=args.batch_size,
            use_amp=(args.backend == 'torch_fp16')
        )

        if args.backend.startswith('torch'):
            engine = TorchInferenceEngine(config)
            engine.load_model(args.model)
        elif args.backend == 'onnx':
            if not args.model.endswith('.onnx'):
                logger.error("ONNX backend requires .onnx model file")
                return
            engine = ONNXInferenceEngine(config)
            engine.load_model(args.model)
        else:
            logger.error(f"Unknown backend: {args.backend}")
            return

        # Run benchmark
        stats = engine.benchmark(input_shape, num_runs=args.num_runs)

        # Print results
        logger.info(f"\n{'='*60}")
        logger.info("Benchmark Results")
        logger.info(f"{'='*60}")
        logger.info(f"Backend:       {args.backend}")
        logger.info(f"Device:        {args.device}")
        logger.info(f"Input shape:   {input_shape}")
        logger.info(f"Num runs:      {args.num_runs}")
        logger.info("-"*60)
        logger.info(f"Mean latency:  {stats['mean_latency_ms']:.2f} ± {stats['std_latency_ms']:.2f} ms")
        logger.info(f"Min latency:   {stats['min_latency_ms']:.2f} ms")
        logger.info(f"Max latency:   {stats['max_latency_ms']:.2f} ms")
        logger.info(f"Median:        {stats['median_latency_ms']:.2f} ms")
        logger.info(f"P95:           {stats['p95_latency_ms']:.2f} ms")
        logger.info(f"P99:           {stats['p99_latency_ms']:.2f} ms")
        logger.info(f"Throughput:    {stats['throughput_samples_per_sec']:.1f} samples/sec")
        logger.info(f"{'='*60}\n")

        # Save results
        if args.save_results:
            with open(args.save_results, 'w') as f:
                json.dump({args.backend: stats}, f, indent=2)
            logger.info(f"✓ Results saved to {args.save_results}\n")


if __name__ == '__main__':
    main()
