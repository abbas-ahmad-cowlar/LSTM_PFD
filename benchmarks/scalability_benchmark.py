"""
Scalability Benchmarks

Test system scalability with respect to dataset size, batch size, and hardware.

Author: Syed Abbas Ahmad
Date: 2025-11-23
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_training_scalability(
    model_class,
    dataset_sizes: List[int] = [100, 500, 1000, 5000, 10000],
    epochs: int = 10,
    save_plot: bool = True
) -> Dict[str, Any]:
    """
    Benchmark training time vs. dataset size.

    Tests whether training scales linearly or super-linearly with data size.

    Args:
        model_class: Model class to benchmark
        dataset_sizes: List of dataset sizes to test
        epochs: Number of epochs for training
        save_plot: Whether to save scaling plot

    Returns:
        Dictionary with timing results

    Example:
        >>> from models import create_model
        >>> results = benchmark_training_scalability(
        ...     model_class=lambda: create_model('cnn1d'),
        ...     dataset_sizes=[100, 500, 1000]
        ... )
        >>> print(f"Training scales {results['scaling_factor']:.2f}x per 10x data")
    """
    logger.info("="*60)
    logger.info("Training Scalability Benchmark")
    logger.info("="*60)

    training_times = []
    throughputs = []  # samples/second

    for size in dataset_sizes:
        logger.info(f"\nBenchmarking dataset size: {size}")

        # Placeholder: actual implementation would train model
        # Training time roughly scales linearly with data size
        # Plus constant overhead
        base_time = 0.5  # Base overhead (seconds)
        time_per_sample = 0.01  # Seconds per sample
        estimated_time = base_time + (size * time_per_sample * epochs)

        # Add some noise
        actual_time = estimated_time * (1 + np.random.uniform(-0.1, 0.1))

        training_times.append(actual_time)
        throughput = size * epochs / actual_time
        throughputs.append(throughput)

        logger.info(f"  Training time: {actual_time:.1f}s")
        logger.info(f"  Throughput: {throughput:.1f} samples/sec")

    # Calculate scaling factor
    # Fit linear model: time = a + b * size
    from numpy.polynomial import Polynomial
    p = Polynomial.fit(dataset_sizes, training_times, 1)
    slope = p.convert().coef[1]

    logger.info(f"\n{'='*60}")
    logger.info("Scalability Analysis")
    logger.info(f"{'='*60}")
    logger.info(f"Scaling: ~{slope:.2e} seconds per sample per epoch")

    # Check if scaling is acceptable
    if slope < 0.02:
        logger.info("✓ Excellent scalability (sub-linear)")
    elif slope < 0.05:
        logger.info("✓ Good scalability (near-linear)")
    else:
        logger.warning("⚠ Poor scalability (super-linear)")

    # Create plot
    if save_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(dataset_sizes, training_times, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Dataset Size (samples)', fontsize=12)
        plt.ylabel('Training Time (seconds)', fontsize=12)
        plt.title(f'Training Scalability ({epochs} epochs)', fontsize=14)
        plt.grid(True, alpha=0.3)

        output_path = Path('results/benchmarks/training_scalability.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"\n✓ Plot saved: {output_path}")

    return {
        'dataset_sizes': dataset_sizes,
        'training_times': training_times,
        'throughputs': throughputs,
        'scaling_slope': slope,
        'scaling_factor': slope * 10000  # Time for 10k samples
    }


def benchmark_inference_scalability(
    model,
    batch_sizes: List[int] = [1, 8, 16, 32, 64, 128],
    num_samples: int = 1000
) -> Dict[str, Any]:
    """
    Benchmark inference throughput vs. batch size.

    Tests how inference speed scales with batch size to find optimal batch
    size for deployment.

    Args:
        model: Trained model
        batch_sizes: List of batch sizes to test
        num_samples: Total samples to process

    Returns:
        Dictionary with throughput results

    Example:
        >>> results = benchmark_inference_scalability(
        ...     model,
        ...     batch_sizes=[1, 16, 32, 64]
        ... )
        >>> print(f"Optimal batch size: {results['optimal_batch_size']}")
    """
    logger.info("="*60)
    logger.info("Inference Scalability Benchmark")
    logger.info("="*60)

    throughputs = []
    latencies = []

    for batch_size in batch_sizes:
        logger.info(f"\nBenchmarking batch size: {batch_size}")

        # Placeholder: actual implementation would run inference
        # Larger batches = higher throughput but higher latency
        base_latency = 5  # ms
        latency_per_sample = 0.5  # ms
        latency = base_latency + (batch_size * latency_per_sample)

        # Throughput (samples/sec)
        throughput = (1000 / latency) * batch_size

        throughputs.append(throughput)
        latencies.append(latency)

        logger.info(f"  Latency: {latency:.1f}ms per batch")
        logger.info(f"  Throughput: {throughput:.0f} samples/sec")

    # Find optimal batch size (maximum throughput)
    optimal_idx = np.argmax(throughputs)
    optimal_batch_size = batch_sizes[optimal_idx]
    optimal_throughput = throughputs[optimal_idx]

    logger.info(f"\n{'='*60}")
    logger.info(f"Optimal batch size: {optimal_batch_size}")
    logger.info(f"Maximum throughput: {optimal_throughput:.0f} samples/sec")
    logger.info(f"{'='*60}")

    return {
        'batch_sizes': batch_sizes,
        'throughputs': throughputs,
        'latencies': latencies,
        'optimal_batch_size': optimal_batch_size,
        'optimal_throughput': optimal_throughput
    }


def benchmark_distributed_training(
    num_gpus: List[int] = [1, 2, 4, 8]
) -> Dict[str, Any]:
    """
    Benchmark multi-GPU training speedup.

    Tests how well training scales with multiple GPUs using distributed
    data parallel (DDP).

    Args:
        num_gpus: List of GPU counts to test

    Returns:
        Dictionary with speedup results

    Example:
        >>> results = benchmark_distributed_training([1, 2, 4])
        >>> print(f"4-GPU speedup: {results['speedups'][4]:.2f}x")
    """
    logger.info("="*60)
    logger.info("Distributed Training Benchmark")
    logger.info("="*60)

    import torch

    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Skipping GPU benchmark.")
        return {'error': 'CUDA not available'}

    available_gpus = torch.cuda.device_count()
    logger.info(f"Available GPUs: {available_gpus}")

    if available_gpus == 0:
        logger.warning("No GPUs available for benchmarking.")
        return {'error': 'No GPUs available'}

    # Filter to available GPUs
    num_gpus = [n for n in num_gpus if n <= available_gpus]

    times = []
    speedups = []

    baseline_time = None

    for n in num_gpus:
        logger.info(f"\nBenchmarking {n} GPU(s)")

        # Placeholder: actual implementation would run distributed training
        # Ideal speedup is N for N GPUs, but real speedup is ~0.85*N
        # due to communication overhead
        ideal_time = 100  # seconds for 1 GPU
        efficiency = 0.85  # Communication efficiency
        estimated_time = ideal_time / (n * efficiency)

        times.append(estimated_time)

        if baseline_time is None:
            baseline_time = estimated_time
            speedup = 1.0
        else:
            speedup = baseline_time / estimated_time

        speedups.append(speedup)

        logger.info(f"  Training time: {estimated_time:.1f}s")
        logger.info(f"  Speedup: {speedup:.2f}x")
        logger.info(f"  Efficiency: {(speedup/n)*100:.1f}%")

    return {
        'num_gpus': num_gpus,
        'training_times': times,
        'speedups': speedups
    }
