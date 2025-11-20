"""
Architecture Comparison Framework

Systematic comparison of all Phase 3 architectures:
- Accuracy metrics
- Model complexity (parameters, FLOPs)
- Inference time
- Memory usage
- Pareto frontier visualization

Provides comprehensive analysis to select best models for deployment.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns


def count_parameters(model: nn.Module) -> int:
    """
    Count total number of trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 1, 102400),
    device: str = 'cpu'
) -> int:
    """
    Estimate FLOPs (floating point operations) for model.

    Note: This is a simplified estimation. For more accurate FLOPs
    calculation, consider using libraries like fvcore or ptflops.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to run on

    Returns:
        Estimated FLOPs
    """
    model = model.to(device)
    model.eval()

    # Count multiply-add operations
    flops = 0

    def conv1d_flops(module, input, output):
        nonlocal flops
        batch_size, in_channels, input_length = input[0].shape
        out_channels, _, kernel_size = module.weight.shape
        output_length = output.shape[2]

        # Each output element requires kernel_size * in_channels multiply-adds
        flops += batch_size * out_channels * output_length * kernel_size * in_channels

    def linear_flops(module, input, output):
        nonlocal flops
        batch_size = input[0].shape[0]
        in_features = module.in_features
        out_features = module.out_features

        flops += batch_size * in_features * out_features

    # Register hooks
    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(conv1d_flops))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_flops))

    # Forward pass
    with torch.no_grad():
        dummy_input = torch.randn(input_shape).to(device)
        model(dummy_input)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return flops


def measure_inference_time(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 1, 102400),
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Measure inference time statistics.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs
        device: Device to run on

    Returns:
        Dictionary with timing statistics (mean, std, min, max)
    """
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(input_shape).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    # Synchronize if using GPU
    if device == 'cuda':
        torch.cuda.synchronize()

    # Measure inference time
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)

            if device == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to milliseconds

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'throughput_samples_per_sec': 1000.0 / np.mean(times)
    }


def measure_memory_usage(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 1, 102400),
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Measure memory usage of model.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        device: Device to run on

    Returns:
        Dictionary with memory statistics (MB)
    """
    model = model.to(device)

    # Model parameters memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    # Buffer memory
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 ** 2)

    # Forward pass memory (approximate)
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        dummy_input = torch.randn(input_shape).to(device)

        with torch.no_grad():
            _ = model(dummy_input)

        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_memory = param_memory + buffer_memory  # Rough estimate for CPU

    return {
        'parameters_mb': param_memory,
        'buffers_mb': buffer_memory,
        'peak_memory_mb': peak_memory
    }


def evaluate_model_accuracy(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Evaluate model accuracy on test set.

    Args:
        model: PyTorch model
        test_loader: Test data loader
        device: Device to run on

    Returns:
        Dictionary with accuracy metrics
    """
    model = model.to(device)
    model.eval()

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100.0 * correct / total

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'predictions': all_preds,
        'labels': all_labels
    }


def compare_architectures(
    model_dict: Dict[str, nn.Module],
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    input_shape: Tuple[int, ...] = (1, 1, 102400),
    device: str = 'cpu',
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Comprehensive comparison of multiple architectures.

    Args:
        model_dict: Dictionary of {model_name: model}
        test_loader: Test data loader for accuracy evaluation
        input_shape: Input shape for FLOPs/timing estimation
        device: Device to run on
        save_path: Path to save comparison results

    Returns:
        DataFrame with comparison results
    """
    results = []

    for name, model in model_dict.items():
        print(f"Evaluating {name}...")

        result = {'model_name': name}

        # Count parameters
        result['parameters'] = count_parameters(model)
        result['parameters_millions'] = result['parameters'] / 1e6

        # Compute FLOPs
        try:
            result['flops'] = compute_flops(model, input_shape, device)
            result['flops_billions'] = result['flops'] / 1e9
        except Exception as e:
            print(f"Warning: Could not compute FLOPs for {name}: {e}")
            result['flops'] = None
            result['flops_billions'] = None

        # Measure inference time
        try:
            timing = measure_inference_time(model, input_shape, num_runs=50, device=device)
            result.update(timing)
        except Exception as e:
            print(f"Warning: Could not measure inference time for {name}: {e}")

        # Measure memory
        try:
            memory = measure_memory_usage(model, input_shape, device)
            result.update(memory)
        except Exception as e:
            print(f"Warning: Could not measure memory for {name}: {e}")

        # Evaluate accuracy if test loader provided
        if test_loader is not None:
            try:
                accuracy_metrics = evaluate_model_accuracy(model, test_loader, device)
                result['test_accuracy'] = accuracy_metrics['accuracy']
            except Exception as e:
                print(f"Warning: Could not evaluate accuracy for {name}: {e}")

        results.append(result)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by test accuracy (if available)
    if 'test_accuracy' in df.columns:
        df = df.sort_values('test_accuracy', ascending=False)

    # Save results
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Results saved to {save_path}")

    return df


def plot_accuracy_vs_params(
    df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot accuracy vs. parameters (Pareto frontier).

    Args:
        df: DataFrame with comparison results
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))

    plt.scatter(
        df['parameters_millions'],
        df['test_accuracy'],
        s=100,
        alpha=0.7
    )

    # Annotate points
    for idx, row in df.iterrows():
        plt.annotate(
            row['model_name'],
            (row['parameters_millions'], row['test_accuracy']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )

    plt.xlabel('Parameters (Millions)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Model Accuracy vs. Parameters', fontsize=14)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_accuracy_vs_inference_time(
    df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot accuracy vs. inference time.

    Args:
        df: DataFrame with comparison results
        save_path: Path to save plot
    """
    plt.figure(figsize=(10, 6))

    plt.scatter(
        df['mean_ms'],
        df['test_accuracy'],
        s=100,
        alpha=0.7
    )

    # Annotate points
    for idx, row in df.iterrows():
        plt.annotate(
            row['model_name'],
            (row['mean_ms'], row['test_accuracy']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )

    plt.xlabel('Inference Time (ms)', fontsize=12)
    plt.ylabel('Test Accuracy (%)', fontsize=12)
    plt.title('Model Accuracy vs. Inference Time', fontsize=14)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_pareto_frontier(
    df: pd.DataFrame,
    save_path: Optional[str] = None
):
    """
    Plot multi-dimensional Pareto frontier.

    Args:
        df: DataFrame with comparison results
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Accuracy vs. Parameters
    axes[0].scatter(
        df['parameters_millions'],
        df['test_accuracy'],
        s=100,
        alpha=0.7,
        c=df['mean_ms'],
        cmap='viridis'
    )
    for idx, row in df.iterrows():
        axes[0].annotate(
            row['model_name'],
            (row['parameters_millions'], row['test_accuracy']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    axes[0].set_xlabel('Parameters (Millions)')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Accuracy vs. Parameters (color=inference time)')
    axes[0].grid(True, alpha=0.3)

    # Accuracy vs. Inference Time
    axes[1].scatter(
        df['mean_ms'],
        df['test_accuracy'],
        s=100,
        alpha=0.7,
        c=df['parameters_millions'],
        cmap='plasma'
    )
    for idx, row in df.iterrows():
        axes[1].annotate(
            row['model_name'],
            (row['mean_ms'], row['test_accuracy']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8
        )
    axes[1].set_xlabel('Inference Time (ms)')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Accuracy vs. Inference Time (color=parameters)')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pareto frontier plot saved to {save_path}")

    plt.close()


def print_comparison_summary(df: pd.DataFrame):
    """
    Print formatted comparison summary.

    Args:
        df: DataFrame with comparison results
    """
    print("\n" + "="*80)
    print("ARCHITECTURE COMPARISON SUMMARY")
    print("="*80 + "\n")

    # Format for display
    display_df = df.copy()

    # Select key columns
    columns_to_display = [
        'model_name',
        'test_accuracy',
        'parameters_millions',
        'mean_ms',
        'flops_billions'
    ]

    # Filter existing columns
    columns_to_display = [col for col in columns_to_display if col in display_df.columns]

    display_df = display_df[columns_to_display]

    # Rename for clarity
    display_df = display_df.rename(columns={
        'model_name': 'Model',
        'test_accuracy': 'Accuracy (%)',
        'parameters_millions': 'Params (M)',
        'mean_ms': 'Time (ms)',
        'flops_billions': 'FLOPs (B)'
    })

    # Print table
    print(display_df.to_string(index=False))
    print("\n" + "="*80)

    # Best models
    if 'Accuracy (%)' in display_df.columns:
        best_acc = display_df.loc[display_df['Accuracy (%)'].idxmax()]
        print(f"\n🏆 Best Accuracy: {best_acc['Model']} ({best_acc['Accuracy (%)']:.2f}%)")

    if 'Time (ms)' in display_df.columns:
        fastest = display_df.loc[display_df['Time (ms)'].idxmin()]
        print(f"⚡ Fastest: {fastest['Model']} ({fastest['Time (ms)']:.2f} ms)")

    if 'Params (M)' in display_df.columns:
        smallest = display_df.loc[display_df['Params (M)'].idxmin()]
        print(f"📦 Smallest: {smallest['Model']} ({smallest['Params (M)']:.2f}M params)")

    print("\n" + "="*80)


# Example usage
if __name__ == "__main__":
    print("Architecture Comparison Framework")
    print("This module provides tools for systematic model comparison.")
    print("\nUsage example:")
    print("""
    from evaluation.architecture_comparison import compare_architectures

    models = {
        'ResNet-18': resnet18_model,
        'ResNet-50': resnet50_model,
        'EfficientNet-B3': efficientnet_model
    }

    results = compare_architectures(
        models,
        test_loader=test_loader,
        device='cuda'
    )

    print_comparison_summary(results)
    plot_pareto_frontier(results, 'pareto_frontier.png')
    """)
