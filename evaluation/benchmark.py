"""
Benchmark Models Against Baselines

Compare deep learning models against:
- Classical ML baselines
- Previous model versions
- Literature benchmarks
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import torch
from torch.utils.data import DataLoader


def benchmark_against_classical(
    dl_model: torch.nn.Module,
    test_loader: DataLoader,
    classical_results: Dict[str, float],
    device: str = 'cuda'
) -> pd.DataFrame:
    """
    Compare DL model against classical ML baselines.

    Args:
        dl_model: Deep learning model to evaluate
        test_loader: Test data loader
        classical_results: Dictionary of classical model results
                          {'model_name': accuracy, ...}
        device: Device for evaluation

    Returns:
        Comparison DataFrame
    """
    # Evaluate DL model
    dl_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = dl_model(inputs)
            _, predicted = outputs.max(1)

            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    dl_accuracy = 100.0 * correct / total

    # Create comparison table
    results = []

    # Add classical models
    for model_name, accuracy in classical_results.items():
        results.append({
            'Model': model_name,
            'Type': 'Classical',
            'Accuracy': accuracy
        })

    # Add DL model
    results.append({
        'Model': dl_model.__class__.__name__,
        'Type': 'Deep Learning',
        'Accuracy': dl_accuracy
    })

    df = pd.DataFrame(results)
    df = df.sort_values('Accuracy', ascending=False)

    return df


def generate_comparison_table(results: List[Dict]) -> pd.DataFrame:
    """
    Generate comparison table from results.

    Args:
        results: List of result dictionaries

    Returns:
        Formatted comparison DataFrame
    """
    df = pd.DataFrame(results)
    return df.sort_values('Accuracy', ascending=False)
