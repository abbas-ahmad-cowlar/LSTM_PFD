"""
Compare Multiple Experiments

Provides tools for:
- Loading experiment results
- Statistical comparison
- Generating comparison reports
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


def load_experiments(experiment_ids: List[str]) -> pd.DataFrame:
    """
    Load multiple experiments from MLflow.

    Args:
        experiment_ids: List of experiment IDs to load

    Returns:
        DataFrame with experiment results
    """
    try:
        import mlflow
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH

        data = []

        for exp_id in experiment_ids:
            # Get runs for this experiment
            runs = mlflow.search_runs(experiment_ids=[exp_id])

            for _, run in runs.iterrows():
                data.append({
                    'experiment_id': exp_id,
                    'run_id': run['run_id'],
                    'metrics': run.filter(regex=r'^metrics\.').to_dict(),
                    'params': run.filter(regex=r'^params\.').to_dict()
                })

        return pd.DataFrame(data)

    except ImportError:
        print("MLflow not available")
        return pd.DataFrame()


def compare_metrics(
    experiments: pd.DataFrame,
    metric_names: List[str]
) -> pd.DataFrame:
    """
    Compare metrics across experiments.

    Args:
        experiments: DataFrame from load_experiments
        metric_names: List of metric names to compare

    Returns:
        Comparison DataFrame
    """
    comparison_data = []

    for exp_id in experiments['experiment_id'].unique():
        exp_data = experiments[experiments['experiment_id'] == exp_id]

        for metric_name in metric_names:
            metric_col = f'metrics.{metric_name}'

            if metric_col in exp_data.columns:
                values = exp_data[metric_col].dropna()

                if len(values) > 0:
                    comparison_data.append({
                        'experiment': exp_id,
                        'metric': metric_name,
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max()
                    })

    return pd.DataFrame(comparison_data)


def generate_comparison_report(
    experiments: pd.DataFrame,
    metric_names: List[str]
) -> str:
    """
    Generate text report comparing experiments.

    Args:
        experiments: DataFrame from load_experiments
        metric_names: List of metric names to compare

    Returns:
        Formatted comparison report
    """
    comparison = compare_metrics(experiments, metric_names)

    report = "=" * 60 + "\n"
    report += "EXPERIMENT COMPARISON REPORT\n"
    report += "=" * 60 + "\n\n"

    for metric in metric_names:
        metric_data = comparison[comparison['metric'] == metric]

        if len(metric_data) == 0:
            continue

        report += f"\n{metric}:\n"
        report += "-" * 40 + "\n"

        for _, row in metric_data.iterrows():
            report += f"  {row['experiment']}:\n"
            report += f"    Mean: {row['mean']:.4f} ± {row['std']:.4f}\n"
            report += f"    Range: [{row['min']:.4f}, {row['max']:.4f}]\n"

    report += "\n" + "=" * 60 + "\n"

    return report
