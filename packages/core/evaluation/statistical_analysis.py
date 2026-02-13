#!/usr/bin/env python3
"""
Statistical Significance Testing for Bearing Fault Diagnosis

This script runs experiments with multiple random seeds and computes
statistical significance using confidence intervals and hypothesis tests.

Features:
- Multi-seed experiments (configurable, default=5)
- 95% confidence intervals
- Paired t-tests for model comparison
- Wilcoxon signed-rank tests (non-parametric alternative)
- Publication-ready statistics tables

Usage:
    # Run multi-seed experiments
    python scripts/utilities/statistical_analysis.py --data data/processed/dataset.h5 --seeds 5
    
    # Compare two models
    python scripts/utilities/statistical_analysis.py --data data/processed/dataset.h5 \
        --model1 cnn1d --model2 attention --seeds 5

Author: Critical Deficiency Fix #5 (Priority: 90)
Date: 2026-01-18
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats

from utils.reproducibility import set_seed
from utils.device_manager import get_device
from utils.logging import get_logger


logger = get_logger(__name__)


def compute_confidence_interval(
    values: List[float], 
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for a list of values.
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    n = len(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample standard deviation
    
    # t-critical value for two-tailed test
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    
    margin = t_crit * std / np.sqrt(n)
    
    return mean, mean - margin, mean + margin


def paired_ttest(values1: List[float], values2: List[float]) -> Dict[str, float]:
    """
    Perform paired t-test between two sets of results.
    
    Returns:
        Dictionary with t-statistic, p-value, and significance
    """
    t_stat, p_value = stats.ttest_rel(values1, values2)
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant_005': p_value < 0.05,
        'significant_001': p_value < 0.01
    }


def wilcoxon_test(values1: List[float], values2: List[float]) -> Dict[str, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative).
    
    Returns:
        Dictionary with statistic, p-value, and significance
    """
    try:
        stat, p_value = stats.wilcoxon(values1, values2)
        return {
            'statistic': float(stat),
            'p_value': float(p_value),
            'significant_005': p_value < 0.05,
            'significant_001': p_value < 0.01
        }
    except ValueError as e:
        # Wilcoxon fails if differences are all zero
        return {
            'statistic': None,
            'p_value': None,
            'error': str(e)
        }


class MultiSeedExperiment:
    """Runs experiments across multiple random seeds."""
    
    def __init__(
        self,
        model_class: str = 'cnn1d',
        num_seeds: int = 5,
        epochs: int = 30,
        batch_size: int = 32,
        device: Optional[torch.device] = None
    ):
        self.model_class = model_class
        self.num_seeds = num_seeds
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device or get_device(prefer_gpu=True)
        self.base_seed = 42
    
    def train_single_seed(
        self,
        seed: int,
        train_loader,
        val_loader,
        test_loader,
        num_classes: int
    ) -> Dict[str, float]:
        """Train model with a specific seed and return test metrics."""
        from models.cnn.cnn_1d import CNN1D
        from models.cnn.attention_cnn import AttentionCNN1D, LightweightAttentionCNN
        from models.cnn.multi_scale_cnn import MultiScaleCNN1D
        from training.cnn_trainer import CNNTrainer
        from training.cnn_optimizer import create_optimizer
        from training.cnn_losses import create_criterion
        from training.cnn_schedulers import create_cosine_scheduler
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        set_seed(seed)
        
        model_registry = {
            'cnn1d': CNN1D,
            'attention': AttentionCNN1D,
            'attention-lite': LightweightAttentionCNN,
            'multiscale': MultiScaleCNN1D
        }
        
        model_cls = model_registry.get(self.model_class)
        if model_cls is None:
            raise ValueError(f"Unknown model: {self.model_class}")
        
        # Create model
        if self.model_class == 'cnn1d':
            model = model_cls(num_classes=num_classes, input_channels=1, dropout=0.3)
        else:
            model = model_cls(num_classes=num_classes, input_length=102400, in_channels=1)
        
        # Training setup
        optimizer = create_optimizer('adamw', model.parameters(), lr=0.001)
        criterion = create_criterion('cross_entropy', num_classes=num_classes)
        scheduler = create_cosine_scheduler(optimizer, self.epochs)
        
        trainer = CNNTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            lr_scheduler=scheduler
        )
        
        # Train
        best_val_acc = 0
        best_state = None
        
        for epoch in range(self.epochs):
            trainer.current_epoch = epoch
            trainer.train_epoch()
            val_metrics = trainer.validate_epoch()
            
            if scheduler:
                scheduler.step()
            
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        # Load best and evaluate on test
        if best_state:
            model.load_state_dict(best_state)
        
        model.eval()
        model.to(self.device)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_signals, batch_labels in test_loader:
                batch_signals = batch_signals.to(self.device)
                outputs = model(batch_signals)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        return {
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0)
        }
    
    def run_experiments(self, hdf5_path: str) -> Dict[str, Any]:
        """Run experiments across all seeds."""
        import h5py
        from data.cnn_dataset import RawSignalDataset
        from data.cnn_transforms import get_train_transforms, get_test_transforms
        from data.cnn_dataloader import create_cnn_dataloaders
        
        logger.info(f"Running {self.num_seeds}-seed experiment for {self.model_class}")
        
        # Load data
        with h5py.File(hdf5_path, 'r') as f:
            train_signals = f['train']['signals'][:]
            train_labels = f['train']['labels'][:]
            val_signals = f['val']['signals'][:]
            val_labels = f['val']['labels'][:]
            test_signals = f['test']['signals'][:]
            test_labels = f['test']['labels'][:]
            num_classes = f.attrs.get('num_classes', 11)
        
        # Create datasets
        train_ds = RawSignalDataset(train_signals, train_labels, get_train_transforms(True))
        val_ds = RawSignalDataset(val_signals, val_labels, get_test_transforms())
        test_ds = RawSignalDataset(test_signals, test_labels, get_test_transforms())
        
        loaders = create_cnn_dataloaders(train_ds, val_ds, test_ds, self.batch_size, num_workers=4)
        
        # Run multiple seeds
        seed_results = []
        
        for i in range(self.num_seeds):
            seed = self.base_seed + i * 100
            logger.info(f"  Seed {i+1}/{self.num_seeds} (seed={seed})...")
            
            metrics = self.train_single_seed(
                seed=seed,
                train_loader=loaders['train'],
                val_loader=loaders['val'],
                test_loader=loaders['test'],
                num_classes=num_classes
            )
            
            seed_results.append({
                'seed': seed,
                **metrics
            })
            
            logger.info(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")
        
        # Compute statistics
        metric_keys = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
        statistics = {}
        
        for key in metric_keys:
            values = [r[key] for r in seed_results]
            mean, ci_low, ci_high = compute_confidence_interval(values)
            
            statistics[key] = {
                'mean': mean,
                'std': float(np.std(values, ddof=1)),
                'ci_95_low': ci_low,
                'ci_95_high': ci_high,
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': values
            }
        
        return {
            'model': self.model_class,
            'num_seeds': self.num_seeds,
            'epochs': self.epochs,
            'seed_results': seed_results,
            'statistics': statistics,
            'timestamp': datetime.now().isoformat()
        }


def compare_models(
    results1: Dict[str, Any],
    results2: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare two models using statistical tests."""
    model1 = results1['model']
    model2 = results2['model']
    
    logger.info(f"Comparing {model1} vs {model2}")
    
    comparisons = {}
    
    for metric in ['accuracy', 'f1_macro']:
        values1 = results1['statistics'][metric]['values']
        values2 = results2['statistics'][metric]['values']
        
        ttest = paired_ttest(values1, values2)
        wilcox = wilcoxon_test(values1, values2)
        
        # Effect size (Cohen's d)
        diff = np.array(values1) - np.array(values2)
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
        
        comparisons[metric] = {
            'model1_mean': results1['statistics'][metric]['mean'],
            'model2_mean': results2['statistics'][metric]['mean'],
            'difference': results1['statistics'][metric]['mean'] - results2['statistics'][metric]['mean'],
            'paired_ttest': ttest,
            'wilcoxon': wilcox,
            'cohens_d': float(cohens_d)
        }
    
    return {
        'model1': model1,
        'model2': model2,
        'comparisons': comparisons
    }


def print_results(results: Dict[str, Any]) -> None:
    """Print formatted multi-seed experiment results."""
    print("\n" + "=" * 80)
    print("MULTI-SEED EXPERIMENT RESULTS")
    print("=" * 80)
    
    print(f"\nModel: {results['model']}")
    print(f"Seeds: {results['num_seeds']}")
    print(f"Epochs: {results['epochs']}")
    
    print("\n" + "-" * 80)
    print("PER-SEED RESULTS")
    print("-" * 80)
    print(f"{'Seed':^8} | {'Accuracy':^10} | {'F1 Macro':^10} | {'Precision':^10} | {'Recall':^10}")
    print("-" * 60)
    
    for r in results['seed_results']:
        print(f"{r['seed']:^8} | {r['accuracy']:^10.4f} | {r['f1_macro']:^10.4f} | "
              f"{r['precision_macro']:^10.4f} | {r['recall_macro']:^10.4f}")
    
    print("\n" + "-" * 80)
    print("STATISTICAL SUMMARY (with 95% CI)")
    print("-" * 80)
    
    stats = results['statistics']
    
    for metric in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']:
        s = stats[metric]
        print(f"\n{metric.upper()}:")
        print(f"  Mean:    {s['mean']:.4f} ± {s['std']:.4f}")
        print(f"  95% CI:  [{s['ci_95_low']:.4f}, {s['ci_95_high']:.4f}]")
        print(f"  Range:   [{s['min']:.4f}, {s['max']:.4f}]")
    
    # Publication-ready summary
    print("\n" + "=" * 80)
    print("PUBLICATION-READY SUMMARY")
    print("=" * 80)
    
    acc = stats['accuracy']
    f1 = stats['f1_macro']
    
    print(f"\n  Accuracy: {acc['mean']*100:.2f}% ± {acc['std']*100:.2f}% (95% CI: [{acc['ci_95_low']*100:.2f}%, {acc['ci_95_high']*100:.2f}%])")
    print(f"  F1 Score: {f1['mean']*100:.2f}% ± {f1['std']*100:.2f}% (95% CI: [{f1['ci_95_low']*100:.2f}%, {f1['ci_95_high']*100:.2f}%])")
    print(f"\n  (Results computed over {results['num_seeds']} independent runs)")
    print("=" * 80)


def print_comparison(comparison: Dict[str, Any]) -> None:
    """Print model comparison results."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    print(f"\n{comparison['model1']} vs {comparison['model2']}")
    
    for metric, data in comparison['comparisons'].items():
        print(f"\n{metric.upper()}:")
        print(f"  {comparison['model1']}: {data['model1_mean']:.4f}")
        print(f"  {comparison['model2']}: {data['model2_mean']:.4f}")
        print(f"  Difference: {data['difference']:.4f}")
        print(f"  Cohen's d: {data['cohens_d']:.3f}")
        
        ttest = data['paired_ttest']
        print(f"  Paired t-test: t={ttest['t_statistic']:.3f}, p={ttest['p_value']:.4f}", end="")
        if ttest['significant_001']:
            print(" (p < 0.01 **)")
        elif ttest['significant_005']:
            print(" (p < 0.05 *)")
        else:
            print(" (not significant)")
    
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run multi-seed experiments with statistical analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to HDF5 dataset file')
    parser.add_argument('--model1', type=str, default='cnn1d',
                       help='First model to test')
    parser.add_argument('--model2', type=str, default=None,
                       help='Second model for comparison (optional)')
    parser.add_argument('--seeds', type=int, default=5,
                       help='Number of random seeds')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Epochs per run')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (3 seeds, 5 epochs)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.quick:
        args.seeds = 3
        args.epochs = 5
        logger.info("Quick mode: 3 seeds, 5 epochs")
    
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        sys.exit(1)
    
    # Run experiments for first model
    exp1 = MultiSeedExperiment(
        model_class=args.model1,
        num_seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    results1 = exp1.run_experiments(str(data_path))
    print_results(results1)
    
    all_results = {'model1': results1}
    
    # Optionally compare with second model
    if args.model2:
        exp2 = MultiSeedExperiment(
            model_class=args.model2,
            num_seeds=args.seeds,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        results2 = exp2.run_experiments(str(data_path))
        print_results(results2)
        
        comparison = compare_models(results1, results2)
        print_comparison(comparison)
        
        all_results['model2'] = results2
        all_results['comparison'] = comparison
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / 'results' / f'stats_{args.model1}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
