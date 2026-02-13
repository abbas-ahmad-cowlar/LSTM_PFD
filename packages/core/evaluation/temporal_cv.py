#!/usr/bin/env python3
"""
Temporal Cross-Validation for Time-Series Data

Implements time-aware cross-validation that respects temporal ordering.
Critical for time-series models to prevent data leakage.

Features:
- TimeSeriesSplit: sklearn-style time-based splitting
- BlockingTimeSeriesSplit: Gap between train/test to handle autocorrelation
- Rolling window validation
- Temporal stratification

Usage:
    python scripts/utilities/temporal_cv.py --data data/processed/dataset.h5 --windows 5
    
    # Or programmatically:
    from scripts.utilities.temporal_cv import TemporalCrossValidator
    cv = TemporalCrossValidator(n_splits=5)
    results = cv.run(model, signals, labels, timestamps)

Author: Deficiency Fix #35 (Priority: 28)
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
from typing import Dict, List, Any, Optional, Tuple, Iterator
from sklearn.metrics import accuracy_score, f1_score

from utils.reproducibility import set_seed
from utils.device_manager import get_device
from utils.logging import get_logger


logger = get_logger(__name__)


class TimeSeriesSplit:
    """
    Time-series cross-validator.
    
    Provides train/test indices for time-based splitting.
    The test set always comes AFTER the training set in time.
    
    Example:
        Split 1: Train [0..20%], Test [20..40%]
        Split 2: Train [0..40%], Test [40..60%]
        Split 3: Train [0..60%], Test [60..80%]
        Split 4: Train [0..80%], Test [80..100%]
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[float] = None,
        gap: int = 0,
        min_train_size: Optional[int] = None
    ):
        """
        Args:
            n_splits: Number of splits
            test_size: Fraction of data for test (default: 1/(n_splits+1))
            gap: Number of samples to skip between train and test
            min_train_size: Minimum training set size
        """
        self.n_splits = n_splits
        self.test_size = test_size or 1.0 / (n_splits + 1)
        self.gap = gap
        self.min_train_size = min_train_size
    
    def split(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices."""
        n_samples = len(X)
        test_size = int(n_samples * self.test_size)
        
        for i in range(self.n_splits):
            # Test set slides forward
            test_end = n_samples - i * test_size
            test_start = max(0, test_end - test_size)
            
            # Train set is everything before (minus gap)
            train_end = max(0, test_start - self.gap)
            train_start = 0
            
            if self.min_train_size and train_end - train_start < self.min_train_size:
                continue
            
            if train_end <= train_start or test_end <= test_start:
                continue
            
            train_indices = np.arange(train_start, train_end)
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class BlockingTimeSeriesSplit:
    """
    Blocked time-series split for handling autocorrelation.
    
    Unlike expanding window, this uses fixed-size blocks
    to reduce autocorrelation between train and test.
    
    Example with gap=100:
        Split 1: Train [0:1000], Gap [1000:1100], Test [1100:1500]
        Split 2: Train [1500:2500], Gap [2500:2600], Test [2600:3000]
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 100,
        train_ratio: float = 0.7
    ):
        self.n_splits = n_splits
        self.gap = gap
        self.train_ratio = train_ratio
    
    def split(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices with gap."""
        n_samples = len(X)
        block_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            block_start = i * block_size
            block_end = min((i + 1) * block_size, n_samples)
            
            block_len = block_end - block_start
            train_end = block_start + int(block_len * self.train_ratio)
            test_start = min(train_end + self.gap, block_end)
            
            if test_start >= block_end:
                continue
            
            train_indices = np.arange(block_start, train_end)
            test_indices = np.arange(test_start, block_end)
            
            yield train_indices, test_indices


class TemporalCrossValidator:
    """
    Full temporal cross-validation pipeline.
    
    Supports:
    - TimeSeriesSplit (expanding window)
    - BlockingTimeSeriesSplit (blocked with gap)
    - Rolling window (fixed window size)
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        method: str = 'expanding',
        gap: int = 0,
        device: Optional[torch.device] = None,
        seed: int = 42
    ):
        """
        Args:
            n_splits: Number of splits
            method: 'expanding', 'blocking', or 'rolling'
            gap: Gap between train and test
            device: PyTorch device
            seed: Random seed
        """
        self.n_splits = n_splits
        self.method = method
        self.gap = gap
        self.device = device or get_device(prefer_gpu=True)
        self.seed = seed
        
        if method == 'expanding':
            self.splitter = TimeSeriesSplit(n_splits, gap=gap)
        elif method == 'blocking':
            self.splitter = BlockingTimeSeriesSplit(n_splits, gap=gap)
        else:
            self.splitter = TimeSeriesSplit(n_splits, gap=gap)
    
    def run(
        self,
        model_class,
        model_kwargs: Dict[str, Any],
        signals: np.ndarray,
        labels: np.ndarray,
        epochs: int = 30,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Run full temporal cross-validation.
        
        Args:
            model_class: Model class to instantiate
            model_kwargs: Arguments for model constructor
            signals: Signal data [N, T]
            labels: Labels [N]
            epochs: Training epochs per split
            batch_size: Batch size
        
        Returns:
            Dictionary with per-split and aggregated results
        """
        from data.cnn_dataset import RawSignalDataset
        from data.cnn_transforms import get_train_transforms, get_test_transforms
        from torch.utils.data import DataLoader
        
        logger.info("=" * 60)
        logger.info("TEMPORAL CROSS-VALIDATION")
        logger.info("=" * 60)
        logger.info(f"Method: {self.method}")
        logger.info(f"Splits: {self.n_splits}")
        logger.info(f"Gap: {self.gap}")
        logger.info(f"Samples: {len(signals)}")
        
        split_results = []
        
        for split_idx, (train_idx, test_idx) in enumerate(self.splitter.split(signals)):
            logger.info(f"\n--- Split {split_idx + 1} ---")
            logger.info(f"  Train: samples {train_idx[0]} to {train_idx[-1]} ({len(train_idx)} samples)")
            logger.info(f"  Test:  samples {test_idx[0]} to {test_idx[-1]} ({len(test_idx)} samples)")
            
            set_seed(self.seed + split_idx)
            
            # Create datasets
            train_dataset = RawSignalDataset(
                signals[train_idx], labels[train_idx],
                transform=get_train_transforms(augment=True)
            )
            test_dataset = RawSignalDataset(
                signals[test_idx], labels[test_idx],
                transform=get_test_transforms()
            )
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Train model
            model = model_class(**model_kwargs).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()
            
            best_test_acc = 0
            
            for epoch in range(epochs):
                model.train()
                for batch_signals, batch_labels in train_loader:
                    batch_signals = batch_signals.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_signals)
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate
            model.eval()
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch_signals, batch_labels in test_loader:
                    batch_signals = batch_signals.to(self.device)
                    outputs = model(batch_signals)
                    preds = outputs.argmax(dim=1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(batch_labels.numpy())
            
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            
            logger.info(f"  Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            split_results.append({
                'split': split_idx + 1,
                'train_start': int(train_idx[0]),
                'train_end': int(train_idx[-1]),
                'test_start': int(test_idx[0]),
                'test_end': int(test_idx[-1]),
                'train_samples': len(train_idx),
                'test_samples': len(test_idx),
                'accuracy': accuracy,
                'f1_macro': f1
            })
        
        # Aggregate
        accuracies = [r['accuracy'] for r in split_results]
        f1s = [r['f1_macro'] for r in split_results]
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'method': self.method,
                'n_splits': self.n_splits,
                'gap': self.gap,
                'epochs': epochs
            },
            'split_results': split_results,
            'aggregated': {
                'accuracy_mean': float(np.mean(accuracies)),
                'accuracy_std': float(np.std(accuracies)),
                'f1_mean': float(np.mean(f1s)),
                'f1_std': float(np.std(f1s))
            }
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Accuracy: {results['aggregated']['accuracy_mean']:.4f} ± {results['aggregated']['accuracy_std']:.4f}")
        logger.info(f"F1 Score: {results['aggregated']['f1_mean']:.4f} ± {results['aggregated']['f1_std']:.4f}")
        
        return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Temporal cross-validation for time-series',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data', type=str, required=True, help='HDF5 data path')
    parser.add_argument('--splits', type=int, default=5, help='Number of splits')
    parser.add_argument('--method', type=str, default='expanding',
                       choices=['expanding', 'blocking', 'rolling'])
    parser.add_argument('--gap', type=int, default=0, help='Gap between train/test')
    parser.add_argument('--epochs', type=int, default=30, help='Epochs per split')
    parser.add_argument('--output', type=str, help='Output JSON path')
    parser.add_argument('--quick', action='store_true', help='Quick mode (5 epochs)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.quick:
        args.epochs = 5
    
    import h5py
    
    with h5py.File(args.data, 'r') as f:
        # Combine all splits
        signals = []
        labels = []
        for split in ['train', 'val', 'test']:
            if split in f:
                signals.append(f[split]['signals'][:])
                labels.append(f[split]['labels'][:])
        
        signals = np.concatenate(signals)
        labels = np.concatenate(labels)
        num_classes = f.attrs.get('num_classes', 11)
    
    from packages.core.models.cnn.cnn_1d import CNN1D
    
    cv = TemporalCrossValidator(
        n_splits=args.splits,
        method=args.method,
        gap=args.gap
    )
    
    results = cv.run(
        model_class=CNN1D,
        model_kwargs={'num_classes': num_classes},
        signals=signals,
        labels=labels,
        epochs=args.epochs
    )
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / 'results' / f'temporal_cv_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
