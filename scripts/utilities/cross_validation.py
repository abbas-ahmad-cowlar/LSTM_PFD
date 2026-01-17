#!/usr/bin/env python3
"""
K-Fold Cross-Validation Script for Bearing Fault Diagnosis

This script implements proper K-fold cross-validation to substantiate
accuracy claims with statistical confidence.

Usage:
    python scripts/utilities/cross_validation.py --data data/processed/dataset.h5 --k 5 --epochs 50
    
    # Quick validation (for testing)
    python scripts/utilities/cross_validation.py --data data/processed/dataset.h5 --k 3 --epochs 1 --quick

Author: Critical Deficiency Fix #1 (Priority: 100)
Date: 2026-01-18
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np
import torch
import h5py
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from utils.reproducibility import set_seed
from utils.device_manager import get_device
from utils.logging import get_logger


logger = get_logger(__name__)


class CrossValidationTrainer:
    """Handles K-fold cross-validation for CNN models."""
    
    def __init__(
        self,
        model_class: str = 'cnn1d',
        k_folds: int = 5,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 0.001,
        device: Optional[torch.device] = None,
        seed: int = 42
    ):
        self.model_class = model_class
        self.k_folds = k_folds
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or get_device(prefer_gpu=True)
        self.seed = seed
        
        # Import model registry
        from models.cnn.cnn_1d import CNN1D
        from models.cnn.attention_cnn import AttentionCNN1D, LightweightAttentionCNN
        from models.cnn.multi_scale_cnn import MultiScaleCNN1D, DilatedMultiScaleCNN
        
        self.model_registry = {
            'cnn1d': CNN1D,
            'attention': AttentionCNN1D,
            'attention-lite': LightweightAttentionCNN,
            'multiscale': MultiScaleCNN1D,
            'dilated': DilatedMultiScaleCNN
        }
    
    def load_full_dataset(self, hdf5_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load all signals and labels from HDF5 for cross-validation.
        
        Combines train/val/test splits into a single dataset for proper CV.
        """
        logger.info(f"Loading dataset from {hdf5_path}")
        
        with h5py.File(hdf5_path, 'r') as f:
            signals_list = []
            labels_list = []
            
            for split in ['train', 'val', 'test']:
                if split in f:
                    signals_list.append(f[split]['signals'][:])
                    labels_list.append(f[split]['labels'][:])
            
            self.sampling_rate = f.attrs.get('sampling_rate', 20480)
            self.num_classes = f.attrs.get('num_classes', 11)
        
        signals = np.concatenate(signals_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        
        logger.info(f"Loaded {len(signals)} samples, {self.num_classes} classes")
        return signals, labels
    
    def create_model(self, num_classes: int) -> torch.nn.Module:
        """Create a fresh model instance for each fold."""
        model_cls = self.model_registry.get(self.model_class)
        if model_cls is None:
            raise ValueError(f"Unknown model: {self.model_class}")
        
        if self.model_class == 'cnn1d':
            return model_cls(
                num_classes=num_classes,
                input_channels=1,
                dropout=0.3,
                use_batch_norm=True
            )
        else:
            return model_cls(
                num_classes=num_classes,
                input_length=102400,
                in_channels=1,
                dropout=0.3
            )
    
    def train_fold(
        self,
        fold_idx: int,
        train_signals: np.ndarray,
        train_labels: np.ndarray,
        val_signals: np.ndarray,
        val_labels: np.ndarray
    ) -> Dict[str, Any]:
        """Train and evaluate a single fold."""
        from data.cnn_dataset import RawSignalDataset
        from data.cnn_transforms import get_train_transforms, get_test_transforms
        from data.cnn_dataloader import create_cnn_dataloaders
        from training.cnn_trainer import CNNTrainer
        from training.cnn_optimizer import create_optimizer
        from training.cnn_losses import create_criterion
        from training.cnn_schedulers import create_cosine_scheduler
        
        logger.info(f"\n{'='*60}")
        logger.info(f"FOLD {fold_idx + 1}/{self.k_folds}")
        logger.info(f"{'='*60}")
        logger.info(f"  Train: {len(train_signals)} | Val: {len(val_signals)}")
        
        # Create datasets
        train_dataset = RawSignalDataset(
            signals=train_signals,
            labels=train_labels,
            transform=get_train_transforms(augment=True)
        )
        val_dataset = RawSignalDataset(
            signals=val_signals,
            labels=val_labels,
            transform=get_test_transforms()
        )
        
        # Create dataloaders
        loaders = create_cnn_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=self.batch_size,
            num_workers=4
        )
        
        # Create fresh model for this fold
        model = self.create_model(self.num_classes)
        
        # Optimizer and scheduler
        optimizer = create_optimizer(
            'adamw',
            model.parameters(),
            lr=self.lr,
            weight_decay=0.0001
        )
        criterion = create_criterion('cross_entropy', num_classes=self.num_classes)
        scheduler = create_cosine_scheduler(optimizer, num_epochs=self.epochs, eta_min=1e-6)
        
        # Trainer
        trainer = CNNTrainer(
            model=model,
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            lr_scheduler=scheduler,
            mixed_precision=False
        )
        
        # Training loop
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(self.epochs):
            trainer.current_epoch = epoch
            train_metrics = trainer.train_epoch()
            val_metrics = trainer.validate_epoch()
            
            if scheduler:
                scheduler.step()
            
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"  Epoch {epoch+1:3d}: Train Acc={train_metrics['accuracy']:.4f}, "
                           f"Val Acc={val_metrics['accuracy']:.4f}")
        
        # Load best model and compute final metrics
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        model.eval()
        model.to(self.device)
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_signals, batch_labels in loaders['val']:
                batch_signals = batch_signals.to(self.device)
                outputs = model(batch_signals)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_labels.numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        fold_metrics = {
            'fold': fold_idx + 1,
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
            'f1_weighted': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
            'precision_macro': precision_score(all_labels, all_preds, average='macro', zero_division=0),
            'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
            'num_train_samples': len(train_signals),
            'num_val_samples': len(val_signals)
        }
        
        logger.info(f"  ✓ Fold {fold_idx + 1} complete: Accuracy={fold_metrics['accuracy']:.4f}, "
                   f"F1={fold_metrics['f1_macro']:.4f}")
        
        return fold_metrics
    
    def run_cv(self, hdf5_path: str) -> Dict[str, Any]:
        """Run full K-fold cross-validation."""
        set_seed(self.seed)
        
        # Load full dataset
        signals, labels = self.load_full_dataset(hdf5_path)
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
        
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(signals, labels)):
            fold_seed = self.seed + fold_idx
            set_seed(fold_seed)
            
            train_signals = signals[train_idx]
            train_labels = labels[train_idx]
            val_signals = signals[val_idx]
            val_labels = labels[val_idx]
            
            fold_metrics = self.train_fold(
                fold_idx=fold_idx,
                train_signals=train_signals,
                train_labels=train_labels,
                val_signals=val_signals,
                val_labels=val_labels
            )
            fold_results.append(fold_metrics)
        
        # Aggregate results
        metric_keys = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
        aggregated = {}
        
        for key in metric_keys:
            values = [f[key] for f in fold_results]
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': values
            }
        
        # Final results
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'model_class': self.model_class,
                'k_folds': self.k_folds,
                'epochs': self.epochs,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'seed': self.seed,
                'device': str(self.device)
            },
            'dataset': {
                'path': str(hdf5_path),
                'total_samples': len(signals),
                'num_classes': self.num_classes,
                'sampling_rate': self.sampling_rate
            },
            'fold_results': fold_results,
            'aggregated': aggregated
        }
        
        return results


def print_results(results: Dict[str, Any]) -> None:
    """Print formatted cross-validation results."""
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 80)
    
    config = results['config']
    print(f"\nConfiguration:")
    print(f"  Model:      {config['model_class']}")
    print(f"  K-Folds:    {config['k_folds']}")
    print(f"  Epochs:     {config['epochs']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Seed:       {config['seed']}")
    
    print(f"\nDataset:")
    print(f"  Samples: {results['dataset']['total_samples']}")
    print(f"  Classes: {results['dataset']['num_classes']}")
    
    print(f"\n{'='*80}")
    print("PER-FOLD RESULTS")
    print("=" * 80)
    print(f"{'Fold':^6} | {'Accuracy':^10} | {'F1 Macro':^10} | {'Precision':^10} | {'Recall':^10}")
    print("-" * 60)
    
    for fold in results['fold_results']:
        print(f"{fold['fold']:^6} | {fold['accuracy']:^10.4f} | {fold['f1_macro']:^10.4f} | "
              f"{fold['precision_macro']:^10.4f} | {fold['recall_macro']:^10.4f}")
    
    print("-" * 60)
    
    agg = results['aggregated']
    print(f"{'Mean':^6} | {agg['accuracy']['mean']:^10.4f} | {agg['f1_macro']['mean']:^10.4f} | "
          f"{agg['precision_macro']['mean']:^10.4f} | {agg['recall_macro']['mean']:^10.4f}")
    print(f"{'±Std':^6} | {agg['accuracy']['std']:^10.4f} | {agg['f1_macro']['std']:^10.4f} | "
          f"{agg['precision_macro']['std']:^10.4f} | {agg['recall_macro']['std']:^10.4f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY (Publication-Ready)")
    print("=" * 80)
    print(f"\n  Accuracy: {agg['accuracy']['mean']*100:.2f}% ± {agg['accuracy']['std']*100:.2f}%")
    print(f"  F1 Score: {agg['f1_macro']['mean']*100:.2f}% ± {agg['f1_macro']['std']*100:.2f}%")
    print("\n" + "=" * 80)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run K-fold cross-validation for bearing fault diagnosis CNN',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to HDF5 dataset file')
    parser.add_argument('--model', type=str, default='cnn1d',
                       choices=['cnn1d', 'attention', 'attention-lite', 'multiscale', 'dilated'],
                       help='CNN architecture')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of folds')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Epochs per fold')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (1 epoch, 3 folds) for testing')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.epochs = 1
        args.k = 3
        logger.info("Quick mode enabled: 1 epoch, 3 folds")
    
    # Check data path
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        sys.exit(1)
    
    # Run cross-validation
    trainer = CrossValidationTrainer(
        model_class=args.model,
        k_folds=args.k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )
    
    results = trainer.run_cv(str(data_path))
    
    # Print results
    print_results(results)
    
    # Save to JSON
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / 'results' / f'cv_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
