#!/usr/bin/env python3
"""
Transformer Baselines Benchmark Script

Runs benchmarks comparing CNN vs Transformer baselines:
- CNN1D (current SOTA in project)
- SignalTransformer (existing)
- PatchTST (new)
- TSMixer (new)

Usage:
    # Compare all models
    python scripts/research/transformer_benchmark.py --data data/processed/dataset.h5
    
    # Quick comparison
    python scripts/research/transformer_benchmark.py --data data/processed/dataset.h5 --quick

Author: Critical Deficiency Fix #9 (Priority: 80)
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
from typing import Dict, List, Any, Optional

from utils.reproducibility import set_seed
from utils.device_manager import get_device
from utils.logging import get_logger
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


logger = get_logger(__name__)


# Model registry
MODEL_CONFIGS = {
    'cnn1d': {
        'module': 'packages.core.models.cnn.cnn_1d',
        'class': 'CNN1D',
        'kwargs': {'input_channels': 1, 'dropout': 0.3, 'use_batch_norm': True}
    },
    'signal_transformer': {
        'module': 'packages.core.models.transformer.signal_transformer',
        'class': 'SignalTransformer',
        'kwargs': {'input_channels': 1, 'patch_size': 1024, 'd_model': 128, 'num_heads': 4, 'num_layers': 3}
    },
    'patchtst': {
        'module': 'packages.core.models.transformer.patchtst',
        'class': 'PatchTST',
        'kwargs': {'patch_size': 1024, 'd_model': 128, 'n_heads': 4, 'n_layers': 3}
    },
    'tsmixer': {
        'module': 'packages.core.models.transformer.tsmixer',
        'class': 'TSMixer',
        'kwargs': {'n_features': 1, 'n_blocks': 4, 'd_model': 256, 'stride': 100}
    }
}


def create_model(model_name: str, num_classes: int, input_length: int = 102400) -> torch.nn.Module:
    """Dynamically create model from registry."""
    import importlib
    
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_name]
    
    # Import module and class
    module = importlib.import_module(config['module'])
    model_class = getattr(module, config['class'])
    
    # Build kwargs
    kwargs = config['kwargs'].copy()
    kwargs['num_classes'] = num_classes
    
    if model_name in ['patchtst', 'tsmixer']:
        kwargs['input_length'] = input_length
    
    return model_class(**kwargs)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_evaluate(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    epochs: int = 30,
    lr: float = 0.001
) -> Dict[str, Any]:
    """Train model and return test metrics."""
    from training.cnn_optimizer import create_optimizer
    from training.cnn_losses import create_criterion
    from training.cnn_schedulers import create_cosine_scheduler
    
    model = model.to(device)
    num_classes = model.num_classes
    
    optimizer = create_optimizer('adamw', model.parameters(), lr=lr, weight_decay=0.0001)
    criterion = create_criterion('cross_entropy', num_classes=num_classes)
    scheduler = create_cosine_scheduler(optimizer, num_epochs=epochs)
    
    # Training loop
    best_val_acc = 0
    best_model_state = None
    train_history = []
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_signals, batch_labels in train_loader:
            batch_signals = batch_signals.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_signals)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_labels.size(0)
            train_correct += predicted.eq(batch_labels).sum().item()
        
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_signals, batch_labels in val_loader:
                batch_signals = batch_signals.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_signals)
                _, predicted = outputs.max(1)
                val_total += batch_labels.size(0)
                val_correct += predicted.eq(batch_labels).sum().item()
        
        val_acc = val_correct / val_total
        
        if scheduler:
            scheduler.step()
        
        # Track best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        train_history.append({
            'epoch': epoch + 1,
            'train_acc': train_acc,
            'val_acc': val_acc
        })
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}: Train {train_acc:.4f}, Val {val_acc:.4f}")
    
    # Load best model and evaluate on test
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_signals, batch_labels in test_loader:
            batch_signals = batch_signals.to(device)
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
        'recall_macro': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'best_val_acc': best_val_acc,
        'train_history': train_history
    }


def run_benchmark(
    hdf5_path: str,
    models: List[str],
    epochs: int = 30,
    batch_size: int = 32,
    seed: int = 42
) -> Dict[str, Any]:
    """Run benchmark across all models."""
    from data.cnn_dataset import RawSignalDataset
    from data.cnn_transforms import get_train_transforms, get_test_transforms
    from data.cnn_dataloader import create_cnn_dataloaders
    
    set_seed(seed)
    device = get_device(prefer_gpu=True)
    
    logger.info("=" * 60)
    logger.info("TRANSFORMER BASELINES BENCHMARK")
    logger.info("=" * 60)
    logger.info(f"Models: {models}")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {epochs}")
    
    # Load data
    logger.info("\nLoading data...")
    with h5py.File(hdf5_path, 'r') as f:
        train_signals = f['train']['signals'][:]
        train_labels = f['train']['labels'][:]
        val_signals = f['val']['signals'][:]
        val_labels = f['val']['labels'][:]
        test_signals = f['test']['signals'][:]
        test_labels = f['test']['labels'][:]
        num_classes = f.attrs.get('num_classes', 11)
        signal_length = train_signals.shape[1]
    
    logger.info(f"  Train: {len(train_signals)}, Val: {len(val_signals)}, Test: {len(test_signals)}")
    logger.info(f"  Signal length: {signal_length}")
    logger.info(f"  Classes: {num_classes}")
    
    # Create dataloaders
    train_ds = RawSignalDataset(train_signals, train_labels, get_train_transforms(True))
    val_ds = RawSignalDataset(val_signals, val_labels, get_test_transforms())
    test_ds = RawSignalDataset(test_signals, test_labels, get_test_transforms())
    
    loaders = create_cnn_dataloaders(train_ds, val_ds, test_ds, batch_size, num_workers=4)
    
    # Benchmark each model
    results = {}
    
    for model_name in models:
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'='*60}")
        
        try:
            # Create model
            model = create_model(model_name, num_classes, signal_length)
            num_params = count_parameters(model)
            logger.info(f"  Parameters: {num_params:,}")
            
            # Train and evaluate
            metrics = train_and_evaluate(
                model=model,
                train_loader=loaders['train'],
                val_loader=loaders['val'],
                test_loader=loaders['test'],
                device=device,
                epochs=epochs
            )
            
            results[model_name] = {
                'num_parameters': num_params,
                **metrics
            }
            
            logger.info(f"\n  Test Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  Test F1 Macro: {metrics['f1_macro']:.4f}")
            
        except Exception as e:
            logger.error(f"  Error: {e}")
            results[model_name] = {'error': str(e)}
    
    return {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'seed': seed,
            'models': models
        },
        'dataset': {
            'path': str(hdf5_path),
            'train_samples': len(train_signals),
            'val_samples': len(val_signals),
            'test_samples': len(test_signals),
            'num_classes': num_classes,
            'signal_length': signal_length
        },
        'results': results
    }


def print_summary(benchmark: Dict[str, Any]) -> None:
    """Print benchmark summary table."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Model':<20} | {'Params':>12} | {'Accuracy':>10} | {'F1 Macro':>10} | {'Best Val':>10}")
    print("-" * 75)
    
    for model_name, metrics in benchmark['results'].items():
        if 'error' in metrics:
            print(f"{model_name:<20} | {'ERROR':>12} | {'-':>10} | {'-':>10} | {'-':>10}")
        else:
            print(f"{model_name:<20} | {metrics['num_parameters']:>12,} | "
                  f"{metrics['accuracy']:>10.4f} | {metrics['f1_macro']:>10.4f} | "
                  f"{metrics['best_val_acc']:>10.4f}")
    
    print("=" * 80)
    
    # Best model
    valid_results = {k: v for k, v in benchmark['results'].items() if 'error' not in v}
    if valid_results:
        best_model = max(valid_results.items(), key=lambda x: x[1]['accuracy'])
        print(f"\nBest model: {best_model[0]} ({best_model[1]['accuracy']*100:.2f}%)")
    
    print("=" * 80)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Benchmark Transformer baselines for time-series classification',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data', type=str, required=True,
                       help='Path to HDF5 dataset file')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['cnn1d', 'patchtst', 'tsmixer'],
                       choices=list(MODEL_CONFIGS.keys()),
                       help='Models to benchmark')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Training epochs per model')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode (10 epochs)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.quick:
        args.epochs = 10
        logger.info("Quick mode: 10 epochs")
    
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Dataset not found: {data_path}")
        sys.exit(1)
    
    # Run benchmark
    benchmark = run_benchmark(
        hdf5_path=str(data_path),
        models=args.models,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Print summary
    print_summary(benchmark)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = project_root / 'results' / f'transformer_benchmark_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove train_history for JSON serialization
    for model_name in benchmark['results']:
        if 'train_history' in benchmark['results'][model_name]:
            del benchmark['results'][model_name]['train_history']
    
    with open(output_path, 'w') as f:
        json.dump(benchmark, f, indent=2)
    
    logger.info(f"\n✓ Results saved to: {output_path}")


if __name__ == '__main__':
    main()
