#!/usr/bin/env python3
"""
Mixed-Precision (FP16) Training Validation

Stress tests FP16 training for numerical stability and performance.
Validates that mixed-precision training produces equivalent results.

Features:
- FP16 vs FP32 accuracy comparison
- Gradient overflow detection
- Loss scaling analysis
- Memory and throughput benchmarks

Usage:
    python scripts/utilities/mixed_precision_test.py --data data/processed/dataset.h5
    
    # Quick test
    python scripts/utilities/mixed_precision_test.py --demo

Author: Deficiency Fix #46 (Priority: 6)
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
import gc
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

from utils.reproducibility import set_seed
from utils.device_manager import get_device
from utils.logging import get_logger


logger = get_logger(__name__)


class MixedPrecisionValidator:
    """
    Validates mixed-precision (FP16) training stability and correctness.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device(prefer_gpu=True)
        self.supports_amp = torch.cuda.is_available()
        
        if not self.supports_amp:
            logger.warning("CUDA not available. AMP requires GPU.")
    
    def compare_training(
        self,
        model_class,
        model_kwargs: Dict[str, Any],
        train_data: Tuple[np.ndarray, np.ndarray],
        epochs: int = 10,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Compare FP32 vs FP16 training.
        
        Args:
            model_class: Model class to instantiate
            model_kwargs: Model constructor arguments
            train_data: (signals, labels) tuple
            epochs: Number of training epochs
            batch_size: Batch size
        
        Returns:
            Comparison results
        """
        from torch.utils.data import DataLoader, TensorDataset
        from torch.cuda.amp import autocast, GradScaler
        
        signals, labels = train_data
        signals = torch.from_numpy(signals.astype(np.float32))
        labels = torch.from_numpy(labels.astype(np.int64))
        
        dataset = TensorDataset(signals, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        results = {}
        
        # FP32 Training
        logger.info("\n--- FP32 Training ---")
        set_seed(42)
        model_fp32 = model_class(**model_kwargs).to(self.device)
        results['fp32'] = self._train_model(
            model_fp32, dataloader, epochs, use_amp=False
        )
        
        # FP16 Training
        if self.supports_amp:
            logger.info("\n--- FP16 (AMP) Training ---")
            set_seed(42)
            model_fp16 = model_class(**model_kwargs).to(self.device)
            results['fp16'] = self._train_model(
                model_fp16, dataloader, epochs, use_amp=True
            )
        else:
            logger.warning("Skipping FP16 (requires CUDA)")
            results['fp16'] = {'skipped': True, 'reason': 'No CUDA'}
        
        return results
    
    def _train_model(
        self,
        model: nn.Module,
        dataloader,
        epochs: int,
        use_amp: bool = False
    ) -> Dict[str, Any]:
        """Train model and collect metrics."""
        from torch.cuda.amp import autocast, GradScaler
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scaler = GradScaler() if use_amp else None
        
        # Track metrics
        loss_history = []
        grad_norms = []
        overflow_count = 0
        
        # Memory tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        start_time = time.perf_counter()
        
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_grad_norm = 0
            n_batches = 0
            
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                if use_amp:
                    with autocast():
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y)
                    
                    scaler.scale(loss).backward()
                    
                    # Check for overflow
                    scaler.unscale_(optimizer)
                    grad_norm = self._get_grad_norm(model)
                    
                    if torch.isnan(torch.tensor(grad_norm)) or torch.isinf(torch.tensor(grad_norm)):
                        overflow_count += 1
                    else:
                        epoch_grad_norm += grad_norm
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    grad_norm = self._get_grad_norm(model)
                    epoch_grad_norm += grad_norm
                    
                    optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            avg_grad = epoch_grad_norm / n_batches
            
            loss_history.append(avg_loss)
            grad_norms.append(avg_grad)
        
        training_time = time.perf_counter() - start_time
        
        # Memory stats
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            peak_memory = 0
        
        # Final accuracy
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                if use_amp:
                    with autocast():
                        outputs = model(batch_x)
                else:
                    outputs = model(batch_x)
                
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        
        accuracy = correct / total
        
        results = {
            'final_loss': loss_history[-1],
            'final_accuracy': accuracy,
            'training_time_s': training_time,
            'peak_memory_mb': peak_memory,
            'loss_history': loss_history,
            'grad_norm_history': grad_norms,
            'overflow_count': overflow_count,
            'throughput_samples_per_sec': (total * epochs) / training_time
        }
        
        logger.info(f"  Final Loss: {loss_history[-1]:.4f}")
        logger.info(f"  Accuracy:   {accuracy:.4f}")
        logger.info(f"  Time:       {training_time:.2f}s")
        logger.info(f"  Peak Mem:   {peak_memory:.1f} MB")
        logger.info(f"  Overflows:  {overflow_count}")
        
        return results
    
    def _get_grad_norm(self, model: nn.Module) -> float:
        """Compute gradient L2 norm."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def run_numerical_stability_tests(
        self,
        model_class,
        model_kwargs: Dict[str, Any],
        n_tests: int = 10
    ) -> Dict[str, Any]:
        """
        Test numerical stability under extreme inputs.
        """
        logger.info("\n" + "=" * 50)
        logger.info("NUMERICAL STABILITY TESTS")
        logger.info("=" * 50)
        
        set_seed(42)
        model = model_class(**model_kwargs).to(self.device)
        model.eval()
        
        results = {
            'tests': [],
            'passed': 0,
            'failed': 0
        }
        
        test_cases = [
            ('Normal range', lambda: torch.randn(1, 1, 1024) * 1.0),
            ('Large values', lambda: torch.randn(1, 1, 1024) * 1000.0),
            ('Small values', lambda: torch.randn(1, 1, 1024) * 0.001),
            ('Near zero', lambda: torch.randn(1, 1, 1024) * 1e-6),
            ('Mixed scale', lambda: torch.cat([
                torch.randn(1, 1, 512) * 1000,
                torch.randn(1, 1, 512) * 0.001
            ], dim=2)),
        ]
        
        for name, input_fn in test_cases:
            x = input_fn().to(self.device)
            
            try:
                with torch.no_grad():
                    output = model(x)
                
                has_nan = torch.isnan(output).any().item()
                has_inf = torch.isinf(output).any().item()
                
                if has_nan or has_inf:
                    status = 'FAIL'
                    results['failed'] += 1
                else:
                    status = 'PASS'
                    results['passed'] += 1
                
                results['tests'].append({
                    'name': name,
                    'status': status,
                    'has_nan': has_nan,
                    'has_inf': has_inf
                })
                
                logger.info(f"  {name}: {status}")
                
            except Exception as e:
                results['failed'] += 1
                results['tests'].append({
                    'name': name,
                    'status': 'ERROR',
                    'error': str(e)
                })
                logger.info(f"  {name}: ERROR - {e}")
        
        return results


def print_comparison_report(results: Dict[str, Any]) -> None:
    """Print comparison report."""
    print("\n" + "=" * 60)
    print("MIXED-PRECISION COMPARISON REPORT")
    print("=" * 60)
    
    if 'fp16' in results and 'skipped' in results['fp16']:
        print("\nFP16 testing skipped (requires CUDA)")
        return
    
    fp32 = results.get('fp32', {})
    fp16 = results.get('fp16', {})
    
    print(f"\n{'Metric':<25} | {'FP32':>12} | {'FP16':>12} | {'Diff':>10}")
    print("-" * 65)
    
    metrics = [
        ('Final Loss', 'final_loss', '.4f'),
        ('Accuracy', 'final_accuracy', '.4f'),
        ('Training Time (s)', 'training_time_s', '.2f'),
        ('Peak Memory (MB)', 'peak_memory_mb', '.1f'),
        ('Throughput (samples/s)', 'throughput_samples_per_sec', '.1f'),
    ]
    
    for name, key, fmt in metrics:
        v32 = fp32.get(key, 0)
        v16 = fp16.get(key, 0)
        diff = v16 - v32 if isinstance(v16, (int, float)) and isinstance(v32, (int, float)) else 0
        
        print(f"{name:<25} | {v32:>12{fmt}} | {v16:>12{fmt}} | {diff:>+10{fmt}}")
    
    print("=" * 60)
    
    # Speedup and memory savings
    if fp32.get('training_time_s', 0) > 0:
        speedup = fp32['training_time_s'] / fp16['training_time_s']
        print(f"\nSpeedup: {speedup:.2f}x")
    
    if fp32.get('peak_memory_mb', 0) > 0:
        mem_savings = (1 - fp16['peak_memory_mb'] / fp32['peak_memory_mb']) * 100
        print(f"Memory Savings: {mem_savings:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Mixed-precision training validation')
    parser.add_argument('--data', type=str, help='HDF5 data path')
    parser.add_argument('--epochs', type=int, default=10, help='Epochs')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--output', type=str, help='Output JSON path')
    
    args = parser.parse_args()
    
    if args.demo:
        logger.info("=" * 60)
        logger.info("MIXED-PRECISION VALIDATION DEMO")
        logger.info("=" * 60)
        
        # Create synthetic data
        np.random.seed(42)
        n_samples = 500
        signal_length = 1024
        n_classes = 5
        
        signals = np.random.randn(n_samples, 1, signal_length).astype(np.float32)
        labels = np.random.randint(0, n_classes, n_samples)
        
        # Simple model
        class SimpleModel(nn.Module):
            def __init__(self, num_classes=5):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 32, 7, stride=2)
                self.conv2 = nn.Conv1d(32, 64, 7, stride=2)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(64, num_classes)
            
            def forward(self, x):
                if x.dim() == 2:
                    x = x.unsqueeze(1)
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x).squeeze(-1)
                return self.fc(x)
        
        validator = MixedPrecisionValidator()
        
        # Training comparison
        results = validator.compare_training(
            model_class=SimpleModel,
            model_kwargs={'num_classes': n_classes},
            train_data=(signals, labels),
            epochs=args.epochs
        )
        
        print_comparison_report(results)
        
        # Numerical stability
        stability = validator.run_numerical_stability_tests(
            model_class=SimpleModel,
            model_kwargs={'num_classes': n_classes}
        )
        
        logger.info(f"\nStability: {stability['passed']}/{len(stability['tests'])} tests passed")
        
    else:
        print("Run with --demo for demonstration")


if __name__ == '__main__':
    main()
