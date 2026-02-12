"""
Contrastive Physics Pretraining Framework

This module implements a SimCLR-style contrastive learning approach where
physics similarity defines positive/negative pairs for bearing fault diagnosis.

Addresses Deficiency #26 (Priority 46): No Contrastive Physics Pretraining

Key Innovation:
- Signals with similar physics parameters (eccentricity, clearance, viscosity)
  are treated as positive pairs, even if from different fault types
- This encourages the encoder to learn physics-aware representations
- Fine-tuning on fault classification uses physics-informed features

Core classes and losses are now in packages/core/ — this script retains
the CLI, benchmarking, and visualization utilities.

Usage:
    # Pretraining
    python scripts/research/contrastive_physics.py pretrain --epochs 100
    
    # Fine-tuning
    python scripts/research/contrastive_physics.py finetune --checkpoint pretrained.pt
    
    # Full pipeline with benchmark
    python scripts/research/contrastive_physics.py --full-pipeline

Author: AI Research Team
Date: January 2026
"""

import argparse
import copy
import gc
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Error: PyTorch is required for contrastive pretraining")
    sys.exit(1)

try:
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import matplotlib.pyplot as plt
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import production modules (extracted from this script into packages/)
# ---------------------------------------------------------------------------
from packages.core.training.contrastive import (
    compute_physics_similarity,
    select_positive_negative_pairs,
    PhysicsInfoNCELoss,
    NTXentLoss,
    PhysicsContrastiveDataset,
    FineTuneDataset,
    ContrastivePretrainer,
    ContrastiveFineTuner,
)
from packages.core.models.contrastive import (
    SignalEncoder,
    ContrastiveClassifier,
)

OUTPUT_DIR = Path('results/contrastive_physics')


# ============================================================================
# Benchmark: Contrastive vs Supervised
# ============================================================================

def run_benchmark(
    signals: np.ndarray,
    labels: np.ndarray,
    physics_params: List[Dict[str, float]],
    num_seeds: int = 3,
    pretrain_epochs: int = 50,
    finetune_epochs: int = 30,
    device: str = None
) -> Dict[str, Any]:
    """
    Benchmark contrastive pretraining vs supervised-only baseline.
    
    Returns metrics for both approaches with confidence intervals.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(np.unique(labels))
    
    results = {
        'supervised': {'accuracy': [], 'f1': []},
        'contrastive_frozen': {'accuracy': [], 'f1': []},
        'contrastive_finetuned': {'accuracy': [], 'f1': []}
    }
    
    for seed in range(num_seeds):
        logger.info(f"\n=== Seed {seed + 1}/{num_seeds} ===")
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Split data
        indices = np.arange(len(signals))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.2, stratify=labels, random_state=seed
        )
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.15, stratify=labels[train_idx], random_state=seed
        )
        
        # Create datasets
        train_signals = signals[train_idx]
        train_labels = labels[train_idx]
        train_physics = [physics_params[i] for i in train_idx]
        
        val_signals = signals[val_idx]
        val_labels = labels[val_idx]
        
        test_signals = signals[test_idx]
        test_labels = labels[test_idx]
        
        # 1. Supervised baseline
        logger.info("Training supervised baseline...")
        encoder_sup = SignalEncoder()
        classifier_sup = ContrastiveClassifier(encoder_sup, num_classes).to(device)
        optimizer_sup = optim.Adam(classifier_sup.parameters(), lr=0.001)
        
        train_loader = DataLoader(
            FineTuneDataset(train_signals, train_labels),
            batch_size=32, shuffle=True
        )
        test_loader = DataLoader(
            FineTuneDataset(test_signals, test_labels),
            batch_size=32
        )
        
        for _ in range(finetune_epochs):
            classifier_sup.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                loss = F.cross_entropy(classifier_sup(x), y)
                optimizer_sup.zero_grad()
                loss.backward()
                optimizer_sup.step()
                
        classifier_sup.eval()
        all_preds, all_labels_list = [], []
        with torch.no_grad():
            for x, y in test_loader:
                preds = classifier_sup(x.to(device)).argmax(1).cpu()
                all_preds.extend(preds.numpy())
                all_labels_list.extend(y.numpy())
                
        sup_acc = accuracy_score(all_labels_list, all_preds)
        sup_f1 = f1_score(all_labels_list, all_preds, average='weighted')
        results['supervised']['accuracy'].append(sup_acc)
        results['supervised']['f1'].append(sup_f1)
        logger.info(f"Supervised: Acc={sup_acc:.4f}, F1={sup_f1:.4f}")
        
        # 2. Contrastive pretraining
        logger.info("Contrastive pretraining...")
        encoder_con = SignalEncoder()
        
        contrastive_dataset = PhysicsContrastiveDataset(
            train_signals, train_physics,
            similarity_threshold=0.75,
            num_negatives=3
        )
        contrastive_loader = DataLoader(contrastive_dataset, batch_size=32, shuffle=True)
        
        pretrainer = ContrastivePretrainer(encoder_con, device=device)
        pretrainer.pretrain(contrastive_loader, epochs=pretrain_epochs)
        
        # 2a. Linear probe (frozen encoder)
        logger.info("Linear probe (frozen encoder)...")
        finetuner_frozen = ContrastiveFineTuner(
            encoder_con, num_classes, freeze_encoder=True, device=device
        )
        val_loader = DataLoader(FineTuneDataset(val_signals, val_labels), batch_size=32)
        finetuner_frozen.finetune(train_loader, val_loader, epochs=finetune_epochs)
        
        frozen_acc, frozen_f1 = finetuner_frozen.evaluate(test_loader)
        results['contrastive_frozen']['accuracy'].append(frozen_acc)
        results['contrastive_frozen']['f1'].append(frozen_f1)
        logger.info(f"Frozen encoder: Acc={frozen_acc:.4f}, F1={frozen_f1:.4f}")
        
        # 2b. Full fine-tuning
        logger.info("Full fine-tuning...")
        finetuner_full = ContrastiveFineTuner(
            encoder_con, num_classes, freeze_encoder=False, device=device
        )
        finetuner_full.finetune(train_loader, val_loader, epochs=finetune_epochs)
        
        full_acc, full_f1 = finetuner_full.evaluate(test_loader)
        results['contrastive_finetuned']['accuracy'].append(full_acc)
        results['contrastive_finetuned']['f1'].append(full_f1)
        logger.info(f"Fine-tuned: Acc={full_acc:.4f}, F1={full_f1:.4f}")
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Compute statistics
    summary = {}
    for method, metrics in results.items():
        summary[method] = {
            'accuracy_mean': np.mean(metrics['accuracy']),
            'accuracy_std': np.std(metrics['accuracy']),
            'f1_mean': np.mean(metrics['f1']),
            'f1_std': np.std(metrics['f1'])
        }
        
    return {'results': results, 'summary': summary}


# ============================================================================
# Data Generation (for testing)
# ============================================================================

def generate_synthetic_data(
    n_samples: int = 1000,
    signal_length: int = 4096,
    n_classes: int = 5
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, float]]]:
    """Generate synthetic signals with physics parameters."""
    
    signals = []
    labels = []
    physics_params = []
    
    for i in range(n_samples):
        label = i % n_classes
        
        # Generate physics parameters
        eccentricity = np.random.uniform(0.1, 0.9)
        clearance = np.random.uniform(0.02, 0.3)
        viscosity = np.random.uniform(20, 300)
        load = np.random.uniform(500, 5000)
        speed = np.random.uniform(1000, 4000)
        
        physics = {
            'eccentricity': eccentricity,
            'clearance': clearance,
            'viscosity': viscosity,
            'load': load,
            'speed': speed
        }
        
        # Generate signal based on physics + class
        t = np.linspace(0, 1, signal_length)
        freq = 50 + label * 30 + speed / 100
        
        signal = np.sin(2 * np.pi * freq * t)
        signal += 0.3 * np.sin(2 * np.pi * freq * 2 * t * (1 + eccentricity))
        signal *= (1 + 0.2 * clearance)
        signal += np.random.randn(signal_length) * 0.1
        
        signals.append(signal.astype(np.float32))
        labels.append(label)
        physics_params.append(physics)
        
    return np.array(signals), np.array(labels), physics_params


# ============================================================================
# Visualization
# ============================================================================

def plot_training_curves(pretrain_history: Dict, finetune_history: Dict, 
                         output_path: Path):
    """Plot training curves for pretraining and fine-tuning."""
    if not HAS_PLOTTING:
        return
        
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Pretraining loss
    ax = axes[0]
    ax.plot(pretrain_history['loss'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Contrastive Loss')
    ax.set_title('Contrastive Pretraining')
    ax.grid(True, alpha=0.3)
    
    # Fine-tuning accuracy
    ax = axes[1]
    ax.plot(finetune_history['val_acc'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Fine-tuning')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training curves to {output_path}")


def plot_benchmark_results(summary: Dict[str, Dict], output_path: Path):
    """Plot benchmark comparison bar chart."""
    if not HAS_PLOTTING:
        return
        
    methods = list(summary.keys())
    means = [summary[m]['accuracy_mean'] for m in methods]
    stds = [summary[m]['accuracy_std'] for m in methods]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    bars = ax.bar(methods, means, yerr=stds, color=colors, 
                  capsize=5, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Contrastive Physics Pretraining vs Supervised Baseline')
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                f'{mean:.3f}±{std:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved benchmark results to {output_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Contrastive Physics Pretraining Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run contrastive pretraining
    python scripts/research/contrastive_physics.py pretrain --epochs 100
    
    # Fine-tune pretrained model
    python scripts/research/contrastive_physics.py finetune --checkpoint results/contrastive.pt
    
    # Run full benchmark comparison
    python scripts/research/contrastive_physics.py benchmark --seeds 5
    
    # Quick test
    python scripts/research/contrastive_physics.py --quick
        """
    )
    
    parser.add_argument('mode', nargs='?', default='benchmark',
                        choices=['pretrain', 'finetune', 'benchmark'],
                        help='Mode: pretrain, finetune, or benchmark')
    parser.add_argument('--epochs', '-e', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--seeds', '-s', type=int, default=3,
                        help='Number of seeds for benchmark (default: 3)')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Checkpoint path for fine-tuning')
    parser.add_argument('--output-dir', '-o', type=str, default='results/contrastive_physics',
                        help='Output directory')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with reduced epochs and samples')
    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Quick mode
    if args.quick:
        args.epochs = 10
        args.seeds = 2
        n_samples = 200
    else:
        n_samples = 1000
    
    # Generate synthetic data
    logger.info("Generating synthetic data with physics parameters...")
    signals, labels, physics_params = generate_synthetic_data(n_samples=n_samples)
    logger.info(f"Generated {len(signals)} samples, {len(np.unique(labels))} classes")
    
    if args.mode == 'pretrain':
        logger.info("Running contrastive pretraining...")
        
        dataset = PhysicsContrastiveDataset(signals, physics_params)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        encoder = SignalEncoder()
        pretrainer = ContrastivePretrainer(encoder, device=device)
        history = pretrainer.pretrain(
            loader, 
            epochs=args.epochs,
            save_path=output_dir / 'pretrained_encoder.pt'
        )
        
        # Save history
        with open(output_dir / 'pretrain_history.json', 'w') as f:
            json.dump(history, f, indent=2)
            
    elif args.mode == 'finetune':
        if not args.checkpoint:
            logger.error("--checkpoint required for fine-tuning")
            return
            
        logger.info(f"Fine-tuning from {args.checkpoint}")
        
        # Load pretrained encoder
        encoder = SignalEncoder()
        checkpoint = torch.load(args.checkpoint, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        
        # Split data
        train_idx, test_idx = train_test_split(
            np.arange(len(signals)), test_size=0.2, stratify=labels
        )
        
        train_loader = DataLoader(
            FineTuneDataset(signals[train_idx], labels[train_idx]),
            batch_size=32, shuffle=True
        )
        test_loader = DataLoader(
            FineTuneDataset(signals[test_idx], labels[test_idx]),
            batch_size=32
        )
        
        finetuner = ContrastiveFineTuner(
            encoder, num_classes=len(np.unique(labels)), device=device
        )
        history = finetuner.finetune(train_loader, test_loader, epochs=args.epochs)
        
        acc, f1 = finetuner.evaluate(test_loader)
        logger.info(f"Final Test Accuracy: {acc:.4f}, F1: {f1:.4f}")
        
    elif args.mode == 'benchmark':
        logger.info("Running benchmark: Contrastive vs Supervised")
        
        benchmark_results = run_benchmark(
            signals, labels, physics_params,
            num_seeds=args.seeds,
            pretrain_epochs=args.epochs,
            finetune_epochs=args.epochs // 2,
            device=device
        )
        
        # Print summary
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        for method, stats in benchmark_results['summary'].items():
            print(f"\n{method}:")
            print(f"  Accuracy: {stats['accuracy_mean']:.4f} ± {stats['accuracy_std']:.4f}")
            print(f"  F1 Score: {stats['f1_mean']:.4f} ± {stats['f1_std']:.4f}")
        print("="*60)
        
        # Save results
        with open(output_dir / 'benchmark_results.json', 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=float)
            
        # Plot
        plot_benchmark_results(
            benchmark_results['summary'],
            output_dir / 'benchmark_comparison.png'
        )
        
    logger.info(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
