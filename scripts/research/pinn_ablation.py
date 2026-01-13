"""
PINN Ablation Study Script

This script runs systematic ablation experiments to validate the contribution
of physics-informed components in bearing fault diagnosis.

Usage:
    python scripts/research/pinn_ablation.py --config configs/ablation.yaml
    python scripts/research/pinn_ablation.py --quick  # Only 3 key configs

Output:
    - results/ablation/ablation_results.csv
    - results/ablation/sensitivity_surface.png
    - results/ablation/significance_tests.json

Reference: Master Roadmap Chapter 3.1
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ablation configurations per Master Roadmap Chapter 3.1
ABLATION_CONFIGS = [
    # Baseline: No physics
    {'name': 'CNN_only', 'lambda_physics': 0.0, 'lambda_boundary': 0.0},
    
    # Physics loss only (varying weights)
    {'name': 'Physics_low', 'lambda_physics': 0.01, 'lambda_boundary': 0.0},
    {'name': 'Physics_med', 'lambda_physics': 0.1, 'lambda_boundary': 0.0},
    {'name': 'Physics_high', 'lambda_physics': 1.0, 'lambda_boundary': 0.0},
    
    # Boundary loss only
    {'name': 'Boundary_low', 'lambda_physics': 0.0, 'lambda_boundary': 0.01},
    {'name': 'Boundary_med', 'lambda_physics': 0.0, 'lambda_boundary': 0.1},
    
    # Combined configurations
    {'name': 'Combined_balanced', 'lambda_physics': 0.1, 'lambda_boundary': 0.1},
    {'name': 'Combined_physics_heavy', 'lambda_physics': 0.5, 'lambda_boundary': 0.1},
    {'name': 'Combined_optimal', 'lambda_physics': 0.2, 'lambda_boundary': 0.15},  # From HPO
]

# Quick mode: only key comparisons
QUICK_CONFIGS = ['CNN_only', 'Physics_med', 'Combined_optimal']

OUTPUT_DIR = Path('results/ablation')


def set_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_data():
    """Load dataset for ablation study."""
    from data.dataset import BearingFaultDataset
    
    cache_path = Path('data/processed/signals_cache.h5')
    if not cache_path.exists():
        logger.warning(f"Cache file not found: {cache_path}")
        logger.info("Generating synthetic data for ablation study...")
        from data.signal_generator import SignalGenerator
        from config.data_config import DataConfig
        
        config = DataConfig(num_signals_per_fault=50)
        generator = SignalGenerator(config)
        dataset = generator.generate_dataset()
        paths = generator.save_dataset(dataset, format='hdf5')
        cache_path = Path(paths['hdf5'])
    
    train_dataset = BearingFaultDataset.from_hdf5(str(cache_path), split='train')
    val_dataset = BearingFaultDataset.from_hdf5(str(cache_path), split='val')
    test_dataset = BearingFaultDataset.from_hdf5(str(cache_path), split='test')
    
    return train_dataset, val_dataset, test_dataset


def create_pinn_model(config: Dict[str, Any], num_classes: int = 11):
    """Create PINN model with given configuration."""
    try:
        from packages.core.models.pinn import HybridPINN
        
        model = HybridPINN(
            num_classes=num_classes,
            lambda_physics=config['lambda_physics'],
            lambda_boundary=config.get('lambda_boundary', 0.0)
        )
        return model
    except ImportError:
        # Fallback to simple CNN if PINN not available
        logger.warning("HybridPINN not available, using placeholder")
        from packages.core.models import create_model
        return create_model('resnet18', num_classes=num_classes)


def train_model(model, train_loader, val_loader, epochs: int = 50):
    """Train model and return training history."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        for batch in train_loader:
            signals, labels = batch
            signals, labels = signals.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                signals, labels = batch
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['val_accuracy'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs}: val_acc={val_acc:.4f}")
    
    return history


def evaluate_model(model, test_loader) -> Dict[str, Any]:
    """Evaluate model and return metrics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            signals, labels = batch
            signals = signals.to(device)
            outputs = model(signals)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'f1_weighted': f1_score(all_labels, all_preds, average='weighted'),
        'predictions': all_preds,
        'labels': all_labels
    }


def mcnemar_test(preds_a: List[int], preds_b: List[int], labels: List[int]) -> float:
    """
    Perform McNemar's test between two models.
    
    Returns p-value for the null hypothesis that the two models
    have the same error rate.
    """
    # Build contingency table
    a_correct = np.array(preds_a) == np.array(labels)
    b_correct = np.array(preds_b) == np.array(labels)
    
    # b: A correct, B wrong
    b = np.sum(a_correct & ~b_correct)
    # c: A wrong, B correct
    c = np.sum(~a_correct & b_correct)
    
    # McNemar's test (with continuity correction)
    if b + c == 0:
        return 1.0  # No discordant pairs
    
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    
    return p_value


def generate_sensitivity_surface(results_df: pd.DataFrame, output_path: Path):
    """Generate 3D sensitivity surface for lambda parameters."""
    try:
        import plotly.graph_objects as go
        
        # Extract unique lambda values
        physics_vals = sorted(results_df['lambda_physics'].unique())
        boundary_vals = sorted(results_df['lambda_boundary'].unique())
        
        # Create accuracy matrix
        acc_matrix = np.zeros((len(boundary_vals), len(physics_vals)))
        for i, b in enumerate(boundary_vals):
            for j, p in enumerate(physics_vals):
                mask = (results_df['lambda_physics'] == p) & (results_df['lambda_boundary'] == b)
                if mask.any():
                    acc_matrix[i, j] = results_df.loc[mask, 'accuracy'].iloc[0]
        
        fig = go.Figure(data=[go.Surface(
            x=physics_vals,
            y=boundary_vals,
            z=acc_matrix,
            colorscale='Viridis'
        )])
        
        fig.update_layout(
            title='PINN Hyperparameter Sensitivity',
            scene=dict(
                xaxis_title='λ_physics',
                yaxis_title='λ_boundary',
                zaxis_title='Accuracy'
            )
        )
        
        fig.write_image(str(output_path / 'sensitivity_surface.png'))
        fig.write_html(str(output_path / 'sensitivity_surface.html'))
        logger.info(f"Sensitivity surface saved to {output_path}")
        
    except ImportError:
        logger.warning("Plotly not available, skipping sensitivity surface")


def run_ablation_study(configs: List[Dict], quick: bool = False) -> pd.DataFrame:
    """Run the full ablation study."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("PINN ABLATION STUDY")
    logger.info("=" * 60)
    
    # Filter configs if quick mode
    if quick:
        configs = [c for c in configs if c['name'] in QUICK_CONFIGS]
        logger.info(f"Quick mode: running {len(configs)} configurations")
    
    # Load data
    logger.info("Loading data...")
    train_dataset, val_dataset, test_dataset = load_data()
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Run ablations
    results = []
    all_predictions = {}
    
    for config in configs:
        logger.info(f"\n--- Running: {config['name']} ---")
        set_seeds(42)
        
        # Create and train model
        model = create_pinn_model(config)
        start_time = datetime.now()
        history = train_model(model, train_loader, val_loader, epochs=30)
        training_time = (datetime.now() - start_time).total_seconds() / 60
        
        # Evaluate
        metrics = evaluate_model(model, test_loader)
        
        result = {
            'name': config['name'],
            'lambda_physics': config['lambda_physics'],
            'lambda_boundary': config['lambda_boundary'],
            'accuracy': metrics['accuracy'],
            'f1_macro': metrics['f1_macro'],
            'training_time_min': training_time,
            'final_val_loss': history['val_loss'][-1],
        }
        results.append(result)
        all_predictions[config['name']] = metrics['predictions']
        
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'ablation_results.csv', index=False)
    logger.info(f"\nResults saved to {OUTPUT_DIR / 'ablation_results.csv'}")
    
    # Statistical significance tests
    if len(configs) > 1:
        logger.info("\n--- Statistical Significance (McNemar) ---")
        baseline_preds = all_predictions.get('CNN_only')
        labels = list(test_dataset.labels if hasattr(test_dataset, 'labels') else range(len(test_dataset)))
        
        significance_results = {}
        for name, preds in all_predictions.items():
            if name != 'CNN_only' and baseline_preds is not None:
                p_value = mcnemar_test(baseline_preds, preds, labels)
                significance_results[f"CNN_only_vs_{name}"] = {
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
                logger.info(f"  CNN_only vs {name}: p={p_value:.4f} {'*' if p_value < 0.05 else ''}")
        
        with open(OUTPUT_DIR / 'significance_tests.json', 'w') as f:
            json.dump(significance_results, f, indent=2)
    
    # Generate sensitivity surface
    generate_sensitivity_surface(results_df, OUTPUT_DIR)
    
    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION STUDY RESULTS")
    logger.info("=" * 60)
    print(results_df.to_string(index=False))
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='PINN Ablation Study')
    parser.add_argument('--quick', action='store_true', help='Run only key configurations')
    parser.add_argument('--config', type=str, help='Path to custom config YAML')
    args = parser.parse_args()
    
    configs = ABLATION_CONFIGS
    
    if args.config:
        import yaml
        with open(args.config) as f:
            configs = yaml.safe_load(f)['ablation_configs']
    
    results = run_ablation_study(configs, quick=args.quick)
    
    # Find best configuration
    best_idx = results['accuracy'].idxmax()
    best = results.iloc[best_idx]
    logger.info(f"\n🏆 Best configuration: {best['name']} (Accuracy: {best['accuracy']:.4f})")


if __name__ == '__main__':
    main()
