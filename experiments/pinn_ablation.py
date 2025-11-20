"""
PINN Ablation Study

Systematic ablation study to quantify the impact of each physics component:
1. Baseline CNN (no physics)
2. + Physics Loss (frequency consistency)
3. + Physics Features (Sommerfeld, Reynolds, etc.)
4. + Knowledge Graph (fault relationships)
5. + Multi-task Learning (auxiliary tasks)

Goals:
- Understand which physics components contribute most to performance
- Measure sample efficiency gains from each component
- Validate that physics constraints improve generalization
- Provide evidence for design decisions

Expected findings:
- Physics loss: +1-2% accuracy
- Physics features: +2-3% accuracy
- Knowledge graph: +0.5-1% on mixed faults
- Multi-task learning: +1-2% accuracy
- Combined (full PINN): +4-6% total improvement
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.resnet.resnet_1d import ResNet1D
from models.pinn.physics_constrained_cnn import PhysicsConstrainedCNN
from models.pinn.hybrid_pinn import HybridPINN
from models.pinn.knowledge_graph_pinn import KnowledgeGraphPINN
from models.pinn.multitask_pinn import MultitaskPINN
from evaluation.pinn_evaluator import PINNEvaluator
from training.trainer import Trainer
from training.pinn_trainer import PINNTrainer


class PINNAblationStudy:
    """
    Conducts comprehensive ablation study on PINN components.
    """

    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        device: str = 'cuda',
        num_epochs: int = 30,
        batch_size: int = 32,
        learning_rate: float = 1e-4
    ):
        """
        Initialize ablation study.

        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            device: Device to train on
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.results = {}

    def run_ablation(
        self,
        configurations: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Run ablation study on specified configurations.

        Args:
            configurations: List of configurations to test
                          If None, tests all configurations

        Returns:
            Dictionary mapping configuration name to results
        """
        if configurations is None:
            configurations = [
                'baseline_cnn',
                'physics_loss',
                'hybrid_pinn',
                'knowledge_graph',
                'multitask_pinn'
            ]

        print("=" * 60)
        print("PINN ABLATION STUDY")
        print("=" * 60)
        print(f"Configurations: {configurations}")
        print(f"Training epochs: {self.num_epochs}")
        print(f"Device: {self.device}")
        print("=" * 60)

        for config_name in configurations:
            print(f"\n{'='*60}")
            print(f"Testing Configuration: {config_name.upper()}")
            print(f"{'='*60}")

            try:
                results = self._train_and_evaluate_config(config_name)
                self.results[config_name] = results

                print(f"\nResults for {config_name}:")
                print(f"  Accuracy: {results['accuracy']:.2f}%")
                print(f"  Training time: {results['training_time']:.2f}s")

            except Exception as e:
                print(f"Error with configuration {config_name}: {e}")
                self.results[config_name] = {'error': str(e)}

        # Print comparison
        self._print_comparison()

        return self.results

    def _train_and_evaluate_config(self, config_name: str) -> Dict[str, float]:
        """
        Train and evaluate a specific configuration.

        Args:
            config_name: Configuration identifier

        Returns:
            Results dictionary
        """
        import time

        # Create model based on configuration
        model = self._create_model(config_name)

        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        # Create optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Create trainer
        if 'physics' in config_name or 'pinn' in config_name or 'graph' in config_name:
            trainer = PINNTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
                lambda_physics=0.5 if config_name != 'baseline_cnn' else 0.0,
                adaptive_lambda=True
            )
        else:
            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device
            )

        # Train
        start_time = time.time()
        trainer.train(num_epochs=self.num_epochs)
        training_time = time.time() - start_time

        # Evaluate
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

        evaluator = PINNEvaluator(model, device=self.device)
        test_results = evaluator.evaluate(test_loader)

        results = {
            'accuracy': test_results['accuracy'],
            'training_time': training_time,
            'num_params': sum(p.numel() for p in model.parameters()),
        }

        return results

    def _create_model(self, config_name: str) -> nn.Module:
        """
        Create model based on configuration name.

        Args:
            config_name: Configuration identifier

        Returns:
            PyTorch model
        """
        if config_name == 'baseline_cnn':
            # Standard ResNet without physics
            model = ResNet1D(
                num_classes=11,
                input_channels=1,
                layers=[2, 2, 2, 2],
                dropout=0.3,
                input_length=102400
            )

        elif config_name == 'physics_loss':
            # ResNet + physics-based loss constraints
            model = PhysicsConstrainedCNN(
                num_classes=11,
                backbone='resnet18',
                dropout=0.3,
                sample_rate=51200
            )

        elif config_name == 'hybrid_pinn':
            # Hybrid PINN with physics features
            model = HybridPINN(
                num_classes=11,
                backbone='resnet18',
                physics_feature_dim=64,
                fusion_dim=256,
                dropout=0.3
            )

        elif config_name == 'knowledge_graph':
            # Knowledge Graph PINN
            model = KnowledgeGraphPINN(
                num_classes=11,
                backbone='resnet18',
                node_feature_dim=64,
                gcn_hidden_dim=128,
                num_gcn_layers=2,
                dropout=0.3
            )

        elif config_name == 'multitask_pinn':
            # Multi-task PINN
            model = MultitaskPINN(
                num_fault_classes=11,
                num_severity_levels=4,
                backbone='resnet18',
                dropout=0.3
            )

        else:
            raise ValueError(f"Unknown configuration: {config_name}")

        return model

    def _print_comparison(self):
        """Print comparison table of all configurations."""
        print("\n" + "=" * 80)
        print("ABLATION STUDY RESULTS")
        print("=" * 80)

        if not self.results:
            print("No results to display")
            return

        # Find baseline accuracy
        baseline_acc = self.results.get('baseline_cnn', {}).get('accuracy', 0)

        print(f"\n{'Configuration':<25} {'Accuracy':<12} {'Δ vs Baseline':<15} {'Time (s)':<12} {'Params (M)'}")
        print("-" * 80)

        for config_name, results in self.results.items():
            if 'error' in results:
                print(f"{config_name:<25} ERROR: {results['error']}")
                continue

            acc = results['accuracy']
            delta = acc - baseline_acc
            time_s = results['training_time']
            params_m = results['num_params'] / 1e6

            delta_str = f"{delta:+.2f}%" if config_name != 'baseline_cnn' else "-"

            print(f"{config_name:<25} {acc:>6.2f}%      {delta_str:<15} {time_s:>8.1f}      {params_m:>6.2f}")

        print("=" * 80)

    def test_sample_efficiency(
        self,
        sample_sizes: List[int] = None,
        num_trials: int = 3
    ) -> Dict[str, Dict[int, Dict[str, float]]]:
        """
        Test sample efficiency: accuracy vs training set size.

        Args:
            sample_sizes: List of training set sizes
            num_trials: Number of random trials per size

        Returns:
            Results for each configuration and sample size
        """
        if sample_sizes is None:
            sample_sizes = [50, 100, 200, 500, 1000]

        configurations = ['baseline_cnn', 'hybrid_pinn']
        efficiency_results = {}

        print("\n" + "=" * 60)
        print("SAMPLE EFFICIENCY TEST")
        print("=" * 60)

        for config_name in configurations:
            print(f"\nTesting {config_name}:")
            config_results = {}

            for size in sample_sizes:
                print(f"\n  Sample size: {size}")
                size_accuracies = []

                for trial in range(num_trials):
                    # Create subset
                    indices = torch.randperm(len(self.train_dataset))[:size]
                    subset = Subset(self.train_dataset, indices)

                    # Create temporary datasets
                    original_train = self.train_dataset
                    self.train_dataset = subset

                    # Train and evaluate
                    try:
                        results = self._train_and_evaluate_config(config_name)
                        size_accuracies.append(results['accuracy'])
                        print(f"    Trial {trial+1}: {results['accuracy']:.2f}%")
                    except Exception as e:
                        print(f"    Trial {trial+1} failed: {e}")

                    # Restore original dataset
                    self.train_dataset = original_train

                if size_accuracies:
                    config_results[size] = {
                        'mean': np.mean(size_accuracies),
                        'std': np.std(size_accuracies),
                        'trials': size_accuracies
                    }
                    print(f"    Average: {config_results[size]['mean']:.2f}% ± {config_results[size]['std']:.2f}%")

            efficiency_results[config_name] = config_results

        self._print_sample_efficiency_results(efficiency_results)

        return efficiency_results

    def _print_sample_efficiency_results(self, results: Dict):
        """Print sample efficiency comparison."""
        print("\n" + "=" * 60)
        print("SAMPLE EFFICIENCY COMPARISON")
        print("=" * 60)

        # Get all sample sizes
        sample_sizes = sorted(list(results[list(results.keys())[0]].keys()))

        print(f"\n{'Sample Size':<15}", end='')
        for config in results.keys():
            print(f"{config:<20}", end='')
        print()
        print("-" * (15 + 20 * len(results)))

        for size in sample_sizes:
            print(f"{size:<15}", end='')
            for config in results.keys():
                mean = results[config][size]['mean']
                std = results[config][size]['std']
                print(f"{mean:>6.2f}% ± {std:>4.2f}%     ", end='')
            print()

        print("=" * 60)

    def save_results(self, save_path: str):
        """
        Save ablation study results to JSON.

        Args:
            save_path: Path to save results
        """
        # Convert numpy types to native Python types
        results_serializable = {}
        for config, result in self.results.items():
            results_serializable[config] = {
                k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in result.items()
            }

        with open(save_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"\nResults saved to {save_path}")


def run_pinn_ablation_study(
    train_dataset,
    val_dataset,
    test_dataset,
    device: str = 'cuda',
    save_path: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Convenience function to run complete PINN ablation study.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        device: Device
        save_path: Optional path to save results

    Returns:
        Ablation study results
    """
    study = PINNAblationStudy(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        device=device,
        num_epochs=30,
        batch_size=32,
        learning_rate=1e-4
    )

    # Run main ablation
    results = study.run_ablation()

    # Optionally test sample efficiency
    # efficiency_results = study.test_sample_efficiency()

    # Save results
    if save_path:
        study.save_results(save_path)

    return results


if __name__ == "__main__":
    # Test ablation study structure
    print("=" * 60)
    print("PINN Ablation Study - Structure Validation")
    print("=" * 60)

    from torch.utils.data import TensorDataset

    # Create dummy datasets
    n_train = 1000
    n_val = 200
    n_test = 200

    train_signals = torch.randn(n_train, 1, 102400)
    train_labels = torch.randint(0, 11, (n_train,))
    train_dataset = TensorDataset(train_signals, train_labels)

    val_signals = torch.randn(n_val, 1, 102400)
    val_labels = torch.randint(0, 11, (n_val,))
    val_dataset = TensorDataset(val_signals, val_labels)

    test_signals = torch.randn(n_test, 1, 102400)
    test_labels = torch.randint(0, 11, (n_test,))
    test_dataset = TensorDataset(test_signals, test_labels)

    print("\nAblation Study Structure:")
    print("  ✓ Train dataset: {} samples".format(len(train_dataset)))
    print("  ✓ Val dataset: {} samples".format(len(val_dataset)))
    print("  ✓ Test dataset: {} samples".format(len(test_dataset)))

    print("\nConfigurations to test:")
    configs = ['baseline_cnn', 'physics_loss', 'hybrid_pinn', 'knowledge_graph', 'multitask_pinn']
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config}")

    print("\n" + "=" * 60)
    print("Structure Validation Complete!")
    print("=" * 60)
    print("\nNote: Run actual ablation study with real data and training for full results")
