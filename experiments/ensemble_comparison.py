"""
Ensemble Comparison Experiment

Compare all ensemble methods:
- Best Single Model (baseline)
- Soft Voting
- Hard Voting
- Stacking
- Boosting
- Mixture of Experts
- Early Fusion
- Late Fusion

Output comprehensive benchmarking table with accuracy and diversity metrics.

Author: LSTM_PFD Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import json
import time
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('/home/user/LSTM_PFD')

from models.legacy_ensemble import VotingEnsemble, StackingEnsemble
from models.ensemble import (
    BoostingEnsemble, MixtureOfExperts,
    train_stacking, train_boosting
)
from models.fusion import EarlyFusion, LateFusion, create_late_fusion
from evaluation.ensemble_evaluator import EnsembleEvaluator


class EnsembleComparison:
    """
    Comprehensive comparison of all ensemble methods.

    Args:
        base_models: List of trained base models
        model_names: Names of base models
        num_classes: Number of classes (default: 11)
        device: Device to use (default: 'cuda')

    Example:
        >>> models = [cnn_model, resnet_model, transformer_model, pinn_model]
        >>> names = ['CNN', 'ResNet', 'Transformer', 'PINN']
        >>> comparison = EnsembleComparison(models, names)
        >>> results = comparison.run_comparison(train_loader, val_loader, test_loader)
    """
    def __init__(
        self,
        base_models: List[nn.Module],
        model_names: List[str],
        num_classes: int = NUM_CLASSES,
        device: str = 'cuda'
    ):
        self.base_models = base_models
        self.model_names = model_names
        self.num_classes = num_classes
        self.device = device

        self.evaluator = EnsembleEvaluator(num_classes=num_classes)

        self.results = {}

    def run_comparison(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        save_dir: Optional[Path] = None
    ) -> Dict:
        """
        Run complete ensemble comparison.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            save_dir: Directory to save results

        Returns:
            Dictionary with all results
        """
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*80)
        print("ENSEMBLE METHODS COMPARISON")
        print("="*80)

        # Evaluate individual models (baseline)
        print("\n[1/8] Evaluating Individual Models (Baseline)...")
        self._evaluate_individuals(test_loader)

        # Evaluate ensemble diversity
        print("\n[2/8] Computing Ensemble Diversity Metrics...")
        diversity = self.evaluator.evaluate_ensemble_diversity(
            self.base_models, test_loader, self.device
        )
        self.results['diversity'] = diversity

        # Soft Voting
        print("\n[3/8] Evaluating Soft Voting Ensemble...")
        self._evaluate_soft_voting(test_loader)

        # Hard Voting
        print("\n[4/8] Evaluating Hard Voting Ensemble...")
        self._evaluate_hard_voting(test_loader)

        # Stacking
        print("\n[5/8] Training and Evaluating Stacking Ensemble...")
        self._evaluate_stacking(train_loader, val_loader, test_loader)

        # Late Fusion
        print("\n[6/8] Evaluating Late Fusion (Weighted Average)...")
        self._evaluate_late_fusion(test_loader)

        # Mixture of Experts
        print("\n[7/8] Training and Evaluating Mixture of Experts...")
        self._evaluate_mixture_of_experts(train_loader, val_loader, test_loader)

        # Early Fusion (if feature extractors available)
        print("\n[8/8] Evaluating Early Fusion (if applicable)...")
        # Skip early fusion for now as it requires feature extraction setup

        # Generate comparison table
        print("\n" + "="*80)
        self._print_comparison_table()

        # Save results
        if save_dir is not None:
            self._save_results(save_dir)
            self._plot_comparison(save_dir)

        return self.results

    def _evaluate_individuals(self, test_loader: torch.utils.data.DataLoader):
        """Evaluate individual models."""
        individual_results = []

        for i, (model, name) in enumerate(zip(self.base_models, self.model_names)):
            print(f"  Evaluating {name}...")
            model.eval()
            model = model.to(self.device)

            all_predictions = []
            all_labels = []

            with torch.no_grad():
                for batch in test_loader:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)

                    logits = model(x)
                    _, predicted = logits.max(1)

                    all_predictions.append(predicted.cpu().numpy())
                    all_labels.append(y.cpu().numpy())

            predictions = np.concatenate(all_predictions)
            labels = np.concatenate(all_labels)

            from sklearn.metrics import accuracy_score, f1_score
            accuracy = accuracy_score(labels, predictions) * 100
            f1 = f1_score(labels, predictions, average='weighted', zero_division=0) * 100

            individual_results.append({
                'model_name': name,
                'accuracy': accuracy,
                'f1_score': f1
            })

            print(f"    {name}: Accuracy = {accuracy:.2f}%, F1 = {f1:.2f}%")

        self.results['individuals'] = individual_results

        # Find best individual
        best_individual = max(individual_results, key=lambda x: x['accuracy'])
        self.results['best_individual'] = best_individual

        print(f"\n  Best Individual Model: {best_individual['model_name']} "
              f"({best_individual['accuracy']:.2f}%)")

    def _evaluate_soft_voting(self, test_loader: torch.utils.data.DataLoader):
        """Evaluate soft voting ensemble."""
        ensemble = VotingEnsemble(
            models=self.base_models,
            voting_type='soft',
            num_classes=self.num_classes
        )
        ensemble = ensemble.to(self.device)

        results = self.evaluator.evaluate_ensemble(ensemble, test_loader, self.device)

        self.results['soft_voting'] = {
            'accuracy': results['accuracy'],
            'f1_score': results['f1_score']
        }

    def _evaluate_hard_voting(self, test_loader: torch.utils.data.DataLoader):
        """Evaluate hard voting ensemble."""
        ensemble = VotingEnsemble(
            models=self.base_models,
            voting_type='hard',
            num_classes=self.num_classes
        )
        ensemble = ensemble.to(self.device)

        results = self.evaluator.evaluate_ensemble(ensemble, test_loader, self.device)

        self.results['hard_voting'] = {
            'accuracy': results['accuracy'],
            'f1_score': results['f1_score']
        }

    def _evaluate_stacking(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader
    ):
        """Train and evaluate stacking ensemble."""
        try:
            stacking = train_stacking(
                base_models=self.base_models,
                meta_learner='mlp',
                train_loader=train_loader,
                val_loader=val_loader,
                num_classes=self.num_classes,
                num_epochs=10,
                lr=0.001,
                device=self.device,
                verbose=False
            )

            results = self.evaluator.evaluate_ensemble(stacking, test_loader, self.device)

            self.results['stacking'] = {
                'accuracy': results['accuracy'],
                'f1_score': results['f1_score']
            }

        except Exception as e:
            print(f"    Error training stacking ensemble: {e}")
            self.results['stacking'] = {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'error': str(e)
            }

    def _evaluate_late_fusion(self, test_loader: torch.utils.data.DataLoader):
        """Evaluate late fusion."""
        ensemble = create_late_fusion(
            models=self.base_models,
            fusion_method='weighted_average',
            num_classes=self.num_classes
        )
        ensemble = ensemble.to(self.device)

        results = self.evaluator.evaluate_ensemble(ensemble, test_loader, self.device)

        self.results['late_fusion'] = {
            'accuracy': results['accuracy'],
            'f1_score': results['f1_score']
        }

    def _evaluate_mixture_of_experts(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader
    ):
        """Train and evaluate mixture of experts."""
        try:
            moe = MixtureOfExperts(
                experts=self.base_models,
                num_classes=self.num_classes
            )
            moe = moe.to(self.device)

            # Train gating network
            optimizer = torch.optim.Adam(moe.gating_network.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            moe.train_gating_network(
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                num_epochs=10,
                device=self.device,
                verbose=False
            )

            results = self.evaluator.evaluate_ensemble(moe, test_loader, self.device)

            self.results['mixture_of_experts'] = {
                'accuracy': results['accuracy'],
                'f1_score': results['f1_score']
            }

        except Exception as e:
            print(f"    Error training mixture of experts: {e}")
            self.results['mixture_of_experts'] = {
                'accuracy': 0.0,
                'f1_score': 0.0,
                'error': str(e)
            }

    def _print_comparison_table(self):
        """Print comparison table."""
        print("\nCOMPREHENSIVE ENSEMBLE COMPARISON")
        print("="*80)

        # Prepare table data
        table_data = []

        # Best individual
        best_ind = self.results['best_individual']
        table_data.append([
            'Best Single Model',
            best_ind['model_name'],
            f"{best_ind['accuracy']:.2f}%",
            f"{best_ind['f1_score']:.2f}%",
            'N/A',
            '+0.00%'
        ])

        # Ensemble methods
        ensemble_methods = [
            ('soft_voting', 'Soft Voting'),
            ('hard_voting', 'Hard Voting'),
            ('stacking', 'Stacking'),
            ('late_fusion', 'Late Fusion'),
            ('mixture_of_experts', 'Mixture of Experts')
        ]

        baseline_acc = best_ind['accuracy']
        diversity_score = self.results['diversity']['mean_disagreement']

        for key, name in ensemble_methods:
            if key in self.results:
                result = self.results[key]
                if 'error' not in result:
                    improvement = result['accuracy'] - baseline_acc
                    table_data.append([
                        name,
                        f"{len(self.base_models)} models",
                        f"{result['accuracy']:.2f}%",
                        f"{result['f1_score']:.2f}%",
                        f"{diversity_score:.3f}",
                        f"{improvement:+.2f}%"
                    ])

        # Print table
        headers = ['Ensemble Method', 'Components', 'Test Accuracy', 'F1 Score',
                   'Diversity', 'Improvement']
        print(tabulate(table_data, headers=headers, tablefmt='grid'))

    def _save_results(self, save_dir: Path):
        """Save results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}

        for key, value in self.results.items():
            if isinstance(value, dict):
                results_serializable[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        results_serializable[key][k] = v.tolist()
                    else:
                        results_serializable[key][k] = v
            else:
                results_serializable[key] = value

        # Save to JSON
        with open(save_dir / 'ensemble_comparison.json', 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"\nResults saved to {save_dir / 'ensemble_comparison.json'}")

    def _plot_comparison(self, save_dir: Path):
        """Plot comparison charts."""
        # Accuracy comparison
        methods = []
        accuracies = []

        # Best individual
        best_ind = self.results['best_individual']
        methods.append(f"Best Single\n({best_ind['model_name']})")
        accuracies.append(best_ind['accuracy'])

        # Ensemble methods
        ensemble_methods = [
            ('soft_voting', 'Soft Voting'),
            ('hard_voting', 'Hard Voting'),
            ('stacking', 'Stacking'),
            ('late_fusion', 'Late Fusion'),
            ('mixture_of_experts', 'MoE')
        ]

        for key, name in ensemble_methods:
            if key in self.results and 'error' not in self.results[key]:
                methods.append(name)
                accuracies.append(self.results[key]['accuracy'])

        # Plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(methods)), accuracies, color='steelblue', alpha=0.7)

        # Highlight best method
        best_idx = np.argmax(accuracies)
        bars[best_idx].set_color('darkgreen')
        bars[best_idx].set_alpha(1.0)

        plt.xlabel('Method')
        plt.ylabel('Test Accuracy (%)')
        plt.title('Ensemble Methods Comparison')
        plt.xticks(range(len(methods)), methods, rotation=15, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.ylim(min(accuracies) - 1, max(accuracies) + 1)

        # Add value labels on bars
        for i, (method, acc) in enumerate(zip(methods, accuracies)):
            plt.text(i, acc + 0.1, f'{acc:.2f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig(save_dir / 'ensemble_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Comparison plot saved to {save_dir / 'ensemble_comparison.png'}")


def run_ensemble_comparison(
    base_models: List[nn.Module],
    model_names: List[str],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int = NUM_CLASSES,
    device: str = 'cuda',
    save_dir: Optional[Path] = None
) -> Dict:
    """
    Run comprehensive ensemble comparison.

    Args:
        base_models: List of trained base models
        model_names: Names of base models
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        num_classes: Number of classes
        device: Device to use
        save_dir: Directory to save results

    Returns:
        Dictionary with all results

    Example:
        >>> models = [cnn_model, resnet_model, transformer_model, pinn_model]
        >>> names = ['CNN-1D', 'ResNet-18', 'Transformer', 'PINN']
        >>> results = run_ensemble_comparison(
        ...     models, names, train_loader, val_loader, test_loader,
        ...     save_dir=Path('results/phase8')
        ... )
    """
    comparison = EnsembleComparison(base_models, model_names, num_classes, device)
    results = comparison.run_comparison(train_loader, val_loader, test_loader, save_dir)

    return results


if __name__ == "__main__":
    # Example usage
    print("Ensemble Comparison Experiment")
    print("To use this script, import and call run_ensemble_comparison()")
    print("\nExample:")
    print("  from experiments.ensemble_comparison import run_ensemble_comparison")
    print("  results = run_ensemble_comparison(models, names, train_loader, val_loader, test_loader)")
