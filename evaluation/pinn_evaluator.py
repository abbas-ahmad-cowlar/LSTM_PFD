"""
PINN Model Evaluator

Specialized evaluator for Physics-Informed Neural Networks that includes:
- Physics-aware metrics (frequency consistency, Sommerfeld consistency)
- Sample efficiency testing (accuracy vs training set size)
- Out-of-distribution generalization (unseen operating conditions)
- Comparison with baseline models

This evaluator validates that PINN models:
1. Achieve higher accuracy than baseline CNNs
2. Require less training data for same performance
3. Generalize better to unseen operating conditions
4. Make physically plausible predictions
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.evaluator import ModelEvaluator
from models.physics.bearing_dynamics import BearingDynamics
from models.physics.fault_signatures import FaultSignatureDatabase
from training.physics_loss_functions import FrequencyConsistencyLoss


class PINNEvaluator(ModelEvaluator):
    """
    Evaluator for Physics-Informed Neural Networks.

    Extends base evaluator with physics-aware metrics and sample efficiency testing.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        sample_rate: int = 51200
    ):
        """
        Initialize PINN evaluator.

        Args:
            model: PINN model to evaluate
            device: Device to run evaluation on
            sample_rate: Signal sampling rate for physics loss
        """
        super().__init__(model, device)

        self.sample_rate = sample_rate
        self.bearing_dynamics = BearingDynamics()
        self.signature_db = FaultSignatureDatabase()
        self.freq_loss_fn = FrequencyConsistencyLoss(sample_rate=sample_rate)

    @torch.no_grad()
    def evaluate_with_physics_metrics(
        self,
        dataloader: DataLoader,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Evaluate PINN model with physics-aware metrics.

        Args:
            dataloader: Test data loader
            metadata: Optional dict with 'rpm', 'load', etc.
            class_names: Optional list of class names

        Returns:
            Dictionary with standard + physics metrics
        """
        # Get standard metrics
        results = self.evaluate(dataloader, class_names)

        # Compute physics-specific metrics
        all_signals = []
        all_predictions = []
        all_targets = []

        for batch in tqdm(dataloader, desc="Computing physics metrics"):
            if len(batch) == 2:
                inputs, targets = batch
                batch_metadata = metadata
            else:
                inputs, targets, batch_metadata = batch

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            if batch_metadata is not None:
                if isinstance(batch_metadata, dict):
                    batch_metadata = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                     for k, v in batch_metadata.items()}
                outputs = self.model(inputs, batch_metadata)
            else:
                outputs = self.model(inputs)

            _, predicted = outputs.max(1)

            all_signals.append(inputs)
            all_predictions.append(predicted)
            all_targets.append(targets)

        # Concatenate
        all_signals = torch.cat(all_signals, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Compute frequency consistency
        freq_consistency = self._compute_frequency_consistency(
            all_signals, all_predictions, metadata
        )

        # Compute prediction plausibility
        plausibility_score = self._compute_prediction_plausibility(
            all_signals, all_predictions, metadata
        )

        # Add physics metrics to results
        results['physics_metrics'] = {
            'frequency_consistency': freq_consistency,
            'prediction_plausibility': plausibility_score
        }

        return results

    def _compute_frequency_consistency(
        self,
        signals: torch.Tensor,
        predictions: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]] = None
    ) -> float:
        """
        Compute what fraction of predictions are consistent with observed frequencies.

        Args:
            signals: Signal tensor [N, 1, T]
            predictions: Predicted classes [N]
            metadata: Operating conditions

        Returns:
            Consistency score (0-1)
        """
        # Sample subset for efficiency (checking all can be slow)
        n_samples = min(500, signals.shape[0])
        indices = torch.randperm(signals.shape[0])[:n_samples]

        signals_subset = signals[indices]
        predictions_subset = predictions[indices]

        # Use frequency consistency loss (lower = more consistent)
        rpm = 3600.0
        if metadata is not None and 'rpm' in metadata:
            rpm = metadata['rpm']
            if isinstance(rpm, torch.Tensor):
                rpm = rpm[indices]

        try:
            # Package metadata
            meta = {'rpm': rpm} if not isinstance(rpm, dict) else metadata

            # Compute frequency loss (lower is better)
            freq_loss = self.freq_loss_fn(signals_subset,
                                         torch.nn.functional.one_hot(predictions_subset, num_classes=11).float(),
                                         meta)

            # Convert to consistency score (1 = perfect, 0 = worst)
            # Normalize by typical loss value (~0.5)
            consistency = max(0.0, 1.0 - float(freq_loss) / 0.5)

        except Exception as e:
            print(f"Warning: Could not compute frequency consistency: {e}")
            consistency = 0.5  # Neutral score on error

        return consistency

    def _compute_prediction_plausibility(
        self,
        signals: torch.Tensor,
        predictions: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]] = None
    ) -> float:
        """
        Check if predictions are physically plausible given operating conditions.

        For example:
        - High-speed + low-load → unlikely to have severe bearing faults
        - Low Sommerfeld number → expect wear/lubrication faults

        Args:
            signals: Signal tensor
            predictions: Predicted classes
            metadata: Operating conditions

        Returns:
            Plausibility score (0-1)
        """
        if metadata is None:
            return 0.5  # Unknown without metadata

        try:
            rpm = metadata.get('rpm', torch.tensor(3600.0))
            load = metadata.get('load', torch.tensor(500.0))
            viscosity = metadata.get('viscosity', torch.tensor(0.03))

            # Compute Sommerfeld number
            S = self.bearing_dynamics.sommerfeld_number(load, rpm, viscosity, return_torch=True)
            S = S.to(self.device)

            # Check plausibility rules
            plausible_count = 0
            total_count = 0

            pred_np = predictions.cpu().numpy()

            # Rule 1: Low Sommerfeld → expect lubrication/wear faults (9, 10)
            low_S_mask = S < 0.1
            if low_S_mask.any():
                low_S_preds = pred_np[low_S_mask.cpu().numpy()]
                # Should predict lubrication (10) or wear (9) faults
                plausible_count += np.sum((low_S_preds == 9) | (low_S_preds == 10))
                total_count += len(low_S_preds)

            # Rule 2: High Sommerfeld → unlikely to have severe bearing damage (3, 4)
            high_S_mask = S > 1.0
            if high_S_mask.any():
                high_S_preds = pred_np[high_S_mask.cpu().numpy()]
                # Should NOT predict race defects frequently
                plausible_count += np.sum((high_S_preds != 3) & (high_S_preds != 4))
                total_count += len(high_S_preds)

            if total_count > 0:
                plausibility = plausible_count / total_count
            else:
                plausibility = 0.5

        except Exception as e:
            print(f"Warning: Could not compute plausibility: {e}")
            plausibility = 0.5

        return plausibility

    def test_sample_efficiency(
        self,
        train_dataset,
        val_dataset,
        train_function,
        sample_sizes: List[int] = None,
        num_trials: int = 3
    ) -> Dict[str, any]:
        """
        Test how model performance scales with training set size.

        This demonstrates PINN's improved sample efficiency compared to baseline models.

        Args:
            train_dataset: Full training dataset
            val_dataset: Validation dataset
            train_function: Function that trains model given subset
            sample_sizes: List of training set sizes to test
            num_trials: Number of random trials per size

        Returns:
            Dictionary mapping sample size to accuracy statistics
        """
        if sample_sizes is None:
            sample_sizes = [50, 100, 200, 500, 1000]

        results = {}

        for size in sample_sizes:
            print(f"\nTesting with {size} training samples:")
            size_accuracies = []

            for trial in range(num_trials):
                # Random subset of training data
                indices = torch.randperm(len(train_dataset))[:size]
                subset = Subset(train_dataset, indices)

                # Train model on subset
                trained_model = train_function(subset)

                # Evaluate on validation set
                self.model = trained_model
                val_results = self.evaluate(val_dataset)

                accuracy = val_results['accuracy']
                size_accuracies.append(accuracy)

                print(f"  Trial {trial+1}/{num_trials}: {accuracy:.2f}%")

            results[size] = {
                'mean': np.mean(size_accuracies),
                'std': np.std(size_accuracies),
                'trials': size_accuracies
            }

            print(f"  Average: {results[size]['mean']:.2f}% ± {results[size]['std']:.2f}%")

        return results

    def test_ood_generalization(
        self,
        ood_dataloaders: Dict[str, DataLoader],
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Test out-of-distribution (OOD) generalization to unseen operating conditions.

        PINN models should generalize better to new operating conditions due to
        physics constraints.

        Args:
            ood_dataloaders: Dict mapping condition name to dataloader
                            e.g., {'high_speed': loader1, 'high_load': loader2}
            class_names: Optional class names

        Returns:
            Dictionary mapping condition to performance metrics
        """
        results = {}

        for condition_name, dataloader in ood_dataloaders.items():
            print(f"\nEvaluating on {condition_name}:")

            condition_results = self.evaluate(dataloader, class_names)

            results[condition_name] = {
                'accuracy': condition_results['accuracy'],
                'num_samples': len(dataloader.dataset)
            }

            print(f"  Accuracy: {condition_results['accuracy']:.2f}%")

        return results

    def compare_with_baseline(
        self,
        baseline_model: nn.Module,
        test_dataloader: DataLoader,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, any]:
        """
        Compare PINN performance with baseline CNN.

        Args:
            baseline_model: Baseline model (standard CNN without physics)
            test_dataloader: Test data
            class_names: Class names

        Returns:
            Comparison results
        """
        print("\nComparing PINN vs Baseline CNN:")

        # Evaluate PINN
        print("\nEvaluating PINN:")
        pinn_results = self.evaluate(test_dataloader, class_names)

        # Evaluate baseline
        print("\nEvaluating Baseline:")
        baseline_evaluator = ModelEvaluator(baseline_model, self.device)
        baseline_results = baseline_evaluator.evaluate(test_dataloader, class_names)

        # Compute improvements
        acc_improvement = pinn_results['accuracy'] - baseline_results['accuracy']

        print("\n" + "=" * 60)
        print("COMPARISON RESULTS:")
        print(f"  PINN Accuracy:     {pinn_results['accuracy']:.2f}%")
        print(f"  Baseline Accuracy: {baseline_results['accuracy']:.2f}%")
        print(f"  Improvement:       {acc_improvement:+.2f}%")
        print("=" * 60)

        return {
            'pinn_accuracy': pinn_results['accuracy'],
            'baseline_accuracy': baseline_results['accuracy'],
            'improvement': acc_improvement,
            'pinn_results': pinn_results,
            'baseline_results': baseline_results
        }


def evaluate_pinn_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = 'cuda',
    metadata: Optional[Dict] = None,
    class_names: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Convenience function to evaluate PINN model with physics metrics.

    Args:
        model: PINN model
        test_loader: Test data loader
        device: Device
        metadata: Operating conditions
        class_names: Class names

    Returns:
        Evaluation results with physics metrics
    """
    evaluator = PINNEvaluator(model, device)
    results = evaluator.evaluate_with_physics_metrics(test_loader, metadata, class_names)

    print("\n" + "=" * 60)
    print("PINN EVALUATION RESULTS:")
    print(f"  Overall Accuracy: {results['accuracy']:.2f}%")

    if 'physics_metrics' in results:
        print(f"\n  Physics Metrics:")
        print(f"    Frequency Consistency: {results['physics_metrics']['frequency_consistency']:.3f}")
        print(f"    Prediction Plausibility: {results['physics_metrics']['prediction_plausibility']:.3f}")

    print("=" * 60)

    return results


if __name__ == "__main__":
    # Test PINN evaluator
    print("=" * 60)
    print("PINN Evaluator - Validation")
    print("=" * 60)

    from models.pinn.hybrid_pinn import HybridPINN

    # Create dummy model and data
    model = HybridPINN(num_classes=11, backbone='resnet18')

    from torch.utils.data import TensorDataset

    # Dummy test data
    n_samples = 100
    signals = torch.randn(n_samples, 1, 102400)
    labels = torch.randint(0, 11, (n_samples,))
    rpm = torch.tensor([3600.0] * n_samples)

    test_dataset = TensorDataset(signals, labels, rpm)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Create evaluator
    evaluator = PINNEvaluator(model, device='cpu')

    print("\nTesting Standard Evaluation:")
    results = evaluator.evaluate(test_loader)
    print(f"  Accuracy: {results['accuracy']:.2f}%")

    print("\nTesting Physics-Aware Evaluation:")
    metadata = {'rpm': rpm}
    results_physics = evaluator.evaluate_with_physics_metrics(
        test_loader, metadata
    )
    print(f"  Accuracy: {results_physics['accuracy']:.2f}%")
    if 'physics_metrics' in results_physics:
        print(f"  Frequency Consistency: {results_physics['physics_metrics']['frequency_consistency']:.3f}")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
