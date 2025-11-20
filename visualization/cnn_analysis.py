"""
CNN Analysis Tools - Deep analysis of CNN model behavior and performance

This module provides advanced analysis tools for understanding CNN model behavior:
- Gradient flow analysis (detect vanishing/exploding gradients)
- Layer-wise performance contribution
- Failure case analysis
- Saliency maps for interpretability
- Feature importance via occlusion

Author: Phase 2 - CNN Implementation
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict


class CNNAnalyzer:
    """
    Advanced analysis tools for 1D CNN models

    Provides:
    - Gradient flow visualization
    - Layer-wise ablation study
    - Failure case analysis
    - Saliency maps (input attribution)
    - Occlusion-based feature importance

    Examples:
        >>> from models.cnn.cnn_1d import CNN1D
        >>> model = CNN1D(num_classes=11, input_length=102400)
        >>> analyzer = CNNAnalyzer(model)
        >>>
        >>> # Analyze gradient flow during training
        >>> analyzer.analyze_gradient_flow()
        >>>
        >>> # Generate saliency map for a signal
        >>> signal = torch.randn(1, 1, 102400, requires_grad=True)
        >>> saliency = analyzer.compute_saliency_map(signal, target_class=3)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None
    ):
        """
        Initialize CNN analyzer

        Args:
            model: PyTorch CNN model
            device: Device to run analysis on
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Storage for gradients
        self.gradient_dict = defaultdict(list)
        self.hooks = []

    def _register_gradient_hooks(self):
        """Register backward hooks to capture gradients"""
        self._remove_hooks()

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                hook = module.register_backward_hook(
                    lambda mod, grad_in, grad_out, name=name:
                        self.gradient_dict[name].append(grad_out[0].detach().cpu())
                )
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.gradient_dict = defaultdict(list)

    def analyze_gradient_flow(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int = 10,
        figsize: Tuple[int, int] = (15, 8),
        save_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, float]:
        """
        Analyze gradient flow through the network

        Useful for detecting:
        - Vanishing gradients (very small magnitudes)
        - Exploding gradients (very large magnitudes)
        - Dead layers (zero gradients)

        Args:
            dataloader: DataLoader with training data
            num_batches: Number of batches to analyze
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Dictionary with gradient statistics per layer
        """
        self.model.train()
        self._register_gradient_hooks()

        criterion = nn.CrossEntropyLoss()
        gradient_stats = {}

        # Accumulate gradients over batches
        for batch_idx, (signals, labels) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            signals = signals.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(signals)
            loss = criterion(outputs, labels)

            # Backward pass
            self.model.zero_grad()
            loss.backward()

        # Compute statistics
        layer_names = []
        mean_grads = []
        max_grads = []
        min_grads = []

        for name, grads in self.gradient_dict.items():
            if len(grads) > 0:
                grads_tensor = torch.stack(grads)
                grad_magnitude = grads_tensor.abs().mean(dim=0).flatten()

                gradient_stats[name] = {
                    'mean': float(grad_magnitude.mean()),
                    'std': float(grad_magnitude.std()),
                    'max': float(grad_magnitude.max()),
                    'min': float(grad_magnitude.min()),
                    'zero_ratio': float((grad_magnitude == 0).sum().float() / grad_magnitude.numel())
                }

                layer_names.append(name)
                mean_grads.append(gradient_stats[name]['mean'])
                max_grads.append(gradient_stats[name]['max'])
                min_grads.append(gradient_stats[name]['min'])

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Mean gradient magnitudes
        x = range(len(layer_names))
        ax1.bar(x, mean_grads, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
        ax1.set_ylabel('Mean Gradient Magnitude', fontsize=12)
        ax1.set_title('Gradient Flow: Mean Magnitudes', fontsize=14, fontweight='bold')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=1e-6, color='red', linestyle='--', label='Vanishing threshold (1e-6)')
        ax1.axhline(y=1.0, color='orange', linestyle='--', label='Exploding threshold (1.0)')
        ax1.legend()

        # Min/Max range
        ax2.bar(x, max_grads, alpha=0.5, label='Max', color='red')
        ax2.bar(x, mean_grads, alpha=0.7, label='Mean', color='blue')
        ax2.set_xticks(x)
        ax2.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Gradient Magnitude', fontsize=12)
        ax2.set_title('Gradient Range (Mean vs Max)', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved gradient flow analysis to {save_path}")

        self._remove_hooks()
        self.model.eval()

        return gradient_stats

    def compute_saliency_map(
        self,
        input_signal: torch.Tensor,
        target_class: int,
        figsize: Tuple[int, int] = (15, 5),
        save_path: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """
        Compute saliency map (input gradient) for interpretability

        Shows which parts of the input signal are most important for the prediction.

        Args:
            input_signal: Input signal [1, 1, length] with requires_grad=True
            target_class: Target class for saliency computation
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Saliency map (numpy array)
        """
        self.model.eval()

        if not input_signal.requires_grad:
            input_signal.requires_grad = True

        input_signal = input_signal.to(self.device)

        # Forward pass
        output = self.model(input_signal)

        # Backward pass w.r.t. target class
        self.model.zero_grad()
        output[0, target_class].backward()

        # Get gradient
        saliency = input_signal.grad.abs().cpu().numpy()[0, 0]  # [length]

        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Original signal
        signal_np = input_signal.detach().cpu().numpy()[0, 0]
        ax1.plot(signal_np, linewidth=0.5, color='black', alpha=0.7)
        ax1.set_title('Input Signal', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)

        # Saliency map
        ax2.plot(saliency, linewidth=0.5, color='red')
        ax2.fill_between(range(len(saliency)), 0, saliency, alpha=0.3, color='red')
        ax2.set_title(f'Saliency Map (Target Class: {target_class})', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Index')
        ax2.set_ylabel('Importance')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved saliency map to {save_path}")

        return saliency

    def occlusion_sensitivity(
        self,
        input_signal: torch.Tensor,
        target_class: int,
        window_size: int = 1024,
        stride: int = 512,
        figsize: Tuple[int, int] = (15, 5),
        save_path: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """
        Compute occlusion sensitivity map

        Systematically occlude parts of the input and measure impact on prediction.
        More robust than gradient-based methods but computationally expensive.

        Args:
            input_signal: Input signal [1, 1, length]
            target_class: Target class to analyze
            window_size: Size of occlusion window
            stride: Stride between occlusions
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Sensitivity map (numpy array)
        """
        self.model.eval()
        input_signal = input_signal.to(self.device)
        signal_length = input_signal.shape[2]

        # Get baseline prediction
        with torch.no_grad():
            baseline_output = self.model(input_signal)
            baseline_prob = F.softmax(baseline_output, dim=1)[0, target_class].item()

        # Occlusion loop
        sensitivities = []
        positions = []

        with torch.no_grad():
            for start in range(0, signal_length - window_size, stride):
                end = start + window_size

                # Create occluded signal (set window to zero)
                occluded_signal = input_signal.clone()
                occluded_signal[:, :, start:end] = 0

                # Predict
                output = self.model(occluded_signal)
                prob = F.softmax(output, dim=1)[0, target_class].item()

                # Sensitivity = drop in probability
                sensitivity = baseline_prob - prob
                sensitivities.append(sensitivity)
                positions.append(start + window_size // 2)

        # Interpolate to full signal length
        sensitivity_map = np.interp(
            range(signal_length),
            positions,
            sensitivities
        )

        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Original signal
        signal_np = input_signal.cpu().numpy()[0, 0]
        ax1.plot(signal_np, linewidth=0.5, color='black', alpha=0.7)
        ax1.set_title('Input Signal', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)

        # Sensitivity map
        ax2.plot(sensitivity_map, linewidth=1, color='blue')
        ax2.fill_between(range(len(sensitivity_map)), 0, sensitivity_map,
                        alpha=0.3, color='blue')
        ax2.set_title(f'Occlusion Sensitivity (Target Class: {target_class})',
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Index')
        ax2.set_ylabel('Sensitivity (Δ Probability)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved occlusion sensitivity to {save_path}")

        return sensitivity_map

    def analyze_failure_cases(
        self,
        dataloader: torch.utils.data.DataLoader,
        class_names: List[str],
        n_cases: int = 10,
        figsize: Tuple[int, int] = (15, 12),
        save_path: Optional[Union[str, Path]] = None
    ) -> Dict:
        """
        Analyze misclassified samples (failure cases)

        Args:
            dataloader: Test/validation dataloader
            class_names: List of class names
            n_cases: Number of failure cases to analyze
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Dictionary with failure case analysis
        """
        self.model.eval()

        failures = []

        with torch.no_grad():
            for signals, labels in dataloader:
                signals = signals.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(signals)
                probs = F.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                # Find misclassifications
                mask = preds != labels

                for i in range(signals.size(0)):
                    if mask[i] and len(failures) < n_cases:
                        failures.append({
                            'signal': signals[i].cpu().numpy(),
                            'true_label': labels[i].item(),
                            'pred_label': preds[i].item(),
                            'confidence': probs[i, preds[i]].item(),
                            'true_prob': probs[i, labels[i]].item()
                        })

                if len(failures) >= n_cases:
                    break

        if len(failures) == 0:
            print("✓ No failure cases found (perfect classification!)")
            return {}

        # Plot failure cases
        n_cols = 2
        n_rows = (len(failures) + 1) // 2

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

        for idx, failure in enumerate(failures):
            ax = axes[idx]

            signal = failure['signal'][0][:2000]  # Plot first 2000 samples
            true_label = class_names[failure['true_label']]
            pred_label = class_names[failure['pred_label']]

            ax.plot(signal, linewidth=0.5, color='red', alpha=0.7)
            ax.set_title(
                f"True: {true_label} | Pred: {pred_label}\n"
                f"Conf: {failure['confidence']:.3f}",
                fontsize=9
            )
            ax.set_xlabel('Time Index', fontsize=8)
            ax.set_ylabel('Amplitude', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Remove empty subplots
        for idx in range(len(failures), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved failure case analysis to {save_path}")

        # Summary statistics
        confusion_pairs = defaultdict(int)
        for failure in failures:
            pair = (failure['true_label'], failure['pred_label'])
            confusion_pairs[pair] += 1

        print(f"\n✓ Found {len(failures)} failure cases:")
        print("Most common confusions:")
        for (true_idx, pred_idx), count in sorted(confusion_pairs.items(),
                                                   key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {class_names[true_idx]} → {class_names[pred_idx]}: {count} cases")

        return {
            'failures': failures,
            'confusion_pairs': dict(confusion_pairs)
        }

    def layer_ablation_study(
        self,
        dataloader: torch.utils.data.DataLoader,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, float]:
        """
        Perform layer ablation study

        Systematically remove each layer and measure performance impact
        to understand layer importance.

        Args:
            dataloader: Validation dataloader
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Dictionary with accuracy per ablation
        """
        # Note: This is a simplified ablation that zeros out layer outputs
        # Full ablation would require retraining

        print("Running layer ablation study (zeroing outputs)...")

        self.model.eval()
        results = {'baseline': self._compute_accuracy(dataloader)}

        # Get all conv layers
        conv_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv1d):
                conv_layers.append(name)

        # Ablate each layer
        for layer_name in conv_layers:
            print(f"  Ablating {layer_name}...")
            accuracy = self._ablate_layer_accuracy(dataloader, layer_name)
            results[layer_name] = accuracy

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        layers = list(results.keys())
        accuracies = list(results.values())
        drops = [results['baseline'] - acc for acc in accuracies]

        x = range(len(layers))
        colors = ['green' if l == 'baseline' else 'red' for l in layers]

        ax.bar(x, accuracies, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Layer Ablation Study', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=results['baseline'], color='blue', linestyle='--',
                  label=f"Baseline: {results['baseline']:.3f}")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved ablation study to {save_path}")

        return results

    def _compute_accuracy(self, dataloader):
        """Compute accuracy on dataloader"""
        correct = 0
        total = 0

        with torch.no_grad():
            for signals, labels in dataloader:
                signals = signals.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(signals)
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total if total > 0 else 0.0

    def _ablate_layer_accuracy(self, dataloader, layer_name):
        """Compute accuracy with one layer ablated (output zeroed)"""
        hook_handle = None

        def ablation_hook(module, input, output):
            return torch.zeros_like(output)

        # Register hook
        for name, module in self.model.named_modules():
            if name == layer_name:
                hook_handle = module.register_forward_hook(ablation_hook)
                break

        accuracy = self._compute_accuracy(dataloader)

        # Remove hook
        if hook_handle:
            hook_handle.remove()

        return accuracy

    def __del__(self):
        """Cleanup"""
        self._remove_hooks()


def test_cnn_analyzer():
    """Test CNN analysis functions"""
    print("Testing CNNAnalyzer...")

    from models.cnn.cnn_1d import CNN1D

    # Create model
    model = CNN1D(num_classes=11, input_length=102400)
    analyzer = CNNAnalyzer(model)

    # Test saliency map
    print("\n✓ Testing saliency map...")
    signal = torch.randn(1, 1, 102400, requires_grad=True)
    saliency = analyzer.compute_saliency_map(signal, target_class=3)
    print(f"  Saliency shape: {saliency.shape}")
    print(f"  Saliency range: [{saliency.min():.6f}, {saliency.max():.6f}]")
    plt.close('all')

    # Test occlusion sensitivity
    print("\n✓ Testing occlusion sensitivity...")
    sensitivity = analyzer.occlusion_sensitivity(signal.detach(), target_class=3,
                                                window_size=512, stride=256)
    print(f"  Sensitivity shape: {sensitivity.shape}")
    print(f"  Sensitivity range: [{sensitivity.min():.6f}, {sensitivity.max():.6f}]")
    plt.close('all')

    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_cnn_analyzer()
