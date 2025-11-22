"""
Integrated Gradients for Attribution

Implements Integrated Gradients (Sundararajan et al., 2017) for attributing
predictions to input features. This method computes the integral of gradients
along a path from a baseline to the input.

Key Properties:
- Sensitivity: If inputs differ only in one feature and predictions differ,
  the differing feature should have non-zero attribution
- Implementation Invariance: Attribution is the same for functionally equivalent networks
- Completeness: Attributions sum to difference between output at input and baseline

Reference:
Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks.
International Conference on Machine Learning (ICML).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Union, List
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))


class IntegratedGradientsExplainer:
    """
    Integrated Gradients explainer for neural network models.

    Computes feature attributions by integrating gradients along a straight
    line path from a baseline input to the actual input.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        """
        Initialize Integrated Gradients explainer.

        Args:
            model: PyTorch model to explain
            device: Device to run computations on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def explain(
        self,
        input_signal: torch.Tensor,
        target_class: Optional[int] = None,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50,
        internal_batch_size: int = 8
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients attributions.

        Args:
            input_signal: Input signal to explain [1, C, T] or [C, T]
            target_class: Target class to explain (if None, uses predicted class)
            baseline: Baseline input (if None, uses zero baseline)
            steps: Number of integration steps
            internal_batch_size: Batch size for gradient computation

        Returns:
            Attributions with same shape as input_signal
        """
        # Ensure input has batch dimension
        if input_signal.dim() == 2:
            input_signal = input_signal.unsqueeze(0)

        input_signal = input_signal.to(self.device)

        # Create baseline if not provided (zero baseline)
        if baseline is None:
            baseline = torch.zeros_like(input_signal)
        else:
            baseline = baseline.to(self.device)
            if baseline.dim() == 2:
                baseline = baseline.unsqueeze(0)

        # Get target class if not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_signal)
                target_class = output.argmax(dim=1).item()

        # Compute integrated gradients
        attributions = self._compute_integrated_gradients(
            input_signal,
            baseline,
            target_class,
            steps,
            internal_batch_size
        )

        return attributions

    def _compute_integrated_gradients(
        self,
        input_signal: torch.Tensor,
        baseline: torch.Tensor,
        target_class: int,
        steps: int,
        batch_size: int
    ) -> torch.Tensor:
        """
        Core Integrated Gradients computation.

        IG(x) = (x - x') × ∫[α=0 to 1] ∂F(x' + α(x - x'))/∂x dα

        Approximated using Riemann sum with n steps.
        """
        # Generate interpolated inputs along straight line path
        # x(α) = x' + α(x - x') for α in [0, 1]
        alphas = torch.linspace(0, 1, steps + 1, device=self.device)

        # Compute interpolated inputs: [steps+1, C, T]
        input_diff = input_signal - baseline
        interpolated_inputs = baseline + alphas.view(-1, 1, 1) * input_diff

        # Compute gradients in batches
        all_gradients = []

        for i in range(0, len(interpolated_inputs), batch_size):
            batch = interpolated_inputs[i:i+batch_size]
            batch.requires_grad = True

            # Forward pass
            outputs = self.model(batch)

            # Get target class scores
            target_scores = outputs[:, target_class]

            # Backward pass
            gradients = torch.autograd.grad(
                outputs=target_scores,
                inputs=batch,
                grad_outputs=torch.ones_like(target_scores),
                create_graph=False
            )[0]

            all_gradients.append(gradients.detach())

        # Concatenate all gradients
        all_gradients = torch.cat(all_gradients, dim=0)  # [steps+1, C, T]

        # Approximate integral using trapezoidal rule
        # ∫f(x)dx ≈ Δx/2 × (f(x0) + 2f(x1) + 2f(x2) + ... + f(xn))
        avg_gradients = (all_gradients[:-1] + all_gradients[1:]) / 2.0
        avg_gradients = avg_gradients.mean(dim=0, keepdim=True)  # [1, C, T]

        # Multiply by input difference
        integrated_gradients = input_diff * avg_gradients

        return integrated_gradients

    def explain_batch(
        self,
        input_signals: torch.Tensor,
        target_classes: Optional[torch.Tensor] = None,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients for a batch of inputs.

        Args:
            input_signals: Batch of input signals [B, C, T]
            target_classes: Target classes for each sample [B]
            baseline: Baseline (if None, uses zero)
            steps: Integration steps

        Returns:
            Attributions [B, C, T]
        """
        batch_size = input_signals.shape[0]
        attributions = []

        for i in range(batch_size):
            target = target_classes[i].item() if target_classes is not None else None
            attr = self.explain(
                input_signals[i:i+1],
                target_class=target,
                baseline=baseline,
                steps=steps
            )
            attributions.append(attr)

        return torch.cat(attributions, dim=0)

    def compute_convergence_delta(
        self,
        input_signal: torch.Tensor,
        attributions: torch.Tensor,
        target_class: int,
        baseline: Optional[torch.Tensor] = None
    ) -> float:
        """
        Compute convergence delta to verify attribution quality.

        Completeness property: sum of attributions should equal
        F(x) - F(x'), where x' is baseline.

        Returns:
            Absolute difference between attribution sum and output difference
        """
        if baseline is None:
            baseline = torch.zeros_like(input_signal)

        baseline = baseline.to(self.device)
        input_signal = input_signal.to(self.device)

        with torch.no_grad():
            # Output at input
            output_input = self.model(input_signal)
            score_input = output_input[0, target_class].item()

            # Output at baseline
            output_baseline = self.model(baseline)
            score_baseline = output_baseline[0, target_class].item()

        # Sum of attributions
        attribution_sum = attributions.sum().item()

        # Expected difference
        expected_diff = score_input - score_baseline

        # Delta
        delta = abs(attribution_sum - expected_diff)

        return delta


def plot_attribution_map(
    signal: Union[torch.Tensor, np.ndarray],
    attributions: Union[torch.Tensor, np.ndarray],
    predicted_class: int,
    true_class: Optional[int] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
):
    """
    Visualize signal with attribution overlay.

    Args:
        signal: Original signal [C, T] or [T]
        attributions: Attribution values [C, T] or [T]
        predicted_class: Predicted fault class
        true_class: True fault class (optional)
        class_names: List of class names
        save_path: Path to save figure
        show_plot: Whether to display plot
    """
    # Convert to numpy
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()
    if isinstance(attributions, torch.Tensor):
        attributions = attributions.cpu().numpy()

    # Remove batch and channel dimensions
    signal = signal.squeeze()
    attributions = attributions.squeeze()

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    time = np.arange(len(signal))

    # Plot 1: Original Signal
    axes[0].plot(time, signal, 'b-', linewidth=0.8)
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title('Original Vibration Signal', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Attribution Values
    axes[1].plot(time, attributions, 'r-', linewidth=0.8)
    axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    axes[1].set_ylabel('Attribution', fontsize=11)
    axes[1].set_title('Integrated Gradients Attribution', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Signal with Attribution Overlay
    axes[2].plot(time, signal, 'b-', linewidth=0.8, alpha=0.5, label='Signal')

    # Color-code attribution importance
    attr_abs = np.abs(attributions)
    attr_normalized = attr_abs / (attr_abs.max() + 1e-8)

    # Create colormap overlay
    for i in range(len(time) - 1):
        color_intensity = attr_normalized[i]
        color = plt.cm.Reds(color_intensity)
        axes[2].axvspan(time[i], time[i+1], alpha=color_intensity * 0.5, color=color, linewidth=0)

    axes[2].set_xlabel('Time Steps', fontsize=11)
    axes[2].set_ylabel('Amplitude', fontsize=11)

    # Create title with class information
    if class_names is not None:
        pred_name = class_names[predicted_class]
        title = f'Signal with Attribution Overlay (Predicted: {pred_name})'
        if true_class is not None:
            true_name = class_names[true_class]
            title += f' | True: {true_name}'
    else:
        title = f'Signal with Attribution Overlay (Predicted Class: {predicted_class})'
        if true_class is not None:
            title += f' | True Class: {true_class}'

    axes[2].set_title(title, fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Attribution plot saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_attribution_heatmap(
    attributions: Union[torch.Tensor, np.ndarray],
    predicted_class: int,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot attribution as a heatmap (useful for multi-channel signals).

    Args:
        attributions: Attribution values [C, T]
        predicted_class: Predicted class
        class_names: List of class names
        save_path: Path to save figure
    """
    if isinstance(attributions, torch.Tensor):
        attributions = attributions.cpu().numpy()

    attributions = attributions.squeeze()

    # If 1D, make it 2D for heatmap
    if attributions.ndim == 1:
        attributions = attributions.reshape(1, -1)

    fig, ax = plt.subplots(figsize=(14, 4))

    im = ax.imshow(
        attributions,
        aspect='auto',
        cmap='RdBu_r',
        interpolation='nearest'
    )

    ax.set_xlabel('Time Steps', fontsize=11)
    ax.set_ylabel('Channel', fontsize=11)

    title = f'Attribution Heatmap (Predicted'
    if class_names:
        title += f': {class_names[predicted_class]}'
    else:
        title += f' Class: {predicted_class}'
    title += ')'
    ax.set_title(title, fontsize=12, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Attribution Value')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    # Test Integrated Gradients
    print("=" * 60)
    print("Integrated Gradients - Validation")
    print("=" * 60)

    # Create dummy model and data
    from models.cnn.cnn_1d import CNN1D

    model = CNN1D(num_classes=NUM_CLASSES, input_channels=1, dropout=0.3)

    # Dummy input
    signal = torch.randn(1, 1, 10240)

    # Create explainer
    explainer = IntegratedGradientsExplainer(model, device='cpu')

    print("\nComputing Integrated Gradients...")
    attributions = explainer.explain(
        signal,
        target_class=3,
        steps=50
    )

    print(f"  Input shape: {signal.shape}")
    print(f"  Attribution shape: {attributions.shape}")
    print(f"  Attribution range: [{attributions.min().item():.4f}, {attributions.max().item():.4f}]")
    print(f"  Attribution sum: {attributions.sum().item():.4f}")

    # Check convergence
    delta = explainer.compute_convergence_delta(signal, attributions, target_class=3)
    print(f"  Convergence delta: {delta:.6f}")

    if delta < 0.01:
        print("  ✓ Convergence check PASSED (delta < 0.01)")
    else:
        print(f"  ⚠ Convergence delta is {delta:.6f} (expected < 0.01)")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
