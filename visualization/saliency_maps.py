"""
Saliency Maps and Gradient-Based Visualization

Implements various gradient-based visualization techniques for interpreting
neural network predictions:
- Vanilla Gradients (Simonyan et al., 2013)
- SmoothGrad (Smilkov et al., 2017)
- Guided Backpropagation
- GradCAM adaptation for 1D signals

These methods show which parts of the input have the largest gradient with
respect to the output, indicating importance for the prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union, List
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.constants import NUM_CLASSES


class SaliencyMapGenerator:
    """
    Generates saliency maps using gradient-based methods.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        """
        Initialize saliency map generator.

        Args:
            model: PyTorch model to visualize
            device: Device to run computations on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def vanilla_gradient(
        self,
        input_signal: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute vanilla gradient saliency map.

        Simply computes gradient of target class score w.r.t. input.

        Args:
            input_signal: Input signal [1, C, T] or [C, T]
            target_class: Target class (if None, uses predicted class)

        Returns:
            Saliency map with same shape as input
        """
        # Ensure batch dimension
        if input_signal.dim() == 2:
            input_signal = input_signal.unsqueeze(0)

        input_signal = input_signal.to(self.device)
        input_signal.requires_grad = True

        # Forward pass
        output = self.model(input_signal)

        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()

        # Get gradients
        saliency = input_signal.grad.abs()

        return saliency.detach()

    def smooth_grad(
        self,
        input_signal: torch.Tensor,
        target_class: Optional[int] = None,
        noise_level: float = 0.1,
        n_samples: int = 50
    ) -> torch.Tensor:
        """
        Compute SmoothGrad saliency map.

        Averages gradients over multiple noisy versions of the input to
        reduce visual diffusion and noise.

        Reference: Smilkov et al., "SmoothGrad: removing noise by adding noise", 2017

        Args:
            input_signal: Input signal [1, C, T]
            target_class: Target class
            noise_level: Standard deviation of Gaussian noise (relative to input std)
            n_samples: Number of noisy samples to average

        Returns:
            Smooth saliency map
        """
        if input_signal.dim() == 2:
            input_signal = input_signal.unsqueeze(0)

        input_signal = input_signal.to(self.device)

        # Get target class
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_signal)
                target_class = output.argmax(dim=1).item()

        # Compute std for noise
        std = input_signal.std() * noise_level

        # Accumulate gradients
        total_gradients = torch.zeros_like(input_signal)

        for _ in range(n_samples):
            # Add Gaussian noise
            noise = torch.randn_like(input_signal) * std
            noisy_input = input_signal + noise
            noisy_input.requires_grad = True

            # Forward pass
            output = self.model(noisy_input)

            # Backward pass
            self.model.zero_grad()
            output[0, target_class].backward()

            # Accumulate gradients
            total_gradients += noisy_input.grad.abs()

        # Average gradients
        smooth_saliency = total_gradients / n_samples

        return smooth_saliency.detach()

    def gradient_times_input(
        self,
        input_signal: torch.Tensor,
        target_class: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute Gradient × Input saliency.

        Multiplies gradient by input value, giving signed importance.

        Args:
            input_signal: Input signal [1, C, T]
            target_class: Target class

        Returns:
            Saliency map
        """
        if input_signal.dim() == 2:
            input_signal = input_signal.unsqueeze(0)

        input_signal = input_signal.to(self.device)
        input_signal.requires_grad = True

        # Forward pass
        output = self.model(input_signal)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()

        # Gradient × Input
        saliency = input_signal.grad * input_signal

        return saliency.abs().detach()

    def grad_cam_1d(
        self,
        input_signal: torch.Tensor,
        target_class: Optional[int] = None,
        target_layer: Optional[nn.Module] = None
    ) -> torch.Tensor:
        """
        Compute GradCAM-style attention for 1D signals.

        Uses gradients flowing into a convolutional layer to produce a
        coarse localization map highlighting important regions.

        Args:
            input_signal: Input signal [1, C, T]
            target_class: Target class
            target_layer: Target convolutional layer (if None, uses last conv layer)

        Returns:
            GradCAM map [1, 1, T'] (may be shorter due to convolutions)
        """
        if input_signal.dim() == 2:
            input_signal = input_signal.unsqueeze(0)

        input_signal = input_signal.to(self.device)
        input_signal.requires_grad = True

        # Find target layer if not specified
        if target_layer is None:
            target_layer = self._find_last_conv_layer()

        # Hook to capture activations and gradients
        activations = {}
        gradients = {}

        def forward_hook(module, input, output):
            activations['value'] = output

        def backward_hook(module, grad_input, grad_output):
            gradients['value'] = grad_output[0]

        # Register hooks
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            # Forward pass
            output = self.model(input_signal)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            # Backward pass
            self.model.zero_grad()
            output[0, target_class].backward()

            # Get activations and gradients
            acts = activations['value']  # [1, C, T']
            grads = gradients['value']  # [1, C, T']

            # Global average pooling of gradients (weights)
            weights = grads.mean(dim=2, keepdim=True)  # [1, C, 1]

            # Weighted combination of activation maps
            cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, T']

            # ReLU (only positive influences)
            cam = F.relu(cam)

            # Normalize to [0, 1]
            cam = cam / (cam.max() + 1e-8)

            # Upsample to input size
            if cam.shape[2] != input_signal.shape[2]:
                cam = F.interpolate(
                    cam,
                    size=input_signal.shape[2],
                    mode='linear',
                    align_corners=False
                )

        finally:
            # Remove hooks
            forward_handle.remove()
            backward_handle.remove()

        return cam.detach()

    def _find_last_conv_layer(self) -> nn.Module:
        """Find the last convolutional layer in the model."""
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d)):
                last_conv = module
        return last_conv


def plot_saliency_map(
    signal: Union[torch.Tensor, np.ndarray],
    saliency: Union[torch.Tensor, np.ndarray],
    predicted_class: int,
    true_class: Optional[int] = None,
    method_name: str = "Saliency",
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Visualize signal with saliency map overlay.

    Args:
        signal: Original signal
        saliency: Saliency map
        predicted_class: Predicted class
        true_class: True class (optional)
        method_name: Name of saliency method
        class_names: List of class names
        save_path: Path to save figure
    """
    # Convert to numpy
    if isinstance(signal, torch.Tensor):
        signal = signal.cpu().numpy()
    if isinstance(saliency, torch.Tensor):
        saliency = saliency.cpu().numpy()

    signal = signal.squeeze()
    saliency = saliency.squeeze()

    # Normalize saliency for visualization
    saliency_norm = saliency / (saliency.max() + 1e-8)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    time = np.arange(len(signal))

    # Plot 1: Original Signal
    axes[0].plot(time, signal, 'b-', linewidth=0.8)
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title('Original Signal', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Saliency Map
    axes[1].plot(time, saliency, 'r-', linewidth=0.8)
    axes[1].fill_between(time, 0, saliency, alpha=0.3, color='red')
    axes[1].set_ylabel('Saliency', fontsize=11)
    axes[1].set_title(f'{method_name} Map', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Overlay
    axes[2].plot(time, signal, 'b-', linewidth=0.8, alpha=0.5, label='Signal')

    # Heatmap overlay
    for i in range(len(time) - 1):
        alpha_val = saliency_norm[i]
        color = plt.cm.Reds(alpha_val)
        axes[2].axvspan(time[i], time[i+1], alpha=alpha_val * 0.6, color=color, linewidth=0)

    axes[2].set_xlabel('Time Steps', fontsize=11)
    axes[2].set_ylabel('Amplitude', fontsize=11)

    # Title
    if class_names:
        title = f'Prediction: {class_names[predicted_class]}'
        if true_class is not None:
            title += f' | True: {class_names[true_class]}'
    else:
        title = f'Predicted: {predicted_class}'
        if true_class is not None:
            title += f' | True: {true_class}'

    axes[2].set_title(title, fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saliency plot saved to {save_path}")

    plt.show()


def compare_saliency_methods(
    signal: torch.Tensor,
    generator: SaliencyMapGenerator,
    target_class: Optional[int] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Compare multiple saliency methods side-by-side.

    Args:
        signal: Input signal
        generator: SaliencyMapGenerator instance
        target_class: Target class
        class_names: Class names
        save_path: Path to save comparison figure
    """
    # Compute different saliency maps
    print("Computing saliency maps...")

    vanilla = generator.vanilla_gradient(signal, target_class)
    print("  ✓ Vanilla Gradient")

    smooth = generator.smooth_grad(signal, target_class, n_samples=30)
    print("  ✓ SmoothGrad")

    grad_input = generator.gradient_times_input(signal, target_class)
    print("  ✓ Gradient × Input")

    # Convert to numpy
    signal_np = signal.cpu().numpy().squeeze()
    vanilla_np = vanilla.cpu().numpy().squeeze()
    smooth_np = smooth.cpu().numpy().squeeze()
    grad_input_np = grad_input.cpu().numpy().squeeze()

    # Normalize for visualization
    vanilla_np = vanilla_np / (vanilla_np.max() + 1e-8)
    smooth_np = smooth_np / (smooth_np.max() + 1e-8)
    grad_input_np = grad_input_np / (grad_input_np.max() + 1e-8)

    # Plot comparison
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    time = np.arange(len(signal_np))

    # Original Signal
    axes[0].plot(time, signal_np, 'b-', linewidth=0.8)
    axes[0].set_ylabel('Amplitude', fontsize=10)
    axes[0].set_title('Original Signal', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Method 1: Vanilla Gradient
    axes[1].fill_between(time, 0, vanilla_np, alpha=0.5, color='red')
    axes[1].plot(time, vanilla_np, 'r-', linewidth=0.8)
    axes[1].set_ylabel('Saliency', fontsize=10)
    axes[1].set_title('Vanilla Gradient', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Method 2: SmoothGrad
    axes[2].fill_between(time, 0, smooth_np, alpha=0.5, color='green')
    axes[2].plot(time, smooth_np, 'g-', linewidth=0.8)
    axes[2].set_ylabel('Saliency', fontsize=10)
    axes[2].set_title('SmoothGrad (n=30)', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    # Method 3: Gradient × Input
    axes[3].fill_between(time, 0, grad_input_np, alpha=0.5, color='purple')
    axes[3].plot(time, grad_input_np, color='purple', linewidth=0.8)
    axes[3].set_xlabel('Time Steps', fontsize=10)
    axes[3].set_ylabel('Saliency', fontsize=10)
    axes[3].set_title('Gradient × Input', fontsize=11, fontweight='bold')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Test saliency map generation
    print("=" * 60)
    print("Saliency Maps - Validation")
    print("=" * 60)

    from models.cnn.cnn_1d import CNN1D

    # Create model and dummy data
    model = CNN1D(num_classes=NUM_CLASSES, input_channels=1, dropout=0.3)
    signal = torch.randn(1, 1, 10240)

    # Create generator
    generator = SaliencyMapGenerator(model, device='cpu')

    print("\nTesting Vanilla Gradient...")
    vanilla = generator.vanilla_gradient(signal, target_class=3)
    print(f"  Output shape: {vanilla.shape}")
    print(f"  Range: [{vanilla.min().item():.4f}, {vanilla.max().item():.4f}]")

    print("\nTesting SmoothGrad...")
    smooth = generator.smooth_grad(signal, target_class=3, n_samples=20)
    print(f"  Output shape: {smooth.shape}")
    print(f"  Range: [{smooth.min().item():.4f}, {smooth.max().item():.4f}]")

    print("\nTesting Gradient × Input...")
    grad_input = generator.gradient_times_input(signal, target_class=3)
    print(f"  Output shape: {grad_input.shape}")
    print(f"  Range: [{grad_input.min().item():.4f}, {grad_input.max().item():.4f}]")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
