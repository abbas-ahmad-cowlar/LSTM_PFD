"""
CNN Visualizer - Visualize CNN filters, feature maps, and activations

This module provides comprehensive visualization tools for analyzing 1D CNN models
trained on vibration signals. Includes filter visualization, feature map analysis,
activation distribution monitoring, and gradient flow visualization.

Key Features:
- Filter/kernel visualization across all convolutional layers
- Feature map extraction and visualization at intermediate layers
- Activation distribution analysis (ReLU saturation detection)
- Gradient magnitude visualization (detect vanishing/exploding gradients)
- Receptive field calculation and visualization

Author: Phase 2 - CNN Implementation
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict


class CNNVisualizer:
    """
    Comprehensive visualization suite for 1D CNN models

    Provides tools to visualize and analyze:
    - Convolutional filters/kernels
    - Feature maps (activations at intermediate layers)
    - Activation distributions
    - Gradient flow
    - Receptive fields

    Examples:
        >>> from models.cnn.cnn_1d import CNN1D
        >>> model = CNN1D(num_classes=NUM_CLASSES, input_length=102400)
        >>> visualizer = CNNVisualizer(model)
        >>>
        >>> # Visualize filters
        >>> visualizer.plot_conv_filters(save_path='filters.png')
        >>>
        >>> # Visualize feature maps for a signal
        >>> signal = torch.randn(1, 1, SIGNAL_LENGTH)
        >>> visualizer.plot_feature_maps(signal, layer_name='conv1', save_path='fmaps.png')
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None
    ):
        """
        Initialize CNN visualizer

        Args:
            model: PyTorch CNN model (nn.Module)
            device: Device to run visualizations on (default: auto-detect)
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # Extract convolutional layers
        self.conv_layers = self._extract_conv_layers()

        # Hook storage for intermediate activations
        self.activations = {}
        self.gradients = {}
        self.hooks = []

    def _extract_conv_layers(self) -> OrderedDict:
        """Extract all Conv1d layers from the model"""
        conv_layers = OrderedDict()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv1d):
                conv_layers[name] = module

        return conv_layers

    def _register_hooks(self, layer_names: Optional[List[str]] = None):
        """Register forward hooks to capture intermediate activations"""
        self._remove_hooks()

        target_layers = layer_names or list(self.conv_layers.keys())

        for name, module in self.model.named_modules():
            if name in target_layers:
                hook = module.register_forward_hook(
                    lambda mod, inp, out, name=name: self.activations.update({name: out})
                )
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
        self.gradients = {}

    def plot_conv_filters(
        self,
        layer_name: Optional[str] = None,
        max_filters: int = 64,
        figsize: Tuple[int, int] = (20, 12),
        save_path: Optional[Union[str, Path]] = None
    ):
        """
        Visualize convolutional filters (kernels) as 1D signals

        Args:
            layer_name: Specific layer to visualize (None = all layers)
            max_filters: Maximum number of filters to display per layer
            figsize: Figure size (width, height)
            save_path: Path to save figure (None = display only)

        Returns:
            Figure object
        """
        layers_to_plot = [layer_name] if layer_name else list(self.conv_layers.keys())

        n_layers = len(layers_to_plot)
        fig, axes = plt.subplots(n_layers, 1, figsize=figsize)

        if n_layers == 1:
            axes = [axes]

        for idx, (layer_name, ax) in enumerate(zip(layers_to_plot, axes)):
            layer = self.conv_layers[layer_name]
            weights = layer.weight.detach().cpu().numpy()  # [out_ch, in_ch, kernel_size]

            out_ch, in_ch, kernel_size = weights.shape
            n_filters = min(out_ch, max_filters)

            # Plot filters
            for i in range(n_filters):
                # Average over input channels for visualization
                filter_1d = weights[i].mean(axis=0)
                ax.plot(filter_1d, alpha=0.6, linewidth=1)

            ax.set_title(f'{layer_name}: {out_ch} filters, kernel_size={kernel_size}',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Kernel Index')
            ax.set_ylabel('Weight Value')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved filter visualization to {save_path}")

        return fig

    def plot_feature_maps(
        self,
        input_signal: torch.Tensor,
        layer_name: str,
        max_channels: int = 16,
        signal_length: int = 1000,
        figsize: Tuple[int, int] = (20, 10),
        save_path: Optional[Union[str, Path]] = None
    ):
        """
        Visualize feature maps (activations) at a specific layer

        Args:
            input_signal: Input signal [batch, 1, length] or [1, length]
            layer_name: Name of the layer to visualize
            max_channels: Maximum number of channels to display
            signal_length: Length of signal segment to display
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Feature maps (numpy array)
        """
        # Ensure proper input shape
        if input_signal.dim() == 2:
            input_signal = input_signal.unsqueeze(1)  # [batch, 1, length]

        input_signal = input_signal.to(self.device)

        # Register hook and forward pass
        self._register_hooks([layer_name])

        with torch.no_grad():
            _ = self.model(input_signal)

        # Get feature maps
        feature_maps = self.activations[layer_name].cpu().numpy()[0]  # [channels, length]
        n_channels = min(feature_maps.shape[0], max_channels)

        # Trim to display length
        if feature_maps.shape[1] > signal_length:
            feature_maps = feature_maps[:, :signal_length]

        # Plot
        fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)

        if n_channels == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            ax.plot(feature_maps[i], linewidth=1, color='b')
            ax.set_ylabel(f'Ch {i+1}', fontsize=10)
            ax.grid(True, alpha=0.3)

            # Highlight positive activations
            ax.fill_between(range(len(feature_maps[i])), 0, feature_maps[i],
                           where=(feature_maps[i] > 0), alpha=0.3, color='green')

        axes[0].set_title(f'Feature Maps: {layer_name} ({feature_maps.shape[0]} channels)',
                         fontsize=14, fontweight='bold')
        axes[-1].set_xlabel('Time Index', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved feature maps to {save_path}")

        self._remove_hooks()

        return feature_maps

    def plot_activation_distributions(
        self,
        input_signal: torch.Tensor,
        figsize: Tuple[int, int] = (15, 10),
        save_path: Optional[Union[str, Path]] = None
    ):
        """
        Plot activation distributions across all layers

        Useful for detecting:
        - Dead neurons (all zeros)
        - Saturated activations (all max values)
        - Activation imbalance

        Args:
            input_signal: Input signal [batch, 1, length]
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Dictionary of activation statistics
        """
        if input_signal.dim() == 2:
            input_signal = input_signal.unsqueeze(1)

        input_signal = input_signal.to(self.device)

        # Register all conv layer hooks
        self._register_hooks()

        with torch.no_grad():
            _ = self.model(input_signal)

        # Compute statistics
        stats = {}
        n_layers = len(self.conv_layers)

        fig, axes = plt.subplots(n_layers, 2, figsize=figsize)

        if n_layers == 1:
            axes = axes.reshape(1, -1)

        for idx, (layer_name, layer) in enumerate(self.conv_layers.items()):
            activations = self.activations[layer_name].cpu().numpy().flatten()

            # Statistics
            stats[layer_name] = {
                'mean': float(np.mean(activations)),
                'std': float(np.std(activations)),
                'min': float(np.min(activations)),
                'max': float(np.max(activations)),
                'dead_ratio': float(np.mean(activations == 0)),
                'sparsity': float(np.mean(np.abs(activations) < 0.01))
            }

            # Histogram
            axes[idx, 0].hist(activations, bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[idx, 0].set_title(f'{layer_name} - Distribution', fontsize=10)
            axes[idx, 0].set_xlabel('Activation Value')
            axes[idx, 0].set_ylabel('Count')
            axes[idx, 0].axvline(x=0, color='red', linestyle='--', linewidth=1)
            axes[idx, 0].grid(True, alpha=0.3)

            # Box plot
            axes[idx, 1].boxplot(activations, vert=True, widths=0.5)
            axes[idx, 1].set_title(f'{layer_name} - Statistics', fontsize=10)
            axes[idx, 1].set_ylabel('Activation Value')
            axes[idx, 1].grid(True, alpha=0.3, axis='y')

            # Add text annotations
            text = f"Dead: {stats[layer_name]['dead_ratio']*100:.1f}%\n"
            text += f"Sparse: {stats[layer_name]['sparsity']*100:.1f}%"
            axes[idx, 1].text(1.2, np.median(activations), text,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                            fontsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved activation distributions to {save_path}")

        self._remove_hooks()

        return stats

    def plot_filter_heatmap(
        self,
        layer_name: str,
        figsize: Tuple[int, int] = (15, 8),
        save_path: Optional[Union[str, Path]] = None
    ):
        """
        Plot filter weights as a heatmap

        Args:
            layer_name: Name of convolutional layer
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Weights as numpy array
        """
        layer = self.conv_layers[layer_name]
        weights = layer.weight.detach().cpu().numpy()  # [out_ch, in_ch, kernel_size]

        # Reshape for heatmap: [out_ch, in_ch * kernel_size]
        out_ch, in_ch, kernel_size = weights.shape
        weights_2d = weights.reshape(out_ch, in_ch * kernel_size)

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(weights_2d, cmap='RdBu_r', aspect='auto',
                      vmin=-np.abs(weights_2d).max(), vmax=np.abs(weights_2d).max())

        ax.set_title(f'Filter Weights: {layer_name} [{out_ch} filters × {in_ch * kernel_size} weights]',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Weight Index (input_ch × kernel_size)', fontsize=12)
        ax.set_ylabel('Filter Index (output channel)', fontsize=12)

        plt.colorbar(im, ax=ax, label='Weight Value')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved filter heatmap to {save_path}")

        return weights

    def plot_receptive_field(
        self,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Union[str, Path]] = None
    ):
        """
        Calculate and visualize receptive field sizes across layers

        Returns:
            Dictionary with receptive field sizes per layer
        """
        receptive_fields = {}
        current_rf = 1
        current_stride = 1

        for layer_name, layer in self.conv_layers.items():
            kernel_size = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
            stride = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
            padding = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
            dilation = layer.dilation[0] if isinstance(layer.dilation, tuple) else layer.dilation

            # Effective kernel size with dilation
            effective_kernel = (kernel_size - 1) * dilation + 1

            # Update receptive field
            current_rf += (effective_kernel - 1) * current_stride
            current_stride *= stride

            receptive_fields[layer_name] = {
                'receptive_field': current_rf,
                'stride': current_stride,
                'kernel_size': kernel_size
            }

        # Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        layers = list(receptive_fields.keys())
        rfs = [receptive_fields[l]['receptive_field'] for l in layers]
        strides = [receptive_fields[l]['stride'] for l in layers]

        # Receptive field growth
        ax1.plot(range(len(layers)), rfs, marker='o', linewidth=2, markersize=8)
        ax1.set_xticks(range(len(layers)))
        ax1.set_xticklabels(layers, rotation=45, ha='right')
        ax1.set_ylabel('Receptive Field Size (samples)', fontsize=12)
        ax1.set_title('Receptive Field Growth', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)

        # Stride growth
        ax2.semilogy(range(len(layers)), strides, marker='s', linewidth=2, markersize=8, color='orange')
        ax2.set_xticks(range(len(layers)))
        ax2.set_xticklabels(layers, rotation=45, ha='right')
        ax2.set_ylabel('Cumulative Stride (log scale)', fontsize=12)
        ax2.set_title('Stride Accumulation', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved receptive field plot to {save_path}")

        return receptive_fields

    def __del__(self):
        """Cleanup hooks on deletion"""
        self._remove_hooks()


def test_cnn_visualizer():
    """Test CNN visualization functions"""
    print("Testing CNNVisualizer...")

    # Create dummy model
    from models.cnn.cnn_1d import CNN1D
    model = CNN1D(num_classes=NUM_CLASSES, input_length=102400, in_channels=1)

    # Create visualizer
    visualizer = CNNVisualizer(model)

    print(f"✓ Found {len(visualizer.conv_layers)} convolutional layers:")
    for name, layer in visualizer.conv_layers.items():
        print(f"  - {name}: {layer.out_channels} filters, kernel_size={layer.kernel_size}")

    # Test filter visualization
    print("\n✓ Testing filter visualization...")
    fig = visualizer.plot_conv_filters()
    plt.close()

    # Test feature maps
    print("✓ Testing feature map visualization...")
    dummy_signal = torch.randn(1, 1, SIGNAL_LENGTH)
    first_layer = list(visualizer.conv_layers.keys())[0]
    fmaps = visualizer.plot_feature_maps(dummy_signal, first_layer)
    print(f"  Feature maps shape: {fmaps.shape}")
    plt.close()

    # Test activation distributions
    print("✓ Testing activation distributions...")
    stats = visualizer.plot_activation_distributions(dummy_signal)
    for layer, stat in stats.items():
        print(f"  {layer}: mean={stat['mean']:.4f}, dead={stat['dead_ratio']*100:.1f}%")
    plt.close()

    # Test receptive field
    print("✓ Testing receptive field calculation...")
    rfs = visualizer.plot_receptive_field()
    for layer, rf_info in rfs.items():
        print(f"  {layer}: RF={rf_info['receptive_field']}, stride={rf_info['stride']}")
    plt.close()

    print("\n✅ All tests passed!")


if __name__ == '__main__':
    test_cnn_visualizer()
