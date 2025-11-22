"""
2D CNN Activation Map Visualization

Visualizes what 2D CNNs learn when processing spectrograms:
- Convolutional filter visualization
- Feature map activations at different layers
- Grad-CAM attention maps
- Filter response analysis

Usage:
    from visualization.activation_maps_2d import visualize_filters, visualize_feature_maps

    # Visualize learned filters
    visualize_filters(model, layer_name='layer1', save_path='filters.png')

    # Visualize feature maps
    visualize_feature_maps(
        model, spectrogram, layer_name='layer2',
        save_path='feature_maps.png'
    )

    # Generate Grad-CAM
    grad_cam_heatmap = generate_grad_cam(
        model, spectrogram, target_class=3
    )
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Tuple, Dict
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def visualize_filters(
    model: nn.Module,
    layer_name: str,
    num_filters: int = 64,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 16)
):
    """
    Visualize learned convolutional filters.

    Args:
        model: Trained 2D CNN model
        layer_name: Name of convolutional layer to visualize
        num_filters: Number of filters to display (default: 64)
        save_path: Optional path to save figure
        figsize: Figure size
    """
    # Get layer
    try:
        layer = dict(model.named_modules())[layer_name]
    except KeyError:
        raise ValueError(f"Layer '{layer_name}' not found in model")

    if not isinstance(layer, nn.Conv2d):
        raise ValueError(f"Layer '{layer_name}' is not a Conv2d layer")

    # Get filters
    filters = layer.weight.data.cpu().numpy()  # [out_channels, in_channels, H, W]

    num_filters = min(num_filters, filters.shape[0])
    ncols = 8
    nrows = (num_filters + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i in range(num_filters):
        ax = axes[i]

        # Get filter (average over input channels for visualization)
        filt = filters[i].mean(axis=0)  # [H, W]

        # Normalize for visualization
        filt_norm = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)

        ax.imshow(filt_norm, cmap='viridis', interpolation='nearest')
        ax.set_title(f'Filter {i}', fontsize=8)
        ax.axis('off')

    # Hide unused subplots
    for i in range(num_filters, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Learned Filters in Layer: {layer_name}', fontsize=14, weight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved filter visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_feature_maps(
    model: nn.Module,
    spectrogram: torch.Tensor,
    layer_name: str,
    num_maps: int = 32,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    Visualize feature map activations.

    Args:
        model: Trained 2D CNN model
        spectrogram: Input spectrogram [1, 1, H, W]
        layer_name: Name of layer to visualize
        num_maps: Number of feature maps to display
        save_path: Optional path to save figure
        figsize: Figure size
    """
    model.eval()

    # Hook to capture activations
    activations = {}

    def hook_fn(module, input, output):
        activations['value'] = output

    # Register hook
    try:
        layer = dict(model.named_modules())[layer_name]
        handle = layer.register_forward_hook(hook_fn)
    except KeyError:
        raise ValueError(f"Layer '{layer_name}' not found in model")

    # Forward pass
    with torch.no_grad():
        _ = model(spectrogram)

    # Get activations
    feature_maps = activations['value'][0].cpu().numpy()  # [C, H, W]

    # Remove hook
    handle.remove()

    # Visualize
    num_maps = min(num_maps, feature_maps.shape[0])
    ncols = 8
    nrows = (num_maps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i in range(num_maps):
        ax = axes[i]

        feat = feature_maps[i]  # [H, W]

        # Normalize
        feat_norm = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)

        ax.imshow(feat_norm, cmap='hot', aspect='auto', interpolation='bilinear')
        ax.set_title(f'Map {i}', fontsize=8)
        ax.axis('off')

    # Hide unused subplots
    for i in range(num_maps, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Feature Maps from Layer: {layer_name}', fontsize=14, weight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved feature maps to {save_path}")
    else:
        plt.show()

    plt.close()


def generate_grad_cam(
    model: nn.Module,
    spectrogram: torch.Tensor,
    target_layer: str = 'layer4',
    target_class: Optional[int] = None,
    device: str = 'cuda'
) -> Tuple[np.ndarray, int]:
    """
    Generate Grad-CAM heatmap.

    Args:
        model: Trained 2D CNN model
        spectrogram: Input spectrogram [1, 1, H, W]
        target_layer: Layer to compute Grad-CAM for
        target_class: Target class (None = predicted class)
        device: Device to run on

    Returns:
        Tuple of (heatmap [H, W], predicted_class)
    """
    model.eval()
    model.to(device)

    spectrogram = spectrogram.to(device)
    spectrogram.requires_grad = True

    # Forward pass
    outputs = model(spectrogram)
    _, predicted = outputs.max(1)
    pred_class = predicted.item()

    if target_class is None:
        target_class = pred_class

    # Storage for activations and gradients
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    # Register hooks
    try:
        target_module = dict(model.named_modules())[target_layer]
        forward_handle = target_module.register_forward_hook(forward_hook)
        backward_handle = target_module.register_full_backward_hook(backward_hook)
    except KeyError:
        raise ValueError(f"Layer '{target_layer}' not found in model")

    # Backward pass
    model.zero_grad()
    outputs[0, target_class].backward()

    # Compute Grad-CAM
    pooled_gradients = torch.mean(gradients['value'], dim=[0, 2, 3])
    activation = activations['value'][0]

    for i in range(activation.shape[0]):
        activation[i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activation, dim=0).cpu().detach().numpy()

    # Normalize heatmap
    heatmap = np.maximum(heatmap, 0)  # ReLU
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Resize to input size
    from scipy.ndimage import zoom
    H, W = spectrogram.shape[2:]
    if heatmap.shape[0] != H or heatmap.shape[1] != W:
        heatmap = zoom(heatmap, (H / heatmap.shape[0], W / heatmap.shape[1]))

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    return heatmap, pred_class


def visualize_grad_cam(
    model: nn.Module,
    spectrogram: torch.Tensor,
    target_class: Optional[int] = None,
    class_name: Optional[str] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
    device: str = 'cuda'
):
    """
    Visualize Grad-CAM heatmap overlaid on spectrogram.

    Args:
        model: Trained 2D CNN model
        spectrogram: Input spectrogram [1, 1, H, W]
        target_class: Target class for Grad-CAM
        class_name: Name of predicted class
        save_path: Optional path to save figure
        figsize: Figure size
        device: Device to run on
    """
    # Generate Grad-CAM
    heatmap, pred_class = generate_grad_cam(
        model, spectrogram, target_class=target_class, device=device
    )

    spec_np = spectrogram[0, 0].cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1. Original spectrogram
    ax = axes[0]
    im1 = ax.imshow(spec_np, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title('Original Spectrogram', fontsize=12, weight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    # 2. Grad-CAM heatmap
    ax = axes[1]
    im2 = ax.imshow(heatmap, aspect='auto', origin='lower', cmap='jet')
    title = f'Grad-CAM'
    if class_name:
        title += f'\n(Predicted: {class_name})'
    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # 3. Overlay
    ax = axes[2]
    ax.imshow(spec_np, aspect='auto', origin='lower', cmap='gray', alpha=0.7)
    im3 = ax.imshow(heatmap, aspect='auto', origin='lower', cmap='jet', alpha=0.5)
    ax.set_title('Overlay', fontsize=12, weight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Grad-CAM visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def analyze_layer_responses(
    model: nn.Module,
    spectrogram: torch.Tensor,
    layers: List[str],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 10)
):
    """
    Analyze and visualize responses across multiple layers.

    Args:
        model: Trained 2D CNN model
        spectrogram: Input spectrogram [1, 1, H, W]
        layers: List of layer names to analyze
        save_path: Optional path to save figure
        figsize: Figure size
    """
    model.eval()

    # Collect activations
    activations = {}

    def make_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks
    handles = []
    for layer_name in layers:
        try:
            layer = dict(model.named_modules())[layer_name]
            handle = layer.register_forward_hook(make_hook(layer_name))
            handles.append(handle)
        except KeyError:
            print(f"Warning: Layer '{layer_name}' not found, skipping")

    # Forward pass
    with torch.no_grad():
        _ = model(spectrogram)

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Visualize
    num_layers = len([k for k in activations.keys()])
    fig, axes = plt.subplots(num_layers, 2, figsize=figsize)

    if num_layers == 1:
        axes = axes.reshape(1, -1)

    for i, (layer_name, activation) in enumerate(activations.items()):
        feat = activation[0].cpu().numpy()  # [C, H, W]

        # Average feature map
        avg_feat = feat.mean(axis=0)

        # Max activation
        max_feat = feat.max(axis=0)

        # Plot average
        ax = axes[i, 0]
        im = ax.imshow(avg_feat, aspect='auto', origin='lower', cmap='hot')
        ax.set_title(f'{layer_name} - Average Activation', fontsize=10, weight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Plot max
        ax = axes[i, 1]
        im = ax.imshow(max_feat, aspect='auto', origin='lower', cmap='hot')
        ax.set_title(f'{layer_name} - Max Activation', fontsize=10, weight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved layer response analysis to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_filter_responses_to_frequency(
    model: nn.Module,
    layer_name: str,
    spectrograms: torch.Tensor,
    freq_bins: int = 129,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Analyze how filters respond to different frequency ranges.

    Args:
        model: Trained 2D CNN model
        layer_name: Layer to analyze
        spectrograms: Multiple spectrograms [B, 1, H, W]
        freq_bins: Number of frequency bins
        save_path: Optional path to save figure
        figsize: Figure size
    """
    model.eval()

    # Collect activations
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach())

    # Register hook
    try:
        layer = dict(model.named_modules())[layer_name]
        handle = layer.register_forward_hook(hook_fn)
    except KeyError:
        raise ValueError(f"Layer '{layer_name}' not found in model")

    # Forward pass
    with torch.no_grad():
        _ = model(spectrograms)

    handle.remove()

    # Analyze frequency responses
    feat = torch.cat(activations, dim=0).cpu().numpy()  # [B, C, H, W]

    # Average over batch and time
    freq_response = feat.mean(axis=(0, 3))  # [C, H]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        freq_response.T,
        aspect='auto',
        origin='lower',
        cmap='hot',
        interpolation='bilinear'
    )

    ax.set_xlabel('Filter Index', fontsize=12)
    ax.set_ylabel('Frequency Bin', fontsize=12)
    ax.set_title(f'Filter Responses Across Frequencies - {layer_name}', fontsize=14, weight='bold')
    plt.colorbar(im, ax=ax, label='Average Activation')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved filter frequency response to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    # Test visualization
    from models.spectrogram_cnn import resnet18_2d

    print("Testing 2D CNN activation visualization...")

    # Create model
    model = resnet18_2d(num_classes=NUM_CLASSES)

    # Create dummy spectrogram
    spectrogram = torch.randn(1, 1, 129, 400)

    # Test filter visualization
    print("\n1. Visualizing filters...")
    visualize_filters(model, layer_name='conv1', num_filters=64)

    # Test feature map visualization
    print("\n2. Visualizing feature maps...")
    visualize_feature_maps(model, spectrogram, layer_name='layer1', num_maps=32)

    # Test Grad-CAM
    print("\n3. Visualizing Grad-CAM...")
    visualize_grad_cam(model, spectrogram, class_name='Normal', device='cpu')

    # Test layer response analysis
    print("\n4. Analyzing layer responses...")
    analyze_layer_responses(
        model, spectrogram,
        layers=['layer1', 'layer2', 'layer3', 'layer4']
    )

    print("\nAll visualization tests complete!")
