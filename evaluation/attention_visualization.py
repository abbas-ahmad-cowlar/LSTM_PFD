"""
Attention Visualization for Transformer Models

Provides tools to visualize and analyze attention patterns in transformer models
for bearing fault diagnosis. This helps understand which time regions the model
focuses on when making predictions.

Key functions:
- plot_attention_heatmap: Visualize attention weights as a heatmap
- plot_signal_with_attention: Overlay attention importance on the signal
- attention_rollout: Aggregate attention across all layers
- find_most_attended_patches: Identify critical time regions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
from pathlib import Path


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    patch_size: int = 512,
    head_idx: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    cmap: str = 'viridis',
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot attention weights as a heatmap.

    Args:
        attention_weights: Attention weights tensor of shape [B, n_heads, n_patches, n_patches]
                          or [n_heads, n_patches, n_patches] or [n_patches, n_patches]
        patch_size: Size of each patch in samples
        head_idx: If provided, plot only this attention head. If None, average over all heads.
        figsize: Figure size (width, height)
        save_path: If provided, save figure to this path
        cmap: Colormap name
        title: Custom title for the plot

    Returns:
        Matplotlib figure

    Example:
        >>> model = SignalTransformer()
        >>> signal = torch.randn(1, 1, 102400)
        >>> attention = model.get_attention_weights(signal, layer_idx=-1)
        >>> fig = plot_attention_heatmap(attention, patch_size=512)
        >>> plt.show()
    """
    # Handle different tensor shapes
    if attention_weights.dim() == 4:
        attention_weights = attention_weights[0]  # Remove batch dimension
    if attention_weights.dim() == 3:
        if head_idx is not None:
            attention_weights = attention_weights[head_idx]
        else:
            # Average over heads
            attention_weights = attention_weights.mean(dim=0)

    # Convert to numpy
    attention_np = attention_weights.detach().cpu().numpy()
    n_patches = attention_np.shape[0]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(attention_np, cmap=cmap, aspect='auto', interpolation='nearest')

    # Set labels
    ax.set_xlabel('Key Patch Index', fontsize=12)
    ax.set_ylabel('Query Patch Index', fontsize=12)

    # Add patch time ranges as labels
    if n_patches <= 20:  # Only show labels for small number of patches
        patch_labels = [f'{i*patch_size}-{(i+1)*patch_size}' for i in range(n_patches)]
        ax.set_xticks(range(n_patches))
        ax.set_yticks(range(n_patches))
        ax.set_xticklabels(patch_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(patch_labels, fontsize=8)

    # Set title
    if title is None:
        if head_idx is not None:
            title = f'Attention Heatmap (Head {head_idx})'
        else:
            title = 'Attention Heatmap (Averaged over Heads)'
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention heatmap to {save_path}")

    return fig


def plot_signal_with_attention(
    signal: torch.Tensor,
    attention_weights: torch.Tensor,
    patch_size: int = 512,
    true_label: Optional[int] = None,
    predicted_label: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 8),
    save_path: Optional[str] = None,
    label_names: Optional[List[str]] = None
) -> plt.Figure:
    """
    Plot signal with attention importance overlay.

    Args:
        signal: Signal tensor of shape [1, 1, T] or [1, T] or [T]
        attention_weights: Attention weights of shape [B, n_heads, n_patches, n_patches]
        patch_size: Size of each patch in samples
        true_label: True fault label (optional)
        predicted_label: Predicted fault label (optional)
        figsize: Figure size
        save_path: If provided, save figure to this path
        label_names: List of label names for display

    Returns:
        Matplotlib figure

    Example:
        >>> signal = torch.randn(1, 1, 102400)
        >>> attention = model.get_attention_weights(signal, layer_idx=-1)
        >>> fig = plot_signal_with_attention(signal, attention, true_label=0, predicted_label=0)
    """
    # Prepare signal
    if signal.dim() > 1:
        signal = signal.squeeze()
    signal_np = signal.detach().cpu().numpy()

    # Prepare attention weights
    if attention_weights.dim() == 4:
        attention_weights = attention_weights[0]  # Remove batch dim
    if attention_weights.dim() == 3:
        # Average over heads
        attention_weights = attention_weights.mean(dim=0)

    attention_np = attention_weights.detach().cpu().numpy()
    n_patches = attention_np.shape[0]

    # Compute attention importance (how much attention each patch receives)
    attention_importance = attention_np.mean(axis=0)  # Average over queries

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Plot 1: Signal with attention overlay
    axes[0].plot(signal_np, alpha=0.7, linewidth=0.5, label='Signal')

    # Overlay attention importance as colored spans
    for i in range(n_patches):
        start_idx = i * patch_size
        end_idx = min((i + 1) * patch_size, len(signal_np))
        alpha_val = min(0.7, max(0.1, attention_importance[i]))
        axes[0].axvspan(start_idx, end_idx, alpha=alpha_val, color='red')

    # Title with labels if provided
    title = 'Signal with Attention Importance Overlay'
    if true_label is not None or predicted_label is not None:
        if label_names:
            true_str = label_names[true_label] if true_label is not None else '?'
            pred_str = label_names[predicted_label] if predicted_label is not None else '?'
        else:
            true_str = str(true_label) if true_label is not None else '?'
            pred_str = str(predicted_label) if predicted_label is not None else '?'
        title += f'\nTrue: {true_str}, Predicted: {pred_str}'

    axes[0].set_title(title, fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Sample Index', fontsize=10)
    axes[0].set_ylabel('Amplitude', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Attention importance per patch
    patch_indices = np.arange(n_patches)
    axes[1].bar(patch_indices, attention_importance, color='steelblue', alpha=0.7)
    axes[1].set_xlabel('Patch Index', fontsize=10)
    axes[1].set_ylabel('Attention Importance', fontsize=10)
    axes[1].set_title('Attention Importance by Patch', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')

    # Plot 3: Attention heatmap
    im = axes[2].imshow(attention_np, cmap='viridis', aspect='auto', interpolation='nearest')
    axes[2].set_xlabel('Key Patch', fontsize=10)
    axes[2].set_ylabel('Query Patch', fontsize=10)
    axes[2].set_title('Attention Heatmap', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=axes[2])
    cbar.set_label('Attention Weight', rotation=270, labelpad=15, fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved signal with attention to {save_path}")

    return fig


def attention_rollout(
    all_attention_weights: List[torch.Tensor],
    discard_ratio: float = 0.1
) -> torch.Tensor:
    """
    Compute attention rollout across all transformer layers.

    Attention rollout recursively multiplies attention matrices from all layers
    to get the effective attention from input to output.

    Args:
        all_attention_weights: List of attention tensors, one per layer.
                              Each tensor has shape [B, n_heads, n_patches, n_patches]
        discard_ratio: Ratio of lowest attention weights to discard (helps focus on important paths)

    Returns:
        Rolled out attention of shape [n_patches, n_patches]

    Reference:
        "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, 2020)

    Example:
        >>> all_attentions = model.get_all_attention_weights(signal)
        >>> rollout = attention_rollout(all_attentions)
        >>> plot_attention_heatmap(rollout)
    """
    # Average over heads and remove batch dimension
    attention_matrices = []
    for attn in all_attention_weights:
        if attn.dim() == 4:
            attn = attn[0]  # Remove batch
        if attn.dim() == 3:
            attn = attn.mean(dim=0)  # Average over heads
        attention_matrices.append(attn)

    # Start with identity matrix
    n_patches = attention_matrices[0].shape[0]
    rollout = torch.eye(n_patches, device=attention_matrices[0].device)

    # Add residual connection (identity) to each attention matrix
    for attn in attention_matrices:
        # Add identity (residual connection)
        attn_with_residual = attn + torch.eye(n_patches, device=attn.device)

        # Renormalize
        attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)

        # Optional: Discard low attention weights
        if discard_ratio > 0:
            flat_attn = attn_with_residual.view(-1)
            threshold = torch.quantile(flat_attn, discard_ratio)
            attn_with_residual = torch.where(
                attn_with_residual < threshold,
                torch.zeros_like(attn_with_residual),
                attn_with_residual
            )
            attn_with_residual = attn_with_residual / attn_with_residual.sum(dim=-1, keepdim=True)

        # Multiply with previous rollout
        rollout = torch.matmul(attn_with_residual, rollout)

    return rollout


def find_most_attended_patches(
    attention_weights: torch.Tensor,
    top_k: int = 10
) -> np.ndarray:
    """
    Find the patches that receive the most attention.

    Args:
        attention_weights: Attention tensor [B, n_heads, n_patches, n_patches] or [n_patches, n_patches]
        top_k: Number of top patches to return

    Returns:
        Array of patch indices sorted by importance (descending)

    Example:
        >>> attention = model.get_attention_weights(signal, layer_idx=-1)
        >>> top_patches = find_most_attended_patches(attention, top_k=5)
        >>> print(f"Most important patches: {top_patches}")
    """
    # Average over dimensions
    if attention_weights.dim() == 4:
        attention_weights = attention_weights[0]  # Remove batch
    if attention_weights.dim() == 3:
        attention_weights = attention_weights.mean(dim=0)  # Average over heads

    # Compute importance (average attention received)
    importance = attention_weights.mean(dim=0).detach().cpu().numpy()

    # Get top-k patches
    top_indices = np.argsort(importance)[::-1][:top_k]

    return top_indices


def compare_attention_heads(
    attention_weights: torch.Tensor,
    patch_size: int = 512,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare attention patterns across different attention heads.

    Args:
        attention_weights: Attention tensor [B, n_heads, n_patches, n_patches]
        patch_size: Size of each patch
        figsize: Figure size
        save_path: If provided, save figure to this path

    Returns:
        Matplotlib figure showing all attention heads

    Example:
        >>> attention = model.get_attention_weights(signal, layer_idx=-1)
        >>> fig = compare_attention_heads(attention)
    """
    if attention_weights.dim() == 4:
        attention_weights = attention_weights[0]  # Remove batch

    n_heads = attention_weights.shape[0]

    # Create subplot grid
    n_cols = min(4, n_heads)
    n_rows = (n_heads + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_heads == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for head_idx in range(n_heads):
        attn = attention_weights[head_idx].detach().cpu().numpy()

        im = axes[head_idx].imshow(attn, cmap='viridis', aspect='auto')
        axes[head_idx].set_title(f'Head {head_idx}', fontsize=10)
        axes[head_idx].set_xlabel('Key', fontsize=8)
        axes[head_idx].set_ylabel('Query', fontsize=8)
        plt.colorbar(im, ax=axes[head_idx], fraction=0.046, pad=0.04)

    # Hide unused subplots
    for idx in range(n_heads, len(axes)):
        axes[idx].axis('off')

    fig.suptitle('Attention Patterns Across Heads', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved head comparison to {save_path}")

    return fig


def analyze_attention_entropy(
    attention_weights: torch.Tensor
) -> dict:
    """
    Compute attention entropy to measure how focused the attention is.

    Lower entropy = more focused attention on specific patches
    Higher entropy = more distributed attention across patches

    Args:
        attention_weights: Attention tensor [B, n_heads, n_patches, n_patches]

    Returns:
        Dictionary with entropy statistics

    Example:
        >>> attention = model.get_attention_weights(signal, layer_idx=-1)
        >>> entropy_stats = analyze_attention_entropy(attention)
        >>> print(f"Average entropy: {entropy_stats['mean_entropy']:.4f}")
    """
    if attention_weights.dim() == 4:
        attention_weights = attention_weights[0]  # Remove batch

    attention_np = attention_weights.detach().cpu().numpy()

    # Compute entropy for each query position
    epsilon = 1e-10
    entropy = -np.sum(attention_np * np.log(attention_np + epsilon), axis=-1)

    return {
        'mean_entropy': float(np.mean(entropy)),
        'std_entropy': float(np.std(entropy)),
        'min_entropy': float(np.min(entropy)),
        'max_entropy': float(np.max(entropy)),
        'entropy_per_head': entropy.mean(axis=-1).tolist()  # Average per head
    }


if __name__ == '__main__':
    # Example usage and testing
    print("Testing attention visualization utilities...")

    # Create dummy attention weights
    batch_size = 1
    n_heads = 8
    n_patches = 200
    attention = torch.softmax(torch.randn(batch_size, n_heads, n_patches, n_patches), dim=-1)

    # Test heatmap plotting
    print("\n1. Testing attention heatmap...")
    fig = plot_attention_heatmap(attention, patch_size=512, save_path='test_attention_heatmap.png')
    plt.close(fig)

    # Test signal with attention
    print("\n2. Testing signal with attention overlay...")
    signal = torch.randn(1, 1, 102400)
    fig = plot_signal_with_attention(
        signal, attention, patch_size=512,
        true_label=0, predicted_label=0,
        save_path='test_signal_attention.png'
    )
    plt.close(fig)

    # Test finding most attended patches
    print("\n3. Testing most attended patches...")
    top_patches = find_most_attended_patches(attention, top_k=10)
    print(f"Top 10 patches: {top_patches}")

    # Test attention entropy
    print("\n4. Testing attention entropy analysis...")
    entropy_stats = analyze_attention_entropy(attention)
    print(f"Entropy stats: {entropy_stats}")

    # Test head comparison
    print("\n5. Testing attention head comparison...")
    fig = compare_attention_heads(attention, save_path='test_head_comparison.png')
    plt.close(fig)

    # Test attention rollout
    print("\n6. Testing attention rollout...")
    all_attentions = [torch.softmax(torch.randn(1, n_heads, n_patches, n_patches), dim=-1) for _ in range(6)]
    rollout = attention_rollout(all_attentions)
    print(f"Rollout shape: {rollout.shape}")

    print("\nAll tests passed! ✓")
