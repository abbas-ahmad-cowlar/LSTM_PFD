#!/usr/bin/env python3
"""
Attention Visualization for Transformer Models

Visualizes attention patterns in SignalTransformer, PatchTST, and other
attention-based models to show which signal regions are attended.

Features:
- Extract attention weights from any transformer layer
- Visualize attention heatmaps overlaid on signal
- Head-wise attention analysis
- Aggregate attention rollout

Usage:
    python visualization/attention_viz.py --model checkpoints/patchtst.pth --signal data/sample.npy
    
    # Or programmatically:
    from visualization.attention_viz import AttentionVisualizer
    viz = AttentionVisualizer(model)
    viz.plot_attention(signal, save_path='attention.png')

Author: Deficiency Fix #27 (Priority: 44)
Date: 2026-01-18
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any
from matplotlib.colors import LinearSegmentedColormap


# Custom colormap for attention
ATTENTION_CMAP = LinearSegmentedColormap.from_list(
    'attention', ['white', 'yellow', 'orange', 'red']
)


class AttentionVisualizer:
    """
    Visualize attention patterns in Transformer models.
    
    Supports:
    - SignalTransformer (packages/core/models/transformer/)
    - PatchTST
    - Any model with get_attention_maps() or similar method
    """
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    def get_attention_weights(
        self, 
        x: torch.Tensor, 
        layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Extract attention weights from model.
        
        Args:
            x: Input tensor [1, channels, seq_len]
            layer_idx: Layer index (-1 for last layer)
        
        Returns:
            Attention weights [n_heads, seq_len, seq_len]
        """
        x = x.to(self.device)
        
        # Try different methods based on model type
        if hasattr(self.model, 'get_attention_maps'):
            attn = self.model.get_attention_maps(x)
            return attn[0]  # Remove batch dim
        
        elif hasattr(self.model, 'get_attention_weights'):
            attn = self.model.get_attention_weights(x, layer_idx=layer_idx)
            return attn[0]
        
        elif hasattr(self.model, 'get_all_attention_weights'):
            all_attn = self.model.get_all_attention_weights(x)
            return all_attn[layer_idx][0]
        
        else:
            # Manual extraction via hooks
            return self._extract_via_hooks(x, layer_idx)
    
    def _extract_via_hooks(
        self, 
        x: torch.Tensor, 
        layer_idx: int
    ) -> torch.Tensor:
        """Extract attention using forward hooks."""
        attention_weights = []
        
        def hook_fn(module, input, output):
            # MultiheadAttention returns (attn_output, attn_weights)
            if isinstance(output, tuple) and len(output) >= 2:
                attention_weights.append(output[1])
        
        # Find attention layers
        attn_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.MultiheadAttention):
                attn_layers.append(module)
        
        if not attn_layers:
            raise ValueError("No MultiheadAttention layers found in model")
        
        # Register hook on target layer
        target_layer = attn_layers[layer_idx]
        
        # Temporarily enable need_weights
        original_need_weights = True
        handle = target_layer.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                self.model(x)
        finally:
            handle.remove()
        
        if not attention_weights:
            raise ValueError("No attention weights captured")
        
        return attention_weights[0][0]  # [n_heads, seq, seq]
    
    def attention_rollout(
        self, 
        attention_weights: List[torch.Tensor],
        discard_ratio: float = 0.0
    ) -> torch.Tensor:
        """
        Compute attention rollout across layers.
        
        Follows "Quantifying Attention Flow in Transformers" (Abnar & Zuidema, 2020).
        
        Args:
            attention_weights: List of [n_heads, seq, seq] tensors per layer
            discard_ratio: Fraction of lowest attention to discard
        
        Returns:
            Rolled attention [seq, seq]
        """
        result = torch.eye(attention_weights[0].shape[-1], device=self.device)
        
        for attn in attention_weights:
            # Average across heads
            attn = attn.mean(dim=0)
            
            # Add residual connection
            attn = attn + torch.eye(attn.shape[0], device=self.device)
            attn = attn / attn.sum(dim=-1, keepdim=True)
            
            # Accumulate
            result = torch.matmul(attn, result)
        
        return result
    
    def plot_attention_heatmap(
        self,
        x: torch.Tensor,
        layer_idx: int = -1,
        head_idx: Optional[int] = None,
        title: str = "Attention Weights",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot attention weights as heatmap.
        
        Args:
            x: Input tensor [1, channels, seq_len]
            layer_idx: Transformer layer index
            head_idx: Specific head to visualize (None for average)
            title: Plot title
            save_path: Path to save figure
        
        Returns:
            matplotlib Figure
        """
        attn = self.get_attention_weights(x, layer_idx)
        
        if head_idx is not None:
            attn = attn[head_idx]
            title = f"{title} (Head {head_idx})"
        else:
            attn = attn.mean(dim=0)
            title = f"{title} (All Heads Average)"
        
        attn = attn.cpu().numpy()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(attn, cmap='viridis', aspect='auto')
        ax.set_xlabel('Key Position (Patch)')
        ax.set_ylabel('Query Position (Patch)')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label='Attention Weight')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_signal_attention(
        self,
        signal: np.ndarray,
        x: torch.Tensor,
        layer_idx: int = -1,
        patch_size: Optional[int] = None,
        title: str = "Signal with Attention Overlay",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plot signal with attention weights overlaid.
        
        Shows which parts of the signal the model attends to.
        
        Args:
            signal: Raw signal [seq_len]
            x: Preprocessed input tensor [1, channels, seq_len]
            layer_idx: Transformer layer index
            patch_size: Size of patches (for mapping attention to signal)
            title: Plot title
            save_path: Path to save figure
        """
        attn = self.get_attention_weights(x, layer_idx)
        
        # Average attention to CLS token (first position) across heads
        if attn.dim() == 3:
            # [n_heads, seq, seq] -> attention from CLS to all patches
            cls_attention = attn[:, 0, 1:].mean(dim=0)  # Skip CLS-to-CLS
        else:
            cls_attention = attn[0, 1:]
        
        cls_attention = cls_attention.cpu().numpy()
        
        # Map attention to signal positions
        if patch_size is None:
            # Infer from model if possible
            patch_size = getattr(self.model, 'patch_size', len(signal) // len(cls_attention))
        
        # Create attention overlay for full signal
        signal_attention = np.zeros(len(signal))
        for i, attn_val in enumerate(cls_attention):
            start = i * patch_size
            end = min(start + patch_size, len(signal))
            signal_attention[start:end] = attn_val
        
        # Normalize
        signal_attention = (signal_attention - signal_attention.min()) / (signal_attention.max() - signal_attention.min() + 1e-8)
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Signal with attention overlay
        ax1 = axes[0]
        t = np.arange(len(signal))
        ax1.plot(t, signal, color='steelblue', linewidth=0.5, alpha=0.7)
        
        # Overlay attention as background color
        for i in range(len(cls_attention)):
            start = i * patch_size
            end = min(start + patch_size, len(signal))
            alpha = cls_attention[i] * 0.8
            ax1.axvspan(start, end, alpha=alpha, color='red', linewidth=0)
        
        ax1.set_ylabel('Amplitude')
        ax1.set_title(title)
        
        # Attention values per patch
        ax2 = axes[1]
        patch_centers = np.arange(len(cls_attention)) * patch_size + patch_size // 2
        ax2.bar(patch_centers, cls_attention, width=patch_size * 0.8, color='crimson', alpha=0.7)
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Attention Weight')
        ax2.set_title('Attention to CLS Token by Patch')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    def plot_head_comparison(
        self,
        x: torch.Tensor,
        layer_idx: int = -1,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Plot attention patterns for each head."""
        attn = self.get_attention_weights(x, layer_idx)
        n_heads = attn.shape[0]
        
        # Determine grid size
        cols = min(4, n_heads)
        rows = (n_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
        axes = np.atleast_2d(axes)
        
        for i in range(n_heads):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            head_attn = attn[i].cpu().numpy()
            im = ax.imshow(head_attn, cmap='viridis', aspect='auto')
            ax.set_title(f'Head {i}')
            ax.set_xlabel('Key')
            ax.set_ylabel('Query')
        
        # Hide unused axes
        for i in range(n_heads, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.suptitle(f'Attention Patterns by Head (Layer {layer_idx})', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def demo_attention_visualization():
    """Run demo with synthetic data."""
    print("=" * 60)
    print("ATTENTION VISUALIZATION DEMO")
    print("=" * 60)
    
    from packages.core.models.transformer.patchtst import PatchTST
    
    # Create model
    model = PatchTST(
        num_classes=11,
        input_length=10240,  # Smaller for demo
        patch_size=512,
        d_model=64,
        n_heads=4,
        n_layers=2
    )
    
    print(f"\nModel: PatchTST")
    print(f"  Patches: {model.num_patches}")
    print(f"  Heads: 4")
    
    # Create visualizer
    viz = AttentionVisualizer(model)
    
    # Generate synthetic signal with a fault
    np.random.seed(42)
    t = np.linspace(0, 1, 10240)
    signal = np.sin(2 * np.pi * 50 * t)  # Base frequency
    
    # Add fault impulse in middle section
    fault_start = 4000
    fault_end = 6000
    signal[fault_start:fault_end] += 0.5 * np.sin(2 * np.pi * 200 * t[fault_start:fault_end])
    signal += 0.1 * np.random.randn(len(signal))
    
    # Prepare input
    x = torch.from_numpy(signal.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    
    # Create output directory
    output_dir = Path('results/attention_viz')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Attention heatmap
    fig1 = viz.plot_attention_heatmap(x, title="PatchTST Attention", 
                                       save_path=output_dir / 'heatmap.png')
    plt.close(fig1)
    print("  ✓ Attention heatmap saved")
    
    # 2. Signal with attention overlay
    fig2 = viz.plot_signal_attention(signal, x, patch_size=512,
                                      title="Signal Attention Overlay",
                                      save_path=output_dir / 'signal_attention.png')
    plt.close(fig2)
    print("  ✓ Signal attention overlay saved")
    
    # 3. Head comparison
    fig3 = viz.plot_head_comparison(x, save_path=output_dir / 'heads.png')
    plt.close(fig3)
    print("  ✓ Head comparison saved")
    
    print(f"\n✓ Visualizations saved to: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Visualize Transformer attention')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    parser.add_argument('--signal', type=str, help='Path to signal file (.npy)')
    parser.add_argument('--output', type=str, default='attention.png', help='Output path')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_attention_visualization()
    elif args.model and args.signal:
        # Load model and signal
        checkpoint = torch.load(args.model, map_location='cpu')
        # Would need model class info to instantiate
        print("Model loading from checkpoint requires model class information")
    else:
        print("Run with --demo for demonstration")


if __name__ == '__main__':
    main()
