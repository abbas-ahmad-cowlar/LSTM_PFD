"""
Multi-Scale CNN

Processes signals at multiple resolutions simultaneously using parallel branches.

Key idea:
- Different kernel sizes capture different frequency patterns
- Fine-grained patterns: Small kernels (k=16)
- Coarse-grained patterns: Large kernels (k=128)
- Fuse multi-scale features for robust classification

Architecture:
    Input [B, 1, 102400]
      ↓
    ┌─────────────┬─────────────┬─────────────┐
    Branch 1      Branch 2      Branch 3      (parallel)
    k=16, s=4     k=64, s=4     k=128, s=4
    [B, 64, ...]  [B, 64, ...]  [B, 64, ...]
    └─────────────┴─────────────┴─────────────┘
      ↓ Concatenate
    [B, 192, ...]
      ↓ Fusion convolutions
    [B, 512, ...]
      ↓ Global pooling
    [B, 11]

Reference:
- Szegedy et al. (2015). "Going Deeper with Convolutions" (Inception)
- Multi-scale feature learning for time series
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from models.base_model import BaseModel


class MultiScaleBranch(nn.Module):
    """
    Single branch for multi-scale processing.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Kernel size for this scale
        stride: Stride
        num_layers: Number of conv layers in branch
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 4,
        num_layers: int = 3
    ):
        super().__init__()

        layers = []

        # First conv with specified kernel and stride
        layers.extend([
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        ])

        # Additional layers with smaller kernels
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ])

        self.branch = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through branch."""
        return self.branch(x)


class MultiScaleCNN(BaseModel):
    """
    Multi-scale CNN with parallel processing branches.

    Args:
        num_classes: Number of output classes
        input_channels: Number of input channels
        branch_configs: List of (kernel_size, out_channels) for each branch
        fusion_channels: Channels for fusion layers
        dropout: Dropout probability
    """

    def __init__(
        self,
        num_classes: int = 11,
        input_channels: int = 1,
        branch_configs: Optional[List[tuple]] = None,
        fusion_channels: Optional[List[int]] = None,
        dropout: float = 0.2
    ):
        super().__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.dropout_p = dropout

        # Default branch configurations: (kernel_size, out_channels)
        if branch_configs is None:
            branch_configs = [
                (16, 64),   # Fine-grained (high frequency)
                (64, 64),   # Medium-grained
                (128, 64),  # Coarse-grained (low frequency)
            ]

        # Create parallel branches
        self.branches = nn.ModuleList()
        total_branch_channels = 0

        for kernel_size, out_channels in branch_configs:
            self.branches.append(
                MultiScaleBranch(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=4,
                    num_layers=3
                )
            )
            total_branch_channels += out_channels

        # Fusion layers
        if fusion_channels is None:
            fusion_channels = [256, 512]

        fusion_layers = []
        in_channels = total_branch_channels

        for out_channels in fusion_channels:
            fusion_layers.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ])
            in_channels = out_channels

        self.fusion = nn.Sequential(*fusion_layers)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fusion_channels[-1], num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-scale processing."""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Process through parallel branches
        branch_outputs = []
        for branch in self.branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)

        # Align temporal dimensions if needed
        # Find minimum length
        min_length = min(out.size(2) for out in branch_outputs)

        # Crop all outputs to same length
        aligned_outputs = []
        for out in branch_outputs:
            if out.size(2) > min_length:
                # Center crop
                start = (out.size(2) - min_length) // 2
                out = out[:, :, start:start + min_length]
            aligned_outputs.append(out)

        # Concatenate multi-scale features
        fused = torch.cat(aligned_outputs, dim=1)

        # Fusion layers
        fused = self.fusion(fused)

        # Global pooling
        pooled = self.avgpool(fused)
        pooled = torch.flatten(pooled, 1)

        # Classification
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)

        return logits

    def get_branch_outputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get outputs from each branch (for visualization).

        Args:
            x: Input tensor

        Returns:
            List of branch outputs
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        branch_outputs = []
        for branch in self.branches:
            branch_out = branch(x)
            branch_outputs.append(branch_out)

        return branch_outputs

    def get_config(self) -> dict:
        """Return model configuration."""
        return {
            'model_type': 'MultiScaleCNN',
            'num_classes': self.num_classes,
            'input_channels': self.input_channels,
            'num_branches': len(self.branches),
            'num_parameters': self.get_num_params(),
            'dropout': self.dropout_p
        }


def create_multiscale_cnn(
    num_classes: int = 11,
    num_scales: int = 3,
    **kwargs
) -> MultiScaleCNN:
    """
    Factory function to create multi-scale CNN.

    Args:
        num_classes: Number of output classes
        num_scales: Number of parallel branches (3, 4, or 5)
        **kwargs: Additional arguments

    Returns:
        MultiScaleCNN model instance
    """
    # Define scale configurations based on num_scales
    if num_scales == 3:
        branch_configs = [
            (16, 64),   # Fine
            (64, 64),   # Medium
            (128, 64),  # Coarse
        ]
    elif num_scales == 4:
        branch_configs = [
            (8, 64),    # Very fine
            (32, 64),   # Fine
            (64, 64),   # Medium
            (128, 64),  # Coarse
        ]
    elif num_scales == 5:
        branch_configs = [
            (8, 64),    # Very fine
            (16, 64),   # Fine
            (32, 64),   # Medium-fine
            (64, 64),   # Medium-coarse
            (128, 64),  # Coarse
        ]
    else:
        raise ValueError(f"num_scales must be 3, 4, or 5, got {num_scales}")

    return MultiScaleCNN(
        num_classes=num_classes,
        branch_configs=branch_configs,
        **kwargs
    )


# Test
if __name__ == "__main__":
    print("Testing Multi-Scale CNN...")

    # Test 3-scale
    print("\n3-scale CNN:")
    model = create_multiscale_cnn(num_classes=11, num_scales=3)
    x = torch.randn(2, 1, 102400)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
    print(f"Parameters: {model.get_num_params():,}")
    assert y.shape == (2, 11)

    # Test branch outputs
    branch_outputs = model.get_branch_outputs(x)
    print(f"\nBranch outputs:")
    for i, out in enumerate(branch_outputs):
        print(f"  Branch {i+1}: {out.shape}")

    print("\n✓ Multi-Scale CNN tests passed!")
