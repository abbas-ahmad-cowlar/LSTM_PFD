"""
Simple CNN Model for Testing

A lightweight CNN model used across test files. Defined at module level
so it can be properly pickled for ONNX export and other serialization tests.

Author: Syed Abbas Ahmad
Date: 2025-11-23
"""

import torch
import torch.nn as nn
from utils.constants import NUM_CLASSES


class SimpleCNN(nn.Module):
    """
    Simple 1D CNN model for testing.

    Architecture:
    - Conv1d(1, 32, kernel_size=7)
    - MaxPool1d(2)
    - Conv1d(32, 64, kernel_size=5)
    - AdaptiveAvgPool1d(1)
    - Linear(64, num_classes)

    This model is intentionally simple for fast testing.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        """
        Initialize SimpleCNN.

        Args:
            num_classes: Number of output classes (default: 11)
        """
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, 1, signal_length]

        Returns:
            Output tensor [batch_size, num_classes]
        """
        # x: [B, 1, T]
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = torch.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x
