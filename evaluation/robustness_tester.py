"""
Robustness Testing Suite

Tests model robustness against:
- Sensor noise
- Missing features
- Temporal drift
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
from torch.utils.data import DataLoader


class RobustnessTester:
    """
    Test model robustness to various perturbations.

    Args:
        model: Trained model to test
        device: Device to run tests on
    """
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def test_sensor_noise(
        self,
        dataloader: DataLoader,
        noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2]
    ) -> Dict[float, float]:
        """
        Test robustness to sensor noise.

        Args:
            dataloader: Test data loader
            noise_levels: List of noise standard deviations

        Returns:
            Dictionary mapping noise level to accuracy
        """
        results = {}

        for noise_level in noise_levels:
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    # Add Gaussian noise
                    noisy_inputs = inputs + torch.randn_like(inputs) * noise_level

                    # Evaluate
                    outputs = self.model(noisy_inputs)
                    _, predicted = outputs.max(1)

                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)

            accuracy = 100.0 * correct / total
            results[noise_level] = accuracy

        return results

    def test_missing_features(
        self,
        dataloader: DataLoader,
        dropout_rates: List[float] = [0.1, 0.2, 0.3, 0.5]
    ) -> Dict[float, float]:
        """
        Test robustness to missing/zeroed features.

        Args:
            dataloader: Test data loader
            dropout_rates: List of dropout probabilities

        Returns:
            Dictionary mapping dropout rate to accuracy
        """
        results = {}

        for dropout_rate in dropout_rates:
            correct = 0
            total = 0

            with torch.no_grad():
                for inputs, targets in dataloader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    # Apply dropout (zero out random features)
                    mask = torch.bernoulli(torch.ones_like(inputs) * (1 - dropout_rate))
                    masked_inputs = inputs * mask

                    # Evaluate
                    outputs = self.model(masked_inputs)
                    _, predicted = outputs.max(1)

                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)

            accuracy = 100.0 * correct / total
            results[dropout_rate] = accuracy

        return results
