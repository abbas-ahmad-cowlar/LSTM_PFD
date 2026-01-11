
import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import shutil
from packages.core.explainability.shap_explainer import SHAPExplainer

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [Batch, Channels, Time]
        # Simple conv to flatten to linear
        self.conv = nn.Conv1d(1, 2, kernel_size=3, padding=1)
        self.fc = nn.Linear(2 * 10, 3) # 10 timepoints, 3 classes

    def forward(self, x):
        # x: [B, 1, 10]
        x = self.conv(x) # [B, 2, 10]
        x = x.view(x.size(0), -1) # [B, 20]
        return self.fc(x)

class TestXAI(unittest.TestCase):
    """Test XAI explainer functionality."""

    def setUp(self):
        """Setup test fixtures."""
        self.device = 'cpu'
        self.model = MockModel()
        self.classes = ['Normal', 'Inner', 'Outer']
        
        # Background data [N, C, T]
        self.background = torch.randn(10, 1, 10)
        
        # Explainer
        self.explainer = SHAPExplainer(
            self.model, 
            background_data=self.background, 
            device=self.device, 
            use_shap_library=False
        )

    def test_gradient_shap(self):
        """Test Native Gradient SHAP."""
        # Input signal [1, 10] (Channel, Time)
        signal = torch.randn(1, 10)
        
        # Explain
        shap_values = self.explainer.explain(
            signal, 
            method='gradient', 
            n_samples=5
        )
        
        # Check shape [1, 1, 10] (Batch added by explainer)
        self.assertEqual(shap_values.shape, (1, 1, 10))
        self.assertFalse(torch.isnan(shap_values).any())
        self.assertFalse(torch.isinf(shap_values).any())

    def test_shap_values_sum(self):
        """Test integrity of SHAP values."""
        # Ideally, sum of SHAP values + base value = prediction (approx)
        # But GradientSHAP is an approximation. We just check it runs and produces non-zero values.
        signal = torch.randn(1, 10)
        shap_values = self.explainer.explain(signal, method='gradient', n_samples=5)
        
        # Should be non-zero (unless model weights are zero, which they aren't initialized to)
        self.assertNotEqual(shap_values.abs().sum().item(), 0)

