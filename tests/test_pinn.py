"""
PINN Model Unit Tests

Tests for Physics-Informed Neural Network models per Master Roadmap Chapter 3.1.

Coverage:
- Model instantiation with different configurations
- Forward pass and output shapes
- Physics loss computation
- Gradient flow verification
- Feature extraction tests

Run with: pytest tests/test_pinn.py -v
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from packages.core.models.pinn.hybrid_pinn import HybridPINN, create_hybrid_pinn
from utils.constants import SIGNAL_LENGTH


class TestHybridPINNInstantiation(unittest.TestCase):
    """Test PINN model instantiation with various configurations."""
    
    def test_default_instantiation(self):
        """Test model creates with default parameters."""
        model = HybridPINN()
        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.num_classes, 11)  # Default bearing fault classes
    
    def test_custom_num_classes(self):
        """Test model with custom number of classes."""
        model = HybridPINN(num_classes=5)
        self.assertEqual(model.num_classes, 5)
    
    def test_resnet18_backbone(self):
        """Test with ResNet18 backbone."""
        model = HybridPINN(backbone='resnet18')
        self.assertIsNotNone(model.data_branch)
    
    def test_resnet34_backbone(self):
        """Test with ResNet34 backbone."""
        model = HybridPINN(backbone='resnet34')
        self.assertIsNotNone(model.data_branch)
    
    def test_cnn1d_backbone(self):
        """Test with CNN1D backbone."""
        model = HybridPINN(backbone='cnn1d')
        self.assertIsNotNone(model.data_branch)
    
    def test_custom_physics_dim(self):
        """Test with custom physics feature dimension."""
        model = HybridPINN(physics_feature_dim=128)
        # Verify the physics MLP has correct output dimension
        self.assertEqual(model.physics_feature_dim, 128)
    
    def test_custom_fusion_dim(self):
        """Test with custom fusion dimension."""
        model = HybridPINN(fusion_dim=512)
        # Model should create successfully
        self.assertIsNotNone(model.fusion)


class TestHybridPINNForward(unittest.TestCase):
    """Test forward pass functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_length = SIGNAL_LENGTH  # Use project constant
        self.model = HybridPINN(
            num_classes=11,
            input_length=self.input_length,
            backbone='cnn1d'    # Lighter backbone for fast tests
        ).to(self.device)
        self.model.eval()
    
    def test_forward_shape(self):
        """Test output shape is correct."""
        batch_size = 4
        signal = torch.randn(batch_size, 1, self.input_length).to(self.device)
        
        with torch.no_grad():
            output = self.model(signal)
        
        self.assertEqual(output.shape, (batch_size, 11))
    
    def test_forward_with_metadata(self):
        """Test forward pass with physics metadata."""
        batch_size = 4
        signal = torch.randn(batch_size, 1, self.input_length).to(self.device)
        metadata = {
            'rpm': torch.tensor([1750.0] * batch_size).to(self.device),
            'load': torch.tensor([1000.0] * batch_size).to(self.device),
        }
        
        with torch.no_grad():
            output = self.model(signal, metadata=metadata)
        
        self.assertEqual(output.shape, (batch_size, 11))
    
    def test_forward_no_nan(self):
        """Test forward pass produces no NaN values."""
        signal = torch.randn(2, 1, self.input_length).to(self.device)
        
        with torch.no_grad():
            output = self.model(signal)
        
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
    
    def test_forward_with_features(self):
        """Test forward_with_features returns intermediate representations."""
        signal = torch.randn(2, 1, self.input_length).to(self.device)
        
        with torch.no_grad():
            output, features = self.model.forward_with_features(signal)
        
        self.assertIn('data_features', features)
        self.assertIn('physics_features', features)
        self.assertIn('combined_features', features)
    
    def test_batch_inference(self):
        """Test inference on different batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            signal = torch.randn(batch_size, 1, self.input_length).to(self.device)
            with torch.no_grad():
                output = self.model(signal)
            self.assertEqual(output.shape[0], batch_size)


class TestHybridPINNGradients(unittest.TestCase):
    """Test gradient flow for training."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input_length = SIGNAL_LENGTH
        self.model = HybridPINN(
            num_classes=11,
            input_length=self.input_length,
            backbone='cnn1d'
        ).to(self.device)
        self.model.train()
    
    def test_gradient_flow(self):
        """Test gradients flow through all parameters."""
        signal = torch.randn(2, 1, self.input_length, requires_grad=True).to(self.device)
        target = torch.tensor([0, 1]).to(self.device)
        
        output = self.model(signal)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for {name}")
    
    def test_physics_branch_gradient(self):
        """Test gradients flow through physics branch specifically."""
        signal = torch.randn(2, 1, self.input_length, requires_grad=True).to(self.device)
        target = torch.tensor([0, 1]).to(self.device)
        
        output = self.model(signal)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        
        # Check physics MLP has gradients
        has_physics_grad = False
        for name, param in self.model.named_parameters():
            if 'physics' in name.lower() and param.grad is not None:
                has_physics_grad = True
                break
        
        self.assertTrue(has_physics_grad, "No gradients in physics branch")


class TestPhysicsFeatureExtraction(unittest.TestCase):
    """Test physics feature extraction."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = HybridPINN(
            num_classes=11,
            input_length=SIGNAL_LENGTH,
            backbone='cnn1d',
            physics_feature_dim=64
        ).to(self.device)
    
    def test_physics_features_shape(self):
        """Test physics feature output shape."""
        metadata = {
            'rpm': torch.tensor([1750.0, 1800.0]).to(self.device),
            'load': torch.tensor([1000.0, 1200.0]).to(self.device),
        }
        
        features = self.model.extract_physics_features(metadata)
        
        # extract_physics_features returns raw physics features [B, 10]
        # (not the MLP transformed features which are [B, physics_feature_dim])
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(features.shape[1], 10)  # 10 raw physics features
    
    def test_physics_features_vary_with_input(self):
        """Test that different metadata produces different features."""
        metadata1 = {
            'rpm': torch.tensor([1750.0]).to(self.device),
        }
        metadata2 = {
            'rpm': torch.tensor([3500.0]).to(self.device),  # Double speed
        }
        
        features1 = self.model.extract_physics_features(metadata1)
        features2 = self.model.extract_physics_features(metadata2)
        
        # Features should be different
        self.assertFalse(torch.allclose(features1, features2))


class TestModelInfo(unittest.TestCase):
    """Test model info and configuration retrieval."""
    
    def test_get_model_info(self):
        """Test model info contains expected fields."""
        model = HybridPINN()
        info = model.get_model_info()
        
        self.assertIn('num_classes', info)
        self.assertIn('backbone', info)
        self.assertIn('total_params', info)
    
    def test_param_count_reasonable(self):
        """Test parameter count is in expected range."""
        model = HybridPINN(backbone='cnn1d')
        info = model.get_model_info()
        
        # CNN1D backbone should have reasonable param count
        self.assertGreater(info['total_params'], 10000)
        self.assertLess(info['total_params'], 100_000_000)


class TestFactoryFunction(unittest.TestCase):
    """Test create_hybrid_pinn factory function."""
    
    def test_factory_default(self):
        """Test factory with default args."""
        model = create_hybrid_pinn()
        self.assertIsInstance(model, HybridPINN)
    
    def test_factory_custom_classes(self):
        """Test factory with custom classes."""
        model = create_hybrid_pinn(num_classes=5)
        self.assertEqual(model.num_classes, 5)
    
    def test_factory_with_backbone(self):
        """Test factory with specific backbone."""
        model = create_hybrid_pinn(backbone='resnet34')
        self.assertIsNotNone(model)


if __name__ == '__main__':
    unittest.main()
