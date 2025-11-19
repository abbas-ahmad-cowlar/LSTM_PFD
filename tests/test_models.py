"""
Unit Tests for Model Architectures

Tests:
- Forward pass shapes
- Gradient flow
- Model serialization/deserialization
- Factory functions
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile


class TestCNN1D:
    """Test CNN1D model."""

    def test_forward_pass(self):
        """Test forward pass output shape."""
        from models import CNN1D

        model = CNN1D(num_classes=11, input_channels=1)
        x = torch.randn(4, 1, 5000)  # Batch of 4 signals

        output = model(x)

        assert output.shape == (4, 11), f"Expected shape (4, 11), got {output.shape}"

    def test_gradient_flow(self):
        """Test gradient flow through model."""
        from models import CNN1D

        model = CNN1D(num_classes=11)
        x = torch.randn(2, 1, 5000)
        target = torch.tensor([0, 1])

        output = model(x)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()

        # Check that gradients exist
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_serialization(self):
        """Test model save/load."""
        from models import CNN1D

        model = CNN1D(num_classes=11)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
            torch.save(model.state_dict(), f.name)

            # Load model
            model2 = CNN1D(num_classes=11)
            model2.load_state_dict(torch.load(f.name))

            # Clean up
            Path(f.name).unlink()


class TestResNet1D:
    """Test ResNet1D model."""

    def test_forward_pass(self):
        """Test forward pass output shape."""
        from models import ResNet1D

        model = ResNet1D(num_classes=11, input_channels=1)
        x = torch.randn(4, 1, 5000)

        output = model(x)

        assert output.shape == (4, 11), f"Expected shape (4, 11), got {output.shape}"

    def test_residual_connections(self):
        """Test that residual connections work."""
        from models import ResNet1D

        model = ResNet1D(num_classes=11)
        model.eval()

        x = torch.randn(2, 1, 5000)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 11)


class TestTransformer:
    """Test Transformer model."""

    def test_forward_pass(self):
        """Test forward pass output shape."""
        from models import SignalTransformer

        model = SignalTransformer(
            num_classes=11,
            input_channels=1,
            patch_size=16,
            d_model=128,
            num_layers=2
        )
        x = torch.randn(4, 1, 5000)

        output = model(x)

        assert output.shape == (4, 11), f"Expected shape (4, 11), got {output.shape}"


class TestHybridPINN:
    """Test HybridPINN model."""

    def test_forward_pass_with_physics(self):
        """Test forward pass with physics features."""
        from models import HybridPINN

        model = HybridPINN(num_classes=11, physics_dim=32)

        x = torch.randn(4, 1, 5000)
        physics_features = torch.randn(4, 32)

        output = model(x, physics_features)

        assert output.shape == (4, 11), f"Expected shape (4, 11), got {output.shape}"

    def test_forward_pass_without_physics(self):
        """Test forward pass without physics features."""
        from models import HybridPINN

        model = HybridPINN(num_classes=11, physics_dim=32)
        x = torch.randn(4, 1, 5000)

        output = model(x)  # No physics features

        assert output.shape == (4, 11), f"Expected shape (4, 11), got {output.shape}"


class TestEnsemble:
    """Test Ensemble models."""

    def test_voting_ensemble(self):
        """Test voting ensemble."""
        from models import CNN1D, create_voting_ensemble

        # Create base models
        model1 = CNN1D(num_classes=11)
        model2 = CNN1D(num_classes=11)

        # Create ensemble
        ensemble = create_voting_ensemble(
            models=[model1, model2],
            voting_type='soft',
            num_classes=11
        )

        x = torch.randn(4, 1, 5000)
        output = ensemble(x)

        assert output.shape == (4, 11), f"Expected shape (4, 11), got {output.shape}"


class TestModelFactory:
    """Test model factory functions."""

    def test_create_model(self):
        """Test create_model factory function."""
        from models import create_model

        model = create_model('cnn1d', num_classes=11)

        assert model is not None
        assert hasattr(model, 'forward')

    def test_list_available_models(self):
        """Test listing available models."""
        from models import list_available_models

        models = list_available_models()

        assert len(models) > 0
        assert 'cnn1d' in models

    def test_save_and_load_checkpoint(self):
        """Test checkpoint save/load."""
        from models import create_model, save_checkpoint, load_pretrained

        model = create_model('cnn1d', num_classes=11)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / 'model.pt'

            # Save checkpoint
            save_checkpoint(model, str(checkpoint_path))

            # Load checkpoint
            loaded_model = load_pretrained(
                'cnn1d',
                str(checkpoint_path),
                num_classes=11
            )

            assert loaded_model is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
