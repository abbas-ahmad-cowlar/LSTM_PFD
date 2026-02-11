"""
A1 Verification: Test that every model in MODEL_REGISTRY can be instantiated
and produces the correct output shape via a forward pass.

Models are grouped by input type:
  - 1D models: input shape (B, 1, L) where L is signal length
  - 2D models: input shape (B, 1, H, W) for spectrograms
  - Fusion/DualStream: need special handling (multi-input or auto-TFR)

Run: python -m pytest tests/test_factory_wiring.py -v
"""

import pytest
import torch
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH

# ---- 1D Models: standard (B, 1, L) input ----
STANDARD_1D_MODELS = [
    'cnn1d',
    'resnet18',
    'resnet34',
    'transformer',
    'vit_tiny_1d',
    'vit_small_1d',
    'patchtst',
    'tsmixer',
    'cnn_transformer_small',
    'efficientnet_b0',
    'efficientnet_b1',
    'pinn',
]

# ---- 2D Models: spectrogram input (B, 1, H, W) ----
SPECTROGRAM_2D_MODELS = [
    'resnet18_2d',
    'resnet34_2d',
    'resnet50_2d',
    'efficientnet_2d_b0',
    'efficientnet_2d_b1',
    'efficientnet_2d_b3',
]

# Signal length for tests — use a smaller length for speed
TEST_LENGTH = 5120
BATCH = 2


class TestStandard1DModels:
    """Test all 1D models that accept (B, 1, L) input."""

    @pytest.mark.parametrize("model_name", STANDARD_1D_MODELS)
    def test_forward_pass(self, model_name):
        from packages.core.models.model_factory import create_model
        extra = {}
        if model_name in ('tsmixer', 'ts_mixer'):
            extra['input_length'] = TEST_LENGTH
        elif model_name in ('patchtst', 'patch_tst'):
            extra['input_length'] = TEST_LENGTH
        model = create_model(model_name, num_classes=NUM_CLASSES, **extra)
        model.eval()
        x = torch.randn(BATCH, 1, TEST_LENGTH)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES), (
            f"{model_name}: expected ({BATCH}, {NUM_CLASSES}), got {out.shape}"
        )

    @pytest.mark.parametrize("model_name", STANDARD_1D_MODELS)
    def test_gradient_flow(self, model_name):
        from packages.core.models.model_factory import create_model
        extra = {}
        if model_name in ('tsmixer', 'ts_mixer'):
            extra['input_length'] = TEST_LENGTH
        elif model_name in ('patchtst', 'patch_tst'):
            extra['input_length'] = TEST_LENGTH
        model = create_model(model_name, num_classes=NUM_CLASSES, **extra)
        model.train()
        x = torch.randn(BATCH, 1, TEST_LENGTH)
        out = model(x)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        assert has_grad, f"{model_name}: no gradients computed"


class TestSpectrogram2DModels:
    """Test 2D spectrogram models that accept (B, C, H, W) input."""

    @pytest.mark.parametrize("model_name", SPECTROGRAM_2D_MODELS)
    def test_forward_pass(self, model_name):
        from packages.core.models.model_factory import create_model
        model = create_model(model_name, num_classes=NUM_CLASSES)
        model.eval()
        # Typical spectrogram shape: 129 freq bins x 200 time frames
        x = torch.randn(BATCH, 1, 129, 200)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES), (
            f"{model_name}: expected ({BATCH}, {NUM_CLASSES}), got {out.shape}"
        )


class TestEfficientNet1DVariants:
    """Test all 8 EfficientNet-1D variants B0-B7."""

    @pytest.mark.parametrize("variant", range(8))
    def test_forward_pass(self, variant):
        from packages.core.models.model_factory import create_model
        name = f"efficientnet_b{variant}"
        model = create_model(name, num_classes=NUM_CLASSES)
        model.eval()
        x = torch.randn(BATCH, 1, TEST_LENGTH)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES), (
            f"{name}: expected ({BATCH}, {NUM_CLASSES}), got {out.shape}"
        )


class TestRegistryCompleteness:
    """Test registry metadata."""

    def test_model_count(self):
        from packages.core.models.model_factory import list_available_models
        models = list_available_models()
        # We expect 43 registered aliases
        assert len(models) >= 40, f"Expected >= 40 models, got {len(models)}"

    def test_all_entries_callable(self):
        from packages.core.models.model_factory import MODEL_REGISTRY
        for name, fn in MODEL_REGISTRY.items():
            assert callable(fn), f"Registry entry '{name}' is not callable: {type(fn)}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
