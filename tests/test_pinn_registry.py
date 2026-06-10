"""
Test: PINN Model Registry

Verifies the curated PINN models (Convergence Plan Tier 1) instantiate via
the factory, run forward passes, and propagate gradients. The registry holds
one honest key per architecture — alias keys were pruned 2026-06.
"""
import pytest
import torch
from packages.core.models.model_factory import create_model, MODEL_REGISTRY

NUM_CLASSES = 11
SIGNAL_LENGTH = 5000
BATCH = 2

# The three physics-informed architectures of the curated zoo
PINN_KEYS = [
    'hybrid_pinn',
    'physics_constrained_cnn',
    'multitask_pinn',
]


class TestPINNInstantiation:
    """All PINN models should instantiate from the factory."""

    @pytest.mark.parametrize('name', PINN_KEYS)
    def test_create(self, name):
        model = create_model(name, num_classes=NUM_CLASSES)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    @pytest.mark.parametrize('name', PINN_KEYS)
    def test_exists_in_registry(self, name):
        assert name in MODEL_REGISTRY, f"'{name}' not found in MODEL_REGISTRY"


class TestPINNForward:
    """PINN models should accept a 1D signal and produce correct output shape."""

    @pytest.mark.parametrize('name', PINN_KEYS)
    def test_forward_pass(self, name):
        model = create_model(name, num_classes=NUM_CLASSES)
        model.eval()
        x = torch.randn(BATCH, 1, SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        assert out.shape == (BATCH, NUM_CLASSES), (
            f"Expected ({BATCH},{NUM_CLASSES}), got {tuple(out.shape)}"
        )


class TestPINNGradient:
    """All PINN models should allow gradient flow."""

    @pytest.mark.parametrize('name', PINN_KEYS)
    def test_gradient_flow(self, name):
        model = create_model(name, num_classes=NUM_CLASSES)
        model.train()
        x = torch.randn(BATCH, 1, SIGNAL_LENGTH, requires_grad=True)
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        loss = out.sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients found"


class TestPINNRegistry:
    """The registry holds exactly the curated PINN keys."""

    def test_all_pinn_keys_exist(self):
        for key in PINN_KEYS:
            assert key in MODEL_REGISTRY, f"Missing key: '{key}'"

    def test_no_stale_pinn_aliases(self):
        pruned = ['pinn', 'physics_informed', 'physics_cnn', 'adaptive_physics_cnn',
                  'adaptive_multitask_pinn', 'knowledge_graph_pinn', 'kg_pinn']
        for key in pruned:
            assert key not in MODEL_REGISTRY, (
                f"Pruned alias '{key}' has crept back into MODEL_REGISTRY"
            )
