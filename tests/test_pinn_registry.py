"""
Test: PINN Model Registry
Verifies all PINN models can be instantiated via the factory and pass a forward test.
"""
import pytest
import torch
from packages.core.models.model_factory import create_model, MODEL_REGISTRY

NUM_CLASSES = 11
SIGNAL_LENGTH = 5000
BATCH = 2


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------
PINN_MODELS_1D = [
    'physics_cnn',
    'physics_constrained_cnn',
    'multitask_pinn',
    'knowledge_graph_pinn',
    'kg_pinn',
]

PINN_MODELS_ADAPTIVE = [
    'adaptive_physics_cnn',
    'adaptive_multitask_pinn',
]


# ------------------------------------------------------------------
# Tests: Instantiation
# ------------------------------------------------------------------
class TestPINNInstantiation:
    """All PINN models should instantiate from factory."""

    @pytest.mark.parametrize('name', PINN_MODELS_1D + PINN_MODELS_ADAPTIVE)
    def test_create(self, name):
        model = create_model(name, num_classes=NUM_CLASSES)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    @pytest.mark.parametrize('name', PINN_MODELS_1D + PINN_MODELS_ADAPTIVE)
    def test_exists_in_registry(self, name):
        assert name in MODEL_REGISTRY, f"'{name}' not found in MODEL_REGISTRY"


# ------------------------------------------------------------------
# Tests: Forward Pass (standard 1D input only, no metadata)
# ------------------------------------------------------------------
class TestPINNForward:
    """PINN models should accept 1D signal and produce correct output shape."""

    @pytest.mark.parametrize('name', PINN_MODELS_1D)
    def test_forward_pass(self, name):
        model = create_model(name, num_classes=NUM_CLASSES)
        model.eval()
        x = torch.randn(BATCH, 1, SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (BATCH, NUM_CLASSES), f"Expected ({BATCH},{NUM_CLASSES}), got {out.shape}"

    @pytest.mark.parametrize('name', PINN_MODELS_ADAPTIVE)
    def test_forward_pass_adaptive(self, name):
        model = create_model(name, num_classes=NUM_CLASSES)
        model.eval()
        x = torch.randn(BATCH, 1, SIGNAL_LENGTH)
        with torch.no_grad():
            out = model(x)
        # Adaptive models may return a tuple (output, loss_dict) or just output
        if isinstance(out, tuple):
            out = out[0]
        assert out.shape[0] == BATCH


# ------------------------------------------------------------------
# Tests: Gradient Flow
# ------------------------------------------------------------------
class TestPINNGradient:
    """All PINN models should allow gradient flow."""

    @pytest.mark.parametrize('name', ['physics_cnn', 'multitask_pinn', 'knowledge_graph_pinn'])
    def test_gradient_flow(self, name):
        model = create_model(name, num_classes=NUM_CLASSES)
        model.train()
        x = torch.randn(BATCH, 1, SIGNAL_LENGTH, requires_grad=True)
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        loss = out.sum()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients found"


# ------------------------------------------------------------------
# Tests: Registry Completeness
# ------------------------------------------------------------------
class TestPINNRegistry:
    """Verify all expected PINN keys exist in registry."""

    EXPECTED_PINN_KEYS = [
        'pinn', 'hybrid_pinn', 'physics_informed',
        'physics_cnn', 'physics_constrained_cnn', 'adaptive_physics_cnn',
        'multitask_pinn', 'adaptive_multitask_pinn',
        'knowledge_graph_pinn', 'kg_pinn',
    ]

    def test_all_pinn_keys_exist(self):
        for key in self.EXPECTED_PINN_KEYS:
            assert key in MODEL_REGISTRY, f"Missing key: '{key}'"

    def test_pinn_count(self):
        pinn_keys = [k for k in MODEL_REGISTRY if 'pinn' in k or 'physics' in k or 'kg_' in k]
        assert len(pinn_keys) >= 10, f"Expected ≥10 PINN keys, found {len(pinn_keys)}"
