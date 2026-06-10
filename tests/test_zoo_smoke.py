"""
Zoo smoke test — the permanent gate for every kept model architecture.

Convergence Plan P1.2: every Tier-1/Tier-2 model must construct via the
factory, run a forward pass on a realistic input, and backpropagate a loss.
A model that cannot pass this test does not stay in the repo (Part I §3).

Tier membership is defined in CONVERGENCE_PLAN.md Part I §4. When pruning
(Phase 2) or changing tiers, update TIER1_KEYS / TIER2_KEYS here in the
same commit.
"""
import pytest
import torch

from packages.core.models.model_factory import create_model
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH

# Tier 1 — core benchmark zoo (trainable PyTorch nets; classical ML models
# are sklearn-based and covered by tests/test_classical_models.py)
TIER1_KEYS = [
    "cnn1d",
    "attention_cnn",
    "cnn_lstm",
    "resnet18",
    "patchtst",
    "hybrid_pinn",
    "physics_constrained_cnn",
    "multitask_pinn",
]

# Tier 2 — extension zoo (kept, smoke-tested, benchmark-optional)
TIER2_KEYS = [
    "multi_scale_cnn",
    "se_resnet18",
    "signal_transformer",
]

ALL_KEYS = TIER1_KEYS + TIER2_KEYS

BATCH = 2


def _forward(model, x):
    """Run a forward pass, normalizing multi-output models to logits."""
    out = model(x)
    if isinstance(out, dict):
        # Multi-task models: pick the fault-classification head
        for key in ("fault_logits", "logits", "fault"):
            if key in out:
                return out[key]
        return next(iter(out.values()))
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


@pytest.mark.parametrize("model_key", ALL_KEYS)
def test_forward_backward(model_key):
    """Model constructs, forwards at full signal length, and backprops."""
    torch.manual_seed(0)
    model = create_model(model_key, num_classes=NUM_CLASSES)
    model.train()

    x = torch.randn(BATCH, 1, SIGNAL_LENGTH)
    logits = _forward(model, x)

    assert logits.shape == (BATCH, NUM_CLASSES), (
        f"{model_key}: expected logits {(BATCH, NUM_CLASSES)}, got {tuple(logits.shape)}"
    )
    assert torch.isfinite(logits).all(), f"{model_key}: non-finite logits"

    loss = torch.nn.functional.cross_entropy(
        logits, torch.randint(0, NUM_CLASSES, (BATCH,))
    )
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads), (
        f"{model_key}: no gradients reached any parameter"
    )


def test_voting_ensemble_forward():
    """Soft-voting ensemble of two zoo members produces valid probabilities."""
    from packages.core.models.ensemble.voting_ensemble import create_voting_ensemble

    torch.manual_seed(0)
    members = [
        create_model("cnn1d", num_classes=NUM_CLASSES),
        create_model("resnet18", num_classes=NUM_CLASSES),
    ]
    ensemble = create_voting_ensemble(
        models=members, voting_type="soft", num_classes=NUM_CLASSES
    )
    ensemble.eval()

    x = torch.randn(BATCH, 1, SIGNAL_LENGTH)
    with torch.no_grad():
        out = _forward(ensemble, x)

    assert out.shape == (BATCH, NUM_CLASSES)
    assert torch.isfinite(out).all()
