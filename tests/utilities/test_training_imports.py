"""
Import smoke tests for the training infrastructure.

Covers the kept training surface after the 2026-06 convergence pruning.
"""
import importlib

import pytest

# (module path, attribute names that must exist on the module)
IMPORT_CASES = [
    # Core trainers
    ("packages.core.training.base_trainer", ["BaseTrainer"]),
    ("packages.core.training.cnn_trainer", ["CNNTrainer"]),
    ("packages.core.training.pinn_trainer", ["PINNTrainer"]),
    ("packages.core.training.trainer", ["Trainer", "TrainingState"]),
    ("packages.core.training.mixed_precision", ["MixedPrecisionTrainer"]),
    # Mixins
    ("packages.core.training.mixins", ["SpecAugmentMixin", "PhysicsLossMixin"]),
    # Callbacks
    ("packages.core.training.callbacks", ["EarlyStopping", "ModelCheckpoint"]),
    # Losses
    ("packages.core.training.losses.classification", ["FocalLoss", "create_criterion"]),
    ("packages.core.training.losses", ["FocalLoss", "create_criterion"]),
    ("packages.core.training.physics_loss_functions", ["FrequencyConsistencyLoss"]),
    # Schedulers
    ("packages.core.training.schedulers", ["create_scheduler", "create_cosine_scheduler"]),
    # Package __init__
    ("packages.core.training", ["BaseTrainer", "CNNTrainer", "FocalLoss", "create_scheduler"]),
]


@pytest.mark.parametrize(
    "module_path,attrs",
    IMPORT_CASES,
    ids=[case[0] for case in IMPORT_CASES],
)
def test_module_imports(module_path, attrs):
    """Every training module imports and exposes its public names."""
    module = importlib.import_module(module_path)
    for attr in attrs:
        assert hasattr(module, attr), f"{module_path} is missing '{attr}'"


TRAINER_SUBCLASS_CASES = [
    "CNNTrainer",
    "PINNTrainer",
    "Trainer",
    "MixedPrecisionTrainer",
]


@pytest.mark.parametrize("trainer_name", TRAINER_SUBCLASS_CASES)
def test_trainer_inherits_base_trainer(trainer_name):
    """All trainers participate in the BaseTrainer hierarchy."""
    from packages.core.training.base_trainer import BaseTrainer

    training = importlib.import_module("packages.core.training")
    trainer_cls = getattr(training, trainer_name)

    assert issubclass(trainer_cls, BaseTrainer), (
        f"{trainer_name} does not inherit from BaseTrainer"
    )
