"""
Import smoke tests for the training infrastructure.

Pytest-native rewrite of the old script-style checker (which called
sys.exit() at module level and crashed pytest collection).

NOTE (Convergence Phase 2): entries marked T3 below cover modules scheduled
for deletion in the pruning phase; remove those rows together with the code.
"""
import importlib

import pytest

# (module path, attribute names that must exist on the module)
IMPORT_CASES = [
    # Core trainers
    ("packages.core.training.base_trainer", ["BaseTrainer"]),
    ("packages.core.training.cnn_trainer", ["CNNTrainer"]),
    ("packages.core.training.pinn_trainer", ["PINNTrainer"]),
    ("packages.core.training.spectrogram_trainer", ["SpectrogramTrainer"]),  # T3
    ("packages.core.training.progressive_resizing", ["ProgressiveResizingTrainer"]),  # T3
    ("packages.core.training.knowledge_distillation", ["DistillationTrainer"]),  # T3
    ("packages.core.training.trainer", ["Trainer", "TrainingState"]),
    ("packages.core.training.mixed_precision", ["MixedPrecisionTrainer"]),
    # Mixins
    ("packages.core.training.mixins", ["SpecAugmentMixin", "PhysicsLossMixin"]),
    # Callbacks
    ("packages.core.training.callbacks", ["EarlyStopping", "ModelCheckpoint"]),
    # Losses (canonical)
    ("packages.core.training.losses.classification", ["FocalLoss", "create_criterion"]),
    ("packages.core.training.losses.distillation", ["DistillationLoss"]),  # T3
    ("packages.core.training.losses", ["FocalLoss", "DistillationLoss"]),
    # Losses (backward-compat shim)  # T3
    ("packages.core.training.cnn_losses", ["FocalLoss", "create_criterion"]),
    # Schedulers
    ("packages.core.training.schedulers", ["create_scheduler"]),
    ("packages.core.training.cnn_schedulers", ["create_cosine_scheduler"]),  # T3
    ("packages.core.training.transformer_schedulers", ["create_warmup_cosine_schedule"]),  # T3
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
    "SpectrogramTrainer",  # T3
    "ProgressiveResizingTrainer",  # T3
    "DistillationTrainer",  # T3
    "Trainer",
    "MixedPrecisionTrainer",
]


@pytest.mark.parametrize("trainer_name", TRAINER_SUBCLASS_CASES)
def test_trainer_inherits_base_trainer(trainer_name):
    """All trainers participate in the BaseTrainer hierarchy."""
    from packages.core.training.base_trainer import BaseTrainer

    training = importlib.import_module("packages.core.training")
    trainer_cls = getattr(training, trainer_name, None)
    if trainer_cls is None:
        # Not re-exported from the package __init__: import from its module
        module_by_name = {
            "CNNTrainer": "packages.core.training.cnn_trainer",
            "PINNTrainer": "packages.core.training.pinn_trainer",
            "SpectrogramTrainer": "packages.core.training.spectrogram_trainer",
            "ProgressiveResizingTrainer": "packages.core.training.progressive_resizing",
            "DistillationTrainer": "packages.core.training.knowledge_distillation",
            "Trainer": "packages.core.training.trainer",
            "MixedPrecisionTrainer": "packages.core.training.mixed_precision",
        }
        module = importlib.import_module(module_by_name[trainer_name])
        trainer_cls = getattr(module, trainer_name)

    assert issubclass(trainer_cls, BaseTrainer), (
        f"{trainer_name} does not inherit from BaseTrainer"
    )
