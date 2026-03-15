"""
Smoke test for training infrastructure imports.
Run from project root: python tests/utilities/test_training_imports.py
"""
import sys
import os

# Project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

passed = 0
failed = 0
errors = []


def check(label, import_fn):
    global passed, failed
    try:
        import_fn()
        print(f"  [PASS] {label}")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] {label}: {type(e).__name__}: {e}")
        errors.append((label, str(e)))
        failed += 1


print("=" * 60)
print("Training Infrastructure Import Smoke Tests")
print("=" * 60)

# --- Phase A: Core trainers ---
print("\n--- Phase A: Core Trainers ---")

check("BaseTrainer", lambda: __import__("packages.core.training.base_trainer", fromlist=["BaseTrainer"]))
check("CNNTrainer", lambda: __import__("packages.core.training.cnn_trainer", fromlist=["CNNTrainer"]))
check("PINNTrainer", lambda: __import__("packages.core.training.pinn_trainer", fromlist=["PINNTrainer"]))
check("SpectrogramTrainer", lambda: __import__("packages.core.training.spectrogram_trainer", fromlist=["SpectrogramTrainer"]))
check("ProgressiveResizingTrainer", lambda: __import__("packages.core.training.progressive_resizing", fromlist=["ProgressiveResizingTrainer"]))
check("DistillationTrainer", lambda: __import__("packages.core.training.knowledge_distillation", fromlist=["DistillationTrainer"]))
check("Trainer (compat shim)", lambda: __import__("packages.core.training.trainer", fromlist=["Trainer", "TrainingState"]))
check("MixedPrecisionTrainer", lambda: __import__("packages.core.training.mixed_precision", fromlist=["MixedPrecisionTrainer"]))

# --- Phase B: Mixins ---
print("\n--- Phase B: Mixins ---")

check("SpecAugmentMixin", lambda: __import__("packages.core.training.mixins", fromlist=["SpecAugmentMixin"]))
check("PhysicsLossMixin", lambda: __import__("packages.core.training.mixins", fromlist=["PhysicsLossMixin"]))

# --- Phase C: Callbacks ---
print("\n--- Phase C: Callbacks ---")

check("Callback classes", lambda: __import__("packages.core.training.callbacks", fromlist=["EarlyStopping", "ModelCheckpoint"]))

# --- Phase D: Losses ---
print("\n--- Phase D: Losses (new canonical) ---")

check("losses.classification", lambda: __import__("packages.core.training.losses.classification", fromlist=["FocalLoss", "create_criterion"]))
check("losses.distillation", lambda: __import__("packages.core.training.losses.distillation", fromlist=["DistillationLoss"]))
check("losses.__init__", lambda: __import__("packages.core.training.losses", fromlist=["FocalLoss", "DistillationLoss"]))

print("\n--- Phase D: Losses (backward-compat shims) ---")

check("cnn_losses (shim)", lambda: __import__("packages.core.training.cnn_losses", fromlist=["FocalLoss", "create_criterion"]))

# --- Phase E: Schedulers ---
print("\n--- Phase E: Schedulers ---")

check("schedulers (unified)", lambda: __import__("packages.core.training.schedulers", fromlist=["create_scheduler"]))
check("cnn_schedulers (shim)", lambda: __import__("packages.core.training.cnn_schedulers", fromlist=["create_cosine_scheduler"]))
check("transformer_schedulers (shim)", lambda: __import__("packages.core.training.transformer_schedulers", fromlist=["create_warmup_cosine_schedule"]))

# --- Phase F: Package __init__ ---
print("\n--- Phase F: Package __init__ ---")

check("training __init__", lambda: __import__("packages.core.training", fromlist=["BaseTrainer", "CNNTrainer", "FocalLoss", "create_scheduler"]))

# --- Inheritance checks ---
print("\n--- Inheritance Checks ---")

try:
    from packages.core.training.base_trainer import BaseTrainer
    from packages.core.training.cnn_trainer import CNNTrainer
    from packages.core.training.pinn_trainer import PINNTrainer
    from packages.core.training.spectrogram_trainer import SpectrogramTrainer
    from packages.core.training.progressive_resizing import ProgressiveResizingTrainer
    from packages.core.training.knowledge_distillation import DistillationTrainer
    from packages.core.training.trainer import Trainer
    from packages.core.training.mixed_precision import MixedPrecisionTrainer

    for name, cls in [
        ("CNNTrainer", CNNTrainer),
        ("PINNTrainer", PINNTrainer),
        ("SpectrogramTrainer", SpectrogramTrainer),
        ("ProgressiveResizingTrainer", ProgressiveResizingTrainer),
        ("DistillationTrainer", DistillationTrainer),
        ("Trainer", Trainer),
        ("MixedPrecisionTrainer", MixedPrecisionTrainer),
    ]:
        if issubclass(cls, BaseTrainer):
            print(f"  [PASS] {name} -> BaseTrainer")
            passed += 1
        else:
            print(f"  [FAIL] {name} does NOT inherit from BaseTrainer!")
            failed += 1
except Exception as e:
    print(f"  [FAIL] Inheritance check error: {type(e).__name__}: {e}")
    failed += 1

# --- Summary ---
print("\n" + "=" * 60)
print(f"Results: {passed} passed, {failed} failed")
if errors:
    print("\nFailed tests:")
    for label, err in errors:
        print(f"  - {label}: {err}")
print("=" * 60)

sys.exit(1 if failed else 0)
