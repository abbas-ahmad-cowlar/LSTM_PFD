"""
Physics-Informed Neural Networks (PINN) Package

Incorporates journal-bearing dynamics (Sommerfeld number, characteristic
frequencies, lubrication regimes) into neural architectures:

- HybridPINN: dual-branch (signal CNN/ResNet features + physics features)
- PhysicsConstrainedCNN: physics constraints as soft losses
- MultitaskPINN: shared encoder, fault/speed/load/severity heads
"""

from .hybrid_pinn import HybridPINN, create_hybrid_pinn
from .physics_constrained_cnn import (
    PhysicsConstrainedCNN,
    AdaptivePhysicsConstrainedCNN,
    create_physics_constrained_cnn
)
from .multitask_pinn import (
    MultitaskPINN,
    AdaptiveMultitaskPINN,
    create_multitask_pinn
)

__all__ = [
    'HybridPINN',
    'create_hybrid_pinn',
    'PhysicsConstrainedCNN',
    'AdaptivePhysicsConstrainedCNN',
    'create_physics_constrained_cnn',
    'MultitaskPINN',
    'AdaptiveMultitaskPINN',
    'create_multitask_pinn',
]
