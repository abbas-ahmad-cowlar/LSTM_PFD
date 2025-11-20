"""
Physics-Informed Neural Networks (PINN) Package

This package contains PINN architectures that incorporate domain knowledge
of bearing dynamics into neural network design and training.

Models:
- HybridPINN: Dual-branch architecture (data + physics features)
- PhysicsConstrainedCNN: Standard CNN with physics-based loss constraints
- AdaptivePhysicsConstrainedCNN: Physics-constrained with adaptive loss weighting

Key Features:
- Physics-based feature extraction (Sommerfeld, Reynolds numbers)
- Frequency consistency constraints
- Improved sample efficiency
- Better generalization to unseen operating conditions
"""

from .hybrid_pinn import HybridPINN, create_hybrid_pinn
from .physics_constrained_cnn import (
    PhysicsConstrainedCNN,
    AdaptivePhysicsConstrainedCNN,
    create_physics_constrained_cnn
)

__all__ = [
    'HybridPINN',
    'create_hybrid_pinn',
    'PhysicsConstrainedCNN',
    'AdaptivePhysicsConstrainedCNN',
    'create_physics_constrained_cnn'
]
