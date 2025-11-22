"""
Physics-Informed Neural Networks (PINN) Package

This package contains PINN architectures that incorporate domain knowledge
of bearing dynamics into neural network design and training.

Models:
- HybridPINN: Dual-branch architecture (data + physics features)
- PhysicsConstrainedCNN: Standard CNN with physics-based loss constraints
- AdaptivePhysicsConstrainedCNN: Physics-constrained with adaptive loss weighting
- MultitaskPINN: Multi-task learning (fault + speed/load/severity)
- AdaptiveMultitaskPINN: Multi-task with learnable task weights
- KnowledgeGraphPINN: Graph neural network leveraging fault relationships

Key Features:
- Physics-based feature extraction (Sommerfeld, Reynolds numbers)
- Frequency consistency constraints
- Improved sample efficiency
- Better generalization to unseen operating conditions
- Fault relationship modeling via knowledge graphs
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
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
from .knowledge_graph_pinn import (
    KnowledgeGraphPINN,
    FaultKnowledgeGraph,
    GraphConvolutionLayer
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
    'KnowledgeGraphPINN',
    'FaultKnowledgeGraph',
    'GraphConvolutionLayer'
]
