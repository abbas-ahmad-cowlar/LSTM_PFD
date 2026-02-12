"""
Contrastive Learning Models Package

Model architectures for contrastive learning:
- SignalEncoder: 1D CNN encoder with projection head
- ContrastiveClassifier: Classifier head for pretrained encoder
- ProjectionHead: MLP projection for contrastive loss
- ContrastiveEncoder: Generic encoder + projection wrapper (2D)
"""

from .signal_encoder import SignalEncoder
from .classifier import ContrastiveClassifier
from .projection import ProjectionHead
from .encoder import ContrastiveEncoder

__all__ = [
    'SignalEncoder',
    'ContrastiveClassifier',
    'ProjectionHead',
    'ContrastiveEncoder',
]
