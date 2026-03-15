"""
ResNet-1D — backward-compatible re-export shim.

Canonical implementation lives in ``resnet/resnet_1d.py``.
This module re-exports all public symbols so that existing imports
(``from .resnet_1d import ...``) continue to work transparently.
"""

from .resnet.resnet_1d import (  # noqa: F401
    ResNet1D,
    create_resnet18_1d,
    create_resnet34_1d,
    create_resnet50_1d,
)
