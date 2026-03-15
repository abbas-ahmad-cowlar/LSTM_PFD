"""
CNN-1D — backward-compatible re-export shim.

Canonical implementation lives in ``cnn/cnn_1d.py``.
This module re-exports all public symbols so that existing imports
(``from .cnn_1d import ...``) continue to work transparently.
"""

from .cnn.cnn_1d import CNN1D, create_cnn1d  # noqa: F401
