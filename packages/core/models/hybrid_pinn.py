"""
HybridPINN — backward-compatible re-export shim.

Canonical implementation lives in ``pinn/hybrid_pinn.py``.
This module re-exports all public symbols so that existing imports
(``from .hybrid_pinn import ...``) continue to work transparently.
"""

from .pinn.hybrid_pinn import (  # noqa: F401
    HybridPINN,
    create_hybrid_pinn,
)
