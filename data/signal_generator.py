"""
Physics-Informed Synthetic Signal Generator — backward-compatible shim.

This module re-exports all public symbols from the ``data.signal_generation``
subpackage so that existing ``from data.signal_generator import …`` statements
throughout the codebase continue to work unchanged.

The canonical implementation now lives in:
    data/signal_generation/generator.py   — SignalGenerator
    data/signal_generation/fault_modeler.py — FaultModeler
    data/signal_generation/noise_generator.py — NoiseGenerator
    data/signal_generation/metadata.py    — SignalMetadata

Author: Syed Abbas Ahmad
Date: 2025-11-19  (refactored 2026-03-15)
"""

from data.signal_generation import (
    SignalGenerator,
    FaultModeler,
    NoiseGenerator,
    SignalMetadata,
)

__all__ = [
    "SignalGenerator",
    "FaultModeler",
    "NoiseGenerator",
    "SignalMetadata",
]
