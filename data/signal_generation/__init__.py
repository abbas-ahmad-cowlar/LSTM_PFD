"""
Signal generation package — physics-informed synthetic vibration signals.

Split from the monolithic ``data/signal_generator.py`` for maintainability.
Original file is kept as a backward-compatible re-export shim.
"""

from .metadata import SignalMetadata
from .fault_modeler import FaultModeler
from .noise_generator import NoiseGenerator
from .generator import SignalGenerator

__all__ = [
    "SignalMetadata",
    "FaultModeler",
    "NoiseGenerator",
    "SignalGenerator",
]
