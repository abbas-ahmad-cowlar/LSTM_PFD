"""
Signal metadata dataclass.

Extracted from: data/signal_generator.py
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class SignalMetadata:
    """Metadata for generated signal (matches MATLAB structure)."""

    # Fault information
    fault: str
    severity: str
    severity_factor_initial: float
    has_evolution: bool

    # Operating conditions
    speed_rpm: float
    speed_variation_factor: float
    load_percent: float
    temperature_C: float
    operating_factor: float

    # Physics parameters
    sommerfeld_number: float
    reynolds_number: float
    clearance_ratio: float
    physics_factor: float
    sommerfeld_calculated: bool

    # Transient behavior
    transient_type: str
    transient_params: Dict[str, Any]

    # Signal properties
    fs: int
    duration_s: float
    num_samples: int
    signal_rms: float
    signal_peak: float
    signal_crest_factor: float

    # Augmentation
    is_augmented: bool
    augmentation: Dict[str, Any]

    # Noise sources applied
    noise_sources: Dict[str, bool]

    # Generation metadata
    generation_timestamp: str
    generator_version: str
    rng_seed: int
    is_overlapping_fault: bool
