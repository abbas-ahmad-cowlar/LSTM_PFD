"""
Physics constants for hydrodynamic bearing fault diagnosis.

Purpose:
    Centralized, frozen dataclasses for physics-related magic numbers used
    throughout the signal generation and fault modeling pipeline.  Replaces
    scattered literals in ``FaultModeler``, ``NoiseGenerator``, and
    ``DataConfig`` with named, documented constants.

    All dataclasses are ``frozen=True`` — they are immutable at runtime.

Usage:
    >>> from utils.physics_constants import BEARING_PHYSICS, VISCOSITY_MODEL
    >>> print(BEARING_PHYSICS.sommerfeld_base)
    0.15
    >>> print(VISCOSITY_MODEL.decay_coefficient)
    -0.03

Author: Syed Abbas Ahmad
Date: 2026-03-15
"""

from dataclasses import dataclass
from typing import Tuple


# ---------------------------------------------------------------------------
# Bearing geometry and fluid-film parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BearingPhysics:
    """Hydrodynamic bearing physics defaults.

    These values are used throughout the signal generation pipeline for
    Sommerfeld number calculation, Reynolds number sampling, and clearance
    ratio sampling.  All values follow ISO 7902 / API 684 conventions.
    """

    # Sommerfeld number
    sommerfeld_base: float = 0.15
    """Base Sommerfeld number at nominal operating conditions."""

    sommerfeld_min: float = 0.05
    """Minimum clipped Sommerfeld number (prevents division instability)."""

    sommerfeld_max: float = 0.50
    """Maximum clipped Sommerfeld number."""

    # Reynolds number
    reynolds_min: float = 500.0
    """Minimum Reynolds number (laminar regime)."""

    reynolds_max: float = 5000.0
    """Maximum Reynolds number (transition to turbulent)."""

    # Clearance ratio (C/R)
    clearance_ratio_min: float = 0.001
    """Minimum radial clearance ratio."""

    clearance_ratio_max: float = 0.003
    """Maximum radial clearance ratio."""

    @property
    def reynolds_range(self) -> Tuple[float, float]:
        """Reynolds number sampling range."""
        return (self.reynolds_min, self.reynolds_max)

    @property
    def clearance_ratio_range(self) -> Tuple[float, float]:
        """Clearance ratio sampling range."""
        return (self.clearance_ratio_min, self.clearance_ratio_max)


# ---------------------------------------------------------------------------
# Viscosity model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ViscosityModel:
    """Temperature-dependent viscosity model parameters.

    Uses exponential decay: μ(T) = μ_ref · exp(decay_coefficient · (T − T_ref))
    """

    decay_coefficient: float = -0.03
    """Exponential decay rate (1/°C) — negative means viscosity drops with T."""

    reference_temperature_C: float = 60.0
    """Reference temperature (°C) at which base Sommerfeld is defined."""


# ---------------------------------------------------------------------------
# Fault signature amplitudes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FaultAmplitudes:
    """Default amplitude coefficients for each fault model.

    These scale the fault signatures before severity and transient modulation.
    """

    # Misalignment (desalignement)
    misalign_2X: float = 0.35
    misalign_3X: float = 0.20

    # Imbalance (desequilibre)
    imbalance_1X: float = 0.50

    # Clearance (jeu)
    clearance_sub: float = 0.25
    clearance_1X: float = 0.18
    clearance_2X: float = 0.10

    # Lubrication (lubrification)
    stick_slip: float = 0.30

    # Cavitation
    cavitation_burst: float = 0.60
    cavitation_burst_duration_s: float = 0.008
    """Duration of a cavitation burst in seconds."""

    # Wear (usure)
    wear_broadband: float = 0.25
    wear_asperity: float = 0.12

    # Oil whirl
    oil_whirl: float = 0.40
    oil_whirl_freq_ratio_base: float = 0.42
    oil_whirl_freq_ratio_spread: float = 0.06


# ---------------------------------------------------------------------------
# Noise model defaults
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NoiseDefaults:
    """Default amplitude/rate parameters for the 7-layer noise model."""

    measurement_std: float = 0.03
    emi_amplitude: float = 0.01
    pink_noise_std: float = 0.02
    drift_amplitude: float = 0.015
    quantization_step: float = 0.001
    sensor_drift_rate: float = 0.001
    """Sensor drift rate (units per second)."""
    impulse_rate: float = 2.0
    """Impulses per second."""
    aliasing_probability: float = 0.10
    """Probability that a signal contains aliasing artifacts."""


# ---------------------------------------------------------------------------
# Rotor dynamics parameters (V2 advanced physics)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RotorDynamics:
    """Rotor dynamics parameters for critical speed and resonance effects.

    Used by AdvancedPhysicsConfig when rotor_dynamics is enabled.
    Values are typical for mid-size industrial turbomachinery (API 684).
    """

    first_critical_hz: float = 45.0
    """First forward critical speed (Hz)."""

    second_critical_hz: float = 120.0
    """Second forward critical speed (Hz)."""

    damping_ratio: float = 0.05
    """Modal damping ratio (dimensionless, typically 0.02-0.10)."""

    amplification_factor: float = 10.0
    """Peak amplification at critical speed (Q = 1/(2*zeta))."""

    resonance_bandwidth_hz: float = 5.0
    """Half-power bandwidth around critical speed."""

    @property
    def quality_factor(self) -> float:
        """Quality factor Q = 1/(2*zeta)."""
        return 1.0 / (2.0 * self.damping_ratio)


@dataclass(frozen=True)
class CrossCouplingDefaults:
    """Default cross-coupling stiffness parameters for hydrodynamic bearings."""

    kxy_ratio: float = 0.30
    """Cross-coupled stiffness ratio Kxy/Kxx (typically 0.2-0.5)."""

    phase_offset_deg: float = 90.0
    """Phase offset between direct and cross-coupled components."""

    damping_cross_ratio: float = 0.10
    """Cross-coupled damping ratio Cxy/Cxx."""


# ---------------------------------------------------------------------------
# Convenience singletons
# ---------------------------------------------------------------------------

BEARING_PHYSICS = BearingPhysics()
VISCOSITY_MODEL = ViscosityModel()
FAULT_AMPLITUDES = FaultAmplitudes()
NOISE_DEFAULTS = NoiseDefaults()
ROTOR_DYNAMICS = RotorDynamics()
CROSS_COUPLING = CrossCouplingDefaults()
