"""
Data generation configuration matching MATLAB generator.m structure.

Purpose:
    Configuration classes for synthetic signal generation including:
    - Signal parameters (fs, duration, speed)
    - Fault types and severity levels
    - 7-layer noise model configuration
    - Operating condition variations
    - Data augmentation settings

Author: LSTM_PFD Team
Date: 2025-11-19
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any
from pathlib import Path

from .base_config import BaseConfig, ConfigValidator


@dataclass
class SignalConfig(BaseConfig):
    """Signal generation parameters."""

    fs: int = 20480  # Sampling frequency (Hz)
    T: float = 5.0  # Signal duration (seconds)
    Omega_base: float = 60.0  # Nominal rotation speed (Hz) = 3600 RPM

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "fs": {"type": "integer", "minimum": 1000, "maximum": 100000},
                "T": {"type": "number", "minimum": 0.1, "maximum": 60.0},
                "Omega_base": {"type": "number", "minimum": 1.0, "maximum": 1000.0}
            },
            "required": ["fs", "T", "Omega_base"]
        }

    @property
    def N(self) -> int:
        """Total number of samples per signal."""
        return int(self.fs * self.T)

    @property
    def omega_base(self) -> float:
        """Angular velocity in rad/s."""
        return 2.0 * 3.14159265359 * self.Omega_base


@dataclass
class FaultConfig(BaseConfig):
    """Fault type selection configuration."""

    # Include fault categories
    include_single: bool = True  # Include 8 single fault types
    include_mixed: bool = True  # Include 3 mixed fault combinations
    include_healthy: bool = True  # Include healthy baseline

    # Individual single fault controls
    single_faults: Dict[str, bool] = field(default_factory=lambda: {
        'desalignement': True,  # Misalignment
        'desequilibre': True,  # Imbalance
        'jeu': True,  # Bearing clearance
        'lubrification': True,  # Lubrication issues
        'cavitation': True,  # Cavitation
        'usure': True,  # Wear
        'oilwhirl': True,  # Oil whirl
    })

    # Mixed fault controls
    mixed_faults: Dict[str, bool] = field(default_factory=lambda: {
        'misalign_imbalance': True,
        'wear_lube': True,
        'cavit_jeu': True,
    })

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "include_single": {"type": "boolean"},
                "include_mixed": {"type": "boolean"},
                "include_healthy": {"type": "boolean"},
            }
        }

    def get_fault_list(self) -> List[str]:
        """
        Build list of active fault types.

        Returns:
            List of fault type names to generate
        """
        fault_types = []

        if self.include_healthy:
            fault_types.append('sain')

        if self.include_single:
            for fault, enabled in self.single_faults.items():
                if enabled:
                    fault_types.append(fault)

        if self.include_mixed:
            for fault, enabled in self.mixed_faults.items():
                if enabled:
                    fault_types.append(f'mixed_{fault}')

        return fault_types


@dataclass
class SeverityConfig(BaseConfig):
    """Multi-severity fault progression configuration."""

    enabled: bool = True
    levels: List[str] = field(default_factory=lambda: [
        'incipient', 'mild', 'moderate', 'severe'
    ])

    # Non-overlapping severity factor ranges
    ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        'incipient': (0.20, 0.45),
        'mild': (0.45, 0.70),
        'moderate': (0.70, 0.90),
        'severe': (0.90, 1.00),
    })

    temporal_evolution: float = 0.30  # 30% of signals show progressive growth

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "temporal_evolution": {"type": "number", "minimum": 0, "maximum": 1}
            }
        }


@dataclass
class NoiseConfig(BaseConfig):
    """7-layer independent noise model configuration."""

    # Enable/disable individual noise sources
    measurement: bool = True  # Sensor electronics thermal noise
    emi: bool = True  # Electromagnetic interference (50/60 Hz)
    pink: bool = True  # 1/f environmental noise
    drift: bool = True  # Low-frequency thermal drift
    quantization: bool = True  # ADC resolution limits
    sensor_drift: bool = True  # Sensor calibration decay
    impulse: bool = True  # Sporadic mechanical impacts

    # Noise level parameters
    levels: Dict[str, float] = field(default_factory=lambda: {
        'measurement': 0.03,  # Gaussian std
        'emi': 0.01,  # EMI amplitude
        'pink': 0.02,  # Pink noise std
        'drift': 0.015,  # Drift amplitude
        'quantization_step': 0.001,  # ADC step size
        'sensor_drift_rate': 0.001,  # Drift per second
        'impulse_rate': 2.0,  # Impulses per second
    })

    aliasing: float = 0.10  # 10% of signals have aliasing artifacts

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "measurement": {"type": "boolean"},
                "emi": {"type": "boolean"},
                "pink": {"type": "boolean"},
                "drift": {"type": "boolean"},
                "quantization": {"type": "boolean"},
                "sensor_drift": {"type": "boolean"},
                "impulse": {"type": "boolean"},
                "aliasing": {"type": "number", "minimum": 0, "maximum": 1}
            }
        }


@dataclass
class OperatingConfig(BaseConfig):
    """Variable operating conditions configuration."""

    speed_variation: float = 0.10  # ±10% from nominal speed
    load_range: Tuple[float, float] = (0.30, 1.00)  # 30-100% of rated load
    temp_range: Tuple[float, float] = (40.0, 80.0)  # Operating temperature (°C)

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "speed_variation": {"type": "number", "minimum": 0, "maximum": 0.5},
            }
        }


@dataclass
class PhysicsConfig(BaseConfig):
    """Physics-based parameter calculation configuration."""

    enabled: bool = True
    calculate_sommerfeld: bool = True  # Calculate from operating conditions (not random)
    sommerfeld_base: float = 0.15  # Base Sommerfeld number
    reynolds_range: Tuple[float, float] = (500.0, 5000.0)
    clearance_ratio_range: Tuple[float, float] = (0.001, 0.003)

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "calculate_sommerfeld": {"type": "boolean"},
                "sommerfeld_base": {"type": "number", "minimum": 0.01, "maximum": 1.0}
            }
        }


@dataclass
class TransientConfig(BaseConfig):
    """Non-stationary behavior (transients) configuration."""

    enabled: bool = True
    probability: float = 0.25  # 25% of signals have transients
    types: List[str] = field(default_factory=lambda: [
        'speed_ramp', 'load_step', 'thermal_expansion'
    ])

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "probability": {"type": "number", "minimum": 0, "maximum": 1}
            }
        }


@dataclass
class AugmentationConfig(BaseConfig):
    """Data augmentation configuration."""

    enabled: bool = True
    ratio: float = 0.30  # 30% additional augmented samples
    methods: List[str] = field(default_factory=lambda: [
        'time_shift', 'amplitude_scale', 'noise_injection'
    ])

    # Augmentation parameter ranges
    time_shift_max: float = 0.02  # 2% of signal length
    amplitude_scale_range: Tuple[float, float] = (0.85, 1.15)
    extra_noise_range: Tuple[float, float] = (0.02, 0.05)

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean"},
                "ratio": {"type": "number", "minimum": 0, "maximum": 1.0},
                "time_shift_max": {"type": "number", "minimum": 0, "maximum": 0.5}
            }
        }


@dataclass
class DataConfig(BaseConfig):
    """
    Master data generation configuration.

    Aggregates all sub-configurations matching MATLAB CONFIG structure.
    """

    # Dataset generation parameters
    num_signals_per_fault: int = 100
    output_dir: str = 'data_signaux_sep_production'

    # Sub-configurations
    signal: SignalConfig = field(default_factory=SignalConfig)
    fault: FaultConfig = field(default_factory=FaultConfig)
    severity: SeverityConfig = field(default_factory=SeverityConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    operating: OperatingConfig = field(default_factory=OperatingConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    transient: TransientConfig = field(default_factory=TransientConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    # Reproducibility
    rng_seed: int = 42
    per_signal_seed_variation: bool = True

    # Output options
    save_metadata: bool = True
    verbose: bool = True

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "num_signals_per_fault": {"type": "integer", "minimum": 1, "maximum": 10000},
                "output_dir": {"type": "string"},
                "rng_seed": {"type": "integer"},
                "per_signal_seed_variation": {"type": "boolean"},
                "save_metadata": {"type": "boolean"},
                "verbose": {"type": "boolean"}
            },
            "required": ["num_signals_per_fault", "output_dir"]
        }

    @classmethod
    def from_matlab_struct(cls, mat_config: Dict) -> 'DataConfig':
        """
        Import configuration from MATLAB .mat file structure.

        Args:
            mat_config: Dictionary from scipy.io.loadmat

        Returns:
            DataConfig object
        """
        # TODO: Implement MATLAB struct parsing
        raise NotImplementedError("MATLAB import coming in data/matlab_importer.py")

    def get_total_signals(self) -> int:
        """Calculate total number of signals to generate."""
        num_faults = len(self.fault.get_fault_list())
        base_signals = num_faults * self.num_signals_per_fault

        if self.augmentation.enabled:
            augmented = int(base_signals * self.augmentation.ratio)
            return base_signals + augmented

        return base_signals
