"""
Physics-Informed Synthetic Signal Generator (Python port of generator.m).

Purpose:
    Generate realistic vibration signals for bearing fault diagnosis with:
    - Physics-based fault models (11 classes)
    - 7-layer independent noise sources
    - Multi-severity temporal evolution
    - Calculated operating conditions (Sommerfeld number)
    - Data augmentation

Author: Syed Abbas Ahmad
Date: 2025-11-19
Reference: generator.m (MATLAB Production v2.0)
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, asdict
import time
from scipy import signal as sp_signal
from scipy.io import savemat
from sklearn.model_selection import train_test_split
import h5py
import json
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.data_config import DataConfig
from utils.reproducibility import set_seed
from utils.logging import get_logger
from utils.constants import FAULT_TYPES, NUM_CLASSES, SAMPLING_RATE

logger = get_logger(__name__)


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


class FaultModeler:
    """
    Physics-based fault modeling for hydrodynamic bearings.

    Implements equations from Section 7.3 of technical report.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.fs = config.signal.fs
        self.T = config.signal.T
        self.N = config.signal.N
        self.t = np.arange(0, self.N) / self.fs

    def generate_fault_signal(
        self,
        fault_type: str,
        severity_curve: np.ndarray,
        transient_modulation: np.ndarray,
        omega: float,
        Omega: float,
        load_factor: float,
        temp_factor: float,
        operating_factor: float,
        physics_factor: float,
        sommerfeld: float,
        speed_variation: float
    ) -> np.ndarray:
        """
        Generate fault-specific vibration signature.

        Args:
            fault_type: Fault class name
            severity_curve: Time-varying severity factor [N,]
            transient_modulation: Transient modulation [N,]
            omega: Angular velocity (rad/s)
            Omega: Rotational speed (Hz)
            load_factor: Load factor (0.3-1.0)
            temp_factor: Temperature factor
            operating_factor: Combined operating factor
            physics_factor: Physics-based scaling
            sommerfeld: Sommerfeld number
            speed_variation: Speed variation factor

        Returns:
            Fault signal array [N,]
        """
        if fault_type == 'sain':
            return np.zeros(self.N)

        elif fault_type == 'desalignement':
            # Misalignment: 2X and 3X harmonics
            phase_2X = np.random.rand() * 2 * np.pi
            phase_3X = np.random.rand() * 2 * np.pi
            misalign_2X = 0.35 * np.sin(2 * omega * self.t + phase_2X)
            misalign_3X = 0.20 * np.sin(3 * omega * self.t + phase_3X)
            x_fault = severity_curve * (misalign_2X + misalign_3X) * transient_modulation
            return x_fault

        elif fault_type == 'desequilibre':
            # Imbalance: 1X dominant, speed-squared dependence
            phase_1X = np.random.rand() * 2 * np.pi
            imbalance_1X = 0.5 * load_factor * np.sin(omega * self.t + phase_1X) * (speed_variation ** 2)
            x_fault = severity_curve * imbalance_1X * transient_modulation
            return x_fault

        elif fault_type == 'jeu':
            # Bearing clearance: sub-synchronous + harmonics
            sub_freq = (0.43 + 0.05 * np.random.rand()) * Omega
            clearance_sub = 0.25 * temp_factor * np.sin(2 * np.pi * sub_freq * self.t)
            clearance_1X = 0.18 * np.sin(omega * self.t)
            clearance_2X = 0.10 * np.sin(2 * omega * self.t)
            x_fault = severity_curve * (clearance_sub + clearance_1X + clearance_2X) * transient_modulation
            return x_fault

        elif fault_type == 'lubrification':
            # Lubrication: stick-slip (INVERSE Sommerfeld)
            stick_slip_freq = 2 + 3 * np.random.rand()
            stick_slip = 0.30 * temp_factor * (0.3 / sommerfeld) * np.sin(2 * np.pi * stick_slip_freq * self.t)

            # Metal contact events
            x_fault = severity_curve * stick_slip * transient_modulation
            impact_rate = int(1 + 3 * np.mean(severity_curve))
            for j in range(impact_rate):
                impact_pos = np.random.randint(0, self.N - 20)
                impact_amp = 0.5 * np.mean(severity_curve)
                impact_len = min(20, self.N - impact_pos)
                decay = np.exp(-0.4 * np.arange(impact_len))
                x_fault[impact_pos:impact_pos + impact_len] += impact_amp * decay * np.random.randn(impact_len)

            return x_fault

        elif fault_type == 'cavitation':
            # Cavitation: high-frequency bursts
            x_fault = np.zeros(self.N)
            burst_rate = int(2 + 5 * np.mean(severity_curve))
            burst_len = int(0.008 * self.fs)

            for i_burst in range(burst_rate):
                if burst_len >= self.N:
                    continue
                pos = np.random.randint(0, self.N - burst_len)
                burst_freq = 1500 + 1000 * np.random.rand()
                burst_t = np.arange(burst_len) / self.fs
                hann_window = sp_signal.windows.hann(burst_len)
                burst = (0.6 * np.mean(severity_curve) *
                        np.sin(2 * np.pi * burst_freq * burst_t) *
                        np.exp(-100 * burst_t) * hann_window)
                x_fault[pos:pos + burst_len] += burst

            return x_fault

        elif fault_type == 'usure':
            # Wear: broadband noise + amplitude modulation
            wear_noise = 0.25 * operating_factor * physics_factor * np.random.randn(self.N)
            asperity_harm = 0.12 * (np.sin(omega * self.t) + 0.5 * np.sin(2 * omega * self.t))
            wear_mod_freq = 0.5 + 1.5 * np.random.rand()
            wear_mod = 1 + 0.3 * np.sin(2 * np.pi * wear_mod_freq * self.t)
            x_fault = severity_curve * (wear_noise + asperity_harm) * wear_mod * transient_modulation
            return x_fault

        elif fault_type == 'oilwhirl':
            # Oil whirl: sub-synchronous (CORRECT inverse Sommerfeld)
            whirl_freq_ratio = 0.42 + 0.06 * np.random.rand()
            whirl_freq = whirl_freq_ratio * Omega
            whirl_amp = 0.40 * (1 / np.sqrt(sommerfeld))
            whirl_signal = whirl_amp * np.sin(2 * np.pi * whirl_freq * self.t)
            subsync_mod_freq = whirl_freq * 0.5
            subsync_mod = 1 + 0.2 * np.sin(2 * np.pi * subsync_mod_freq * self.t)
            x_fault = severity_curve * whirl_signal * subsync_mod * transient_modulation
            return x_fault

        elif fault_type == 'mixed_misalign_imbalance':
            # FIXED: Additive combination
            misalign_sev = np.mean(severity_curve)
            phase_2X = np.random.rand() * 2 * np.pi
            phase_3X = np.random.rand() * 2 * np.pi
            misalign_2X = 0.25 * misalign_sev * np.sin(2 * omega * self.t + phase_2X)
            misalign_3X = 0.15 * misalign_sev * np.sin(3 * omega * self.t + phase_3X)

            imbalance_sev = np.mean(severity_curve)
            phase_1X = np.random.rand() * 2 * np.pi
            imbalance_1X = 0.35 * imbalance_sev * load_factor * np.sin(omega * self.t + phase_1X) * (speed_variation ** 2)

            combined = severity_curve * (misalign_2X + misalign_3X + imbalance_1X)
            x_fault = combined * transient_modulation
            return x_fault

        elif fault_type == 'mixed_wear_lube':
            # Wear + Lubrication (additive)
            wear_sev = np.mean(severity_curve)
            wear_noise = 0.18 * wear_sev * operating_factor * physics_factor * np.random.randn(self.N)
            asperity_harm = 0.08 * wear_sev * (np.sin(omega * self.t) + 0.5 * np.sin(2 * omega * self.t))

            lube_sev = np.mean(severity_curve)
            stick_slip_freq = 2 + 3 * np.random.rand()
            stick_slip = 0.20 * lube_sev * temp_factor * (0.3 / sommerfeld) * np.sin(2 * np.pi * stick_slip_freq * self.t)

            # Contact events
            x_fault = severity_curve * (wear_noise + asperity_harm + stick_slip) * transient_modulation
            contact_rate = int(2 + 3 * lube_sev)
            for jj in range(contact_rate):
                contact_pos = np.random.randint(0, max(1, self.N - 10))
                contact_amp = 0.4 * lube_sev
                contact_len = min(10, self.N - contact_pos)
                decay = np.exp(-0.5 * np.arange(contact_len))
                x_fault[contact_pos:contact_pos + contact_len] += contact_amp * decay * np.random.randn(contact_len)

            return x_fault

        elif fault_type == 'mixed_cavit_jeu':
            # Cavitation + Clearance (additive)
            x_fault = np.zeros(self.N)
            cavit_sev = np.mean(severity_curve)
            burst_rate = int(3 + 4 * cavit_sev)
            burst_len = int(0.008 * self.fs)

            for i_b in range(burst_rate):
                if burst_len >= self.N:
                    continue
                pos = np.random.randint(0, self.N - burst_len)
                burst_freq = 1500 + 1000 * np.random.rand()
                burst_t = np.arange(burst_len) / self.fs
                hann_window = sp_signal.windows.hann(burst_len)
                burst = (0.5 * cavit_sev *
                        np.sin(2 * np.pi * burst_freq * burst_t) *
                        np.exp(-100 * burst_t) * hann_window)
                x_fault[pos:pos + burst_len] += burst

            clearance_sev = np.mean(severity_curve)
            sub_freq = (0.43 + 0.05 * np.random.rand()) * Omega
            clearance_sub = 0.22 * clearance_sev * temp_factor * np.sin(2 * np.pi * sub_freq * self.t)
            clearance_1X = 0.15 * clearance_sev * np.sin(omega * self.t)

            combined = severity_curve * (clearance_sub + clearance_1X)
            x_fault += combined * transient_modulation
            return x_fault

        else:
            raise ValueError(f"Unknown fault type: {fault_type}")


class NoiseGenerator:
    """7-layer independent noise model generator."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.fs = config.signal.fs
        self.T = config.signal.T
        self.N = config.signal.N
        self.t = np.arange(0, self.N) / self.fs

    def apply_noise_layers(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, bool]]:
        """
        Apply 7-layer noise model to signal.

        Args:
            x: Clean signal [N,]

        Returns:
            Tuple of (noisy_signal, noise_sources_applied)
        """
        noise_cfg = self.config.noise
        x_noisy = x.copy()
        applied = {}

        # 1. Measurement noise (sensor electronics)
        if noise_cfg.measurement:
            noise_meas = noise_cfg.levels['measurement'] * np.random.randn(self.N)
            x_noisy += noise_meas
            applied['measurement'] = True
        else:
            applied['measurement'] = False

        # 2. EMI (electromagnetic interference - 50/60 Hz)
        if noise_cfg.emi:
            emi_freq = 50 + 10 * np.random.rand()
            emi_amp = noise_cfg.levels['emi'] * (1 + 0.5 * np.random.rand())
            emi_signal = emi_amp * np.sin(2 * np.pi * emi_freq * self.t + np.random.rand() * 2 * np.pi)
            x_noisy += emi_signal
            applied['emi'] = True
        else:
            applied['emi'] = False

        # 3. Pink noise (1/f)
        if noise_cfg.pink:
            pink_noise = np.cumsum(np.random.randn(self.N))
            pink_noise = noise_cfg.levels['pink'] * (pink_noise / (np.std(pink_noise) + 1e-10))
            x_noisy += pink_noise
            applied['pink'] = True
        else:
            applied['pink'] = False

        # 4. Environmental drift
        if noise_cfg.drift:
            drift_period = 1.5
            drift = noise_cfg.levels['drift'] * np.sin(2 * np.pi * (1 / drift_period) * self.t)
            x_noisy += drift
            applied['drift'] = True
        else:
            applied['drift'] = False

        # 5. Quantization noise (ADC resolution)
        if noise_cfg.quantization:
            quant_step = noise_cfg.levels['quantization_step']
            x_noisy = np.round(x_noisy / quant_step) * quant_step
            applied['quantization'] = True
        else:
            applied['quantization'] = False

        # 6. Sensor drift (cumulative offset)
        if noise_cfg.sensor_drift:
            sensor_drift_rate = noise_cfg.levels['sensor_drift_rate'] / self.T
            sensor_offset = sensor_drift_rate * self.t
            x_noisy += sensor_offset
            applied['sensor_drift'] = True
        else:
            applied['sensor_drift'] = False

        # 7. Aliasing artifacts (10% probability)
        if np.random.rand() < noise_cfg.aliasing:
            alias_freq = self.fs / 2 + 100 + 200 * np.random.rand()
            alias_signal = 0.005 * np.sin(2 * np.pi * alias_freq * self.t)
            x_noisy += alias_signal
            applied['aliasing'] = True
        else:
            applied['aliasing'] = False

        # 8. Impulse noise (sporadic impacts)
        if noise_cfg.impulse:
            num_impulses = int(noise_cfg.levels['impulse_rate'] * self.T)
            for imp in range(num_impulses):
                imp_pos = np.random.randint(0, max(1, self.N - 5))
                imp_amp = 0.02 + 0.03 * np.random.rand()
                imp_len = min(5, self.N - imp_pos)
                decay = np.exp(-0.3 * np.arange(imp_len))
                x_noisy[imp_pos:imp_pos + imp_len] += imp_amp * decay * np.random.randn(imp_len)
            applied['impulse'] = True
        else:
            applied['impulse'] = False

        return x_noisy, applied


class SignalGenerator:
    """
    Main signal generation orchestrator (Python port of generator.m).

    Generates synthetic vibration signals with physics-based fault models,
    7-layer noise, multi-severity progression, and data augmentation.
    """

    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize signal generator.

        Args:
            config: Data generation configuration (uses defaults if None)
        """
        if config is None:
            config = DataConfig()

        self.config = config
        self.fault_modeler = FaultModeler(config)
        self.noise_generator = NoiseGenerator(config)

        # Set random seed for reproducibility
        if config.rng_seed is not None:
            set_seed(config.rng_seed)

        logger.info("Signal Generator initialized")
        logger.info(f"  Sampling rate: {config.signal.fs} Hz")
        logger.info(f"  Signal duration: {config.signal.T} s")
        logger.info(f"  Samples per signal: {config.signal.N}")

    def generate_dataset(self) -> Dict[str, Any]:
        """
        Generate complete dataset of fault signals.

        Returns:
            Dictionary containing:
                - signals: List of generated signals
                - metadata: List of metadata dictionaries
                - config: Configuration used
                - statistics: Generation statistics

        This is the main entry point matching MATLAB's generation loop.
        """
        fault_types = self.config.fault.get_fault_list()
        logger.info(f"Generating dataset for {len(fault_types)} fault types")

        all_signals = []
        all_metadata = []
        all_labels = []

        generation_start = time.time()
        total_signals = 0

        for k, fault in enumerate(fault_types):
            logger.info(f"[{k+1}/{len(fault_types)}] Generating fault: {fault}")

            num_base = self.config.num_signals_per_fault

            if self.config.augmentation.enabled:
                num_augmented = int(num_base * self.config.augmentation.ratio)
            else:
                num_augmented = 0

            num_total_for_fault = num_base + num_augmented

            for n in range(num_total_for_fault):
                # Per-signal seed variation
                if self.config.per_signal_seed_variation and self.config.rng_seed is not None:
                    set_seed(self.config.rng_seed + total_signals)

                is_augmented = (n >= num_base)

                # Generate single signal
                signal, metadata = self.generate_single_signal(fault, is_augmented)

                all_signals.append(signal)
                all_metadata.append(metadata)
                all_labels.append(fault)
                total_signals += 1

            logger.info(f"  ✓ Generated {num_total_for_fault} signals ({num_base} base + {num_augmented} augmented)")

        generation_time = time.time() - generation_start

        logger.info("=" * 60)
        logger.info("✅ DATA GENERATION COMPLETE")
        logger.info(f"  Total signals: {total_signals}")
        logger.info(f"  Fault types: {len(fault_types)}")
        logger.info(f"  Generation time: {generation_time:.2f} s ({total_signals/generation_time:.2f} signals/s)")
        logger.info("=" * 60)

        return {
            'signals': all_signals,
            'metadata': all_metadata,
            'labels': all_labels,
            'config': self.config,
            'statistics': {
                'total_signals': total_signals,
                'num_faults': len(fault_types),
                'generation_time_s': generation_time,
                'signals_per_second': total_signals / generation_time
            }
        }

    def generate_single_signal(
        self,
        fault: str,
        is_augmented: bool = False
    ) -> Tuple[np.ndarray, SignalMetadata]:
        """
        Generate a single fault signal with metadata.

        Args:
            fault: Fault type name
            is_augmented: Whether this is an augmented sample

        Returns:
            Tuple of (signal array, metadata object)
        """
        # Severity configuration
        severity, severity_factor, severity_curve, has_evolution = self._configure_severity(fault)

        # Operating conditions
        Omega, omega, speed_variation, load_factor, load_percent, temperature_C, temp_factor, operating_factor, amp_base = self._configure_operating_conditions()

        # Physics parameters
        sommerfeld, reynolds, clearance_ratio, physics_factor = self._configure_physics(Omega, temperature_C, load_factor)

        # Transients
        transient_modulation, transient_type, transient_params = self._configure_transients()

        # Initialize baseline signal
        x = amp_base * 0.05 * np.random.randn(self.config.signal.N)

        # Apply noise layers
        x, noise_sources_applied = self.noise_generator.apply_noise_layers(x)

        # Apply fault-specific signature
        x_fault = self.fault_modeler.generate_fault_signal(
            fault, severity_curve, transient_modulation, omega, Omega,
            load_factor, temp_factor, operating_factor, physics_factor,
            sommerfeld, speed_variation
        )
        x += x_fault

        # Data augmentation
        aug_params = self._apply_augmentation(x, is_augmented)

        # Assemble metadata
        metadata = SignalMetadata(
            fault=fault,
            severity=severity,
            severity_factor_initial=severity_factor,
            has_evolution=has_evolution,
            speed_rpm=Omega * 60,
            speed_variation_factor=speed_variation,
            load_percent=load_percent,
            temperature_C=temperature_C,
            operating_factor=operating_factor,
            sommerfeld_number=sommerfeld,
            reynolds_number=reynolds,
            clearance_ratio=clearance_ratio,
            physics_factor=physics_factor,
            sommerfeld_calculated=self.config.physics.calculate_sommerfeld,
            transient_type=transient_type,
            transient_params=transient_params,
            fs=self.config.signal.fs,
            duration_s=self.config.signal.T,
            num_samples=self.config.signal.N,
            signal_rms=float(np.sqrt(np.mean(x**2))),
            signal_peak=float(np.max(np.abs(x))),
            signal_crest_factor=float(np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-10)),
            is_augmented=is_augmented,
            augmentation=aug_params,
            noise_sources=noise_sources_applied,
            generation_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            generator_version="Python_v1.0",
            rng_seed=self.config.rng_seed,
            is_overlapping_fault='mixed_' in fault
        )

        return x, metadata

    def _configure_severity(self, fault: str) -> Tuple[str, float, np.ndarray, bool]:
        """Configure severity level and temporal evolution."""
        if self.config.severity.enabled and fault != 'sain':
            severity = np.random.choice(self.config.severity.levels)
            severity_range = self.config.severity.ranges[severity]
            severity_factor = severity_range[0] + (severity_range[1] - severity_range[0]) * np.random.rand()
        else:
            severity = 'nominal'
            severity_factor = 1.0

        # Temporal evolution
        if self.config.severity.temporal_evolution > 0 and np.random.rand() < self.config.severity.temporal_evolution:
            evolution_end = min(1.0, severity_factor + 0.3)
            evolution_curve = np.linspace(severity_factor, evolution_end, self.config.signal.N)
            has_evolution = True
        else:
            evolution_curve = severity_factor * np.ones(self.config.signal.N)
            has_evolution = False

        return severity, severity_factor, evolution_curve, has_evolution

    def _configure_operating_conditions(self) -> Tuple:
        """Configure variable operating conditions."""
        # Speed variation
        speed_variation = 1.0 + (np.random.rand() - 0.5) * 2 * self.config.operating.speed_variation
        Omega = self.config.signal.Omega_base * speed_variation
        omega = 2 * np.pi * Omega

        # Load (30-100% of rated)
        load_percent = (self.config.operating.load_range[0] * 100 +
                       (self.config.operating.load_range[1] - self.config.operating.load_range[0]) * 100 * np.random.rand())
        load_factor = 0.3 + 0.7 * (load_percent / 100)

        # Temperature (40-80°C)
        temperature_C = (self.config.operating.temp_range[0] +
                        (self.config.operating.temp_range[1] - self.config.operating.temp_range[0]) * np.random.rand())
        temp_factor = 0.9 + 0.2 * ((temperature_C - self.config.operating.temp_range[0]) /
                                   (self.config.operating.temp_range[1] - self.config.operating.temp_range[0]))

        # Combined operating factor
        operating_factor = load_factor * temp_factor
        amp_base = (0.2 + 0.1 * np.random.rand()) * operating_factor

        return Omega, omega, speed_variation, load_factor, load_percent, temperature_C, temp_factor, operating_factor, amp_base

    def _configure_physics(self, Omega: float, temperature_C: float, load_factor: float) -> Tuple:
        """Configure physics-based parameters."""
        if self.config.physics.enabled and self.config.physics.calculate_sommerfeld:
            # Calculate Sommerfeld from operating conditions (PHYSICALLY CORRECT)
            viscosity_factor = np.exp(-0.03 * (temperature_C - 60))
            speed_factor = Omega / self.config.signal.Omega_base
            load_factor_somm = 1.0 / load_factor

            sommerfeld = (self.config.physics.sommerfeld_base * viscosity_factor *
                         speed_factor * load_factor_somm)
            sommerfeld = np.clip(sommerfeld, 0.05, 0.5)
        else:
            sommerfeld = self.config.physics.sommerfeld_base + (np.random.rand() - 0.5) * 0.2

        reynolds = (self.config.physics.reynolds_range[0] +
                   (self.config.physics.reynolds_range[1] - self.config.physics.reynolds_range[0]) * np.random.rand())

        clearance_ratio = (self.config.physics.clearance_ratio_range[0] +
                          (self.config.physics.clearance_ratio_range[1] - self.config.physics.clearance_ratio_range[0]) * np.random.rand())

        physics_factor = np.sqrt(sommerfeld / self.config.physics.sommerfeld_base)

        return sommerfeld, reynolds, clearance_ratio, physics_factor

    def _configure_transients(self) -> Tuple:
        """Configure non-stationary transient behavior."""
        transient_modulation = np.ones(self.config.signal.N)
        transient_type = 'none'
        transient_params = {}

        if self.config.transient.enabled and np.random.rand() < self.config.transient.probability:
            transient_type = np.random.choice(self.config.transient.types)

            if transient_type == 'speed_ramp':
                ramp_start_idx = int(0.2 * self.config.signal.N)
                ramp_end_idx = int(0.6 * self.config.signal.N)
                speed_mult = np.linspace(0.85, 1.15, ramp_end_idx - ramp_start_idx)
                transient_modulation[ramp_start_idx:ramp_end_idx] = speed_mult
                transient_params = {'start_idx': ramp_start_idx, 'end_idx': ramp_end_idx, 'speed_range': [0.85, 1.15]}

            elif transient_type == 'load_step':
                step_idx = int(0.4 * self.config.signal.N)
                transient_modulation[:step_idx] = 0.7
                transient_modulation[step_idx:] = 1.0
                transient_params = {'step_idx': step_idx, 'load_values': [0.7, 1.0]}

            elif transient_type == 'thermal_expansion':
                thermal_time_const = 0.3 * self.config.signal.N
                transient_modulation = 0.9 + 0.2 * (1 - np.exp(-np.arange(self.config.signal.N) / thermal_time_const))
                transient_params = {'time_constant': thermal_time_const}

        return transient_modulation, transient_type, transient_params

    def _apply_augmentation(self, x: np.ndarray, is_augmented: bool) -> Dict[str, Any]:
        """Apply data augmentation if enabled."""
        aug_params = {'method': 'none'}

        if is_augmented and self.config.augmentation.enabled:
            aug_method = np.random.choice(self.config.augmentation.methods)

            if aug_method == 'time_shift':
                shift_max = int(self.config.augmentation.time_shift_max * self.config.signal.N)
                shift_samples = np.random.randint(-shift_max, shift_max)
                x[:] = np.roll(x, shift_samples)
                aug_params = {'method': 'time_shift', 'shift_samples': int(shift_samples)}

            elif aug_method == 'amplitude_scale':
                scale_range = self.config.augmentation.amplitude_scale_range
                scale_factor = scale_range[0] + (scale_range[1] - scale_range[0]) * np.random.rand()
                x[:] = x * scale_factor
                aug_params = {'method': 'amplitude_scale', 'scale_factor': float(scale_factor)}

            elif aug_method == 'noise_injection':
                noise_range = self.config.augmentation.extra_noise_range
                extra_noise_level = noise_range[0] + (noise_range[1] - noise_range[0]) * np.random.rand()
                extra_noise = extra_noise_level * np.random.randn(self.config.signal.N)
                x[:] += extra_noise
                aug_params = {'method': 'noise_injection', 'noise_level': float(extra_noise_level)}

        return aug_params

    def save_dataset(
        self,
        dataset: Dict,
        output_dir: Optional[Path] = None,
        format: str = 'hdf5',  # DEFAULT='hdf5' (recommended for performance)
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    ) -> Dict[str, Path]:
        """
        Save generated dataset to disk in .mat and/or HDF5 format.

        Args:
            dataset: Dataset from generate_dataset()
            output_dir: Output directory (uses config if None)
            format: Output format - 'mat', 'hdf5', or 'both' (default: 'hdf5')
                HDF5 is recommended: 25× faster loading, 30% smaller, single file
            train_val_test_split: Train/val/test split ratios for HDF5 format

        Returns:
            Dictionary with paths to saved files:
            - 'mat_dir': Path to directory with .mat files (if format='mat' or 'both')
            - 'hdf5': Path to HDF5 file (if format='hdf5' or 'both')

        Examples:
            >>> # Backward compatible - saves .mat files only (default behavior)
            >>> generator.save_dataset(dataset, output_dir='data/processed')

            >>> # Save as HDF5 only
            >>> paths = generator.save_dataset(dataset, format='hdf5')

            >>> # Save both formats
            >>> paths = generator.save_dataset(dataset, format='both')
        """
        if output_dir is None:
            output_dir = Path(self.config.output_dir)
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = {}

        # Save as .mat files (original format - for backward compatibility)
        if format in ['mat', 'both']:
            # For backward compatibility: save directly in output_dir if format='mat'
            # Otherwise save in subdirectory
            if format == 'mat':
                mat_dir = output_dir
            else:
                mat_dir = output_dir / 'mat_files'
                mat_dir.mkdir(exist_ok=True)

            logger.info(f"Saving .mat files to: {mat_dir}")

            for i, (signal, metadata, label) in enumerate(zip(
                dataset['signals'], dataset['metadata'], dataset['labels']
            )):
                # Determine filename
                signal_idx = (i % self.config.num_signals_per_fault) + 1
                if metadata.is_augmented:
                    filename = f"{label}_{signal_idx:03d}_aug.mat"
                else:
                    filename = f"{label}_{signal_idx:03d}.mat"

                filepath = mat_dir / filename

                # Prepare MATLAB-compatible structure
                mat_data = {
                    'x': signal,
                    'fs': metadata.fs,
                    'fault': label,
                    'metadata': self._metadata_to_matlab_struct(metadata)
                }

                # Save .mat file
                savemat(filepath, mat_data, do_compression=True)

            logger.info(f"✓ Saved {len(dataset['signals'])} .mat files")
            saved_paths['mat_dir'] = mat_dir

        # Save as HDF5 file (new format - faster, more efficient)
        if format in ['hdf5', 'both']:
            hdf5_path = output_dir / 'dataset.h5'
            logger.info(f"Saving HDF5 file to: {hdf5_path}")

            saved_paths['hdf5'] = self._save_as_hdf5(
                dataset,
                hdf5_path,
                train_val_test_split
            )

            logger.info(f"✓ Saved HDF5 file: {hdf5_path}")

        return saved_paths

    def _metadata_to_matlab_struct(self, metadata: SignalMetadata) -> Dict:
        """Convert metadata to MATLAB-compatible structure."""
        return asdict(metadata)

    def _save_as_hdf5(
        self,
        dataset: Dict,
        output_path: Path,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)
    ) -> Path:
        """
        Save dataset as HDF5 file with train/val/test splits.

        This creates an HDF5 file compatible with dash_app and training scripts:
        - f['train']['signals'] and f['train']['labels']
        - f['val']['signals'] and f['val']['labels']
        - f['test']['signals'] and f['test']['labels']

        Args:
            dataset: Dataset dictionary from generate_dataset()
            output_path: Path to HDF5 output file
            split_ratios: (train, val, test) split ratios (default: 0.7, 0.15, 0.15)

        Returns:
            Path to created HDF5 file

        Example:
            >>> generator = SignalGenerator(config)
            >>> dataset = generator.generate_dataset()
            >>> hdf5_path = generator._save_as_hdf5(
            ...     dataset,
            ...     Path('data/processed/dataset.h5')
            ... )
        """
        signals = dataset['signals']
        labels_str = dataset['labels']
        metadata = dataset['metadata']

        # Convert signals to numpy array if it's a list
        if isinstance(signals, list):
            signals = np.array(signals, dtype=np.float32)
        elif not isinstance(signals, np.ndarray):
            raise TypeError(f"Signals must be list or numpy array, got {type(signals)}")

        # Validate signals array
        if len(signals) == 0:
            raise ValueError("Cannot save empty dataset - no signals generated")

        if signals.ndim != 2:
            raise ValueError(f"Signals must be 2D array (num_samples, signal_length), got shape {signals.shape}")

        # Convert string labels to integers using FAULT_TYPES mapping
        label_to_idx = {label: idx for idx, label in enumerate(FAULT_TYPES)}

        # Validate all labels are known
        unknown_labels = set(labels_str) - set(FAULT_TYPES)
        if unknown_labels:
            raise ValueError(f"Unknown fault types in dataset: {unknown_labels}. Valid types: {FAULT_TYPES}")

        labels = np.array([label_to_idx[label] for label in labels_str], dtype=np.int64)

        # Create stratified splits to ensure each split has all classes
        train_ratio, val_ratio, test_ratio = split_ratios

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            signals, labels,
            test_size=test_ratio,
            stratify=labels,
            random_state=self.config.rng_seed
        )

        # Second split: separate train and val from temp
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=self.config.rng_seed
        )

        # Create HDF5 file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, 'w') as f:
            # Store global attributes
            f.attrs['num_classes'] = NUM_CLASSES
            f.attrs['sampling_rate'] = SAMPLING_RATE
            f.attrs['signal_length'] = int(signals.shape[1])
            f.attrs['generation_date'] = datetime.now().isoformat()
            f.attrs['split_ratios'] = split_ratios
            f.attrs['rng_seed'] = self.config.rng_seed

            # Create train group
            train_grp = f.create_group('train')
            train_grp.create_dataset(
                'signals',
                data=X_train,
                compression='gzip',
                compression_opts=4
            )
            train_grp.create_dataset('labels', data=y_train)
            train_grp.attrs['num_samples'] = len(X_train)

            # Create val group
            val_grp = f.create_group('val')
            val_grp.create_dataset(
                'signals',
                data=X_val,
                compression='gzip',
                compression_opts=4
            )
            val_grp.create_dataset('labels', data=y_val)
            val_grp.attrs['num_samples'] = len(X_val)

            # Create test group
            test_grp = f.create_group('test')
            test_grp.create_dataset(
                'signals',
                data=X_test,
                compression='gzip',
                compression_opts=4
            )
            test_grp.create_dataset('labels', data=y_test)
            test_grp.attrs['num_samples'] = len(X_test)

            # Store metadata as JSON (optional, for reference)
            if metadata:
                metadata_json = [json.dumps(asdict(m)) for m in metadata]
                dt = h5py.string_dtype(encoding='utf-8')
                f.create_dataset('metadata', data=metadata_json, dtype=dt)

        logger.info(
            f"Created HDF5 with splits - Train: {len(X_train)}, "
            f"Val: {len(X_val)}, Test: {len(X_test)}"
        )

        return output_path
