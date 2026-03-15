"""
Physics-based fault modeler for hydrodynamic bearings.

Implements equations from Section 7.3 of technical report.
Extracted from: data/signal_generator.py
"""

import numpy as np
from scipy import signal as sp_signal

from config.data_config import DataConfig


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
        speed_variation: float,
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
        if fault_type == "sain":
            return np.zeros(self.N)

        elif fault_type == "desalignement":
            phase_2X = np.random.rand() * 2 * np.pi
            phase_3X = np.random.rand() * 2 * np.pi
            misalign_2X = 0.35 * np.sin(2 * omega * self.t + phase_2X)
            misalign_3X = 0.20 * np.sin(3 * omega * self.t + phase_3X)
            x_fault = severity_curve * (misalign_2X + misalign_3X) * transient_modulation
            return x_fault

        elif fault_type == "desequilibre":
            phase_1X = np.random.rand() * 2 * np.pi
            imbalance_1X = (
                0.5
                * load_factor
                * np.sin(omega * self.t + phase_1X)
                * (speed_variation**2)
            )
            x_fault = severity_curve * imbalance_1X * transient_modulation
            return x_fault

        elif fault_type == "jeu":
            sub_freq = (0.43 + 0.05 * np.random.rand()) * Omega
            clearance_sub = 0.25 * temp_factor * np.sin(2 * np.pi * sub_freq * self.t)
            clearance_1X = 0.18 * np.sin(omega * self.t)
            clearance_2X = 0.10 * np.sin(2 * omega * self.t)
            x_fault = (
                severity_curve
                * (clearance_sub + clearance_1X + clearance_2X)
                * transient_modulation
            )
            return x_fault

        elif fault_type == "lubrification":
            stick_slip_freq = 2 + 3 * np.random.rand()
            stick_slip = (
                0.30
                * temp_factor
                * (0.3 / sommerfeld)
                * np.sin(2 * np.pi * stick_slip_freq * self.t)
            )
            x_fault = severity_curve * stick_slip * transient_modulation
            impact_rate = int(1 + 3 * np.mean(severity_curve))
            for j in range(impact_rate):
                impact_pos = np.random.randint(0, self.N - 20)
                impact_amp = 0.5 * np.mean(severity_curve)
                impact_len = min(20, self.N - impact_pos)
                decay = np.exp(-0.4 * np.arange(impact_len))
                x_fault[impact_pos : impact_pos + impact_len] += (
                    impact_amp * decay * np.random.randn(impact_len)
                )
            return x_fault

        elif fault_type == "cavitation":
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
                burst = (
                    0.6
                    * np.mean(severity_curve)
                    * np.sin(2 * np.pi * burst_freq * burst_t)
                    * np.exp(-100 * burst_t)
                    * hann_window
                )
                x_fault[pos : pos + burst_len] += burst
            return x_fault

        elif fault_type == "usure":
            wear_noise = (
                0.25 * operating_factor * physics_factor * np.random.randn(self.N)
            )
            asperity_harm = 0.12 * (
                np.sin(omega * self.t) + 0.5 * np.sin(2 * omega * self.t)
            )
            wear_mod_freq = 0.5 + 1.5 * np.random.rand()
            wear_mod = 1 + 0.3 * np.sin(2 * np.pi * wear_mod_freq * self.t)
            x_fault = (
                severity_curve
                * (wear_noise + asperity_harm)
                * wear_mod
                * transient_modulation
            )
            return x_fault

        elif fault_type == "oilwhirl":
            whirl_freq_ratio = 0.42 + 0.06 * np.random.rand()
            whirl_freq = whirl_freq_ratio * Omega
            whirl_amp = 0.40 * (1 / np.sqrt(sommerfeld))
            whirl_signal = whirl_amp * np.sin(2 * np.pi * whirl_freq * self.t)
            subsync_mod_freq = whirl_freq * 0.5
            subsync_mod = 1 + 0.2 * np.sin(2 * np.pi * subsync_mod_freq * self.t)
            x_fault = (
                severity_curve * whirl_signal * subsync_mod * transient_modulation
            )
            return x_fault

        elif fault_type == "mixed_misalign_imbalance":
            misalign_sev = np.mean(severity_curve)
            phase_2X = np.random.rand() * 2 * np.pi
            phase_3X = np.random.rand() * 2 * np.pi
            misalign_2X = 0.25 * misalign_sev * np.sin(2 * omega * self.t + phase_2X)
            misalign_3X = 0.15 * misalign_sev * np.sin(3 * omega * self.t + phase_3X)

            imbalance_sev = np.mean(severity_curve)
            phase_1X = np.random.rand() * 2 * np.pi
            imbalance_1X = (
                0.35
                * imbalance_sev
                * load_factor
                * np.sin(omega * self.t + phase_1X)
                * (speed_variation**2)
            )

            combined = severity_curve * (misalign_2X + misalign_3X + imbalance_1X)
            x_fault = combined * transient_modulation
            return x_fault

        elif fault_type == "mixed_wear_lube":
            wear_sev = np.mean(severity_curve)
            wear_noise = (
                0.18
                * wear_sev
                * operating_factor
                * physics_factor
                * np.random.randn(self.N)
            )
            asperity_harm = 0.08 * wear_sev * (
                np.sin(omega * self.t) + 0.5 * np.sin(2 * omega * self.t)
            )

            lube_sev = np.mean(severity_curve)
            stick_slip_freq = 2 + 3 * np.random.rand()
            stick_slip = (
                0.20
                * lube_sev
                * temp_factor
                * (0.3 / sommerfeld)
                * np.sin(2 * np.pi * stick_slip_freq * self.t)
            )

            x_fault = (
                severity_curve
                * (wear_noise + asperity_harm + stick_slip)
                * transient_modulation
            )
            contact_rate = int(2 + 3 * lube_sev)
            for jj in range(contact_rate):
                contact_pos = np.random.randint(0, max(1, self.N - 10))
                contact_amp = 0.4 * lube_sev
                contact_len = min(10, self.N - contact_pos)
                decay = np.exp(-0.5 * np.arange(contact_len))
                x_fault[contact_pos : contact_pos + contact_len] += (
                    contact_amp * decay * np.random.randn(contact_len)
                )
            return x_fault

        elif fault_type == "mixed_cavit_jeu":
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
                burst = (
                    0.5
                    * cavit_sev
                    * np.sin(2 * np.pi * burst_freq * burst_t)
                    * np.exp(-100 * burst_t)
                    * hann_window
                )
                x_fault[pos : pos + burst_len] += burst

            clearance_sev = np.mean(severity_curve)
            sub_freq = (0.43 + 0.05 * np.random.rand()) * Omega
            clearance_sub = (
                0.22
                * clearance_sev
                * temp_factor
                * np.sin(2 * np.pi * sub_freq * self.t)
            )
            clearance_1X = 0.15 * clearance_sev * np.sin(omega * self.t)

            combined = severity_curve * (clearance_sub + clearance_1X)
            x_fault += combined * transient_modulation
            return x_fault

        else:
            raise ValueError(f"Unknown fault type: {fault_type}")

    # ── Advanced Physics Effects (V2) ─────────────────────────────────

    def apply_advanced_physics(
        self,
        x: np.ndarray,
        omega: float,
        Omega: float,
        temperature_C: float,
    ) -> np.ndarray:
        """
        Apply all enabled advanced physics effects to a signal.

        Args:
            x: Input signal [N,]
            omega: Angular velocity (rad/s)
            Omega: Rotational speed (Hz)
            temperature_C: Operating temperature (°C)

        Returns:
            Modified signal with advanced physics applied
        """
        adv = self.config.advanced_physics

        if adv.speed_transients:
            x = self._apply_speed_transient(x, omega)

        if adv.speed_fluctuation:
            x = self._apply_speed_fluctuation(x, omega, adv.speed_fluctuation_pct,
                                               adv.speed_fluctuation_bandwidth_hz)

        if adv.rotor_dynamics:
            x = self._apply_rotor_dynamics(x, Omega, adv.critical_speed_hz,
                                            adv.damping_ratio)

        if adv.cross_coupling:
            x = self._apply_cross_coupling(x, omega, adv.coupling_ratio,
                                            adv.phase_offset_deg)

        if adv.thermal_growth:
            x = self._apply_thermal_growth(x, temperature_C,
                                            adv.thermal_growth_rate,
                                            adv.thermal_time_constant_s)

        if adv.axial_vibration:
            x = self._apply_axial_vibration(x, omega, adv.axial_coupling_ratio,
                                             adv.axial_thrust_freq_ratio)

        return x

    def _apply_speed_transient(
        self,
        x: np.ndarray,
        omega: float,
    ) -> np.ndarray:
        """Apply run-up / coast-down speed profile.

        Creates a linear or exponential frequency sweep in the first
        and/or last portion of the signal, simulating machine startup
        or shutdown transients.
        """
        adv = self.config.advanced_physics
        ramp_samples = int(adv.speed_ramp_duration_s * self.fs)
        ramp_samples = min(ramp_samples, self.N // 4)  # Max 25% of signal

        if ramp_samples < 10:
            return x

        x_out = x.copy()

        # Decide transient type: 0=run-up, 1=coast-down, 2=both
        transient_type = np.random.randint(0, 3)

        if adv.speed_ramp_type == 'exponential':
            ramp_up = 1 - np.exp(-3 * np.linspace(0, 1, ramp_samples))
            ramp_down = np.exp(-3 * np.linspace(0, 1, ramp_samples))
        else:  # linear
            ramp_up = np.linspace(0.1, 1.0, ramp_samples)
            ramp_down = np.linspace(1.0, 0.1, ramp_samples)

        if transient_type in (0, 2):  # run-up
            # Instantaneous frequency sweep: omega varies during ramp
            phase_integral = np.cumsum(omega * ramp_up) / self.fs
            freq_mod = np.sin(phase_integral)
            # Amplitude ramp
            x_out[:ramp_samples] *= ramp_up
            # Add swept tone component
            x_out[:ramp_samples] += 0.05 * ramp_up * freq_mod

        if transient_type in (1, 2):  # coast-down
            phase_integral = np.cumsum(omega * ramp_down) / self.fs
            freq_mod = np.sin(phase_integral)
            x_out[-ramp_samples:] *= ramp_down
            x_out[-ramp_samples:] += 0.05 * ramp_down * freq_mod

        return x_out

    def _apply_speed_fluctuation(
        self,
        x: np.ndarray,
        omega: float,
        fluctuation_pct: float,
        bandwidth_hz: float,
    ) -> np.ndarray:
        """Apply realistic speed fluctuation via frequency modulation.

        Models real-world speed variations from load transients, controller
        response, and mechanical coupling effects. Implemented as
        low-pass-filtered random walk modulating instantaneous frequency.
        """
        # Generate random walk for speed variation
        speed_noise = np.cumsum(np.random.randn(self.N)) / np.sqrt(self.N)
        speed_noise = speed_noise / (np.std(speed_noise) + 1e-10) * fluctuation_pct

        # Low-pass filter to bandwidth
        from scipy.signal import butter, filtfilt
        nyq = self.fs / 2
        if bandwidth_hz < nyq:
            b, a = butter(2, bandwidth_hz / nyq, btype='low')
            speed_noise = filtfilt(b, a, speed_noise)

        # Apply as amplitude modulation (simplified frequency modulation)
        modulation = 1 + speed_noise
        return x * modulation

    def _apply_rotor_dynamics(
        self,
        x: np.ndarray,
        Omega: float,
        critical_speed_hz: float,
        damping_ratio: float,
    ) -> np.ndarray:
        """Apply critical speed resonance amplification.

        When operating speed is near a critical speed, vibration amplitude
        is amplified by the resonance transfer function:
            H(f) = 1 / sqrt((1 - r²)² + (2·ζ·r)²)
        where r = f/f_critical and ζ is the damping ratio.
        """
        r = Omega / critical_speed_hz
        # Transfer function magnitude
        denominator = np.sqrt((1 - r**2)**2 + (2 * damping_ratio * r)**2)
        amplification = 1.0 / max(denominator, 0.1)  # Cap at 10x

        # Also add resonance-excited harmonics
        resonance_component = (
            0.05 * (amplification - 1) *
            np.sin(2 * np.pi * critical_speed_hz * self.t +
                   np.random.rand() * 2 * np.pi)
        )

        return x * min(amplification, 5.0) + resonance_component

    def _apply_cross_coupling(
        self,
        x: np.ndarray,
        omega: float,
        coupling_ratio: float,
        phase_offset_deg: float,
    ) -> np.ndarray:
        """Apply cross-coupling stiffness effects.

        Hydrodynamic bearings have asymmetric stiffness matrices where
        the cross-coupled stiffness Kxy creates a force perpendicular
        to displacement. This adds a 90°-phase-shifted component.
        """
        phase_rad = np.deg2rad(phase_offset_deg)

        # Extract dominant 1X component and create cross-coupled version
        # Use Hilbert transform for phase shifting
        from scipy.signal import hilbert
        analytic = hilbert(x)
        x_shifted = np.real(analytic * np.exp(1j * phase_rad))

        return x + coupling_ratio * x_shifted

    def _apply_thermal_growth(
        self,
        x: np.ndarray,
        temperature_C: float,
        growth_rate: float,
        time_constant_s: float,
    ) -> np.ndarray:
        """Apply thermal growth effects.

        Shaft thermal expansion changes the bearing clearance over time,
        causing a progressive amplitude drift. Modeled as an exponential
        approach to steady-state thermal equilibrium.
        """
        # Temperature rise profile (exponential approach)
        delta_T = temperature_C - 40.0  # Difference from cold start
        thermal_profile = delta_T * (1 - np.exp(-self.t / time_constant_s))

        # Clearance change → amplitude scaling
        # As clearance decreases, vibration increases
        clearance_factor = 1 + growth_rate * thermal_profile
        return x * clearance_factor

    def _apply_axial_vibration(
        self,
        x: np.ndarray,
        omega: float,
        coupling_ratio: float,
        thrust_freq_ratio: float,
    ) -> np.ndarray:
        """Apply axial vibration component.

        Real bearings transmit axial vibration through thrust faces
        and coupling misalignment. This adds an axial component
        proportional to radial amplitude at thrust-related frequencies.
        """
        axial_freq = omega * thrust_freq_ratio
        phase = np.random.rand() * 2 * np.pi

        # Axial component: proportional to RMS of radial signal
        rms = np.sqrt(np.mean(x**2)) + 1e-10
        axial = coupling_ratio * rms * np.sin(axial_freq * self.t + phase)

        # Add 2X axial harmonic (common in angular misalignment)
        axial += 0.3 * coupling_ratio * rms * np.sin(
            2 * axial_freq * self.t + np.random.rand() * 2 * np.pi
        )

        return x + axial
