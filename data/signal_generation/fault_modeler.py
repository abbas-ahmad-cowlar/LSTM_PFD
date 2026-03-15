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
