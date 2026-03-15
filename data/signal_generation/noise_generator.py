"""
7-layer independent noise model generator.

Extracted from: data/signal_generator.py
"""

import numpy as np
from typing import Dict, Tuple

from config.data_config import DataConfig


class NoiseGenerator:
    """7-layer independent noise model generator."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.fs = config.signal.fs
        self.T = config.signal.T
        self.N = config.signal.N
        self.t = np.arange(0, self.N) / self.fs

    def apply_noise_layers(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, bool]]:
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
            noise_meas = noise_cfg.levels["measurement"] * np.random.randn(self.N)
            x_noisy += noise_meas
            applied["measurement"] = True
        else:
            applied["measurement"] = False

        # 2. EMI (electromagnetic interference - 50/60 Hz)
        if noise_cfg.emi:
            emi_freq = 50 + 10 * np.random.rand()
            emi_amp = noise_cfg.levels["emi"] * (1 + 0.5 * np.random.rand())
            emi_signal = emi_amp * np.sin(
                2 * np.pi * emi_freq * self.t + np.random.rand() * 2 * np.pi
            )
            x_noisy += emi_signal
            applied["emi"] = True
        else:
            applied["emi"] = False

        # 3. Pink noise (1/f)
        if noise_cfg.pink:
            pink_noise = np.cumsum(np.random.randn(self.N))
            pink_noise = noise_cfg.levels["pink"] * (
                pink_noise / (np.std(pink_noise) + 1e-10)
            )
            x_noisy += pink_noise
            applied["pink"] = True
        else:
            applied["pink"] = False

        # 4. Environmental drift
        if noise_cfg.drift:
            drift_period = 1.5
            drift = noise_cfg.levels["drift"] * np.sin(
                2 * np.pi * (1 / drift_period) * self.t
            )
            x_noisy += drift
            applied["drift"] = True
        else:
            applied["drift"] = False

        # 5. Quantization noise (ADC resolution)
        if noise_cfg.quantization:
            quant_step = noise_cfg.levels["quantization_step"]
            x_noisy = np.round(x_noisy / quant_step) * quant_step
            applied["quantization"] = True
        else:
            applied["quantization"] = False

        # 6. Sensor drift (cumulative offset)
        if noise_cfg.sensor_drift:
            sensor_drift_rate = noise_cfg.levels["sensor_drift_rate"] / self.T
            sensor_offset = sensor_drift_rate * self.t
            x_noisy += sensor_offset
            applied["sensor_drift"] = True
        else:
            applied["sensor_drift"] = False

        # 7. Aliasing artifacts (10% probability)
        if np.random.rand() < noise_cfg.aliasing:
            alias_freq = self.fs / 2 + 100 + 200 * np.random.rand()
            alias_signal = 0.005 * np.sin(2 * np.pi * alias_freq * self.t)
            x_noisy += alias_signal
            applied["aliasing"] = True
        else:
            applied["aliasing"] = False

        # 8. Impulse noise (sporadic impacts)
        if noise_cfg.impulse:
            num_impulses = int(noise_cfg.levels["impulse_rate"] * self.T)
            for imp in range(num_impulses):
                imp_pos = np.random.randint(0, max(1, self.N - 5))
                imp_amp = 0.02 + 0.03 * np.random.rand()
                imp_len = min(5, self.N - imp_pos)
                decay = np.exp(-0.3 * np.arange(imp_len))
                x_noisy[imp_pos : imp_pos + imp_len] += (
                    imp_amp * decay * np.random.randn(imp_len)
                )
            applied["impulse"] = True
        else:
            applied["impulse"] = False

        return x_noisy, applied
