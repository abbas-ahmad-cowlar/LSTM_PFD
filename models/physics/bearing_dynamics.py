"""
Bearing Dynamics Physics Model

This module encodes bearing fault physics as differentiable functions for use in
Physics-Informed Neural Networks (PINNs). It provides calculations for:
- Characteristic bearing frequencies (FTF, BPFO, BPFI, BSF)
- Dimensionless numbers (Sommerfeld, Reynolds)
- Operating condition validation

References:
- Bearing fault frequency equations from ISO 15243:2017
- Lubrication theory from Hamrock et al., "Fundamentals of Fluid Film Lubrication"
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import numpy as np
import torch
from typing import Dict, Union, Tuple


class BearingDynamics:
    """
    Encapsulates bearing physics calculations for fault diagnosis.

    Supports both numpy and torch tensors for compatibility with:
    - Data preprocessing (numpy)
    - Neural network training with autodiff (torch)
    """

    def __init__(self, bearing_params: Dict[str, float] = None):
        """
        Initialize bearing dynamics calculator.

        Args:
            bearing_params: Dictionary containing bearing geometry
                - n_balls: Number of rolling elements (default: 9)
                - ball_diameter: Ball diameter in mm (default: 7.94)
                - pitch_diameter: Pitch diameter in mm (default: 39.04)
                - contact_angle: Contact angle in degrees (default: 0)
                - shaft_radius: Shaft radius in mm (default: 15)
                - radial_clearance: Radial clearance in μm (default: 20)
        """
        if bearing_params is None:
            # Default parameters for SKF 6205 bearing (common in dataset)
            bearing_params = {
                'n_balls': 9,
                'ball_diameter': 7.94,  # mm
                'pitch_diameter': 39.04,  # mm
                'contact_angle': 0.0,  # degrees (deep groove ball bearing)
                'shaft_radius': 15.0,  # mm
                'radial_clearance': 20.0,  # μm
            }

        self.params = bearing_params
        self._contact_angle_rad = np.radians(bearing_params['contact_angle'])

    def characteristic_frequencies(
        self,
        rpm: Union[float, np.ndarray, torch.Tensor],
        return_torch: bool = False
    ) -> Dict[str, Union[float, np.ndarray, torch.Tensor]]:
        """
        Calculate characteristic bearing fault frequencies.

        These frequencies represent the rate at which defects produce
        periodic impacts in the vibration signal.

        Args:
            rpm: Shaft rotation speed in RPM (can be scalar, array, or tensor)
            return_torch: If True, return torch tensors instead of numpy

        Returns:
            Dictionary with keys:
                - 'FTF': Fundamental Train Frequency (cage frequency)
                - 'BPFO': Ball Pass Frequency Outer race
                - 'BPFI': Ball Pass Frequency Inner race
                - 'BSF': Ball Spin Frequency
                - 'shaft_freq': Shaft rotation frequency (1X)

        Physics:
            For a bearing with n balls, ball diameter d_b, pitch diameter d_p,
            and contact angle β:

            Shaft frequency: f_s = RPM / 60

            FTF = (f_s / 2) * (1 - (d_b / d_p) * cos(β))
            BPFO = (n / 2) * f_s * (1 - (d_b / d_p) * cos(β))
            BPFI = (n / 2) * f_s * (1 + (d_b / d_p) * cos(β))
            BSF = (d_p / (2 * d_b)) * f_s * (1 - ((d_b / d_p) * cos(β))^2)
        """
        n = self.params['n_balls']
        d_b = self.params['ball_diameter']
        d_p = self.params['pitch_diameter']
        beta = self._contact_angle_rad

        # Check if input is torch tensor
        is_torch = isinstance(rpm, torch.Tensor)

        if is_torch:
            # Use torch operations for autodiff compatibility
            cos_beta = torch.tensor(np.cos(beta), dtype=rpm.dtype, device=rpm.device)
            shaft_freq = rpm / 60.0

            ratio = (d_b / d_p) * cos_beta

            ftf = (shaft_freq / 2.0) * (1.0 - ratio)
            bpfo = (n / 2.0) * shaft_freq * (1.0 - ratio)
            bpfi = (n / 2.0) * shaft_freq * (1.0 + ratio)
            bsf = (d_p / (2.0 * d_b)) * shaft_freq * (1.0 - ratio ** 2)

        else:
            # Use numpy operations
            if isinstance(rpm, torch.Tensor):
                rpm = rpm.cpu().numpy()

            shaft_freq = rpm / 60.0
            cos_beta = np.cos(beta)
            ratio = (d_b / d_p) * cos_beta

            ftf = (shaft_freq / 2.0) * (1.0 - ratio)
            bpfo = (n / 2.0) * shaft_freq * (1.0 - ratio)
            bpfi = (n / 2.0) * shaft_freq * (1.0 + ratio)
            bsf = (d_p / (2.0 * d_b)) * shaft_freq * (1.0 - ratio ** 2)

            if return_torch:
                ftf = torch.tensor(ftf, dtype=torch.float32)
                bpfo = torch.tensor(bpfo, dtype=torch.float32)
                bpfi = torch.tensor(bpfi, dtype=torch.float32)
                bsf = torch.tensor(bsf, dtype=torch.float32)
                shaft_freq = torch.tensor(shaft_freq, dtype=torch.float32)

        return {
            'FTF': ftf,
            'BPFO': bpfo,
            'BPFI': bpfi,
            'BSF': bsf,
            'shaft_freq': shaft_freq
        }

    def sommerfeld_number(
        self,
        load: Union[float, np.ndarray, torch.Tensor],
        speed: Union[float, np.ndarray, torch.Tensor],
        viscosity: Union[float, np.ndarray, torch.Tensor],
        return_torch: bool = False
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Calculate Sommerfeld number - dimensionless parameter in lubrication theory.

        The Sommerfeld number characterizes the lubrication regime:
        - S < 0.1: Boundary lubrication (metal-to-metal contact)
        - 0.1 < S < 1: Mixed lubrication
        - S > 1: Hydrodynamic lubrication (full film)

        Args:
            load: Applied radial load in Newtons
            speed: Shaft speed in RPM
            viscosity: Dynamic viscosity in Pa·s (default oil ~0.03 Pa·s at 40°C)
            return_torch: If True, return torch tensor

        Returns:
            Sommerfeld number (dimensionless)

        Physics:
            S = (μ * N * R) / (P * C^2)
            where:
                μ = dynamic viscosity (Pa·s)
                N = rotational speed (rev/s)
                R = shaft radius (m)
                P = bearing pressure = Load / (2 * R * L) (Pa)
                C = radial clearance (m)

            Simplified for journal bearing:
            S = (μ * N) / P * (R / C)^2
        """
        # Input validation (check for positive values)
        def validate_positive(val, name):
            if isinstance(val, (int, float)) and val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")
            elif isinstance(val, np.ndarray) and np.any(val <= 0):
                raise ValueError(f"{name} must be positive (found non-positive values)")
            elif isinstance(val, torch.Tensor) and torch.any(val <= 0):
                raise ValueError(f"{name} must be positive (found non-positive values)")

        validate_positive(load, "Load")
        validate_positive(speed, "Speed")
        validate_positive(viscosity, "Viscosity")

        R = self.params['shaft_radius'] / 1000.0  # Convert mm to m
        C = self.params['radial_clearance'] / 1e6  # Convert μm to m

        # Assumed bearing length (SKF 6205: 15mm width) - TODO: make this configurable
        L = 0.015  # m

        # Check if inputs are torch tensors
        is_torch = isinstance(load, torch.Tensor) or isinstance(speed, torch.Tensor)

        if is_torch:
            # Convert to torch if needed
            if not isinstance(load, torch.Tensor):
                load = torch.tensor(load, dtype=torch.float32)
            if not isinstance(speed, torch.Tensor):
                speed = torch.tensor(speed, dtype=torch.float32)
            if not isinstance(viscosity, torch.Tensor):
                viscosity = torch.tensor(viscosity, dtype=torch.float32)

            # Ensure same device
            device = load.device if isinstance(load, torch.Tensor) else speed.device
            load = load.to(device)
            speed = speed.to(device)
            viscosity = viscosity.to(device)

            N = speed / 60.0  # Convert RPM to rev/s
            P = load / (2.0 * R * L)  # Bearing pressure

            S = (viscosity * N / P) * (R / C) ** 2

        else:
            # Numpy operations
            if isinstance(load, torch.Tensor):
                load = load.cpu().numpy()
            if isinstance(speed, torch.Tensor):
                speed = speed.cpu().numpy()
            if isinstance(viscosity, torch.Tensor):
                viscosity = viscosity.cpu().numpy()

            N = speed / 60.0
            P = load / (2.0 * R * L)

            # Add small epsilon to prevent division by zero (standardized to 1e-8)
            P = np.maximum(P, 1e-8)

            S = (viscosity * N / (P + 1e-8)) * (R / C) ** 2

            if return_torch:
                S = torch.tensor(S, dtype=torch.float32)

        return S

    def reynolds_number(
        self,
        speed: Union[float, np.ndarray, torch.Tensor],
        viscosity: Union[float, np.ndarray, torch.Tensor],
        density: Union[float, np.ndarray, torch.Tensor] = 850.0,
        return_torch: bool = False
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Calculate Reynolds number - characterizes flow regime.

        Determines whether lubricant flow is laminar or turbulent:
        - Re < 2000: Laminar flow
        - 2000 < Re < 4000: Transition
        - Re > 4000: Turbulent flow

        Args:
            speed: Shaft speed in RPM
            viscosity: Dynamic viscosity in Pa·s
            density: Lubricant density in kg/m³ (default: 850 for mineral oil)
            return_torch: If True, return torch tensor

        Returns:
            Reynolds number (dimensionless)

        Physics:
            Re = (ρ * U * C) / μ
            where:
                ρ = fluid density (kg/m³)
                U = surface velocity = ω * R (m/s)
                C = characteristic length (radial clearance, m)
                μ = dynamic viscosity (Pa·s)
        """
        R = self.params['shaft_radius'] / 1000.0  # mm to m
        C = self.params['radial_clearance'] / 1e6  # μm to m

        is_torch = isinstance(speed, torch.Tensor) or isinstance(viscosity, torch.Tensor)

        if is_torch:
            if not isinstance(speed, torch.Tensor):
                speed = torch.tensor(speed, dtype=torch.float32)
            if not isinstance(viscosity, torch.Tensor):
                viscosity = torch.tensor(viscosity, dtype=torch.float32)
            if not isinstance(density, torch.Tensor):
                density = torch.tensor(density, dtype=torch.float32)

            device = speed.device if isinstance(speed, torch.Tensor) else viscosity.device
            speed = speed.to(device)
            viscosity = viscosity.to(device)
            density = density.to(device)

            omega = speed * 2.0 * np.pi / 60.0  # Convert RPM to rad/s
            U = omega * R  # Surface velocity

            Re = (density * U * C) / (viscosity + 1e-8)  # Add epsilon for stability

        else:
            if isinstance(speed, torch.Tensor):
                speed = speed.cpu().numpy()
            if isinstance(viscosity, torch.Tensor):
                viscosity = viscosity.cpu().numpy()
            if isinstance(density, torch.Tensor):
                density = density.cpu().numpy()

            omega = speed * 2.0 * np.pi / 60.0
            U = omega * R

            Re = (density * U * C) / (viscosity + 1e-8)

            if return_torch:
                Re = torch.tensor(Re, dtype=torch.float32)

        return Re

    def compute_all_physics_features(
        self,
        rpm: Union[float, np.ndarray, torch.Tensor],
        load: Union[float, np.ndarray, torch.Tensor],
        viscosity: Union[float, np.ndarray, torch.Tensor] = 0.03,
        density: float = 850.0,
        return_torch: bool = False
    ) -> Dict[str, Union[float, np.ndarray, torch.Tensor]]:
        """
        Compute all physics features for PINN models.

        This is a convenience function that computes all relevant physics
        features in one call.

        Args:
            rpm: Shaft speed in RPM
            load: Applied load in Newtons
            viscosity: Dynamic viscosity in Pa·s (default: 0.03 for typical oil)
            density: Lubricant density in kg/m³ (default: 850)
            return_torch: If True, return torch tensors

        Returns:
            Dictionary containing:
                - All characteristic frequencies (FTF, BPFO, BPFI, BSF, shaft_freq)
                - Sommerfeld number
                - Reynolds number
                - Lubrication regime (0=boundary, 1=mixed, 2=hydrodynamic)
                - Flow regime (0=laminar, 1=transition, 2=turbulent)
        """
        # Get characteristic frequencies
        freqs = self.characteristic_frequencies(rpm, return_torch=return_torch)

        # Calculate dimensionless numbers
        S = self.sommerfeld_number(load, rpm, viscosity, return_torch=return_torch)
        Re = self.reynolds_number(rpm, viscosity, density, return_torch=return_torch)

        # Determine regimes (useful for classification)
        if return_torch or isinstance(S, torch.Tensor):
            lubrication_regime = torch.zeros_like(S)
            lubrication_regime[S < 0.1] = 0  # Boundary
            lubrication_regime[(S >= 0.1) & (S < 1.0)] = 1  # Mixed
            lubrication_regime[S >= 1.0] = 2  # Hydrodynamic

            flow_regime = torch.zeros_like(Re)
            flow_regime[Re < 2000] = 0  # Laminar
            flow_regime[(Re >= 2000) & (Re < 4000)] = 1  # Transition
            flow_regime[Re >= 4000] = 2  # Turbulent
        else:
            lubrication_regime = np.zeros_like(S)
            lubrication_regime[S < 0.1] = 0
            lubrication_regime[(S >= 0.1) & (S < 1.0)] = 1
            lubrication_regime[S >= 1.0] = 2

            flow_regime = np.zeros_like(Re)
            flow_regime[Re < 2000] = 0
            flow_regime[(Re >= 2000) & (Re < 4000)] = 1
            flow_regime[Re >= 4000] = 2

        return {
            **freqs,  # Include all frequencies
            'sommerfeld': S,
            'reynolds': Re,
            'lubrication_regime': lubrication_regime,
            'flow_regime': flow_regime
        }


# Create default instance for convenience
default_bearing = BearingDynamics()


def characteristic_frequencies(rpm, bearing_params=None):
    """Convenience function for characteristic frequency calculation."""
    if bearing_params is None:
        return default_bearing.characteristic_frequencies(rpm)
    else:
        bearing = BearingDynamics(bearing_params)
        return bearing.characteristic_frequencies(rpm)


def sommerfeld_number(load, speed, viscosity, bearing_params=None):
    """Convenience function for Sommerfeld number calculation."""
    if bearing_params is None:
        return default_bearing.sommerfeld_number(load, speed, viscosity)
    else:
        bearing = BearingDynamics(bearing_params)
        return bearing.sommerfeld_number(load, speed, viscosity)


def reynolds_number(speed, viscosity, density=850.0, bearing_params=None):
    """Convenience function for Reynolds number calculation."""
    if bearing_params is None:
        return default_bearing.reynolds_number(speed, viscosity, density)
    else:
        bearing = BearingDynamics(bearing_params)
        return bearing.reynolds_number(speed, viscosity, density)


if __name__ == "__main__":
    # Example usage and validation
    print("=" * 60)
    print("Bearing Dynamics Physics Model - Validation")
    print("=" * 60)

    # Test with typical operating conditions
    rpm = 3600.0
    load = 500.0  # N
    viscosity = 0.03  # Pa·s (typical oil at 40°C)

    bd = BearingDynamics()

    print(f"\nOperating Conditions:")
    print(f"  Speed: {rpm} RPM")
    print(f"  Load: {load} N")
    print(f"  Viscosity: {viscosity} Pa·s")

    # Characteristic frequencies
    print(f"\nCharacteristic Frequencies:")
    freqs = bd.characteristic_frequencies(rpm)
    for name, freq in freqs.items():
        print(f"  {name}: {freq:.2f} Hz")

    # Dimensionless numbers
    print(f"\nDimensionless Numbers:")
    S = bd.sommerfeld_number(load, rpm, viscosity)
    Re = bd.reynolds_number(rpm, viscosity)
    print(f"  Sommerfeld: {S:.4f}")
    print(f"  Reynolds: {Re:.2f}")

    # Full feature computation
    print(f"\nFull Physics Features:")
    features = bd.compute_all_physics_features(rpm, load, viscosity)
    for name, value in features.items():
        if isinstance(value, (int, float, np.ndarray)):
            print(f"  {name}: {value}")

    # Test with torch tensors
    print(f"\nTorch Tensor Test:")
    rpm_torch = torch.tensor([3000.0, 3600.0, 4000.0])
    load_torch = torch.tensor([400.0, 500.0, 600.0])
    freqs_torch = bd.characteristic_frequencies(rpm_torch)
    print(f"  Input RPM: {rpm_torch}")
    print(f"  BPFO: {freqs_torch['BPFO']}")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
