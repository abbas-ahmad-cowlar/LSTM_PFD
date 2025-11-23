"""
Operating Conditions Validation

This module provides utilities for validating and analyzing bearing operating conditions.
It ensures that operating points (load, speed, temperature) are physically valid and
calculates derived parameters useful for fault diagnosis.

Key Functions:
- Validate operating condition combinations
- Calculate film thickness (lubrication theory)
- Determine flow regime (laminar/turbulent)
- Predict bearing temperature rise
- Check for critical operating zones
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import numpy as np
import torch
from typing import Dict, Union, Tuple, Optional
from .bearing_dynamics import BearingDynamics


class OperatingConditionsValidator:
    """
    Validates and analyzes bearing operating conditions.

    This class checks if operating conditions are physically plausible and
    identifies potentially dangerous operating zones.
    """

    def __init__(self, bearing_params: Optional[Dict[str, float]] = None):
        """
        Initialize operating conditions validator.

        Args:
            bearing_params: Bearing geometry parameters (uses default if None)
        """
        self.bearing_dynamics = BearingDynamics(bearing_params)
        self.bearing_params = bearing_params or self.bearing_dynamics.params

        # Typical operating ranges for industrial bearings
        self.safe_ranges = {
            'rpm': (500, 10000),  # RPM
            'load': (50, 2000),  # Newtons
            'temperature': (20, 120),  # Celsius
            'viscosity': (0.001, 0.1),  # Pa·s
        }

        # Critical thresholds
        self.critical_thresholds = {
            'min_sommerfeld': 0.05,  # Below this: severe boundary lubrication
            'max_reynolds': 5000,  # Above this: turbulent flow
            'max_temperature': 100,  # °C - oil degradation risk
            'min_film_thickness': 0.1,  # μm - metal contact risk
        }

    def validate_operating_point(
        self,
        rpm: Union[float, np.ndarray, torch.Tensor],
        load: Union[float, np.ndarray, torch.Tensor],
        temperature: Union[float, np.ndarray, torch.Tensor],
        viscosity: Optional[Union[float, np.ndarray, torch.Tensor]] = None
    ) -> Dict[str, Union[bool, str]]:
        """
        Validate if an operating point is physically plausible.

        Args:
            rpm: Shaft speed in RPM
            load: Applied load in Newtons
            temperature: Temperature in Celsius
            viscosity: Lubricant viscosity in Pa·s (calculated from temp if None)

        Returns:
            Dictionary with:
                - 'valid': Boolean indicating if point is valid
                - 'warnings': List of warning messages
                - 'errors': List of error messages
        """
        warnings = []
        errors = []

        # Convert to numpy for easier checking
        if isinstance(rpm, torch.Tensor):
            rpm = rpm.cpu().numpy()
        if isinstance(load, torch.Tensor):
            load = load.cpu().numpy()
        if isinstance(temperature, torch.Tensor):
            temperature = temperature.cpu().numpy()

        # Ensure scalar or handle arrays
        rpm = np.atleast_1d(rpm)
        load = np.atleast_1d(load)
        temperature = np.atleast_1d(temperature)

        # Calculate viscosity from temperature if not provided
        if viscosity is None:
            viscosity = self.calculate_viscosity_from_temperature(temperature)
        else:
            if isinstance(viscosity, torch.Tensor):
                viscosity = viscosity.cpu().numpy()
            viscosity = np.atleast_1d(viscosity)

        # Check range validity
        if np.any(rpm < self.safe_ranges['rpm'][0]) or np.any(rpm > self.safe_ranges['rpm'][1]):
            warnings.append(f"RPM outside typical range {self.safe_ranges['rpm']}")

        if np.any(load < self.safe_ranges['load'][0]) or np.any(load > self.safe_ranges['load'][1]):
            warnings.append(f"Load outside typical range {self.safe_ranges['load']}")

        if np.any(temperature < self.safe_ranges['temperature'][0]) or \
           np.any(temperature > self.safe_ranges['temperature'][1]):
            warnings.append(f"Temperature outside typical range {self.safe_ranges['temperature']}")

        # Check for physically impossible conditions
        if np.any(rpm <= 0):
            errors.append("RPM must be positive")
        if np.any(load <= 0):
            errors.append("Load must be positive")
        if np.any(temperature < -273.15):  # Absolute zero
            errors.append("Temperature below absolute zero")

        # Calculate Sommerfeld number
        S = self.bearing_dynamics.sommerfeld_number(load, rpm, viscosity)
        if isinstance(S, torch.Tensor):
            S = S.cpu().numpy()
        S = np.atleast_1d(S)

        if np.any(S < self.critical_thresholds['min_sommerfeld']):
            warnings.append(
                f"Sommerfeld number < {self.critical_thresholds['min_sommerfeld']}: "
                "Boundary lubrication - high wear risk"
            )

        # Calculate Reynolds number
        Re = self.bearing_dynamics.reynolds_number(rpm, viscosity)
        if isinstance(Re, torch.Tensor):
            Re = Re.cpu().numpy()
        Re = np.atleast_1d(Re)

        if np.any(Re > self.critical_thresholds['max_reynolds']):
            warnings.append(
                f"Reynolds number > {self.critical_thresholds['max_reynolds']}: "
                "Turbulent flow - unusual for journal bearings"
            )

        # Check temperature
        if np.any(temperature > self.critical_thresholds['max_temperature']):
            warnings.append(
                f"Temperature > {self.critical_thresholds['max_temperature']}°C: "
                "Oil degradation risk"
            )

        # Check film thickness
        film_thickness = self.predict_film_thickness(load, rpm, viscosity)
        if np.any(film_thickness < self.critical_thresholds['min_film_thickness']):
            warnings.append(
                f"Film thickness < {self.critical_thresholds['min_film_thickness']} μm: "
                "Metal-to-metal contact risk"
            )

        valid = len(errors) == 0

        return {
            'valid': valid,
            'warnings': warnings,
            'errors': errors,
            'sommerfeld': float(np.mean(S)),
            'reynolds': float(np.mean(Re)),
            'film_thickness_um': float(np.mean(film_thickness))
        }

    def calculate_viscosity_from_temperature(
        self,
        temperature: Union[float, np.ndarray, torch.Tensor],
        reference_viscosity: float = 0.03,  # Pa·s at 40°C
        reference_temp: float = 40.0  # °C
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Calculate lubricant viscosity from temperature using Vogel equation.

        Viscosity decreases exponentially with temperature.

        Args:
            temperature: Temperature in Celsius
            reference_viscosity: Viscosity at reference temperature
            reference_temp: Reference temperature in Celsius

        Returns:
            Viscosity in Pa·s
        """
        # Vogel equation parameters (typical for mineral oil)
        A = 3.0
        B = 1000.0
        C = 140.0

        is_torch = isinstance(temperature, torch.Tensor)

        if is_torch:
            # Convert to absolute temperature
            T = temperature + 273.15
            T_ref = reference_temp + 273.15

            # Calculate viscosity ratio
            log_ratio = A * (1 / (T + C) - 1 / (T_ref + C))
            viscosity = reference_viscosity * torch.exp(B * log_ratio)
        else:
            if isinstance(temperature, torch.Tensor):
                temperature = temperature.cpu().numpy()

            T = temperature + 273.15
            T_ref = reference_temp + 273.15

            log_ratio = A * (1 / (T + C) - 1 / (T_ref + C))
            viscosity = reference_viscosity * np.exp(B * log_ratio)

        return viscosity

    def predict_film_thickness(
        self,
        load: Union[float, np.ndarray, torch.Tensor],
        speed: Union[float, np.ndarray, torch.Tensor],
        viscosity: Union[float, np.ndarray, torch.Tensor]
    ) -> Union[float, np.ndarray, torch.Tensor]:
        """
        Predict minimum oil film thickness using Reynolds equation.

        Args:
            load: Applied load in Newtons
            speed: Shaft speed in RPM
            viscosity: Dynamic viscosity in Pa·s

        Returns:
            Minimum film thickness in micrometers (μm)
        """
        R = self.bearing_params['shaft_radius'] / 1000.0  # mm to m
        C = self.bearing_params['radial_clearance'] / 1e6  # μm to m
        L = 0.015  # Bearing length in m (assumed)

        is_torch = isinstance(load, torch.Tensor) or isinstance(speed, torch.Tensor)

        if is_torch:
            if not isinstance(load, torch.Tensor):
                load = torch.tensor(load, dtype=torch.float32)
            if not isinstance(speed, torch.Tensor):
                speed = torch.tensor(speed, dtype=torch.float32)
            if not isinstance(viscosity, torch.Tensor):
                viscosity = torch.tensor(viscosity, dtype=torch.float32)

            N = speed / 60.0  # RPM to rev/s
            P = load / (2 * R * L)  # Bearing pressure

            # Sommerfeld number (with epsilon for numerical stability)
            S = (viscosity * N / (P + 1e-8)) * (R / C) ** 2

            # Eccentricity ratio (from Sommerfeld number)
            # Approximate relation: ε ≈ 1 - 1/sqrt(1 + S)
            epsilon = 1.0 - 1.0 / torch.sqrt(1.0 + S + 1e-8)

            # Minimum film thickness
            h_min = C * (1.0 - epsilon) * 1e6  # Convert to μm

        else:
            if isinstance(load, torch.Tensor):
                load = load.cpu().numpy()
            if isinstance(speed, torch.Tensor):
                speed = speed.cpu().numpy()
            if isinstance(viscosity, torch.Tensor):
                viscosity = viscosity.cpu().numpy()

            N = speed / 60.0
            P = load / (2 * R * L)
            P = np.maximum(P, 1e-8)

            S = (viscosity * N / (P + 1e-8)) * (R / C) ** 2
            epsilon = 1.0 - 1.0 / np.sqrt(1.0 + S + 1e-8)
            h_min = C * (1.0 - epsilon) * 1e6

        return h_min

    def check_laminar_turbulent(
        self,
        reynolds_number: Union[float, np.ndarray, torch.Tensor]
    ) -> Union[int, np.ndarray]:
        """
        Classify flow regime based on Reynolds number.

        Args:
            reynolds_number: Reynolds number

        Returns:
            Flow regime: 0 (laminar), 1 (transition), 2 (turbulent)
        """
        is_torch = isinstance(reynolds_number, torch.Tensor)

        if is_torch:
            flow_regime = torch.zeros_like(reynolds_number, dtype=torch.long)
            flow_regime[reynolds_number < 2000] = 0
            flow_regime[(reynolds_number >= 2000) & (reynolds_number < 4000)] = 1
            flow_regime[reynolds_number >= 4000] = 2
        else:
            if isinstance(reynolds_number, torch.Tensor):
                reynolds_number = reynolds_number.cpu().numpy()

            flow_regime = np.zeros_like(reynolds_number, dtype=int)
            flow_regime[reynolds_number < 2000] = 0
            flow_regime[(reynolds_number >= 2000) & (reynolds_number < 4000)] = 1
            flow_regime[reynolds_number >= 4000] = 2

        return flow_regime

    def predict_temperature_rise(
        self,
        rpm: Union[float, np.ndarray],
        load: Union[float, np.ndarray],
        viscosity: Union[float, np.ndarray],
        ambient_temp: float = 25.0
    ) -> Union[float, np.ndarray]:
        """
        Predict bearing temperature rise from friction heating.

        Args:
            rpm: Shaft speed in RPM
            load: Applied load in Newtons
            viscosity: Lubricant viscosity in Pa·s
            ambient_temp: Ambient temperature in Celsius

        Returns:
            Predicted bearing temperature in Celsius
        """
        # Convert to numpy if needed
        if isinstance(rpm, torch.Tensor):
            rpm = rpm.cpu().numpy()
        if isinstance(load, torch.Tensor):
            load = load.cpu().numpy()
        if isinstance(viscosity, torch.Tensor):
            viscosity = viscosity.cpu().numpy()

        R = self.bearing_params['shaft_radius'] / 1000.0  # m
        C = self.bearing_params['radial_clearance'] / 1e6  # m
        L = 0.015  # m

        # Surface velocity
        omega = rpm * 2 * np.pi / 60.0  # rad/s
        U = omega * R

        # Friction coefficient (approximation)
        Re = self.bearing_dynamics.reynolds_number(rpm, viscosity)
        if isinstance(Re, torch.Tensor):
            Re = Re.cpu().numpy()

        # Petroff equation for friction (with epsilon for numerical stability)
        f = 2 * np.pi * viscosity * U / (load / L + 1e-8) * (R / C)

        # Power dissipation (Watts)
        power = f * load * U

        # Temperature rise (simplified, assumes constant heat transfer coefficient)
        h_conv = 20.0  # W/(m²·K) - convective heat transfer coefficient
        A_surface = 2 * np.pi * R * L  # Surface area

        # Steady-state temperature rise
        delta_T = power / (h_conv * A_surface)

        bearing_temp = ambient_temp + delta_T

        return bearing_temp

    def get_operating_regime_name(self, sommerfeld: float) -> str:
        """
        Get human-readable name for lubrication regime.

        Args:
            sommerfeld: Sommerfeld number

        Returns:
            Regime name string
        """
        if sommerfeld < 0.1:
            return "Boundary Lubrication (High Wear)"
        elif sommerfeld < 1.0:
            return "Mixed Lubrication (Moderate Wear)"
        else:
            return "Hydrodynamic Lubrication (Low Wear)"


# Create default validator instance
default_validator = OperatingConditionsValidator()


def validate_operating_point(rpm, load, temperature, viscosity=None):
    """Convenience function for operating point validation."""
    return default_validator.validate_operating_point(rpm, load, temperature, viscosity)


def calculate_viscosity_from_temperature(temperature, reference_viscosity=0.03, reference_temp=40.0):
    """Convenience function for viscosity calculation."""
    return default_validator.calculate_viscosity_from_temperature(
        temperature, reference_viscosity, reference_temp
    )


def predict_film_thickness(load, speed, viscosity):
    """Convenience function for film thickness prediction."""
    return default_validator.predict_film_thickness(load, speed, viscosity)


if __name__ == "__main__":
    # Test operating conditions validator
    print("=" * 60)
    print("Operating Conditions Validator - Validation")
    print("=" * 60)

    validator = OperatingConditionsValidator()

    # Test 1: Normal operating point
    print("\nTest 1: Normal Operating Point")
    rpm = 3600.0
    load = 500.0
    temperature = 60.0

    result = validator.validate_operating_point(rpm, load, temperature)
    print(f"  RPM: {rpm}, Load: {load}N, Temp: {temperature}°C")
    print(f"  Valid: {result['valid']}")
    print(f"  Sommerfeld: {result['sommerfeld']:.4f}")
    print(f"  Reynolds: {result['reynolds']:.2f}")
    print(f"  Film thickness: {result['film_thickness_um']:.3f} μm")
    if result['warnings']:
        print(f"  Warnings: {result['warnings']}")

    # Test 2: High load (boundary lubrication)
    print("\nTest 2: High Load (Boundary Lubrication)")
    rpm = 2000.0
    load = 1500.0
    temperature = 80.0

    result = validator.validate_operating_point(rpm, load, temperature)
    print(f"  RPM: {rpm}, Load: {load}N, Temp: {temperature}°C")
    print(f"  Valid: {result['valid']}")
    print(f"  Sommerfeld: {result['sommerfeld']:.4f} - {validator.get_operating_regime_name(result['sommerfeld'])}")
    if result['warnings']:
        for warning in result['warnings']:
            print(f"  ⚠ {warning}")

    # Test 3: Viscosity calculation
    print("\nTest 3: Viscosity vs Temperature")
    temps = np.array([20, 40, 60, 80, 100])
    viscosities = validator.calculate_viscosity_from_temperature(temps)
    print("  Temperature (°C) | Viscosity (Pa·s)")
    print("  " + "-" * 35)
    for t, v in zip(temps, viscosities):
        print(f"  {t:15.1f} | {v:.6f}")

    # Test 4: Film thickness
    print("\nTest 4: Film Thickness Prediction")
    rpms = np.array([1000, 2000, 3600, 5000])
    loads = np.array([300, 500, 700, 900])
    visc = 0.03

    print("  RPM  | Load (N) | Film Thickness (μm)")
    print("  " + "-" * 40)
    for r, l in zip(rpms, loads):
        h = validator.predict_film_thickness(l, r, visc)
        print(f"  {r:4.0f} | {l:8.0f} | {h:.3f}")

    # Test 5: Temperature rise
    print("\nTest 5: Temperature Rise Prediction")
    rpm = 3600.0
    load = 500.0
    visc = 0.03
    ambient = 25.0

    temp_predicted = validator.predict_temperature_rise(rpm, load, visc, ambient)
    print(f"  Operating: {rpm} RPM, {load}N")
    print(f"  Ambient: {ambient}°C")
    print(f"  Predicted bearing temp: {temp_predicted:.1f}°C")
    print(f"  Temperature rise: {temp_predicted - ambient:.1f}°C")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
