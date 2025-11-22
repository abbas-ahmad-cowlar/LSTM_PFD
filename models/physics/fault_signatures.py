"""
Fault Signatures Database

This module contains expected frequency signatures for each bearing fault type.
These signatures are used to compute physics-based loss functions that constrain
PINN models to make physically plausible predictions.

Each fault type has characteristic frequencies in the vibration spectrum that
can be predicted from bearing geometry and operating conditions.
"""

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
import numpy as np
import torch
from typing import Dict, List, Union, Tuple
from .bearing_dynamics import BearingDynamics


class FaultSignature:
    """
    Represents the expected frequency signature of a bearing fault.

    Attributes:
        fault_type: Name of the fault (e.g., 'misalignment', 'imbalance')
        primary_frequencies: List of frequency components (as functions of shaft freq)
        harmonics: Number of harmonics typically present
        frequency_bands: Typical frequency ranges (low/medium/high)
        amplitude_profile: Expected relative amplitudes
    """

    def __init__(
        self,
        fault_type: str,
        primary_frequencies: List[str],
        harmonics: int = 3,
        frequency_bands: List[str] = None,
        amplitude_profile: str = "exponential_decay"
    ):
        self.fault_type = fault_type
        self.primary_frequencies = primary_frequencies
        self.harmonics = harmonics
        self.frequency_bands = frequency_bands or ["low", "medium"]
        self.amplitude_profile = amplitude_profile


class FaultSignatureDatabase:
    """
    Database of expected fault signatures for physics-informed constraints.

    This class maps each of the 11 fault types to their characteristic
    frequency signatures based on bearing physics and vibration analysis theory.
    """

    # Fault type mappings (consistent with dataset labels)
    FAULT_TYPES = {
        0: 'healthy',
        1: 'misalignment',
        2: 'imbalance',
        3: 'outer_race',
        4: 'inner_race',
        5: 'ball',
        6: 'looseness',
        7: 'oil_whirl',
        8: 'cavitation',
        9: 'wear',
        10: 'lubrication'
    }

    def __init__(self):
        """Initialize fault signature database."""
        self.bearing_dynamics = BearingDynamics()
        self._build_signature_database()

    def _build_signature_database(self):
        """Build the fault signature database with physics-based expectations."""

        self.signatures = {
            'healthy': FaultSignature(
                fault_type='healthy',
                primary_frequencies=['broadband_noise'],  # Low amplitude, broadband
                harmonics=0,
                frequency_bands=['low', 'medium', 'high'],
                amplitude_profile='uniform_low'
            ),

            'misalignment': FaultSignature(
                fault_type='misalignment',
                primary_frequencies=['1X', '2X', '3X'],  # 1X, 2X, 3X shaft speed
                harmonics=3,
                frequency_bands=['low'],  # < 500 Hz typically
                amplitude_profile='harmonic_decay'  # 2X often strongest
            ),

            'imbalance': FaultSignature(
                fault_type='imbalance',
                primary_frequencies=['1X'],  # Strong 1X component
                harmonics=1,
                frequency_bands=['low'],  # At shaft frequency
                amplitude_profile='single_peak'  # Dominant 1X
            ),

            'outer_race': FaultSignature(
                fault_type='outer_race',
                primary_frequencies=['BPFO'],  # Ball Pass Frequency Outer race
                harmonics=5,  # Outer race defects show many harmonics
                frequency_bands=['medium'],  # 100-2000 Hz
                amplitude_profile='exponential_decay'
            ),

            'inner_race': FaultSignature(
                fault_type='inner_race',
                primary_frequencies=['BPFI'],  # Ball Pass Frequency Inner race
                harmonics=5,
                frequency_bands=['medium'],  # 100-2000 Hz
                amplitude_profile='modulated'  # Amplitude modulated by shaft speed
            ),

            'ball': FaultSignature(
                fault_type='ball',
                primary_frequencies=['BSF', '2*BSF'],  # Ball Spin Frequency
                harmonics=3,
                frequency_bands=['medium', 'high'],  # 500-3000 Hz
                amplitude_profile='exponential_decay'
            ),

            'looseness': FaultSignature(
                fault_type='looseness',
                primary_frequencies=['1X', '2X', '3X', 'subsync'],  # Multiple harmonics + subsynchronous
                harmonics=5,
                frequency_bands=['low', 'medium'],
                amplitude_profile='chaotic'  # Non-linear behavior
            ),

            'oil_whirl': FaultSignature(
                fault_type='oil_whirl',
                primary_frequencies=['0.42X', '0.43X', '0.48X'],  # Sub-synchronous (40-48% of shaft speed)
                harmonics=2,
                frequency_bands=['low'],  # Below shaft frequency
                amplitude_profile='subsynchronous'
            ),

            'cavitation': FaultSignature(
                fault_type='cavitation',
                primary_frequencies=['high_freq_bursts'],  # 1000-5000 Hz bursts
                harmonics=0,  # Non-periodic, burst-like
                frequency_bands=['high'],  # High frequency
                amplitude_profile='burst'  # Random bursts
            ),

            'wear': FaultSignature(
                fault_type='wear',
                primary_frequencies=['BPFO', 'BPFI', 'FTF'],  # Mix of bearing frequencies
                harmonics=3,
                frequency_bands=['medium', 'high'],
                amplitude_profile='broadband_elevated'  # Elevated across spectrum
            ),

            'lubrication': FaultSignature(
                fault_type='lubrication',
                primary_frequencies=['high_freq_random'],  # High frequency random noise
                harmonics=0,
                frequency_bands=['high'],  # > 2000 Hz
                amplitude_profile='random_high_freq'
            )
        }

    def get_signature(self, fault_type: Union[int, str]) -> FaultSignature:
        """
        Get fault signature for a given fault type.

        Args:
            fault_type: Either fault index (0-10) or fault name

        Returns:
            FaultSignature object
        """
        if isinstance(fault_type, int):
            fault_name = self.FAULT_TYPES[fault_type]
        else:
            fault_name = fault_type

        return self.signatures[fault_name]

    def get_expected_frequencies(
        self,
        fault_type: Union[int, str],
        rpm: float,
        top_k: int = 5
    ) -> np.ndarray:
        """
        Get expected dominant frequencies for a fault type at given RPM.

        Args:
            fault_type: Fault index or name
            rpm: Shaft speed in RPM
            top_k: Number of top frequencies to return

        Returns:
            Array of expected frequencies in Hz
        """
        signature = self.get_signature(fault_type)

        # Calculate characteristic frequencies
        freqs_dict = self.bearing_dynamics.characteristic_frequencies(rpm)
        shaft_freq = freqs_dict['shaft_freq']

        expected_freqs = []

        for freq_spec in signature.primary_frequencies:
            if freq_spec == '1X':
                expected_freqs.append(shaft_freq)
            elif freq_spec == '2X':
                expected_freqs.append(2 * shaft_freq)
            elif freq_spec == '3X':
                expected_freqs.append(3 * shaft_freq)
            elif freq_spec == 'BPFO':
                expected_freqs.extend([
                    freqs_dict['BPFO'] * (i + 1)
                    for i in range(signature.harmonics)
                ])
            elif freq_spec == 'BPFI':
                expected_freqs.extend([
                    freqs_dict['BPFI'] * (i + 1)
                    for i in range(signature.harmonics)
                ])
            elif freq_spec == 'BSF':
                expected_freqs.extend([
                    freqs_dict['BSF'] * (i + 1)
                    for i in range(signature.harmonics)
                ])
            elif freq_spec == '2*BSF':
                expected_freqs.append(2 * freqs_dict['BSF'])
            elif freq_spec == 'FTF':
                expected_freqs.append(freqs_dict['FTF'])
            elif freq_spec == '0.42X':
                expected_freqs.append(0.42 * shaft_freq)
            elif freq_spec == '0.43X':
                expected_freqs.append(0.43 * shaft_freq)
            elif freq_spec == '0.48X':
                expected_freqs.append(0.48 * shaft_freq)
            elif freq_spec == 'subsync':
                # Add sub-synchronous components
                expected_freqs.extend([0.3 * shaft_freq, 0.5 * shaft_freq])
            elif freq_spec in ['broadband_noise', 'high_freq_bursts',
                              'high_freq_random', 'broadband_elevated']:
                # For broadband/random, return typical frequency range centers
                if 'high' in freq_spec:
                    expected_freqs.extend([2000, 3000, 4000])
                else:
                    expected_freqs.extend([500, 1000, 1500])

        # Convert to numpy array and return top_k
        expected_freqs = np.array(expected_freqs)
        if len(expected_freqs) > top_k:
            expected_freqs = expected_freqs[:top_k]

        return expected_freqs

    def compute_expected_spectrum(
        self,
        fault_type: Union[int, str],
        rpm: float,
        freq_bins: np.ndarray,
        amplitude: float = 1.0
    ) -> np.ndarray:
        """
        Generate expected frequency spectrum for a fault type.

        This creates an idealized spectrum with peaks at expected frequencies,
        useful for computing frequency-domain loss functions.

        Args:
            fault_type: Fault index or name
            rpm: Shaft speed in RPM
            freq_bins: Frequency bins (Hz) for spectrum
            amplitude: Base amplitude for peaks

        Returns:
            Expected spectrum amplitude at each frequency bin
        """
        signature = self.get_signature(fault_type)
        expected_freqs = self.get_expected_frequencies(fault_type, rpm, top_k=10)

        # Initialize spectrum
        spectrum = np.zeros_like(freq_bins)

        # Add Gaussian peaks at expected frequencies
        for i, freq in enumerate(expected_freqs):
            # Amplitude decay for harmonics
            if signature.amplitude_profile == 'exponential_decay':
                amp = amplitude * np.exp(-0.3 * i)
            elif signature.amplitude_profile == 'harmonic_decay':
                amp = amplitude / (i + 1)
            elif signature.amplitude_profile == 'single_peak':
                amp = amplitude if i == 0 else amplitude * 0.1
            else:
                amp = amplitude

            # Add Gaussian peak (width proportional to frequency)
            sigma = freq * 0.05  # 5% bandwidth
            spectrum += amp * np.exp(-((freq_bins - freq) ** 2) / (2 * sigma ** 2))

        # Add noise floor for broadband components
        if 'broadband' in signature.amplitude_profile or signature.amplitude_profile == 'uniform_low':
            noise_level = 0.1 * amplitude if signature.fault_type == 'healthy' else 0.3 * amplitude
            spectrum += noise_level * np.random.randn(len(freq_bins)) * 0.1

        return spectrum

    def get_frequency_band_energy(
        self,
        spectrum: np.ndarray,
        freq_bins: np.ndarray,
        band: str = 'low'
    ) -> float:
        """
        Calculate energy in a specific frequency band.

        Args:
            spectrum: Frequency spectrum
            freq_bins: Frequency bins in Hz
            band: 'low' (<500 Hz), 'medium' (500-2000 Hz), or 'high' (>2000 Hz)

        Returns:
            Total energy in the band
        """
        if band == 'low':
            mask = freq_bins < 500
        elif band == 'medium':
            mask = (freq_bins >= 500) & (freq_bins < 2000)
        elif band == 'high':
            mask = freq_bins >= 2000
        else:
            raise ValueError(f"Unknown band: {band}")

        return np.sum(spectrum[mask] ** 2)

    def check_signature_consistency(
        self,
        predicted_class: int,
        spectrum: np.ndarray,
        freq_bins: np.ndarray,
        rpm: float,
        tolerance: float = 0.1
    ) -> float:
        """
        Check if observed spectrum is consistent with expected fault signature.

        This computes a consistency score between 0 (inconsistent) and 1 (perfect match).
        Used as a physics-based validation metric.

        Args:
            predicted_class: Predicted fault type (0-10)
            spectrum: Observed frequency spectrum
            freq_bins: Frequency bins in Hz
            rpm: Shaft speed in RPM
            tolerance: Frequency tolerance as fraction of expected frequency

        Returns:
            Consistency score (0-1)
        """
        # Get expected frequencies
        expected_freqs = self.get_expected_frequencies(predicted_class, rpm, top_k=5)

        # Find actual peaks in spectrum
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(spectrum, height=0.1 * np.max(spectrum))
        peak_freqs = freq_bins[peaks]

        if len(peak_freqs) == 0:
            return 0.0

        # Check how many expected frequencies are present
        matches = 0
        for expected_freq in expected_freqs:
            # Check if any peak is within tolerance
            freq_tolerance = expected_freq * tolerance
            if np.any(np.abs(peak_freqs - expected_freq) < freq_tolerance):
                matches += 1

        # Consistency = fraction of expected frequencies found
        consistency = matches / len(expected_freqs)

        return consistency


# Create default database instance
default_database = FaultSignatureDatabase()


def get_fault_signature(fault_type: Union[int, str]) -> FaultSignature:
    """Convenience function to get fault signature."""
    return default_database.get_signature(fault_type)


def get_expected_frequencies(fault_type: Union[int, str], rpm: float, top_k: int = 5) -> np.ndarray:
    """Convenience function to get expected frequencies."""
    return default_database.get_expected_frequencies(fault_type, rpm, top_k)


def compute_expected_spectrum(
    fault_type: Union[int, str],
    rpm: float,
    freq_bins: np.ndarray,
    amplitude: float = 1.0
) -> np.ndarray:
    """Convenience function to compute expected spectrum."""
    return default_database.compute_expected_spectrum(fault_type, rpm, freq_bins, amplitude)


if __name__ == "__main__":
    # Example usage and validation
    print("=" * 60)
    print("Fault Signature Database - Validation")
    print("=" * 60)

    db = FaultSignatureDatabase()

    # Test with typical RPM
    rpm = 3600.0

    print(f"\nFault Type Signatures (RPM = {rpm}):\n")

    for fault_id, fault_name in db.FAULT_TYPES.items():
        signature = db.get_signature(fault_id)
        expected_freqs = db.get_expected_frequencies(fault_id, rpm, top_k=3)

        print(f"{fault_id}. {fault_name.upper()}")
        print(f"   Primary: {signature.primary_frequencies}")
        print(f"   Harmonics: {signature.harmonics}")
        print(f"   Bands: {signature.frequency_bands}")
        print(f"   Expected freqs (Hz): {expected_freqs}")
        print()

    # Test expected spectrum generation
    print("\nGenerating Expected Spectrum for Outer Race Fault:")
    freq_bins = np.linspace(0, 5000, 1000)
    spectrum = db.compute_expected_spectrum('outer_race', rpm, freq_bins, amplitude=1.0)
    print(f"   Spectrum shape: {spectrum.shape}")
    print(f"   Max amplitude: {np.max(spectrum):.3f}")
    print(f"   Peak frequency: {freq_bins[np.argmax(spectrum)]:.2f} Hz")

    # Test consistency checking
    print("\nTesting Signature Consistency:")
    # Create synthetic spectrum with outer race fault
    true_spectrum = db.compute_expected_spectrum('outer_race', rpm, freq_bins)

    # Check consistency with correct prediction
    consistency_correct = db.check_signature_consistency(3, true_spectrum, freq_bins, rpm)
    print(f"   Outer race spectrum vs. Outer race prediction: {consistency_correct:.2f}")

    # Check consistency with wrong prediction
    consistency_wrong = db.check_signature_consistency(2, true_spectrum, freq_bins, rpm)
    print(f"   Outer race spectrum vs. Imbalance prediction: {consistency_wrong:.2f}")

    print("\n" + "=" * 60)
    print("Validation Complete!")
    print("=" * 60)
