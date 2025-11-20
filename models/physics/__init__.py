"""
Physics Models Package

This package contains physics-based models for bearing fault diagnosis:
- bearing_dynamics: Characteristic frequencies and dimensionless numbers
- fault_signatures: Expected frequency signatures for each fault type
- operating_conditions: Operating condition validation (to be implemented)
"""

from .bearing_dynamics import (
    BearingDynamics,
    characteristic_frequencies,
    sommerfeld_number,
    reynolds_number,
    default_bearing
)

from .fault_signatures import (
    FaultSignature,
    FaultSignatureDatabase,
    get_fault_signature,
    get_expected_frequencies,
    compute_expected_spectrum,
    default_database
)

__all__ = [
    'BearingDynamics',
    'characteristic_frequencies',
    'sommerfeld_number',
    'reynolds_number',
    'default_bearing',
    'FaultSignature',
    'FaultSignatureDatabase',
    'get_fault_signature',
    'get_expected_frequencies',
    'compute_expected_spectrum',
    'default_database'
]
