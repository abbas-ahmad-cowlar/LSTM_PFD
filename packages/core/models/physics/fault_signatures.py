"""
Fault Signatures Database — journal / hydrodynamic bearings.

Expected spectral signatures per fault class, used by physics-informed losses
to check whether a predicted class is consistent with a signal's spectrum.

**Authoritative source: `docs/PHYSICS.md` §4** (the same physics the generator
implements, enforced by `tests/test_physics_signatures.py`). A separate CI test
(`tests/test_signature_db_consistency.py`) asserts this DB stays consistent with
the actual generated data, so the physics the *models* assume can never silently
diverge from the physics the *data* contains.

NOTE (2026-06-14 rebuild): this module previously encoded ROLLING-ELEMENT
signatures (BPFO/BPFI/BSF/FTF, outer_race/inner_race/ball) that are physically
wrong for journal bearings, and lacked the 3 mixed classes (KeyError). See
`audit_reports/PHYSICS_LOSS_AUDIT_2026-06-14.md`. Rebuilt from PHYSICS.md §4.
Signatures are expressed relative to shaft frequency Ω = rpm/60 (tonal) or as
absolute Hz bands (broadband/impulsive faults).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from utils.constants import FAULT_TYPES


class FaultSignature:
    """Spectral signature of one fault class (journal bearing).

    tonal: list of (multiplier_of_Omega, relative_halfwidth) — narrow peaks at
           multiplier*Omega (e.g. (2.0, 0.06) = 2X ±6%); sub-sync uses wider hw.
    bands_hz: list of (low_hz, high_hz) absolute bands (broadband/impulsive).
    broadband: True if the fault elevates the broadband floor (wear).
    """

    def __init__(self, fault_type: str,
                 tonal: List[Tuple[float, float]] = None,
                 bands_hz: List[Tuple[float, float]] = None,
                 broadband: bool = False):
        self.fault_type = fault_type
        self.tonal = tonal or []
        self.bands_hz = bands_hz or []
        self.broadband = broadband

    @property
    def primary_frequencies(self) -> List[str]:
        """Legacy label list (for any older reader)."""
        return [f"{m:g}X" for m, _ in self.tonal] + \
               [f"{lo:g}-{hi:g}Hz" for lo, hi in self.bands_hz] + \
               (["broadband"] if self.broadband else [])


class FaultSignatureDatabase:
    """Expected journal-bearing fault signatures, keyed by FAULT_TYPES."""

    def __init__(self):
        self._build()

    def _build(self):
        H = 0.06   # harmonic half-width (±6%)
        S = 0.12   # sub-synchronous half-width (the 0.40–0.50·Ω band)
        LOW = (1.0, 6.0)         # stick-slip / low-frequency band (PHYSICS.md §4.5)
        HF = (1400.0, 2600.0)    # cavitation HF burst band (§4.6)
        self.signatures: Dict[str, FaultSignature] = {
            'sain': FaultSignature('sain'),  # §4.1 — no fault tones
            'desalignement': FaultSignature(  # §4.2 — 2X dominant, 3X
                'desalignement', tonal=[(2.0, H), (3.0, H)]),
            'desequilibre': FaultSignature(  # §4.3 — pure 1X
                'desequilibre', tonal=[(1.0, H)]),
            'jeu': FaultSignature(  # §4.4 — sub-sync 0.43–0.48Ω + 1X + 2X
                'jeu', tonal=[(0.45, S), (1.0, H), (2.0, H)]),
            'lubrification': FaultSignature(  # §4.5 — stick-slip 2–5 Hz (1–6 band)
                'lubrification', bands_hz=[LOW]),
            'cavitation': FaultSignature(  # §4.6 — 1.5–2.5 kHz bursts
                'cavitation', bands_hz=[HF]),
            'usure': FaultSignature(  # §4.7 — broadband floor + 1X/2X asperity
                'usure', tonal=[(1.0, H), (2.0, H)], broadband=True),
            'oilwhirl': FaultSignature(  # §4.8 — sub-sync 0.42–0.48Ω
                'oilwhirl', tonal=[(0.45, S)]),
            'mixed_misalign_imbalance': FaultSignature(  # §4.9 — 1X+2X+3X
                'mixed_misalign_imbalance', tonal=[(1.0, H), (2.0, H), (3.0, H)]),
            'mixed_wear_lube': FaultSignature(  # §4.10 — broadband + asperity + stick-slip
                'mixed_wear_lube', tonal=[(1.0, H), (2.0, H)], bands_hz=[LOW], broadband=True),
            'mixed_cavit_jeu': FaultSignature(  # §4.11 — HF bursts + sub-sync + 1X
                'mixed_cavit_jeu', tonal=[(0.45, S), (1.0, H)], bands_hz=[HF]),
        }

    def _name(self, fault_type: Union[int, str]) -> str:
        return FAULT_TYPES[fault_type] if isinstance(fault_type, (int, np.integer)) else fault_type

    def get_signature(self, fault_type: Union[int, str]) -> FaultSignature:
        return self.signatures[self._name(fault_type)]

    def get_expected_frequencies(self, fault_type: Union[int, str], rpm: float,
                                 top_k: int = 5) -> np.ndarray:
        """Tonal expected frequencies (Hz) at this rpm. Empty for purely
        broadband faults (lubrification, cavitation) — they have no tones."""
        omega = rpm / 60.0
        sig = self.get_signature(fault_type)
        freqs = [m * omega for m, _ in sig.tonal]
        return np.array(freqs[:top_k], dtype=float)

    def get_expected_bands(self, fault_type: Union[int, str], rpm: float
                           ) -> Tuple[List[Tuple[float, float]], bool]:
        """Absolute (low_hz, high_hz) bands for this fault at this rpm:
        tonal peaks expanded to narrow bands + the absolute broadband bands.
        Returns (bands, is_broadband). Empty bands for 'sain'."""
        omega = rpm / 60.0
        sig = self.get_signature(fault_type)
        bands = [(m * omega * (1 - hw), m * omega * (1 + hw)) for m, hw in sig.tonal]
        bands += [(lo, hi) for lo, hi in sig.bands_hz]
        return bands, sig.broadband

    def compute_expected_spectrum(self, fault_type: Union[int, str], rpm: float,
                                  freq_bins: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
        """Idealized spectrum: Gaussian bumps over each expected band + a flat
        elevation for broadband faults. (Legacy helper for frequency-domain losses.)"""
        freq_bins = np.asarray(freq_bins, dtype=float)
        spec = np.zeros_like(freq_bins)
        bands, broadband = self.get_expected_bands(fault_type, rpm)
        for lo, hi in bands:
            center, width = 0.5 * (lo + hi), max((hi - lo) / 2.0, 1.0)
            spec += amplitude * np.exp(-0.5 * ((freq_bins - center) / width) ** 2)
        if broadband:
            spec += 0.1 * amplitude
        return spec


# --- frozen healthy-class reference (P6 Step 4) ---------------------------------
_HEALTHY_REFERENCE_PATH = Path(__file__).with_name('healthy_reference.json')
_healthy_reference_cache: Optional[dict] = None


def load_healthy_reference(path: Optional[str] = None) -> Optional[dict]:
    """Frozen healthy-class band-energy reference (owner-ratified 2026-06-14).

    Returns ``{class_name: {'tonal': [H_ref, ...], 'bands_hz': [H_ref, ...]}}`` —
    the mean fraction of total energy HEALTHY training windows carry in each
    class's expected bands — or ``None`` if the artifact is absent. The
    band-energy physics loss and the DB↔data CI test both judge "signature
    present" as energy ABOVE these values (not above a flat spectrum), so
    healthy-shared energy (EMI, low-frequency pink noise) cannot masquerade as a
    fault. Regenerate with ``scripts/compute_healthy_reference.py``. Cached.
    """
    global _healthy_reference_cache
    if path is None and _healthy_reference_cache is not None:
        return _healthy_reference_cache
    p = Path(path) if path else _HEALTHY_REFERENCE_PATH
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding='utf-8'))
    ref = data.get('per_class', data)
    if path is None:
        _healthy_reference_cache = ref
    return ref


# --- module-level convenience API (kept stable for package __init__ + callers) ---
default_database = FaultSignatureDatabase()


def get_fault_signature(fault_type: Union[int, str]) -> FaultSignature:
    return default_database.get_signature(fault_type)


def get_expected_frequencies(fault_type: Union[int, str], rpm: float, top_k: int = 5) -> np.ndarray:
    return default_database.get_expected_frequencies(fault_type, rpm, top_k)


def compute_expected_spectrum(fault_type: Union[int, str], rpm: float,
                              freq_bins: np.ndarray, amplitude: float = 1.0) -> np.ndarray:
    return default_database.compute_expected_spectrum(fault_type, rpm, freq_bins, amplitude)
