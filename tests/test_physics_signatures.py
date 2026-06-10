"""
Spectral-signature validation battery for the synthetic fault generator.

Convergence Plan P3.2 — permanent CI guards so the generator can never
silently drift from its documented physics. The normative reference is
``docs/PHYSICS.md``; every test docstring cites the section it enforces
(e.g. "PHYSICS.md §4.2").

Design:
    - Unit-level determinism: ``FaultModeler.generate_fault_signal`` is
      driven DIRECTLY with controlled inputs (fixed Omega = 60 Hz, flat
      severity curve, unit transient modulation, fixed operating factors).
      ``np.random.seed`` is called immediately before every modeler call
      because the modeler draws phases/positions/frequencies from the
      global numpy RNG. Same-seed pairs are used for monotonicity checks.
    - Frequency resolution: T = 2.0 s (N = 40,960 @ fs = 20,480 Hz) gives
      0.5 Hz bins — enough to separate 1X = 60, 2X = 120, 3X = 180 Hz and
      the sub-synchronous 24-30 Hz region. The low-frequency lubrication
      tests use T = 5.0 s (0.2 Hz bins) to resolve the 2-5 Hz stick-slip.
    - Spectra come from a full-length Hann periodogram
      (``scipy.signal.welch`` with ``nperseg = len(x)``) — deterministic
      and maximally sharp for tonal assertions.
    - Robustness: dominant-band assertions use a >= 3x energy margin, and
      where the modeler randomizes a frequency (whirl ratio 0.42-0.48,
      stick-slip 2-5 Hz, burst carrier 1.5-2.5 kHz) the detected peak is
      only required to lie in the documented band, never at an exact value.

CODE-vs-DOC DISCREPANCY LOG:
    1. [RESOLVED 2026-06-11] An early PHYSICS.md draft claimed lubrification
       "impact spikes -> kurtosis > 3". Numerical probing showed the 2-5 Hz
       stick-slip sine dominates the fourth moment (Pearson kurtosis of a
       pure sine = 1.5); the 1-4 friction impacts (~20 samples each of
       102,400) are fourth-moment-negligible — measured ~1.50 for every
       seed/severity/Sommerfeld combination. PHYSICS.md §4.5 was corrected
       to state the sine-like statistics, and
       ``TestLubrification.test_kurtosis_sine_like`` now pins the true
       behavior as a regression guard.
"""

import numpy as np
import pytest
from scipy import signal as sp_signal
from scipy.stats import kurtosis as _scipy_kurtosis

from config.data_config import DataConfig
from data.signal_generation import FaultModeler, SignalGenerator
from utils.constants import SAMPLING_RATE

# ---------------------------------------------------------------------------
# Controlled modeler inputs (see module docstring / task design)
# ---------------------------------------------------------------------------

FS = SAMPLING_RATE          # 20,480 Hz
OMEGA_HZ = 60.0             # Fixed shaft speed (PHYSICS.md §1: Omega_0)
SEVERITY = 0.8              # Flat severity curve value for signature tests
LOAD_FACTOR = 0.65
TEMP_FACTOR = 1.0
OPERATING_FACTOR = 0.65
PHYSICS_FACTOR = 1.0
SOMMERFELD = 0.15           # PHYSICS.md §3: S_base
SPEED_VARIATION = 1.0

MARGIN = 3.0                # Dominant band must carry >= 3x comparison band
PEAK_TOL_HZ = 1.0           # Tonal peak location tolerance

# Quiet reference band for tonal faults: no fault model places energy here
# (everything tonal lives below ~200 Hz; bursts live above 1.4 kHz).
FAR_BAND = (300.0, 400.0)


def _make_config(duration_s: float) -> DataConfig:
    config = DataConfig()
    config.signal.T = duration_s
    return config


def generate(
    modeler: FaultModeler,
    fault: str,
    seed: int,
    severity: float = SEVERITY,
    sommerfeld: float = SOMMERFELD,
    speed_variation: float = SPEED_VARIATION,
) -> np.ndarray:
    """Call the modeler directly with controlled, seeded inputs."""
    np.random.seed(seed)
    n = modeler.N
    return modeler.generate_fault_signal(
        fault,
        np.full(n, severity),
        np.ones(n),
        omega=2.0 * np.pi * OMEGA_HZ,
        Omega=OMEGA_HZ,
        load_factor=LOAD_FACTOR,
        temp_factor=TEMP_FACTOR,
        operating_factor=OPERATING_FACTOR,
        physics_factor=PHYSICS_FACTOR,
        sommerfeld=sommerfeld,
        speed_variation=speed_variation,
    )


# ---------------------------------------------------------------------------
# Spectral tooling
# ---------------------------------------------------------------------------


def psd(x: np.ndarray, fs: float = FS, nperseg: int = None):
    """Welch PSD; defaults to a full-length Hann periodogram (sharpest)."""
    if nperseg is None:
        nperseg = len(x)
    return sp_signal.welch(x, fs=fs, nperseg=min(nperseg, len(x)))


def band_energy(f: np.ndarray, pxx: np.ndarray, lo: float, hi: float) -> float:
    """Integrated PSD energy in [lo, hi] Hz."""
    mask = (f >= lo) & (f <= hi)
    return float(np.sum(pxx[mask]) * (f[1] - f[0]))


def peak_freq_in_band(f: np.ndarray, pxx: np.ndarray, lo: float, hi: float) -> float:
    """Frequency of the largest PSD bin inside [lo, hi] Hz."""
    mask = (f >= lo) & (f <= hi)
    return float(f[mask][np.argmax(pxx[mask])])


def dominant_freq(f: np.ndarray, pxx: np.ndarray, fmin: float = 1.0) -> float:
    """Globally dominant frequency (excluding the DC/drift region)."""
    return peak_freq_in_band(f, pxx, fmin, f[-1])


def spectral_flatness(pxx: np.ndarray) -> float:
    """Geometric / arithmetic PSD mean: ~0 for tonal, ->1 for white noise."""
    p = np.clip(pxx, 1e-30, None)
    return float(np.exp(np.mean(np.log(p))) / np.mean(p))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2)))


def pearson_kurtosis(x: np.ndarray) -> float:
    """Pearson (non-excess) kurtosis: 3.0 for Gaussian, 1.5 for a sine."""
    return float(_scipy_kurtosis(x, fisher=False))


# ---------------------------------------------------------------------------
# Fixtures (module-scoped: FaultModeler is stateless apart from config)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def modeler_2s():
    """T = 2.0 s -> N = 40,960; 0.5 Hz resolution for tonal tests."""
    return FaultModeler(_make_config(2.0))


@pytest.fixture(scope="module")
def modeler_5s():
    """T = 5.0 s -> N = 102,400; 0.2 Hz resolution for 2-5 Hz stick-slip."""
    return FaultModeler(_make_config(5.0))


# ---------------------------------------------------------------------------
# §4.1 sain (healthy)
# ---------------------------------------------------------------------------


class TestSain:
    def test_modeler_returns_exact_zeros(self, modeler_2s):
        """PHYSICS.md §4.1: s_fault = 0 — the modeler contributes nothing;
        all healthy-signal energy comes from the orchestrator noise floor."""
        x = generate(modeler_2s, "sain", seed=0)
        assert x.shape == (modeler_2s.N,)
        assert np.allclose(x, 0.0)

    def test_generator_sain_has_lowest_mean_rms(self):
        """PHYSICS.md §4.1: healthy = noise floor only, hence the lowest RMS
        of all classes. Checked end-to-end (generator level) against the two
        strongest tonal classes, mean over 8 seeded signals per class."""
        config = DataConfig(num_signals_per_fault=8, rng_seed=42)
        config.signal.T = 0.5  # RMS comparison needs no frequency resolution
        config.augmentation.enabled = False
        config.fault.include_healthy = True
        for name in config.fault.single_faults:
            config.fault.single_faults[name] = name in ("desequilibre", "oilwhirl")
        for name in config.fault.mixed_faults:
            config.fault.mixed_faults[name] = False

        dataset = SignalGenerator(config).generate_dataset()
        rms_by_class = {}
        for sig, label in zip(dataset["signals"], dataset["labels"]):
            rms_by_class.setdefault(label, []).append(rms(sig))
        mean_rms = {label: np.mean(v) for label, v in rms_by_class.items()}

        assert set(mean_rms) == {"sain", "desequilibre", "oilwhirl"}
        assert mean_rms["sain"] < mean_rms["desequilibre"]
        assert mean_rms["sain"] < mean_rms["oilwhirl"]


# ---------------------------------------------------------------------------
# §4.2 desalignement (misalignment)
# ---------------------------------------------------------------------------


class TestDesalignement:
    def test_peaks_at_2x_and_3x(self, modeler_2s):
        """PHYSICS.md §4.2: energy at 2x and 3x shaft speed — PSD peak within
        +/-1 Hz of both 2*Omega (120 Hz) and 3*Omega (180 Hz)."""
        x = generate(modeler_2s, "desalignement", seed=0)
        f, pxx = psd(x)
        assert abs(peak_freq_in_band(f, pxx, 110, 130) - 2 * OMEGA_HZ) <= PEAK_TOL_HZ
        assert abs(peak_freq_in_band(f, pxx, 170, 190) - 3 * OMEGA_HZ) <= PEAK_TOL_HZ

    def test_2x_dominates_1x_and_3x(self, modeler_2s):
        """PHYSICS.md §4.2: 2X dominant (0.35 vs 0.20 coefficient) and NO 1X
        component in this fault — energy(2X) > energy(1X) and > energy(3X)."""
        x = generate(modeler_2s, "desalignement", seed=0)
        f, pxx = psd(x)
        e_1x = band_energy(f, pxx, OMEGA_HZ - 2, OMEGA_HZ + 2)
        e_2x = band_energy(f, pxx, 2 * OMEGA_HZ - 2, 2 * OMEGA_HZ + 2)
        e_3x = band_energy(f, pxx, 3 * OMEGA_HZ - 2, 3 * OMEGA_HZ + 2)
        assert e_2x > MARGIN * e_1x
        assert e_2x > 2.0 * e_3x  # (0.35/0.20)^2 ~ 3.06 expected

    def test_severity_scales_rms(self, modeler_2s):
        """PHYSICS.md §2: the severity curve multiplies the fault signature
        pointwise — RMS at severity 0.9 must exceed RMS at 0.3 (same seed,
        so phases are identical; expected ratio is exactly 3.0)."""
        x_high = generate(modeler_2s, "desalignement", seed=5, severity=0.9)
        x_low = generate(modeler_2s, "desalignement", seed=5, severity=0.3)
        assert rms(x_high) > 2.0 * rms(x_low)


# ---------------------------------------------------------------------------
# §4.3 desequilibre (imbalance)
# ---------------------------------------------------------------------------


class TestDesequilibre:
    def test_dominant_peak_at_1x(self, modeler_2s):
        """PHYSICS.md §4.3: pure 1X tone — the globally dominant PSD peak
        must lie within +/-1 Hz of Omega (60 Hz)."""
        x = generate(modeler_2s, "desequilibre", seed=0)
        f, pxx = psd(x)
        assert abs(dominant_freq(f, pxx) - OMEGA_HZ) <= PEAK_TOL_HZ

    def test_rms_monotone_in_speed_variation(self, modeler_2s):
        """PHYSICS.md §4.3: amplitude grows with the SQUARE of speed
        (centrifugal F = m*r*omega^2) — RMS at speed_variation = 1.1 must
        exceed RMS at 0.9 (same seed; expected ratio (1.1/0.9)^2 ~ 1.49)."""
        x_fast = generate(modeler_2s, "desequilibre", seed=2, speed_variation=1.1)
        x_slow = generate(modeler_2s, "desequilibre", seed=2, speed_variation=0.9)
        assert rms(x_fast) > 1.2 * rms(x_slow)


# ---------------------------------------------------------------------------
# §4.4 jeu (bearing clearance / looseness)
# ---------------------------------------------------------------------------


class TestJeu:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_subsync_peak_dominates(self, modeler_2s, seed):
        """PHYSICS.md §4.4: sub-synchronous component at (0.43-0.48)*Omega is
        the strongest term (0.25 vs 0.18/0.10) — the globally dominant peak
        must lie in the documented [0.40, 0.50]*Omega band (frequency is
        randomized per signal, so only the band is asserted)."""
        x = generate(modeler_2s, "jeu", seed=seed)
        f, pxx = psd(x)
        peak = dominant_freq(f, pxx)
        assert 0.40 * OMEGA_HZ <= peak <= 0.50 * OMEGA_HZ

    def test_1x_and_2x_harmonics_present(self, modeler_2s):
        """PHYSICS.md §4.4: looseness adds 1X and 2X harmonics on top of the
        sub-synchronous motion — both bands must clearly exceed the quiet
        far-band noise reference."""
        x = generate(modeler_2s, "jeu", seed=0)
        f, pxx = psd(x)
        e_far = band_energy(f, pxx, *FAR_BAND)
        e_1x = band_energy(f, pxx, OMEGA_HZ - 2, OMEGA_HZ + 2)
        e_2x = band_energy(f, pxx, 2 * OMEGA_HZ - 2, 2 * OMEGA_HZ + 2)
        assert e_1x > MARGIN * e_far
        assert e_2x > MARGIN * e_far


# ---------------------------------------------------------------------------
# §4.5 lubrification (lubrication deficiency)
# ---------------------------------------------------------------------------


class TestLubrification:
    @pytest.mark.parametrize("seed", [0, 3])
    def test_low_frequency_band_dominates(self, modeler_5s, seed):
        """PHYSICS.md §4.5: 2-5 Hz stick-slip oscillation — energy in the
        1-6 Hz band must dominate the 50-70 Hz (1X region) band. T = 5 s
        gives the 0.2 Hz resolution needed at these frequencies."""
        x = generate(modeler_5s, "lubrification", seed=seed)
        f, pxx = psd(x)
        e_low = band_energy(f, pxx, 1.0, 6.0)
        e_1x_region = band_energy(f, pxx, 50.0, 70.0)
        assert e_low > MARGIN * e_1x_region

    def test_rms_monotone_decreasing_in_sommerfeld(self, modeler_5s):
        """PHYSICS.md §4.5: amplitude ~ 0.3/S (low S = thin film = boundary
        contact) — RMS at S = 0.08 must exceed RMS at S = 0.40 (same seed;
        expected ratio 5.0)."""
        x_thin = generate(modeler_5s, "lubrification", seed=3, sommerfeld=0.08)
        x_thick = generate(modeler_5s, "lubrification", seed=3, sommerfeld=0.40)
        assert rms(x_thin) > 2.0 * rms(x_thick)

    def test_kurtosis_sine_like(self, modeler_5s):
        """PHYSICS.md §4.5 statistical note: the stick-slip sine dominates the
        fourth moment — Pearson kurtosis ~1.5 (sine-like), NOT impulsive (>3).
        The 1-4 friction impacts (~20 samples each of 102,400) are
        fourth-moment-negligible. This pins the measured behavior after an
        earlier draft's 'kurtosis > 3' claim was refuted numerically (P3.2);
        if impacts are ever strengthened, this guard must change with the doc."""
        x = generate(modeler_5s, "lubrification", seed=0)
        k = pearson_kurtosis(x)
        assert 1.0 < k < 3.0, f"expected sine-like kurtosis (~1.5), got {k:.2f}"


# ---------------------------------------------------------------------------
# §4.6 cavitation
# ---------------------------------------------------------------------------


class TestCavitation:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_high_frequency_band_energy_fraction(self, modeler_2s, seed):
        """PHYSICS.md §4.6: bursts of a 1.5-2.5 kHz carrier — the fraction of
        total spectral energy in the 1.4-2.6 kHz band must be elevated
        (> 0.5). Burst carrier frequency is randomized, hence band check."""
        x = generate(modeler_2s, "cavitation", seed=seed)
        f, pxx = psd(x)
        e_band = band_energy(f, pxx, 1400.0, 2600.0)
        e_total = band_energy(f, pxx, 0.0, FS / 2)
        assert e_band / e_total > 0.5

    def test_kurtosis_bursty(self, modeler_2s):
        """PHYSICS.md §4.6: short transient bursts on an otherwise silent
        fault channel -> strongly super-Gaussian, kurtosis > 5 (measured
        ~120+, asserted at the documented threshold)."""
        x = generate(modeler_2s, "cavitation", seed=0)
        assert pearson_kurtosis(x) > 5.0


# ---------------------------------------------------------------------------
# §4.7 usure (wear)
# ---------------------------------------------------------------------------


class TestUsure:
    def test_spectral_flatness_exceeds_tonal_fault(self, modeler_2s):
        """PHYSICS.md §4.7: wear raises the broadband floor — its spectral
        flatness must exceed that of the purely tonal desalignement signal
        (same seed, same controlled inputs)."""
        x_wear = generate(modeler_2s, "usure", seed=0)
        x_tonal = generate(modeler_2s, "desalignement", seed=0)
        _, pxx_wear = psd(x_wear)
        _, pxx_tonal = psd(x_tonal)
        assert spectral_flatness(pxx_wear) > MARGIN * spectral_flatness(pxx_tonal)

    def test_1x_asperity_tone_present(self, modeler_2s):
        """PHYSICS.md §4.7: asperity-contact harmonics add a 1X tone above
        the broadband wear floor — 1X band energy must exceed the far-band
        broadband reference."""
        x = generate(modeler_2s, "usure", seed=0)
        f, pxx = psd(x)
        e_1x = band_energy(f, pxx, OMEGA_HZ - 2, OMEGA_HZ + 2)
        e_far = band_energy(f, pxx, *FAR_BAND)
        assert e_1x > MARGIN * e_far


# ---------------------------------------------------------------------------
# §4.8 oilwhirl
# ---------------------------------------------------------------------------


class TestOilwhirl:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_dominant_subsync_peak(self, modeler_2s, seed):
        """PHYSICS.md §4.8: whirl tone at (0.42-0.48)*Omega is the dominant
        component — globally dominant PSD peak must lie in the documented
        [0.40, 0.49]*Omega band (whirl ratio randomized per signal)."""
        x = generate(modeler_2s, "oilwhirl", seed=seed)
        f, pxx = psd(x)
        peak = dominant_freq(f, pxx)
        assert 0.40 * OMEGA_HZ <= peak <= 0.49 * OMEGA_HZ

    def test_rms_monotone_decreasing_in_sommerfeld(self, modeler_2s):
        """PHYSICS.md §4.8: whirl amplitude ~ 0.40/sqrt(S) — RMS at S = 0.08
        must exceed RMS at S = 0.40 (same seed; expected ratio ~2.24)."""
        x_low_s = generate(modeler_2s, "oilwhirl", seed=1, sommerfeld=0.08)
        x_high_s = generate(modeler_2s, "oilwhirl", seed=1, sommerfeld=0.40)
        assert rms(x_low_s) > 1.5 * rms(x_high_s)


# ---------------------------------------------------------------------------
# §4.9 mixed_misalign_imbalance
# ---------------------------------------------------------------------------


class TestMixedMisalignImbalance:
    @pytest.mark.parametrize("harmonic", [1, 2, 3])
    def test_harmonic_present(self, modeler_2s, harmonic):
        """PHYSICS.md §4.9: 1X + 2X + 3X simultaneously (misalignment 2X/3X
        superposed with imbalance 1X) — each harmonic band must clearly
        exceed the quiet far-band reference."""
        x = generate(modeler_2s, "mixed_misalign_imbalance", seed=0)
        f, pxx = psd(x)
        center = harmonic * OMEGA_HZ
        e_harm = band_energy(f, pxx, center - 2, center + 2)
        e_far = band_energy(f, pxx, *FAR_BAND)
        assert e_harm > MARGIN * e_far


# ---------------------------------------------------------------------------
# §4.10 mixed_wear_lube
# ---------------------------------------------------------------------------


class TestMixedWearLube:
    def test_low_frequency_stick_slip_elevated(self, modeler_5s):
        """PHYSICS.md §4.10: the lubrication component contributes the 2-5 Hz
        stick-slip — 1-6 Hz band energy must clearly exceed the broadband
        wear floor measured in the far band."""
        x = generate(modeler_5s, "mixed_wear_lube", seed=0)
        f, pxx = psd(x)
        e_low = band_energy(f, pxx, 1.0, 6.0)
        e_far = band_energy(f, pxx, *FAR_BAND)
        assert e_low > MARGIN * e_far

    def test_broadband_wear_component_present(self, modeler_5s):
        """PHYSICS.md §4.10: the wear component raises the broadband floor —
        spectral flatness must exceed the purely tonal desalignement's
        (both constituent signatures present simultaneously)."""
        x_mixed = generate(modeler_5s, "mixed_wear_lube", seed=0)
        x_tonal = generate(modeler_5s, "desalignement", seed=0)
        _, pxx_mixed = psd(x_mixed)
        _, pxx_tonal = psd(x_tonal)
        assert spectral_flatness(pxx_mixed) > MARGIN * spectral_flatness(pxx_tonal)


# ---------------------------------------------------------------------------
# §4.11 mixed_cavit_jeu
# ---------------------------------------------------------------------------


class TestMixedCavitJeu:
    @pytest.mark.parametrize("seed", [0, 1])
    def test_high_frequency_bursts_elevated(self, modeler_2s, seed):
        """PHYSICS.md §4.11: the cavitation component contributes 1.5-2.5 kHz
        bursts — energy in 1.4-2.6 kHz must clearly exceed an equally wide
        quiet HF band (4.0-5.2 kHz) where no model places energy."""
        x = generate(modeler_2s, "mixed_cavit_jeu", seed=seed)
        f, pxx = psd(x)
        e_hf = band_energy(f, pxx, 1400.0, 2600.0)
        e_quiet = band_energy(f, pxx, 4000.0, 5200.0)
        assert e_hf > MARGIN * e_quiet

    @pytest.mark.parametrize("seed", [0, 1])
    def test_subsync_clearance_peak_present(self, modeler_2s, seed):
        """PHYSICS.md §4.11: the jeu component contributes the (0.43-0.48)*
        Omega sub-synchronous tone — the [0.40, 0.50]*Omega band must
        clearly exceed an equally wide adjacent band, and the band's peak
        is the clearance tone (frequency randomized, band asserted)."""
        x = generate(modeler_2s, "mixed_cavit_jeu", seed=seed)
        f, pxx = psd(x)
        lo, hi = 0.40 * OMEGA_HZ, 0.50 * OMEGA_HZ
        e_sub = band_energy(f, pxx, lo, hi)
        e_adjacent = band_energy(f, pxx, 0.55 * OMEGA_HZ, 0.65 * OMEGA_HZ)
        assert e_sub > MARGIN * e_adjacent
        peak = peak_freq_in_band(f, pxx, lo, hi)
        assert 0.40 * OMEGA_HZ <= peak <= 0.50 * OMEGA_HZ
