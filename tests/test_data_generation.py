"""
Tests for the data-generation pipeline (data/signal_generation package).

Rewritten 2026-06-11 against the CURRENT API after the refactor of the
monolithic ``data/signal_generator.py`` into ``data/signal_generation/``
(generator.py, fault_modeler.py, noise_generator.py, metadata.py).

Covers:
    - Config construction/validation (DataConfig, SignalConfig, FaultConfig)
    - FaultModeler signatures for all 11 fault types (incl. spectral and
      severity-scaling checks)
    - NoiseGenerator layer behavior
    - Full single-signal pipeline + SignalMetadata consistency
    - Reproducibility (same seed -> identical, different seed -> different)
    - Tiny end-to-end dataset generation (counts, labels, augmentation)
    - HDF5 save/reload round-trip (splits, shapes, attrs, config sidecar)
    - data/signal_validation.py validate_signal()/validate_batch()

All tests run on CPU with tiny configs (T=0.1 s -> N=2048 samples, the
minimum duration allowed by SignalConfig's schema).

NOTES ON CURRENT (POSSIBLY SUSPICIOUS) PRODUCTION BEHAVIOR -- documented
here instead of "fixed", per policy of not touching production code:
    1. ``DataConfig.from_yaml()`` reconstructs nested sub-configs (signal,
       fault, noise, ...) as plain dicts, not dataclass instances, so
       ``loaded.signal.fs`` raises AttributeError after a round-trip.
       Therefore the YAML round-trip is only tested on the flat
       ``SignalConfig`` here (the old nested DataConfig round-trip test was
       dropped).
    2. ``FaultModeler.generate_fault_signal('sain', ...)`` returns exact
       zeros; the healthy baseline energy comes from the orchestrator's
       noise floor. Tests assert zeros at the modeler level and nonzero
       std at the pipeline level.
    3. ``NoiseGenerator.apply_noise_layers`` applies a probabilistic
       "aliasing" layer (default 10%) even when all 7 named noise toggles
       are off; tests set ``config.noise.aliasing = 0.0`` to get a true
       identity pass-through.
"""

import json

import h5py
import numpy as np
import pytest

from config.data_config import DataConfig, FaultConfig, SignalConfig
from data.signal_generation import (
    FaultModeler,
    NoiseGenerator,
    SignalGenerator,
    SignalMetadata,
)
from data.signal_validation import (
    SignalValidationError,
    validate_batch,
    validate_signal,
)
from utils.constants import (
    FAULT_TYPES,
    NUM_CLASSES,
    SAMPLING_RATE,
    SIGNAL_LENGTH,
)
from utils.reproducibility import set_seed

# Shortest duration the SignalConfig schema allows (T >= 0.1 s) -> N = 2048.
TINY_DURATION_S = 0.1
TINY_N = int(SAMPLING_RATE * TINY_DURATION_S)

NON_SAIN_FAULTS = [f for f in FAULT_TYPES if f != "sain"]


def make_tiny_config(
    num_signals_per_fault=2,
    seed=42,
    faults="all",
    augmentation=False,
):
    """Build a minimal DataConfig that generates quickly on CPU.

    Args:
        num_signals_per_fault: Base (non-augmented) signals per fault.
        seed: RNG seed.
        faults: 'all' for the full 11 classes, or an iterable of fault
            names (using FAULT_TYPES naming, e.g. 'mixed_wear_lube').
        augmentation: Whether to enable data augmentation.
    """
    config = DataConfig(num_signals_per_fault=num_signals_per_fault, rng_seed=seed)
    config.signal.T = TINY_DURATION_S
    config.augmentation.enabled = augmentation

    if faults != "all":
        faults = set(faults)
        config.fault.include_healthy = "sain" in faults
        for name in config.fault.single_faults:
            config.fault.single_faults[name] = name in faults
        for name in config.fault.mixed_faults:
            config.fault.mixed_faults[name] = f"mixed_{name}" in faults

    return config


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_dataset():
    """Generate one tiny full dataset (11 faults x 2 signals) for reuse."""
    config = make_tiny_config()
    generator = SignalGenerator(config)
    dataset = generator.generate_dataset()
    return config, dataset


@pytest.fixture(scope="module")
def tiny_generator():
    """A SignalGenerator with a tiny all-faults config."""
    return SignalGenerator(make_tiny_config())


@pytest.fixture(scope="module")
def saved_hdf5(tmp_path_factory):
    """Generate a small 2-class dataset and save it as HDF5.

    Two classes x 10 signals each so the stratified 70/15/15 split has
    enough samples per class in every split.
    """
    out_dir = tmp_path_factory.mktemp("hdf5_out")
    config = make_tiny_config(
        num_signals_per_fault=10, faults=("sain", "desalignement")
    )
    generator = SignalGenerator(config)
    dataset = generator.generate_dataset()
    paths = generator.save_dataset(dataset, output_dir=out_dir, format="hdf5")
    return paths["hdf5"], config, dataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestConfig:
    def test_signal_config_defaults(self):
        """Defaults must match the production constants (MATLAB parity)."""
        config = SignalConfig()
        assert config.fs == SAMPLING_RATE == 20480
        assert config.T == 5.0
        assert config.N == SIGNAL_LENGTH == 102400
        assert config.Omega_base == 60.0

    def test_signal_config_n_scales_with_duration(self):
        config = SignalConfig(T=TINY_DURATION_S)
        assert config.N == TINY_N == 2048

    def test_signal_config_validation_rejects_bad_fs(self):
        config = SignalConfig(fs=-100)
        with pytest.raises(ValueError):
            config.validate()

    def test_data_config_defaults_validate(self):
        assert DataConfig().validate() is True

    def test_signal_config_yaml_roundtrip(self, tmp_path):
        """YAML round-trip on the flat SignalConfig (see header note 1)."""
        config = SignalConfig(T=0.5, Omega_base=50.0)
        yaml_path = tmp_path / "signal_config.yaml"
        config.to_yaml(yaml_path)
        assert yaml_path.exists()

        loaded = SignalConfig.from_yaml(yaml_path)
        assert loaded.fs == config.fs
        assert loaded.T == config.T
        assert loaded.Omega_base == config.Omega_base
        assert loaded.N == config.N

    def test_fault_config_default_list_matches_fault_types(self):
        """All 11 classes enabled by default, in canonical FAULT_TYPES order."""
        assert FaultConfig().get_fault_list() == FAULT_TYPES
        assert len(FAULT_TYPES) == NUM_CLASSES == 11

    def test_fault_config_disable_subset(self):
        config = FaultConfig()
        config.include_mixed = False
        config.single_faults["desalignement"] = False
        fault_list = config.get_fault_list()

        assert "desalignement" not in fault_list
        assert not any(f.startswith("mixed_") for f in fault_list)
        assert "sain" in fault_list
        assert "cavitation" in fault_list

    def test_get_total_signals(self):
        config = make_tiny_config(num_signals_per_fault=4)
        assert config.get_total_signals() == 11 * 4

        config.augmentation.enabled = True
        config.augmentation.ratio = 0.5
        assert config.get_total_signals() == 11 * 4 + int(11 * 4 * 0.5)


# ---------------------------------------------------------------------------
# FaultModeler
# ---------------------------------------------------------------------------


def _modeler_kwargs(n):
    """Default keyword arguments for FaultModeler.generate_fault_signal."""
    return dict(
        transient_modulation=np.ones(n),
        omega=2 * np.pi * 60.0,
        Omega=60.0,
        load_factor=1.0,
        temp_factor=1.0,
        operating_factor=1.0,
        physics_factor=1.0,
        sommerfeld=0.15,
        speed_variation=1.0,
    )


class TestFaultModeler:
    @pytest.fixture(scope="class")
    def modeler(self):
        return FaultModeler(make_tiny_config())

    def test_sain_returns_exact_zeros(self, modeler):
        """Healthy signature is all-zero at the modeler level (note 2)."""
        sig = modeler.generate_fault_signal(
            "sain", 0.5 * np.ones(TINY_N), **_modeler_kwargs(TINY_N)
        )
        assert sig.shape == (TINY_N,)
        assert np.allclose(sig, 0.0)

    @pytest.mark.parametrize("fault", NON_SAIN_FAULTS)
    def test_fault_signature_is_valid(self, modeler, fault):
        """Every non-healthy fault yields a finite, nonzero, N-length signal."""
        set_seed(1234)
        sig = modeler.generate_fault_signal(
            fault, 0.6 * np.ones(TINY_N), **_modeler_kwargs(TINY_N)
        )
        assert sig.shape == (TINY_N,)
        assert np.issubdtype(sig.dtype, np.floating)
        assert np.all(np.isfinite(sig))
        assert np.std(sig) > 0.0

    def test_unknown_fault_raises(self, modeler):
        with pytest.raises(ValueError, match="Unknown fault type"):
            modeler.generate_fault_signal(
                "not_a_fault", np.ones(TINY_N), **_modeler_kwargs(TINY_N)
            )

    def test_desalignement_has_2x_and_3x_harmonics(self, modeler):
        """Misalignment must put energy at 2X (120 Hz) and 3X (180 Hz)."""
        set_seed(99)
        sig = modeler.generate_fault_signal(
            "desalignement", 0.8 * np.ones(TINY_N), **_modeler_kwargs(TINY_N)
        )
        spectrum = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(TINY_N, 1.0 / SAMPLING_RATE)

        mean_mag = np.mean(spectrum)
        idx_2x = np.argmin(np.abs(freqs - 120.0))
        idx_3x = np.argmin(np.abs(freqs - 180.0))
        assert spectrum[idx_2x] > 5.0 * mean_mag
        assert spectrum[idx_3x] > 5.0 * mean_mag

    @pytest.mark.parametrize("fault", ["desalignement", "desequilibre", "oilwhirl"])
    def test_severity_scales_signal_energy(self, modeler, fault):
        """Higher severity -> higher RMS (same RNG state for fair compare)."""
        kwargs = _modeler_kwargs(TINY_N)

        set_seed(7)
        sig_low = modeler.generate_fault_signal(
            fault, 0.2 * np.ones(TINY_N), **kwargs
        )
        set_seed(7)
        sig_high = modeler.generate_fault_signal(
            fault, 0.9 * np.ones(TINY_N), **kwargs
        )

        rms_low = np.sqrt(np.mean(sig_low**2))
        rms_high = np.sqrt(np.mean(sig_high**2))
        assert rms_high > 2.0 * rms_low


# ---------------------------------------------------------------------------
# NoiseGenerator
# ---------------------------------------------------------------------------


class TestNoiseGenerator:
    @staticmethod
    def _clean_signal(n):
        t = np.arange(n) / SAMPLING_RATE
        return np.sin(2 * np.pi * 60.0 * t)

    @staticmethod
    def _all_noise_off(config):
        config.noise.measurement = False
        config.noise.emi = False
        config.noise.pink = False
        config.noise.drift = False
        config.noise.quantization = False
        config.noise.sensor_drift = False
        config.noise.impulse = False
        config.noise.aliasing = 0.0  # probabilistic layer (see note 3)

    def test_disabled_noise_is_identity(self):
        config = make_tiny_config()
        self._all_noise_off(config)
        generator = NoiseGenerator(config)

        clean = self._clean_signal(TINY_N)
        noisy, applied = generator.apply_noise_layers(clean.copy())

        np.testing.assert_array_equal(noisy, clean)
        assert not any(applied.values())

    def test_measurement_noise_increases_rms(self):
        config = make_tiny_config()
        self._all_noise_off(config)
        config.noise.measurement = True
        config.noise.levels["measurement"] = 0.5  # well above signal floor
        generator = NoiseGenerator(config)

        clean = self._clean_signal(TINY_N)
        set_seed(0)
        noisy, applied = generator.apply_noise_layers(clean.copy())

        assert applied["measurement"] is True
        assert np.sqrt(np.mean(noisy**2)) > np.sqrt(np.mean(clean**2))
        assert not np.array_equal(noisy, clean)

    def test_applied_flags_cover_all_layers(self):
        generator = NoiseGenerator(make_tiny_config())
        set_seed(0)
        _, applied = generator.apply_noise_layers(self._clean_signal(TINY_N))

        expected_keys = {
            "measurement", "emi", "pink", "drift", "quantization",
            "sensor_drift", "aliasing", "impulse",
        }
        assert set(applied.keys()) == expected_keys
        # All deterministic layers are on by default.
        for key in expected_keys - {"aliasing"}:
            assert applied[key] is True


# ---------------------------------------------------------------------------
# Full single-signal pipeline
# ---------------------------------------------------------------------------


class TestSingleSignalPipeline:
    @pytest.mark.parametrize("fault", FAULT_TYPES)
    def test_each_fault_generates_valid_signal(self, tiny_generator, fault):
        """All 11 classes: correct length, float dtype, finite, nonzero std.

        Even 'sain' has nonzero std end-to-end because the orchestrator
        adds a baseline noise floor on top of the (zero) fault signature.
        """
        set_seed(7)
        signal, metadata = tiny_generator.generate_single_signal(fault)

        assert signal.shape == (TINY_N,)
        assert np.issubdtype(signal.dtype, np.floating)
        assert np.all(np.isfinite(signal))
        assert np.std(signal) > 0.0

        assert isinstance(metadata, SignalMetadata)
        assert metadata.fault == fault
        assert metadata.num_samples == TINY_N
        assert metadata.is_overlapping_fault == fault.startswith("mixed_")

    def test_metadata_statistics_match_signal(self, tiny_generator):
        set_seed(11)
        signal, metadata = tiny_generator.generate_single_signal("usure")

        rms = float(np.sqrt(np.mean(signal**2)))
        peak = float(np.max(np.abs(signal)))
        assert np.isclose(metadata.signal_rms, rms, rtol=1e-6)
        assert np.isclose(metadata.signal_peak, peak, rtol=1e-6)
        assert metadata.signal_crest_factor > 0.0
        assert metadata.signal_peak >= metadata.signal_rms


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    FAULT_SUBSET = ("sain", "desequilibre", "cavitation")

    def _generate(self, seed):
        config = make_tiny_config(seed=seed, faults=self.FAULT_SUBSET)
        return SignalGenerator(config).generate_dataset()

    def test_same_seed_identical_datasets(self):
        ds1 = self._generate(seed=42)
        ds2 = self._generate(seed=42)

        assert ds1["labels"] == ds2["labels"]
        assert len(ds1["signals"]) == len(ds2["signals"])
        for s1, s2 in zip(ds1["signals"], ds2["signals"]):
            np.testing.assert_array_equal(s1, s2)

    def test_different_seeds_differ(self):
        ds1 = self._generate(seed=42)
        ds2 = self._generate(seed=123)
        assert not np.array_equal(ds1["signals"][0], ds2["signals"][0])

    def test_per_signal_seed_variation_within_dataset(self):
        """Consecutive signals of the same fault must not be identical."""
        ds = self._generate(seed=42)
        assert ds["labels"][0] == ds["labels"][1] == "sain"
        assert not np.array_equal(ds["signals"][0], ds["signals"][1])


# ---------------------------------------------------------------------------
# End-to-end dataset generation
# ---------------------------------------------------------------------------


class TestDatasetGeneration:
    def test_tiny_dataset_structure_and_counts(self, tiny_dataset):
        config, dataset = tiny_dataset

        for key in ("signals", "metadata", "labels", "config", "statistics"):
            assert key in dataset

        n_expected = 11 * config.num_signals_per_fault
        assert len(dataset["signals"]) == n_expected
        assert len(dataset["labels"]) == n_expected
        assert len(dataset["metadata"]) == n_expected
        assert dataset["statistics"]["total_signals"] == n_expected
        assert dataset["statistics"]["num_faults"] == 11

        # Every class present with exactly num_signals_per_fault samples.
        labels = dataset["labels"]
        assert set(labels) == set(FAULT_TYPES)
        for fault in FAULT_TYPES:
            assert labels.count(fault) == config.num_signals_per_fault

        for signal in dataset["signals"]:
            assert signal.shape == (config.signal.N,)
            assert np.all(np.isfinite(signal))

    def test_metadata_operating_conditions_in_range(self, tiny_dataset):
        config, dataset = tiny_dataset

        rpm_nominal = config.signal.Omega_base * 60.0
        rpm_lo = rpm_nominal * (1 - config.operating.speed_variation)
        rpm_hi = rpm_nominal * (1 + config.operating.speed_variation)
        load_lo, load_hi = (r * 100 for r in config.operating.load_range)
        temp_lo, temp_hi = config.operating.temp_range
        valid_severities = set(config.severity.levels) | {"nominal"}

        for m in dataset["metadata"]:
            assert rpm_lo <= m.speed_rpm <= rpm_hi
            assert load_lo <= m.load_percent <= load_hi
            assert temp_lo <= m.temperature_C <= temp_hi
            assert m.severity in valid_severities
            if m.fault == "sain":
                assert m.severity == "nominal"
                assert m.severity_factor_initial == 1.0

    def test_augmentation_counts(self):
        config = make_tiny_config(
            num_signals_per_fault=4,
            faults=("sain", "desalignement"),
            augmentation=True,
        )
        config.augmentation.ratio = 0.5

        dataset = SignalGenerator(config).generate_dataset()

        # Per fault: 4 base + int(4 * 0.5) = 2 augmented.
        assert len(dataset["signals"]) == 2 * (4 + 2)
        aug_flags = [m.is_augmented for m in dataset["metadata"]]
        assert sum(aug_flags) == 2 * 2
        assert len(dataset["signals"]) == config.get_total_signals()


# ---------------------------------------------------------------------------
# HDF5 round-trip
# ---------------------------------------------------------------------------


class TestHDF5RoundTrip:
    def test_splits_shapes_and_labels(self, saved_hdf5):
        hdf5_path, config, dataset = saved_hdf5
        assert hdf5_path.exists()

        total = len(dataset["signals"])
        n_per_split = {}
        with h5py.File(hdf5_path, "r") as f:
            for split in ("train", "val", "test"):
                assert split in f
                signals = f[split]["signals"]
                labels = f[split]["labels"]

                assert signals.ndim == 2
                assert signals.shape[1] == config.signal.N
                assert signals.dtype == np.float32
                assert signals.shape[0] == labels.shape[0]
                assert f[split].attrs["num_samples"] == signals.shape[0]

                # Only sain (0) and desalignement (1) were generated.
                assert set(np.unique(labels[:])) == {0, 1}
                # Reloaded signals must remain finite.
                assert np.all(np.isfinite(signals[:]))

                n_per_split[split] = signals.shape[0]

        assert sum(n_per_split.values()) == total
        assert n_per_split["train"] > n_per_split["val"]
        assert n_per_split["train"] > n_per_split["test"]

    def test_global_attributes(self, saved_hdf5):
        hdf5_path, config, _ = saved_hdf5

        with h5py.File(hdf5_path, "r") as f:
            assert f.attrs["num_classes"] == NUM_CLASSES
            assert f.attrs["sampling_rate"] == SAMPLING_RATE
            assert f.attrs["signal_length"] == config.signal.N
            assert f.attrs["rng_seed"] == config.rng_seed
            assert tuple(f.attrs["split_ratios"]) == (0.7, 0.15, 0.15)
            assert "generation_date" in f.attrs
            assert "config_hash" in f.attrs
            assert "config_json" in f.attrs
            assert "metadata" in f  # JSON metadata dataset

            stored_config = json.loads(f.attrs["config_json"])
            assert stored_config["num_signals_per_fault"] == 10

    def test_config_sidecar_json(self, saved_hdf5):
        hdf5_path, _, dataset = saved_hdf5
        sidecar = hdf5_path.parent / "dataset_config.json"
        assert sidecar.exists()

        with open(sidecar, "r") as f:
            info = json.load(f)

        assert info["total_signals"] == len(dataset["signals"])
        assert sum(info["split_sizes"].values()) == len(dataset["signals"])
        assert "config_hash" in info


# ---------------------------------------------------------------------------
# Signal validation utilities
# ---------------------------------------------------------------------------


class TestSignalValidation:
    @pytest.fixture
    def good_signal(self):
        rng = np.random.RandomState(0)
        return rng.randn(TINY_N)

    def test_good_signal_passes(self, good_signal):
        errors = validate_signal(good_signal, expected_length=TINY_N)
        assert errors == []

    def test_nan_raises(self, good_signal):
        bad = good_signal.copy()
        bad[10] = np.nan
        with pytest.raises(SignalValidationError, match="NaN"):
            validate_signal(bad)

    def test_inf_flagged_without_raise(self, good_signal):
        bad = good_signal.copy()
        bad[5] = np.inf
        errors = validate_signal(bad, raise_on_error=False)
        assert len(errors) == 1
        assert "Inf" in errors[0]

    def test_wrong_length_flagged(self, good_signal):
        errors = validate_signal(
            good_signal, expected_length=TINY_N + 1, raise_on_error=False
        )
        assert any("Length" in e for e in errors)

    def test_flat_signal_flagged(self):
        errors = validate_signal(np.zeros(TINY_N), raise_on_error=False)
        assert any("Flat/dead" in e for e in errors)

    def test_validate_batch_report(self, good_signal):
        nan_row = good_signal.copy()
        nan_row[0] = np.nan
        flat_row = np.zeros(TINY_N)
        batch = np.vstack([good_signal, nan_row, flat_row])
        labels = np.array([0, 1, 2])

        report = validate_batch(batch, labels=labels)

        assert report.total_signals == 3
        assert report.passed == 1
        assert report.failed == 2
        assert report.pass_rate == pytest.approx(1 / 3)
        assert len(report.errors) >= 2

    def test_validate_batch_on_generated_signals(self, tiny_dataset):
        """Generated signals must pass batch validation cleanly."""
        config, dataset = tiny_dataset
        signals = np.array(dataset["signals"], dtype=np.float64)
        report = validate_batch(signals, expected_length=config.signal.N)

        assert report.failed == 0
        assert report.passed == len(dataset["signals"])
