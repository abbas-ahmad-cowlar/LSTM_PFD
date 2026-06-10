"""
Registry integrity: the model factory holds exactly the curated zoo.

Forward/backward verification of every entry lives in tests/test_zoo_smoke.py
(full signal length). This file guards the registry's shape: the tier lists
are the single source of truth — change CONVERGENCE_PLAN.md Part I §4,
test_zoo_smoke.py, and the registry together.
"""
import pytest

from packages.core.models.model_factory import MODEL_REGISTRY, create_model

from tests.test_zoo_smoke import TIER1_KEYS, TIER2_KEYS


class TestRegistryIntegrity:

    def test_registry_matches_tiers_exactly(self):
        """Every registry key is a tier member and vice versa — no aliases."""
        expected = set(TIER1_KEYS) | set(TIER2_KEYS)
        actual = set(MODEL_REGISTRY.keys())
        assert actual == expected, (
            f"Registry drift. Missing: {expected - actual}; "
            f"unexpected: {actual - expected}"
        )

    def test_registry_size(self):
        """Curated zoo: 8 Tier-1 nets + 3 Tier-2 nets = 11 registry entries."""
        assert len(MODEL_REGISTRY) == 11, (
            f"Expected 11 entries, got {len(MODEL_REGISTRY)} — "
            "tiers are fixed-size (Convergence Plan Part I §3)"
        )

    def test_all_entries_callable(self):
        for name, fn in MODEL_REGISTRY.items():
            assert callable(fn), f"Registry entry '{name}' is not callable: {type(fn)}"

    def test_unknown_key_raises_with_listing(self):
        with pytest.raises(ValueError, match="Available models"):
            create_model("efficientnet_b7")  # pruned 2026-06
