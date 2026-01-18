"""
Unit Tests for Fault Type Consistency

Verifies that fault type definitions are consistent across all modules.

Author: Syed Abbas Ahmad
Date: 2026-01-19
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.mark.unit
class TestFaultTypeConsistency:
    """Test fault type definitions are consistent across all modules."""

    def test_all_fault_sources_have_same_count(self):
        """Test that all fault type definitions have exactly 11 types."""
        from utils.constants import (
            FAULT_TYPES, FAULT_LABELS_PINN, FAULT_CLASSES,
            FAULT_TYPE_DISPLAY_NAMES, NUM_CLASSES
        )
        from config.data_config import FaultConfig

        # Core constants
        assert len(FAULT_TYPES) == 11, f"FAULT_TYPES has {len(FAULT_TYPES)} items"
        assert len(FAULT_LABELS_PINN) == 11, f"FAULT_LABELS_PINN has {len(FAULT_LABELS_PINN)} items"
        assert len(FAULT_CLASSES) == 11, f"FAULT_CLASSES has {len(FAULT_CLASSES)} items"
        assert len(FAULT_TYPE_DISPLAY_NAMES) == 11, f"FAULT_TYPE_DISPLAY_NAMES has {len(FAULT_TYPE_DISPLAY_NAMES)} items"
        assert NUM_CLASSES == 11, f"NUM_CLASSES is {NUM_CLASSES}"

        # Data config
        config = FaultConfig()
        fault_list = config.get_fault_list()
        assert len(fault_list) == 11, f"FaultConfig.get_fault_list() returns {len(fault_list)} items"

    def test_pinn_labels_match_phase0_indices(self):
        """Test that FAULT_LABELS_PINN indices match Phase 0 FAULT_TYPES order."""
        from utils.constants import FAULT_TYPES, FAULT_LABELS_PINN, FAULT_TYPE_DISPLAY_NAMES

        # Verify each PINN label corresponds to the correct Phase 0 type
        expected_mappings = {
            0: ('sain', 'healthy'),
            1: ('desalignement', 'misalignment'),
            2: ('desequilibre', 'imbalance'),
            3: ('jeu', 'looseness'),
            4: ('lubrification', 'lubrication'),
            5: ('cavitation', 'cavitation'),
            6: ('usure', 'wear'),
            7: ('oilwhirl', 'oil_whirl'),
            8: ('mixed_misalign_imbalance', 'combined_misalign_imbalance'),
            9: ('mixed_wear_lube', 'combined_wear_lube'),
            10: ('mixed_cavit_jeu', 'combined_cavit_jeu'),
        }

        for idx, (french_name, english_name) in expected_mappings.items():
            # Check Phase 0 order
            assert FAULT_TYPES[idx] == french_name, \
                f"FAULT_TYPES[{idx}] = '{FAULT_TYPES[idx]}', expected '{french_name}'"
            # Check PINN label
            assert FAULT_LABELS_PINN[idx] == english_name, \
                f"FAULT_LABELS_PINN[{idx}] = '{FAULT_LABELS_PINN[idx]}', expected '{english_name}'"
            # Check display name exists
            assert french_name in FAULT_TYPE_DISPLAY_NAMES, \
                f"'{french_name}' missing from FAULT_TYPE_DISPLAY_NAMES"

    def test_dashboard_fault_classes_match_phase0_order(self):
        """Test that dashboard FAULT_CLASSES matches Phase 0 order."""
        from utils.constants import FAULT_TYPES, FAULT_CLASSES

        # The order should be: Phase 0 French -> Dashboard English
        expected_order = [
            "normal", "misalignment", "imbalance", "looseness", "lubrication",
            "cavitation", "wear", "oil_whirl", "combined_misalign_imbalance",
            "combined_wear_lube", "combined_cavit_jeu"
        ]

        assert FAULT_CLASSES == expected_order, \
            f"FAULT_CLASSES order mismatch. Got: {FAULT_CLASSES}"

    def test_dashboard_config_mappings_are_bidirectional(self):
        """Test that dashboard config mappings are bidirectionally consistent."""
        try:
            from packages.dashboard.dashboard_config import (
                DASHBOARD_TO_PHASE0_FAULT_MAP,
                PHASE0_TO_DASHBOARD_FAULT_MAP,
                FAULT_CLASSES
            )
        except ImportError:
            pytest.skip("Dashboard config not available")

        # All FAULT_CLASSES should be in dashboard-to-phase0 map
        for fault in FAULT_CLASSES:
            assert fault in DASHBOARD_TO_PHASE0_FAULT_MAP, \
                f"'{fault}' missing from DASHBOARD_TO_PHASE0_FAULT_MAP"

        # Bidirectional consistency
        for eng, fr in DASHBOARD_TO_PHASE0_FAULT_MAP.items():
            assert PHASE0_TO_DASHBOARD_FAULT_MAP[fr] == eng, \
                f"Bidirectional mismatch: {eng} <-> {fr}"

    def test_no_phantom_fault_types(self):
        """Test that no non-existent fault types are referenced."""
        from utils.constants import FAULT_CLASSES

        # These fault types should NOT exist (they're phantom types)
        phantom_types = ['ball_fault', 'inner_race', 'outer_race', 'combined', 'oil_deficiency']

        for phantom in phantom_types:
            assert phantom not in FAULT_CLASSES, \
                f"Phantom fault type '{phantom}' found in FAULT_CLASSES"

    def test_display_names_cover_all_fault_types(self):
        """Test that display names exist for all Phase 0 fault types."""
        from utils.constants import FAULT_TYPES, FAULT_TYPE_DISPLAY_NAMES

        for fault in FAULT_TYPES:
            assert fault in FAULT_TYPE_DISPLAY_NAMES, \
                f"Missing display name for '{fault}'"
            assert FAULT_TYPE_DISPLAY_NAMES[fault], \
                f"Empty display name for '{fault}'"
