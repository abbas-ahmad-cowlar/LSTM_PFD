"""
Integration Layer Tests

Smoke tests for the integration layer components:
- Phase0Adapter (data generation + MAT import)
- Phase1Adapter (classical ML training)
- DeepLearningAdapter (config validation + registry)
- ModelRegistry (register, query, leaderboard)
- ConfigurationValidator (validate_model_config)
- UnifiedMLPipeline (phase orchestration)

These are import + unit tests that do NOT require GPU or datasets.
"""

import sys
import os
import pytest
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is on path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ================================================================== #
#  1. Import smoke tests
# ================================================================== #
class TestImports:
    """Verify all integration modules import without errors."""

    def test_import_phase0_adapter(self):
        from packages.dashboard.integrations.phase0_adapter import Phase0Adapter
        assert hasattr(Phase0Adapter, 'generate_dataset')
        assert hasattr(Phase0Adapter, 'import_mat_files')

    def test_import_phase1_adapter(self):
        from packages.dashboard.integrations.phase1_adapter import Phase1Adapter
        assert hasattr(Phase1Adapter, 'train')

    def test_import_deep_learning_adapter(self):
        from packages.dashboard.integrations.deep_learning_adapter import DeepLearningAdapter
        assert hasattr(DeepLearningAdapter, 'train')

    def test_import_model_registry(self):
        from integration.model_registry import ModelRegistry
        assert hasattr(ModelRegistry, 'register_model')
        assert hasattr(ModelRegistry, 'auto_register')
        assert hasattr(ModelRegistry, 'get_leaderboard')

    def test_import_configuration_validator(self):
        from integration.configuration_validator import validate_config, validate_model_config
        assert callable(validate_config)
        assert callable(validate_model_config)

    def test_import_data_pipeline_validator(self):
        from integration.data_pipeline_validator import validate_data_compatibility
        assert callable(validate_data_compatibility)

    def test_import_unified_pipeline(self):
        from integration.unified_pipeline import UnifiedMLPipeline
        assert hasattr(UnifiedMLPipeline, 'run_full_pipeline')

    def test_import_integration_package(self):
        """Test the integration/__init__.py exports."""
        from integration import UnifiedMLPipeline, ModelRegistry
        assert UnifiedMLPipeline is not None
        assert ModelRegistry is not None


# ================================================================== #
#  2. ModelRegistry unit tests
# ================================================================== #
class TestModelRegistry:
    """Test ModelRegistry CRUD operations."""

    @pytest.fixture
    def registry(self, tmp_path):
        from integration.model_registry import ModelRegistry
        db_path = tmp_path / "test_registry.db"
        return ModelRegistry(str(db_path))

    def test_register_and_query(self, registry):
        model_id = registry.register_model(
            model_name='TestCNN',
            phase='test',
            accuracy=0.95,
            model_path='/tmp/test.pth',
            precision=0.94,
            recall=0.93,
            f1_score=0.935,
            training_duration_s=120.0,
        )
        assert model_id == 1

        best = registry.get_best_model('accuracy')
        assert best['model_name'] == 'TestCNN'
        assert best['accuracy'] == 0.95

    def test_auto_register_success(self, registry):
        results = {
            'success': True,
            'test_accuracy': 0.88,
            'model_path': '/tmp/model.pth',
            'best_model_name': 'ResNet18',
            'precision': 0.87,
            'recall': 0.86,
            'f1_score': 0.865,
            'training_time': 60.0,
        }
        model_id = registry.auto_register(results, phase='deep_learning')
        assert model_id is not None

    def test_auto_register_failure_skipped(self, registry):
        results = {'success': False, 'error': 'test error'}
        model_id = registry.auto_register(results, phase='test')
        assert model_id is None

    def test_get_leaderboard(self, registry):
        registry.register_model('ModelA', 'test', 0.90, '/a.pth')
        registry.register_model('ModelB', 'test', 0.95, '/b.pth')
        registry.register_model('ModelC', 'other', 0.80, '/c.pth')

        lb = registry.get_leaderboard('accuracy', limit=10)
        assert len(lb) == 3
        assert lb.iloc[0]['model_name'] == 'ModelB'

    def test_get_leaderboard_phase_filter(self, registry):
        registry.register_model('ModelA', 'test', 0.90, '/a.pth')
        registry.register_model('ModelB', 'test', 0.95, '/b.pth')
        registry.register_model('ModelC', 'other', 0.80, '/c.pth')

        lb = registry.get_leaderboard('accuracy', phase='test')
        assert len(lb) == 2

    def test_list_all_models(self, registry):
        registry.register_model('M1', 'p1', 0.9, '/m1')
        registry.register_model('M2', 'p2', 0.8, '/m2')

        df = registry.list_all_models()
        assert len(df) == 2


# ================================================================== #
#  3. ConfigurationValidator tests
# ================================================================== #
class TestConfigValidator:
    """Test validate_model_config."""

    def test_valid_classical_model(self):
        from integration.configuration_validator import validate_model_config
        assert validate_model_config('rf') is True
        assert validate_model_config('svm') is True
        assert validate_model_config('gbm') is True

    def test_invalid_learning_rate(self):
        from integration.configuration_validator import validate_model_config
        with pytest.raises(ValueError, match="learning_rate"):
            validate_model_config('rf', {'lr': -0.1})

    def test_invalid_batch_size(self):
        from integration.configuration_validator import validate_model_config
        with pytest.raises(ValueError, match="batch_size"):
            validate_model_config('rf', {'batch_size': 0})

    def test_invalid_epochs(self):
        from integration.configuration_validator import validate_model_config
        with pytest.raises(ValueError, match="num_epochs"):
            validate_model_config('rf', {'num_epochs': 0})

    def test_negative_weight_decay(self):
        from integration.configuration_validator import validate_model_config
        with pytest.raises(ValueError, match="weight_decay"):
            validate_model_config('rf', {'weight_decay': -1})

    def test_valid_hyperparameters(self):
        from integration.configuration_validator import validate_model_config
        assert validate_model_config('rf', {
            'lr': 0.001,
            'batch_size': 64,
            'num_epochs': 100,
            'weight_decay': 0.0001,
        }) is True


# ================================================================== #
#  4. UnifiedMLPipeline smoke tests
# ================================================================== #
class TestUnifiedMLPipeline:
    """Smoke tests for pipeline initialization and helpers."""

    def test_init(self):
        from integration.unified_pipeline import UnifiedMLPipeline
        config = {'run_phase_0': False, 'run_phase_1': False}
        pipeline = UnifiedMLPipeline(config)
        assert pipeline.results == {}

    def test_get_data_path_default(self):
        from integration.unified_pipeline import UnifiedMLPipeline
        pipeline = UnifiedMLPipeline({})
        assert pipeline._get_data_path() == 'data/processed/signals_cache.h5'

    def test_get_data_path_from_phase0(self):
        from integration.unified_pipeline import UnifiedMLPipeline
        pipeline = UnifiedMLPipeline({})
        pipeline.results['data'] = {
            'success': True,
            'output_path': '/custom/path.h5',
        }
        assert pipeline._get_data_path() == '/custom/path.h5'

    def test_empty_run(self):
        """Pipeline with all phases disabled should succeed."""
        from integration.unified_pipeline import UnifiedMLPipeline
        config = {
            'run_phase_0': False,
            'run_phase_1': False,
            'run_phase_2_4': False,
            'run_phase_5': False,
            'run_phase_6': False,
            'run_phase_7': False,
            'run_phase_8': False,
            'run_phase_9': False,
        }
        pipeline = UnifiedMLPipeline(config)
        results = pipeline.run_full_pipeline()
        assert 'pipeline_time' in results


# ================================================================== #
#  5. Phase0Adapter structure test
# ================================================================== #
class TestPhase0AdapterStructure:
    """Verify Phase0Adapter interface without running actual generation."""

    def test_generate_dataset_interface(self):
        from packages.dashboard.integrations.phase0_adapter import Phase0Adapter
        import inspect
        sig = inspect.signature(Phase0Adapter.generate_dataset)
        params = list(sig.parameters.keys())
        assert 'config' in params
        assert 'progress_callback' in params

    def test_import_mat_files_interface(self):
        from packages.dashboard.integrations.phase0_adapter import Phase0Adapter
        import inspect
        sig = inspect.signature(Phase0Adapter.import_mat_files)
        params = list(sig.parameters.keys())
        assert 'config' in params
        assert 'mat_file_paths' in params
        assert 'progress_callback' in params


# ================================================================== #
#  6. Phase1Adapter structure test
# ================================================================== #
class TestPhase1AdapterStructure:
    """Verify Phase1Adapter interface without running actual training."""

    def test_train_interface(self):
        from packages.dashboard.integrations.phase1_adapter import Phase1Adapter
        import inspect
        sig = inspect.signature(Phase1Adapter.train)
        params = list(sig.parameters.keys())
        assert 'config' in params
        assert 'progress_callback' in params
