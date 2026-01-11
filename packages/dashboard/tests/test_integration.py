"""
Integration tests for LSTM PFD Dashboard (Phase 11D).
Tests for critical workflows and component integration.
"""
import pytest
from app import app, server
from database.connection import init_database, get_db_session
from models.experiment import Experiment, ExperimentStatus
from models.dataset import Dataset
from models.user import User
from middleware.auth import AuthMiddleware
from middleware.security import RateLimiter
from services.monitoring_service import MonitoringService
import time
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


@pytest.fixture
def client():
    """Create test client."""
    server.config['TESTING'] = True
    with server.test_client() as client:
        yield client


@pytest.fixture
def init_test_db():
    """Initialize test database."""
    init_database()
    yield
    # Cleanup handled by conftest.py


class TestAuthentication:
    """Test authentication functionality."""

    def test_create_user(self, init_test_db):
        """Test user creation."""
        success, user_id, error = AuthMiddleware.create_user(
            username="testuser",
            email="test@example.com",
            password="testpassword123",
            role="user"
        )
        assert success is True
        assert user_id is not None
        assert error is None

    def test_authenticate_user_success(self, init_test_db):
        """Test successful authentication."""
        # Create user first
        AuthMiddleware.create_user(
            username="authtest",
            email="authtest@example.com",
            password="password123",
            role="user"
        )

        # Authenticate
        success, token, error = AuthMiddleware.authenticate_user("authtest", "password123")
        assert success is True
        assert token is not None
        assert error is None

    def test_authenticate_user_failure(self, init_test_db):
        """Test failed authentication."""
        success, token, error = AuthMiddleware.authenticate_user("nonexistent", "wrongpass")
        assert success is False
        assert token is None
        assert error is not None

    def test_token_generation_and_verification(self):
        """Test JWT token generation and verification."""
        token = AuthMiddleware.generate_token(user_id=1, username="testuser")
        assert token is not None

        payload = AuthMiddleware.verify_token(token)
        assert payload is not None
        assert payload["user_id"] == 1
        assert payload["username"] == "testuser"


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_rate_limiter_allows_requests(self):
        """Test that rate limiter allows requests within limit."""
        limiter = RateLimiter(requests_per_minute=5)

        for i in range(5):
            assert limiter.is_allowed("test_ip") is True

    def test_rate_limiter_blocks_excess_requests(self):
        """Test that rate limiter blocks requests over limit."""
        limiter = RateLimiter(requests_per_minute=3)

        # First 3 should be allowed
        for i in range(3):
            assert limiter.is_allowed("test_ip") is True

        # 4th should be blocked
        assert limiter.is_allowed("test_ip") is False


class TestDatabaseModels:
    """Test database models."""

    def test_create_dataset(self, init_test_db):
        """Test dataset creation."""
        with get_db_session() as session:
            dataset = Dataset(
                name="Test Dataset",
                description="Test dataset for integration tests",
                num_samples=100,
                num_classes=NUM_CLASSES,
                sampling_rate=20480,
                signal_length = SIGNAL_LENGTH,
                file_path="/path/to/test.h5"
            )
            session.add(dataset)
            session.flush()
            assert dataset.id is not None

    def test_create_experiment(self, init_test_db):
        """Test experiment creation."""
        with get_db_session() as session:
            # Create dataset first
            dataset = Dataset(
                name="Experiment Test Dataset",
                num_samples=100,
                num_classes=NUM_CLASSES,
                sampling_rate=20480,
                signal_length = SIGNAL_LENGTH,
                file_path="/path/to/test.h5"
            )
            session.add(dataset)
            session.flush()

            # Create experiment
            experiment = Experiment(
                name="Test Experiment",
                model_type="resnet18",
                dataset_id=dataset.id,
                status=ExperimentStatus.PENDING,
                config={"learning_rate": 0.001},
                hyperparameters={"dropout": 0.3}
            )
            session.add(experiment)
            session.flush()
            assert experiment.id is not None
            assert experiment.status == ExperimentStatus.PENDING

    def test_experiment_status_transitions(self, init_test_db):
        """Test experiment status transitions."""
        with get_db_session() as session:
            # Create dataset
            dataset = Dataset(
                name="Status Test Dataset",
                num_samples=100,
                num_classes=NUM_CLASSES,
                sampling_rate=20480,
                signal_length = SIGNAL_LENGTH,
                file_path="/path/to/test.h5"
            )
            session.add(dataset)
            session.flush()

            # Create experiment
            experiment = Experiment(
                name="Status Test Experiment",
                model_type="cnn1d",
                dataset_id=dataset.id,
                status=ExperimentStatus.PENDING,
                config={}
            )
            session.add(experiment)
            session.flush()
            exp_id = experiment.id

        # Update status
        with get_db_session() as session:
            experiment = session.query(Experiment).filter_by(id=exp_id).first()
            experiment.status = ExperimentStatus.RUNNING
            session.commit()

        # Verify update
        with get_db_session() as session:
            experiment = session.query(Experiment).filter_by(id=exp_id).first()
            assert experiment.status == ExperimentStatus.RUNNING


class TestMonitoringService:
    """Test monitoring service."""

    def test_metrics_collection(self):
        """Test metrics collection."""
        monitor = MonitoringService()
        monitor._collect_metrics()

        metrics = monitor.get_current_metrics()
        assert "system" in metrics
        assert "cpu_percent" in metrics["system"]
        assert "memory_percent" in metrics["system"]

    def test_health_status(self):
        """Test health status check."""
        monitor = MonitoringService()
        monitor._collect_metrics()

        health = monitor.get_health_status()
        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy", "unknown"]


class TestWorkflows:
    """Test end-to-end workflows."""

    def test_experiment_creation_workflow(self, init_test_db):
        """Test complete experiment creation workflow."""
        # 1. Create dataset
        with get_db_session() as session:
            dataset = Dataset(
                name="Workflow Test Dataset",
                num_samples=1430,
                num_classes=NUM_CLASSES,
                sampling_rate=20480,
                signal_length = SIGNAL_LENGTH,
                file_path="/path/to/test.h5"
            )
            session.add(dataset)
            session.flush()
            dataset_id = dataset.id

        # 2. Create experiment
        with get_db_session() as session:
            experiment = Experiment(
                name="Workflow Test Experiment",
                model_type="resnet34",
                dataset_id=dataset_id,
                status=ExperimentStatus.PENDING,
                config={
                    "num_epochs": 100,
                    "batch_size": 32,
                    "learning_rate": 0.001,
                },
                hyperparameters={"dropout": 0.3}
            )
            session.add(experiment)
            session.flush()
            experiment_id = experiment.id

        # 3. Verify experiment was created
        with get_db_session() as session:
            exp = session.query(Experiment).filter_by(id=experiment_id).first()
            assert exp is not None
            assert exp.name == "Workflow Test Experiment"
            assert exp.model_type == "resnet34"
            assert exp.dataset_id == dataset_id


@pytest.mark.skipif(True, reason="Requires actual dash app running")
class TestDashCallbacks:
    """Test Dash callbacks (requires app instance)."""

    def test_navigation_callback(self, client):
        """Test navigation between pages."""
        # This would require selenium or dash testing tools
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
