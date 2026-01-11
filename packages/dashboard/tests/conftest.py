"""
Pytest configuration and fixtures.
"""
import pytest
from database.connection import engine, SessionLocal
from models.base import Base
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


@pytest.fixture(scope="session")
def test_db():
    """Create test database."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session(test_db):
    """Create a new database session for a test."""
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    return {
        "name": "Test Dataset",
        "num_signals": 100,
        "fault_types": ["normal", "ball_fault"],
        "severity_levels": ["mild", "moderate"],
        "file_path": "/tmp/test.h5"
    }
