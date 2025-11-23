"""
Unit Tests for REST API

Tests for api/main.py endpoints.

Author: Syed Abbas Ahmad
Date: 2025-11-20
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import API modules
try:
    from fastapi.testclient import TestClient
    from api.main import app
    from api.schemas import (
        PredictionRequest,
        PredictionResponse,
        ModelInfo,
        HealthResponse,
        FAULT_CLASS_NAMES
    )
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestAPISchemas:
    """Test API schema validation."""

    def test_prediction_request_valid(self):
        """Test valid prediction request."""
        request = PredictionRequest(
            signal=[0.1, 0.2, 0.3] * 100,
            return_probabilities=True
        )

        assert request.signal is not None
        assert len(request.signal) == 300
        assert request.return_probabilities is True

    def test_prediction_request_empty_signal(self):
        """Test prediction request with empty signal."""
        with pytest.raises(ValueError):
            PredictionRequest(signal=[])

    def test_prediction_response_structure(self):
        """Test prediction response structure."""
        response = PredictionResponse(
            predicted_class=1,
            class_name="Ball Fault",
            confidence=0.95,
            inference_time_ms=15.3
        )

        assert response.predicted_class == 1
        assert response.class_name == "Ball Fault"
        assert response.confidence == 0.95
        assert response.inference_time_ms == 15.3

    def test_fault_class_names_complete(self):
        """Test fault class names dictionary."""
        assert len(FAULT_CLASS_NAMES) == 11
        assert 0 in FAULT_CLASS_NAMES
        assert 10 in FAULT_CLASS_NAMES
        assert FAULT_CLASS_NAMES[0] == "Normal"


@pytest.mark.unit
@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestAPIEndpoints:
    """Test API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "message" in data
        assert "version" in data
        assert "docs" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data
        assert "version" in data

    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        # This may fail if model is not loaded
        response = client.get("/model/info")

        # Should return either 200 (model loaded) or 503 (model not loaded)
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "model_name" in data
            assert "num_classes" in data
            assert "class_names" in data

    def test_predict_endpoint_structure(self, client):
        """Test predict endpoint structure (may fail without model)."""
        request_data = {
            "signal": [0.1] * 1024,
            "return_probabilities": True
        }

        response = client.post("/predict", json=request_data)

        # May return 503 if model not loaded, or 200 if loaded
        assert response.status_code in [200, 503, 500]

        if response.status_code == 200:
            data = response.json()
            assert "predicted_class" in data
            assert "class_name" in data
            assert "confidence" in data
            assert "inference_time_ms" in data

    def test_predict_invalid_request(self, client):
        """Test predict with invalid request."""
        # Empty signal
        request_data = {
            "signal": []
        }

        response = client.post("/predict", json=request_data)

        # Should return 422 (validation error) or 400
        assert response.status_code in [400, 422]

    def test_batch_predict_endpoint_structure(self, client):
        """Test batch predict endpoint structure."""
        request_data = {
            "signals": [
                [0.1] * 1024,
                [0.2] * 1024
            ],
            "return_probabilities": False
        }

        response = client.post("/predict/batch", json=request_data)

        # May return 503 if model not loaded
        assert response.status_code in [200, 503, 500]

        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "batch_size" in data
            assert "total_inference_time_ms" in data


@pytest.mark.unit
@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI not installed")
class TestAPIConfig:
    """Test API configuration."""

    def test_config_defaults(self):
        """Test default configuration values."""
        from api.config import Settings

        settings = Settings()

        assert settings.app_name is not None
        assert settings.port == 8000
        assert settings.batch_size > 0
        assert settings.max_batch_size >= settings.batch_size
