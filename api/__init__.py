"""
REST API for LSTM_PFD Bearing Fault Diagnosis

FastAPI-based REST API for model inference and prediction.

Author: Syed Abbas Ahmad
Date: 2025-11-20
"""

from .main import app
from .schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    HealthResponse
)
from .config import settings

__all__ = [
    'app',
    'PredictionRequest',
    'PredictionResponse',
    'BatchPredictionRequest',
    'BatchPredictionResponse',
    'ModelInfo',
    'HealthResponse',
    'settings',
]
