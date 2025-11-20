"""
API Schemas

Pydantic models for request/response validation.

Author: LSTM_PFD Team
Date: 2025-11-20
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request schema for single prediction."""

    signal: List[float] = Field(
        ...,
        description="Vibration signal data (1D array)",
        min_items=1,
        max_items=102400
    )

    return_probabilities: bool = Field(
        default=True,
        description="Return class probabilities in addition to prediction"
    )

    return_top_k: Optional[int] = Field(
        default=None,
        description="Return top K predictions (default: None, returns all)"
    )

    @validator('signal')
    def validate_signal(cls, v):
        if len(v) == 0:
            raise ValueError("Signal cannot be empty")
        return v


class PredictionResponse(BaseModel):
    """Response schema for single prediction."""

    predicted_class: int = Field(
        ...,
        description="Predicted fault class (0-10)"
    )

    class_name: str = Field(
        ...,
        description="Human-readable class name"
    )

    confidence: float = Field(
        ...,
        description="Confidence score (0.0-1.0)"
    )

    probabilities: Optional[Dict[str, float]] = Field(
        default=None,
        description="Class probabilities"
    )

    inference_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds"
    )

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Prediction timestamp"
    )


class BatchPredictionRequest(BaseModel):
    """Request schema for batch prediction."""

    signals: List[List[float]] = Field(
        ...,
        description="List of vibration signals",
        min_items=1,
        max_items=128
    )

    return_probabilities: bool = Field(
        default=True,
        description="Return class probabilities"
    )

    @validator('signals')
    def validate_signals(cls, v):
        if len(v) == 0:
            raise ValueError("Signals list cannot be empty")

        # Check all signals have same length
        lengths = [len(s) for s in v]
        if len(set(lengths)) > 1:
            raise ValueError("All signals must have the same length")

        return v


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""

    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions"
    )

    batch_size: int = Field(
        ...,
        description="Number of samples in batch"
    )

    total_inference_time_ms: float = Field(
        ...,
        description="Total batch inference time"
    )

    average_inference_time_ms: float = Field(
        ...,
        description="Average inference time per sample"
    )


class ModelInfo(BaseModel):
    """Model information schema."""

    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type (torch, onnx, etc.)")
    num_classes: int = Field(..., description="Number of output classes")
    input_shape: List[int] = Field(..., description="Expected input shape")
    num_parameters: Optional[int] = Field(None, description="Number of model parameters")
    model_size_mb: Optional[float] = Field(None, description="Model size in MB")

    class_names: Dict[int, str] = Field(
        ...,
        description="Mapping of class indices to names"
    )


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Inference device (cuda/cpu)")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Fault class names
FAULT_CLASS_NAMES = {
    0: "Normal",
    1: "Ball Fault",
    2: "Inner Race Fault",
    3: "Outer Race Fault",
    4: "Combined Fault",
    5: "Imbalance",
    6: "Misalignment",
    7: "Oil Whirl",
    8: "Cavitation",
    9: "Looseness",
    10: "Oil Deficiency"
}
