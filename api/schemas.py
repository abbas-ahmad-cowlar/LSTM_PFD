"""
API Schemas

Pydantic models for request/response validation.

Author: Syed Abbas Ahmad
Date: 2025-11-20
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime, timezone
from utils.constants import SIGNAL_LENGTH, FAULT_TYPES, FAULT_TYPE_DISPLAY_NAMES


class PredictionRequest(BaseModel):
    """Request schema for single prediction."""

    signal: List[float] = Field(
        ...,
        description="Vibration signal data (1D array)",
        min_items=1,
        max_items=SIGNAL_LENGTH
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
        if len(v) != SIGNAL_LENGTH:
            raise ValueError(
                f"Signal must be exactly {SIGNAL_LENGTH} samples long. "
                f"Got {len(v)} samples. Please resample or pad your signal."
            )
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
        default_factory=lambda: datetime.now(timezone.utc),
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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Inference device (cuda/cpu)")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """Error response schema."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Fault class names - Generated from constants to ensure consistency
FAULT_CLASS_NAMES = {
    i: FAULT_TYPE_DISPLAY_NAMES[fault]
    for i, fault in enumerate(FAULT_TYPES)
}
