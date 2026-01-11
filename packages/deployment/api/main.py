"""
FastAPI Main Application

REST API server for bearing fault diagnosis.

Author: Syed Abbas Ahmad
Date: 2025-11-20

Usage:
    # Run server
    uvicorn api.main:app --host 0.0.0.0 --port 8000

    # Or with auto-reload for development
    uvicorn api.main:app --reload
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import time
import logging
from pathlib import Path
from typing import Optional
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.constants import NUM_CLASSES, SIGNAL_LENGTH
from api.config import settings
from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ModelInfo,
    HealthResponse,
    ErrorResponse,
    FAULT_CLASS_NAMES
)
from deployment.inference import OptimizedInferenceEngine, InferenceConfig

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.log_file) if settings.log_file else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=settings.app_description,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine
inference_engine: Optional[OptimizedInferenceEngine] = None


def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key if authentication is required."""
    if settings.require_authentication:
        if not x_api_key or x_api_key != settings.api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key"
            )


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global inference_engine

    logger.info("="*60)
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info("="*60)

    try:
        # Initialize inference engine
        config = InferenceConfig(
            device=settings.device,
            batch_size=settings.batch_size,
            use_amp=settings.use_amp,
            num_threads=settings.num_threads
        )

        inference_engine = OptimizedInferenceEngine(config)

        # Load model
        logger.info(f"Loading model from: {settings.model_path}")
        inference_engine.load_model(
            settings.model_path,
            prefer_backend=settings.model_type
        )

        logger.info(f"✓ Model loaded successfully")
        logger.info(f"✓ Backend: {inference_engine.backend}")
        logger.info(f"✓ Device: {settings.device}")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        inference_engine = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API server...")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint.

    Returns service status and model information.
    """
    return HealthResponse(
        status="healthy" if inference_engine is not None else "unhealthy",
        model_loaded=inference_engine is not None,
        device=settings.device,
        version=settings.app_version
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info(api_key: str = Depends(verify_api_key)):
    """
    Get model information.

    Returns model metadata including architecture, input shape, and class names.
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfo(
        model_name="LSTM_PFD Bearing Fault Diagnosis Model",
        model_type=settings.model_type,
        num_classes=NUM_CLASSES,
        input_shape=[1, 1, SIGNAL_LENGTH],
        class_names=FAULT_CLASS_NAMES
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(
    request: PredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Make prediction on single signal.

    Args:
        request: Prediction request with signal data

    Returns:
        Prediction response with class, confidence, and probabilities

    Example:
        ```
        POST /predict
        {
            "signal": [0.1, 0.2, ..., 0.5],
            "return_probabilities": true
        }
        ```
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Prepare input
        signal = np.array(request.signal, dtype=np.float32)

        # Add batch and channel dimensions if needed
        if signal.ndim == 1:
            signal = signal[np.newaxis, np.newaxis, :]  # [1, 1, T]

        # Run inference
        start_time = time.time()
        output = inference_engine.predict(signal)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Process output
        if output.ndim == 1:
            logits = output
        else:
            logits = output[0]

        # Get prediction
        predicted_class = int(np.argmax(logits))
        probabilities = np.exp(logits) / np.sum(np.exp(logits))  # Softmax
        confidence = float(probabilities[predicted_class])

        # Prepare response
        response = PredictionResponse(
            predicted_class=predicted_class,
            class_name=FAULT_CLASS_NAMES[predicted_class],
            confidence=confidence,
            inference_time_ms=inference_time
        )

        # Add probabilities if requested
        if request.return_probabilities:
            prob_dict = {
                FAULT_CLASS_NAMES[i]: float(probabilities[i])
                for i in range(len(probabilities))
            }

            # Filter to top K if requested
            if request.return_top_k:
                sorted_probs = sorted(prob_dict.items(), key=lambda x: -x[1])
                prob_dict = dict(sorted_probs[:request.return_top_k])

            response.probabilities = prob_dict

        logger.info(f"Prediction: {response.class_name} (confidence: {confidence:.3f}, time: {inference_time:.2f}ms)")

        return response

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(
    request: BatchPredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Make predictions on batch of signals.

    Args:
        request: Batch prediction request with list of signals

    Returns:
        Batch prediction response with list of predictions

    Example:
        ```
        POST /predict/batch
        {
            "signals": [
                [0.1, 0.2, ..., 0.5],
                [0.2, 0.3, ..., 0.6]
            ],
            "return_probabilities": true
        }
        ```
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Check batch size limit
    if len(request.signals) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum ({settings.max_batch_size})"
        )

    try:
        # Prepare input
        signals = np.array(request.signals, dtype=np.float32)

        # Add channel dimension if needed
        if signals.ndim == 2:
            signals = signals[:, np.newaxis, :]  # [B, 1, T]

        # Run inference
        start_time = time.time()
        outputs = inference_engine.predict_batch(signals)
        total_inference_time = (time.time() - start_time) * 1000

        # Process outputs
        predictions = []

        for i, logits in enumerate(outputs):
            predicted_class = int(np.argmax(logits))
            probabilities = np.exp(logits) / np.sum(np.exp(logits))
            confidence = float(probabilities[predicted_class])

            pred_response = PredictionResponse(
                predicted_class=predicted_class,
                class_name=FAULT_CLASS_NAMES[predicted_class],
                confidence=confidence,
                inference_time_ms=total_inference_time / len(outputs)
            )

            if request.return_probabilities:
                pred_response.probabilities = {
                    FAULT_CLASS_NAMES[j]: float(probabilities[j])
                    for j in range(len(probabilities))
                }

            predictions.append(pred_response)

        response = BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(predictions),
            total_inference_time_ms=total_inference_time,
            average_inference_time_ms=total_inference_time / len(predictions)
        )

        logger.info(f"Batch prediction: {len(predictions)} samples in {total_inference_time:.2f}ms")

        return response

    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc)
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower()
    )
