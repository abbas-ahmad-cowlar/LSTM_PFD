# API Reference

Complete API documentation for LSTM_PFD prediction system.

**Version:** 1.0.0
**Base URL:** `http://localhost:8000`
**API Docs:** `http://localhost:8000/docs` (Swagger UI)

---

## Table of Contents

- [REST API Endpoints](#rest-api-endpoints)
- [Python SDK](#python-sdk)
- [Request/Response Schemas](#requestresponse-schemas)
- [Error Handling](#error-handling)
- [Rate Limits](#rate-limits)
- [Authentication](#authentication)

---

## REST API Endpoints

### 1. Health Check

**GET** `/health`

Check API server health status.

**Response 200:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "version": "1.0.0",
  "uptime_seconds": 3600
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### 2. Root Information

**GET** `/`

Get API information and available endpoints.

**Response 200:**
```json
{
  "message": "LSTM_PFD Fault Diagnosis API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

---

### 3. Model Information

**GET** `/model/info`

Get information about the loaded model.

**Response 200:**
```json
{
  "model_name": "EnsembleModel",
  "model_version": "1.0.0",
  "num_classes": 11,
  "class_names": {
    "0": "Normal",
    "1": "Ball Fault",
    "2": "Inner Race Fault",
    ...
  },
  "input_shape": [1, 102400],
  "inference_device": "cuda"
}
```

**Response 503:**
```json
{
  "detail": "Model not loaded"
}
```

---

### 4. Single Prediction

**POST** `/predict`

Predict fault class for a single vibration signal.

**Request Body:**
```json
{
  "signal": [0.1, 0.2, ..., 0.5],  // 102,400 floats
  "return_probabilities": true,
  "return_features": false
}
```

**Parameters:**
- `signal` (required): Array of 102,400 float values (vibration signal @ 20,480 Hz for 5 seconds)
- `return_probabilities` (optional, default=false): Include probability distribution
- `return_features` (optional, default=false): Include extracted features

**Response 200:**
```json
{
  "predicted_class": 2,
  "class_name": "Inner Race Fault",
  "confidence": 0.967,
  "inference_time_ms": 42.3,
  "probabilities": {
    "0": 0.001,
    "1": 0.012,
    "2": 0.967,
    ...
  }
}
```

**Response 400:**
```json
{
  "detail": "Invalid signal length. Expected 102400, got 50000"
}
```

**Example (Python):**
```python
import requests
import numpy as np

# Generate or load signal
signal = np.random.randn(102400).tolist()

# Make prediction
response = requests.post(
    'http://localhost:8000/predict',
    json={
        'signal': signal,
        'return_probabilities': True
    }
)

result = response.json()
print(f"Predicted: {result['class_name']}")
print(f"Confidence: {result['confidence']:.1%}")
```

**Example (cURL):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "signal": [0.1, 0.2, ...],
    "return_probabilities": true
  }'
```

---

### 5. Batch Prediction

**POST** `/predict/batch`

Predict fault classes for multiple signals in one request.

**Request Body:**
```json
{
  "signals": [
    [0.1, 0.2, ..., 0.5],  // Signal 1
    [0.3, 0.4, ..., 0.7]   // Signal 2
  ],
  "return_probabilities": false
}
```

**Parameters:**
- `signals` (required): Array of signals (each 102,400 floats)
- `return_probabilities` (optional, default=false): Include probabilities
- Maximum batch size: 32 signals

**Response 200:**
```json
{
  "predictions": [
    {
      "predicted_class": 2,
      "class_name": "Inner Race Fault",
      "confidence": 0.967
    },
    {
      "predicted_class": 0,
      "class_name": "Normal",
      "confidence": 0.991
    }
  ],
  "batch_size": 2,
  "total_inference_time_ms": 87.6,
  "avg_time_per_sample_ms": 43.8
}
```

**Example (Python):**
```python
signals = [
    np.random.randn(102400).tolist(),
    np.random.randn(102400).tolist()
]

response = requests.post(
    'http://localhost:8000/predict/batch',
    json={'signals': signals}
)

for i, pred in enumerate(response.json()['predictions']):
    print(f"Signal {i}: {pred['class_name']} ({pred['confidence']:.1%})")
```

---

## Python SDK

### Installation

```bash
pip install lstm-pfd
```

### Quick Start

```python
from lstm_pfd import InferenceEngine
import numpy as np

# Initialize engine
engine = InferenceEngine(
    model_path='checkpoints/ensemble_model.onnx',
    device='cuda'
)

# Load or generate signal
signal = np.random.randn(102400)

# Predict
result = engine.predict(signal)

print(f"Fault: {result.fault_type}")
print(f"Confidence: {result.confidence:.1%}")
```

### API Classes

#### InferenceEngine

Main class for model inference.

```python
class InferenceEngine:
    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
        batch_size: int = 32
    ):
        """
        Initialize inference engine.

        Args:
            model_path: Path to ONNX model file
            device: 'cuda' or 'cpu'
            batch_size: Batch size for processing
        """

    def predict(
        self,
        signal: np.ndarray,
        return_probabilities: bool = False
    ) -> PredictionResult:
        """
        Predict fault class for single signal.

        Args:
            signal: 1D numpy array (102,400 samples)
            return_probabilities: Include full probability distribution

        Returns:
            PredictionResult with fault_type, confidence, probabilities
        """

    def predict_batch(
        self,
        signals: np.ndarray,
        batch_size: int = 32
    ) -> List[PredictionResult]:
        """
        Predict fault classes for batch of signals.

        Args:
            signals: 2D numpy array (N, 102400)
            batch_size: Batch size for processing

        Returns:
            List of PredictionResult objects
        """
```

#### PredictionResult

Result object from inference.

```python
@dataclass
class PredictionResult:
    fault_type: str           # Human-readable fault name
    predicted_class: int      # Class index (0-10)
    confidence: float         # Confidence score (0-1)
    probabilities: Dict[int, float]  # Full probability distribution
    inference_time_ms: float  # Inference latency
```

### Complete Example

```python
from lstm_pfd import InferenceEngine
import numpy as np
from pathlib import Path

# Initialize
engine = InferenceEngine(
    model_path='models/ensemble_model.onnx',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Load signals from file
signals = np.load('test_signals.npy')

# Batch prediction
results = engine.predict_batch(signals, batch_size=16)

# Analyze results
for i, result in enumerate(results):
    print(f"\nSignal {i}:")
    print(f"  Fault: {result.fault_type}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Time: {result.inference_time_ms:.1f}ms")

    # Check for anomalies (low confidence)
    if result.confidence < 0.7:
        print(f"  ⚠️  Warning: Low confidence!")
```

---

## Request/Response Schemas

### PredictionRequest

```json
{
  "signal": [number],       // Required: Array of 102,400 floats
  "return_probabilities": boolean,  // Optional: default false
  "return_features": boolean       // Optional: default false
}
```

**Validation:**
- `signal` length must be exactly 102,400
- `signal` values must be finite numbers
- `signal` values typically in range [-10, 10] for vibration data

### PredictionResponse

```json
{
  "predicted_class": integer,      // 0-10
  "class_name": string,           // Human-readable name
  "confidence": number,           // 0.0 to 1.0
  "inference_time_ms": number,
  "probabilities": {              // Optional
    "0": number,
    ...
  },
  "features": [number]            // Optional: 36 extracted features
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid input (wrong signal length, etc.) |
| 422 | Unprocessable Entity | Validation error |
| 500 | Internal Server Error | Server error during inference |
| 503 | Service Unavailable | Model not loaded |

### Error Response Format

```json
{
  "detail": "Error message here",
  "error_code": "INVALID_SIGNAL_LENGTH",
  "timestamp": "2025-11-23T12:34:56Z"
}
```

### Common Errors

**Invalid Signal Length:**
```json
{
  "detail": "Invalid signal length. Expected 102400, got 50000"
}
```

**Non-finite Values:**
```json
{
  "detail": "Signal contains non-finite values (NaN or Inf)"
}
```

**Batch Size Exceeded:**
```json
{
  "detail": "Batch size 64 exceeds maximum of 32"
}
```

---

## Rate Limits

- **Default:** 100 requests/minute per IP
- **Batch endpoint:** 20 requests/minute per IP
- **Headers:** Rate limit info included in response headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1700000000
```

---

## Authentication

Optional API key authentication for production deployments.

**Header:**
```
X-API-Key: your-api-key-here
```

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "X-API-Key: abc123def456" \
  -H "Content-Type: application/json" \
  -d '{"signal": [...]}'
```

---

## Changelog

### v1.0.0 (2025-11-23)
- Initial release
- Single and batch prediction endpoints
- Health check and model info endpoints
- Python SDK
- OpenAPI/Swagger documentation

---

## Support

- **Documentation:** See USAGE_GUIDES/
- **GitHub:** https://github.com/abbas-ahmad-cowlar/LSTM_PFD
- **Issues:** https://github.com/abbas-ahmad-cowlar/LSTM_PFD/issues

---

**End of API Reference**
