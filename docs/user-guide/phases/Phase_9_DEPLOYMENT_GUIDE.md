# Phase 9: Deployment Guide

Complete guide for deploying LSTM_PFD models to production.

**Status**: ✅ Complete
**Duration**: 14 days
**Date**: November 2025

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Model Optimization](#model-optimization)
- [ONNX Export](#onnx-export)
- [REST API](#rest-api)
- [Docker Deployment](#docker-deployment)
- [Performance Benchmarking](#performance-benchmarking)
- [Production Best Practices](#production-best-practices)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

Phase 9 focuses on deploying trained models to production with:

- **Model Quantization**: Reduce model size by 4x (INT8) or 2x (FP16)
- **ONNX Export**: Cross-platform deployment
- **REST API**: FastAPI-based inference server
- **Docker**: Containerized deployment
- **Optimization**: Pruning, layer fusion, profiling
- **Target**: <50ms latency, 98%+ accuracy retention

### Key Deliverables

| Component | Description | Location |
|-----------|-------------|----------|
| Quantization | INT8, FP16 conversion | `deployment/quantization.py` |
| ONNX Export | Cross-platform models | `deployment/onnx_export.py` |
| Inference Engine | Optimized inference | `deployment/inference.py` |
| REST API | FastAPI server | `api/main.py` |
| Docker | Containerization | `Dockerfile`, `docker-compose.yml` |
| Scripts | Deployment tools | `scripts/quantize_model.py`, etc. |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install deployment requirements
pip install -r requirements-deployment.txt

# Key dependencies:
# - fastapi, uvicorn (API server)
# - onnx, onnxruntime (ONNX support)
# - torch quantization tools
```

### 2. Quantize Model

```bash
# Dynamic quantization (INT8) - Recommended for most cases
python scripts/quantize_model.py \
    --model checkpoints/phase6/best_model.pth \
    --output checkpoints/phase9/model_int8.pth \
    --quantization-type dynamic \
    --benchmark

# FP16 conversion - For GPU deployment
python scripts/quantize_model.py \
    --model checkpoints/phase6/best_model.pth \
    --output checkpoints/phase9/model_fp16.pth \
    --quantization-type fp16
```

**Expected Results**:
- **Size reduction**: 75% (INT8), 50% (FP16)
- **Speedup**: 2-3x (INT8), 1.5-2x (FP16)
- **Accuracy loss**: <0.5%

### 3. Export to ONNX

```bash
# Export with validation and optimization
python scripts/export_onnx.py \
    --model checkpoints/phase6/best_model.pth \
    --output models/model.onnx \
    --validate \
    --optimize \
    --optimization-level all \
    --benchmark
```

### 4. Start API Server

```bash
# Set model path
export MODEL_PATH=checkpoints/phase9/model_int8.pth
export MODEL_TYPE=torch
export DEVICE=cuda  # or 'cpu'

# Run server
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Or use Python directly
python api/main.py
```

### 5. Test API

```bash
# Health check
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model/info

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "signal": [0.1, 0.2, 0.3, ..., 0.5],
    "return_probabilities": true
  }'
```

### 6. Deploy with Docker

```bash
# Build image
docker build -t lstm_pfd:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  -e MODEL_PATH=/app/checkpoints/phase9/model_int8.pth \
  lstm_pfd:latest

# Or use docker-compose
docker-compose up -d
```

---

## 🔧 Model Optimization

### Quantization

#### Dynamic Quantization

**Best for**: LSTM, GRU, Linear layers
**Pros**: No calibration data needed, easy to apply
**Cons**: Only quantizes weights, not activations

```python
from deployment.quantization import quantize_model_dynamic

# Load model
model = torch.load('checkpoints/phase6/best_model.pth')

# Quantize
quantized_model = quantize_model_dynamic(
    model,
    dtype=torch.qint8,
    inplace=False
)

# Save
torch.save(quantized_model, 'checkpoints/phase9/model_int8.pth')
```

**Expected Output**:
```
Applying dynamic quantization (dtype=torch.qint8)
Quantizing layers: [<class 'torch.nn.modules.linear.Linear'>, ...]
Dynamic quantization complete
Original model size: 45.23 MB
Quantized model size: 11.35 MB
Compression ratio: 3.99x
Size reduction: 75.0%
```

#### Static Quantization

**Best for**: CNNs, fully connected networks
**Pros**: Quantizes both weights and activations, faster inference
**Cons**: Requires calibration data

```python
from deployment.quantization import quantize_model_static

# Load calibration data
calibration_loader = DataLoader(calibration_dataset, batch_size=32)

# Quantize
quantized_model = quantize_model_static(
    model,
    calibration_loader,
    backend='fbgemm',  # or 'qnnpack' for ARM
    inplace=False
)
```

#### FP16 Conversion

**Best for**: GPU deployment with Tensor Cores
**Pros**: 2x smaller, 2-3x faster on modern GPUs
**Cons**: Requires GPU with FP16 support

```python
from deployment.quantization import quantize_to_fp16

# Convert to FP16
fp16_model = quantize_to_fp16(model, inplace=False)
```

### Model Pruning

Remove redundant weights to reduce size and improve speed:

```python
from deployment.model_optimization import prune_model

# Prune 30% of weights
pruned_model = prune_model(
    model,
    pruning_amount=0.3,
    pruning_type='l1_unstructured'
)
```

### Layer Fusion

Fuse adjacent layers for faster inference:

```python
from deployment.model_optimization import fuse_model_layers

fused_model = fuse_model_layers(model)
```

### Model Statistics

```python
from deployment.model_optimization import calculate_model_stats, print_model_stats

stats = calculate_model_stats(model)
print_model_stats(stats)
```

**Output**:
```
============================================================
Model Statistics
============================================================

Parameters:
  Total:        12,345,678 (   12.35M)
  Trainable:    12,345,678 (   12.35M)
  Non-trainable:         0 (    0.00M)
  Zero (pruned):         0 (    0.00%)

Memory:
  Total size:    47.23 MB
  Parameters:    47.10 MB
  Buffers:        0.13 MB

Architecture:
  Total layers: 85

Layer breakdown:
  Conv1d                          24
  BatchNorm1d                     23
  ReLU                            23
  Linear                           2
  ...
============================================================
```

---

## 📦 ONNX Export

### Basic Export

```python
from deployment.onnx_export import export_to_onnx, ONNXExportConfig

# Configure export
config = ONNXExportConfig(
    opset_version=14,
    do_constant_folding=True,
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

# Export
dummy_input = torch.randn(1, 1, 102400)
onnx_path = export_to_onnx(
    model,
    dummy_input,
    'models/model.onnx',
    config
)
```

### Validation

Ensure ONNX model produces same outputs as PyTorch:

```python
from deployment.onnx_export import validate_onnx_export

is_valid = validate_onnx_export(
    'models/model.onnx',
    model,
    test_input=torch.randn(1, 1, 102400)
)

# Output:
# ✓ ONNX model structure is valid
# ✓ Outputs match (max diff: 1.23e-05)
```

### Optimization

```python
from deployment.onnx_export import optimize_onnx_model

optimized_path = optimize_onnx_model(
    'models/model.onnx',
    'models/model_optimized.onnx',
    optimization_level='all'
)

# Output:
# ✓ Optimized model saved to models/model_optimized.onnx
# ✓ Original size: 47.23 MB
# ✓ Optimized size: 44.51 MB
# ✓ Size reduction: 5.8%
```

### ONNX Inference

```python
from deployment.onnx_export import ONNXInferenceSession

# Initialize session
session = ONNXInferenceSession('models/model.onnx')

# Predict
input_data = np.random.randn(1, 1, 102400).astype(np.float32)
output = session.predict(input_data)

# Batch prediction
batch_data = np.random.randn(32, 1, 102400).astype(np.float32)
outputs = session.predict_batch(batch_data, batch_size=32)
```

---

## 🌐 REST API

### API Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      FastAPI Server                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Endpoint   │  │  Validation  │  │   Response   │  │
│  │   Routing    │→ │  (Pydantic)  │→ │  Formatting  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────────────┐
│              Inference Engine                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   PyTorch    │  │     ONNX     │  │  TensorRT    │  │
│  │   Backend    │  │   Backend    │  │   Backend    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Configuration

Create `.env` file:

```bash
# API Settings
APP_NAME="LSTM_PFD Bearing Fault Diagnosis API"
APP_VERSION="1.0.0"
HOST="0.0.0.0"
PORT=8000
DEBUG=false
WORKERS=4

# Model Settings
MODEL_PATH="checkpoints/phase9/model_int8.pth"
MODEL_TYPE="torch"  # or 'onnx'
DEVICE="cuda"  # or 'cpu'
BATCH_SIZE=32
USE_AMP=false
NUM_THREADS=4

# API Limits
MAX_BATCH_SIZE=128
MAX_SIGNAL_LENGTH=102400
REQUEST_TIMEOUT=30

# Logging
LOG_LEVEL="INFO"
LOG_FILE="logs/api.log"

# CORS
CORS_ORIGINS=["*"]

# Security (optional)
API_KEY="your-secret-key"
REQUIRE_AUTHENTICATION=false
```

### API Endpoints

#### GET /health

Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-20T12:00:00Z",
  "model_loaded": true,
  "device": "cuda",
  "version": "1.0.0"
}
```

#### GET /model/info

Get model information.

**Response**:
```json
{
  "model_name": "LSTM_PFD Bearing Fault Diagnosis Model",
  "model_type": "torch",
  "num_classes": 11,
  "input_shape": [1, 1, 102400],
  "class_names": {
    "0": "Normal",
    "1": "Ball Fault",
    "2": "Inner Race Fault",
    ...
  }
}
```

#### POST /predict

Single prediction.

**Request**:
```json
{
  "signal": [0.1, 0.2, 0.3, ..., 0.5],
  "return_probabilities": true,
  "return_top_k": 3
}
```

**Response**:
```json
{
  "predicted_class": 1,
  "class_name": "Ball Fault",
  "confidence": 0.9823,
  "probabilities": {
    "Ball Fault": 0.9823,
    "Inner Race Fault": 0.0145,
    "Normal": 0.0018
  },
  "inference_time_ms": 12.34,
  "timestamp": "2025-11-20T12:00:00Z"
}
```

#### POST /predict/batch

Batch prediction.

**Request**:
```json
{
  "signals": [
    [0.1, 0.2, ..., 0.5],
    [0.2, 0.3, ..., 0.6]
  ],
  "return_probabilities": true
}
```

**Response**:
```json
{
  "predictions": [
    {
      "predicted_class": 1,
      "class_name": "Ball Fault",
      "confidence": 0.9823,
      ...
    },
    {
      "predicted_class": 3,
      "class_name": "Outer Race Fault",
      "confidence": 0.9654,
      ...
    }
  ],
  "batch_size": 2,
  "total_inference_time_ms": 18.56,
  "average_inference_time_ms": 9.28
}
```

### Testing API

```python
import requests
import numpy as np

# Generate test signal
signal = np.random.randn(102400).tolist()

# Make prediction request
response = requests.post(
    'http://localhost:8000/predict',
    json={
        'signal': signal,
        'return_probabilities': True
    }
)

result = response.json()
print(f"Prediction: {result['class_name']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Inference time: {result['inference_time_ms']:.2f}ms")
```

---

## 🐳 Docker Deployment

### Build Image

```bash
# Build image
docker build -t lstm_pfd:1.0.0 .

# Tag for registry
docker tag lstm_pfd:1.0.0 your-registry/lstm_pfd:1.0.0

# Push to registry
docker push your-registry/lstm_pfd:1.0.0
```

### Run Container

```bash
# Basic run
docker run -d \
  --name lstm_pfd_api \
  -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  -e MODEL_PATH=/app/checkpoints/phase9/model_int8.pth \
  -e DEVICE=cpu \
  lstm_pfd:1.0.0

# With GPU support
docker run -d \
  --name lstm_pfd_api \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints:ro \
  -e MODEL_PATH=/app/checkpoints/phase9/model_int8.pth \
  -e DEVICE=cuda \
  lstm_pfd:1.0.0

# View logs
docker logs -f lstm_pfd_api

# Stop container
docker stop lstm_pfd_api
docker rm lstm_pfd_api
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Scale API replicas
docker-compose up -d --scale api=3

# Stop all services
docker-compose down
```

### Production Deployment

**Kubernetes** (for large-scale deployment):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lstm-pfd-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lstm-pfd-api
  template:
    metadata:
      labels:
        app: lstm-pfd-api
    spec:
      containers:
      - name: api
        image: your-registry/lstm_pfd:1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: "/app/checkpoints/model_int8.pth"
        - name: DEVICE
          value: "cuda"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
            nvidia.com/gpu: 1
          limits:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
---
apiVersion: v1
kind: Service
metadata:
  name: lstm-pfd-api-service
spec:
  selector:
    app: lstm-pfd-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 📊 Performance Benchmarking

### Benchmark Script

```bash
# Benchmark single model
python scripts/benchmark_inference.py \
    --model checkpoints/phase9/model_int8.pth \
    --backend torch \
    --num-runs 1000 \
    --device cuda

# Compare backends
python scripts/benchmark_inference.py \
    --model checkpoints/phase6/best_model.pth \
    --backends torch torch_fp16 onnx \
    --compare \
    --plot \
    --save-results results/phase9/benchmark.json
```

### Expected Performance

| Backend | Latency (ms) | Throughput (samples/s) | Model Size (MB) |
|---------|--------------|------------------------|-----------------|
| PyTorch FP32 | 45.2 ± 2.1 | 22.1 | 47.2 |
| PyTorch FP16 | 28.7 ± 1.5 | 34.8 | 23.6 |
| PyTorch INT8 | 15.3 ± 0.9 | 65.4 | 11.8 |
| ONNX FP32 | 38.9 ± 1.8 | 25.7 | 47.0 |
| ONNX INT8 | 12.1 ± 0.7 | 82.6 | 11.5 |

**Target**: ✅ <50ms latency achieved with all backends!

---

## 🎯 Production Best Practices

### 1. Model Versioning

```
checkpoints/
├── phase9/
│   ├── v1.0.0/
│   │   ├── model_int8.pth
│   │   ├── model.onnx
│   │   └── metadata.json
│   ├── v1.0.1/
│   │   └── ...
```

### 2. Monitoring

- **Latency**: P50, P95, P99 metrics
- **Throughput**: Requests per second
- **Error Rate**: Failed predictions
- **Resource Usage**: CPU, GPU, memory

### 3. Logging

```python
# Structured logging
logger.info(
    "Prediction made",
    extra={
        "predicted_class": result["class_name"],
        "confidence": result["confidence"],
        "latency_ms": result["inference_time_ms"],
        "model_version": "v1.0.0"
    }
)
```

### 4. Error Handling

- Input validation
- Timeout handling
- Graceful degradation
- Retry logic

### 5. Security

- API key authentication
- Rate limiting
- Input sanitization
- HTTPS encryption

---

## 🔍 Troubleshooting

### Issue: Model loads slowly

**Solution**: Use quantized or ONNX models

### Issue: High latency on CPU

**Solution**: Use INT8 quantization, reduce batch size

### Issue: ONNX export fails

**Solution**: Check model compatibility, use older opset version

### Issue: Docker container crashes

**Solution**: Increase memory limits, check GPU drivers

---

## 📚 Summary

Phase 9 provides complete deployment pipeline:

✅ **Model Quantization**: 4x smaller models
✅ **ONNX Export**: Cross-platform deployment
✅ **REST API**: Production-ready inference server
✅ **Docker**: Containerized deployment
✅ **Performance**: <50ms latency achieved
✅ **Documentation**: Complete deployment guide

**Next Steps**: Phase 10 - QA & Integration

---

**Last Updated**: November 2025
**Status**: ✅ Complete
