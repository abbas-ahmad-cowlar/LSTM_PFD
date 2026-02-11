# IDB 4.2 Deployment Best Practices

**Domain:** Infrastructure  
**Date:** 2026-01-23

---

## 1. Dockerfile Patterns

### Multi-Stage Builds

```dockerfile
# Stage 1: Base with environment setup
FROM python:3.10-slim as base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# Stage 2: Dependencies (cached layer)
FROM base as dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Stage 3: Application (smallest possible)
FROM base as application
COPY --from=dependencies /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY . /app
```

### Non-Root User Security

```dockerfile
# Create dedicated user/group
RUN groupadd -r appgroup && useradd -r -g appgroup appuser
RUN chown -R appuser:appgroup /app
USER appuser
```

### Container Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

---

## 2. Container Orchestration Conventions

### Docker Compose Service Dependencies

```yaml
services:
  app:
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
```

### Health Checks for Dependencies

```yaml
postgres:
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U user -d database"]
    interval: 10s
    timeout: 5s
    retries: 5

redis:
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
    interval: 10s
    timeout: 5s
    retries: 5
```

### Volume Mount Patterns

```yaml
volumes:
  - ./checkpoints:/app/checkpoints:ro # Read-only for inference
  - ./logs:/app/logs # Read-write for output
  - postgres_data:/var/lib/postgresql/data # Named volume for persistence
```

### Network Isolation

```yaml
networks:
  app_network:
    driver: bridge
```

---

## 3. Model Optimization Patterns

### ONNX Export with Validation

```python
from deployment.optimization.onnx_export import export_to_onnx, validate_onnx_export

# Export with dynamic batch size
config = ONNXExportConfig(
    opset_version=14,
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
export_to_onnx(model, dummy_input, 'model.onnx', config)

# Validate output consistency
validate_onnx_export('model.onnx', model, test_input, rtol=1e-3, atol=1e-5)
```

### Quantization Pipeline

```python
from deployment.optimization.quantization import (
    quantize_model_dynamic,
    compare_model_sizes,
    benchmark_quantized_model
)

# Dynamic quantization (no calibration needed)
quantized = quantize_model_dynamic(model, dtype=torch.qint8)

# Verify compression and speedup
stats = compare_model_sizes(model, quantized)
perf = benchmark_quantized_model(model, quantized, test_input)
```

### Inference Engine Abstraction

```python
from abc import ABC, abstractmethod

class BaseInferenceEngine(ABC):
    @abstractmethod
    def load_model(self, path: str): pass

    @abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray: pass

    @abstractmethod
    def predict_batch(self, data: np.ndarray, batch_size: int) -> np.ndarray: pass
```

### Unified Engine Pattern

```python
engine = OptimizedInferenceEngine(config)
engine.load_model('model.onnx')  # Auto-detects backend
predictions = engine.predict(data)
```

---

## 4. Health Check Conventions

### FastAPI Health Endpoint

```python
from pydantic import BaseModel

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if inference_engine else "unhealthy",
        model_loaded=inference_engine is not None,
        device=settings.device,
        version=settings.app_version
    )
```

### Kubernetes Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

### HPA Autoscaling

```yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

---

## 5. CI/CD Integration Patterns

### Helm Multi-Environment Values

```
deploy/helm/lstm-pfd/
├── values.yaml          # Defaults
├── values-staging.yaml  # Staging overrides
└── values-prod.yaml     # Production overrides
```

### Prometheus Alert Rules (SLO-Based)

```yaml
# 99.5% Availability SLO
- alert: SLOViolation
  expr: |
    sum(rate(requests_total{status="success"}[1h])) 
    / sum(rate(requests_total[1h])) < 0.995
  labels:
    severity: critical
    slo: availability
```

### Resource Limits

```yaml
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"
```

---

## 6. Configuration Patterns

### Pydantic Settings

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_path: str = "checkpoints/best_model.pth"
    device: str = "cuda"
    batch_size: int = 32

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

### Dataclass Configs

```python
from dataclasses import dataclass

@dataclass
class InferenceConfig:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 32
    use_amp: bool = False
    num_threads: int = 4
```

---

## Quick Reference Checklist

| Category         | Best Practice                                               |
| ---------------- | ----------------------------------------------------------- |
| **Dockerfile**   | Multi-stage builds, non-root user, health checks            |
| **Compose**      | Service health conditions, named volumes, network isolation |
| **Optimization** | ONNX export + validation, quantization benchmarking         |
| **Health**       | `/health` endpoint, K8s liveness/readiness probes           |
| **CI/CD**        | Helm multi-env values, SLO-based alerts                     |
| **Config**       | Pydantic settings with `.env`, dataclass configs            |

---

_Extracted from IDB 4.2 Deployment Sub-Block_
