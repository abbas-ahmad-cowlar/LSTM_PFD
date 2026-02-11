# Deployment Guide

> Step-by-step instructions for deploying LSTM_PFD in local, staging, and production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development with Docker](#local-development-with-docker)
3. [Production Deployment — Kubernetes / Helm](#production-deployment--kubernetes--helm)
4. [Environment Configuration](#environment-configuration)
5. [Monitoring and Health Checks](#monitoring-and-health-checks)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Tool           | Minimum Version | Purpose                                     |
| -------------- | --------------- | ------------------------------------------- |
| Docker         | 24+             | Container runtime                           |
| Docker Compose | v2+             | Multi-service orchestration                 |
| kubectl        | 1.26+           | Kubernetes CLI (production)                 |
| Helm           | 3.12+           | Chart-based deployment (production)         |
| Python         | 3.10            | Local development (matches Dockerfile base) |

Required files at repo root:

- `Dockerfile` — Multi-stage build (base → dependencies → application)
- `docker-compose.yml` — Full-stack service definitions
- `requirements.txt` — Core Python dependencies
- `requirements-deployment.txt` — Deployment-specific dependencies (FastAPI, uvicorn, etc.)

---

## Local Development with Docker

### 1. Build and Start

```bash
# Build all images and start in detached mode
docker compose up -d --build

# Watch logs
docker compose logs -f api
```

### 2. Verify Services

```bash
# API health
curl -s http://localhost:8000/health | python -m json.tool
# Expected: {"status": "healthy", "model_loaded": true, "device": "cpu", ...}

# API docs (Swagger)
open http://localhost:8000/docs

# Dashboard
open http://localhost:8050

# Flower (Celery task monitor)
open http://localhost:5555
```

### 3. Test Inference

```bash
# Single prediction (replace SIGNAL with actual values)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"signal": [0.1, 0.2, ...], "return_probabilities": true}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"signals": [[0.1, 0.2, ...], [0.3, 0.4, ...]]}'
```

### 4. Stop Services

```bash
docker compose down            # Stop and remove containers
docker compose down -v         # Also remove named volumes (⚠️ deletes DB data)
```

### Dockerfile Internals

The `Dockerfile` uses a three-stage build for minimal image size:

| Stage          | Base               | Purpose                                                     |
| -------------- | ------------------ | ----------------------------------------------------------- |
| `base`         | `python:3.10-slim` | System deps (`build-essential`, `curl`)                     |
| `dependencies` | `base`             | Installs `requirements.txt` + `requirements-deployment.txt` |
| `application`  | `base`             | Non-root user, copies deps from stage 2, copies source code |

Key details:

- Runs as non-root user `appuser:appgroup`
- Exposes port **8000**
- Built-in `HEALTHCHECK` on `/health` (30 s interval, 10 s timeout)
- Entrypoint: `uvicorn api.main:app --host 0.0.0.0 --port 8000`
- Creates `/app/logs`, `/app/checkpoints`, `/app/models` directories

---

## Production Deployment — Kubernetes / Helm

### Option A: Helm Chart (Recommended)

The Helm chart resides at `deploy/helm/lstm-pfd/`.

```bash
# Add dependency repos
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

# Install with production values
helm install lstm-pfd deploy/helm/lstm-pfd/ \
  -f deploy/helm/lstm-pfd/values-prod.yaml \
  --set secrets.jwtSecret="YOUR_JWT_SECRET" \
  --set secrets.secretKey="YOUR_SECRET_KEY" \
  --set secrets.dbPassword="YOUR_DB_PASSWORD" \
  --namespace lstm-pfd --create-namespace
```

#### Helm Chart Structure

| File                  | Purpose                                                                 |
| --------------------- | ----------------------------------------------------------------------- |
| `Chart.yaml`          | Chart metadata, dependency declarations (Redis, PostgreSQL via Bitnami) |
| `values.yaml`         | Default configuration                                                   |
| `values-staging.yaml` | Staging environment overrides                                           |
| `values-prod.yaml`    | Production environment overrides                                        |

#### Key Helm Values

```yaml
replicaCount: 3 # API replicas

image:
  repository: lstm-pfd
  tag: "latest"

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

worker:
  replicas: 2 # Celery workers
  resources:
    limits:
      cpu: 4000m
      memory: 8Gi

pdb:
  enabled: true
  minAvailable: 1 # PodDisruptionBudget

redis:
  enabled: true # Bitnami Redis subchart

postgresql:
  enabled: true # Bitnami PostgreSQL subchart
  primary:
    persistence:
      size: 10Gi

persistence:
  enabled: true
  size: 10Gi # Model storage PVC
```

#### Upgrade / Rollback

```bash
# Upgrade release
helm upgrade lstm-pfd deploy/helm/lstm-pfd/ \
  -f deploy/helm/lstm-pfd/values-prod.yaml

# Rollback to previous revision
helm rollback lstm-pfd 1
```

### Option B: Bare Kubernetes Manifests

For clusters without Helm, use the standalone manifests:

```bash
# Apply all resources
kubectl apply -f deploy/kubernetes/deployment.yaml -n lstm-pfd
```

This creates:

- **Deployment** — 3 replicas, resource limits (500m–2000m CPU, 1–4 Gi memory), liveness/readiness probes
- **Service** — ClusterIP on port 80 → 8000
- **Ingress** — NGINX ingress with TLS (cert-manager), host: `api.lstm-pfd.example.com`
- **HorizontalPodAutoscaler** — 2–10 replicas, CPU 70% / Memory 80% targets

The K8s manifests pull secrets from a `lstm-pfd-secrets` Secret and mount a `lstm-pfd-models` PVC for model storage.

---

## Environment Configuration

### API Service (`packages/deployment/api/config.py`)

Configuration is managed via `pydantic_settings.BaseSettings`, which reads from environment variables and `.env` file:

| Variable                 | Type | Default                                | Description                     |
| ------------------------ | ---- | -------------------------------------- | ------------------------------- |
| `APP_NAME`               | str  | `LSTM_PFD Bearing Fault Diagnosis API` | API title                       |
| `APP_VERSION`            | str  | `1.0.0`                                | Shown in `/health`              |
| `HOST`                   | str  | `0.0.0.0`                              | Bind address                    |
| `PORT`                   | int  | `8000`                                 | Bind port                       |
| `DEBUG`                  | bool | `False`                                | Debug mode                      |
| `WORKERS`                | int  | `4`                                    | Uvicorn workers                 |
| `MODEL_PATH`             | str  | `checkpoints/best_model.pth`           | Model checkpoint                |
| `MODEL_TYPE`             | str  | `torch`                                | `torch`, `onnx`, or `quantized` |
| `DEVICE`                 | str  | `cuda`                                 | `cuda` or `cpu`                 |
| `BATCH_SIZE`             | int  | `32`                                   | Default batch size              |
| `USE_AMP`                | bool | `False`                                | Automatic mixed precision       |
| `NUM_THREADS`            | int  | `4`                                    | CPU inference threads           |
| `MAX_BATCH_SIZE`         | int  | `128`                                  | Max batch size per request      |
| `REQUEST_TIMEOUT`        | int  | `30`                                   | Request timeout (seconds)       |
| `LOG_LEVEL`              | str  | `INFO`                                 | Python logging level            |
| `LOG_FILE`               | str  | `logs/api.log`                         | Log file path                   |
| `CORS_ORIGINS`           | list | `["*"]`                                | Allowed CORS origins            |
| `API_KEY`                | str  | `None`                                 | API key (if auth enabled)       |
| `REQUIRE_AUTHENTICATION` | bool | `False`                                | Enable API key auth             |

### Security Notes

> ⚠️ **Production checklist:**
>
> - Set `SECRET_KEY` to a cryptographically random value
> - Set `REQUIRE_AUTHENTICATION=True` and configure `API_KEY`
> - Change default PostgreSQL password (`lstm_password`)
> - Restrict `CORS_ORIGINS` to your domain(s)
> - Use TLS (the Helm ingress and NGINX configs support this)

---

## Monitoring and Health Checks

### Prometheus Alerts (`deploy/monitoring/prometheus-alerts.yml`)

The alert rules are organized into three groups:

#### `lstm-pfd-alerts` — Operational Alerts

| Alert                  | Condition                       | Severity | For    |
| ---------------------- | ------------------------------- | -------- | ------ |
| `HighErrorRate`        | Error rate > 5%                 | critical | 5 min  |
| `HighLatency`          | P95 latency > 2 s               | warning  | 5 min  |
| `HighInferenceLatency` | P95 inference > 1 s (per model) | warning  | 5 min  |
| `ModelNotLoaded`       | Model loaded metric = 0         | critical | 2 min  |
| `HighCPUUsage`         | CPU > 90%                       | warning  | 10 min |
| `HighMemoryUsage`      | Memory > 90%                    | warning  | 5 min  |
| `CriticalMemoryUsage`  | Memory > 95%                    | critical | 2 min  |
| `NoRequests`           | 0 requests/5 min                | warning  | 15 min |
| `HighGPUMemory`        | GPU memory > 10 GB              | warning  | 5 min  |

#### `lstm-pfd-slo` — SLO Alerts

| Alert                 | SLO          | Threshold             |
| --------------------- | ------------ | --------------------- |
| `SLOViolation`        | Availability | < 99.5% over 1 h      |
| `LatencySLOViolation` | Latency      | P95 > 500 ms over 1 h |

#### `lstm-pfd-capacity` — Capacity Planning

| Alert                  | Condition                           |
| ---------------------- | ----------------------------------- |
| `ApproachingRateLimit` | Request rate > 40/s for 5 min       |
| `HighInferenceVolume`  | Inference rate > 500/min for 10 min |

### Grafana Dashboard

A pre-built dashboard is available at `deploy/monitoring/grafana-dashboard.json`. Import it into your Grafana instance to visualize request rates, latency histograms, error rates, and system resource metrics.

---

## Troubleshooting

### Container won't start

```bash
# Check logs
docker compose logs api

# Common issues:
# - Missing model checkpoint → ensure checkpoints/best_model.pth exists
# - Port conflict → change the port mapping in docker-compose.yml
# - GPU not found → set DEVICE=cpu in environment
```

### Model not loading

```bash
# Verify model file exists and is accessible
docker compose exec api ls -la /app/checkpoints/

# Check if the model type matches the file format
# MODEL_TYPE=torch  → expects .pth or .pt file
# MODEL_TYPE=onnx   → expects .onnx file
```

### Health check failing

```bash
# Manual health check
docker compose exec api curl -f http://localhost:8000/health

# If model_loaded is false, check:
# 1. MODEL_PATH points to a valid file
# 2. DEVICE is set correctly (cpu vs cuda)
# 3. Sufficient memory for the model
```

### PostgreSQL connection refused

```bash
# Check if postgres is healthy
docker compose ps dashboard_postgres

# Verify credentials match between services
# DATABASE_URL in dashboard/celery must match POSTGRES_* env vars
```

### Celery worker not processing tasks

```bash
# Check worker logs
docker compose logs celery_worker

# Verify Redis is reachable
docker compose exec redis redis-cli ping
# Expected: PONG

# Check Flower for task visibility
open http://localhost:5555
```

---

## Performance

> ⚠️ **Results pending.** Performance metrics below will be populated after experiments are run on the current codebase.

| Metric                     | Value       |
| -------------------------- | ----------- |
| API Cold Start Time        | `[PENDING]` |
| Single Inference Latency   | `[PENDING]` |
| Batch Inference Throughput | `[PENDING]` |
| Docker Image Size          | `[PENDING]` |
| Max Concurrent Requests    | `[PENDING]` |

## Related Documentation

- [Deployment README](./README.md) — Overview and directory structure
- [Model Optimization](../packages/deployment/README.md) — ONNX export, quantization, inference engines
