# Deployment Infrastructure

> Docker, Kubernetes, and Helm deployment configurations for the LSTM_PFD bearing fault diagnosis platform.

## Overview

This directory contains all infrastructure-as-code for deploying the LSTM_PFD platform. The system is designed as a multi-service architecture with an inference API, a Dash-based analytics dashboard, async task workers, and supporting services (PostgreSQL, Redis, NGINX).

## Deployment Options

| Method             | Directory / Files                              | Description                                                   |
| ------------------ | ---------------------------------------------- | ------------------------------------------------------------- |
| **Docker Compose** | `Dockerfile`, `docker-compose.yml` (repo root) | Local / single-host full-stack deployment                     |
| **Kubernetes**     | `deploy/kubernetes/`                           | Bare Kubernetes manifests (Deployment, Service, Ingress, HPA) |
| **Helm**           | `deploy/helm/lstm-pfd/`                        | Templated Helm chart with staging/prod value overrides        |
| **Monitoring**     | `deploy/monitoring/`                           | Prometheus alert rules and Grafana dashboard                  |

## Architecture

```mermaid
graph TD
    Client --> NGINX["NGINX (port 80/443)"]
    NGINX --> API["FastAPI API (port 8000)"]
    NGINX --> Dashboard["Dash Dashboard (port 8050)"]
    Dashboard --> Postgres["PostgreSQL"]
    Dashboard --> Redis["Redis"]
    Redis --> Worker["Celery Worker"]
    Worker --> Postgres
    Flower["Flower UI (port 5555)"] --> Redis
```

## Quick Start — Docker Compose

```bash
# 1. Clone the repo and navigate to root
cd LSTM_PFD

# 2. Start all services
docker compose up -d

# 3. Verify
curl http://localhost:8000/health   # API health check
open http://localhost:8050           # Dashboard
open http://localhost:5555           # Flower task monitor (optional)
```

## Services

| Service              | Container                | Port    | Purpose                                              |
| -------------------- | ------------------------ | ------- | ---------------------------------------------------- |
| `api`                | `lstm_pfd_api`           | 8000    | FastAPI inference server (single & batch prediction) |
| `nginx`              | `lstm_pfd_nginx`         | 80, 443 | Reverse proxy & TLS termination                      |
| `dashboard`          | `lstm_pfd_dashboard`     | 8050    | Plotly Dash analytics dashboard                      |
| `celery_worker`      | `lstm_pfd_celery_worker` | —       | Async task worker (HPO, XAI, Training)               |
| `celery_flower`      | `lstm_pfd_flower`        | 5555    | Celery task monitoring UI                            |
| `dashboard_postgres` | `lstm_pfd_dashboard_db`  | 5432    | PostgreSQL database                                  |
| `redis`              | `lstm_pfd_redis`         | 6379    | Celery broker & result backend                       |

## Environment Variables

The API service accepts the following environment variables:

| Variable     | Default                           | Description                                   |
| ------------ | --------------------------------- | --------------------------------------------- |
| `DEVICE`     | `cpu`                             | Inference device (`cpu` or `cuda`)            |
| `MODEL_PATH` | `/app/checkpoints/best_model.pth` | Path to model checkpoint inside the container |
| `MODEL_TYPE` | `torch`                           | Backend: `torch`, `onnx`, or `quantized`      |
| `LOG_LEVEL`  | `INFO`                            | Logging verbosity                             |
| `WORKERS`    | `4`                               | Uvicorn worker count                          |

Dashboard services share these variables:

| Variable                          | Default                                                                       | Description             |
| --------------------------------- | ----------------------------------------------------------------------------- | ----------------------- |
| `DATABASE_URL`                    | `postgresql://lstm_user:lstm_password@dashboard_postgres:5432/lstm_dashboard` | PostgreSQL connection   |
| `REDIS_URL` / `CELERY_BROKER_URL` | `redis://redis:6379/0`                                                        | Redis / Celery broker   |
| `SECRET_KEY`                      | `development_secret_key_change_in_production`                                 | ⚠️ Change in production |
| `DEBUG`                           | `True`                                                                        | Debug mode              |

## Health Checks

| Service    | Endpoint                   | Interval                                          |
| ---------- | -------------------------- | ------------------------------------------------- |
| API        | `GET /health` on port 8000 | 30 s (timeout 10 s, 3 retries)                    |
| Dashboard  | `GET /` on port 8050       | 30 s (timeout 10 s, 3 retries, 60 s start period) |
| PostgreSQL | `pg_isready`               | 10 s                                              |
| Redis      | `redis-cli ping`           | 10 s                                              |

## Directory Structure

```
deploy/
├── helm/
│   └── lstm-pfd/
│       ├── Chart.yaml              # Chart metadata (v0.1.0, appVersion 1.0.0)
│       ├── values.yaml             # Default values
│       ├── values-staging.yaml     # Staging overrides
│       ├── values-prod.yaml        # Production overrides
│       └── templates/
│           ├── _helpers.tpl        # Template helpers
│           ├── configmap.yaml
│           ├── deployment.yaml
│           ├── hpa.yaml            # HorizontalPodAutoscaler
│           ├── ingress.yaml
│           ├── pdb.yaml            # PodDisruptionBudget
│           ├── secret.yaml
│           ├── service.yaml
│           └── serviceaccount.yaml
├── kubernetes/
│   └── deployment.yaml             # Standalone K8s manifests (Deployment + Service + Ingress + HPA)
├── monitoring/
│   ├── prometheus-alerts.yml       # Alert rules (error rate, latency, SLOs, capacity)
│   └── grafana-dashboard.json      # Pre-built Grafana dashboard
├── README.md                       # ← You are here
└── DEPLOYMENT_GUIDE.md             # Comprehensive deployment guide
```

## Dependencies

- **Requires:** `Dockerfile` and `docker-compose.yml` at repo root
- **Requires:** `packages/deployment/` for inference API code and model optimization
- **Requires:** `packages/dashboard/` for dashboard service code

## Related Documentation

- [Deployment Guide](./DEPLOYMENT_GUIDE.md) — Comprehensive deployment instructions
- [Model Optimization (packages/deployment)](../packages/deployment/README.md) — ONNX export, quantization, inference engines
