# IDB 4.2 Deployment — Consolidated Audit Report

**Domain:** Infrastructure → Deployment  
**Analyst:** IDB 4.2 Agent  
**Date:** 2026-03-15  
**Supersedes:** `IDB_4_2_DEPLOYMENT_ANALYSIS.md`, `IDB_4_2_DEPLOYMENT_BEST_PRACTICES.md`

---

## Executive Summary

This report is a **fresh, source-code-level audit** of the entire Deployment Sub-Block. Every file listed below was read in full and cross-referenced against the implementation plan, the IDB architecture document, and prior reports.

**Overall Verdict:** The deployment domain is architecturally comprehensive but **not production-ready**. The skeleton is strong — multi-stage Docker, K8s with HPA, Helm multi-env, Prometheus SLOs, ONNX/quantization pipelines — but critical gaps remain: missing `nginx.conf`, hardcoded secrets in the dashboard compose, a `sys.path` hack in the API server, deprecated FastAPI lifecycle hooks, and no rate limiting or GPU resource requests.

### Scope

| Area | Files Audited | Total LOC |
|---|---|---|
| Docker | `Dockerfile`, `docker-compose.yml`, `packages/dashboard/Dockerfile`, `packages/dashboard/docker-compose.yml` | ~370 |
| FastAPI API | `api/main.py`, `api/config.py`, `api/schemas.py`, `api/__init__.py` | ~680 |
| Model Optimization | `onnx_export.py`, `quantization.py`, `inference.py`, `model_optimization.py`, `__init__.py` | ~1,910 |
| K8s/Helm | `deployment.yaml`, Helm `values.yaml`, `values-staging.yaml`, `values-prod.yaml`, 9 templates | ~600 |
| Monitoring | `prometheus-alerts.yml`, `grafana-dashboard.json` | ~200 |
| CI/CD | `ci.yml`, `deploy.yml`, `security.yml`, `docs.yml`, `release.yml`, `ci-cd.yml` | ~750 |
| **Total** | **~30 files** | **~4,510** |

---

## Updated Domain Analysis

### What's Done Well ✅

1. **Root Dockerfile** — Proper 3-stage build (base → dependencies → application), non-root user (`appuser`), health check, `build-essential` for native deps.
2. **Root `docker-compose.yml`** — Secrets now use `${VARIABLE:?Set X in .env}` pattern (P0 fixes from Phase 1 ✅). Health checks on Postgres/Redis with `condition: service_healthy`.
3. **API `config.py`** — CORS no longer uses wildcard `["*"]` — correctly defaults to `["http://localhost:8050", "http://localhost:3000"]` and is overridable via `CORS_ORIGINS` env. Pydantic `BaseSettings` with `.env` file support.
4. **K8s `deployment.yaml`** — Secrets via `secretKeyRef` (proper), resource requests/limits defined, liveness/readiness probes, HPA with CPU+memory scaling, PVC for models.
5. **Helm chart** — Multi-env (`values.yaml`, `values-staging.yaml`, `values-prod.yaml`), PDB, ServiceAccount, Configmap, Secret, HPA, Ingress templates — all present.
6. **Prometheus alerts** — 12 rules across 3 groups (operational, SLO, capacity). 99.5% availability SLO, P95 latency SLO, model-loaded check, GPU memory alert.
7. **ONNX pipeline** — Export, validation (output comparison with tolerances), optimization (basic/extended/all passes), benchmarking — all functional.
8. **Quantization pipeline** — Dynamic (INT8), Static (with calibration), FP16, QAT preparation+conversion — comprehensive.
9. **Inference engine abstraction** — Clean ABC pattern (`BaseInferenceEngine` → `TorchInferenceEngine`, `ONNXInferenceEngine`, `OptimizedInferenceEngine` unified selector).
10. **Model optimization** — Pruning (L1 unstructured/structured, random), model stats/comparison, profiling (FLOPs via torchprofile).
11. **CI/CD workflows** — 6 workflows: CI (lint→test→integration→docker→security→docs→benchmark), deploy (tag-triggered Docker Hub push + GitHub Release), security (bandit+safety+pip-audit, weekly schedule).

### What Changed Since the Previous Report

The previous analysis (2026-01-23) raised 3 P0 issues. Here is the current status:

| Previous ID | Issue | Current Status |
|---|---|---|
| P0-1 | Hardcoded secrets in root `docker-compose.yml` | ✅ **FIXED** — now uses `${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD in .env}` |
| P0-2 | CORS allows all origins | ✅ **FIXED** — now defaults to localhost origins |
| P0-3 | Hardcoded password in Helm `values.yaml` | ✅ **FIXED** — now `postgresPassword: ""` with comment `REQUIRED: Set via --set` |

---

## Identified Issues

### P0 — Critical (Production Blockers)

| ID | Issue | Location | Evidence | Impact |
|---|---|---|---|---|
| P0-1 | **Dashboard `docker-compose.yml` still has hardcoded `POSTGRES_PASSWORD=lstm_password`** | `packages/dashboard/docker-compose.yml:9,37,56` | Literal string `lstm_password` in 3 places | Root compose was fixed, but dashboard compose was missed. Credentials exposed in git. |
| P0-2 | **`nginx.conf` does not exist** | Referenced at `docker-compose.yml:48` as `./nginx.conf:/etc/nginx/nginx.conf:ro` | `find_by_name` returned 0 results | `docker-compose up` will fail — NGINX container cannot start without this file. |
| P0-3 | **`ssl/` directory does not exist** | Referenced at `docker-compose.yml:49` as `./ssl:/etc/nginx/ssl:ro` | `find_by_name` returned 0 results | Same — NGINX bind mount fails if path is missing. |

### P1 — High Priority

| ID | Issue | Location | Evidence | Impact |
|---|---|---|---|---|
| P1-1 | **`sys.path.insert()` hack in API server** | `api/main.py:28` | `sys.path.insert(0, str(Path(__file__).parent.parent))` | Breaks when deployed from Docker root `/app`; fragile import resolution. Should use proper package install or relative imports. |
| P1-2 | **Deprecated `@app.on_event("startup"/"shutdown")`** | `api/main.py:87,124` | FastAPI deprecated these in favor of lifespan context managers | Will emit deprecation warnings; will break in a future FastAPI release. |
| P1-3 | **Dashboard Dockerfile has no non-root user** | `packages/dashboard/Dockerfile` (26 lines) | No `RUN groupadd`, no `USER`, no `HEALTHCHECK` | Runs as root inside container — security risk. No health check for orchestrator integration. |
| P1-4 | **Dashboard Dockerfile is single-stage** | `packages/dashboard/Dockerfile` | Only one `FROM` directive | Build tools (`gcc`, `g++`) remain in production image, increasing attack surface and image size. |
| P1-5 | **No resource limits in dashboard `docker-compose.yml`** | `packages/dashboard/docker-compose.yml` | No `deploy.resources` or `mem_limit` on any service | A runaway Celery worker or Dash process can consume all host resources. |
| P1-6 | **No GPU resource requests in K8s** | `deploy/kubernetes/deployment.yaml:42-48` | No `nvidia.com/gpu` in resource requests | GPU-accelerated inference won't get scheduled to GPU nodes. |
| P1-7 | **Redis auth disabled in Helm** | `deploy/helm/lstm-pfd/values.yaml:81` | `auth.enabled: false` | Production Redis has no authentication — any pod on the cluster network can read/write. |
| P1-8 | **No rate limiting in FastAPI API** | `api/main.py` (entire file) | No SlowAPI or middleware import | No protection against request flooding, DoS, or scraping abuse. |
| P1-9 | **Static quantization stub in ONNX export** | `onnx_export.py:474-476` | `logger.warning("Static quantization not yet implemented for ONNX")` then returns `temp_path` | Caller expects a quantized model path but gets the raw FP32 file. This is a silent correctness bug. |
| P1-10 | **`benchmark_inference()` skips PyTorch backend** | `inference.py:474-479` | Logs a warning and `continue`s for `torch` backend | The multi-backend benchmark function cannot actually benchmark the most common (PyTorch) backend. |

### P2 — Medium Priority

| ID | Issue | Location | Evidence | Impact |
|---|---|---|---|---|
| P2-1 | **Placeholder domains in K8s Ingress and Helm** | `deployment.yaml:100,103`, `values.yaml:33,40` | `api.lstm-pfd.example.com` | Must be replaced before any real deployment. |
| P2-2 | **No K8s network policies** | `deploy/kubernetes/` | No `NetworkPolicy` resource defined | No pod-to-pod isolation — all pods can talk to all other pods on the cluster. |
| P2-3 | **ONNX optimizer uses deprecated `onnx.optimizer`** | `onnx_export.py:225,236,250` | `from onnx import optimizer as onnx_optimizer` | The `onnx.optimizer` module has been deprecated since ONNX 1.14; `onnxoptimizer` is now a separate package. This will fail on modern ONNX versions. |
| P2-4 | **`torch.load()` missing `weights_only=True`** | `quantization.py:456`, `inference.py:106` | `torch.load(checkpoint_path, map_location='cpu')` with no `weights_only` | Security: allows arbitrary code execution via pickle. PyTorch 2.0+ emits a warning; will default to `weights_only=True` in a future release, breaking current code. |
| P2-5 | **`optimize_for_deployment()` ignores caller's `pruning_amount`** | `model_optimization.py:183-189` | Function accepts `pruning_amount` parameter, then overwrites it based on `optimization_level` | Misleading API — the parameter is a no-op. |
| P2-6 | **`fuse_model_layers()` is a no-op stub** | `model_optimization.py:115-154` | Logs a warning and returns the model unchanged | Code and docstring promise layer fusion but it does nothing. |
| P2-7 | **CI/CD workflows use deprecated Actions** | `.github/workflows/ci.yml:18,22,204`, `deploy.yml:17,23,57` | `actions/checkout@v3`, `actions/setup-python@v4`, `actions/create-release@v1` | `create-release@v1` is archived and no longer maintained. `checkout@v3` and `setup-python@v4` should be updated to v4/v5. |
| P2-8 | **No `.env.example` file** | Root of repo | Root compose uses `:?Set X in .env` error pattern but no template exists | Developer onboarding friction — no reference for required environment variables. |

### P3 — Low Priority / Cosmetic

| ID | Issue | Location | Impact |
|---|---|---|---|
| P3-1 | Version `"3.8"` in compose files | Both `docker-compose.yml` files | Docker Compose v2+ ignores the `version` key; can be removed for cleanliness. |
| P3-2 | `Dockerfile` hardcodes `python:3.10-slim` | Root `Dockerfile:5` | Should be parameterized via `ARG PYTHON_VERSION=3.10` for CI matrix builds. |
| P3-3 | Duplicate benchmark code between `TorchInferenceEngine.benchmark()` and `ONNXInferenceEngine.benchmark()` | `inference.py:194-231` and `inference.py:336-364` | Nearly identical code; should be pulled into `BaseInferenceEngine`. |
| P3-4 | `model_optimization.py` imports `SIGNAL_LENGTH` | `model_optimization.py:22` | Used only in default args of `compare_models()` and `export_model_summary()`. Tight coupling to a specific signal length. |
| P3-5 | `logging.basicConfig()` repeated in every file | `onnx_export.py:22`, `quantization.py:23`, `inference.py:25`, `model_optimization.py:24` | Each module resets the root logger configuration, which can stomp on other modules' settings when imported. |
| P3-6 | Runbook URL in Prometheus alerts points to `example.com` | `prometheus-alerts.yml:24` | Dead link if not updated. |

---

## Best Practices — Updated Reference

### Docker

| Practice | Status | Notes |
|---|---|---|
| Multi-stage builds | ✅ Root, ❌ Dashboard | Dashboard needs conversion to multi-stage |
| Non-root user | ✅ Root, ❌ Dashboard | Dashboard runs as root |
| Health checks | ✅ Root, ❌ Dashboard Dockerfile | Dashboard compose has a health check, but Dockerfile itself doesn't |
| `.dockerignore` | ⚠️ Not verified | Should exclude `*.pyc`, `__pycache__`, `.git`, `tests/`, `docs/` |
| Parameterized base image | ❌ | Use `ARG PYTHON_VERSION` |
| Secrets via env vars | ✅ Root compose, ❌ Dashboard compose | Dashboard compose still hardcodes |

### Kubernetes / Helm

| Practice | Status | Notes |
|---|---|---|
| Secret management | ✅ K8s secretKeyRef | Helm values use `""` with comments |
| Resource limits | ✅ | requests/limits defined in deployment.yaml and Helm |
| HPA autoscaling | ✅ | CPU + memory targets |
| PDB | ✅ | `minAvailable: 1` |
| Network policies | ❌ | Must be added |
| GPU resource requests | ❌ | No `nvidia.com/gpu` |
| Configurable domains | ❌ | Hardcoded `example.com` |

### FastAPI API

| Practice | Status | Notes |
|---|---|---|
| Pydantic validation | ✅ | Schemas are well-defined |
| CORS config | ✅ | Fixed, configurable via env |
| Health endpoint | ✅ | `/health` returns model status |
| Rate limiting | ❌ | No middleware |
| FastAPI lifespan | ❌ | Uses deprecated `@app.on_event` |
| API key auth | ✅ | Optional, via header |
| Error handling | ✅ | Custom exception handlers |

### Model Optimization

| Practice | Status | Notes |
|---|---|---|
| ONNX export+validation | ✅ | Output comparison with tolerances |
| Dynamic quantization | ✅ | INT8 with benchmarking |
| Static quantization | ⚠️ PyTorch yes, ONNX stub | ONNX path silently returns FP32 |
| QAT | ✅ | Prepare + convert workflow |
| Inference engine abstraction | ✅ | Clean ABC pattern |
| Model pruning | ✅ | L1/structured/random |
| `weights_only=True` for `torch.load` | ❌ | Security vulnerability |

### CI/CD

| Practice | Status | Notes |
|---|---|---|
| Multi-OS test matrix | ✅ | Ubuntu + Windows, Python 3.10 + 3.11 |
| Security scanning | ✅ | bandit + safety + pip-audit |
| Docker build in CI | ✅ | Build verification |
| Tag-triggered deploy | ✅ | Version-tagged Docker Hub push |
| Action versions | ⚠️ | Several using v1/v3 (deprecated) |

---

## Prioritized Actions

### 🔴 Phase 6 Pre-requisites (Must-Fix Before `docker-compose up`)

| Priority | Action | Est. Effort | Plan Step |
|---|---|---|---|
| **P0** | Create `nginx.conf` with reverse proxy config for API (:8000) + Dashboard (:8050) | 30 min | **6.1** |
| **P0** | Create `ssl/` directory + self-signed cert generation script | 30 min | **6.2** |
| **P0** | Fix dashboard `docker-compose.yml` — replace hardcoded `lstm_password` with env-var refs | 15 min | *NEW 6.0a* |

### 🟡 Phase 6 Core Work

| Priority | Action | Est. Effort | Plan Step |
|---|---|---|---|
| **P1** | Remove `sys.path.insert()` from `api/main.py` — install packages properly or use relative imports | 30 min | *NEW 6.0b* |
| **P1** | Replace `@app.on_event` with FastAPI lifespan context manager | 30 min | **6.12** |
| **P1** | Harden dashboard `Dockerfile` (non-root user, health check, multi-stage) | 1 hr | **6.3** |
| **P1** | Add resource limits to dashboard `docker-compose.yml` | 15 min | **6.4** |
| **P1** | Add GPU resource requests to K8s deployment | 15 min | **6.6** |
| **P1** | Enable Redis auth in Helm values | 15 min | **1.4** |
| **P1** | Add rate limiting middleware (SlowAPI) to FastAPI | 1 hr | **6.13** |
| **P1** | Fix ONNX static quantization stub — either implement or raise `NotImplementedError` | 30 min | *NEW 6.5a* |
| **P1** | Fix `benchmark_inference()` to support PyTorch backend | 1 hr | *NEW 6.5b* |

### 🟢 Phase 6 Polish

| Priority | Action | Est. Effort | Plan Step |
|---|---|---|---|
| **P2** | Replace placeholder domains with configurable Helm values | 15 min | **6.8** |
| **P2** | Add K8s network policies | 30 min | **6.7** |
| **P2** | Add `weights_only=True` to all `torch.load()` calls | 15 min | *NEW 6.5c* |
| **P2** | Fix deprecated `onnx.optimizer` usage → `onnxoptimizer` package | 30 min | *NEW 6.5d* |
| **P2** | Fix `optimize_for_deployment()` parameter shadowing | 10 min | *NEW 6.5e* |
| **P2** | Update CI/CD workflow Action versions (v3→v4, create-release→softprops/action-gh-release) | 30 min | *NEW 6.16a* |
| **P2** | Create `.env.example` file with all required variables | 15 min | *NEW 6.0c* |

### 📋 Deferred (Phase 7+)

| Priority | Action | Plan Step |
|---|---|---|
| P3 | Remove `version: "3.8"` from compose files | Phase 8 cleanup |
| P3 | Parameterize Dockerfile base image with `ARG` | Phase 6 optional |
| P3 | Deduplicate `benchmark()` methods into `BaseInferenceEngine` | Phase 2 refactoring |
| P3 | Extract `logging.basicConfig()` to a shared setup | Phase 2 refactoring |
| P3 | Update Prometheus runbook URLs | Phase 8 docs |

---

## File Inventory (Verified 2026-03-15)

| File | LOC | Purpose | Issues |
|---|---|---|---|
| `Dockerfile` | 64 | Multi-stage API container | ✅ Clean |
| `docker-compose.yml` | 197 | Full-stack orchestration | P0-2, P0-3 (missing nginx.conf, ssl/) |
| `packages/dashboard/Dockerfile` | 26 | Dashboard container | P1-3, P1-4 |
| `packages/dashboard/docker-compose.yml` | 84 | Dashboard stack | P0-1, P1-5 |
| `packages/deployment/api/main.py` | 379 | FastAPI inference server | P1-1, P1-2, P1-8 |
| `packages/deployment/api/config.py` | 64 | API configuration | ✅ Clean |
| `packages/deployment/api/schemas.py` | 173 | Pydantic request/response | ✅ Clean |
| `packages/deployment/optimization/onnx_export.py` | 483 | ONNX export utilities | P1-9, P2-3 |
| `packages/deployment/optimization/quantization.py` | 468 | Model quantization | P2-4 |
| `packages/deployment/optimization/inference.py` | 525 | Inference engines | P1-10, P2-4 |
| `packages/deployment/optimization/model_optimization.py` | 433 | Pruning, profiling | P2-5, P2-6 |
| `deploy/kubernetes/deployment.yaml` | 139 | K8s manifests | P1-6, P2-1, P2-2 |
| `deploy/helm/lstm-pfd/values.yaml` | 140 | Helm defaults | P1-7, P2-1 |
| `deploy/monitoring/prometheus-alerts.yml` | 173 | Alert rules | P3-6 |
| `.github/workflows/ci.yml` | 241 | CI pipeline | P2-7 |
| `.github/workflows/deploy.yml` | 80 | Deploy pipeline | P2-7 |
| `.github/workflows/security.yml` | 77 | Security scans | ✅ Uses v4/v5 |

---

_This report fully supersedes `IDB_4_2_DEPLOYMENT_ANALYSIS.md` and `IDB_4_2_DEPLOYMENT_BEST_PRACTICES.md`._
