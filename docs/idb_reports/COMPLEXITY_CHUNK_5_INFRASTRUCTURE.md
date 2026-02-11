# Chunk 5: Infrastructure (Domain 4)

**Verdict: 🟡 Premature Enterprise Hardening**

---

## The Numbers

| Sub-module                          | Files                              | Assessment                                  |
| ----------------------------------- | ---------------------------------- | ------------------------------------------- |
| Database (migrations)               | 10 SQL files                       | 🟡 Half are for enterprise features         |
| Deployment (`deploy/`)              | 15 files (Helm + K8s + monitoring) | 🔴 Full K8s platform for a single-user app  |
| Deployment (`packages/deployment/`) | 10 .py                             | 🟢 Inference API + optimization — justified |
| Testing (`tests/`)                  | ~30 files across 6 dirs            | 🟡 Load/stress tests are overkill           |
| Configuration (`config/`)           | 5 .py                              | 🟢 **Clean — best module in the project**   |

---

## Problem 1: 🔴 Kubernetes Infrastructure for a Research Tool

The `deploy/` directory contains a **full production Kubernetes platform**:

### Helm Chart (`deploy/helm/lstm-pfd/`)

| File                                                       | What It Does                  |
| ---------------------------------------------------------- | ----------------------------- |
| `deployment.yaml`                                          | Multi-container pod spec      |
| `service.yaml`                                             | ClusterIP service             |
| `ingress.yaml`                                             | External ingress rules        |
| `hpa.yaml`                                                 | **Horizontal Pod Autoscaler** |
| `pdb.yaml`                                                 | **Pod Disruption Budget**     |
| `serviceaccount.yaml`                                      | RBAC service account          |
| `configmap.yaml`                                           | Environment config            |
| `secret.yaml`                                              | Secret management             |
| `_helpers.tpl`                                             | Template helpers              |
| `values.yaml` + `values-staging.yaml` + `values-prod.yaml` | 3 environment configs         |

Plus **monitoring**:
| File | Size |
|------|------|
| `grafana-dashboard.json` | 15 KB |
| `prometheus-alerts.yml` | 5 KB |

> [!WARNING]
> **HPA + PDB + Ingress + Prometheus + Grafana** is infrastructure for a team running 24/7 production workloads. If this project runs on a single machine or a dev laptop, this is ~15 files of unused infrastructure.

### Duplicate K8s Manifest

`deploy/kubernetes/deployment.yaml` (3 KB) duplicates what the Helm chart generates. It's a standalone K8s manifest that predates the Helm chart.

**Verdict:**

- **If running on K8s in production:** Keep the Helm chart, delete `deploy/kubernetes/` (Helm supersedes it)
- **If running locally:** Consider removing the entire `deploy/` directory and the root `Dockerfile`. Keep only `docker-compose.yml` for local development

---

## Problem 2: 🟡 Database Migrations — Enterprise Half

10 migrations, roughly split:

| Migration                               | Purpose                                                | Needed for Research? |
| --------------------------------------- | ------------------------------------------------------ | -------------------- |
| `007_add_tags_and_search.sql`           | Tags, saved searches                                   | ✅ Useful            |
| `008_add_dataset_generation_import.sql` | Dataset tracking                                       | ✅ Core              |
| `009_add_xai_indexes.sql`               | XAI query indexes                                      | ✅ Core              |
| `001_add_api_keys.sql`                  | API key management                                     | 🔴 Enterprise        |
| `002_add_notification_preferences.sql`  | Notification settings                                  | 🔴 Enterprise        |
| `003_add_email_logs.sql`                | Email audit logs                                       | 🔴 Enterprise        |
| `004_add_email_digest_queue.sql`        | Email digest queue                                     | 🔴 Enterprise        |
| `005_add_webhook_configurations.sql`    | Webhook storage                                        | 🔴 Enterprise        |
| `006_add_webhook_logs.sql`              | Webhook audit logs                                     | 🔴 Enterprise        |
| `010_add_2fa_sessions.sql` (**11 KB**)  | Two-factor auth, sessions, login history, backup codes | 🔴 Enterprise        |

> Migration `010` alone is **11 KB of SQL** adding 2FA, backup codes, session tracking, and login history. This is enterprise security infrastructure.

**Verdict:** These migrations are **coupled to the enterprise dashboard features** analyzed in Chunk 3. If you remove those features, these migrations go too.

---

## Problem 3: 🟡 Testing — Load/Stress Tests for a Research Project

| File                                  | Size      | What It Does                                                                              |
| ------------------------------------- | --------- | ----------------------------------------------------------------------------------------- |
| `load_tests.py`                       | **36 KB** | Full load testing framework with concurrent users, response time SLAs, throughput targets |
| `stress_tests.py`                     | **17 KB** | Stress testing with memory pressure, CPU saturation, recovery testing                     |
| `test_data_generation.py`             | **29 KB** | Comprehensive data generation tests                                                       |
| `tests/benchmarks/benchmark_suite.py` | 12 KB     | Performance benchmark suite                                                               |

**53 KB of load/stress testing.** That's valuable for production systems handling concurrent requests. For a research dashboard used by 1-3 people, it's overkill.

**Additional issue — fragmented test locations:**

- `tests/` (root) — 30 files
- `packages/dashboard/tests/` — 3 files
- `tests/unit/` — 5 files
- `tests/integration/` — 3 files
- `tests/benchmarks/` — 2 files
- `tests/utilities/` — 2 files

**Verdict:**

- **KEEP** unit/integration tests — valuable regardless of scale
- **MOVE** `packages/dashboard/tests/` → `tests/dashboard/` (pick one convention)
- **FLAG** load/stress tests as deferrable — keep if going to production, archive if not

---

## Problem 4: 🟢 Configuration — Actually Well-Designed

| File                   | Size  | What It Does                            |
| ---------------------- | ----- | --------------------------------------- |
| `base_config.py`       | 5 KB  | Base configuration with env var support |
| `data_config.py`       | 12 KB | Data/signal generation params           |
| `model_config.py`      | 11 KB | Model architecture params               |
| `experiment_config.py` | 5 KB  | Experiment settings                     |
| `training_config.py`   | 9 KB  | Training hyperparams                    |

Uses Python `dataclasses` with `@dataclass`, clear hierarchy, typed fields. **This is the best-designed module in the entire project.** No duplication, clean separation of concerns.

**Verdict:** **KEEP as-is.** The only suggestion (from Chunk 1) is to move from `config/` at root to `packages/core/config/`.

---

## Problem 5: 🟢 `packages/deployment/` — Justified

| File                                 | Size  | Purpose                          |
| ------------------------------------ | ----- | -------------------------------- |
| `api/main.py`                        | 10 KB | FastAPI inference server         |
| `api/schemas.py`                     | 4 KB  | Pydantic request/response models |
| `optimization/inference.py`          | 16 KB | Optimized inference pipeline     |
| `optimization/onnx_export.py`        | 14 KB | ONNX model export                |
| `optimization/quantization.py`       | 13 KB | INT8/FP16 quantization           |
| `optimization/model_optimization.py` | 12 KB | General optimization utilities   |

This is a legitimate deployment/inference package. The ONNX export and quantization tooling is standard for ML model deployment.

**Verdict:** **KEEP.** Well-scoped and useful.

---

## Summary Scorecard

| Action                                                 | Impact            |
| ------------------------------------------------------ | ----------------- |
| Remove `deploy/kubernetes/` (superseded by Helm)       | -1 file           |
| Consider removing `deploy/` if running locally only    | -15 files         |
| Enterprise migrations removed with enterprise features | -6 SQL files      |
| Consolidate test locations                             | Cleaner structure |
| Archive load/stress tests if not needed                | -2 files, -53 KB  |

> [!IMPORTANT]
> The infrastructure is **premature optimization for scale that hasn't arrived**. Helm + HPA + PDB + Prometheus + load testing is what you add when you have production users, not before.

---

_Next: Chunk 6 — Research & Science (Domain 5) + Scripts_
