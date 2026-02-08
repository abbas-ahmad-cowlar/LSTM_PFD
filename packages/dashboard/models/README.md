# ORM Models

> SQLAlchemy ORM models for the PFD Fault Diagnosis Dashboard.

## Overview

This package contains all database model definitions for the dashboard application. Every model inherits from `BaseModel` (defined in `base.py`), which provides `id`, `created_at`, and `updated_at` fields plus a generic `to_dict()` method. The one exception is `APIKey` and `APIUsage`, which inherit directly from `Base` (the declarative base) and define their own timestamp columns.

Models are organized by domain and are all re-exported from `__init__.py` for convenient imports.

## Base Model

```python
from models.base import BaseModel

class BaseModel(Base):
    __abstract__ = True

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    def to_dict(self):
        """Convert model to dictionary."""
        ...
```

## Model Catalog

### Core ML Domain

| Model Class   | Table           | File              | Description                                                                             |
| ------------- | --------------- | ----------------- | --------------------------------------------------------------------------------------- |
| `Dataset`     | `datasets`      | `dataset.py`      | Dataset metadata (name, fault types, severity levels, HDF5 file path)                   |
| `Signal`      | `signals`       | `signal.py`       | Individual signal records with statistical features (RMS, kurtosis, dominant frequency) |
| `Experiment`  | `experiments`   | `experiment.py`   | Training experiment metadata, config, metrics, and file references                      |
| `TrainingRun` | `training_runs` | `training_run.py` | Per-epoch training metrics (loss, accuracy, learning rate)                              |
| `HPOCampaign` | `hpo_campaigns` | `hpo_campaign.py` | Hyperparameter optimization campaigns (grid, random, Bayesian, Hyperband)               |
| `Explanation` | `explanations`  | `explanation.py`  | Cached XAI explanation results (SHAP, LIME, Grad-CAM, etc.)                             |

### Data Pipeline

| Model Class         | Table                 | File                    | Description                                                 |
| ------------------- | --------------------- | ----------------------- | ----------------------------------------------------------- |
| `DatasetGeneration` | `dataset_generations` | `dataset_generation.py` | Phase 0 signal generation job tracking with Celery task IDs |
| `DatasetImport`     | `dataset_imports`     | `dataset_import.py`     | MAT file import job tracking with progress percentage       |

### API & Security

| Model Class         | Table                 | File                 | Description                                                                |
| ------------------- | --------------------- | -------------------- | -------------------------------------------------------------------------- |
| `APIKey`            | `api_keys`            | `api_key.py`         | API key management with bcrypt hashing, scopes, rate limits, expiration    |
| `APIUsage`          | `api_usage`           | `api_key.py`         | Per-request API usage tracking for analytics                               |
| `APIRequestLog`     | `api_request_logs`    | `api_request_log.py` | Detailed API request logs with timing, size, and error tracking            |
| `APIMetricsSummary` | `api_metrics_summary` | `api_request_log.py` | Pre-aggregated API metrics (hourly/daily/weekly) with percentile latencies |

### Notifications

| Model Class              | Table                      | File                         | Description                                                |
| ------------------------ | -------------------------- | ---------------------------- | ---------------------------------------------------------- |
| `NotificationPreference` | `notification_preferences` | `notification_preference.py` | Per-user, per-event notification channel preferences       |
| `EmailLog`               | `email_logs`               | `email_log.py`               | Email delivery audit trail with provider tracking          |
| `EmailDigestQueue`       | `email_digest_queue`       | `email_digest_queue.py`      | Batched digest email queue for daily/weekly digests        |
| `WebhookConfiguration`   | `webhook_configurations`   | `webhook_configuration.py`   | Slack/Teams/custom webhook integrations with event routing |
| `WebhookLog`             | `webhook_logs`             | `webhook_log.py`             | Webhook delivery attempt logs with retry tracking          |

### Tags & Search

| Model Class     | Table             | File              | Description                                                 |
| --------------- | ----------------- | ----------------- | ----------------------------------------------------------- |
| `Tag`           | `tags`            | `tag.py`          | Experiment categorization tags with colors and usage counts |
| `ExperimentTag` | `experiment_tags` | `tag.py`          | Many-to-many junction between experiments and tags          |
| `SavedSearch`   | `saved_searches`  | `saved_search.py` | Bookmarked search queries with pinning and usage tracking   |

### Security & Authentication

| Model Class    | Table           | File               | Description                                             |
| -------------- | --------------- | ------------------ | ------------------------------------------------------- |
| `User`         | `users`         | `user.py`          | User accounts with roles, 2FA/TOTP support              |
| `SessionLog`   | `session_logs`  | `session_log.py`   | Active session tracking (token, device, location, IP)   |
| `LoginHistory` | `login_history` | `login_history.py` | Login attempt audit trail (success/failure, method, IP) |
| `BackupCode`   | `backup_codes`  | `backup_code.py`   | 2FA recovery codes (bcrypt-hashed, single-use)          |

### Neural Architecture Search

| Model Class   | Table           | File              | Description                                                                 |
| ------------- | --------------- | ----------------- | --------------------------------------------------------------------------- |
| `NASCampaign` | `nas_campaigns` | `nas_campaign.py` | NAS campaign tracking (random, Bayesian, evolution algorithms)              |
| `NASTrial`    | `nas_trials`    | `nas_campaign.py` | Individual architecture evaluations with complexity metrics (params, FLOPs) |

### System

| Model Class | Table         | File            | Description                                     |
| ----------- | ------------- | --------------- | ----------------------------------------------- |
| `SystemLog` | `system_logs` | `system_log.py` | System event log (action, status, details JSON) |

## Non-ORM Modules

| Module           | File                         | Description                                                                                                                                                                                                                          |
| ---------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| RBAC Permissions | `permissions.py`             | Role-Based Access Control with 4 roles: `VIEWER`, `OPERATOR`, `ADMIN`, `SUPER_ADMIN`. Provides `has_permission()`, `require_permission()` decorator, `require_role()` decorator, and convenience decorators (`admin_required`, etc.) |
| Event Types      | `notification_preference.py` | `EventType` constants class with 12 standard notification events (training, HPO, system)                                                                                                                                             |
| Email Status     | `email_log.py`               | `EmailStatus` constants: pending, sent, delivered, failed, bounced, opened, clicked                                                                                                                                                  |
| Webhook Status   | `webhook_log.py`             | `WebhookStatus` constants: sent, failed, rate_limited, timeout, invalid_url, provider_error                                                                                                                                          |

## Enumerations

| Enum                      | File                    | Values                                                             |
| ------------------------- | ----------------------- | ------------------------------------------------------------------ |
| `ExperimentStatus`        | `experiment.py`         | `pending`, `running`, `paused`, `completed`, `failed`, `cancelled` |
| `HPOMethod`               | `hpo_campaign.py`       | `grid_search`, `random_search`, `bayesian`, `hyperband`            |
| `DatasetGenerationStatus` | `dataset_generation.py` | `pending`, `running`, `completed`, `failed`, `cancelled`           |
| `DatasetImportStatus`     | `dataset_import.py`     | `pending`, `running`, `completed`, `failed`, `cancelled`           |
| `Role`                    | `permissions.py`        | `viewer`, `operator`, `admin`, `super_admin`                       |

## Usage

```python
# Import individual models
from models.user import User
from models.experiment import Experiment

# Or import everything via __init__.py
from models import User, Experiment, Dataset, APIKey
```

## Dependencies

- **Requires:** `sqlalchemy`, `models.base.Base`, `utils.constants`, `utils.logger`
- **Provides:** All ORM model classes, RBAC utilities, status constants, enumerations

## Related Documentation

- [Database README](../database/README.md) — Connection management, migrations, seed data
- [Schema Guide](../database/SCHEMA_GUIDE.md) — ER diagram, table catalog, relationship map
