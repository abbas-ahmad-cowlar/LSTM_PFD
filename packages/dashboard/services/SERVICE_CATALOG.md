# Backend Services — Service Catalog

> Detailed reference for each service class in `packages/dashboard/services/`.

---

## 1. APIKeyService

**File:** `api_key_service.py` · **Pattern:** Static methods

Handles generation, verification, and management of API keys with bcrypt hashing and prefix-based lookup.

**Public Methods:**

| Method                | Signature                                                                                 | Description                                                      |
| --------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `generate_key`        | `(user_id, name, environment='live', rate_limit=1000, expires_in_days=None, scopes=None)` | Generate new API key with bcrypt-hashed storage                  |
| `verify_key`          | `(api_key: str)`                                                                          | Verify key using prefix-based lookup + bcrypt compare            |
| `revoke_key`          | `(api_key_id, user_id)`                                                                   | Deactivate an API key (authorization check)                      |
| `list_user_keys`      | `(user_id, include_inactive=False)`                                                       | List all API keys for a user                                     |
| `get_key_usage_stats` | `(api_key_id, hours=24)`                                                                  | Usage statistics (total requests, error rate, avg response time) |
| `log_usage`           | `(api_key_id, endpoint, method, status_code, response_time_ms=None)`                      | Log API usage for analytics                                      |

**Dependencies:** `models.api_key.APIKey`, `models.api_key.APIUsage`, `models.user.User`, `bcrypt`, `secrets`

**Error Handling:** Returns `None` or `False` on failure; logs errors via `utils.logger`.

**Usage Example:**

```python
from services.api_key_service import APIKeyService

# Generate a new API key
result = APIKeyService.generate_key(user_id=1, name="CI/CD Pipeline")
print(result["key"])  # sk_live_...

# Verify an API key
key_record = APIKeyService.verify_key("sk_live_abc...")
if key_record:
    print(f"Authenticated as user {key_record.user_id}")
```

---

## 2. APIMonitoringService

**File:** `api_monitoring_service.py` · **Pattern:** Static methods

Tracks API requests and provides analytics — latency percentiles, endpoint metrics, and error logs.

**Public Methods:**

| Method                    | Signature                                                            | Description                  |
| ------------------------- | -------------------------------------------------------------------- | ---------------------------- |
| `log_request`             | `(endpoint, method, status_code, response_time_ms, ip_address, ...)` | Log an API request           |
| `get_recent_requests`     | `(limit=100, hours=24)`                                              | Recent API request list      |
| `get_request_stats`       | `(hours=24)`                                                         | Aggregate request statistics |
| `get_endpoint_metrics`    | `(hours=24)`                                                         | Per-endpoint metrics         |
| `get_latency_percentiles` | `(hours=24)`                                                         | P50, P95, P99 latency        |
| `get_request_timeline`    | `(hours=24, interval_minutes=5)`                                     | Requests per time interval   |
| `get_top_api_keys`        | `(limit=10, hours=24)`                                               | Most active API keys         |
| `get_error_logs`          | `(limit=50, hours=24)`                                               | Recent error logs            |

**Dependencies:** `models.api_request_log.APIRequestLog`, `models.api_request_log.APIMetricsSummary`, `models.api_key.APIKey`, `numpy`, `sqlalchemy`

**Usage Example:**

```python
from services.api_monitoring_service import APIMonitoringService

stats = APIMonitoringService.get_request_stats(hours=24)
percentiles = APIMonitoringService.get_latency_percentiles(hours=24)
print(f"P99 latency: {percentiles.get('p99')} ms")
```

---

## 3. AuthenticationService

**File:** `authentication_service.py` · **Pattern:** Static methods

Centralized authentication and security — TOTP/2FA, backup codes, sessions, login history. Includes in-memory rate limiting for 2FA attempts.

**Public Methods:**

| Method                  | Signature                                                                                                           | Description                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `generate_totp_secret`  | `()`                                                                                                                | Generate 32-char Base32 TOTP secret                           |
| `setup_2fa`             | `(user_id)`                                                                                                         | Set up 2FA — returns (success, qr_code_base64, secret, error) |
| `verify_totp`           | `(user_id, code, ip_address=None)`                                                                                  | Verify TOTP code with rate limiting                           |
| `generate_backup_codes` | `(user_id)`                                                                                                         | Generate new backup codes (invalidates previous)              |
| `verify_backup_code`    | `(user_id, code, ip_address=None)`                                                                                  | Verify and consume a backup code                              |
| `create_session`        | `(user_id, ip_address=None, user_agent=None, device_type=None, browser=None, location=None)`                        | Create user session                                           |
| `revoke_session`        | `(session_id, user_id)`                                                                                             | Terminate a session                                           |
| `record_login_attempt`  | `(user_id, success, login_method='password', ip_address=None, user_agent=None, location=None, failure_reason=None)` | Record login attempt                                          |

**Dependencies:** `models.user`, `utils.constants` (`MAX_2FA_ATTEMPTS`, `LOCKOUT_DURATION_MINUTES`), `pyotp` (implied by TOTP)

**Usage Example:**

```python
from services.authentication_service import AuthenticationService

success, qr, secret, error = AuthenticationService.setup_2fa(user_id=1)
if success:
    print(f"Scan QR: {qr[:20]}...")
```

---

## 4. CacheService

**File:** `cache_service.py` · **Pattern:** Static methods

Redis caching wrapper. Initializes a module-level `redis_client`; all methods gracefully degrade to no-ops if Redis is unavailable.

**Public Methods:**

| Method               | Signature                            | Description                            |
| -------------------- | ------------------------------------ | -------------------------------------- |
| `get`                | `(key: str)`                         | Get JSON-deserialized value from cache |
| `set`                | `(key, value, ttl=CACHE_TTL_MEDIUM)` | Set JSON-serialized value with TTL     |
| `delete`             | `(key: str)`                         | Delete key                             |
| `invalidate_pattern` | `(pattern: str)`                     | Delete all keys matching glob pattern  |
| `get_stats`          | `()`                                 | Redis info: memory, clients, hit rate  |

**Dependencies:** `redis`, `dashboard_config.REDIS_URL`, `dashboard_config.CACHE_TTL_MEDIUM`

**Usage Example:**

```python
from services.cache_service import CacheService

CacheService.set("dataset:1:stats", {"count": 500}, ttl=600)
stats = CacheService.get("dataset:1:stats")
CacheService.invalidate_pattern("dataset:1:*")
```

---

## 5. ComparisonService

**File:** `comparison_service.py` · **Pattern:** Static methods

Compares 2–3 experiments using confusion matrices, predictions, and statistical tests (McNemar for pairs, Friedman for 3-way).

**Public Methods:**

| Method                        | Signature                        | Description                                     |
| ----------------------------- | -------------------------------- | ----------------------------------------------- |
| `validate_comparison_request` | `(experiment_ids, user_id=None)` | Validate comparison is valid (2–3 experiments)  |
| `get_comparison_data`         | `(experiment_ids)`               | Load all comparison data + statistical tests    |
| `identify_key_differences`    | `(comparison_data)`              | Auto-detect key differences between experiments |

**Dependencies:** `models.experiment`, `models.training_run`, `dashboard_config.STORAGE_RESULTS_DIR`, `utils.statistical_tests` (`mcnemar_test`, `friedman_test`)

**Usage Example:**

```python
from services.comparison_service import ComparisonService

valid, error = ComparisonService.validate_comparison_request([1, 2])
if valid:
    data = ComparisonService.get_comparison_data([1, 2])
    diffs = ComparisonService.identify_key_differences(data)
```

---

## 6. DataService

**File:** `data_service.py` · **Pattern:** Static methods

Core dataset operations — CRUD, HDF5 signal loading, and cached statistics.

**Public Methods:**

| Method              | Signature                                                                               | Description                                |
| ------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------ |
| `list_datasets`     | `()`                                                                                    | List all datasets                          |
| `get_dataset`       | `(dataset_id)`                                                                          | Get dataset by ID                          |
| `get_dataset_stats` | `(dataset_id)`                                                                          | Get statistics (cached via `CacheService`) |
| `load_signals`      | `(dataset_id, fault_filter=None, limit=None)`                                           | Load signals with optional filtering       |
| `load_signal_data`  | `(dataset_id, signal_id)`                                                               | Load raw signal data from HDF5             |
| `create_dataset`    | `(name, description, fault_types, severity_levels, file_path, num_signals, created_by)` | Create dataset entry                       |
| `delete_dataset`    | `(dataset_id)`                                                                          | Delete dataset and signals                 |

**Dependencies:** `models.dataset`, `models.signal`, `services.cache_service.CacheService`, `h5py`, `numpy`

---

## 7. DatasetService

**File:** `dataset_service.py` · **Pattern:** Static methods

Advanced dataset management with pagination, preview, export, archival, and HDF5 statistics.

**Public Methods:**

| Method                   | Signature                                                     | Description                           |
| ------------------------ | ------------------------------------------------------------- | ------------------------------------- |
| `list_datasets`          | `(limit=MAX_DATASET_LIST_LIMIT, offset=0, search_query=None)` | Paginated dataset listing             |
| `get_dataset_details`    | `(dataset_id)`                                                | Full details + statistics             |
| `get_dataset_preview`    | `(dataset_id, num_samples=3)`                                 | Signal preview per fault type         |
| `delete_dataset`         | `(dataset_id, delete_file=False)`                             | Delete dataset (optionally HDF5 file) |
| `archive_dataset`        | `(dataset_id)`                                                | Mark dataset as archived              |
| `export_dataset`         | `(dataset_id, format='hdf5', output_dir='exports')`           | Export to HDF5/MAT/CSV                |
| `get_dataset_statistics` | `(dataset_id)`                                                | Compute dataset statistics            |

**Dependencies:** `models.dataset`, `h5py`, `numpy`

---

## 8. DeploymentService

**File:** `deployment_service.py` · **Pattern:** Static methods

Model deployment operations — export, quantization, pruning, and benchmarking.

**Public Methods:**

| Method                   | Signature                                                                                  | Description                        |
| ------------------------ | ------------------------------------------------------------------------------------------ | ---------------------------------- |
| `get_model_path`         | `(experiment_id)`                                                                          | Get model checkpoint path          |
| `load_model`             | `(experiment_id)`                                                                          | Load PyTorch model from experiment |
| `get_model_size`         | `(model_path)`                                                                             | File size in MB                    |
| `save_model`             | `(model, save_path, metadata=None)`                                                        | Save model with metadata           |
| `quantize_model_dynamic` | `(model)`                                                                                  | Dynamic INT8 quantization          |
| `quantize_model_static`  | `(model, calibration_data)`                                                                | Static INT8 quantization           |
| `convert_to_fp16`        | `(model)`                                                                                  | Half-precision conversion          |
| `export_to_onnx`         | `(model, save_path, input_shape=..., opset_version=..., optimize=True, dynamic_axes=True)` | ONNX export                        |
| `prune_model`            | `(model, amount=DEFAULT_PRUNING_AMOUNT, method='l1_unstructured')`                         | Model pruning                      |
| `benchmark_model`        | `(model, input_shape=..., num_runs=...)`                                                   | Inference speed benchmark          |

**Dependencies:** `torch`, `torch.nn`, `models.experiment`, `utils.constants`

---

## 9. EmailDigestService

**File:** `email_digest_service.py` · **Pattern:** Static methods

Manages the email digest queue — statistics, filtering, history, and Celery task triggering.

**Public Methods:**

| Method                      | Signature                                                                                   | Description                   |
| --------------------------- | ------------------------------------------------------------------------------------------- | ----------------------------- |
| `get_queue_stats`           | `()`                                                                                        | Pending/included/today counts |
| `get_pending_digests`       | `(event_type_filter='all', user_id_filter=None, time_filter='all', page=1, page_size=None)` | Filtered pending items        |
| `get_digest_history`        | `(limit=50)`                                                                                | Recently processed items      |
| `get_users_with_digests`    | `()`                                                                                        | Users with queued items       |
| `trigger_digest_processing` | `()`                                                                                        | Trigger Celery digest task    |

**Dependencies:** `models.email_digest_queue`, `models.user`, `dashboard_config.EMAIL_DIGEST_PAGE_SIZE`

---

## 10. EmailProvider (+ Implementations)

**File:** `email_provider.py` · **Pattern:** ABC + Concrete implementations

Abstract email provider with SendGrid, SMTP implementations, a factory, and a rate limiter.

**Classes:**

| Class                  | Description                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
| `EmailProvider` (ABC)  | Abstract: `send(to_email, subject, html_body, text_body, ...)`, `get_message_status(message_id)` |
| `SendGridProvider`     | SendGrid API integration                                                                         |
| `SMTPProvider`         | SMTP fallback provider                                                                           |
| `EmailProviderFactory` | `create_provider(provider_type, config)` — returns appropriate provider                          |
| `EmailRateLimiter`     | Token-bucket rate limiter backed by Redis                                                        |

**Dependencies:** `sendgrid`, `smtplib` (stdlib), `redis`

---

## 11. EvaluationService

**File:** `evaluation_service.py` · **Pattern:** Static methods

Advanced model evaluation — ROC curves, error analysis, architecture comparison, noise robustness.

**Public Methods:**

| Method                        | Signature                                              | Description                              |
| ----------------------------- | ------------------------------------------------------ | ---------------------------------------- |
| `load_experiment_predictions` | `(experiment_id)`                                      | Load predictions, probabilities, targets |
| `generate_roc_data`           | `(probabilities, targets, class_names)`                | ROC curves per class                     |
| `analyze_errors`              | `(predictions, probabilities, targets, class_names)`   | Comprehensive error analysis             |
| `compare_architectures`       | `(experiment_ids)`                                     | Multi-experiment comparison              |
| `test_robustness`             | `(experiment_id, noise_levels=[0.01, 0.05, 0.1, 0.2])` | Noise robustness testing                 |
| `cache_evaluation_results`    | `(experiment_id, evaluation_type, results)`            | Cache results                            |
| `get_cached_evaluation`       | `(experiment_id, evaluation_type)`                     | Retrieve cached results                  |

**Dependencies:** `models.experiment`, `torch`, `numpy`

---

## 12. ExplanationCache

**File:** `explanation_cache.py` · **Pattern:** Static methods

Database-backed XAI explanation cache. Stores, retrieves, and prunes explanation results.

**Public Methods:**

| Method                    | Signature                                              | Description                 |
| ------------------------- | ------------------------------------------------------ | --------------------------- |
| `get_explanation`         | `(experiment_id, signal_id, method)`                   | Retrieve cached explanation |
| `cache_explanation`       | `(experiment_id, signal_id, method, explanation_data)` | Store explanation           |
| `get_recent_explanations` | `(experiment_id=None, limit=10)`                       | Recent explanations         |
| `delete_explanation`      | `(explanation_id)`                                     | Delete by ID                |
| `clear_old_explanations`  | `(days=30)`                                            | Prune old explanations      |
| `get_cache_statistics`    | `()`                                                   | Cache usage stats           |
| `explanation_exists`      | `(experiment_id, signal_id, method)`                   | Check existence             |

**Dependencies:** `models.explanation`

---

## 13. FeatureService

**File:** `feature_service.py` · **Pattern:** Static methods

Feature engineering — extraction across multiple domains, importance analysis, selection, and persistence.

**Public Methods:**

| Method                        | Signature                                                                                         | Description                                                                                |
| ----------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `extract_features`            | `(dataset_id, domain, config=None)`                                                               | Extract features (domain: 'all', 'time', 'frequency', 'wavelet', 'bispectrum', 'envelope') |
| `compute_feature_importance`  | `(features, labels, feature_names, method='mutual_info')`                                         | Importance analysis (mutual_info, random_forest, permutation)                              |
| `select_features`             | `(features, labels, feature_names, method, num_features=15, threshold=0.01)`                      | Feature selection (mrmr, variance, mutual_info, rfe)                                       |
| `compute_feature_correlation` | `(features, feature_names)`                                                                       | Correlation matrix                                                                         |
| `save_feature_set`            | `(dataset_id, features, labels, feature_names, domain, metadata=None, output_dir='feature_sets')` | Persist feature set                                                                        |

**Dependencies:** `models.dataset`, `h5py`, `numpy`

---

## 14. HPOService

**File:** `hpo_service.py` · **Pattern:** Static methods

Hyperparameter optimization campaign management — creation, tracking, export, and analysis.

**Public Methods:**

| Method                     | Signature                                                                                                                             | Description                          |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| `create_campaign`          | `(name, method, base_model_type, dataset_id, search_space, num_trials, metric='val_accuracy', direction='maximize', created_by=None)` | Create HPO campaign                  |
| `get_campaign`             | `(campaign_id)`                                                                                                                       | Get campaign by ID                   |
| `get_all_campaigns`        | `()`                                                                                                                                  | List all campaigns                   |
| `update_campaign_progress` | `(campaign_id, trials_completed, best_experiment_id=None, best_accuracy=None)`                                                        | Update progress                      |
| `update_campaign_status`   | `(campaign_id, status)`                                                                                                               | Update status                        |
| `get_campaign_experiments` | `(campaign_id)`                                                                                                                       | Get campaign experiments             |
| `get_default_search_space` | `(model_type)`                                                                                                                        | Default search space per model type  |
| `resume_campaign`          | `(campaign_id)`                                                                                                                       | Resume paused/cancelled campaign     |
| `export_results`           | `(campaign_id, format='json')`                                                                                                        | Export (JSON/YAML/Python)            |
| `get_trials_dataframe`     | `(campaign_id)`                                                                                                                       | Trial data for visualization         |
| `get_parameter_importance` | `(campaign_id)`                                                                                                                       | Parameter importance via correlation |
| `save_research_artifact`   | `(campaign_id)`                                                                                                                       | Save results as JSON artifact        |

**Dependencies:** `models.hpo_campaign`, `models.experiment`, `models.dataset`

---

## 15. MonitoringService

**File:** `monitoring_service.py` · **Pattern:** Instance methods (stateful)

Application health monitoring with background thread, `psutil`-based metrics, and alerting.

> **Note:** This is the only service that uses **instance state** and a background thread. A global singleton `monitoring_service` is created at module level.

**Public Methods:**

| Method                | Signature               | Description                         |
| --------------------- | ----------------------- | ----------------------------------- |
| `start_monitoring`    | `(interval_seconds=60)` | Start background metrics collection |
| `stop_monitoring`     | `()`                    | Stop background thread              |
| `get_current_metrics` | `()`                    | Current CPU/RAM/disk metrics        |
| `get_recent_alerts`   | `(hours=24)`            | Recent alert list                   |
| `get_health_status`   | `()`                    | Overall health status               |

**Dependencies:** `psutil`, `models.system_log`, `threading`

---

## 16. NASService

**File:** `nas_service.py` · **Pattern:** Static methods

Neural Architecture Search — campaign management, architecture sampling/deduplication, and code export.

**Public Methods:**

| Method                      | Signature                                                                                                    | Description                    |
| --------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------ |
| `create_nas_campaign`       | `(name, dataset_id, search_space_config, search_algorithm='random', num_trials=20, max_epochs_per_trial=10)` | Create NAS campaign            |
| `sample_architecture`       | `(search_space_config)`                                                                                      | Random architecture sampling   |
| `compute_architecture_hash` | `(architecture)`                                                                                             | SHA-256 deduplication hash     |
| `get_campaign_details`      | `(campaign_id)`                                                                                              | Campaign with all trials       |
| `get_best_architecture`     | `(campaign_id)`                                                                                              | Best-performing trial          |
| `export_architecture`       | `(trial_id, format='pytorch')`                                                                               | Export as PyTorch code or JSON |
| `list_campaigns`            | `(limit=50, offset=0)`                                                                                       | List all campaigns             |

**Dependencies:** `models.nas_campaign`, `models.dataset`

---

## 17. NotificationService

**File:** `notification_service.py` · **Pattern:** Class methods + Static methods

Multi-channel notification system — routes events to toast, email (immediate/digest), and webhooks based on user preferences.

**Public Methods:**

| Method         | Signature                                                       | Description                        |
| -------------- | --------------------------------------------------------------- | ---------------------------------- |
| `initialize`   | `(cls, email_config, redis_client)`                             | One-time initialization at startup |
| `emit_event`   | `(event_type, user_id, data)`                                   | Main entry — routes to channels    |
| `create_toast` | `(message, notification_type='info', duration=5000, link=None)` | Create toast (legacy)              |
| `success`      | `(message, link=None)`                                          | Success toast shorthand            |
| `error`        | `(message)`                                                     | Error toast shorthand              |
| `warning`      | `(message)`                                                     | Warning toast shorthand            |
| `info`         | `(message, link=None)`                                          | Info toast shorthand               |

**Dependencies:** `EmailProvider`, `notification_providers`, `models.user`, `models.webhook_configuration`, `redis`, `jinja2`

---

## 18. SearchService

**File:** `search_service.py` · **Pattern:** Static methods

Structured experiment search with query parsing, SQL query building, relevance ranking, and saved searches.

**Query Syntax:**

- `tag:baseline` — filter by tag
- `accuracy:>0.95` — accuracy filter
- `status:completed` — status filter
- `model:resnet` — model type filter
- Free text keywords for full-text search

**Public Methods:**

| Method                | Signature                                          | Description             |
| --------------------- | -------------------------------------------------- | ----------------------- |
| `search`              | `(session, query, user_id=None, limit=100)`        | Main search entry point |
| `save_search`         | `(session, user_id, name, query, is_pinned=False)` | Save search query       |
| `get_saved_searches`  | `(session, user_id)`                               | List saved searches     |
| `use_saved_search`    | `(session, saved_search_id, user_id)`              | Execute saved search    |
| `delete_saved_search` | `(session, saved_search_id, user_id)`              | Delete saved search     |

**Dependencies:** `models.experiment`, `models.tag`, `models.saved_search`

---

## 19. SignalService

**File:** `signal_service.py` · **Pattern:** Static methods

Signal processing — FFT, spectrogram (STFT), and basic statistical feature extraction.

**Public Methods:**

| Method                     | Signature                                      | Description                                                     |
| -------------------------- | ---------------------------------------------- | --------------------------------------------------------------- |
| `compute_fft`              | `(signal_data, fs=SAMPLING_RATE)`              | FFT computation → (freq, magnitude)                             |
| `compute_spectrogram`      | `(signal_data, fs=SAMPLING_RATE, nperseg=256)` | STFT spectrogram → (f, t, Sxx)                                  |
| `extract_basic_features`   | `(signal_data, fs=SAMPLING_RATE)`              | RMS, kurtosis, skewness, peak, dominant freq, spectral centroid |
| `get_signal_with_features` | `(dataset_id, signal_id)`                      | Load signal + compute features                                  |

**Dependencies:** `scipy.signal`, `scipy.stats`, `services.data_service.DataService`, `numpy`

---

## 20. TagService

**File:** `tag_service.py` · **Pattern:** Static methods

Experiment tag management — CRUD, autocomplete, bulk operations, and statistics.

**Public Methods:**

| Method                       | Signature                                                      | Description                   |
| ---------------------------- | -------------------------------------------------------------- | ----------------------------- |
| `slugify`                    | `(text)`                                                       | Convert to URL-safe slug      |
| `create_or_get_tag`          | `(session, name, color=None, user_id=None)`                    | Create or return existing tag |
| `add_tag_to_experiment`      | `(session, experiment_id, tag_name, user_id=None, color=None)` | Tag an experiment             |
| `remove_tag_from_experiment` | `(session, experiment_id, tag_id)`                             | Remove tag                    |
| `get_experiment_tags`        | `(session, experiment_id)`                                     | List tags on experiment       |
| `get_popular_tags`           | `(session, limit=20, min_usage=1)`                             | Tags sorted by usage          |
| `suggest_tags`               | `(session, query, limit=10)`                                   | Autocomplete                  |
| `bulk_add_tags`              | `(session, experiment_ids, tag_names, user_id=None)`           | Bulk add                      |
| `bulk_remove_tags`           | `(session, experiment_ids, tag_ids)`                           | Bulk remove                   |
| `get_tag_statistics`         | `(session)`                                                    | Overall tag stats             |

**Dependencies:** `models.tag`, `models.experiment`

---

## 21. TestingService

**File:** `testing_service.py` · **Pattern:** Static methods

Programmatic test execution, coverage analysis, benchmarking, and code quality checks.

**Public Methods:**

| Method                    | Signature                                                               | Description               |
| ------------------------- | ----------------------------------------------------------------------- | ------------------------- |
| `run_pytest`              | `(test_path='tests/', markers=None, verbose=True, capture_output=True)` | Run pytest                |
| `run_coverage`            | `(test_path='tests/', source_path='.', min_coverage=80.0)`              | Coverage analysis         |
| `run_benchmarks`          | `(model_path=None, api_url=None, num_samples=100)`                      | Performance benchmarks    |
| `parse_pytest_output`     | `(stdout)`                                                              | Parse pytest text output  |
| `get_test_files`          | `()`                                                                    | List available test files |
| `validate_code_quality`   | `(path='.')`                                                            | Flake8/mypy checks        |
| `get_recent_test_history` | `(limit=10)`                                                            | Recent test run history   |

**Dependencies:** `subprocess`, `pathlib`

---

## 22. WebhookService

**File:** `webhook_service.py` · **Pattern:** Static methods

CRUD for webhook configurations with test delivery, logging, and statistics.

**Public Methods:**

| Method               | Signature                                                                                                      | Description                   |
| -------------------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| `list_user_webhooks` | `(user_id, include_inactive=False)`                                                                            | List user's webhooks          |
| `get_webhook`        | `(webhook_id, user_id)`                                                                                        | Get webhook (ownership check) |
| `create_webhook`     | `(user_id, provider_type, webhook_url, name, enabled_events, description=None, is_active=True, settings=None)` | Create webhook                |
| `update_webhook`     | `(webhook_id, user_id, **updates)`                                                                             | Update webhook                |
| `delete_webhook`     | `(webhook_id, user_id)`                                                                                        | Delete webhook                |
| `toggle_webhook`     | `(webhook_id, user_id, is_active)`                                                                             | Enable/disable                |
| `test_webhook`       | `(webhook_id, user_id)`                                                                                        | Send test notification        |
| `get_webhook_logs`   | `(webhook_id, user_id, limit=10)`                                                                              | Delivery logs                 |
| `get_webhook_stats`  | `(webhook_id, user_id)`                                                                                        | Webhook statistics            |

**Dependencies:** `models.webhook_configuration`, `models.webhook_log`, `notification_providers`

---

## 23. XAIService

**File:** `xai_service.py` · **Pattern:** Static methods

Explainability — SHAP, LIME, Integrated Gradients, and Grad-CAM explanation generation.

**Public Methods:**

| Method                          | Signature                                                                   | Description                |
| ------------------------------- | --------------------------------------------------------------------------- | -------------------------- |
| `get_device`                    | `()`                                                                        | Detect GPU/CPU             |
| `generate_shap_explanation`     | `(model, signal, background_data=None, method='gradient', num_samples=100)` | SHAP explanation           |
| `generate_lime_explanation`     | `(model, signal, num_segments=20, num_samples=1000, target_class=None)`     | LIME explanation           |
| `generate_integrated_gradients` | `(model, signal, baseline=None, steps=50)`                                  | Integrated Gradients       |
| `generate_gradcam`              | `(model, signal, target_layer=None)`                                        | Grad-CAM heatmap           |
| `load_model`                    | `(experiment_id)`                                                           | Load model from experiment |
| `get_model_prediction`          | `(model, signal)`                                                           | Get prediction results     |

**Dependencies:** `torch`, `captum` (optional), Phase 7 explainers

---

## 24. notification_providers (Sub-Package)

**Directory:** `notification_providers/`

| Module                       | Class                         | Description                                                    |
| ---------------------------- | ----------------------------- | -------------------------------------------------------------- |
| `base.py`                    | `NotificationProvider` (ABC)  | Abstract base: `send(message)`, `validate_config(config)`      |
| `base.py`                    | `NotificationMessage`         | Dataclass: `title`, `body`, `color`, `fields`, `footer`, `url` |
| `factory.py`                 | `NotificationProviderFactory` | `create_provider(provider_type, config)` → provider instance   |
| `slack_notifier.py`          | `SlackNotifier`               | Slack webhook with Block Kit formatting                        |
| `teams_notifier.py`          | `TeamsNotifier`               | MS Teams webhook with Adaptive Cards                           |
| `custom_webhook_notifier.py` | `CustomWebhookNotifier`       | Generic webhook with retry logic                               |

**Dependencies:** `requests`, `json`
