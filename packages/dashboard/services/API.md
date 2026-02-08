# Backend Services — API Reference

> Full API reference for all service classes in `packages/dashboard/services/`.

---

## APIKeyService

> API key generation, verification, and management.

**File:** `api_key_service.py`

### `generate_key(user_id, name, environment, rate_limit, expires_in_days, scopes) → Dict`

Generate a new API key with bcrypt-hashed storage.

| Parameter         | Type                  | Default  | Description                               |
| ----------------- | --------------------- | -------- | ----------------------------------------- |
| `user_id`         | `int`                 | required | User ID                                   |
| `name`            | `str`                 | required | Descriptive name (e.g., "CI/CD Pipeline") |
| `environment`     | `str`                 | `'live'` | `'live'` or `'test'`                      |
| `rate_limit`      | `int`                 | `1000`   | Requests per hour                         |
| `expires_in_days` | `Optional[int]`       | `None`   | Expiration (None = never)                 |
| `scopes`          | `Optional[List[str]]` | `None`   | Permission scopes                         |

**Returns:** Dict with `key` (full key string), `prefix`, `api_key_id`.

**Example:**

```python
result = APIKeyService.generate_key(user_id=1, name="CI/CD")
print(result["key"])  # sk_live_...
```

### `verify_key(api_key) → Optional[APIKey]`

Verify API key using prefix-based lookup + bcrypt comparison.

| Parameter | Type  | Default  | Description         |
| --------- | ----- | -------- | ------------------- |
| `api_key` | `str` | required | Full API key string |

**Returns:** `APIKey` record or `None`.

### `revoke_key(api_key_id, user_id) → bool`

Deactivate an API key.

| Parameter    | Type  | Default  | Description                   |
| ------------ | ----- | -------- | ----------------------------- |
| `api_key_id` | `int` | required | Key ID                        |
| `user_id`    | `int` | required | User ID (authorization check) |

### `list_user_keys(user_id, include_inactive) → List[APIKey]`

| Parameter          | Type   | Default  | Description          |
| ------------------ | ------ | -------- | -------------------- |
| `user_id`          | `int`  | required | User ID              |
| `include_inactive` | `bool` | `False`  | Include revoked keys |

### `get_key_usage_stats(api_key_id, hours) → Dict`

| Parameter    | Type  | Default  | Description     |
| ------------ | ----- | -------- | --------------- |
| `api_key_id` | `int` | required | Key ID          |
| `hours`      | `int` | `24`     | Lookback window |

**Returns:** Dict with `total_requests`, `error_rate`, `avg_response_time_ms`.

### `log_usage(api_key_id, endpoint, method, status_code, response_time_ms) → None`

| Parameter          | Type            | Default  | Description       |
| ------------------ | --------------- | -------- | ----------------- |
| `api_key_id`       | `int`           | required | Key ID            |
| `endpoint`         | `str`           | required | API endpoint path |
| `method`           | `str`           | required | HTTP method       |
| `status_code`      | `int`           | required | HTTP status code  |
| `response_time_ms` | `Optional[int]` | `None`   | Response time     |

---

## APIMonitoringService

> API request tracking and analytics.

**File:** `api_monitoring_service.py`

### `log_request(endpoint, method, status_code, response_time_ms, ip_address, ...) → None`

| Parameter          | Type   | Default  | Description         |
| ------------------ | ------ | -------- | ------------------- |
| `endpoint`         | `str`  | required | API endpoint        |
| `method`           | `str`  | required | HTTP method         |
| `status_code`      | `int`  | required | HTTP status         |
| `response_time_ms` | `int`  | required | Response time in ms |
| `ip_address`       | `str`  | `None`   | Client IP           |
| `user_agent`       | `str`  | `None`   | Browser user agent  |
| `api_key_id`       | `int`  | `None`   | API key ID          |
| `request_size`     | `int`  | `0`      | Request body size   |
| `response_size`    | `int`  | `0`      | Response body size  |
| `error_message`    | `str`  | `None`   | Error message       |
| `request_sample`   | `dict` | `None`   | Request sample      |
| `response_sample`  | `dict` | `None`   | Response sample     |

### `get_recent_requests(limit, hours) → List[Dict]`

| Parameter | Type  | Default | Description     |
| --------- | ----- | ------- | --------------- |
| `limit`   | `int` | `100`   | Max results     |
| `hours`   | `int` | `24`    | Lookback window |

### `get_request_stats(hours) → Dict`

| Parameter | Type  | Default | Description     |
| --------- | ----- | ------- | --------------- |
| `hours`   | `int` | `24`    | Lookback window |

### `get_endpoint_metrics(hours) → List[Dict]`

Per-endpoint aggregated metrics.

### `get_latency_percentiles(hours) → Dict`

Returns P50, P95, P99 percentiles.

### `get_request_timeline(hours, interval_minutes) → List[Dict]`

| Parameter          | Type  | Default | Description      |
| ------------------ | ----- | ------- | ---------------- |
| `hours`            | `int` | `24`    | Lookback window  |
| `interval_minutes` | `int` | `5`     | Time bucket size |

### `get_top_api_keys(limit, hours) → List[Dict]`

### `get_error_logs(limit, hours) → List[Dict]`

---

## AuthenticationService

> TOTP/2FA, backup codes, sessions, and login history.

**File:** `authentication_service.py`

### `generate_totp_secret() → str`

Returns a 32-character Base32-encoded TOTP secret.

### `setup_2fa(user_id) → Tuple[bool, str, str, str]`

| Parameter | Type  | Default  | Description |
| --------- | ----- | -------- | ----------- |
| `user_id` | `int` | required | User ID     |

**Returns:** `(success, qr_code_base64, secret_formatted, error_message)`

### `verify_totp(user_id, code, ip_address) → Tuple[bool, str]`

| Parameter    | Type            | Default  | Description       |
| ------------ | --------------- | -------- | ----------------- |
| `user_id`    | `int`           | required | User ID           |
| `code`       | `str`           | required | 6-digit TOTP code |
| `ip_address` | `Optional[str]` | `None`   | IP for logging    |

**Returns:** `(success, error_message)`. Rate-limited to `MAX_2FA_ATTEMPTS`.

### `generate_backup_codes(user_id) → Tuple[bool, List[str], str]`

**Returns:** `(success, codes, error_message)`. Invalidates all previous codes.

### `verify_backup_code(user_id, code, ip_address) → Tuple[bool, str]`

Verify and consume a one-time backup code.

### `create_session(user_id, ip_address, user_agent, device_type, browser, location) → Optional[Dict]`

### `revoke_session(session_id, user_id) → Tuple[bool, str]`

### `record_login_attempt(user_id, success, login_method, ip_address, user_agent, location, failure_reason) → None`

---

## CacheService

> Redis caching wrapper with graceful degradation.

**File:** `cache_service.py`

### `get(key) → Optional[Any]`

Returns JSON-deserialized value, or `None` if key not found or Redis unavailable.

### `set(key, value, ttl) → bool`

| Parameter | Type  | Default            | Description             |
| --------- | ----- | ------------------ | ----------------------- |
| `key`     | `str` | required           | Cache key               |
| `value`   | `Any` | required           | JSON-serializable value |
| `ttl`     | `int` | `CACHE_TTL_MEDIUM` | TTL in seconds          |

### `delete(key) → bool`

### `invalidate_pattern(pattern) → int`

Delete all keys matching glob pattern. Returns count deleted.

### `get_stats() → Dict`

Returns `status`, `used_memory`, `connected_clients`, `hit_rate`, etc.

---

## ComparisonService

> Multi-experiment statistical comparison.

**File:** `comparison_service.py`

### `validate_comparison_request(experiment_ids, user_id) → Tuple[bool, Optional[str]]`

### `get_comparison_data(experiment_ids) → Dict`

**Returns:** Dict with `experiments` (list of dicts with metrics, confusion_matrix, config) and `statistical_tests` (McNemar for pairs, Friedman for 3-way).

### `identify_key_differences(comparison_data) → List[str]`

Returns human-readable difference descriptions.

---

## DataService

> Core dataset CRUD and HDF5 signal loading.

**File:** `data_service.py`

### `list_datasets() → List[Dict]`

### `get_dataset(dataset_id) → Optional[Dict]`

### `get_dataset_stats(dataset_id) → Dict`

Cached via `CacheService`.

### `load_signals(dataset_id, fault_filter, limit) → List`

### `load_signal_data(dataset_id, signal_id) → np.ndarray`

### `create_dataset(name, description, fault_types, severity_levels, file_path, num_signals, created_by) → Optional[Dict]`

### `delete_dataset(dataset_id) → bool`

---

## DatasetService

> Advanced dataset management with pagination, export, and archival.

**File:** `dataset_service.py`

### `list_datasets(limit, offset, search_query) → List[Dict]`

| Parameter      | Type  | Default                  | Description   |
| -------------- | ----- | ------------------------ | ------------- |
| `limit`        | `int` | `MAX_DATASET_LIST_LIMIT` | Page size     |
| `offset`       | `int` | `0`                      | Offset        |
| `search_query` | `str` | `None`                   | Search filter |

### `get_dataset_details(dataset_id) → Optional[Dict]`

### `get_dataset_preview(dataset_id, num_samples) → Dict`

### `delete_dataset(dataset_id, delete_file) → bool`

| Parameter     | Type   | Default  | Description           |
| ------------- | ------ | -------- | --------------------- |
| `dataset_id`  | `int`  | required | Dataset ID            |
| `delete_file` | `bool` | `False`  | Also delete HDF5 file |

### `archive_dataset(dataset_id) → bool`

### `export_dataset(dataset_id, format, output_dir) → Optional[str]`

| Parameter    | Type  | Default     | Description                |
| ------------ | ----- | ----------- | -------------------------- |
| `format`     | `str` | `'hdf5'`    | `'hdf5'`, `'mat'`, `'csv'` |
| `output_dir` | `str` | `'exports'` | Output directory           |

### `get_dataset_statistics(dataset_id) → Dict`

---

## DeploymentService

> Model export, quantization, pruning, and benchmarking.

**File:** `deployment_service.py`

### `get_model_path(experiment_id) → Optional[Path]`

### `load_model(experiment_id) → Optional[nn.Module]`

### `get_model_size(model_path) → float`

Returns size in MB.

### `save_model(model, save_path, metadata) → None`

### `quantize_model_dynamic(model) → nn.Module`

Dynamic INT8 quantization.

### `quantize_model_static(model, calibration_data) → nn.Module`

Static INT8 quantization with calibration.

### `convert_to_fp16(model) → nn.Module`

### `export_to_onnx(model, save_path, input_shape, opset_version, optimize, dynamic_axes) → bool`

| Parameter       | Type        | Default                      | Description         |
| --------------- | ----------- | ---------------------------- | ------------------- |
| `model`         | `nn.Module` | required                     | PyTorch model       |
| `save_path`     | `Path`      | required                     | Output path         |
| `input_shape`   | `tuple`     | `DEFAULT_ONNX_INPUT_SHAPE`   | Input shape         |
| `opset_version` | `int`       | `DEFAULT_ONNX_OPSET_VERSION` | ONNX opset          |
| `optimize`      | `bool`      | `True`                       | Apply optimizations |
| `dynamic_axes`  | `bool`      | `True`                       | Dynamic batch size  |

### `prune_model(model, amount, method) → nn.Module`

| Parameter | Type    | Default                  | Description                 |
| --------- | ------- | ------------------------ | --------------------------- |
| `amount`  | `float` | `DEFAULT_PRUNING_AMOUNT` | Fraction to prune (0.0–1.0) |
| `method`  | `str`   | `'l1_unstructured'`      | Pruning method              |

### `benchmark_model(model, input_shape, num_runs) → Dict`

## Performance

> ⚠️ **Results pending.** Performance metrics below will be populated after experiments are run on the current codebase.

| Metric                      | Value       |
| --------------------------- | ----------- |
| Inference Time              | `[PENDING]` |
| Model Size Reduction (INT8) | `[PENDING]` |
| Model Size Reduction (FP16) | `[PENDING]` |

---

## EmailDigestService

> Digest queue management and processing.

**File:** `email_digest_service.py`

### `get_queue_stats() → Dict`

Returns `pending_count`, `included_count`, `today_count`.

### `get_pending_digests(event_type_filter, user_id_filter, time_filter, page, page_size) → Tuple[List, int]`

### `get_digest_history(limit) → List[Tuple]`

### `get_users_with_digests() → List[User]`

### `trigger_digest_processing() → bool`

Triggers Celery task; returns `False` if Celery unavailable.

---

## EmailProvider (ABC)

> Abstract email provider interface.

**File:** `email_provider.py`

### `send(to_email, subject, html_body, text_body, from_email, from_name) → Dict`

| Parameter    | Type            | Default  | Description          |
| ------------ | --------------- | -------- | -------------------- |
| `to_email`   | `str`           | required | Recipient            |
| `subject`    | `str`           | required | Subject line         |
| `html_body`  | `str`           | required | HTML body            |
| `text_body`  | `str`           | required | Plain-text fallback  |
| `from_email` | `Optional[str]` | `None`   | Override sender      |
| `from_name`  | `Optional[str]` | `None`   | Override sender name |

**Returns:** Dict with `success`, `message_id`, `error`.

### `get_message_status(message_id) → str`

---

### SendGridProvider

Constructor: `SendGridProvider(api_key, default_from_email, default_from_name='LSTM Dashboard')`

### SMTPProvider

Constructor: `SMTPProvider(smtp_host, smtp_port, username, password, default_from_email, default_from_name='LSTM Dashboard')`

### EmailProviderFactory

#### `create_provider(provider_type, config) → EmailProvider`

| Parameter       | Type             | Description                     |
| --------------- | ---------------- | ------------------------------- |
| `provider_type` | `str`            | `'sendgrid'` or `'smtp'`        |
| `config`        | `Dict[str, Any]` | Provider-specific configuration |

### EmailRateLimiter

Constructor: `EmailRateLimiter(redis_client, max_emails_per_minute=100)`

#### `can_send() → bool`

#### `reset() → None`

---

## EvaluationService

> Advanced model evaluation and analysis.

**File:** `evaluation_service.py`

### `load_experiment_predictions(experiment_id) → Dict`

Returns `predictions`, `probabilities`, `targets`, `class_names`.

### `generate_roc_data(probabilities, targets, class_names) → Dict`

### `analyze_errors(predictions, probabilities, targets, class_names) → Dict`

### `compare_architectures(experiment_ids) → Dict`

### `test_robustness(experiment_id, noise_levels) → Dict`

| Parameter      | Type          | Default                  | Description |
| -------------- | ------------- | ------------------------ | ----------- |
| `noise_levels` | `List[float]` | `[0.01, 0.05, 0.1, 0.2]` | SNR levels  |

### `cache_evaluation_results(experiment_id, evaluation_type, results) → None`

### `get_cached_evaluation(experiment_id, evaluation_type) → Optional[Dict]`

---

## ExplanationCache

> Database-backed XAI explanation caching.

**File:** `explanation_cache.py`

### `get_explanation(experiment_id, signal_id, method) → Optional[Dict]`

### `cache_explanation(experiment_id, signal_id, method, explanation_data) → bool`

### `get_recent_explanations(experiment_id, limit) → List[Dict]`

### `delete_explanation(explanation_id) → bool`

### `clear_old_explanations(days) → int`

| Parameter | Type  | Default | Description      |
| --------- | ----- | ------- | ---------------- |
| `days`    | `int` | `30`    | Retention period |

### `get_cache_statistics() → Dict`

### `explanation_exists(experiment_id, signal_id, method) → bool`

---

## FeatureService

> Feature engineering: extraction, importance, selection.

**File:** `feature_service.py`

### `extract_features(dataset_id, domain, config) → Dict`

| Parameter    | Type             | Default  | Description                                                                 |
| ------------ | ---------------- | -------- | --------------------------------------------------------------------------- |
| `dataset_id` | `int`            | required | Dataset                                                                     |
| `domain`     | `str`            | required | `'all'`, `'time'`, `'frequency'`, `'wavelet'`, `'bispectrum'`, `'envelope'` |
| `config`     | `Optional[Dict]` | `None`   | Extraction config                                                           |

**Returns:** Dict with `features`, `feature_names`, `extraction_time`.

### `compute_feature_importance(features, labels, feature_names, method) → Dict`

| Parameter | Type  | Default         | Description                                         |
| --------- | ----- | --------------- | --------------------------------------------------- |
| `method`  | `str` | `'mutual_info'` | `'mutual_info'`, `'random_forest'`, `'permutation'` |

### `select_features(features, labels, feature_names, method, num_features, threshold) → Dict`

| Parameter      | Type    | Default  | Description                                      |
| -------------- | ------- | -------- | ------------------------------------------------ |
| `method`       | `str`   | required | `'mrmr'`, `'variance'`, `'mutual_info'`, `'rfe'` |
| `num_features` | `int`   | `15`     | Number to select                                 |
| `threshold`    | `float` | `0.01`   | Minimum importance                               |

### `compute_feature_correlation(features, feature_names) → Dict`

### `save_feature_set(dataset_id, features, labels, feature_names, domain, metadata, output_dir) → str`

---

## HPOService

> Hyperparameter optimization campaign management.

**File:** `hpo_service.py`

### `create_campaign(name, method, base_model_type, dataset_id, search_space, num_trials, metric, direction, created_by) → Optional[Dict]`

| Parameter   | Type  | Default          | Description                                       |
| ----------- | ----- | ---------------- | ------------------------------------------------- |
| `method`    | `str` | required         | `'bayesian'`, `'random'`, `'grid'`, `'hyperband'` |
| `metric`    | `str` | `'val_accuracy'` | Optimization metric                               |
| `direction` | `str` | `'maximize'`     | `'maximize'` or `'minimize'`                      |

### `get_campaign(campaign_id) → Optional[Dict]`

### `get_all_campaigns() → List[Dict]`

### `update_campaign_progress(campaign_id, trials_completed, best_experiment_id, best_accuracy) → bool`

### `update_campaign_status(campaign_id, status) → bool`

### `get_campaign_experiments(campaign_id) → List[Dict]`

### `get_default_search_space(model_type) → Dict`

### `resume_campaign(campaign_id) → bool`

### `export_results(campaign_id, format) → Optional[str]`

| Parameter | Type  | Default  | Description                    |
| --------- | ----- | -------- | ------------------------------ |
| `format`  | `str` | `'json'` | `'json'`, `'yaml'`, `'python'` |

### `get_trials_dataframe(campaign_id) → Dict`

### `get_parameter_importance(campaign_id) → Dict`

### `save_research_artifact(campaign_id) → Optional[str]`

---

## MonitoringService

> Application health monitoring (instance-based).

**File:** `monitoring_service.py`

> **Note:** This service uses instance methods. A global `monitoring_service` singleton is available.

### `start_monitoring(interval_seconds) → None`

| Parameter          | Type  | Default | Description         |
| ------------------ | ----- | ------- | ------------------- |
| `interval_seconds` | `int` | `60`    | Collection interval |

### `stop_monitoring() → None`

### `get_current_metrics() → Dict`

### `get_recent_alerts(hours) → List[Dict]`

### `get_health_status() → Dict`

Returns `status` (`'healthy'`, `'degraded'`, `'unhealthy'`) and component details.

---

## NASService

> Neural Architecture Search campaign management.

**File:** `nas_service.py`

### `create_nas_campaign(name, dataset_id, search_space_config, search_algorithm, num_trials, max_epochs_per_trial) → Optional[Dict]`

| Parameter              | Type  | Default    | Description                             |
| ---------------------- | ----- | ---------- | --------------------------------------- |
| `search_algorithm`     | `str` | `'random'` | `'random'`, `'bayesian'`, `'evolution'` |
| `num_trials`           | `int` | `20`       | Total trials                            |
| `max_epochs_per_trial` | `int` | `10`       | Epochs per trial                        |

### `sample_architecture(search_space_config) → Dict`

### `compute_architecture_hash(architecture) → str`

SHA-256 hash for deduplication.

### `get_campaign_details(campaign_id) → Optional[Dict]`

### `get_best_architecture(campaign_id) → Optional[Dict]`

### `export_architecture(trial_id, format) → Optional[str]`

| Parameter | Type  | Default     | Description             |
| --------- | ----- | ----------- | ----------------------- |
| `format`  | `str` | `'pytorch'` | `'pytorch'` or `'json'` |

### `list_campaigns(limit, offset) → List[Dict]`

---

## NotificationService

> Multi-channel event notification routing.

**File:** `notification_service.py`

### `initialize(cls, email_config, redis_client) → None`

Call once at startup.

### `emit_event(event_type, user_id, data) → Dict`

Main entry point. Routes to toast, email (immediate/digest), and webhooks.

| Parameter    | Type             | Default  | Description                 |
| ------------ | ---------------- | -------- | --------------------------- |
| `event_type` | `str`            | required | e.g., `'training.complete'` |
| `user_id`    | `int`            | required | User to notify              |
| `data`       | `Dict[str, Any]` | required | Event data                  |

### `create_toast(message, notification_type, duration, link) → Dict`

### `success(message, link) → Dict`

### `error(message) → Dict`

### `warning(message) → Dict`

### `info(message, link) → Dict`

**Standalone functions:**

### `create_default_notification_preferences(user_id) → None`

### `get_error_suggestion(error_message) → str`

---

## SearchService

> Structured experiment search with query parsing and ranking.

**File:** `search_service.py`

### `search(session, query, user_id, limit) → Dict`

| Parameter | Type            | Default  | Description                                                       |
| --------- | --------------- | -------- | ----------------------------------------------------------------- |
| `session` | `Session`       | required | Database session                                                  |
| `query`   | `str`           | required | Query string (supports `tag:`, `accuracy:>`, `status:`, keywords) |
| `user_id` | `Optional[int]` | `None`   | For authorization                                                 |
| `limit`   | `int`           | `100`    | Max results                                                       |

**Returns:** Dict with `results`, `total_count`, `suggestions`.

### `save_search(session, user_id, name, query, is_pinned) → SavedSearch`

### `get_saved_searches(session, user_id) → List[SavedSearch]`

### `use_saved_search(session, saved_search_id, user_id) → Dict`

### `delete_saved_search(session, saved_search_id, user_id) → bool`

---

## SignalService

> Signal processing: FFT, spectrogram, features.

**File:** `signal_service.py`

### `compute_fft(signal_data, fs) → Tuple[ndarray, ndarray]`

Returns `(frequencies, magnitudes)`.

### `compute_spectrogram(signal_data, fs, nperseg) → Tuple[ndarray, ndarray, ndarray]`

Returns `(f, t, Sxx)`.

### `extract_basic_features(signal_data, fs) → Dict[str, float]`

Returns `rms`, `kurtosis`, `skewness`, `peak_value`, `mean`, `std`, `dominant_frequency`, `spectral_centroid`.

### `get_signal_with_features(dataset_id, signal_id) → Dict`

---

## TagService

> Experiment tag management and autocomplete.

**File:** `tag_service.py`

### `slugify(text) → str`

### `create_or_get_tag(session, name, color, user_id) → Tag`

### `add_tag_to_experiment(session, experiment_id, tag_name, user_id, color) → Dict`

### `remove_tag_from_experiment(session, experiment_id, tag_id) → Dict`

### `get_experiment_tags(session, experiment_id) → List[Tag]`

### `get_popular_tags(session, limit, min_usage) → List[Tag]`

### `suggest_tags(session, query, limit) → List[Tag]`

### `bulk_add_tags(session, experiment_ids, tag_names, user_id) → Dict`

### `bulk_remove_tags(session, experiment_ids, tag_ids) → Dict`

### `get_tag_statistics(session) → Dict`

---

## TestingService

> Programmatic test execution, coverage, and quality checks.

**File:** `testing_service.py`

### `run_pytest(test_path, markers, verbose, capture_output) → Dict`

### `run_coverage(test_path, source_path, min_coverage) → Dict`

| Parameter      | Type    | Default | Description       |
| -------------- | ------- | ------- | ----------------- |
| `min_coverage` | `float` | `80.0`  | Minimum threshold |

### `run_benchmarks(model_path, api_url, num_samples) → Dict`

### `parse_pytest_output(stdout) → Dict`

### `get_test_files() → List[Dict]`

### `validate_code_quality(path) → Dict`

### `get_recent_test_history(limit) → List[Dict]`

---

## WebhookService

> Webhook configuration CRUD and delivery management.

**File:** `webhook_service.py`

### `list_user_webhooks(user_id, include_inactive) → List[Dict]`

### `get_webhook(webhook_id, user_id) → Optional[Dict]`

### `create_webhook(user_id, provider_type, webhook_url, name, enabled_events, description, is_active, settings) → Optional[Dict]`

| Parameter        | Type        | Default  | Description                       |
| ---------------- | ----------- | -------- | --------------------------------- |
| `provider_type`  | `str`       | required | `'slack'`, `'teams'`, `'webhook'` |
| `enabled_events` | `List[str]` | required | Event types to subscribe          |

### `update_webhook(webhook_id, user_id, **updates) → Optional[Dict]`

### `delete_webhook(webhook_id, user_id) → bool`

### `toggle_webhook(webhook_id, user_id, is_active) → bool`

### `test_webhook(webhook_id, user_id) → Dict`

### `get_webhook_logs(webhook_id, user_id, limit) → List[Dict]`

### `get_webhook_stats(webhook_id, user_id) → Dict`

---

## XAIService

> Explainability: SHAP, LIME, Integrated Gradients, Grad-CAM.

**File:** `xai_service.py`

### `get_device() → torch.device`

### `generate_shap_explanation(model, signal, background_data, method, num_samples) → Dict`

| Parameter     | Type  | Default      | Description       |
| ------------- | ----- | ------------ | ----------------- |
| `method`      | `str` | `'gradient'` | SHAP method       |
| `num_samples` | `int` | `100`        | Number of samples |

### `generate_lime_explanation(model, signal, num_segments, num_samples, target_class) → Dict`

| Parameter      | Type            | Default | Description          |
| -------------- | --------------- | ------- | -------------------- |
| `num_segments` | `int`           | `20`    | Signal segments      |
| `num_samples`  | `int`           | `1000`  | Perturbation samples |
| `target_class` | `Optional[int]` | `None`  | Target class         |

### `generate_integrated_gradients(model, signal, baseline, steps) → Dict`

| Parameter  | Type               | Default | Description               |
| ---------- | ------------------ | ------- | ------------------------- |
| `baseline` | `Optional[Tensor]` | `None`  | Baseline (default: zeros) |
| `steps`    | `int`              | `50`    | Integration steps         |

### `generate_gradcam(model, signal, target_layer) → Dict`

| Parameter      | Type                  | Default | Description           |
| -------------- | --------------------- | ------- | --------------------- |
| `target_layer` | `Optional[nn.Module]` | `None`  | Auto-detected if None |

### `load_model(experiment_id) → Optional[nn.Module]`

### `get_model_prediction(model, signal) → Dict`

Returns `predicted_class`, `confidence`, `probabilities`.
