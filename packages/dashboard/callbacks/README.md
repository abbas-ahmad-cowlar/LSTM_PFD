# Callbacks

> Central callback layer connecting Dashboard UI layouts to backend services and database models.

## Overview

The `callbacks/` package contains 28 Dash callback modules that power the interactive behavior of the LSTM-PFD Dashboard. Each module follows a consistent `register_*_callbacks(app)` pattern and is registered centrally through `__init__.py`'s `register_all_callbacks()` function.

Callbacks act as the "glue" between the UI (layouts, IDB 2.1) and backend logic (services, IDB 2.2). They translate user interactions (button clicks, dropdown selections, URL changes) into service calls, database queries, and Celery task launches, then render the results back into Dash components.

## Architecture

```mermaid
graph TD
    subgraph Entry Point
        APP[app.py] -->|calls| RAC[register_all_callbacks]
    end

    subgraph __init__.py
        RAC -->|URL routing| DP[display_page callback]
        RAC -->|try/except import| CB1[register_home_callbacks]
        RAC -->|try/except import| CB2[register_data_generation_callbacks]
        RAC -->|try/except import| CBN[... 26 more modules]
    end

    subgraph Callback Module Pattern
        CB1 -->|@app.callback| SVC[Services Layer]
        CB1 -->|get_db_session| DB[(Database)]
        CB1 -->|.delay| CELERY[Celery Tasks]
    end

    DP -->|pathname match| LAYOUTS[Layout Functions]
```

### Registration Flow

1. `app.py` calls `register_all_callbacks(app)`
2. `__init__.py` registers the URL-routing callback (`display_page`)
3. Each callback module is imported inside a `try/except ImportError` block for resilience
4. Each module's `register_*_callbacks(app)` function decorates inner functions with `@app.callback`

### URL Routing

The `display_page` callback in `__init__.py` maps URL pathnames to layout functions:

| Pathname                   | Layout Function                            | Module                          |
| -------------------------- | ------------------------------------------ | ------------------------------- |
| `/`                        | `create_home_layout()`                     | `layouts.home`                  |
| `/data-generation`         | `create_data_generation_layout()`          | `layouts.data_generation`       |
| `/data-explorer`           | `create_data_explorer_layout()`            | `layouts.data_explorer`         |
| `/signal-viewer`           | `create_signal_viewer_layout()`            | `layouts.signal_viewer`         |
| `/datasets`                | `create_datasets_layout()`                 | `layouts.datasets`              |
| `/experiments`             | `create_experiments_layout()`              | `layouts.experiments`           |
| `/experiment/new`          | `create_experiment_wizard_layout()`        | `layouts.experiment_wizard`     |
| `/experiment/{id}/monitor` | `create_training_monitor_layout(id)`       | `layouts.training_monitor`      |
| `/experiment/{id}/results` | `create_experiment_results_layout(id)`     | `layouts.experiment_results`    |
| `/compare?ids=...`         | `create_experiment_comparison_layout(ids)` | `layouts.experiment_comparison` |
| `/xai`                     | `create_xai_dashboard_layout()`            | `layouts.xai_dashboard`         |
| `/system-health`           | `create_system_health_layout()`            | `layouts.system_health`         |
| `/hpo/campaigns`           | `create_hpo_campaigns_layout()`            | `layouts.hpo_campaigns`         |
| `/deployment`              | `create_deployment_layout()`               | `layouts.deployment`            |
| `/api-monitoring`          | `create_api_monitoring_layout()`           | `layouts.api_monitoring`        |
| `/evaluation`              | `create_evaluation_dashboard_layout()`     | `layouts.evaluation_dashboard`  |
| `/testing`                 | `create_testing_dashboard_layout()`        | `layouts.testing_dashboard`     |
| `/feature-engineering`     | `create_feature_engineering_layout()`      | `layouts.feature_engineering`   |
| `/visualization`           | `create_visualization_layout()`            | `layouts.visualization`         |
| `/nas`                     | `create_nas_dashboard_layout()`            | `layouts.nas_dashboard`         |
| `/settings`                | `create_settings_layout()`                 | `layouts.settings`              |

## Key Components

### Callback-Layout-Service Mapping

| Callback Module                  | Layout(s) Served                              | Service(s) / Backend Used                                                                      | Inner Callbacks                                                                                                                                                                                                                                                                                                |
| -------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `home_callbacks.py`              | Home (`/`)                                    | DB: `Experiment`, `Dataset`                                                                    | `update_home_dashboard`                                                                                                                                                                                                                                                                                        |
| `data_generation_callbacks.py`   | Data Generation (`/data-generation`)          | DB: `DatasetGeneration`; Celery tasks                                                          | `save_configuration`, `load_configuration`, `update_config_summary`, `start_generation`, `poll_generation_status`, `load_recent_generations`, `toggle_augmentation_settings`                                                                                                                                   |
| `data_explorer_callbacks.py`     | Data Explorer (`/data-explorer`)              | `DataService`, `DatasetService`                                                                | `load_datasets`, `update_class_distribution`, `update_summary_stats`, `load_feature_options`, `update_feature_distribution`, `calculate_projection`, `update_spectral_analysis`, `update_feature_comparison`                                                                                                   |
| `signal_viewer_callbacks.py`     | Signal Viewer (`/signal-viewer`)              | `SignalService`, `DatasetService`                                                              | Signal loading and visualization                                                                                                                                                                                                                                                                               |
| `datasets_callbacks.py`          | Datasets (`/datasets`)                        | `DatasetService`                                                                               | Dataset CRUD operations                                                                                                                                                                                                                                                                                        |
| `experiment_wizard_callbacks.py` | Experiment Wizard (`/experiment/new`)         | `DataService`; DB: `Dataset`, `Experiment`                                                     | Wizard step navigation, experiment creation                                                                                                                                                                                                                                                                    |
| `training_monitor_callbacks.py`  | Training Monitor (`/experiment/{id}/monitor`) | DB: `Experiment`, `TrainingRun`; Celery                                                        | `load_experiment_info`, `fetch_training_data`, `update_progress_indicators`, `update_metrics_display`, `update_loss_curve`, `update_accuracy_curve`, `update_lr_schedule`, `update_training_logs`                                                                                                              |
| `experiments_callbacks.py`       | Experiments List (`/experiments`)             | DB: `Experiment`, `ExperimentTag`                                                              | `load_experiments`, `load_experiments_summary`, `load_model_filter_options`, `handle_experiment_selection`, `navigate_to_comparison`, `toggle_comparison_cart`                                                                                                                                                 |
| `comparison_callbacks.py`        | Experiment Comparison (`/compare`)            | `ComparisonService`                                                                            | `render_tab_content`, `render_key_differences`, `toggle_share_modal`, `copy_link_to_clipboard`, `export_comparison_pdf`                                                                                                                                                                                        |
| `xai_callbacks.py`               | XAI Dashboard (`/xai`)                        | `XAIService`, `ExplanationCache`; DB: `Experiment`, `Signal`, `Dataset`, `Explanation`; Celery | `load_available_models`, `load_signals_for_model`, `generate_explanation`, `update_visualizations`, `load_cached_explanations`                                                                                                                                                                                 |
| `system_health_callbacks.py`     | System Health (`/system-health`)              | `monitoring_service`; DB: `SystemLog`                                                          | System metrics, log viewer                                                                                                                                                                                                                                                                                     |
| `hpo_callbacks.py`               | HPO Campaigns (`/hpo/campaigns`)              | `HPOService`; DB: `Dataset`                                                                    | `toggle_hpo_modal`, `update_campaigns_list`, `launch_hpo_campaign`, `create_and_launch_campaign`, `stop_campaign`, `resume_campaign`, `toggle_export_modal`, `update_export_preview`, `download_export`, `toggle_viz_card`, `update_parallel_coords`, `update_param_importance`, `update_optimization_history` |
| `deployment_callbacks.py`        | Deployment (`/deployment`)                    | `DeploymentService`; DB: `Experiment`                                                          | `load_completed_experiments`, `display_model_info`, `quantize_model`, `export_onnx`, `optimize_model`, `run_benchmark`                                                                                                                                                                                         |
| `api_monitoring_callbacks.py`    | API Monitoring (`/api-monitoring`)            | `APIMonitoringService`                                                                         | Endpoint metrics display                                                                                                                                                                                                                                                                                       |
| `evaluation_callbacks.py`        | Evaluation (`/evaluation`)                    | Celery: `evaluation_tasks`; DB: `Experiment`                                                   | `load_experiments`, `generate_roc_analysis`, `analyze_errors`, `compare_architectures`                                                                                                                                                                                                                         |
| `testing_callbacks.py`           | Testing (`/testing`)                          | Celery: `testing_tasks`                                                                        | `run_tests`, `run_coverage`, `run_benchmarks`, `run_quality_checks`                                                                                                                                                                                                                                            |
| `feature_callbacks.py`           | Feature Engineering (`/feature-engineering`)  | `DatasetService`                                                                               | Feature extraction and analysis                                                                                                                                                                                                                                                                                |
| `visualization_callbacks.py`     | Visualization (`/visualization`)              | DB: `Dataset`, `Experiment`                                                                    | `load_datasets`, `load_experiments`, `toggle_embedding_params`, `generate_embedding`, `load_signal_options`, `generate_signal_visualization`, `generate_feature_visualization`, `update_layer_options`, `load_sample_options`, `generate_model_visualization`                                                  |
| `nas_callbacks.py`               | NAS Dashboard (`/nas`)                        | `NASService`; DB: `Dataset`                                                                    | NAS campaign management                                                                                                                                                                                                                                                                                        |
| `notification_callbacks.py`      | Settings (`/settings`)                        | `NotificationService`; DB: `NotificationPreference`, `EmailLog`                                | `load_notification_preferences`, `update_notification_preferences`, `save_email_configuration`, `send_test_notification`, `load_notification_history`                                                                                                                                                          |
| `email_digest_callbacks.py`      | Settings (`/settings`)                        | DB: `EmailDigestQueue`, `User`                                                                 | `update_digest_stats`, `load_digest_queue_table`, `load_digest_history`, `trigger_digest_processing`, `populate_user_filter`                                                                                                                                                                                   |
| `api_key_callbacks.py`           | Settings (`/settings`)                        | `APIKeyService`                                                                                | `load_api_keys_table`, `toggle_generate_modal`, `generate_new_key`, `handle_revoke_key`, `update_api_stats_summary`, `update_api_usage_timeline`, `update_top_keys_chart`, `update_endpoints_chart`, `update_api_usage_detail_table`                                                                           |
| `webhook_callbacks.py`           | Settings (`/settings`)                        | `WebhookService`                                                                               | `load_webhooks_table`, `manage_webhook_modal`, `save_webhook`, `test_webhook_handler`, `toggle_webhook_handler`, `manage_delete_webhook_modal`, `manage_webhook_details_modal`                                                                                                                                 |
| `tag_callbacks.py`               | Settings, Experiments                         | `TagService`; DB: `Tag`                                                                        | Tag CRUD and assignment                                                                                                                                                                                                                                                                                        |
| `saved_search_callbacks.py`      | Settings, Experiments                         | `SearchService`; DB direct                                                                     | Saved search management                                                                                                                                                                                                                                                                                        |
| `profile_callbacks.py`           | Settings (`/settings`)                        | DB: `User`                                                                                     | `load_profile_data`, `update_profile`                                                                                                                                                                                                                                                                          |
| `security_callbacks.py`          | Settings (`/settings`)                        | DB: `User`, `SessionLog`, `LoginHistory`; `middleware.auth`                                    | `check_password_strength`, `change_password`, `load_active_sessions`, `load_login_history`, `manage_2fa_setup`, `verify_2fa_code`                                                                                                                                                                              |
| `mat_import_callbacks.py`        | Data Generation (`/data-generation`)          | DB: `DatasetImport`; Celery tasks                                                              | `handle_file_upload`, `update_import_summary`, `clear_uploaded_files`, `start_import`, `poll_import_status`, `load_recent_imports`                                                                                                                                                                             |

### Functional Groups

| Group                  | Modules                                                                                            | Purpose                                         |
| ---------------------- | -------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| **Data Pipeline**      | `data_generation`, `mat_import`, `data_explorer`, `signal_viewer`, `datasets`                      | Data ingestion, exploration, and management     |
| **ML Workflow**        | `experiment_wizard`, `training_monitor`, `experiments`, `comparison`, `evaluation`                 | Experiment creation, monitoring, and analysis   |
| **Advanced Analytics** | `xai`, `hpo`, `nas`, `feature`, `visualization`                                                    | Explainability, optimization, and visualization |
| **Operations**         | `system_health`, `deployment`, `api_monitoring`, `testing`                                         | System health, deployment, and testing          |
| **User & Settings**    | `api_key`, `webhook`, `notification`, `email_digest`, `tag`, `saved_search`, `profile`, `security` | User preferences, integrations, and security    |

## Dependencies

- **Requires:**
  - `layouts/` — Layout functions that produce the Dash component trees (IDB 2.1)
  - `services/` — Business logic and data access services (IDB 2.2)
  - `tasks/` — Celery async task definitions (IDB 2.4)
  - `database/` — SQLAlchemy session management (IDB 4.1)
  - `models/` — ORM model classes (IDB 4.1)
  - `utils/` — Logger, constants, auth utilities (IDB 6.0)
  - `middleware/` — Authentication middleware (`auth.py`)
- **Provides:**
  - All interactive behavior for the Dashboard UI
  - URL-based page routing via `display_page`

## Related Documentation

- [Layouts / UI Guide](../layouts/README.md) — IDB 2.1
- [Services Catalog](../services/README.md) — IDB 2.2
- [Async Tasks Guide](../tasks/README.md) — IDB 2.4
- [Database Schema](../database/README.md) — IDB 4.1
- [CALLBACK_GUIDE.md](./CALLBACK_GUIDE.md) — Developer guide for writing new callbacks
