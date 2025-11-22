"""
Configuration management for the Dash application.
Loads settings from environment variables following 12-factor app methodology.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DASH_APP_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"

# Environment
ENV = os.getenv("ENV", "development")
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# Database Configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://lstm_user:lstm_password@localhost:5432/lstm_dashboard"
)

# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# Celery Configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

# File Storage Configuration
STORAGE_DATASETS_DIR = STORAGE_DIR / "datasets"
STORAGE_MODELS_DIR = STORAGE_DIR / "models"
STORAGE_RESULTS_DIR = STORAGE_DIR / "results"
STORAGE_UPLOADS_DIR = STORAGE_DIR / "uploads"

# Ensure storage directories exist
for dir_path in [STORAGE_DATASETS_DIR, STORAGE_MODELS_DIR, STORAGE_RESULTS_DIR, STORAGE_UPLOADS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dash App Configuration
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8050"))

# Security
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO" if not DEBUG else "DEBUG")
LOG_FILE = DASH_APP_DIR / "app.log"

# Cache TTL (seconds)
CACHE_TTL_SHORT = 300  # 5 minutes
CACHE_TTL_MEDIUM = 600  # 10 minutes
CACHE_TTL_LONG = 3600  # 1 hour
CACHE_TTL_DAY = 86400  # 24 hours

# Task Configuration
MAX_TRAINING_DURATION = 7200  # 2 hours max per training task
TASK_POLL_INTERVAL = 2  # seconds between progress polls

# Model Configuration
FAULT_CLASSES = [
    "normal", "ball_fault", "inner_race", "outer_race", "combined",
    "imbalance", "misalignment", "oil_whirl", "cavitation", "looseness", "oil_deficiency"
]
NUM_CLASSES = len(FAULT_CLASSES)

# Visualization Configuration
PLOT_THEME = "plotly_white"
COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8"
]

# Phase 0-10 Integration Paths
PHASE_0_DATA_PATH = BASE_DIR / "data"
PHASE_0_CACHE_PATH = PHASE_0_DATA_PATH / "processed" / "signals_cache.h5"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

# GPU Configuration
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# API Key & Rate Limiting Configuration (Feature #1)
API_KEY_RATE_LIMIT_DEFAULT = int(os.getenv("API_KEY_RATE_LIMIT_DEFAULT", "1000"))  # Requests per hour
API_KEY_RATE_LIMIT_WINDOW = 3600  # 1 hour in seconds
API_KEY_EXPIRY_REDIS = 7200  # 2 hours (for cleanup)
RATE_LIMIT_FAIL_OPEN = os.getenv("RATE_LIMIT_FAIL_OPEN", "True").lower() == "true"  # Allow requests if Redis down

# Email Notification Configuration (Feature #3)
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "False").lower() == "true"
EMAIL_PROVIDER = os.getenv("EMAIL_PROVIDER", "sendgrid")  # 'sendgrid', 'smtp'
EMAIL_FROM = os.getenv("EMAIL_FROM", "noreply@lstm-dashboard.com")
EMAIL_FROM_NAME = os.getenv("EMAIL_FROM_NAME", "LSTM Bearing Fault Diagnosis")

# SendGrid Configuration
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")

# SMTP Configuration (alternative to SendGrid)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

# Email Rate Limiting
EMAIL_RATE_LIMIT = int(os.getenv("EMAIL_RATE_LIMIT", "100"))  # Max emails per minute

# Webhook Notification Configuration (Feature #4)
NOTIFICATIONS_SLACK_ENABLED = os.getenv("NOTIFICATIONS_SLACK_ENABLED", "True").lower() == "true"
NOTIFICATIONS_TEAMS_ENABLED = os.getenv("NOTIFICATIONS_TEAMS_ENABLED", "True").lower() == "true"
NOTIFICATIONS_WEBHOOK_ENABLED = os.getenv("NOTIFICATIONS_WEBHOOK_ENABLED", "True").lower() == "true"

# Slack Configuration
SLACK_RATE_LIMIT_PER_WEBHOOK = int(os.getenv("SLACK_RATE_LIMIT_PER_WEBHOOK", "1"))  # 1 msg/sec
SLACK_RETRY_ATTEMPTS = int(os.getenv("SLACK_RETRY_ATTEMPTS", "3"))
SLACK_TIMEOUT_SECONDS = int(os.getenv("SLACK_TIMEOUT_SECONDS", "10"))

# Teams Configuration
TEAMS_RATE_LIMIT_PER_WEBHOOK = int(os.getenv("TEAMS_RATE_LIMIT_PER_WEBHOOK", "2"))  # 2 msg/sec
TEAMS_RETRY_ATTEMPTS = int(os.getenv("TEAMS_RETRY_ATTEMPTS", "3"))
TEAMS_TIMEOUT_SECONDS = int(os.getenv("TEAMS_TIMEOUT_SECONDS", "10"))

# Custom Webhook Configuration
WEBHOOK_CUSTOM_TIMEOUT_SECONDS = int(os.getenv("WEBHOOK_CUSTOM_TIMEOUT_SECONDS", "5"))
WEBHOOK_CUSTOM_RETRY_ATTEMPTS = int(os.getenv("WEBHOOK_CUSTOM_RETRY_ATTEMPTS", "2"))

# Webhook Feature Toggles
NOTIFICATIONS_ENABLE_RICH_FORMATTING = os.getenv("NOTIFICATIONS_ENABLE_RICH_FORMATTING", "True").lower() == "true"
NOTIFICATIONS_ENABLE_MENTIONS = os.getenv("NOTIFICATIONS_ENABLE_MENTIONS", "True").lower() == "true"

# Feature #5: Tags & Search Configuration
FEATURE_TAGS_ENABLED = os.getenv("FEATURE_TAGS_ENABLED", "True").lower() == "true"
FEATURE_SEARCH_ENABLED = os.getenv("FEATURE_SEARCH_ENABLED", "True").lower() == "true"
FEATURE_SAVED_SEARCHES_ENABLED = os.getenv("FEATURE_SAVED_SEARCHES_ENABLED", "True").lower() == "true"
FEATURE_TAG_SUGGESTIONS_ENABLED = os.getenv("FEATURE_TAG_SUGGESTIONS_ENABLED", "True").lower() == "true"

# Search Engine Configuration
SEARCH_ENGINE = os.getenv("SEARCH_ENGINE", "postgres")  # Options: 'postgres', 'elasticsearch'
SEARCH_MAX_RESULTS = int(os.getenv("SEARCH_MAX_RESULTS", "100"))

# Tag System Configuration
TAGS_MAX_PER_EXPERIMENT = int(os.getenv("TAGS_MAX_PER_EXPERIMENT", "10"))
TAGS_MAX_LENGTH = int(os.getenv("TAGS_MAX_LENGTH", "50"))
TAGS_CASE_SENSITIVE = os.getenv("TAGS_CASE_SENSITIVE", "False").lower() == "true"
TAGS_ALLOW_SPACES = os.getenv("TAGS_ALLOW_SPACES", "False").lower() == "true"
TAGS_RESERVED_WORDS = ["all", "none", "system"]

# Performance Tuning
SEARCH_DEBOUNCE_MS = int(os.getenv("SEARCH_DEBOUNCE_MS", "300"))  # Delay before search
TAG_AUTOCOMPLETE_MIN_CHARS = int(os.getenv("TAG_AUTOCOMPLETE_MIN_CHARS", "2"))
TAG_AUTOCOMPLETE_MAX_RESULTS = int(os.getenv("TAG_AUTOCOMPLETE_MAX_RESULTS", "10"))
SEARCH_CACHE_TTL_SECONDS = int(os.getenv("SEARCH_CACHE_TTL_SECONDS", "60"))
