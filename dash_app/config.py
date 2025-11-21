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
