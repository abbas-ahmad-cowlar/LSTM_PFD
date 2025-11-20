"""
API Configuration

Configuration settings for the FastAPI application.

Author: LSTM_PFD Team
Date: 2025-11-20
"""

from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """API configuration settings."""

    # API metadata
    app_name: str = "LSTM_PFD Bearing Fault Diagnosis API"
    app_version: str = "1.0.0"
    app_description: str = "REST API for bearing fault diagnosis using deep learning"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 4

    # Model settings
    model_path: str = "checkpoints/best_model.pth"
    model_type: str = "torch"  # 'torch', 'onnx', 'quantized'
    device: str = "cuda"  # 'cuda' or 'cpu'
    batch_size: int = 32

    # Inference settings
    use_amp: bool = False  # Automatic mixed precision
    num_threads: int = 4  # CPU threads

    # API limits
    max_batch_size: int = 128
    max_signal_length: int = 102400
    request_timeout: int = 30  # seconds

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/api.log"

    # CORS
    cors_origins: list = ["*"]

    # Security (optional)
    api_key: Optional[str] = None
    require_authentication: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()
