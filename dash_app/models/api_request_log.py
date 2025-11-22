"""API Request Log model for tracking API usage and performance."""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Text, ForeignKey
from sqlalchemy.orm import relationship
from models.base import BaseModel
from datetime import datetime


class APIRequestLog(BaseModel):
    """
    API Request Log model.

    Tracks all API requests for monitoring and analytics.
    """
    __tablename__ = 'api_request_logs'

    # Request details
    endpoint = Column(String(255), nullable=False, index=True)
    method = Column(String(10), nullable=False)  # GET, POST, etc.
    status_code = Column(Integer, nullable=False, index=True)

    # Timing
    request_time = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    response_time_ms = Column(Float, nullable=False)

    # Size
    request_size_bytes = Column(Integer, default=0)
    response_size_bytes = Column(Integer, default=0)

    # Client info
    ip_address = Column(String(45))  # IPv6 max length
    user_agent = Column(Text)

    # API key (if authenticated)
    api_key_id = Column(Integer, ForeignKey('api_keys.id'), nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)

    # Payload samples (limited size for debugging)
    request_payload_sample = Column(JSON, nullable=True)
    response_payload_sample = Column(JSON, nullable=True)

    # Relationships
    api_key = relationship("APIKey", back_populates="request_logs")

    def __repr__(self):
        return f"<APIRequestLog(endpoint='{self.endpoint}', status={self.status_code}, time={self.response_time_ms}ms)>"


class APIMetricsSummary(BaseModel):
    """
    Aggregated API metrics summary.

    Pre-computed statistics for faster dashboard loading.
    """
    __tablename__ = 'api_metrics_summary'

    # Time window
    period_start = Column(DateTime, nullable=False, index=True)
    period_end = Column(DateTime, nullable=False)
    aggregation_type = Column(String(20), nullable=False)  # 'hourly', 'daily', 'weekly'

    # Metrics
    total_requests = Column(Integer, default=0)
    successful_requests = Column(Integer, default=0)
    failed_requests = Column(Integer, default=0)

    # Latency stats
    avg_response_time_ms = Column(Float)
    p50_response_time_ms = Column(Float)
    p95_response_time_ms = Column(Float)
    p99_response_time_ms = Column(Float)

    # Throughput
    requests_per_second = Column(Float)

    # Errors
    error_rate = Column(Float)  # Percentage

    # Per endpoint breakdown (JSON)
    endpoint_metrics = Column(JSON)

    def __repr__(self):
        return f"<APIMetricsSummary(period={self.period_start}, type={self.aggregation_type})>"
