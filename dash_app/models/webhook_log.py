"""Webhook log model for tracking webhook delivery status (Feature #4)."""
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from models.base import BaseModel
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


class WebhookLog(BaseModel):
    """
    Log webhook delivery attempts for debugging and monitoring.
    Separate from email_logs for modularity (each channel has its own logs).
    """
    __tablename__ = 'webhook_logs'

    webhook_config_id = Column(
        Integer,
        ForeignKey('webhook_configurations.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    user_id = Column(
        Integer,
        ForeignKey('users.id', ondelete='SET NULL'),
        nullable=True,
        index=True
    )

    event_type = Column(String(50), nullable=False, index=True)
    provider_type = Column(String(50), nullable=False)

    # Request details
    webhook_url = Column(Text, nullable=False)  # Logged for debugging (can't rely on config if deleted)
    payload = Column(JSONB, nullable=True)  # Full JSON payload sent

    # Response details
    status = Column(String(20), nullable=False, index=True)  # 'sent', 'failed', 'rate_limited'
    http_status_code = Column(Integer, nullable=True)  # 200, 429, 500, etc.
    response_body = Column(Text, nullable=True)  # Provider response (if error)
    error_message = Column(Text, nullable=True)

    retry_count = Column(Integer, default=0)
    sent_at = Column(DateTime, nullable=True)

    # Relationships
    webhook_config = relationship("WebhookConfiguration", backref="logs")
    user = relationship("User", backref="webhook_logs")

    # Performance indexes
    # Note: webhook_config_id (FK), user_id (index=True), event_type (index=True),
    #       status (index=True) already have column-level indexes
    __table_args__ = (
        Index('ix_webhook_logs_sent_at', 'sent_at'),
        Index('ix_webhook_logs_created_at', 'created_at'),
        # Composite indexes removed - log tables should minimize indexes for write performance
    )

    def __repr__(self):
        return f"<WebhookLog(id={self.id}, provider='{self.provider_type}', status='{self.status}', event='{self.event_type}')>"


# Webhook status constants
class WebhookStatus:
    """Standard webhook delivery statuses."""
    SENT = 'sent'
    FAILED = 'failed'
    RATE_LIMITED = 'rate_limited'
    TIMEOUT = 'timeout'
    INVALID_URL = 'invalid_url'
    PROVIDER_ERROR = 'provider_error'
