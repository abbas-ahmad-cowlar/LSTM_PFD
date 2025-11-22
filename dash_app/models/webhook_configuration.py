"""Webhook configuration model for Slack/Teams integrations (Feature #4)."""
from sqlalchemy import Column, Integer, String, Boolean, Text, ForeignKey, UniqueConstraint, DateTime, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from models.base import BaseModel
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


class WebhookConfiguration(BaseModel):
    """
    Webhook configuration for multi-provider notifications (Slack, Teams, custom).
    Allows users to receive ML experiment notifications in team channels.
    """
    __tablename__ = 'webhook_configurations'

    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)

    # Provider info
    provider_type = Column(String(50), nullable=False, index=True)  # 'slack', 'teams', 'webhook'
    webhook_url = Column(Text, nullable=False)  # Provider-specific webhook URL

    # User-provided metadata
    name = Column(String(200), nullable=True)  # e.g., "#ml-experiments channel"
    description = Column(Text, nullable=True)

    # Configuration
    is_active = Column(Boolean, default=True, index=True)

    # Event routing (JSON array of enabled events)
    # Example: ["training.complete", "training.failed", "hpo.campaign_complete"]
    enabled_events = Column(JSONB, default=list, nullable=False)

    # Provider-specific settings (flexible JSON for different providers)
    # Example for Slack: {"mention_on_failure": true, "mention_user": "@abbas"}
    settings = Column(JSONB, default=dict, nullable=False)

    # Status tracking
    last_used_at = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)
    consecutive_failures = Column(Integer, default=0)

    # Relationships
    user = relationship("User", backref="webhook_configurations")

    # Constraints and performance indexes
    # Note: user_id, provider_type, is_active already have column-level indexes
    __table_args__ = (
        UniqueConstraint('user_id', 'webhook_url', name='uq_user_webhook_url'),
        Index('ix_webhook_configurations_created_at', 'created_at'),
        # Composite indexes removed - columns already individually indexed
    )

    def __repr__(self):
        return f"<WebhookConfiguration(id={self.id}, user={self.user_id}, provider='{self.provider_type}', active={self.is_active})>"

    def to_dict(self):
        """Convert to dictionary with safe representation."""
        data = super().to_dict()
        # Mask webhook URL for security (show only last 10 chars)
        if self.webhook_url and len(self.webhook_url) > 10:
            data['webhook_url_masked'] = f"...{self.webhook_url[-10:]}"
        return data
