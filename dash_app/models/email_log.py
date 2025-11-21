"""Email log model for tracking sent emails (audit trail + debugging)."""
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, JSON, Index
from sqlalchemy.orm import relationship
from models.base import BaseModel
from datetime import datetime


class EmailLog(BaseModel):
    """Email log model for audit trail and debugging."""
    __tablename__ = 'email_logs'

    user_id = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True)
    recipient_email = Column(String(255), nullable=False)

    event_type = Column(String(50), nullable=False, index=True)
    subject = Column(String(255), nullable=False)
    template_name = Column(String(100), nullable=False)

    # Metadata
    event_data = Column(JSON)  # Store event details (experiment_id, accuracy, etc.)

    # Sending details
    provider = Column(String(50))  # 'sendgrid', 'ses', 'smtp'
    message_id = Column(String(255))  # Provider's message ID (for tracking)

    status = Column(String(20), nullable=False, index=True)  # 'sent', 'failed', 'bounced', 'pending'
    error_message = Column(Text)  # If status = 'failed'

    sent_at = Column(DateTime)
    delivered_at = Column(DateTime)  # From webhook (if provider supports)
    opened_at = Column(DateTime)  # From tracking pixel (optional)
    clicked_at = Column(DateTime)  # From link tracking (optional)

    retry_count = Column(Integer, default=0)

    # Relationships
    user = relationship("User", backref="email_logs")

    # Indexes
    __table_args__ = (
        Index('idx_email_logs_sent_at', 'sent_at'),
    )

    def __repr__(self):
        return f"<EmailLog(recipient='{self.recipient_email}', event='{self.event_type}', status='{self.status}')>"


class EmailStatus:
    """Email status constants."""
    PENDING = 'pending'
    SENT = 'sent'
    DELIVERED = 'delivered'
    FAILED = 'failed'
    BOUNCED = 'bounced'
    OPENED = 'opened'
    CLICKED = 'clicked'
