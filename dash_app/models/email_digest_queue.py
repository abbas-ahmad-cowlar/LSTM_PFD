"""Email digest queue model for daily/weekly digest emails."""
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean, JSON, Index
from sqlalchemy.orm import relationship
from models.base import BaseModel
from datetime import datetime
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


class EmailDigestQueue(BaseModel):
    """Queue for batching notifications into digest emails."""
    __tablename__ = 'email_digest_queue'

    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    event_type = Column(String(50), nullable=False)
    event_data = Column(JSON, nullable=False)
    scheduled_for = Column(DateTime, nullable=False, index=True)  # When to send digest
    included_in_digest = Column(Boolean, default=False, index=True)  # Has been included in sent digest

    # Relationships
    user = relationship("User", backref="digest_queue")

    # Indexes - partial index for pending items
    __table_args__ = (
        Index('idx_digest_queue_scheduled', 'scheduled_for'),
    )

    def __repr__(self):
        return f"<EmailDigestQueue(user_id={self.user_id}, event='{self.event_type}', scheduled={self.scheduled_for})>"
