"""Notification preference model for user notification settings."""
from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from models.base import BaseModel
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


class NotificationPreference(BaseModel):
    """User notification preferences for different event types."""
    __tablename__ = 'notification_preferences'

    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)

    # Channel preferences (true = enabled)
    email_enabled = Column(Boolean, default=True)
    in_app_enabled = Column(Boolean, default=True)  # Toast notifications
    slack_enabled = Column(Boolean, default=False)  # Future feature
    webhook_enabled = Column(Boolean, default=False)  # Future feature

    # Email-specific settings
    email_frequency = Column(String(20), default='immediate')  # 'immediate', 'digest_daily', 'digest_weekly'

    # Relationships
    user = relationship("User", backref="notification_preferences")

    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'event_type', name='uq_user_event_type'),
    )

    def __repr__(self):
        return f"<NotificationPreference(user_id={self.user_id}, event='{self.event_type}', email={self.email_enabled})>"


# Event types constants
class EventType:
    """Standard event types for notifications."""
    TRAINING_STARTED = 'training.started'
    TRAINING_COMPLETE = 'training.complete'
    TRAINING_FAILED = 'training.failed'
    TRAINING_PAUSED = 'training.paused'
    TRAINING_RESUMED = 'training.resumed'

    HPO_CAMPAIGN_STARTED = 'hpo.campaign_started'
    HPO_TRIAL_COMPLETE = 'hpo.trial_complete'
    HPO_CAMPAIGN_COMPLETE = 'hpo.campaign_complete'
    HPO_CAMPAIGN_FAILED = 'hpo.campaign_failed'

    ACCURACY_MILESTONE = 'accuracy.milestone'
    MODEL_DEPLOYED = 'model.deployed'
    SYSTEM_MAINTENANCE = 'system.maintenance'

    @classmethod
    def get_default_preferences(cls):
        """
        Get default notification preferences for new users.

        Returns:
            List of (event_type, email_enabled, frequency) tuples
        """
        return [
            (cls.TRAINING_STARTED, False, 'immediate'),  # Too noisy
            (cls.TRAINING_COMPLETE, True, 'immediate'),
            (cls.TRAINING_FAILED, True, 'immediate'),
            (cls.TRAINING_PAUSED, False, 'immediate'),
            (cls.TRAINING_RESUMED, False, 'immediate'),

            (cls.HPO_CAMPAIGN_STARTED, False, 'immediate'),
            (cls.HPO_TRIAL_COMPLETE, False, 'immediate'),  # Would send 100+ emails
            (cls.HPO_CAMPAIGN_COMPLETE, True, 'immediate'),
            (cls.HPO_CAMPAIGN_FAILED, True, 'immediate'),

            (cls.ACCURACY_MILESTONE, False, 'immediate'),  # Optional feature
            (cls.MODEL_DEPLOYED, False, 'immediate'),  # Future feature
            (cls.SYSTEM_MAINTENANCE, True, 'immediate'),
        ]
