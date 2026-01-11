"""System log model for tracking events."""
from sqlalchemy import Column, String, JSON, ForeignKey, Index
from models.base import BaseModel
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


class SystemLog(BaseModel):
    """System event log model."""
    __tablename__ = 'system_logs'

    user_id = Column(ForeignKey('users.id'), index=True)
    action = Column(String(100), nullable=False, index=True)
    status = Column(String(50), nullable=False, index=True)  # success, error, warning
    details = Column(JSON)  # Additional event details
    error_message = Column(String(1000))

    # Indexes - optimized for filtering and search
    __table_args__ = (
        # Composite index for time-based status filtering
        Index('idx_system_log_time_status', 'created_at', 'status'),
        # Composite index for user activity logs
        Index('idx_system_log_user_time', 'user_id', 'created_at'),
    )

    def __repr__(self):
        return f"<SystemLog(action='{self.action}', status='{self.status}')>"
