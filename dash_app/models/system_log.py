"""System log model for tracking events."""
from sqlalchemy import Column, String, JSON, ForeignKey
from models.base import BaseModel


class SystemLog(BaseModel):
    """System event log model."""
    __tablename__ = 'system_logs'

    user_id = Column(ForeignKey('users.id'))
    action = Column(String(100), nullable=False, index=True)
    status = Column(String(50), nullable=False)  # success, error, warning
    details = Column(JSON)  # Additional event details
    error_message = Column(String(1000))

    def __repr__(self):
        return f"<SystemLog(action='{self.action}', status='{self.status}')>"
