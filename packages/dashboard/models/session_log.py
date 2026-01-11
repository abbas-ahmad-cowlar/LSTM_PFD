"""Session tracking model for user security (Phase 6, Feature 3)."""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Index
from sqlalchemy.orm import relationship
from datetime import datetime
from models.base import BaseModel


class SessionLog(BaseModel):
    """
    Model for tracking user sessions.

    Attributes:
        user_id: Foreign key to User model
        session_token: Unique token identifying the session
        ip_address: IP address of the session
        user_agent: Browser/client user agent string
        device_type: Type of device (desktop, mobile, tablet)
        browser: Browser name and version
        location: Geographic location (City, Country)
        last_active: Timestamp of last activity
        is_active: Whether the session is currently active
        logged_out_at: Timestamp when session was terminated
    """
    __tablename__ = "session_logs"

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_token = Column(String(255), unique=True, nullable=False, index=True)
    ip_address = Column(String(45))  # Supports both IPv4 and IPv6
    user_agent = Column(String(500))
    device_type = Column(String(50))  # desktop, mobile, tablet
    browser = Column(String(100))
    location = Column(String(200))  # City, Country
    last_active = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    logged_out_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="sessions")

    # Indexes for performance
    __table_args__ = (
        Index('ix_session_logs_user_id', 'user_id'),
        Index('ix_session_logs_session_token', 'session_token'),
        Index('ix_session_logs_is_active', 'is_active'),
    )

    def __repr__(self):
        return f"<SessionLog(user_id={self.user_id}, device='{self.device_type}', active={self.is_active})>"
