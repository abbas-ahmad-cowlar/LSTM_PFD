"""Login history tracking model for user security (Phase 6, Feature 3)."""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean, Index
from datetime import datetime
from models.base import BaseModel


class LoginHistory(BaseModel):
    """
    Model for tracking login attempts and history.

    Attributes:
        user_id: Foreign key to User model
        ip_address: IP address of the login attempt
        user_agent: Browser/client user agent string
        location: Geographic location (City, Country)
        login_method: Authentication method used (password, 2fa, oauth)
        success: Whether the login attempt was successful
        failure_reason: Reason for failure if login was unsuccessful
        timestamp: When the login attempt occurred
    """
    __tablename__ = "login_history"

    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    ip_address = Column(String(45))  # Supports both IPv4 and IPv6
    user_agent = Column(String(500))
    location = Column(String(200))  # City, Country
    login_method = Column(String(50))  # password, 2fa, oauth, api_key
    success = Column(Boolean, default=True)
    failure_reason = Column(String(200), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Indexes for performance
    __table_args__ = (
        Index('ix_login_history_user_id', 'user_id'),
        Index('ix_login_history_timestamp', 'timestamp'),
        Index('ix_login_history_success', 'success'),
    )

    def __repr__(self):
        return f"<LoginHistory(user_id={self.user_id}, success={self.success}, timestamp={self.timestamp})>"
