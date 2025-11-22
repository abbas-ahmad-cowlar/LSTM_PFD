"""User model for authentication (Phase 11D)."""
from sqlalchemy import Column, String, Boolean
from sqlalchemy.orm import relationship
from models.base import BaseModel
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


class User(BaseModel):
    """User account model."""
    __tablename__ = 'users'

    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=True)  # For API key feature integration
    role = Column(String(50), default='user')  # user, admin
    is_active = Column(Boolean, default=True)

    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")

    # Note: Password hashing and authentication logic will be added in Phase 11D

    def __repr__(self):
        return f"<User(username='{self.username}', role='{self.role}')>"
