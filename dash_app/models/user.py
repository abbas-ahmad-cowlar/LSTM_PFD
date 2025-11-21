"""User model for authentication (Phase 11D)."""
from sqlalchemy import Column, String, Boolean
from models.base import BaseModel


class User(BaseModel):
    """User account model."""
    __tablename__ = 'users'

    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False)
    role = Column(String(50), default='user')  # user, admin
    is_active = Column(Boolean, default=True)

    # Note: Password hashing and authentication logic will be added in Phase 11D

    def __repr__(self):
        return f"<User(username='{self.username}', role='{self.role}')>"
