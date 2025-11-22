"""
API Key model for authentication and rate limiting (Feature #1).
Implements secure API key management with bcrypt hashing.
"""
from sqlalchemy import Column, Integer, String, Boolean, TIMESTAMP, ARRAY, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime

from models.base import Base
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


class APIKey(Base):
    """
    API Key model for programmatic access authentication.

    Security:
        - Keys are hashed with bcrypt (cost factor 12)
        - Only prefix is stored for display
        - Full key shown ONCE at creation
        - Supports expiration and revocation

    Attributes:
        id: Primary key
        user_id: Foreign key to users table
        key_hash: bcrypt hash of the full API key
        name: User-provided descriptive name
        prefix: First 20 chars for display (e.g., "sk_live_abc12345678")
        scopes: Array of permissions ['read', 'write']
        rate_limit: Requests per hour (default 1000)
        last_used_at: Timestamp of last successful authentication
        is_active: Whether key is active (revoked keys are inactive)
        expires_at: Expiration timestamp (NULL = never expires)
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    __tablename__ = 'api_keys'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True)
    name = Column(String(100), nullable=False)
    prefix = Column(String(20), nullable=False, index=True)  # Indexed for fast lookup
    scopes = Column(ARRAY(String), default=['read', 'write'])
    rate_limit = Column(Integer, default=1000)
    last_used_at = Column(TIMESTAMP)
    is_active = Column(Boolean, default=True, index=True)
    expires_at = Column(TIMESTAMP)
    created_at = Column(TIMESTAMP, default=func.now())
    updated_at = Column(TIMESTAMP, default=func.now(), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="api_keys")
    usage_records = relationship("APIUsage", back_populates="api_key", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<APIKey(id={self.id}, name='{self.name}', prefix='{self.prefix}', active={self.is_active})>"

    def to_dict(self, include_hash=False):
        """
        Convert to dictionary for API responses.

        Args:
            include_hash: If True, include key_hash (USE WITH CAUTION)

        Returns:
            Dictionary representation
        """
        data = {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'prefix': self.prefix,
            'scopes': self.scopes,
            'rate_limit': self.rate_limit,
            'is_active': self.is_active,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
        }

        if include_hash:
            data['key_hash'] = self.key_hash

        return data

    def is_expired(self):
        """Check if API key has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at

    def is_valid(self):
        """Check if API key is valid (active and not expired)."""
        return self.is_active and not self.is_expired()


class APIUsage(Base):
    """
    API Usage tracking for analytics and abuse detection.

    Attributes:
        id: Primary key
        api_key_id: Foreign key to api_keys table
        endpoint: API endpoint called
        method: HTTP method (GET, POST, etc.)
        status_code: HTTP status code returned
        response_time_ms: Response time in milliseconds
        timestamp: Request timestamp
    """
    __tablename__ = 'api_usage'

    id = Column(Integer, primary_key=True)
    api_key_id = Column(Integer, ForeignKey('api_keys.id', ondelete='CASCADE'), nullable=False)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    response_time_ms = Column(Integer)
    timestamp = Column(TIMESTAMP, default=func.now(), index=True)

    # Relationships
    api_key = relationship("APIKey", back_populates="usage_records")

    def __repr__(self):
        return f"<APIUsage(id={self.id}, endpoint='{self.endpoint}', status={self.status_code})>"

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'api_key_id': self.api_key_id,
            'endpoint': self.endpoint,
            'method': self.method,
            'status_code': self.status_code,
            'response_time_ms': self.response_time_ms,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }
