"""
Backup codes model for 2FA recovery (Phase 6, Feature 3).

Backup codes provide a recovery mechanism if users lose access to their
authenticator app. Each user gets 10 single-use codes during 2FA setup.
"""
from sqlalchemy import Column, Integer, String, ForeignKey, Boolean, DateTime, Index
from datetime import datetime
from models.base import BaseModel


class BackupCode(BaseModel):
    """
    Model for storing 2FA backup codes.

    Backup codes are one-time use recovery codes that allow users to access
    their account if they lose their authenticator device.

    Attributes:
        user_id: Foreign key to User model
        code_hash: bcrypt hash of the backup code (never store plaintext)
        is_used: Whether this code has been used
        used_at: Timestamp when the code was used
        ip_address: IP address where code was used (for audit trail)

    Security Notes:
        - Codes are 12 characters (alphanumeric, no ambiguous chars)
        - Hashed with bcrypt (cost factor 12) before storage
        - Each code can only be used once
        - Users get 10 codes per 2FA setup
        - New codes replace old ones when regenerated
    """
    __tablename__ = "backup_codes"

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    code_hash = Column(String(255), nullable=False, unique=True)
    is_used = Column(Boolean, default=False, nullable=False)
    used_at = Column(DateTime, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IP where code was used

    # Indexes for performance
    __table_args__ = (
        Index('ix_backup_codes_user_id', 'user_id'),
        Index('ix_backup_codes_is_used', 'is_used'),
        # Composite index for finding unused codes for a user
        Index('ix_backup_codes_user_unused', 'user_id', 'is_used'),
    )

    def mark_as_used(self, ip_address: str = None):
        """
        Mark this backup code as used.

        Args:
            ip_address: IP address where code was used
        """
        self.is_used = True
        self.used_at = datetime.utcnow()
        if ip_address:
            self.ip_address = ip_address

    def __repr__(self):
        return f"<BackupCode(user_id={self.user_id}, used={self.is_used})>"
