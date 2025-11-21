"""Saved search model for bookmarking frequent queries."""
from sqlalchemy import Column, Integer, String, Text, Boolean, ForeignKey, UniqueConstraint, DateTime
from sqlalchemy.orm import relationship
from models.base import BaseModel


class SavedSearch(BaseModel):
    """
    Saved search queries for quick access.

    Allows users to bookmark complex search queries (e.g., "tag:baseline accuracy:>0.95")
    for easy re-use without having to remember or retype the query syntax.
    """
    __tablename__ = 'saved_searches'

    user_id = Column(
        Integer,
        ForeignKey('users.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )

    name = Column(String(200), nullable=False)  # User-provided name
    query = Column(Text, nullable=False)        # The search query string

    # Metadata
    is_pinned = Column(Boolean, default=False, nullable=False)
    usage_count = Column(Integer, default=0, nullable=False)
    last_used_at = Column(DateTime)

    # Relationships
    user = relationship("User", backref="saved_searches")

    # Unique constraint: user can't have duplicate saved search names
    __table_args__ = (
        UniqueConstraint('user_id', 'name', name='uq_user_saved_search_name'),
    )

    def __repr__(self):
        return f"<SavedSearch(user_id={self.user_id}, name='{self.name}')>"

    def to_dict(self):
        """Convert saved search to dictionary."""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'query': self.query,
            'is_pinned': self.is_pinned,
            'usage_count': self.usage_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
        }
