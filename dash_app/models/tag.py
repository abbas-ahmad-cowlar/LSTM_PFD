"""Tag models for experiment organization."""
from sqlalchemy import Column, Integer, String, ForeignKey, UniqueConstraint, Index
from sqlalchemy.orm import relationship
from models.base import BaseModel


class Tag(BaseModel):
    """
    Tag model for categorizing experiments.

    Tags allow users to organize and filter experiments by custom categories
    like 'baseline', 'production', 'research', etc.
    """
    __tablename__ = 'tags'

    name = Column(String(50), unique=True, nullable=False, index=True)
    slug = Column(String(50), unique=True, nullable=False, index=True)
    color = Column(String(7))  # Hex color code (e.g., "#3498db")

    # Metadata
    created_by = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'))
    usage_count = Column(Integer, default=0, nullable=False)

    # Relationships
    experiment_tags = relationship(
        "ExperimentTag",
        back_populates="tag",
        cascade="all, delete-orphan"
    )

    # Performance indexes
    # Note: name and slug already have column-level unique indexes
    # Note: created_by is ForeignKey (auto-indexed)
    __table_args__ = (
        Index('ix_tags_created_at', 'created_at'),
        # Removed duplicate on created_by (FK)
    )

    def __repr__(self):
        return f"<Tag(name='{self.name}', usage_count={self.usage_count})>"

    def to_dict(self):
        """Convert tag to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'slug': self.slug,
            'color': self.color,
            'usage_count': self.usage_count,
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }


class ExperimentTag(BaseModel):
    """
    Many-to-many relationship between experiments and tags.

    Tracks which tags are applied to which experiments, including
    audit information about who added the tag and when.
    """
    __tablename__ = 'experiment_tags'

    experiment_id = Column(
        Integer,
        ForeignKey('experiments.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )
    tag_id = Column(
        Integer,
        ForeignKey('tags.id', ondelete='CASCADE'),
        nullable=False,
        index=True
    )

    # Audit trail
    added_by = Column(Integer, ForeignKey('users.id', ondelete='SET NULL'))

    # Relationships
    experiment = relationship("Experiment", backref="experiment_tags")
    tag = relationship("Tag", back_populates="experiment_tags")

    # Constraints and performance indexes
    # Note: experiment_id and tag_id already have column-level indexes
    # Note: UniqueConstraint automatically creates composite index on (experiment_id, tag_id)
    # Note: added_by is ForeignKey (auto-indexed)
    __table_args__ = (
        UniqueConstraint('experiment_id', 'tag_id', name='uq_experiment_tag'),
        Index('ix_experiment_tags_created_at', 'created_at'),
        # Removed all duplicates - UniqueConstraint handles composite index
    )

    def __repr__(self):
        return f"<ExperimentTag(experiment_id={self.experiment_id}, tag_id={self.tag_id})>"

    def to_dict(self):
        """Convert experiment tag to dictionary."""
        return {
            'id': self.id,
            'experiment_id': self.experiment_id,
            'tag_id': self.tag_id,
            'tag_name': self.tag.name if self.tag else None,
            'tag_color': self.tag.color if self.tag else None,
            'added_by': self.added_by,
            'added_at': self.created_at.isoformat() if self.created_at else None,
        }
