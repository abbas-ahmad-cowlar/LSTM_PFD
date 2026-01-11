"""Experiment model for storing training experiments."""
from sqlalchemy import Column, Integer, String, Float, JSON, ForeignKey, Enum, Text, Index
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import TSVECTOR
from models.base import BaseModel
import enum
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


class ExperimentStatus(enum.Enum):
    """Experiment status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Experiment(BaseModel):
    """Experiment metadata model."""
    __tablename__ = 'experiments'

    name = Column(String(255), unique=True, nullable=False, index=True)
    model_type = Column(String(50), nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    status = Column(Enum(ExperimentStatus), default=ExperimentStatus.PENDING, index=True)
    config = Column(JSON, nullable=False)  # Full experiment configuration
    hyperparameters = Column(JSON)  # Extracted hyperparameters for easy querying

    # Results
    metrics = Column(JSON)  # {accuracy, f1, precision, recall, ...}
    best_epoch = Column(Integer)
    total_epochs = Column(Integer)
    duration_seconds = Column(Float)

    # File references
    model_path = Column(String(500))  # Path to saved model
    onnx_path = Column(String(500))  # Path to ONNX export
    results_dir = Column(String(500))  # Directory containing all results

    # Feature #5: Tags & Search
    notes = Column(Text)  # User notes for experiment (searchable)
    search_vector = Column(TSVECTOR)  # Full-text search vector (auto-updated by trigger)

    # Relationships
    created_by = Column(Integer, ForeignKey('users.id'))
    hpo_campaign_id = Column(Integer, ForeignKey('hpo_campaigns.id'))  # If part of HPO

    dataset = relationship("Dataset", backref="experiments")
    training_runs = relationship("TrainingRun", back_populates="experiment", cascade="all, delete-orphan")

    # Performance indexes
    # Note: name, model_type, status already have column-level indexes
    # Note: dataset_id, created_by, hpo_campaign_id are ForeignKeys (auto-indexed in PostgreSQL)
    __table_args__ = (
        Index('ix_experiments_created_at', 'created_at'),
        # Removed duplicate indexes on ForeignKey columns
        # Removed composite on created_by+status (status already indexed, low cardinality)
    )

    def __repr__(self):
        return f"<Experiment(name='{self.name}', status='{self.status.value}')>"

    def get_tags(self):
        """Get list of tags for this experiment."""
        from models.tag import ExperimentTag
        return [et.tag for et in self.experiment_tags if et.tag]

    def to_dict_with_tags(self):
        """Convert experiment to dictionary including tags."""
        data = self.to_dict()
        data['tags'] = [
            {
                'id': tag.id,
                'name': tag.name,
                'slug': tag.slug,
                'color': tag.color
            }
            for tag in self.get_tags()
        ]
        return data
