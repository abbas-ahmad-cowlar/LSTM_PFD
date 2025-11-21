"""Experiment model for storing training experiments."""
from sqlalchemy import Column, Integer, String, Float, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship
from models.base import BaseModel
import enum


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

    # Relationships
    created_by = Column(Integer, ForeignKey('users.id'))
    hpo_campaign_id = Column(Integer, ForeignKey('hpo_campaigns.id'))  # If part of HPO

    dataset = relationship("Dataset", backref="experiments")
    training_runs = relationship("TrainingRun", back_populates="experiment", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Experiment(name='{self.name}', status='{self.status.value}')>"
