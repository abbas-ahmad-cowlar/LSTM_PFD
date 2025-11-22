"""Dataset Generation model for tracking Phase 0 data generation jobs."""
from sqlalchemy import Column, Integer, String, Float, JSON, Enum
from models.base import BaseModel
import enum


class DatasetGenerationStatus(enum.Enum):
    """Dataset generation status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DatasetGeneration(BaseModel):
    """Model for tracking dataset generation jobs."""
    __tablename__ = 'dataset_generations'

    name = Column(String(255), nullable=False, index=True)
    config = Column(JSON, nullable=False)  # Full generation configuration
    status = Column(Enum(DatasetGenerationStatus), default=DatasetGenerationStatus.PENDING, index=True)

    # Generation parameters
    num_signals = Column(Integer)  # Total signals generated
    num_faults = Column(Integer)  # Number of fault types
    output_path = Column(String(500))  # Path to generated dataset

    # Progress tracking
    progress = Column(Integer, default=0)  # Progress percentage (0-100)
    celery_task_id = Column(String(255), index=True)  # Celery task ID for monitoring

    # Performance metrics
    duration_seconds = Column(Float)  # Total generation time

    def __repr__(self):
        return f"<DatasetGeneration(name='{self.name}', status='{self.status.value}')>"
