"""Dataset Import model for tracking MAT file import jobs."""
from sqlalchemy import Column, Integer, String, Float, JSON, Enum
from models.base import BaseModel
import enum


class DatasetImportStatus(enum.Enum):
    """Dataset import status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DatasetImport(BaseModel):
    """Model for tracking MAT file import jobs."""
    __tablename__ = 'dataset_imports'

    name = Column(String(255), nullable=False, index=True)
    config = Column(JSON, nullable=False)  # Full import configuration
    status = Column(Enum(DatasetImportStatus), default=DatasetImportStatus.PENDING, index=True)

    # Import parameters
    num_files = Column(Integer)  # Number of MAT files
    num_signals = Column(Integer)  # Total signals imported
    output_path = Column(String(500))  # Path to output HDF5/directory

    # Progress tracking
    progress = Column(Integer, default=0)  # Progress percentage (0-100)
    celery_task_id = Column(String(255), index=True)  # Celery task ID for monitoring

    # Performance metrics
    duration_seconds = Column(Float)  # Total import time

    def __repr__(self):
        return f"<DatasetImport(name='{self.name}', status='{self.status.value}')>"
