"""Dataset model for storing dataset metadata."""
from sqlalchemy import Column, Integer, String, JSON, ForeignKey
from models.base import BaseModel
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


class Dataset(BaseModel):
    """Dataset metadata model."""
    __tablename__ = 'datasets'

    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(String(1000))
    num_signals = Column(Integer, nullable=False)
    fault_types = Column(JSON, nullable=False)  # List of fault types included
    severity_levels = Column(JSON, nullable=False)  # List of severity levels
    file_path = Column(String(500), nullable=False)  # Path to HDF5 file
    metadata = Column(JSON)  # Additional metadata
    created_by = Column(Integer, ForeignKey('users.id'))

    def __repr__(self):
        return f"<Dataset(name='{self.name}', num_signals={self.num_signals})>"
