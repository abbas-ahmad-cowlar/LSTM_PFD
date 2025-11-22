"""Signal model for storing signal records."""
from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import relationship
from models.base import BaseModel
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


class Signal(BaseModel):
    """Signal record model."""
    __tablename__ = 'signals'

    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False, index=True)
    signal_id = Column(String(100), nullable=False, index=True)
    fault_class = Column(String(50), nullable=False, index=True)
    severity = Column(String(50))
    file_path = Column(String(500))  # Path to signal file if stored separately
    rms = Column(Float)
    kurtosis = Column(Float)
    dominant_frequency = Column(Float)

    dataset = relationship("Dataset", backref="signals")

    def __repr__(self):
        return f"<Signal(id='{self.signal_id}', fault='{self.fault_class}')>"
