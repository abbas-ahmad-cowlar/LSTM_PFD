"""Explanation model for caching XAI results."""
from sqlalchemy import Column, Integer, String, JSON, ForeignKey
from models.base import BaseModel
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


class Explanation(BaseModel):
    """Cached explanation model."""
    __tablename__ = 'explanations'

    experiment_id = Column(Integer, ForeignKey('experiments.id'), nullable=False, index=True)
    signal_id = Column(String(100), nullable=False, index=True)
    method = Column(String(50), nullable=False)  # shap, lime, grad_cam, etc.
    explanation_data = Column(JSON, nullable=False)  # Serialized explanation
    file_path = Column(String(500))  # Path to visualization file if saved

    def __repr__(self):
        return f"<Explanation(experiment_id={self.experiment_id}, signal_id='{self.signal_id}', method='{self.method}')>"
