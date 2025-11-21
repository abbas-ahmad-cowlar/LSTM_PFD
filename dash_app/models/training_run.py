"""Training run model for epoch-level metrics."""
from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship
from models.base import BaseModel


class TrainingRun(BaseModel):
    """Training run (per-epoch) model."""
    __tablename__ = 'training_runs'

    experiment_id = Column(Integer, ForeignKey('experiments.id'), nullable=False, index=True)
    epoch = Column(Integer, nullable=False)
    train_loss = Column(Float, nullable=False)
    val_loss = Column(Float, nullable=False)
    val_accuracy = Column(Float, nullable=False)
    checkpoint_path = Column(String(500))

    experiment = relationship("Experiment", back_populates="training_runs")

    def __repr__(self):
        return f"<TrainingRun(experiment_id={self.experiment_id}, epoch={self.epoch})>"
