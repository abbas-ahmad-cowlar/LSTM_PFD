"""Training run model for epoch-level metrics."""
from sqlalchemy import Column, Integer, Float, String, ForeignKey, Index
from sqlalchemy.orm import relationship
from models.base import BaseModel
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


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

    # Performance indexes
    # Note: experiment_id already has column-level index (FK)
    __table_args__ = (
        Index('ix_training_runs_created_at', 'created_at'),
        Index('ix_training_runs_experiment_epoch', 'experiment_id', 'epoch'),  # Composite for ORDER BY
        # Composite is useful for: SELECT * FROM training_runs WHERE experiment_id=X ORDER BY epoch
    )

    def __repr__(self):
        return f"<TrainingRun(experiment_id={self.experiment_id}, epoch={self.epoch})>"
