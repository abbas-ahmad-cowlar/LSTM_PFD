"""HPO Campaign model for hyperparameter optimization."""
from sqlalchemy import Column, Integer, String, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship
from models.base import BaseModel
from models.experiment import ExperimentStatus
import enum
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


class HPOMethod(enum.Enum):
    """HPO method enumeration."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    HYPERBAND = "hyperband"


class HPOCampaign(BaseModel):
    """HPO Campaign model."""
    __tablename__ = 'hpo_campaigns'

    name = Column(String(255), unique=True, nullable=False, index=True)
    method = Column(Enum(HPOMethod), nullable=False)
    status = Column(Enum(ExperimentStatus), default=ExperimentStatus.PENDING)

    # Configuration
    base_model_type = Column(String(50), nullable=False)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    search_space = Column(JSON, nullable=False)  # Hyperparameter search space
    budget = Column(JSON, nullable=False)  # {max_trials, max_duration}

    # Progress
    trials_completed = Column(Integer, default=0)
    trials_total = Column(Integer, nullable=False)
    best_experiment_id = Column(Integer, ForeignKey('experiments.id'))
    best_accuracy = Column(Float)

    created_by = Column(Integer, ForeignKey('users.id'))

    # Relationships
    dataset = relationship("Dataset")
    best_experiment = relationship("Experiment", foreign_keys=[best_experiment_id])

    def __repr__(self):
        return f"<HPOCampaign(name='{self.name}', method='{self.method.value}')>"
