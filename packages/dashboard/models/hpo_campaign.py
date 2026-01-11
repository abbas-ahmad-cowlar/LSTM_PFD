"""HPO Campaign model for hyperparameter optimization."""
from sqlalchemy import Column, Integer, String, Float, JSON, ForeignKey, Enum, Index
from sqlalchemy.orm import relationship
from models.base import BaseModel
from models.experiment import ExperimentStatus
import enum


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

    # Performance indexes
    # Note: name already has column-level index
    # Note: created_by, dataset_id are ForeignKeys (auto-indexed)
    __table_args__ = (
        Index('ix_hpo_campaigns_status', 'status'),
        Index('ix_hpo_campaigns_created_at', 'created_at'),
        # Removed duplicate indexes on ForeignKey columns
        # Removed composite - status has low cardinality, minimal benefit
    )

    def __repr__(self):
        return f"<HPOCampaign(name='{self.name}', method='{self.method.value}')>"
