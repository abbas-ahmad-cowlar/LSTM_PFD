"""
Neural Architecture Search (NAS) Campaign and Trial models.
Tracks NAS campaigns and individual architecture trials.
"""
from sqlalchemy import Column, Integer, String, Float, BigInteger, ForeignKey, JSON, Text, Index
from sqlalchemy.orm import relationship
from models.base import BaseModel


class NASCampaign(BaseModel):
    """
    NAS Campaign model for tracking architecture search campaigns.

    A campaign runs multiple trials to find optimal architectures.
    """
    __tablename__ = 'nas_campaigns'

    name = Column(String(200), nullable=False, index=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id', ondelete='CASCADE'), nullable=False)
    search_algorithm = Column(String(50), default='random')  # 'random', 'bayesian', 'evolution'
    num_trials = Column(Integer, default=20)
    max_epochs_per_trial = Column(Integer, default=10)
    search_space_config = Column(JSON)  # SearchSpaceConfig parameters
    status = Column(String(50), default='pending')  # 'pending', 'running', 'completed', 'failed'

    # Results
    best_trial_id = Column(Integer, ForeignKey('nas_trials.id', use_alter=True), nullable=True)
    best_accuracy = Column(Float, nullable=True)

    # Task tracking
    task_id = Column(String(200), nullable=True)  # Celery task ID
    error_message = Column(Text, nullable=True)

    # Relationships
    dataset = relationship("Dataset", backref="nas_campaigns")
    trials = relationship("NASTrial", backref="campaign", foreign_keys="NASTrial.campaign_id",
                         cascade="all, delete-orphan")

    # Performance indexes
    # Note: name already has column-level index
    # Note: dataset_id is ForeignKey (auto-indexed)
    __table_args__ = (
        Index('ix_nas_campaigns_status', 'status'),
        Index('ix_nas_campaigns_created_at', 'created_at'),
        # Removed duplicate index on dataset_id (ForeignKey)
        # Removed composite - status has low cardinality
    )

    def __repr__(self):
        return f"<NASCampaign(name='{self.name}', algorithm='{self.search_algorithm}', status='{self.status}')>"


class NASTrial(BaseModel):
    """
    Individual NAS Trial model.

    Represents one architecture evaluation within a campaign.
    """
    __tablename__ = 'nas_trials'

    campaign_id = Column(Integer, ForeignKey('nas_campaigns.id', ondelete='CASCADE'), nullable=False)
    trial_number = Column(Integer, nullable=False)  # Sequential trial number

    # Architecture specification
    architecture = Column(JSON, nullable=False)  # Full architecture dict
    architecture_hash = Column(String(64), index=True)  # For deduplication

    # Training results
    validation_accuracy = Column(Float, nullable=True)
    validation_loss = Column(Float, nullable=True)
    training_time = Column(Float, nullable=True)  # seconds

    # Model complexity metrics
    num_parameters = Column(Integer, nullable=True)
    flops = Column(BigInteger, nullable=True)  # FLOPs count
    model_size_mb = Column(Float, nullable=True)  # Model size in MB

    # Additional metrics
    metrics = Column(JSON, nullable=True)  # Additional training metrics

    # Performance indexes
    # Note: campaign_id is ForeignKey (auto-indexed)
    # Note: architecture_hash already has column-level index
    __table_args__ = (
        Index('ix_nas_trials_created_at', 'created_at'),
        Index('ix_nas_trials_campaign_trial', 'campaign_id', 'trial_number'),  # Composite for ORDER BY
        # Kept composite for query pattern: SELECT * FROM nas_trials WHERE campaign_id=X ORDER BY trial_number
    )

    def __repr__(self):
        return f"<NASTrial(campaign_id={self.campaign_id}, trial={self.trial_number}, acc={self.validation_accuracy})>"
