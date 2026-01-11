"""
HPO Campaign Service (Phase 11C).
Business logic for hyperparameter optimization campaigns.
"""
from typing import Dict, List, Optional, Any
from database.connection import get_db_session
from models.hpo_campaign import HPOCampaign, HPOMethod
from models.experiment import Experiment, ExperimentStatus
from models.dataset import Dataset
from utils.logger import setup_logger
from datetime import datetime
import json

logger = setup_logger(__name__)


class HPOService:
    """Service for managing HPO campaigns."""

    @staticmethod
    def create_campaign(
        name: str,
        method: str,
        base_model_type: str,
        dataset_id: int,
        search_space: Dict[str, Any],
        num_trials: int,
        metric: str = "val_accuracy",
        direction: str = "maximize",
        created_by: Optional[int] = None
    ) -> Optional[int]:
        """
        Create a new HPO campaign.

        Args:
            name: Campaign name
            method: Optimization method (bayesian, random, grid, hyperband)
            base_model_type: Base model type to optimize
            dataset_id: Dataset ID
            search_space: Hyperparameter search space dictionary
            num_trials: Number of trials to run
            metric: Metric to optimize (val_accuracy, val_loss, f1_score)
            direction: Optimization direction (maximize, minimize)
            created_by: User ID

        Returns:
            Campaign ID if successful, None otherwise
        """
        try:
            with get_db_session() as session:
                # Validate dataset exists
                dataset = session.query(Dataset).filter_by(id=dataset_id).first()
                if not dataset:
                    logger.error(f"Dataset {dataset_id} not found")
                    return None

                # Convert method string to enum
                method_enum = HPOMethod[method.upper()]

                # Create campaign
                campaign = HPOCampaign(
                    name=name,
                    method=method_enum,
                    status=ExperimentStatus.PENDING,
                    base_model_type=base_model_type,
                    dataset_id=dataset_id,
                    search_space=search_space,
                    budget={
                        "max_trials": num_trials,
                        "metric": metric,
                        "direction": direction
                    },
                    trials_completed=0,
                    trials_total=num_trials,
                    best_accuracy=None,
                    created_by=created_by
                )

                session.add(campaign)
                session.flush()
                campaign_id = campaign.id
                session.commit()

                logger.info(f"Created HPO campaign {campaign_id}: {name}")
                return campaign_id

        except Exception as e:
            logger.error(f"Failed to create HPO campaign: {e}", exc_info=True)
            return None

    @staticmethod
    def get_campaign(campaign_id: int) -> Optional[Dict]:
        """
        Get campaign by ID.

        Args:
            campaign_id: Campaign ID

        Returns:
            Campaign dictionary or None
        """
        try:
            with get_db_session() as session:
                campaign = session.query(HPOCampaign).filter_by(id=campaign_id).first()
                if not campaign:
                    return None

                return HPOService._campaign_to_dict(campaign)

        except Exception as e:
            logger.error(f"Failed to get campaign {campaign_id}: {e}", exc_info=True)
            return None

    @staticmethod
    def get_all_campaigns() -> List[Dict]:
        """
        Get all HPO campaigns.

        Returns:
            List of campaign dictionaries
        """
        try:
            with get_db_session() as session:
                # Apply pagination to prevent loading too many campaigns
                from utils.query_utils import paginate_with_default_limit
                campaigns = paginate_with_default_limit(
                    session.query(HPOCampaign).order_by(HPOCampaign.created_at.desc()),
                    limit=100
                )

                return [HPOService._campaign_to_dict(c) for c in campaigns]

        except Exception as e:
            logger.error(f"Failed to get campaigns: {e}", exc_info=True)
            return []

    @staticmethod
    def update_campaign_progress(
        campaign_id: int,
        trials_completed: int,
        best_experiment_id: Optional[int] = None,
        best_accuracy: Optional[float] = None
    ) -> bool:
        """
        Update campaign progress.

        Args:
            campaign_id: Campaign ID
            trials_completed: Number of completed trials
            best_experiment_id: ID of best experiment so far
            best_accuracy: Best accuracy achieved

        Returns:
            True if successful, False otherwise
        """
        try:
            with get_db_session() as session:
                campaign = session.query(HPOCampaign).filter_by(id=campaign_id).first()
                if not campaign:
                    return False

                campaign.trials_completed = trials_completed

                if best_experiment_id is not None:
                    campaign.best_experiment_id = best_experiment_id

                if best_accuracy is not None:
                    campaign.best_accuracy = best_accuracy

                # Update status
                if trials_completed >= campaign.trials_total:
                    campaign.status = ExperimentStatus.COMPLETED
                elif trials_completed > 0:
                    campaign.status = ExperimentStatus.RUNNING

                session.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to update campaign progress: {e}", exc_info=True)
            return False

    @staticmethod
    def update_campaign_status(campaign_id: int, status: ExperimentStatus) -> bool:
        """
        Update campaign status.

        Args:
            campaign_id: Campaign ID
            status: New status

        Returns:
            True if successful, False otherwise
        """
        try:
            with get_db_session() as session:
                campaign = session.query(HPOCampaign).filter_by(id=campaign_id).first()
                if not campaign:
                    return False

                campaign.status = status
                session.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to update campaign status: {e}", exc_info=True)
            return False

    @staticmethod
    def get_campaign_experiments(campaign_id: int) -> List[Dict]:
        """
        Get all experiments for a campaign.

        Args:
            campaign_id: Campaign ID

        Returns:
            List of experiment dictionaries
        """
        try:
            with get_db_session() as session:
                # Get experiments with matching campaign tag
                # Apply pagination to prevent loading too many experiments
                from utils.query_utils import paginate_with_default_limit
                experiments = paginate_with_default_limit(
                    session.query(Experiment).filter(
                        Experiment.tags.contains([f"hpo_campaign_{campaign_id}"])
                    ).order_by(Experiment.created_at.asc()),
                    limit=500
                )

                return [HPOService._experiment_to_dict(e) for e in experiments]

        except Exception as e:
            logger.error(f"Failed to get campaign experiments: {e}", exc_info=True)
            return []

    @staticmethod
    def _campaign_to_dict(campaign: HPOCampaign) -> Dict:
        """Convert campaign model to dictionary."""
        return {
            "id": campaign.id,
            "name": campaign.name,
            "method": campaign.method.value if campaign.method else None,
            "status": campaign.status.value if campaign.status else "pending",
            "base_model_type": campaign.base_model_type,
            "dataset_id": campaign.dataset_id,
            "search_space": campaign.search_space,
            "budget": campaign.budget,
            "trials_completed": campaign.trials_completed,
            "trials_total": campaign.trials_total,
            "best_experiment_id": campaign.best_experiment_id,
            "best_accuracy": campaign.best_accuracy,
            "created_by": campaign.created_by,
            "created_at": campaign.created_at.isoformat() if campaign.created_at else None,
            "updated_at": campaign.updated_at.isoformat() if campaign.updated_at else None,
        }

    @staticmethod
    def _experiment_to_dict(experiment: Experiment) -> Dict:
        """Convert experiment model to dictionary."""
        return {
            "id": experiment.id,
            "name": experiment.name,
            "model_type": experiment.model_type,
            "status": experiment.status.value if experiment.status else "pending",
            "test_accuracy": experiment.metrics.get("test_accuracy") if experiment.metrics else None,
            "test_loss": experiment.metrics.get("test_loss") if experiment.metrics else None,
            "hyperparameters": experiment.config.get("hyperparameters") if experiment.config else {},
            "created_at": experiment.created_at.isoformat() if experiment.created_at else None,
            "duration_seconds": experiment.duration_seconds,
        }

    @staticmethod
    def get_default_search_space(model_type: str) -> Dict[str, Any]:
        """
        Get default search space for a model type.

        Args:
            model_type: Model type

        Returns:
            Default search space dictionary
        """
        # Common search spaces for different model types
        if model_type in ["resnet18", "resnet34"]:
            return {
                "learning_rate": {
                    "type": "float",
                    "min": 1e-5,
                    "max": 1e-2,
                    "log": True
                },
                "batch_size": {
                    "type": "categorical",
                    "choices": [16, 32, 64, 128]
                },
                "optimizer": {
                    "type": "categorical",
                    "choices": ["adam", "adamw", "sgd"]
                },
                "weight_decay": {
                    "type": "float",
                    "min": 1e-6,
                    "max": 1e-3,
                    "log": True
                },
                "dropout": {
                    "type": "float",
                    "min": 0.1,
                    "max": 0.5
                }
            }
        elif model_type == "transformer":
            return {
                "learning_rate": {
                    "type": "float",
                    "min": 1e-5,
                    "max": 1e-2,
                    "log": True
                },
                "d_model": {
                    "type": "categorical",
                    "choices": [128, 256, 512]
                },
                "num_layers": {
                    "type": "int",
                    "min": 2,
                    "max": 8
                },
                "num_heads": {
                    "type": "categorical",
                    "choices": [4, 8, 16]
                },
                "dropout": {
                    "type": "float",
                    "min": 0.1,
                    "max": 0.5
                }
            }
        else:
            # Generic search space
            return {
                "learning_rate": {
                    "type": "float",
                    "min": 1e-5,
                    "max": 1e-2,
                    "log": True
                },
                "batch_size": {
                    "type": "categorical",
                    "choices": [16, 32, 64]
                },
                "optimizer": {
                    "type": "categorical",
                    "choices": ["adam", "adamw"]
                }
            }
