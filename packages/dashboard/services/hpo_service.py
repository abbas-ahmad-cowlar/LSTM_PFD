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

    @staticmethod
    def resume_campaign(campaign_id: int) -> bool:
        """
        Resume a paused or cancelled campaign.

        Args:
            campaign_id: Campaign ID

        Returns:
            True if campaign was resumed, False otherwise
        """
        try:
            with get_db_session() as session:
                campaign = session.query(HPOCampaign).filter_by(id=campaign_id).first()
                if not campaign:
                    logger.error(f"Campaign {campaign_id} not found")
                    return False

                # Only resume if paused or cancelled
                if campaign.status not in [ExperimentStatus.CANCELLED, ExperimentStatus.PAUSED]:
                    logger.warning(f"Campaign {campaign_id} cannot be resumed (status: {campaign.status})")
                    return False

                # Update status to pending for resumption
                campaign.status = ExperimentStatus.PENDING
                session.commit()

                logger.info(f"Campaign {campaign_id} marked for resumption")
                return True

        except Exception as e:
            logger.error(f"Failed to resume campaign {campaign_id}: {e}", exc_info=True)
            return False

    @staticmethod
    def export_results(campaign_id: int, format: str = "json") -> Optional[str]:
        """
        Export campaign results in the specified format.

        Args:
            campaign_id: Campaign ID
            format: Export format (json, yaml, python)

        Returns:
            Exported content as string, or None on failure
        """
        try:
            campaign = HPOService.get_campaign(campaign_id)
            if not campaign:
                return None

            experiments = HPOService.get_campaign_experiments(campaign_id)

            # Build export data
            export_data = {
                "campaign": {
                    "name": campaign["name"],
                    "model_type": campaign["base_model_type"],
                    "method": campaign["method"],
                    "metric": campaign["budget"].get("metric", "val_accuracy"),
                    "direction": campaign["budget"].get("direction", "maximize"),
                    "trials_completed": campaign["trials_completed"],
                    "best_accuracy": campaign["best_accuracy"],
                },
                "search_space": campaign["search_space"],
                "best_params": {},
                "trials": []
            }

            # Find best experiment and extract params
            best_exp_id = campaign.get("best_experiment_id")
            for exp in experiments:
                trial_data = {
                    "trial_id": exp["id"],
                    "name": exp["name"],
                    "status": exp["status"],
                    "hyperparameters": exp.get("hyperparameters", {}),
                    "test_accuracy": exp.get("test_accuracy"),
                    "test_loss": exp.get("test_loss"),
                    "duration_seconds": exp.get("duration_seconds")
                }
                export_data["trials"].append(trial_data)

                if exp["id"] == best_exp_id:
                    export_data["best_params"] = exp.get("hyperparameters", {})

            # Format output
            if format == "json":
                return json.dumps(export_data, indent=2)
            elif format == "yaml":
                try:
                    import yaml
                    return yaml.dump(export_data, default_flow_style=False, sort_keys=False)
                except ImportError:
                    # Fallback to JSON if yaml not installed
                    return json.dumps(export_data, indent=2)
            elif format == "python":
                # Return best params as Python dict literal
                params_str = "best_hyperparameters = " + repr(export_data["best_params"])
                return params_str
            else:
                return json.dumps(export_data, indent=2)

        except Exception as e:
            logger.error(f"Failed to export campaign {campaign_id}: {e}", exc_info=True)
            return None

    @staticmethod
    def get_trials_dataframe(campaign_id: int) -> Optional[Dict]:
        """
        Get trial data in a format suitable for visualization.

        Args:
            campaign_id: Campaign ID

        Returns:
            Dictionary with trial data for parallel coordinates, etc.
        """
        try:
            experiments = HPOService.get_campaign_experiments(campaign_id)
            if not experiments:
                return None

            # Build dataframe-like structure
            data = {
                "trial_id": [],
                "test_accuracy": [],
                "test_loss": [],
            }

            # Collect all hyperparameter names
            all_params = set()
            for exp in experiments:
                params = exp.get("hyperparameters", {})
                all_params.update(params.keys())

            # Initialize param columns
            for param in all_params:
                data[param] = []

            # Populate data
            for exp in experiments:
                if exp.get("status") not in ["completed"]:
                    continue  # Only include completed trials

                data["trial_id"].append(exp["id"])
                data["test_accuracy"].append(exp.get("test_accuracy") or 0)
                data["test_loss"].append(exp.get("test_loss") or 0)

                params = exp.get("hyperparameters", {})
                for param in all_params:
                    data[param].append(params.get(param, None))

            return data

        except Exception as e:
            logger.error(f"Failed to get trials dataframe for campaign {campaign_id}: {e}", exc_info=True)
            return None

    @staticmethod
    def get_parameter_importance(campaign_id: int) -> Optional[Dict]:
        """
        Calculate parameter importance using correlation analysis.

        Args:
            campaign_id: Campaign ID

        Returns:
            Dictionary with parameter names and importance scores
        """
        try:
            trials_data = HPOService.get_trials_dataframe(campaign_id)
            if not trials_data or len(trials_data.get("trial_id", [])) < 3:
                return None

            # Calculate correlation between each param and accuracy
            target = trials_data.get("test_accuracy", [])
            if not target:
                return None

            import numpy as np

            importance = {}
            skip_keys = {"trial_id", "test_accuracy", "test_loss"}

            for param, values in trials_data.items():
                if param in skip_keys:
                    continue

                # Convert to numeric if possible
                try:
                    numeric_values = []
                    for v in values:
                        if v is None:
                            numeric_values.append(0)
                        elif isinstance(v, (int, float)):
                            numeric_values.append(float(v))
                        elif isinstance(v, str):
                            # Encode categorical as hash
                            numeric_values.append(hash(v) % 1000 / 1000)
                        else:
                            numeric_values.append(0)

                    if len(numeric_values) == len(target) and len(target) > 1:
                        # Calculate absolute correlation
                        correlation = np.corrcoef(numeric_values, target)[0, 1]
                        if not np.isnan(correlation):
                            importance[param] = abs(correlation)
                        else:
                            importance[param] = 0.0
                    else:
                        importance[param] = 0.0
                except Exception:
                    importance[param] = 0.0

            # Normalize to sum to 1
            total = sum(importance.values())
            if total > 0:
                importance = {k: v / total for k, v in importance.items()}

            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

            return importance

        except Exception as e:
            logger.error(f"Failed to calculate parameter importance for campaign {campaign_id}: {e}", exc_info=True)
            return None

    @staticmethod
    def save_research_artifact(campaign_id: int) -> Optional[str]:
        """
        Save campaign results as a research artifact (JSON file).

        Args:
            campaign_id: Campaign ID

        Returns:
            Path to saved artifact, or None on failure
        """
        try:
            import os
            from datetime import datetime

            campaign = HPOService.get_campaign(campaign_id)
            if not campaign:
                return None

            # Export as JSON
            export_content = HPOService.export_results(campaign_id, "json")
            if not export_content:
                return None

            # Create artifacts directory
            artifacts_dir = os.path.join(os.getcwd(), "storage", "research_artifacts", "hpo")
            os.makedirs(artifacts_dir, exist_ok=True)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = campaign["name"].replace(" ", "_").replace("/", "_")
            filename = f"hpo_{safe_name}_{timestamp}.json"
            filepath = os.path.join(artifacts_dir, filename)

            # Write artifact
            with open(filepath, "w") as f:
                f.write(export_content)

            logger.info(f"Saved research artifact: {filepath}")
            return filepath

        except Exception as e:
            logger.error(f"Failed to save research artifact for campaign {campaign_id}: {e}", exc_info=True)
            return None

