"""
MLflow Experiment Management

Provides interface for:
- Creating experiments
- Starting/ending runs
- Logging parameters, metrics, and artifacts
- Querying experiment results
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json


class ExperimentManager:
    """
    Manage MLflow experiments and runs.

    Args:
        experiment_name: Name of the experiment
        tracking_uri: Optional MLflow tracking URI
    """
    def __init__(
        self,
        experiment_name: str = 'bearing_fault_diagnosis',
        tracking_uri: Optional[str] = None
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.mlflow = None
        self.current_run = None

        self._init_mlflow()

    def _init_mlflow(self):
        """Initialize MLflow."""
        try:
            import mlflow
            self.mlflow = mlflow

            if self.tracking_uri:
                mlflow.set_tracking_uri(self.tracking_uri)

            # Set experiment
            mlflow.set_experiment(self.experiment_name)

        except ImportError:
            print("MLflow not available. Install with: pip install mlflow")

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Start a new MLflow run.

        Args:
            run_name: Optional name for the run
            tags: Optional tags for the run
        """
        if self.mlflow is None:
            return

        self.current_run = self.mlflow.start_run(run_name=run_name)

        if tags:
            self.mlflow.set_tags(tags)

    def end_run(self):
        """End the current run."""
        if self.mlflow is not None and self.current_run is not None:
            self.mlflow.end_run()
            self.current_run = None

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters.

        Args:
            params: Dictionary of parameters
        """
        if self.mlflow is None:
            return

        self.mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if self.mlflow is None:
            return

        self.mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, artifact_path: str):
        """
        Log an artifact file.

        Args:
            artifact_path: Path to artifact file
        """
        if self.mlflow is None:
            return

        self.mlflow.log_artifact(artifact_path)

    def log_model(self, model, artifact_path: str = 'model'):
        """
        Log a PyTorch model.

        Args:
            model: PyTorch model
            artifact_path: Artifact path for the model
        """
        if self.mlflow is None:
            return

        self.mlflow.pytorch.log_model(model, artifact_path)

    def log_config_to_file(self, config: Dict[str, Any], filename: str = 'config.json'):
        """
        Log configuration as a JSON artifact.

        Args:
            config: Configuration dictionary
            filename: Filename for the config
        """
        if self.mlflow is None:
            return

        # Save to temp file
        temp_path = Path(filename)
        with open(temp_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Log as artifact
        self.mlflow.log_artifact(str(temp_path))

        # Clean up
        temp_path.unlink()
