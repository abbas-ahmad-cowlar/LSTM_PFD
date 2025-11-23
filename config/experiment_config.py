"""
Experiment tracking configuration for MLflow.

Purpose:
    Configuration for experiment tracking and management:
    - MLflow tracking setup
    - Experiment naming and tagging
    - Logging configuration
    - Artifact storage

Author: Syed Abbas Ahmad
Date: 2025-11-19
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from config.base_config import BaseConfig


@dataclass
class MLflowConfig(BaseConfig):
    """
    MLflow tracking configuration.

    Example:
        >>> config = MLflowConfig(
        ...     tracking_uri='./mlruns',
        ...     experiment_name='bearing_fault_diagnosis'
        ... )
    """
    # Tracking server
    tracking_uri: str = './mlruns'
    registry_uri: Optional[str] = None

    # Experiment settings
    experiment_name: str = 'bearing_fault_diagnosis'
    run_name: Optional[str] = None  # Auto-generated if None

    # Tags
    tags: Dict[str, str] = field(default_factory=dict)

    # Auto-logging
    autolog: bool = True
    log_models: bool = True
    log_every_n_steps: int = 10

    # Artifact storage
    artifact_location: Optional[str] = None

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tracking_uri": {"type": "string"},
                "experiment_name": {"type": "string", "minLength": 1},
                "autolog": {"type": "boolean"},
                "log_every_n_steps": {"type": "integer", "minimum": 1}
            }
        }


@dataclass
class LoggingConfig(BaseConfig):
    """
    Configuration for what to log during experiments.

    Example:
        >>> config = LoggingConfig(
        ...     log_params=True,
        ...     log_metrics=True,
        ...     log_artifacts=True
        ... )
    """
    # What to log
    log_params: bool = True
    log_metrics: bool = True
    log_artifacts: bool = True
    log_models: bool = True
    log_system_info: bool = True

    # Metrics logging
    metrics_to_log: List[str] = field(default_factory=lambda: [
        'loss', 'accuracy', 'f1_score', 'precision', 'recall'
    ])
    log_confusion_matrix: bool = True
    log_roc_curves: bool = True

    # Artifact logging
    artifacts_to_log: List[str] = field(default_factory=lambda: [
        'config', 'model', 'plots', 'checkpoints'
    ])
    save_best_model: bool = True
    save_final_model: bool = True

    # Plotting
    log_training_plots: bool = True
    log_evaluation_plots: bool = True
    plot_frequency: int = 5  # Plot every N epochs

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "log_params": {"type": "boolean"},
                "log_metrics": {"type": "boolean"},
                "log_artifacts": {"type": "boolean"},
                "plot_frequency": {"type": "integer", "minimum": 1}
            }
        }


@dataclass
class ExperimentConfig(BaseConfig):
    """
    Master experiment configuration combining MLflow and logging settings.

    Example:
        >>> config = ExperimentConfig(
        ...     experiment_name='cnn_baseline',
        ...     description='1D CNN baseline model'
        ... )
        >>> config.to_yaml('configs/experiment.yaml')
    """
    # Experiment metadata
    experiment_name: str = 'default_experiment'
    description: str = ''
    tags: Dict[str, str] = field(default_factory=dict)

    # MLflow configuration
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    # Logging configuration
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Reproducibility
    seed: int = 42
    deterministic: bool = True

    # Checkpointing
    checkpoint_dir: str = 'checkpoints'
    checkpoint_frequency: int = 5  # Save every N epochs
    keep_n_checkpoints: int = 3  # Keep only last N checkpoints

    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_metric: str = 'val_loss'
    early_stopping_mode: str = 'min'  # 'min' or 'max'

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "experiment_name": {"type": "string", "minLength": 1},
                "description": {"type": "string"},
                "seed": {"type": "integer", "minimum": 0},
                "checkpoint_frequency": {"type": "integer", "minimum": 1},
                "early_stopping_patience": {"type": "integer", "minimum": 1}
            }
        }

    def get_run_name(self, model_name: str, timestamp: Optional[str] = None) -> str:
        """
        Generate run name for MLflow.

        Args:
            model_name: Model architecture name
            timestamp: Optional timestamp string

        Returns:
            Run name string

        Example:
            >>> run_name = config.get_run_name('cnn1d', '20231119_143022')
            >>> print(run_name)  # 'cnn1d_20231119_143022'
        """
        if timestamp is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        return f"{model_name}_{timestamp}"

    def get_tags_with_defaults(self, **additional_tags) -> Dict[str, str]:
        """
        Get tags with default system tags.

        Args:
            **additional_tags: Additional tags to merge

        Returns:
            Complete tags dictionary

        Example:
            >>> tags = config.get_tags_with_defaults(model='cnn1d', lr=0.001)
        """
        import platform

        default_tags = {
            'experiment': self.experiment_name,
            'seed': str(self.seed),
            'platform': platform.system(),
            'python_version': platform.python_version(),
        }

        # Merge: default < config < additional
        all_tags = {**default_tags, **self.tags, **additional_tags}

        return all_tags
