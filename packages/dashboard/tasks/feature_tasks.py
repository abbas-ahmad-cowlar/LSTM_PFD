"""
Feature Engineering Celery Tasks.
Background tasks for feature extraction, selection, and importance analysis.
"""
from tasks import celery_app
from utils.logger import setup_logger
from services.feature_service import FeatureService
import traceback
import numpy as np

logger = setup_logger(__name__)


@celery_app.task(bind=True)
def extract_features_task(self, dataset_id: int, domain: str, config: dict = None):
    """
    Extract features from dataset in background.

    Args:
        dataset_id: Dataset ID
        domain: Feature domain ('all', 'time', 'frequency', etc.)
        config: Optional configuration

    Returns:
        Feature extraction results
    """
    task_id = self.request.id
    logger.info(f"Starting feature extraction task {task_id} for dataset {dataset_id}")

    try:
        self.update_state(state='PROGRESS', meta={
            'progress': 0.1,
            'status': 'Loading dataset...'
        })

        # Extract features
        result = FeatureService.extract_features(dataset_id, domain, config)

        if not result.get('success'):
            return {
                "success": False,
                "error": result.get('error', 'Unknown error'),
                "task_id": task_id
            }

        self.update_state(state='PROGRESS', meta={
            'progress': 0.9,
            'status': 'Finalizing results...'
        })

        # Convert numpy arrays to lists for JSON serialization
        features = result['features']
        labels = result['labels']

        logger.info(f"Feature extraction task {task_id} completed successfully")

        return {
            "success": True,
            "dataset_id": dataset_id,
            "domain": domain,
            "num_features": result['num_features'],
            "num_samples": result['num_samples'],
            "extraction_time": result['extraction_time'],
            "feature_names": result['feature_names'],
            # Store features and labels as lists (for JSON)
            "features_shape": list(features.shape),
            "labels_shape": list(labels.shape),
            "task_id": task_id
        }

    except Exception as e:
        logger.error(f"Feature extraction task {task_id} failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "task_id": task_id
        }


@celery_app.task(bind=True)
def compute_importance_task(
    self,
    dataset_id: int,
    domain: str,
    method: str = 'mutual_info'
):
    """
    Compute feature importance in background.

    Args:
        dataset_id: Dataset ID
        domain: Feature domain
        method: Importance method

    Returns:
        Feature importance results
    """
    task_id = self.request.id
    logger.info(f"Starting feature importance task {task_id}")

    try:
        self.update_state(state='PROGRESS', meta={
            'progress': 0.1,
            'status': 'Extracting features...'
        })

        # First extract features
        extraction_result = FeatureService.extract_features(dataset_id, domain)

        if not extraction_result.get('success'):
            return {
                "success": False,
                "error": extraction_result.get('error'),
                "task_id": task_id
            }

        features = extraction_result['features']
        labels = extraction_result['labels']
        feature_names = extraction_result['feature_names']

        self.update_state(state='PROGRESS', meta={
            'progress': 0.5,
            'status': f'Computing {method} importance...'
        })

        # Compute importance
        importance_result = FeatureService.compute_feature_importance(
            features, labels, feature_names, method
        )

        if not importance_result.get('success'):
            return {
                "success": False,
                "error": importance_result.get('error'),
                "task_id": task_id
            }

        logger.info(f"Feature importance task {task_id} completed successfully")

        return {
            "success": True,
            "dataset_id": dataset_id,
            "domain": domain,
            "method": method,
            "importances": importance_result['importances'],
            "sorted_features": importance_result['sorted_features'],
            "feature_names": importance_result['feature_names'],
            "task_id": task_id
        }

    except Exception as e:
        logger.error(f"Feature importance task {task_id} failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "task_id": task_id
        }


@celery_app.task(bind=True)
def select_features_task(
    self,
    dataset_id: int,
    domain: str,
    method: str,
    num_features: int = 15,
    threshold: float = 0.01
):
    """
    Select features in background.

    Args:
        dataset_id: Dataset ID
        domain: Feature domain
        method: Selection method
        num_features: Number of features to select
        threshold: Threshold for variance method

    Returns:
        Feature selection results
    """
    task_id = self.request.id
    logger.info(f"Starting feature selection task {task_id}")

    try:
        self.update_state(state='PROGRESS', meta={
            'progress': 0.1,
            'status': 'Extracting features...'
        })

        # First extract features
        extraction_result = FeatureService.extract_features(dataset_id, domain)

        if not extraction_result.get('success'):
            return {
                "success": False,
                "error": extraction_result.get('error'),
                "task_id": task_id
            }

        features = extraction_result['features']
        labels = extraction_result['labels']
        feature_names = extraction_result['feature_names']

        self.update_state(state='PROGRESS', meta={
            'progress': 0.5,
            'status': f'Selecting features using {method}...'
        })

        # Select features
        selection_result = FeatureService.select_features(
            features, labels, feature_names, method, num_features, threshold
        )

        if not selection_result.get('success'):
            return {
                "success": False,
                "error": selection_result.get('error'),
                "task_id": task_id
            }

        logger.info(f"Feature selection task {task_id} completed successfully")

        return {
            "success": True,
            "dataset_id": dataset_id,
            "domain": domain,
            "method": method,
            "selected_indices": selection_result['selected_indices'],
            "selected_names": selection_result['selected_names'],
            "num_selected": selection_result['num_selected'],
            "task_id": task_id
        }

    except Exception as e:
        logger.error(f"Feature selection task {task_id} failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "task_id": task_id
        }


@celery_app.task(bind=True)
def compute_correlation_task(
    self,
    dataset_id: int,
    domain: str
):
    """
    Compute feature correlation in background.

    Args:
        dataset_id: Dataset ID
        domain: Feature domain

    Returns:
        Correlation results
    """
    task_id = self.request.id
    logger.info(f"Starting correlation computation task {task_id}")

    try:
        self.update_state(state='PROGRESS', meta={
            'progress': 0.1,
            'status': 'Extracting features...'
        })

        # First extract features
        extraction_result = FeatureService.extract_features(dataset_id, domain)

        if not extraction_result.get('success'):
            return {
                "success": False,
                "error": extraction_result.get('error'),
                "task_id": task_id
            }

        features = extraction_result['features']
        feature_names = extraction_result['feature_names']

        self.update_state(state='PROGRESS', meta={
            'progress': 0.5,
            'status': 'Computing correlation matrix...'
        })

        # Compute correlation
        corr_result = FeatureService.compute_feature_correlation(features, feature_names)

        if not corr_result.get('success'):
            return {
                "success": False,
                "error": corr_result.get('error'),
                "task_id": task_id
            }

        logger.info(f"Correlation computation task {task_id} completed successfully")

        return {
            "success": True,
            "dataset_id": dataset_id,
            "domain": domain,
            "correlation_matrix": corr_result['correlation_matrix'],
            "feature_names": corr_result['feature_names'],
            "high_corr_pairs": corr_result['high_corr_pairs'],
            "num_features": corr_result['num_features'],
            "task_id": task_id
        }

    except Exception as e:
        logger.error(f"Correlation computation task {task_id} failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "task_id": task_id
        }
