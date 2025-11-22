"""
Evaluation tasks for Enhanced Evaluation Dashboard.
Celery tasks for CPU-intensive evaluation computations.
"""
from tasks import celery_app
from utils.logger import setup_logger
from services.evaluation_service import EvaluationService
import traceback

logger = setup_logger(__name__)


@celery_app.task(bind=True)
def generate_roc_analysis_task(self, experiment_id: int):
    """
    Generate ROC analysis for an experiment.

    Args:
        experiment_id: Experiment ID

    Returns:
        ROC analysis results
    """
    task_id = self.request.id
    logger.info(f"Starting ROC analysis task {task_id} for experiment {experiment_id}")

    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={
            'progress': 0.1,
            'status': 'Loading experiment predictions...'
        })

        # Load predictions
        data = EvaluationService.load_experiment_predictions(experiment_id)
        if not data:
            raise ValueError(f"Failed to load predictions for experiment {experiment_id}")

        self.update_state(state='PROGRESS', meta={
            'progress': 0.4,
            'status': 'Computing ROC curves...'
        })

        # Generate ROC data
        roc_data = EvaluationService.generate_roc_data(
            data['probabilities'],
            data['targets'],
            data['class_names']
        )

        self.update_state(state='PROGRESS', meta={
            'progress': 0.9,
            'status': 'Finalizing results...'
        })

        # Cache results
        EvaluationService.cache_evaluation_results(experiment_id, 'roc', roc_data)

        logger.info(f"ROC analysis task {task_id} completed successfully")

        return {
            "success": True,
            "experiment_id": experiment_id,
            "experiment_name": data['experiment_name'],
            "roc_data": roc_data,
            "num_classes": len(data['class_names']),
            "class_names": data['class_names']
        }

    except Exception as e:
        logger.error(f"ROC analysis task {task_id} failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@celery_app.task(bind=True)
def error_analysis_task(self, experiment_id: int):
    """
    Perform error analysis for an experiment.

    Args:
        experiment_id: Experiment ID

    Returns:
        Error analysis results
    """
    task_id = self.request.id
    logger.info(f"Starting error analysis task {task_id} for experiment {experiment_id}")

    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={
            'progress': 0.1,
            'status': 'Loading experiment predictions...'
        })

        # Load predictions
        data = EvaluationService.load_experiment_predictions(experiment_id)
        if not data:
            raise ValueError(f"Failed to load predictions for experiment {experiment_id}")

        self.update_state(state='PROGRESS', meta={
            'progress': 0.4,
            'status': 'Analyzing errors...'
        })

        # Analyze errors
        error_data = EvaluationService.analyze_errors(
            data['predictions'],
            data['probabilities'],
            data['targets'],
            data['class_names']
        )

        self.update_state(state='PROGRESS', meta={
            'progress': 0.9,
            'status': 'Finalizing results...'
        })

        # Cache results
        EvaluationService.cache_evaluation_results(experiment_id, 'error_analysis', error_data)

        logger.info(f"Error analysis task {task_id} completed successfully")

        return {
            "success": True,
            "experiment_id": experiment_id,
            "experiment_name": data['experiment_name'],
            "error_data": error_data,
            "class_names": data['class_names']
        }

    except Exception as e:
        logger.error(f"Error analysis task {task_id} failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@celery_app.task(bind=True)
def architecture_comparison_task(self, experiment_ids: list):
    """
    Compare multiple architectures.

    Args:
        experiment_ids: List of experiment IDs

    Returns:
        Architecture comparison results
    """
    task_id = self.request.id
    logger.info(f"Starting architecture comparison task {task_id}")

    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={
            'progress': 0.3,
            'status': 'Comparing architectures...'
        })

        # Compare architectures
        comparison = EvaluationService.compare_architectures(experiment_ids)

        logger.info(f"Architecture comparison task {task_id} completed successfully")

        return {
            "success": True,
            "comparison": comparison
        }

    except Exception as e:
        logger.error(f"Architecture comparison task {task_id} failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
