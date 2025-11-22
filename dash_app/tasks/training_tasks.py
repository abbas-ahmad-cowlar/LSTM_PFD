"""
Training tasks for Phase 11B.
Real training execution using phase adapters.
"""
from tasks import celery_app
from utils.logger import setup_logger
from database.connection import get_db_session
from models.experiment import Experiment, ExperimentStatus
from models.training_run import TrainingRun
from services.notification_service import NotificationService, get_error_suggestion
from models.notification_preference import EventType
import time
import traceback

logger = setup_logger(__name__)


@celery_app.task(bind=True)
def train_model_task(self, config: dict):
    """
    Celery task for model training.

    Args:
        config: Training configuration
            - model_type: Model type to train
            - dataset_id: Dataset ID
            - hyperparameters: Training hyperparameters
            - experiment_id: Experiment ID in database

    Returns:
        Training results dictionary
    """
    task_id = self.request.id
    experiment_id = config.get("experiment_id")
    logger.info(f"Starting training task {task_id} for experiment {experiment_id}")

    try:
        # Update experiment status
        with get_db_session() as session:
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
            if experiment:
                experiment.status = ExperimentStatus.RUNNING
                experiment.config["celery_task_id"] = task_id
                session.commit()

        # Update task state
        self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Initializing...'})

        # Determine which adapter to use based on model type
        model_type = config["model_type"]

        # Classical ML models (Phase 1)
        if model_type in ["rf", "svm", "gbm"]:
            from integrations.phase1_adapter import Phase1Adapter

            def progress_callback(epoch, metrics):
                self.update_state(state='PROGRESS', meta=metrics)
                # Save to database
                _save_training_run(experiment_id, epoch, metrics)

            results = Phase1Adapter.train(config, progress_callback=progress_callback)

        # Deep learning models (Phases 2-8)
        else:
            from integrations.deep_learning_adapter import DeepLearningAdapter
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

            def progress_callback(epoch, metrics):
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'epoch': epoch,
                        'progress': epoch / config.get('num_epochs', 100),
                        **metrics
                    }
                )
                # Save to database
                _save_training_run(experiment_id, epoch, metrics)

            results = DeepLearningAdapter.train(config, progress_callback=progress_callback)

        # Update experiment with final results
        with get_db_session() as session:
            experiment = session.query(Experiment).filter_by(id=experiment_id).first()
            if experiment:
                if results.get("success"):
                    experiment.status = ExperimentStatus.COMPLETED
                    experiment.metrics = {
                        "test_accuracy": results.get("test_accuracy", 0),
                        "test_loss": results.get("test_loss", 0),
                        "best_val_loss": results.get("best_val_loss", 0),
                    }
                    experiment.total_epochs = results.get("total_epochs", config.get("num_epochs", 1))
                    experiment.best_epoch = results.get("best_epoch", 1)
                    experiment.duration_seconds = results.get("training_time", 0)

                    # EMIT EMAIL NOTIFICATION: Training Complete
                    try:
                        duration_mins = int(experiment.duration_seconds // 60)
                        duration_secs = int(experiment.duration_seconds % 60)
                        duration_str = f"{duration_mins}m {duration_secs}s" if duration_mins > 0 else f"{duration_secs}s"

                        NotificationService.emit_event(
                            event_type=EventType.TRAINING_COMPLETE,
                            user_id=experiment.created_by or 1,  # Default to user 1 if not set
                            data={
                                'experiment_id': experiment_id,
                                'experiment_name': experiment.name,
                                'accuracy': experiment.metrics.get('test_accuracy', 0),
                                'precision': results.get('precision', 0),
                                'recall': results.get('recall', 0),
                                'f1_score': results.get('f1_score', 0),
                                'duration': duration_str,
                                'total_epochs': experiment.total_epochs,
                                'results_url': f"http://localhost:8050/experiments/{experiment_id}/results",
                                'dashboard_url': 'http://localhost:8050'
                            }
                        )
                    except Exception as e:
                        logger.error(f"Failed to send training complete notification: {e}")

                else:
                    experiment.status = ExperimentStatus.FAILED
                session.commit()

        logger.info(f"Training task {task_id} completed successfully")
        return results

    except Exception as e:
        logger.error(f"Training task {task_id} failed: {e}", exc_info=True)

        # Update experiment status to failed
        try:
            with get_db_session() as session:
                experiment = session.query(Experiment).filter_by(id=experiment_id).first()
                if experiment:
                    experiment.status = ExperimentStatus.FAILED
                    session.commit()

                    # EMIT EMAIL NOTIFICATION: Training Failed
                    try:
                        error_msg = str(e)
                        suggestion = get_error_suggestion(error_msg)

                        NotificationService.emit_event(
                            event_type=EventType.TRAINING_FAILED,
                            user_id=experiment.created_by or 1,
                            data={
                                'experiment_id': experiment_id,
                                'experiment_name': experiment.name,
                                'error_message': error_msg,
                                'error_suggestion': suggestion,
                                'error_details_url': f"http://localhost:8050/experiments/{experiment_id}/logs",
                                'new_training_url': 'http://localhost:8050/training/new',
                                'troubleshooting_url': 'http://localhost:8050/help/troubleshooting',
                                'dashboard_url': 'http://localhost:8050'
                            }
                        )
                    except Exception as notif_error:
                        logger.error(f"Failed to send training failed notification: {notif_error}")

        except Exception as update_error:
            logger.error(f"Failed to update experiment status: {update_error}")

        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


def _save_training_run(experiment_id, epoch, metrics):
    """Save training run metrics to database."""
    try:
        with get_db_session() as session:
            training_run = TrainingRun(
                experiment_id=experiment_id,
                epoch=epoch,
                train_loss=metrics.get("train_loss", 0),
                val_loss=metrics.get("val_loss", 0),
                train_accuracy=metrics.get("train_accuracy", 0),
                val_accuracy=metrics.get("val_accuracy", 0),
                learning_rate=metrics.get("learning_rate", 0),
                duration_seconds=metrics.get("epoch_time", 0),
            )
            session.add(training_run)
            session.commit()
    except Exception as e:
        logger.error(f"Failed to save training run: {e}")
