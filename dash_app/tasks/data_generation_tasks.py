"""
Data generation tasks for Phase 0 integration.
Async Celery tasks for generating synthetic vibration datasets.
"""
from tasks import celery_app
from utils.logger import setup_logger
from database.connection import get_db_session
from models.dataset_generation import DatasetGeneration, DatasetGenerationStatus
from integrations.phase0_adapter import Phase0Adapter
from services.notification_service import NotificationService
from models.notification_preference import EventType
import time
import traceback

logger = setup_logger(__name__)


@celery_app.task(bind=True)
def generate_dataset_task(self, config: dict):
    """
    Celery task for dataset generation.

    Args:
        config: Generation configuration
            - name: Dataset name
            - generation_id: Database record ID
            - num_signals_per_fault: Signals per fault type
            - fault_types: List of fault types to generate
            - severity_levels: List of severity levels
            - noise_layers: Dict of noise layer enables
            - augmentation: Augmentation config
            - output_format: 'mat', 'hdf5', or 'both'
            - random_seed: Random seed for reproducibility

    Returns:
        Generation results dictionary
    """
    task_id = self.request.id
    generation_id = config.get("generation_id")
    logger.info(f"Starting data generation task {task_id} for generation {generation_id}")

    try:
        # Update generation status
        with get_db_session() as session:
            generation = session.query(DatasetGeneration).filter_by(id=generation_id).first()
            if generation:
                generation.status = DatasetGenerationStatus.RUNNING
                generation.celery_task_id = task_id
                session.commit()

        # Update task state
        self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Initializing...'})

        # Define progress callback
        def progress_callback(current, total, fault_type=None):
            """Update progress during generation."""
            progress = int((current / total) * 100)
            status_msg = f"Generating {fault_type} ({current}/{total})" if fault_type else f"Processing {current}/{total}"

            self.update_state(
                state='PROGRESS',
                meta={
                    'progress': progress,
                    'current': current,
                    'total': total,
                    'status': status_msg
                }
            )

            # Update database
            try:
                with get_db_session() as session:
                    generation = session.query(DatasetGeneration).filter_by(id=generation_id).first()
                    if generation:
                        generation.progress = progress
                        session.commit()
            except Exception as e:
                logger.error(f"Failed to update progress: {e}")

        # Generate dataset using Phase0Adapter
        start_time = time.time()
        results = Phase0Adapter.generate_dataset(config, progress_callback=progress_callback)
        generation_time = time.time() - start_time

        # Update generation with final results
        with get_db_session() as session:
            generation = session.query(DatasetGeneration).filter_by(id=generation_id).first()
            if generation:
                if results.get("success"):
                    generation.status = DatasetGenerationStatus.COMPLETED
                    generation.progress = 100
                    generation.output_path = results.get("output_path")
                    generation.num_signals = results.get("total_signals", 0)
                    generation.num_faults = results.get("num_faults", 0)
                    generation.duration_seconds = generation_time

                    # Emit notification
                    try:
                        duration_mins = int(generation_time // 60)
                        duration_secs = int(generation_time % 60)
                        duration_str = f"{duration_mins}m {duration_secs}s" if duration_mins > 0 else f"{duration_secs}s"

                        NotificationService.emit_event(
                            event_type=EventType.TRAINING_COMPLETE,  # Reuse training complete for now
                            user_id=config.get('user_id', 1),  # Get user_id from config or default to 1 for background tasks
                            data={
                                'generation_id': generation_id,
                                'dataset_name': generation.name,
                                'num_signals': generation.num_signals,
                                'num_faults': generation.num_faults,
                                'duration': duration_str,
                                'output_path': generation.output_path,
                                'dashboard_url': f'http://localhost:8050/data-explorer'
                            }
                        )
                    except Exception as e:
                        logger.error(f"Failed to send completion notification: {e}")

                else:
                    generation.status = DatasetGenerationStatus.FAILED
                    generation.progress = 0

                session.commit()

        logger.info(f"Data generation task {task_id} completed in {generation_time:.2f}s")
        return results

    except Exception as e:
        logger.error(f"Data generation task {task_id} failed: {e}", exc_info=True)

        # Update generation status to failed
        try:
            with get_db_session() as session:
                generation = session.query(DatasetGeneration).filter_by(id=generation_id).first()
                if generation:
                    generation.status = DatasetGenerationStatus.FAILED
                    generation.progress = 0
                    session.commit()

                    # Emit failure notification
                    try:
                        NotificationService.emit_event(
                            event_type=EventType.TRAINING_FAILED,  # Reuse training failed
                            user_id=config.get('user_id', 1),  # Get user_id from config or default to 1 for background tasks
                            data={
                                'generation_id': generation_id,
                                'dataset_name': generation.name,
                                'error_message': str(e),
                                'error_suggestion': 'Check logs for details and verify configuration.',
                                'dashboard_url': 'http://localhost:8050/data-generation'
                            }
                        )
                    except Exception as notif_error:
                        logger.error(f"Failed to send failure notification: {notif_error}")

        except Exception as update_error:
            logger.error(f"Failed to update generation status: {update_error}")

        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise
