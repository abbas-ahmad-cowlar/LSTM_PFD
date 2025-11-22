"""
MAT file import tasks for Phase 0 integration.
Async Celery tasks for importing MATLAB .mat files.
"""
from tasks import celery_app
from utils.logger import setup_logger
from database.connection import get_db_session
from models.dataset_import import DatasetImport, DatasetImportStatus
from integrations.phase0_adapter import Phase0Adapter
from services.notification_service import NotificationService
from models.notification_preference import EventType
import time
import traceback

logger = setup_logger(__name__)


@celery_app.task(bind=True)
def import_mat_dataset_task(self, config: dict, mat_file_paths: list):
    """
    Celery task for MAT file import.

    Args:
        config: Import configuration
            - name: Dataset name
            - import_id: Database record ID
            - output_dir: Output directory
            - signal_length: Target signal length
            - validate: Enable validation
            - auto_normalize: Enable normalization
            - output_format: 'mat', 'hdf5', or 'both'
        mat_file_paths: List of paths to uploaded MAT files

    Returns:
        Import results dictionary
    """
    task_id = self.request.id
    import_id = config.get("import_id")
    logger.info(f"Starting MAT import task {task_id} for import {import_id}")

    try:
        # Update import status
        with get_db_session() as session:
            import_job = session.query(DatasetImport).filter_by(id=import_id).first()
            if import_job:
                import_job.status = DatasetImportStatus.RUNNING
                import_job.celery_task_id = task_id
                session.commit()

        # Update task state
        self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Initializing...'})

        # Define progress callback
        def progress_callback(current, total, filename=None):
            """Update progress during import."""
            progress = int((current / total) * 100)
            status_msg = f"Processing {filename} ({current}/{total})" if filename else f"Processing {current}/{total}"

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
                    import_job = session.query(DatasetImport).filter_by(id=import_id).first()
                    if import_job:
                        import_job.progress = progress
                        session.commit()
            except Exception as e:
                logger.error(f"Failed to update progress: {e}")

        # Import MAT files using Phase0Adapter
        start_time = time.time()
        results = Phase0Adapter.import_mat_files(config, mat_file_paths, progress_callback=progress_callback)
        import_time = time.time() - start_time

        # Update import with final results
        with get_db_session() as session:
            import_job = session.query(DatasetImport).filter_by(id=import_id).first()
            if import_job:
                if results.get("success"):
                    import_job.status = DatasetImportStatus.COMPLETED
                    import_job.progress = 100
                    import_job.output_path = results.get("output_path")
                    import_job.num_signals = results.get("total_signals", 0)
                    import_job.num_files = results.get("num_files", 0)
                    import_job.duration_seconds = import_time

                    # Emit notification
                    try:
                        duration_mins = int(import_time // 60)
                        duration_secs = int(import_time % 60)
                        duration_str = f"{duration_mins}m {duration_secs}s" if duration_mins > 0 else f"{duration_secs}s"

                        failed_count = len(results.get("failed_files", []))

                        NotificationService.emit_event(
                            event_type=EventType.TRAINING_COMPLETE,  # Reuse training complete for now
                            user_id=1,  # Default user
                            data={
                                'import_id': import_id,
                                'dataset_name': import_job.name,
                                'num_signals': import_job.num_signals,
                                'num_files': import_job.num_files,
                                'failed_files': failed_count,
                                'duration': duration_str,
                                'output_path': import_job.output_path,
                                'dashboard_url': f'http://localhost:8050/data-explorer'
                            }
                        )
                    except Exception as e:
                        logger.error(f"Failed to send completion notification: {e}")

                else:
                    import_job.status = DatasetImportStatus.FAILED
                    import_job.progress = 0

                session.commit()

        logger.info(f"MAT import task {task_id} completed in {import_time:.2f}s")
        return results

    except Exception as e:
        logger.error(f"MAT import task {task_id} failed: {e}", exc_info=True)

        # Update import status to failed
        try:
            with get_db_session() as session:
                import_job = session.query(DatasetImport).filter_by(id=import_id).first()
                if import_job:
                    import_job.status = DatasetImportStatus.FAILED
                    import_job.progress = 0
                    session.commit()

                    # Emit failure notification
                    try:
                        NotificationService.emit_event(
                            event_type=EventType.TRAINING_FAILED,  # Reuse training failed
                            user_id=1,
                            data={
                                'import_id': import_id,
                                'dataset_name': import_job.name,
                                'error_message': str(e),
                                'error_suggestion': 'Check MAT file formats and try again.',
                                'dashboard_url': 'http://localhost:8050/data-generation'
                            }
                        )
                    except Exception as notif_error:
                        logger.error(f"Failed to send failure notification: {notif_error}")

        except Exception as update_error:
            logger.error(f"Failed to update import status: {update_error}")

        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise
