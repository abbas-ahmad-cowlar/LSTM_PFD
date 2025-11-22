"""
XAI explanation generation tasks.
Async Celery tasks for generating model explanations (SHAP, LIME, IG, Grad-CAM).
"""
from tasks import celery_app
from utils.logger import setup_logger
from database.connection import get_db_session
from models.explanation import Explanation
from services.xai_service import XAIService
from utils.signal_loader import SignalLoader
from services.notification_service import NotificationService
from models.notification_preference import EventType
import time

logger = setup_logger(__name__)


@celery_app.task(bind=True)
def generate_explanation_task(self, config: dict):
    """
    Celery task for generating XAI explanations.

    Args:
        config: Explanation configuration
            - experiment_id: int
            - signal_id: int
            - method: 'shap'|'lime'|'integrated_gradients'|'gradcam'
            - params: dict with method-specific parameters
                - shap_method: 'gradient'|'deep'|'kernel'
                - num_samples: int (for SHAP gradient)
                - num_segments: int (for LIME)
                - num_perturbations: int (for LIME)
                - ig_steps: int (for Integrated Gradients)

    Returns:
        Explanation results dictionary
    """
    task_id = self.request.id
    experiment_id = config.get('experiment_id')
    signal_id = config.get('signal_id')
    method = config.get('method')

    logger.info(f"Starting XAI explanation task {task_id}: {method} for experiment {experiment_id}, signal {signal_id}")

    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={'progress': 10, 'status': 'Loading model...'})

        # Load model from experiment
        model = XAIService.load_model(experiment_id)
        if model is None:
            raise ValueError(f"Failed to load model for experiment {experiment_id}")

        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 20, 'status': 'Loading signal...'})

        # Load signal
        signal = SignalLoader.load_signal_by_id(signal_id)
        if signal is None:
            raise ValueError(f"Failed to load signal {signal_id}")

        # Get signal metadata
        signal_metadata = SignalLoader.get_signal_metadata(signal_id)

        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 40, 'status': f'Generating {method} explanation...'})

        # Generate explanation based on method
        start_time = time.time()

        if method == 'shap':
            # Load background data for SHAP
            background_data = SignalLoader.load_background_data(
                experiment_id,
                num_samples=config.get('params', {}).get('num_samples', 100)
            )

            result = XAIService.generate_shap_explanation(
                model=model,
                signal=signal,
                background_data=background_data,
                method=config.get('params', {}).get('shap_method', 'gradient'),
                num_samples=config.get('params', {}).get('num_samples', 100)
            )

        elif method == 'lime':
            result = XAIService.generate_lime_explanation(
                model=model,
                signal=signal,
                num_segments=config.get('params', {}).get('num_segments', 20),
                num_samples=config.get('params', {}).get('num_perturbations', 1000),
                target_class=None
            )

        elif method == 'integrated_gradients':
            result = XAIService.generate_integrated_gradients(
                model=model,
                signal=signal,
                baseline=None,
                steps=config.get('params', {}).get('ig_steps', 50)
            )

        elif method == 'gradcam':
            result = XAIService.generate_gradcam(
                model=model,
                signal=signal,
                target_layer=None
            )

        else:
            raise ValueError(f"Unknown XAI method: {method}")

        generation_time = time.time() - start_time

        # Check for errors
        if not result.get('success', False):
            raise Exception(result.get('error', 'Unknown error during explanation generation'))

        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 80, 'status': 'Caching explanation...'})

        # Cache explanation to database
        try:
            with get_db_session() as session:
                # Check if explanation already exists
                existing = session.query(Explanation).filter_by(
                    experiment_id=experiment_id,
                    signal_id=str(signal_id),
                    method=method
                ).first()

                if existing:
                    # Update existing
                    existing.explanation_data = result
                    session.commit()
                    explanation_db_id = existing.id
                    logger.info(f"Updated existing explanation {explanation_db_id}")
                else:
                    # Create new
                    explanation = Explanation(
                        experiment_id=experiment_id,
                        signal_id=str(signal_id),
                        method=method,
                        explanation_data=result
                    )
                    session.add(explanation)
                    session.commit()
                    explanation_db_id = explanation.id
                    logger.info(f"Created new explanation {explanation_db_id}")

        except Exception as cache_error:
            logger.error(f"Failed to cache explanation: {cache_error}", exc_info=True)
            # Continue even if caching fails

        # Update progress
        self.update_state(state='PROGRESS', meta={'progress': 95, 'status': 'Sending notification...'})

        # Send notification
        try:
            NotificationService.emit_event(
                event_type=EventType.TRAINING_COMPLETE,  # Reuse for XAI completion
                user_id=config.get('user_id', 1),  # Get user_id from config or default to 1 for background tasks
                data={
                    'experiment_id': experiment_id,
                    'signal_id': signal_id,
                    'method': method.upper(),
                    'predicted_class': result.get('predicted_class'),
                    'confidence': result.get('confidence', 0),
                    'generation_time': f"{generation_time:.2f}s",
                    'signal_metadata': signal_metadata,
                    'dashboard_url': f'http://localhost:8050/xai?exp={experiment_id}&sig={signal_id}'
                }
            )
        except Exception as notif_error:
            logger.error(f"Failed to send notification: {notif_error}")

        # Final result
        logger.info(f"XAI explanation task {task_id} completed in {generation_time:.2f}s")

        return {
            'success': True,
            'task_id': task_id,
            'experiment_id': experiment_id,
            'signal_id': signal_id,
            'method': method,
            'result': result,
            'generation_time': generation_time,
            'explanation_db_id': explanation_db_id if 'explanation_db_id' in locals() else None
        }

    except Exception as e:
        logger.error(f"XAI explanation task {task_id} failed: {e}", exc_info=True)

        # Send failure notification
        try:
            NotificationService.emit_event(
                event_type=EventType.TRAINING_FAILED,  # Reuse for XAI failure
                user_id=config.get('user_id', 1),  # Get user_id from config or default to 1 for background tasks
                data={
                    'experiment_id': experiment_id,
                    'signal_id': signal_id,
                    'method': method.upper(),
                    'error_message': str(e),
                    'error_suggestion': 'Check model compatibility and signal format.',
                    'dashboard_url': 'http://localhost:8050/xai'
                }
            )
        except Exception as notif_error:
            logger.error(f"Failed to send failure notification: {notif_error}")

        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise


@celery_app.task(bind=True)
def generate_batch_explanations_task(self, config: dict):
    """
    Generate explanations for multiple signals.

    Args:
        config: Batch configuration
            - experiment_id: int
            - signal_ids: List[int]
            - method: str
            - params: dict

    Returns:
        Batch results
    """
    task_id = self.request.id
    experiment_id = config.get('experiment_id')
    signal_ids = config.get('signal_ids', [])
    method = config.get('method')

    logger.info(f"Starting batch XAI task {task_id}: {len(signal_ids)} signals for {method}")

    try:
        results = []
        total = len(signal_ids)

        for i, signal_id in enumerate(signal_ids):
            progress = int((i / total) * 100)
            self.update_state(
                state='PROGRESS',
                meta={
                    'progress': progress,
                    'current': i + 1,
                    'total': total,
                    'status': f'Processing signal {i+1}/{total}'
                }
            )

            # Generate individual explanation
            individual_config = {
                'experiment_id': experiment_id,
                'signal_id': signal_id,
                'method': method,
                'params': config.get('params', {})
            }

            try:
                result = generate_explanation_task(individual_config)
                results.append({
                    'signal_id': signal_id,
                    'success': True,
                    'result': result
                })
            except Exception as e:
                logger.error(f"Failed to generate explanation for signal {signal_id}: {e}")
                results.append({
                    'signal_id': signal_id,
                    'success': False,
                    'error': str(e)
                })

        logger.info(f"Batch XAI task {task_id} completed: {len(results)} explanations")

        return {
            'success': True,
            'task_id': task_id,
            'total_signals': total,
            'successful': len([r for r in results if r['success']]),
            'failed': len([r for r in results if not r['success']]),
            'results': results
        }

    except Exception as e:
        logger.error(f"Batch XAI task {task_id} failed: {e}", exc_info=True)
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise
