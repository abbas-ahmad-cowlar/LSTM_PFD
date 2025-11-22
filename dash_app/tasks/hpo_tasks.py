"""
HPO Campaign tasks for Phase 11C.
Celery tasks for running hyperparameter optimization campaigns.
"""
from tasks import celery_app
from utils.logger import setup_logger
from database.connection import get_db_session
from models.hpo_campaign import HPOCampaign
from models.experiment import Experiment, ExperimentStatus
from services.hpo_service import HPOService
import traceback

logger = setup_logger(__name__)


@celery_app.task(bind=True)
def run_hpo_campaign_task(self, campaign_id: int):
    """
    Celery task for running an HPO campaign.

    Args:
        campaign_id: HPO campaign ID

    Returns:
        Campaign results dictionary
    """
    task_id = self.request.id
    logger.info(f"Starting HPO campaign task {task_id} for campaign {campaign_id}")

    try:
        # Get campaign details
        campaign = HPOService.get_campaign(campaign_id)
        if not campaign:
            raise ValueError(f"Campaign {campaign_id} not found")

        # Update campaign status to running
        HPOService.update_campaign_status(campaign_id, ExperimentStatus.RUNNING)

        # Update task state
        self.update_state(state='PROGRESS', meta={
            'progress': 0,
            'status': 'Initializing HPO campaign...',
            'trials_completed': 0,
            'trials_total': campaign['trials_total']
        })

        # Get optimization method
        method = campaign['method']
        search_space = campaign['search_space']
        budget = campaign['budget']
        num_trials = budget['max_trials']
        metric = budget.get('metric', 'val_accuracy')
        direction = budget.get('direction', 'maximize')

        # Import optuna
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna not installed. Install with: pip install optuna")

        # Create optuna study
        study = optuna.create_study(
            direction=direction,
            study_name=f"hpo_campaign_{campaign_id}"
        )

        # Track best experiment
        best_experiment_id = None
        best_score = float('-inf') if direction == 'maximize' else float('inf')

        def objective(trial):
            """Optuna objective function."""
            nonlocal best_experiment_id, best_score

            # Suggest hyperparameters based on search space
            suggested_params = {}
            for param_name, param_config in search_space.items():
                param_type = param_config.get('type')

                if param_type == 'float':
                    suggested_params[param_name] = trial.suggest_float(
                        param_name,
                        param_config['min'],
                        param_config['max'],
                        log=param_config.get('log', False)
                    )
                elif param_type == 'int':
                    suggested_params[param_name] = trial.suggest_int(
                        param_name,
                        param_config['min'],
                        param_config['max']
                    )
                elif param_type == 'categorical':
                    suggested_params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config['choices']
                    )

            # Create experiment with suggested hyperparameters
            experiment_name = f"{campaign['name']}_trial_{trial.number}"

            # Import experiment creation
            from integrations.deep_learning_adapter import DeepLearningAdapter

            # Build experiment config
            experiment_config = {
                "name": experiment_name,
                "model_type": campaign['base_model_type'],
                "dataset_id": campaign['dataset_id'],
                "hyperparameters": suggested_params,
                "num_epochs": 50,  # Can be added to search space
                "tags": [f"hpo_campaign_{campaign_id}"],
            }

            # Create experiment in database
            with get_db_session() as session:
                experiment = Experiment(
                    name=experiment_name,
                    model_type=campaign['base_model_type'],
                    dataset_id=campaign['dataset_id'],
                    config=experiment_config,
                    status=ExperimentStatus.RUNNING,
                    created_by=campaign['created_by']
                )
                session.add(experiment)
                session.flush()
                experiment_id = experiment.id
                session.commit()

            # Update experiment config with ID
            experiment_config['experiment_id'] = experiment_id

            try:
                # Train model
                logger.info(f"Training trial {trial.number} with params: {suggested_params}")

                def progress_callback(epoch, metrics):
                    """Progress callback for training."""
                    # Update task state
                    self.update_state(state='PROGRESS', meta={
                        'progress': (trial.number + epoch / 50) / num_trials,
                        'status': f'Trial {trial.number}/{num_trials} - Epoch {epoch}/50',
                        'trials_completed': trial.number,
                        'trials_total': num_trials,
                        'current_trial_params': suggested_params,
                        'current_trial_epoch': epoch
                    })

                results = DeepLearningAdapter.train(
                    experiment_config,
                    progress_callback=progress_callback
                )

                # Get metric value
                if metric == 'val_accuracy':
                    score = results.get('val_accuracy', results.get('test_accuracy', 0))
                elif metric == 'val_loss':
                    score = results.get('val_loss', results.get('test_loss', float('inf')))
                elif metric == 'f1_score':
                    score = results.get('f1_score', 0)
                else:
                    score = results.get('test_accuracy', 0)

                # Update experiment status
                with get_db_session() as session:
                    exp = session.query(Experiment).filter_by(id=experiment_id).first()
                    if exp:
                        exp.status = ExperimentStatus.COMPLETED
                        exp.metrics = {
                            "test_accuracy": results.get('test_accuracy', 0),
                            "test_loss": results.get('test_loss', 0),
                            metric: score
                        }
                        exp.duration_seconds = results.get('training_time', 0)
                        session.commit()

                # Update best experiment
                is_better = (
                    (direction == 'maximize' and score > best_score) or
                    (direction == 'minimize' and score < best_score)
                )

                if is_better:
                    best_score = score
                    best_experiment_id = experiment_id

                # Update campaign progress
                HPOService.update_campaign_progress(
                    campaign_id,
                    trials_completed=trial.number + 1,
                    best_experiment_id=best_experiment_id,
                    best_accuracy=best_score
                )

                logger.info(f"Trial {trial.number} completed with {metric}={score:.4f}")
                return score

            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)

                # Mark experiment as failed
                with get_db_session() as session:
                    exp = session.query(Experiment).filter_by(id=experiment_id).first()
                    if exp:
                        exp.status = ExperimentStatus.FAILED
                        session.commit()

                # Return worst possible score
                if direction == 'maximize':
                    return float('-inf')
                else:
                    return float('inf')

        # Run optimization
        study.optimize(objective, n_trials=num_trials)

        # Get best trial
        best_trial = study.best_trial
        best_params = study.best_params
        best_value = study.best_value

        logger.info(f"HPO campaign {campaign_id} completed")
        logger.info(f"Best {metric}: {best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")

        # Update campaign to completed
        HPOService.update_campaign_status(campaign_id, ExperimentStatus.COMPLETED)

        return {
            "success": True,
            "campaign_id": campaign_id,
            "trials_completed": num_trials,
            "best_experiment_id": best_experiment_id,
            "best_score": best_value,
            "best_params": best_params
        }

    except Exception as e:
        logger.error(f"HPO campaign task {task_id} failed: {e}", exc_info=True)

        # Update campaign status to failed
        try:
            HPOService.update_campaign_status(campaign_id, ExperimentStatus.FAILED)
        except:
            pass

        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@celery_app.task
def stop_hpo_campaign_task(campaign_id: int):
    """
    Stop a running HPO campaign.

    Args:
        campaign_id: Campaign ID

    Returns:
        Success boolean
    """
    try:
        logger.info(f"Stopping HPO campaign {campaign_id}")
        HPOService.update_campaign_status(campaign_id, ExperimentStatus.CANCELLED)
        return True
    except Exception as e:
        logger.error(f"Failed to stop campaign {campaign_id}: {e}")
        return False
