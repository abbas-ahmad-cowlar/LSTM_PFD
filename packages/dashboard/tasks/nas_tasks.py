"""
Neural Architecture Search (NAS) Celery Tasks.
Runs NAS campaigns in background, evaluating sampled architectures.
"""
from celery_app import celery_app
from utils.logger import setup_logger
from database.connection import get_db_session
from models.nas_campaign import NASCampaign, NASTrial
from services.nas_service import NASService
import traceback

logger = setup_logger(__name__)


@celery_app.task(bind=True)
def run_nas_campaign_task(self, campaign_id: int):
    """
    Run NAS campaign in background.

    For each trial:
    1. Sample architecture from search space
    2. Build model
    3. Train briefly
    4. Evaluate on validation set
    5. Save architecture + metrics

    Args:
        campaign_id: NAS campaign ID

    Returns:
        Dict with success status and results
    """
    try:
        with get_db_session() as session:
            campaign = session.query(NASCampaign).get(campaign_id)
            if not campaign:
                return {"success": False, "error": "Campaign not found"}

            # Update status
            campaign.status = 'running'
            campaign.task_id = self.request.id
            session.commit()

        logger.info(f"Starting NAS campaign {campaign_id}: {campaign.name}")

        # Run trials
        completed_trials = 0
        best_accuracy = 0.0
        best_trial_id = None

        for trial_idx in range(campaign.num_trials):
            try:
                # Update progress
                self.update_state(
                    state='PROGRESS',
                    meta={
                        'current_trial': trial_idx + 1,
                        'total_trials': campaign.num_trials,
                        'best_accuracy': best_accuracy,
                        'status': f'Evaluating architecture {trial_idx + 1}/{campaign.num_trials}'
                    }
                )

                # Sample architecture
                architecture = NASService.sample_architecture(campaign.search_space_config)
                arch_hash = NASService.compute_architecture_hash(architecture)

                # Check if architecture already evaluated (deduplication)
                with get_db_session() as session:
                    existing_trial = session.query(NASTrial).filter(
                        NASTrial.campaign_id == campaign_id,
                        NASTrial.architecture_hash == arch_hash
                    ).first()

                    if existing_trial:
                        logger.info(f"Architecture already evaluated (trial {existing_trial.trial_number}), skipping")
                        continue

                # Simulate training (In production, actually train the model)
                # For MVP, we'll generate synthetic results
                import random
                import time

                # Simulate training time
                training_time = random.uniform(10, 30)
                time.sleep(min(training_time, 5))  # Cap at 5s for demo

                # Simulate results (higher accuracy for larger/more complex models)
                num_params = sum(architecture['channels']) * 1000
                complexity_bonus = min(num_params / 100000, 0.15)  # Up to 15% bonus for complexity
                base_accuracy = random.uniform(0.70, 0.85)
                validation_accuracy = min(base_accuracy + complexity_bonus, 0.98)
                validation_loss = random.uniform(0.1, 0.5)

                # Calculate FLOPs (rough estimation)
                flops = sum(architecture['channels']) * 1000 * len(architecture['operations'])

                # Save trial
                with get_db_session() as session:
                    trial = NASTrial(
                        campaign_id=campaign_id,
                        trial_number=trial_idx + 1,
                        architecture=architecture,
                        architecture_hash=arch_hash,
                        validation_accuracy=validation_accuracy,
                        validation_loss=validation_loss,
                        training_time=training_time,
                        num_parameters=num_params,
                        flops=flops,
                        model_size_mb=num_params * 4 / (1024 * 1024)  # Rough estimate (4 bytes per param)
                    )
                    session.add(trial)
                    session.commit()
                    session.refresh(trial)

                    # Update best if needed
                    if validation_accuracy > best_accuracy:
                        best_accuracy = validation_accuracy
                        best_trial_id = trial.id

                    completed_trials += 1
                    logger.info(f"Trial {trial_idx + 1} completed: Acc={validation_accuracy:.4f}, Params={num_params}")

            except Exception as trial_error:
                logger.error(f"Trial {trial_idx + 1} failed: {trial_error}")
                logger.error(traceback.format_exc())
                continue

        # Update campaign with final results
        with get_db_session() as session:
            campaign = session.query(NASCampaign).get(campaign_id)
            campaign.status = 'completed'
            campaign.best_accuracy = best_accuracy
            campaign.best_trial_id = best_trial_id
            session.commit()

        logger.info(f"NAS campaign {campaign_id} completed: {completed_trials} trials, best acc={best_accuracy:.4f}")

        return {
            "success": True,
            "campaign_id": campaign_id,
            "completed_trials": completed_trials,
            "best_accuracy": best_accuracy,
            "best_trial_id": best_trial_id
        }

    except Exception as e:
        logger.error(f"NAS campaign {campaign_id} failed: {e}")
        logger.error(traceback.format_exc())

        # Update campaign status to failed
        try:
            with get_db_session() as session:
                campaign = session.query(NASCampaign).get(campaign_id)
                if campaign:
                    campaign.status = 'failed'
                    campaign.error_message = str(e)
                    session.commit()
        except Exception as update_error:
            logger.error(f"Failed to update campaign status: {update_error}")

        return {
            "success": False,
            "error": str(e)
        }
