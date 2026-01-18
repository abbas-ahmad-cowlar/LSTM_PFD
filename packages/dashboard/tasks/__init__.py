"""
Celery tasks for background processing.
"""
from celery import Celery
from dashboard_config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE
import os

# Initialize Celery app with task autodiscovery
celery_app = Celery(
    'tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    # Include all task modules for autodiscovery
    include=[
        'tasks.data_generation_tasks',
        'tasks.deployment_tasks',
        'tasks.evaluation_tasks',
        'tasks.feature_tasks',
        'tasks.hpo_tasks',
        'tasks.mat_import_tasks',
        'tasks.nas_tasks',
        'tasks.testing_tasks',
        'tasks.training_tasks',
        'tasks.xai_tasks',
    ]
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Check for eager execution (useful for debugging/development without worker)
if os.getenv('CELERY_ALWAYS_EAGER', 'False').lower() == 'true':
    celery_app.conf.update(task_always_eager=True)

