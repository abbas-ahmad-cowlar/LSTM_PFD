"""
Celery tasks for background processing.
"""
from celery import Celery
from config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

# Initialize Celery app
celery_app = Celery(
    'dash_tasks',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# Check for eager execution (useful for debugging/development without worker)
import os
if os.getenv('CELERY_ALWAYS_EAGER', 'False').lower() == 'true':
    celery_app.conf.update(task_always_eager=True)
