"""
API routes for task status and real-time updates.
"""
from flask import Blueprint, jsonify, request
from tasks import celery_app
from utils.logger import setup_logger

logger = setup_logger(__name__)

api_bp = Blueprint('api', __name__, url_prefix='/api')


@api_bp.route('/task-status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get status of a Celery task."""
    try:
        task = celery_app.AsyncResult(task_id)

        if task.state == 'PENDING':
            response = {
                'state': task.state,
                'status': 'Pending...'
            }
        elif task.state == 'PROGRESS':
            response = {
                'state': task.state,
                'progress': task.info.get('progress', 0),
                'status': task.info.get('status', ''),
                **task.info
            }
        elif task.state == 'SUCCESS':
            response = {
                'state': task.state,
                'result': task.result
            }
        else:  # FAILURE or other states
            response = {
                'state': task.state,
                'error': str(task.info)
            }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/task-cancel/<task_id>', methods=['POST'])
def cancel_task(task_id):
    """Cancel a running task."""
    try:
        celery_app.control.revoke(task_id, terminate=True)
        return jsonify({'status': 'cancelled', 'task_id': task_id})
    except Exception as e:
        logger.error(f"Error cancelling task: {e}")
        return jsonify({'error': str(e)}), 500
