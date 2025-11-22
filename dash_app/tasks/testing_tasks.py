"""
Testing & QA Celery Tasks.
Background tasks for running tests, coverage, and benchmarks.
"""
from tasks import celery_app
from utils.logger import setup_logger
from services.testing_service import TestingService
import traceback

logger = setup_logger(__name__)


@celery_app.task(bind=True)
def run_tests_task(self, test_path: str = "tests/", markers: str = None):
    """
    Run pytest in background.

    Args:
        test_path: Path to tests
        markers: Pytest markers to filter

    Returns:
        Test results
    """
    task_id = self.request.id
    logger.info(f"Starting test execution task {task_id}")

    try:
        self.update_state(state='PROGRESS', meta={
            'progress': 0.1,
            'status': 'Initializing test runner...'
        })

        # Run tests
        results = TestingService.run_pytest(
            test_path=test_path,
            markers=markers,
            verbose=True
        )

        self.update_state(state='PROGRESS', meta={
            'progress': 0.9,
            'status': 'Parsing test results...'
        })

        # Parse output
        if results.get('stdout'):
            stats = TestingService.parse_pytest_output(results['stdout'])
            results['stats'] = stats

        logger.info(f"Test execution task {task_id} completed")

        return {
            "success": results.get('success', False),
            "results": results,
            "task_id": task_id
        }

    except Exception as e:
        logger.error(f"Test execution task {task_id} failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@celery_app.task(bind=True)
def run_coverage_task(
    self,
    test_path: str = "tests/",
    source_path: str = ".",
    min_coverage: float = 80.0
):
    """
    Run coverage analysis in background.

    Args:
        test_path: Path to tests
        source_path: Source code path
        min_coverage: Minimum coverage threshold

    Returns:
        Coverage results
    """
    task_id = self.request.id
    logger.info(f"Starting coverage analysis task {task_id}")

    try:
        self.update_state(state='PROGRESS', meta={
            'progress': 0.2,
            'status': 'Running tests with coverage...'
        })

        # Run coverage
        results = TestingService.run_coverage(
            test_path=test_path,
            source_path=source_path,
            min_coverage=min_coverage
        )

        self.update_state(state='PROGRESS', meta={
            'progress': 0.9,
            'status': 'Generating coverage report...'
        })

        logger.info(f"Coverage analysis task {task_id} completed")

        return {
            "success": results.get('success', False),
            "coverage_data": results,
            "task_id": task_id
        }

    except Exception as e:
        logger.error(f"Coverage analysis task {task_id} failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@celery_app.task(bind=True)
def run_benchmarks_task(
    self,
    model_path: str = None,
    api_url: str = None,
    num_samples: int = 100
):
    """
    Run performance benchmarks in background.

    Args:
        model_path: Path to model
        api_url: API URL for latency benchmarks
        num_samples: Number of samples

    Returns:
        Benchmark results
    """
    task_id = self.request.id
    logger.info(f"Starting benchmarks task {task_id}")

    try:
        self.update_state(state='PROGRESS', meta={
            'progress': 0.1,
            'status': 'Initializing benchmarks...'
        })

        # Run benchmarks
        results = TestingService.run_benchmarks(
            model_path=model_path,
            api_url=api_url,
            num_samples=num_samples
        )

        self.update_state(state='PROGRESS', meta={
            'progress': 0.9,
            'status': 'Finalizing benchmark results...'
        })

        logger.info(f"Benchmarks task {task_id} completed")

        return {
            "success": results.get('success', False),
            "benchmark_data": results,
            "task_id": task_id
        }

    except Exception as e:
        logger.error(f"Benchmarks task {task_id} failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@celery_app.task(bind=True)
def run_quality_checks_task(self, path: str = "."):
    """
    Run code quality checks in background.

    Args:
        path: Path to check

    Returns:
        Quality check results
    """
    task_id = self.request.id
    logger.info(f"Starting quality checks task {task_id}")

    try:
        self.update_state(state='PROGRESS', meta={
            'progress': 0.3,
            'status': 'Running code quality checks...'
        })

        # Run quality checks
        results = TestingService.validate_code_quality(path=path)

        logger.info(f"Quality checks task {task_id} completed")

        return {
            "success": True,
            "quality_data": results,
            "task_id": task_id
        }

    except Exception as e:
        logger.error(f"Quality checks task {task_id} failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
