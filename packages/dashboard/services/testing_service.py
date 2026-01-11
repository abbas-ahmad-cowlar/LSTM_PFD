"""
Testing & QA Service.
Business logic for running tests, coverage analysis, and benchmarks.
"""
from typing import Dict, List, Optional, Any
from pathlib import Path
import subprocess
import json
import re
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger(__name__)


class TestingService:
    """Service for testing and QA operations."""

    @staticmethod
    def run_pytest(
        test_path: str = "tests/",
        markers: Optional[str] = None,
        verbose: bool = True,
        capture_output: bool = True
    ) -> Dict[str, Any]:
        """
        Run pytest programmatically.

        Args:
            test_path: Path to tests (file or directory)
            markers: Pytest markers to filter tests (e.g., "unit", "integration")
            verbose: Verbose output
            capture_output: Capture stdout/stderr

        Returns:
            Dictionary with test results
        """
        try:
            cmd = ["python", "-m", "pytest", test_path]

            if markers:
                cmd.extend(["-m", markers])

            if verbose:
                cmd.append("-v")

            # JSON output for parsing
            cmd.extend(["--json-report", "--json-report-indent=2"])

            logger.info(f"Running pytest: {' '.join(cmd)}")

            start_time = datetime.utcnow()
            result = subprocess.run(
                cmd,
                capture_output=capture_output,
                text=True,
                timeout=600  # 10 minute timeout
            )
            end_time = datetime.utcnow()

            # Parse JSON report
            report_path = Path(".report.json")
            test_results = {}

            if report_path.exists():
                with open(report_path, 'r') as f:
                    test_results = json.load(f)
                report_path.unlink()  # Clean up

            return {
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'duration_seconds': (end_time - start_time).total_seconds(),
                'test_results': test_results,
                'timestamp': start_time.isoformat()
            }

        except subprocess.TimeoutExpired:
            logger.error("Pytest execution timed out")
            return {
                'success': False,
                'error': 'Test execution timed out (10 minutes)',
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to run pytest: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    @staticmethod
    def run_coverage(
        test_path: str = "tests/",
        source_path: str = ".",
        min_coverage: float = 80.0
    ) -> Dict[str, Any]:
        """
        Run tests with coverage analysis.

        Args:
            test_path: Path to tests
            source_path: Source code path to measure coverage
            min_coverage: Minimum coverage threshold

        Returns:
            Coverage results
        """
        try:
            # Run coverage
            cmd = [
                "python", "-m", "pytest",
                test_path,
                f"--cov={source_path}",
                "--cov-report=json",
                "--cov-report=term"
            ]

            logger.info(f"Running coverage: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )

            # Parse coverage JSON
            coverage_path = Path("coverage.json")
            coverage_data = {}

            if coverage_path.exists():
                with open(coverage_path, 'r') as f:
                    coverage_data = json.load(f)

            # Extract summary
            totals = coverage_data.get('totals', {})
            percent_covered = totals.get('percent_covered', 0)

            return {
                'success': result.returncode == 0,
                'total_coverage': percent_covered,
                'meets_threshold': percent_covered >= min_coverage,
                'min_coverage': min_coverage,
                'num_statements': totals.get('num_statements', 0),
                'missing_lines': totals.get('missing_lines', 0),
                'covered_lines': totals.get('covered_lines', 0),
                'files': coverage_data.get('files', {}),
                'stdout': result.stdout,
                'stderr': result.stderr
            }

        except subprocess.TimeoutExpired:
            logger.error("Coverage execution timed out")
            return {
                'success': False,
                'error': 'Coverage execution timed out'
            }
        except Exception as e:
            logger.error(f"Failed to run coverage: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    @staticmethod
    def run_benchmarks(
        model_path: Optional[str] = None,
        api_url: Optional[str] = None,
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Run performance benchmarks.

        Args:
            model_path: Path to model for benchmarking
            api_url: API URL for latency benchmarks
            num_samples: Number of samples for benchmarks

        Returns:
            Benchmark results
        """
        try:
            cmd = [
                "python",
                "tests/benchmarks/benchmark_suite.py",
                "--output", "benchmark_results.json",
                "--num-samples", str(num_samples)
            ]

            if model_path:
                cmd.extend(["--model", model_path])

            if api_url:
                cmd.extend(["--api-url", api_url])

            logger.info(f"Running benchmarks: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )

            # Load results
            results_path = Path("benchmark_results.json")
            benchmark_data = {}

            if results_path.exists():
                with open(results_path, 'r') as f:
                    benchmark_data = json.load(f)

            return {
                'success': result.returncode == 0,
                'benchmarks': benchmark_data,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

        except subprocess.TimeoutExpired:
            logger.error("Benchmark execution timed out")
            return {
                'success': False,
                'error': 'Benchmark execution timed out'
            }
        except Exception as e:
            logger.error(f"Failed to run benchmarks: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    @staticmethod
    def parse_pytest_output(stdout: str) -> Dict[str, Any]:
        """
        Parse pytest text output.

        Args:
            stdout: Pytest stdout

        Returns:
            Parsed test statistics
        """
        try:
            stats = {
                'passed': 0,
                'failed': 0,
                'skipped': 0,
                'errors': 0,
                'warnings': 0,
                'duration': 0.0
            }

            # Extract summary line
            summary_pattern = r'(\d+) passed(?:, (\d+) failed)?(?:, (\d+) skipped)?(?:, (\d+) error)?'
            match = re.search(summary_pattern, stdout)

            if match:
                stats['passed'] = int(match.group(1) or 0)
                stats['failed'] = int(match.group(2) or 0)
                stats['skipped'] = int(match.group(3) or 0)
                stats['errors'] = int(match.group(4) or 0)

            # Extract duration
            duration_pattern = r'in ([\d.]+)s'
            duration_match = re.search(duration_pattern, stdout)
            if duration_match:
                stats['duration'] = float(duration_match.group(1))

            return stats

        except Exception as e:
            logger.error(f"Failed to parse pytest output: {e}")
            return {}

    @staticmethod
    def get_test_files() -> List[Dict[str, str]]:
        """
        Get list of available test files.

        Returns:
            List of test file info
        """
        try:
            test_dir = Path("tests")
            test_files = []

            for test_file in test_dir.rglob("test_*.py"):
                relative_path = test_file.relative_to(Path.cwd())

                # Determine test type
                if "unit" in str(test_file):
                    test_type = "Unit"
                elif "integration" in str(test_file):
                    test_type = "Integration"
                elif "benchmarks" in str(test_file):
                    test_type = "Benchmark"
                else:
                    test_type = "Other"

                test_files.append({
                    'path': str(relative_path),
                    'name': test_file.name,
                    'type': test_type
                })

            return sorted(test_files, key=lambda x: x['path'])

        except Exception as e:
            logger.error(f"Failed to get test files: {e}")
            return []

    @staticmethod
    def validate_code_quality(path: str = ".") -> Dict[str, Any]:
        """
        Run code quality checks (flake8, mypy, etc.).

        Args:
            path: Path to check

        Returns:
            Quality check results
        """
        try:
            results = {}

            # Run flake8
            try:
                flake8_result = subprocess.run(
                    ["flake8", path, "--count", "--statistics"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                results['flake8'] = {
                    'success': flake8_result.returncode == 0,
                    'output': flake8_result.stdout,
                    'errors': flake8_result.stderr
                }
            except Exception as e:
                results['flake8'] = {'error': str(e)}

            # Run pylint (optional)
            try:
                pylint_result = subprocess.run(
                    ["pylint", path, "--output-format=json"],
                    capture_output=True,
                    text=True,
                    timeout=300
                )

                if pylint_result.stdout:
                    pylint_data = json.loads(pylint_result.stdout)
                    results['pylint'] = {
                        'issues': pylint_data,
                        'count': len(pylint_data)
                    }
            except Exception as e:
                results['pylint'] = {'error': str(e)}

            return results

        except Exception as e:
            logger.error(f"Failed to validate code quality: {e}")
            return {'error': str(e)}

    @staticmethod
    def get_recent_test_history(limit: int = 10) -> List[Dict]:
        """
        Get recent test run history (from stored results).

        Args:
            limit: Number of recent runs to return

        Returns:
            List of test run summaries
        """
        try:
            # In production, this would query a database
            # For now, return placeholder
            return []

        except Exception as e:
            logger.error(f"Failed to get test history: {e}")
            return []
