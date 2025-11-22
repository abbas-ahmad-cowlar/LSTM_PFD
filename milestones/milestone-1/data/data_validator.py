"""
Validation utilities for comparing Python vs MATLAB signal generation.

Purpose:
    Ensure Python implementation matches MATLAB generator.m within 1% tolerance.
    Performs statistical and signal-level comparisons to validate correctness.

Author: LSTM_PFD Team
Date: 2025-11-19
"""

from utils.constants import SAMPLING_RATE
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import scipy.stats as stats
from scipy.signal import correlate, coherence

from utils.logging import get_logger
from data.matlab_importer import MatlabSignalData, MatlabImporter

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """
    Container for validation comparison results.

    Attributes:
        passed: Whether validation passed
        tolerance: Tolerance threshold used
        metrics: Dictionary of comparison metrics
        errors: List of validation errors
        warnings: List of warnings
    """
    passed: bool
    tolerance: float
    metrics: Dict[str, float]
    errors: List[str]
    warnings: List[str]

    def __repr__(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return f"ValidationResult({status}, {len(self.errors)} errors, {len(self.warnings)} warnings)"


class SignalValidator:
    """
    Validate Python-generated signals against MATLAB reference.

    Performs multiple levels of validation:
    1. Statistical comparison (mean, std, RMS, percentiles)
    2. Time-domain correlation
    3. Frequency-domain coherence
    4. Point-wise error analysis

    Example:
        >>> validator = SignalValidator(tolerance=0.01)
        >>> result = validator.compare_signals(python_signal, matlab_signal)
        >>> if result.passed:
        ...     print("✅ Validation passed!")
        >>> else:
        ...     print(f"❌ Errors: {result.errors}")
    """

    def __init__(
        self,
        tolerance: float = 0.01,
        correlation_threshold: float = 0.99,
        coherence_threshold: float = 0.95
    ):
        """
        Initialize validator with tolerance settings.

        Args:
            tolerance: Maximum allowed relative error (default: 1%)
            correlation_threshold: Minimum correlation coefficient (default: 0.99)
            coherence_threshold: Minimum coherence (default: 0.95)
        """
        self.tolerance = tolerance
        self.correlation_threshold = correlation_threshold
        self.coherence_threshold = coherence_threshold

    def compare_signals(
        self,
        python_signal: np.ndarray,
        matlab_signal: np.ndarray,
        label: str = "signal"
    ) -> ValidationResult:
        """
        Compare Python and MATLAB signals with comprehensive metrics.

        Args:
            python_signal: Signal from Python generator
            matlab_signal: Signal from MATLAB generator
            label: Descriptive label for logging

        Returns:
            ValidationResult with detailed comparison

        Example:
            >>> result = validator.compare_signals(py_sig, mat_sig, "desalignement")
            >>> print(f"Mean error: {result.metrics['mean_relative_error']:.4%}")
        """
        logger.info(f"Validating {label}...")

        errors = []
        warnings = []
        metrics = {}

        # Check shape compatibility
        if python_signal.shape != matlab_signal.shape:
            errors.append(
                f"Shape mismatch: Python {python_signal.shape} vs MATLAB {matlab_signal.shape}"
            )
            return ValidationResult(False, self.tolerance, metrics, errors, warnings)

        # 1. Statistical metrics comparison
        stat_metrics = self._compare_statistics(python_signal, matlab_signal)
        metrics.update(stat_metrics)

        # Check statistical errors
        for key, value in stat_metrics.items():
            if 'error' in key and value > self.tolerance:
                errors.append(f"{key} = {value:.4%} exceeds tolerance {self.tolerance:.2%}")

        # 2. Point-wise error analysis
        pointwise_metrics = self._compute_pointwise_error(python_signal, matlab_signal)
        metrics.update(pointwise_metrics)

        if pointwise_metrics['max_relative_error'] > self.tolerance:
            warnings.append(
                f"Max point-wise error {pointwise_metrics['max_relative_error']:.4%} "
                f"exceeds tolerance (may be acceptable if localized)"
            )

        # 3. Time-domain correlation
        corr_metrics = self._compute_correlation(python_signal, matlab_signal)
        metrics.update(corr_metrics)

        if corr_metrics['correlation'] < self.correlation_threshold:
            errors.append(
                f"Correlation {corr_metrics['correlation']:.4f} below threshold "
                f"{self.correlation_threshold:.4f}"
            )

        # 4. Frequency-domain coherence (if signals long enough)
        if len(python_signal) >= 256:
            coh_metrics = self._compute_coherence(python_signal, matlab_signal)
            metrics.update(coh_metrics)

            if coh_metrics['mean_coherence'] < self.coherence_threshold:
                warnings.append(
                    f"Mean coherence {coh_metrics['mean_coherence']:.4f} below threshold "
                    f"{self.coherence_threshold:.4f}"
                )

        # Overall pass/fail
        passed = len(errors) == 0

        # Log results
        if passed:
            logger.info(f"✅ {label} validation PASSED")
            logger.debug(f"  Mean error: {metrics['mean_relative_error']:.4%}")
            logger.debug(f"  RMS error: {metrics['rms_relative_error']:.4%}")
            logger.debug(f"  Correlation: {metrics['correlation']:.6f}")
        else:
            logger.error(f"❌ {label} validation FAILED")
            for error in errors:
                logger.error(f"  - {error}")

        return ValidationResult(passed, self.tolerance, metrics, errors, warnings)

    def _compare_statistics(
        self,
        python_signal: np.ndarray,
        matlab_signal: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare statistical properties of signals.

        Returns:
            Dictionary with statistical comparison metrics
        """
        py = python_signal
        mat = matlab_signal

        metrics = {}

        # Mean
        py_mean = np.mean(py)
        mat_mean = np.mean(mat)
        metrics['python_mean'] = py_mean
        metrics['matlab_mean'] = mat_mean
        metrics['mean_absolute_error'] = abs(py_mean - mat_mean)
        metrics['mean_relative_error'] = abs(py_mean - mat_mean) / (abs(mat_mean) + 1e-10)

        # Standard deviation
        py_std = np.std(py)
        mat_std = np.std(mat)
        metrics['python_std'] = py_std
        metrics['matlab_std'] = mat_std
        metrics['std_absolute_error'] = abs(py_std - mat_std)
        metrics['std_relative_error'] = abs(py_std - mat_std) / (mat_std + 1e-10)

        # RMS
        py_rms = np.sqrt(np.mean(py**2))
        mat_rms = np.sqrt(np.mean(mat**2))
        metrics['python_rms'] = py_rms
        metrics['matlab_rms'] = mat_rms
        metrics['rms_absolute_error'] = abs(py_rms - mat_rms)
        metrics['rms_relative_error'] = abs(py_rms - mat_rms) / (mat_rms + 1e-10)

        # Min/Max
        metrics['python_min'] = np.min(py)
        metrics['matlab_min'] = np.min(mat)
        metrics['python_max'] = np.max(py)
        metrics['matlab_max'] = np.max(mat)

        # Percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        for p in percentiles:
            py_p = np.percentile(py, p)
            mat_p = np.percentile(mat, p)
            metrics[f'python_p{p}'] = py_p
            metrics[f'matlab_p{p}'] = mat_p
            rel_err = abs(py_p - mat_p) / (abs(mat_p) + 1e-10)
            metrics[f'p{p}_relative_error'] = rel_err

        return metrics

    def _compute_pointwise_error(
        self,
        python_signal: np.ndarray,
        matlab_signal: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute point-wise error between signals.
        """
        abs_error = np.abs(python_signal - matlab_signal)
        rel_error = abs_error / (np.abs(matlab_signal) + 1e-10)

        metrics = {
            'mean_absolute_error': float(np.mean(abs_error)),
            'max_absolute_error': float(np.max(abs_error)),
            'std_absolute_error': float(np.std(abs_error)),
            'mean_relative_error': float(np.mean(rel_error)),
            'max_relative_error': float(np.max(rel_error)),
            'std_relative_error': float(np.std(rel_error)),
            'rmse': float(np.sqrt(np.mean(abs_error**2))),
            'nrmse': float(np.sqrt(np.mean(abs_error**2)) / (np.std(matlab_signal) + 1e-10))
        }

        return metrics

    def _compute_correlation(
        self,
        python_signal: np.ndarray,
        matlab_signal: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute time-domain correlation metrics.
        """
        # Pearson correlation
        correlation = np.corrcoef(python_signal, matlab_signal)[0, 1]

        # Cross-correlation (normalized)
        xcorr = correlate(python_signal, matlab_signal, mode='full')
        xcorr_normalized = xcorr / (len(python_signal) * np.std(python_signal) * np.std(matlab_signal))
        max_xcorr = np.max(np.abs(xcorr_normalized))

        # Lag at maximum correlation
        max_lag = np.argmax(np.abs(xcorr_normalized)) - len(python_signal) + 1

        metrics = {
            'correlation': float(correlation),
            'max_cross_correlation': float(max_xcorr),
            'lag_at_max_correlation': int(max_lag)
        }

        return metrics

    def _compute_coherence(
        self,
        python_signal: np.ndarray,
        matlab_signal: np.ndarray,
        fs: float = 20480.0
    ) -> Dict[str, float]:
        """
        Compute frequency-domain coherence.

        Coherence measures linear relationship in frequency domain.
        Values close to 1 indicate high similarity.
        """
        # Compute coherence
        f, Cxy = coherence(python_signal, matlab_signal, fs=fs, nperseg=256)

        # Focus on meaningful frequency range (avoid DC and Nyquist extremes)
        valid_idx = (f > 1.0) & (f < fs/2 - 1.0)
        f_valid = f[valid_idx]
        Cxy_valid = Cxy[valid_idx]

        metrics = {
            'mean_coherence': float(np.mean(Cxy_valid)),
            'min_coherence': float(np.min(Cxy_valid)),
            'median_coherence': float(np.median(Cxy_valid)),
            'coherence_below_threshold': float(np.mean(Cxy_valid < self.coherence_threshold))
        }

        return metrics

    def compare_batch(
        self,
        python_signals: List[np.ndarray],
        matlab_signals: List[MatlabSignalData],
        labels: Optional[List[str]] = None
    ) -> Dict[str, ValidationResult]:
        """
        Compare batch of Python signals against MATLAB references.

        Args:
            python_signals: List of Python-generated signals
            matlab_signals: List of MATLAB reference signals
            labels: Optional labels for each signal

        Returns:
            Dictionary mapping label to ValidationResult

        Example:
            >>> results = validator.compare_batch(py_signals, mat_signals, labels)
            >>> passed = sum(1 for r in results.values() if r.passed)
            >>> print(f"{passed}/{len(results)} signals passed validation")
        """
        if len(python_signals) != len(matlab_signals):
            raise ValueError(
                f"Signal count mismatch: {len(python_signals)} Python vs "
                f"{len(matlab_signals)} MATLAB"
            )

        if labels is None:
            labels = [f"signal_{i:03d}" for i in range(len(python_signals))]

        results = {}
        for py_sig, mat_data, label in zip(python_signals, matlab_signals, labels):
            result = self.compare_signals(py_sig, mat_data.signal, label)
            results[label] = result

        # Summary statistics
        passed_count = sum(1 for r in results.values() if r.passed)
        total_count = len(results)

        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH VALIDATION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total signals: {total_count}")
        logger.info(f"Passed: {passed_count} ({passed_count/total_count:.1%})")
        logger.info(f"Failed: {total_count - passed_count}")

        if passed_count == total_count:
            logger.info(f"✅ All signals passed validation!")
        else:
            logger.warning(f"⚠️  {total_count - passed_count} signals failed validation")

        return results

    def generate_report(
        self,
        results: Dict[str, ValidationResult],
        output_path: Optional[Path] = None
    ) -> str:
        """
        Generate human-readable validation report.

        Args:
            results: Dictionary of ValidationResults
            output_path: Optional path to save report

        Returns:
            Report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("PYTHON vs MATLAB SIGNAL VALIDATION REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {np.datetime64('now')}")
        lines.append(f"Tolerance: {self.tolerance:.2%}")
        lines.append(f"Total signals: {len(results)}")
        lines.append("")

        # Summary
        passed = sum(1 for r in results.values() if r.passed)
        failed = len(results) - passed
        lines.append(f"SUMMARY:")
        lines.append(f"  ✅ Passed: {passed} ({passed/len(results):.1%})")
        lines.append(f"  ❌ Failed: {failed} ({failed/len(results):.1%})")
        lines.append("")

        # Per-signal results
        lines.append("DETAILED RESULTS:")
        lines.append("-" * 80)

        for label, result in results.items():
            status = "✅ PASS" if result.passed else "❌ FAIL"
            lines.append(f"{label}: {status}")

            if result.metrics:
                lines.append(f"  Mean error: {result.metrics.get('mean_relative_error', 0):.4%}")
                lines.append(f"  RMS error: {result.metrics.get('rms_relative_error', 0):.4%}")
                lines.append(f"  Correlation: {result.metrics.get('correlation', 0):.6f}")

            if result.errors:
                lines.append(f"  Errors:")
                for error in result.errors:
                    lines.append(f"    - {error}")

            if result.warnings:
                lines.append(f"  Warnings:")
                for warning in result.warnings:
                    lines.append(f"    - {warning}")

            lines.append("")

        report = "\n".join(lines)

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")

        return report


def validate_against_matlab(
    python_generator_output: Dict[str, Any],
    matlab_mat_dir: Path,
    tolerance: float = 0.01,
    report_path: Optional[Path] = None
) -> bool:
    """
    Convenience function to validate Python generator against MATLAB .mat files.

    Args:
        python_generator_output: Output from SignalGenerator.generate_dataset()
        matlab_mat_dir: Directory containing MATLAB .mat reference files
        tolerance: Maximum allowed error (default: 1%)
        report_path: Optional path to save validation report

    Returns:
        True if all signals pass validation

    Example:
        >>> from data.signal_generator import SignalGenerator
        >>> from config.data_config import DataConfig
        >>>
        >>> # Generate Python signals
        >>> config = DataConfig.from_yaml('config/data_config.yaml')
        >>> generator = SignalGenerator(config)
        >>> python_output = generator.generate_dataset()
        >>>
        >>> # Validate against MATLAB
        >>> passed = validate_against_matlab(
        ...     python_output,
        ...     Path('./matlab_reference_signals'),
        ...     report_path=Path('./validation_report.txt')
        ... )
    """
    logger.info("Starting Python vs MATLAB validation...")

    # Load MATLAB reference signals
    importer = MatlabImporter()
    matlab_signals = importer.load_batch(matlab_mat_dir)

    logger.info(f"Loaded {len(matlab_signals)} MATLAB reference signals")

    # Extract Python signals
    python_signals = python_generator_output['signals']
    python_labels = python_generator_output['labels']

    # Match signals by label
    validator = SignalValidator(tolerance=tolerance)

    # Create label-indexed dictionaries
    matlab_by_label = {sig.label: sig for sig in matlab_signals}
    python_by_label = {}
    for sig, label in zip(python_signals, python_labels):
        if label not in python_by_label:
            python_by_label[label] = []
        python_by_label[label].append(sig)

    # Compare matching labels
    results = {}
    for label in matlab_by_label.keys():
        if label in python_by_label:
            matlab_sig = matlab_by_label[label]
            # Use first Python signal with matching label
            python_sig = python_by_label[label][0]

            result = validator.compare_signals(python_sig, matlab_sig.signal, label)
            results[label] = result
        else:
            logger.warning(f"No Python signal found for MATLAB label: {label}")

    # Generate report
    if report_path or results:
        report = validator.generate_report(results, report_path)
        print("\n" + report)

    # Overall pass/fail
    all_passed = all(r.passed for r in results.values())
    return all_passed
