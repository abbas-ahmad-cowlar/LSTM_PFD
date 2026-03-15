"""
Post-generation signal validation utilities.

Purpose:
    Validate signals produced by SignalGenerator or loaded from HDF5 for
    numerical integrity (NaN, Inf), statistical sanity (std range, flat
    signal detection), and structural correctness (expected length).

    This is distinct from ``data_validator.py``, which compares Python
    signals against MATLAB reference signals.

Usage:
    >>> from data.signal_validation import validate_signal, validate_batch
    >>> validate_signal(signal, expected_length=102400)  # raises on failure
    >>> report = validate_batch(signals, labels)
    >>> print(report)

Author: Syed Abbas Ahmad
Date: 2026-03-15
"""

import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from utils.logging import get_logger

logger = get_logger(__name__)


class SignalValidationError(Exception):
    """Raised when a signal fails validation checks."""
    pass


@dataclass
class ValidationReport:
    """Batch-level validation report."""

    total_signals: int = 0
    passed: int = 0
    failed: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    statistics: Dict[str, float] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        if self.total_signals == 0:
            return 0.0
        return self.passed / self.total_signals

    def __repr__(self) -> str:
        status = "✅ ALL PASSED" if self.failed == 0 else f"❌ {self.failed} FAILED"
        return (
            f"ValidationReport({status}, "
            f"{self.passed}/{self.total_signals} passed, "
            f"{len(self.warnings)} warnings)"
        )


def validate_signal(
    signal: np.ndarray,
    expected_length: Optional[int] = None,
    std_min: float = 1e-6,
    std_max: float = 100.0,
    peak_max: float = 1000.0,
    label: str = "signal",
    raise_on_error: bool = True,
) -> List[str]:
    """
    Validate a single generated signal for numerical integrity.

    Checks:
        1. NaN values
        2. Inf values
        3. Expected length (if provided)
        4. Standard deviation within [std_min, std_max]
        5. Peak amplitude within [-peak_max, peak_max]
        6. Flat/dead signal detection (std < std_min)

    Args:
        signal: 1-D numpy array of signal samples.
        expected_length: Expected number of samples (None = skip check).
        std_min: Minimum acceptable standard deviation.
        std_max: Maximum acceptable standard deviation.
        peak_max: Maximum acceptable absolute amplitude.
        label: Descriptive label for log messages.
        raise_on_error: If True, raises ``SignalValidationError`` on first
            critical failure. If False, returns list of error strings.

    Returns:
        List of error/warning strings (empty if all checks pass).

    Raises:
        SignalValidationError: If ``raise_on_error`` is True and a check fails.
    """
    errors: List[str] = []

    # 1. NaN check
    nan_count = int(np.sum(np.isnan(signal)))
    if nan_count > 0:
        msg = f"[{label}] Contains {nan_count} NaN values"
        errors.append(msg)

    # 2. Inf check
    inf_count = int(np.sum(np.isinf(signal)))
    if inf_count > 0:
        msg = f"[{label}] Contains {inf_count} Inf values"
        errors.append(msg)

    # 3. Length check
    if expected_length is not None and len(signal) != expected_length:
        msg = f"[{label}] Length {len(signal)} != expected {expected_length}"
        errors.append(msg)

    # Skip remaining stats if NaN/Inf present (would produce meaningless results)
    if nan_count > 0 or inf_count > 0:
        if raise_on_error and errors:
            raise SignalValidationError("; ".join(errors))
        return errors

    # 4. Std range check
    signal_std = float(np.std(signal))
    if signal_std < std_min:
        msg = (
            f"[{label}] Flat/dead signal detected — "
            f"std={signal_std:.2e} < {std_min:.2e}"
        )
        errors.append(msg)
    elif signal_std > std_max:
        msg = (
            f"[{label}] Abnormally high variance — "
            f"std={signal_std:.2e} > {std_max:.2e}"
        )
        errors.append(msg)

    # 5. Peak amplitude check
    signal_peak = float(np.max(np.abs(signal)))
    if signal_peak > peak_max:
        msg = (
            f"[{label}] Peak amplitude {signal_peak:.2f} "
            f"exceeds maximum {peak_max:.2f}"
        )
        errors.append(msg)

    if raise_on_error and errors:
        raise SignalValidationError("; ".join(errors))

    return errors


def validate_batch(
    signals: np.ndarray,
    labels: Optional[np.ndarray] = None,
    expected_length: Optional[int] = None,
    std_min: float = 1e-6,
    std_max: float = 100.0,
    peak_max: float = 1000.0,
) -> ValidationReport:
    """
    Validate a batch of signals and produce a summary report.

    Does NOT raise exceptions — all issues are collected in the returned
    ``ValidationReport``.

    Args:
        signals: 2-D array (num_signals, signal_length).
        labels: Optional 1-D array of integer labels.
        expected_length: Expected signal length (None = infer from first signal).
        std_min: Minimum acceptable standard deviation.
        std_max: Maximum acceptable standard deviation.
        peak_max: Maximum acceptable absolute amplitude.

    Returns:
        ValidationReport with aggregate statistics and per-signal errors.
    """
    report = ValidationReport(total_signals=len(signals))

    if len(signals) == 0:
        report.warnings.append("Empty signal batch")
        return report

    # Infer expected length from first signal if not provided
    if expected_length is None:
        expected_length = signals.shape[1] if signals.ndim == 2 else len(signals[0])

    # Per-signal validation
    for i, sig in enumerate(signals):
        errs = validate_signal(
            sig,
            expected_length=expected_length,
            std_min=std_min,
            std_max=std_max,
            peak_max=peak_max,
            label=f"signal_{i:04d}",
            raise_on_error=False,
        )
        if errs:
            report.failed += 1
            report.errors.extend(errs)
        else:
            report.passed += 1

    # Aggregate statistics (skip if all signals have NaN/Inf)
    try:
        stds = np.std(signals, axis=1)
        peaks = np.max(np.abs(signals), axis=1)
        means = np.mean(signals, axis=1)
        report.statistics = {
            "mean_of_means": float(np.mean(means)),
            "std_of_means": float(np.std(means)),
            "mean_std": float(np.mean(stds)),
            "min_std": float(np.min(stds)),
            "max_std": float(np.max(stds)),
            "mean_peak": float(np.mean(peaks)),
            "max_peak": float(np.max(peaks)),
        }
    except Exception:
        report.warnings.append("Could not compute aggregate statistics (NaN/Inf present)")

    # Class balance check
    if labels is not None and len(labels) > 0:
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) > 1:
            imbalance_ratio = float(max(counts) / (min(counts) + 1e-10))
            if imbalance_ratio > 3.0:
                report.warnings.append(
                    f"Class imbalance detected: ratio {imbalance_ratio:.1f}x "
                    f"(max_class={max(counts)}, min_class={min(counts)})"
                )
            report.statistics["num_classes"] = len(unique)
            report.statistics["class_imbalance_ratio"] = imbalance_ratio

    # Log summary
    if report.failed > 0:
        logger.warning(f"Signal validation: {report}")
    else:
        logger.info(f"Signal validation: {report}")

    return report
