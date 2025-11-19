"""
Pipeline validation utilities.

Purpose:
    Validate pipeline correctness and compare with baselines.

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import numpy as np
from typing import Dict, Optional


class PipelineValidator:
    """
    Validate pipeline outputs and performance.

    Example:
        >>> validator = PipelineValidator()
        >>> validator.validate_pipeline_output(results, expected_accuracy=0.92)
    """

    @staticmethod
    def validate_pipeline_output(results: Dict,
                                expected_accuracy: float = 0.92,
                                tolerance: float = 0.05) -> bool:
        """
        Validate pipeline output against expected performance.

        Args:
            results: Pipeline results dictionary
            expected_accuracy: Expected test accuracy
            tolerance: Acceptable deviation

        Returns:
            True if validation passes
        """
        test_accuracy = results.get('test_accuracy', 0.0)

        print("\n=== Pipeline Validation ===")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Expected: {expected_accuracy:.4f} +/- {tolerance:.4f}")

        if test_accuracy >= expected_accuracy - tolerance:
            print("✓ Validation PASSED")
            return True
        else:
            print("✗ Validation FAILED")
            return False

    @staticmethod
    def compare_with_baseline(current_results: Dict,
                            baseline_results: Dict) -> Dict:
        """
        Compare current results with baseline.

        Args:
            current_results: Current pipeline results
            baseline_results: Baseline results

        Returns:
            Comparison dictionary
        """
        current_acc = current_results['test_accuracy']
        baseline_acc = baseline_results.get('test_accuracy', 0.0)

        diff = current_acc - baseline_acc
        rel_diff = (diff / baseline_acc * 100) if baseline_acc > 0 else 0

        comparison = {
            'current_accuracy': current_acc,
            'baseline_accuracy': baseline_acc,
            'absolute_difference': diff,
            'relative_difference_pct': rel_diff,
            'improved': diff > 0
        }

        print("\n=== Baseline Comparison ===")
        print(f"Current: {current_acc:.4f}")
        print(f"Baseline: {baseline_acc:.4f}")
        print(f"Difference: {diff:+.4f} ({rel_diff:+.2f}%)")

        return comparison

    @staticmethod
    def check_class_balance(labels: np.ndarray,
                          min_samples_per_class: int = 10) -> bool:
        """
        Check if dataset has sufficient samples per class.

        Args:
            labels: Label array
            min_samples_per_class: Minimum required samples

        Returns:
            True if balanced
        """
        unique, counts = np.unique(labels, return_counts=True)

        print("\n=== Class Balance Check ===")
        for label, count in zip(unique, counts):
            status = "✓" if count >= min_samples_per_class else "✗"
            print(f"  {status} Class {label}: {count} samples")

        min_count = np.min(counts)
        if min_count < min_samples_per_class:
            print(f"✗ Insufficient samples (min: {min_count})")
            return False

        print("✓ Class balance OK")
        return True
