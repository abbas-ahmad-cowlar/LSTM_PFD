"""
MATLAB compatibility layer for importing/exporting data.

Purpose:
    Interface with existing MATLAB code and data.

Author: Syed Abbas Ahmad
Date: 2025-11-19
"""

import numpy as np
from pathlib import Path
from typing import Dict
from scipy.io import loadmat, savemat
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


class MatlabCompatibility:
    """
    MATLAB compatibility utilities.

    Example:
        >>> compat = MatlabCompatibility()
        >>> features = compat.load_matlab_features('features.mat')
        >>> compat.export_to_matlab(python_features, 'python_features.mat')
    """

    @staticmethod
    def load_matlab_features(mat_file: Path) -> Dict:
        """
        Load features from MATLAB .mat file.

        Args:
            mat_file: Path to .mat file

        Returns:
            Dictionary with features and labels
        """
        data = loadmat(mat_file)

        # MATLAB typically saves as 'features' and 'labels'
        features = data.get('features', data.get('X'))
        labels = data.get('labels', data.get('y'))

        return {
            'features': features,
            'labels': labels.flatten() if labels is not None else None
        }

    @staticmethod
    def export_to_matlab(features: np.ndarray, labels: np.ndarray,
                        save_path: Path):
        """
        Export Python features to MATLAB format.

        Args:
            features: Feature matrix
            labels: Label array
            save_path: Path to save .mat file
        """
        savemat(save_path, {
            'features': features,
            'labels': labels,
            'feature_names': [f'feature_{i}' for i in range(features.shape[1])]
        })

        print(f"Exported to MATLAB format: {save_path}")

    @staticmethod
    def compare_with_matlab(python_features: np.ndarray,
                          matlab_features: np.ndarray,
                          rtol: float = 0.01) -> bool:
        """
        Compare Python and MATLAB features for parity.

        Args:
            python_features: Features from Python
            matlab_features: Features from MATLAB
            rtol: Relative tolerance (default 1%)

        Returns:
            True if features match within tolerance
        """
        if python_features.shape != matlab_features.shape:
            print(f"Shape mismatch: Python {python_features.shape} vs MATLAB {matlab_features.shape}")
            return False

        # Check element-wise closeness
        close = np.allclose(python_features, matlab_features, rtol=rtol, atol=0.01)

        if close:
            print(f"✓ Features match within {rtol*100}% tolerance")
        else:
            max_diff = np.max(np.abs(python_features - matlab_features))
            print(f"✗ Features differ (max diff: {max_diff:.6f})")

        return close
