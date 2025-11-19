"""
MATLAB .mat file importer for validation against Python generator.

Purpose:
    Load MATLAB-generated signals and metadata for comparing against
    Python implementation. Ensures numerical equivalence within 1% tolerance.

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import scipy.io as sio
from datetime import datetime

from utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MatlabSignalData:
    """
    Container for MATLAB signal data.

    Attributes:
        signal: Time-domain signal array (N,)
        metadata: Dictionary of signal properties
        label: Fault class label
        severity: Severity level if available
        config: Original MATLAB configuration if available
    """
    signal: np.ndarray
    metadata: Dict[str, Any]
    label: str
    severity: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class MatlabImporter:
    """
    Import and parse MATLAB .mat files from generator.m output.

    Handles different .mat file formats:
    - Single signal files
    - Batch files with multiple signals
    - Metadata extraction
    - Type conversion from MATLAB to Python

    Example:
        >>> importer = MatlabImporter()
        >>> data = importer.load_mat_file('signal_desalignement_001.mat')
        >>> print(f"Signal shape: {data.signal.shape}")
        >>> print(f"Fault: {data.label}, Severity: {data.severity}")
    """

    def __init__(self, squeeze_me: bool = True, struct_as_record: bool = False):
        """
        Initialize MATLAB importer.

        Args:
            squeeze_me: Remove singleton dimensions
            struct_as_record: Load MATLAB structs as numpy records
        """
        self.squeeze_me = squeeze_me
        self.struct_as_record = struct_as_record

    def load_mat_file(self, path: Path) -> MatlabSignalData:
        """
        Load a single .mat file.

        Args:
            path: Path to .mat file

        Returns:
            MatlabSignalData object with signal and metadata

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If .mat file has unexpected structure

        Example:
            >>> data = importer.load_mat_file('signal_sain_001.mat')
            >>> assert data.signal.shape == (102400,)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"MATLAB file not found: {path}")

        logger.info(f"Loading MATLAB file: {path.name}")

        # Load .mat file
        mat_data = sio.loadmat(
            path,
            squeeze_me=self.squeeze_me,
            struct_as_record=self.struct_as_record
        )

        # Parse structure based on expected format
        signal_data = self._parse_mat_structure(mat_data, path.name)

        logger.debug(f"Loaded {signal_data.label} signal: shape={signal_data.signal.shape}")
        return signal_data

    def _parse_mat_structure(self, mat_data: Dict, filename: str) -> MatlabSignalData:
        """
        Parse MATLAB structure into MatlabSignalData.

        Expected MATLAB structure:
            signal: [N x 1] or [1 x N] array
            metadata: struct with fields (fault_type, severity, fs, etc.)
            label: string or char array

        Args:
            mat_data: Dictionary from scipy.io.loadmat
            filename: Original filename for error messages

        Returns:
            Parsed MatlabSignalData
        """
        # Remove MATLAB metadata fields
        mat_data = {k: v for k, v in mat_data.items()
                   if not k.startswith('__')}

        # Try to find signal array (largest numeric array)
        signal = self._extract_signal(mat_data)

        # Try to find metadata struct
        metadata = self._extract_metadata(mat_data)

        # Extract label (from metadata or filename)
        label = self._extract_label(mat_data, metadata, filename)

        # Extract severity if available
        severity = self._extract_severity(mat_data, metadata)

        # Extract config if available
        config = self._extract_config(mat_data)

        return MatlabSignalData(
            signal=signal,
            metadata=metadata,
            label=label,
            severity=severity,
            config=config
        )

    def _extract_signal(self, mat_data: Dict) -> np.ndarray:
        """
        Extract signal array from MATLAB data.

        Looks for common field names: 'signal', 'x', 'data', or largest array.
        """
        # Try common field names
        for field_name in ['signal', 'x', 'data', 'y']:
            if field_name in mat_data:
                signal = np.asarray(mat_data[field_name]).flatten()
                return signal

        # Find largest numeric array
        candidates = []
        for key, value in mat_data.items():
            if isinstance(value, np.ndarray) and value.size > 100:
                candidates.append((key, value))

        if not candidates:
            raise ValueError("No signal array found in .mat file")

        # Take largest array
        largest_key, largest_array = max(candidates, key=lambda x: x[1].size)
        logger.warning(f"Using field '{largest_key}' as signal (no standard name found)")
        return largest_array.flatten()

    def _extract_metadata(self, mat_data: Dict) -> Dict[str, Any]:
        """
        Extract metadata struct from MATLAB data.

        Returns:
            Dictionary of metadata fields
        """
        metadata = {}

        # Try common metadata field names
        for field_name in ['metadata', 'info', 'params']:
            if field_name in mat_data:
                meta_struct = mat_data[field_name]
                if isinstance(meta_struct, dict):
                    metadata = meta_struct
                elif hasattr(meta_struct, '_fieldnames'):
                    # MATLAB struct as numpy record
                    metadata = {f: getattr(meta_struct, f) for f in meta_struct._fieldnames}
                break

        # Also collect scalar fields directly in mat_data
        for key, value in mat_data.items():
            if isinstance(value, (int, float, str, np.number)):
                metadata[key] = value
            elif isinstance(value, np.ndarray) and value.size == 1:
                metadata[key] = value.item()

        return metadata

    def _extract_label(self, mat_data: Dict, metadata: Dict, filename: str) -> str:
        """
        Extract fault label from metadata or filename.
        """
        # Try metadata fields
        for field_name in ['label', 'fault_type', 'class', 'fault']:
            if field_name in metadata:
                label = metadata[field_name]
                if isinstance(label, np.ndarray):
                    label = str(label.item())
                return str(label)
            if field_name in mat_data:
                label = mat_data[field_name]
                if isinstance(label, np.ndarray):
                    label = str(label.item())
                return str(label)

        # Parse from filename: "signal_<fault>_<num>.mat"
        parts = Path(filename).stem.split('_')
        if len(parts) >= 2:
            # Assume format: signal_faultname_001
            fault_part = '_'.join(parts[1:-1]) if len(parts) > 2 else parts[1]
            logger.info(f"Extracted label '{fault_part}' from filename")
            return fault_part

        logger.warning(f"Could not extract label from {filename}, using 'unknown'")
        return 'unknown'

    def _extract_severity(self, mat_data: Dict, metadata: Dict) -> Optional[str]:
        """
        Extract severity level if available.
        """
        for field_name in ['severity', 'severity_level', 'sev']:
            if field_name in metadata:
                sev = metadata[field_name]
                if isinstance(sev, np.ndarray):
                    sev = sev.item()
                return str(sev)
            if field_name in mat_data:
                sev = mat_data[field_name]
                if isinstance(sev, np.ndarray):
                    sev = sev.item()
                return str(sev)
        return None

    def _extract_config(self, mat_data: Dict) -> Optional[Dict[str, Any]]:
        """
        Extract configuration struct if available.
        """
        for field_name in ['config', 'CONFIG', 'params']:
            if field_name in mat_data:
                config = mat_data[field_name]
                if isinstance(config, dict):
                    return config
                elif hasattr(config, '_fieldnames'):
                    return {f: getattr(config, f) for f in config._fieldnames}
        return None

    def load_batch(self, mat_dir: Path, pattern: str = '*.mat') -> List[MatlabSignalData]:
        """
        Load multiple .mat files from directory.

        Args:
            mat_dir: Directory containing .mat files
            pattern: Glob pattern for file matching

        Returns:
            List of MatlabSignalData objects

        Example:
            >>> batch = importer.load_batch(Path('./matlab_signals'))
            >>> print(f"Loaded {len(batch)} signals")
        """
        mat_dir = Path(mat_dir)
        if not mat_dir.exists():
            raise FileNotFoundError(f"Directory not found: {mat_dir}")

        mat_files = sorted(mat_dir.glob(pattern))
        if not mat_files:
            raise ValueError(f"No .mat files found in {mat_dir} with pattern {pattern}")

        logger.info(f"Loading {len(mat_files)} .mat files from {mat_dir}")

        batch_data = []
        for mat_file in mat_files:
            try:
                data = self.load_mat_file(mat_file)
                batch_data.append(data)
            except Exception as e:
                logger.error(f"Failed to load {mat_file.name}: {e}")
                continue

        logger.info(f"Successfully loaded {len(batch_data)}/{len(mat_files)} files")
        return batch_data

    def extract_signals_and_labels(
        self,
        batch: List[MatlabSignalData]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract signals and labels as arrays for validation.

        Args:
            batch: List of MatlabSignalData objects

        Returns:
            (signals, labels) tuple
            - signals: (num_signals, signal_length) array
            - labels: List of fault labels

        Example:
            >>> batch = importer.load_batch(mat_dir)
            >>> signals, labels = importer.extract_signals_and_labels(batch)
            >>> print(f"Signals: {signals.shape}, Labels: {len(labels)}")
        """
        signals = np.array([data.signal for data in batch])
        labels = [data.label for data in batch]
        return signals, labels

    def get_statistics(self, batch: List[MatlabSignalData]) -> Dict[str, Any]:
        """
        Compute statistics on batch of MATLAB signals.

        Args:
            batch: List of MatlabSignalData objects

        Returns:
            Dictionary with statistics (mean, std, min, max, rms)
        """
        signals, labels = self.extract_signals_and_labels(batch)

        stats = {
            'num_signals': len(signals),
            'signal_length': signals.shape[1],
            'global_mean': float(np.mean(signals)),
            'global_std': float(np.std(signals)),
            'global_min': float(np.min(signals)),
            'global_max': float(np.max(signals)),
            'global_rms': float(np.sqrt(np.mean(signals**2))),
            'per_signal_rms': np.sqrt(np.mean(signals**2, axis=1)),
            'labels': np.unique(labels).tolist(),
            'label_counts': {label: labels.count(label) for label in np.unique(labels)}
        }

        return stats


def load_matlab_reference(
    mat_path: Path,
    verbose: bool = True
) -> MatlabSignalData:
    """
    Convenience function to load a single MATLAB reference signal.

    Args:
        mat_path: Path to .mat file
        verbose: Enable logging

    Returns:
        MatlabSignalData object

    Example:
        >>> ref_signal = load_matlab_reference('matlab_outputs/signal_sain_001.mat')
        >>> print(ref_signal.signal.shape)
    """
    importer = MatlabImporter()
    data = importer.load_mat_file(mat_path)

    if verbose:
        logger.info(f"Loaded reference: {data.label} (shape={data.signal.shape})")
        logger.info(f"  RMS: {np.sqrt(np.mean(data.signal**2)):.6f}")
        logger.info(f"  Range: [{np.min(data.signal):.6f}, {np.max(data.signal):.6f}]")

    return data
