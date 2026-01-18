"""
Dataset Management Service.
Business logic for dataset CRUD operations.
"""
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from database.connection import get_db_session
from models.dataset import Dataset
from utils.logger import setup_logger
import h5py
import numpy as np
import os
from utils.constants import (
    MAX_DATASET_LIST_LIMIT,
    BYTES_PER_MB,
    PERCENT_MULTIPLIER,
)

logger = setup_logger(__name__)


class DatasetService:
    """Service for dataset management operations."""

    @staticmethod
    def list_datasets(limit: int = MAX_DATASET_LIST_LIMIT, offset: int = 0, search_query: str = None) -> List[Dict]:
        """
        Get list of all datasets with pagination.

        Args:
            limit: Maximum number of datasets to return
            offset: Number of datasets to skip
            search_query: Optional search string

        Returns:
            List of dataset dictionaries
        """
        try:
            with get_db_session() as session:
                query = session.query(Dataset)

                # Apply search filter
                if search_query:
                    query = query.filter(Dataset.name.ilike(f'%{search_query}%'))

                # Apply pagination
                datasets = query.order_by(Dataset.created_at.desc()).limit(limit).offset(offset).all()

                result = []
                for ds in datasets:
                    # Calculate file size
                    file_size_mb = 0
                    if ds.file_path and Path(ds.file_path).exists():
                        file_size_mb = Path(ds.file_path).stat().st_size / BYTES_PER_MB

                    result.append({
                        'id': ds.id,
                        'name': ds.name,
                        'description': ds.description or '',
                        'num_signals': ds.num_signals,
                        'fault_types': ds.fault_types or [],
                        'severity_levels': ds.severity_levels or [],
                        'file_path': ds.file_path,
                        'file_size_mb': file_size_mb,
                        'created_at': ds.created_at,
                        'metadata': ds.meta_data or {}
                    })

                return result

        except Exception as e:
            logger.error(f"Failed to list datasets: {e}", exc_info=True)
            return []

    @staticmethod
    def get_dataset_details(dataset_id: int) -> Optional[Dict]:
        """
        Get detailed dataset information.

        Args:
            dataset_id: Dataset ID

        Returns:
            Dataset details including statistics
        """
        try:
            with get_db_session() as session:
                dataset = session.query(Dataset).filter_by(id=dataset_id).first()

                if not dataset:
                    logger.error(f"Dataset {dataset_id} not found")
                    return None

                # Basic info
                details = {
                    'id': dataset.id,
                    'name': dataset.name,
                    'description': dataset.description or '',
                    'num_signals': dataset.num_signals,
                    'fault_types': dataset.fault_types or [],
                    'severity_levels': dataset.severity_levels or [],
                    'file_path': dataset.file_path,
                    'created_at': dataset.created_at,
                    'metadata': dataset.meta_data or {}
                }

                # Load statistics from HDF5 file
                if dataset.file_path and Path(dataset.file_path).exists():
                    file_stats = DatasetService._load_file_statistics(dataset.file_path)
                    details.update(file_stats)

                return details

        except Exception as e:
            logger.error(f"Failed to get dataset details: {e}", exc_info=True)
            return None

    @staticmethod
    def _resolve_file_path(file_path: str) -> Path:
        """
        Resolve file path, trying multiple base directories.
        
        Handles paths that may be:
        1. Absolute paths
        2. Relative to project root (LSTM_PFD/)
        3. Relative to dashboard dir (packages/dashboard/)
        """
        path = Path(file_path)
        
        # Already absolute and exists
        if path.is_absolute() and path.exists():
            return path
        
        # Try relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        project_relative = project_root / file_path
        if project_relative.exists():
            return project_relative
        
        # Try relative to dashboard dir
        dashboard_dir = Path(__file__).resolve().parent.parent
        dashboard_relative = dashboard_dir / file_path
        if dashboard_relative.exists():
            return dashboard_relative
        
        # Return original path (will fail with clear error)
        return path

    @staticmethod
    def _load_file_statistics(file_path: str) -> Dict:
        """
        Load statistics from HDF5 file.
        
        Handles two HDF5 structures:
        1. Flat: signals, labels at root level
        2. Grouped: train/val/test groups with signals, labels inside each

        Args:
            file_path: Path to HDF5 file

        Returns:
            Dictionary with file statistics
        """
        try:
            # Resolve path (handles relative paths)
            resolved_path = DatasetService._resolve_file_path(file_path)
            
            stats = {
                'file_size_mb': 0,
                'signal_length': 0,
                'sampling_rate': 0,
                'class_distribution': {},
                'signal_statistics': {}
            }

            # File size (always works)
            if resolved_path.exists():
                stats['file_size_mb'] = resolved_path.stat().st_size / BYTES_PER_MB

            with h5py.File(str(resolved_path), 'r') as f:
                signals = None
                labels = None
                
                # Try flat structure first (signals, labels at root)
                if 'signals' in f and 'labels' in f:
                    signals = f['signals'][:]
                    labels = f['labels'][:]
                
                # Try grouped structure (train/val/test groups)
                elif 'train' in f or 'val' in f or 'test' in f:
                    all_signals = []
                    all_labels = []
                    
                    for split in ['train', 'val', 'test']:
                        if split in f:
                            grp = f[split]
                            if 'signals' in grp:
                                all_signals.append(grp['signals'][:])
                            if 'labels' in grp:
                                all_labels.append(grp['labels'][:])
                    
                    if all_signals:
                        signals = np.concatenate(all_signals, axis=0)
                    if all_labels:
                        labels = np.concatenate(all_labels, axis=0)
                
                # Extract statistics from loaded data
                if signals is not None:
                    stats['signal_length'] = signals.shape[1] if len(signals.shape) > 1 else len(signals)
                
                # Sampling rate from attributes
                if 'sampling_rate' in f.attrs:
                    stats['sampling_rate'] = f.attrs['sampling_rate']
                elif 'metadata' in f and 'sampling_rate' in f['metadata'].attrs:
                    stats['sampling_rate'] = f['metadata'].attrs['sampling_rate']
                
                # Get class names from metadata if available
                fault_class_names = {}
                if 'metadata' in f and 'fault_classes' in f['metadata']:
                    fault_classes = f['metadata']['fault_classes'][:]
                    # Build name mapping from unique labels
                    if labels is not None:
                        unique_labels = np.unique(labels)
                        for i, label_idx in enumerate(unique_labels):
                            # Try to find a matching fault class name
                            if label_idx < len(fault_classes):
                                fault_class_names[int(label_idx)] = fault_classes[label_idx]
                
                # Class distribution
                if labels is not None:
                    unique, counts = np.unique(labels, return_counts=True)
                    for k, v in zip(unique, counts):
                        label_key = int(k)
                        # Use fault class name if available, otherwise use index
                        if label_key in fault_class_names:
                            display_name = fault_class_names[label_key]
                            if isinstance(display_name, bytes):
                                display_name = display_name.decode('utf-8')
                            stats['class_distribution'][display_name] = int(v)
                        else:
                            stats['class_distribution'][label_key] = int(v)
                
                # Signal statistics (mean, std per class)
                if signals is not None and labels is not None:
                    for fault_class in np.unique(labels):
                        class_signals = signals[labels == fault_class]
                        class_key = int(fault_class)
                        if class_key in fault_class_names:
                            display_name = fault_class_names[class_key]
                            if isinstance(display_name, bytes):
                                display_name = display_name.decode('utf-8')
                            class_key = display_name
                        
                        stats['signal_statistics'][class_key] = {
                            'mean': float(np.mean(class_signals)),
                            'std': float(np.std(class_signals)),
                            'min': float(np.min(class_signals)),
                            'max': float(np.max(class_signals)),
                            'count': len(class_signals)
                        }

            return stats

        except Exception as e:
            logger.error(f"Failed to load file statistics: {e}", exc_info=True)
            return {'file_size_mb': Path(file_path).stat().st_size / BYTES_PER_MB if Path(file_path).exists() else 0}

    @staticmethod
    def get_dataset_preview(dataset_id: int, num_samples: int = 3) -> Dict:
        """
        Get preview of dataset signals.

        Args:
            dataset_id: Dataset ID
            num_samples: Number of samples per fault type

        Returns:
            Signal preview data
        """
        try:
            with get_db_session() as session:
                dataset = session.query(Dataset).filter_by(id=dataset_id).first()

                if not dataset or not Path(dataset.file_path).exists():
                    return {}

                preview = {}

                with h5py.File(dataset.file_path, 'r') as f:
                    if 'signals' not in f or 'labels' not in f:
                        return {}

                    signals = f['signals'][:]
                    labels = f['labels'][:]

                    # Get samples for each fault type
                    for fault_class in np.unique(labels):
                        class_indices = np.where(labels == fault_class)[0]
                        sample_indices = class_indices[:num_samples]

                        preview[int(fault_class)] = [
                            signals[idx].tolist() for idx in sample_indices
                        ]

                return preview

        except Exception as e:
            logger.error(f"Failed to get dataset preview: {e}", exc_info=True)
            return {}

    @staticmethod
    def delete_dataset(dataset_id: int, delete_file: bool = False) -> bool:
        """
        Delete dataset from database (and optionally file).

        Args:
            dataset_id: Dataset ID
            delete_file: If True, also delete HDF5 file

        Returns:
            True if successful
        """
        try:
            with get_db_session() as session:
                dataset = session.query(Dataset).filter_by(id=dataset_id).first()

                if not dataset:
                    logger.error(f"Dataset {dataset_id} not found")
                    return False

                file_path = dataset.file_path

                # Delete from database
                session.delete(dataset)
                session.commit()

                # Delete file if requested
                if delete_file and file_path and Path(file_path).exists():
                    try:
                        os.remove(file_path)
                        logger.info(f"Deleted file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete file {file_path}: {e}")

                logger.info(f"Deleted dataset {dataset_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete dataset: {e}", exc_info=True)
            return False

    @staticmethod
    def archive_dataset(dataset_id: int) -> bool:
        """
        Archive dataset (mark as archived in metadata).

        Args:
            dataset_id: Dataset ID

        Returns:
            True if successful
        """
        try:
            with get_db_session() as session:
                dataset = session.query(Dataset).filter_by(id=dataset_id).first()

                if not dataset:
                    logger.error(f"Dataset {dataset_id} not found")
                    return False

                # Update metadata
                metadata = dataset.meta_data or {}
                metadata['archived'] = True
                metadata['archived_at'] = datetime.utcnow().isoformat()
                dataset.meta_data = metadata

                session.commit()

                logger.info(f"Archived dataset {dataset_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to archive dataset: {e}", exc_info=True)
            return False

    @staticmethod
    def export_dataset(dataset_id: int, format: str = 'hdf5', output_dir: str = 'exports') -> Optional[str]:
        """
        Export dataset to different format.

        Args:
            dataset_id: Dataset ID
            format: 'hdf5', 'mat', 'csv'
            output_dir: Output directory

        Returns:
            Path to exported file or None
        """
        try:
            with get_db_session() as session:
                dataset = session.query(Dataset).filter_by(id=dataset_id).first()

                if not dataset or not Path(dataset.file_path).exists():
                    logger.error(f"Dataset {dataset_id} not found or file missing")
                    return None

                # Create output directory
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)

                # Load data
                with h5py.File(dataset.file_path, 'r') as f:
                    signals = f['signals'][:]
                    labels = f['labels'][:]

                # Export based on format
                if format == 'hdf5':
                    export_file = output_path / f"{dataset.name}.h5"
                    with h5py.File(export_file, 'w') as f:
                        f.create_dataset('signals', data=signals)
                        f.create_dataset('labels', data=labels)
                        f.attrs['name'] = dataset.name
                        f.attrs['num_signals'] = dataset.num_signals

                elif format == 'mat':
                    import scipy.io
                    export_file = output_path / f"{dataset.name}.mat"
                    scipy.io.savemat(export_file, {
                        'signals': signals,
                        'labels': labels,
                        'name': dataset.name
                    })

                elif format == 'csv':
                    import pandas as pd
                    export_file = output_path / f"{dataset.name}.csv"

                    # Flatten signals and create DataFrame
                    data = {
                        'label': labels,
                    }

                    # Add signal columns
                    for i in range(signals.shape[1]):
                        data[f'signal_{i}'] = signals[:, i]

                    df = pd.DataFrame(data)
                    df.to_csv(export_file, index=False)

                else:
                    logger.error(f"Unsupported format: {format}")
                    return None

                logger.info(f"Exported dataset {dataset_id} to {export_file}")
                return str(export_file)

        except Exception as e:
            logger.error(f"Failed to export dataset: {e}", exc_info=True)
            return None

    @staticmethod
    def get_dataset_statistics(dataset_id: int) -> Dict:
        """
        Compute dataset statistics.

        Args:
            dataset_id: Dataset ID

        Returns:
            Statistics dictionary
        """
        try:
            details = DatasetService.get_dataset_details(dataset_id)

            if not details:
                return {}

            stats = {
                'total_signals': details.get('num_signals', 0),
                'fault_types': details.get('fault_types', []),
                'file_size_mb': details.get('file_size_mb', 0),
                'signal_length': details.get('signal_length', 0),
                'sampling_rate': details.get('sampling_rate', 0),
                'class_distribution': details.get('class_distribution', {}),
                'signal_statistics': details.get('signal_statistics', {})
            }

            # Calculate percentages
            total = stats['total_signals']
            if total > 0:
                stats['class_distribution_pct'] = {
                    k: (v / total * PERCENT_MULTIPLIER) for k, v in stats['class_distribution'].items()
                }

            return stats

        except Exception as e:
            logger.error(f"Failed to get dataset statistics: {e}", exc_info=True)
            return {}
