"""
Efficient dataset caching using HDF5.

Purpose:
    Cache generated signals to disk for faster repeated experiments:
    - HDF5 storage for efficient random access
    - Compression support
    - Metadata storage
    - Cache invalidation

Author: Syed Abbas Ahmad
Date: 2025-11-19
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import hashlib
from datetime import datetime

from utils.logging import get_logger
from utils.file_io import ensure_dir

logger = get_logger(__name__)


class CacheManager:
    """
    Manage dataset caching with HDF5.

    Example:
        >>> cache = CacheManager(cache_dir='./cache')
        >>> cache.cache_dataset(signals, labels, metadata, cache_name='train')
        >>> signals, labels, metadata = cache.load_cached_dataset('train')
    """

    def __init__(self, cache_dir: Path = Path('./cache')):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        ensure_dir(self.cache_dir)

    def cache_dataset(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        cache_name: str = 'dataset',
        compression: Optional[str] = 'gzip',
        compression_opts: Optional[int] = 4
    ) -> Path:
        """
        Cache dataset to HDF5 file.

        Args:
            signals: Signal array (num_signals, signal_length)
            labels: Label array (num_signals,)
            metadata: Optional list of metadata dicts
            cache_name: Name for cache file
            compression: HDF5 compression ('gzip', 'lzf', None)
            compression_opts: Compression level (1-9 for gzip)

        Returns:
            Path to cache file

        Example:
            >>> cache_path = cache.cache_dataset(
            ...     signals, labels, metadata, cache_name='train_set'
            ... )
        """
        cache_path = self.cache_dir / f'{cache_name}.h5'

        logger.info(f"Caching dataset to {cache_path}")

        with h5py.File(cache_path, 'w') as f:
            # Store signals
            f.create_dataset(
                'signals',
                data=signals,
                compression=compression,
                compression_opts=compression_opts
            )

            # Store labels
            f.create_dataset('labels', data=labels)

            # Store metadata as JSON strings
            if metadata is not None:
                metadata_json = [json.dumps(m) for m in metadata]
                dt = h5py.string_dtype(encoding='utf-8')
                f.create_dataset('metadata', data=metadata_json, dtype=dt)

            # Store cache info
            f.attrs['cache_name'] = cache_name
            f.attrs['num_signals'] = len(signals)
            f.attrs['signal_length'] = signals.shape[1]
            f.attrs['cached_at'] = datetime.now().isoformat()
            f.attrs['compression'] = compression if compression else 'none'

        logger.info(f"Cached {len(signals)} signals ({self._format_size(cache_path)})")

        return cache_path

    def load_cached_dataset(
        self,
        cache_name: str
    ) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict]]]:
        """
        Load dataset from cache.

        Args:
            cache_name: Name of cached dataset

        Returns:
            (signals, labels, metadata) tuple

        Example:
            >>> signals, labels, metadata = cache.load_cached_dataset('train_set')
        """
        cache_path = self.cache_dir / f'{cache_name}.h5'

        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found: {cache_path}")

        logger.info(f"Loading dataset from cache: {cache_path}")

        with h5py.File(cache_path, 'r') as f:
            signals = f['signals'][:]
            labels = f['labels'][:]

            # Load metadata if exists
            if 'metadata' in f:
                metadata_json = f['metadata'][:]
                metadata = [json.loads(m) for m in metadata_json]
            else:
                metadata = None

            # Log cache info
            logger.info(
                f"Loaded {f.attrs['num_signals']} signals "
                f"(cached at {f.attrs.get('cached_at', 'unknown')})"
            )

        return signals, labels, metadata

    def cache_exists(self, cache_name: str) -> bool:
        """
        Check if cache exists.

        Args:
            cache_name: Name of cache to check

        Returns:
            True if cache exists

        Example:
            >>> if not cache.cache_exists('train'):
            ...     generate_and_cache_data()
        """
        cache_path = self.cache_dir / f'{cache_name}.h5'
        return cache_path.exists()

    def invalidate_cache(self, cache_name: str) -> None:
        """
        Delete cache file.

        Args:
            cache_name: Name of cache to delete

        Example:
            >>> cache.invalidate_cache('old_dataset')
        """
        cache_path = self.cache_dir / f'{cache_name}.h5'

        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Invalidated cache: {cache_name}")
        else:
            logger.warning(f"Cache not found: {cache_name}")

    def list_caches(self) -> List[Dict[str, Any]]:
        """
        List all available caches.

        Returns:
            List of cache info dictionaries

        Example:
            >>> for cache_info in cache.list_caches():
            ...     print(f"{cache_info['name']}: {cache_info['num_signals']} signals")
        """
        cache_files = list(self.cache_dir.glob('*.h5'))
        cache_list = []

        for cache_file in cache_files:
            try:
                with h5py.File(cache_file, 'r') as f:
                    info = {
                        'name': cache_file.stem,
                        'path': str(cache_file),
                        'num_signals': f.attrs.get('num_signals', 0),
                        'signal_length': f.attrs.get('signal_length', 0),
                        'cached_at': f.attrs.get('cached_at', 'unknown'),
                        'size_mb': cache_file.stat().st_size / (1024**2)
                    }
                    cache_list.append(info)
            except Exception as e:
                logger.warning(f"Could not read cache info from {cache_file}: {e}")

        return sorted(cache_list, key=lambda x: x['name'])

    def get_cache_info(self, cache_name: str) -> Dict[str, Any]:
        """
        Get detailed cache information.

        Args:
            cache_name: Name of cache

        Returns:
            Dictionary with cache details

        Example:
            >>> info = cache.get_cache_info('train_set')
            >>> print(f"Size: {info['size_mb']:.2f} MB")
        """
        cache_path = self.cache_dir / f'{cache_name}.h5'

        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found: {cache_path}")

        with h5py.File(cache_path, 'r') as f:
            info = {
                'name': cache_name,
                'path': str(cache_path),
                'num_signals': f.attrs.get('num_signals', 0),
                'signal_length': f.attrs.get('signal_length', 0),
                'cached_at': f.attrs.get('cached_at', 'unknown'),
                'compression': f.attrs.get('compression', 'none'),
                'size_mb': cache_path.stat().st_size / (1024**2),
                'datasets': list(f.keys())
            }

        return info

    def compute_cache_key(self, config: Dict[str, Any]) -> str:
        """
        Compute hash key from configuration.

        Useful for automatic cache invalidation when config changes.

        Args:
            config: Configuration dictionary

        Returns:
            Hash string

        Example:
            >>> key = cache.compute_cache_key(data_config.to_dict())
            >>> cache_name = f'dataset_{key[:8]}'
        """
        # Convert config to canonical JSON string
        config_str = json.dumps(config, sort_keys=True)

        # Compute hash
        hash_obj = hashlib.sha256(config_str.encode())
        return hash_obj.hexdigest()

    def cache_dataset_with_splits(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None,
        cache_name: str = 'dataset',
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        stratify: bool = True,
        random_seed: int = 42
    ) -> Path:
        """
        Cache dataset with train/val/test splits.

        Creates HDF5 file with structure compatible with signal_generator and dash_app:
            - f['train']['signals'], f['train']['labels']
            - f['val']['signals'], f['val']['labels']
            - f['test']['signals'], f['test']['labels']

        Args:
            signals: Signal array (num_signals, signal_length)
            labels: Label array (num_signals,) - integer labels
            metadata: Optional metadata dictionaries
            cache_name: Name for cache file (without extension)
            split_ratios: (train, val, test) ratios (default: 0.7, 0.15, 0.15)
            stratify: Whether to stratify splits by label (default: True)
            random_seed: Random seed for reproducibility (default: 42)

        Returns:
            Path to created HDF5 cache file

        Example:
            >>> cache = CacheManager(cache_dir='data/processed')
            >>> cache_path = cache.cache_dataset_with_splits(
            ...     signals=signal_array,
            ...     labels=label_array,
            ...     cache_name='bearing_dataset',
            ...     split_ratios=(0.7, 0.15, 0.15)
            ... )
            >>> print(f"Dataset cached at: {cache_path}")
        """
        from sklearn.model_selection import train_test_split

        cache_path = self.cache_dir / f'{cache_name}.h5'

        logger.info(f"Caching dataset with splits to {cache_path}")

        # Create stratified splits
        train_ratio, val_ratio, test_ratio = split_ratios

        # First split: separate test set
        stratify_arg = labels if stratify else None
        X_temp, X_test, y_temp, y_test = train_test_split(
            signals, labels,
            test_size=test_ratio,
            stratify=stratify_arg,
            random_state=random_seed
        )

        # Second split: separate train and val from temp
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        stratify_arg = y_temp if stratify else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=stratify_arg,
            random_state=random_seed
        )

        # Create HDF5 file
        with h5py.File(cache_path, 'w') as f:
            # Store global attributes
            f.attrs['cache_name'] = cache_name
            f.attrs['num_signals'] = len(signals)
            f.attrs['signal_length'] = signals.shape[1]
            f.attrs['cached_at'] = datetime.now().isoformat()
            f.attrs['split_ratios'] = split_ratios
            f.attrs['stratified'] = stratify
            f.attrs['random_seed'] = random_seed

            # Create train group
            train_grp = f.create_group('train')
            train_grp.create_dataset(
                'signals',
                data=X_train,
                compression='gzip',
                compression_opts=4
            )
            train_grp.create_dataset('labels', data=y_train)
            train_grp.attrs['num_samples'] = len(X_train)

            # Create val group
            val_grp = f.create_group('val')
            val_grp.create_dataset(
                'signals',
                data=X_val,
                compression='gzip',
                compression_opts=4
            )
            val_grp.create_dataset('labels', data=y_val)
            val_grp.attrs['num_samples'] = len(X_val)

            # Create test group
            test_grp = f.create_group('test')
            test_grp.create_dataset(
                'signals',
                data=X_test,
                compression='gzip',
                compression_opts=4
            )
            test_grp.create_dataset('labels', data=y_test)
            test_grp.attrs['num_samples'] = len(X_test)

            # Store metadata if provided
            if metadata is not None:
                metadata_json = [json.dumps(m) for m in metadata]
                dt = h5py.string_dtype(encoding='utf-8')
                f.create_dataset('metadata', data=metadata_json, dtype=dt)

        logger.info(
            f"Cached with splits - Train: {len(X_train)}, "
            f"Val: {len(X_val)}, Test: {len(X_test)} "
            f"({self._format_size(cache_path)})"
        )

        return cache_path

    def _format_size(self, path: Path) -> str:
        """Format file size for display."""
        size_bytes = path.stat().st_size
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes / (1024**2):.2f} MB"
        else:
            return f"{size_bytes / (1024**3):.2f} GB"


def cache_dataset_simple(
    signals: np.ndarray,
    labels: np.ndarray,
    cache_path: Path,
    metadata: Optional[List[Dict]] = None
) -> None:
    """
    Simple function to cache dataset to HDF5.

    Args:
        signals: Signal array
        labels: Label array
        cache_path: Output cache file path
        metadata: Optional metadata

    Example:
        >>> cache_dataset_simple(signals, labels, Path('cache/train.h5'))
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(cache_path, 'w') as f:
        f.create_dataset('signals', data=signals, compression='gzip')
        f.create_dataset('labels', data=labels)

        if metadata is not None:
            metadata_json = [json.dumps(m) for m in metadata]
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('metadata', data=metadata_json, dtype=dt)

    logger.info(f"Cached {len(signals)} signals to {cache_path}")


def load_cached_dataset_simple(
    cache_path: Path
) -> Tuple[np.ndarray, np.ndarray, Optional[List[Dict]]]:
    """
    Simple function to load cached dataset.

    Args:
        cache_path: Cache file path

    Returns:
        (signals, labels, metadata) tuple

    Example:
        >>> signals, labels, meta = load_cached_dataset_simple(Path('cache/train.h5'))
    """
    cache_path = Path(cache_path)

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")

    with h5py.File(cache_path, 'r') as f:
        signals = f['signals'][:]
        labels = f['labels'][:]

        if 'metadata' in f:
            metadata_json = f['metadata'][:]
            metadata = [json.loads(m) for m in metadata_json]
        else:
            metadata = None

    return signals, labels, metadata
