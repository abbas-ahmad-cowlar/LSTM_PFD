"""
File I/O utilities for serialization and data persistence.

Purpose:
    Common file operations:
    - Pickle serialization
    - JSON save/load
    - YAML operations
    - Directory management

Author: LSTM_PFD Team
Date: 2025-11-19
"""

import pickle
import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import shutil

from utils.logging import get_logger

logger = get_logger(__name__)


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    """
    Save object to pickle file.

    Args:
        obj: Object to serialize
        path: Output file path

    Example:
        >>> data = {'key': 'value'}
        >>> save_pickle(data, 'data.pkl')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.debug(f"Saved pickle to {path}")


def load_pickle(path: Union[str, Path]) -> Any:
    """
    Load object from pickle file.

    Args:
        path: Pickle file path

    Returns:
        Deserialized object

    Example:
        >>> data = load_pickle('data.pkl')
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle file not found: {path}")

    with open(path, 'rb') as f:
        obj = pickle.load(f)

    logger.debug(f"Loaded pickle from {path}")
    return obj


def save_json(
    data: Union[Dict, List],
    path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False
) -> None:
    """
    Save data to JSON file.

    Args:
        data: Dictionary or list to save
        path: Output file path
        indent: JSON indentation (None for compact)
        ensure_ascii: Escape non-ASCII characters

    Example:
        >>> config = {'lr': 0.001, 'batch_size': 32}
        >>> save_json(config, 'config.json')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

    logger.debug(f"Saved JSON to {path}")


def load_json(path: Union[str, Path]) -> Union[Dict, List]:
    """
    Load data from JSON file.

    Args:
        path: JSON file path

    Returns:
        Loaded dictionary or list

    Example:
        >>> config = load_json('config.json')
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, 'r') as f:
        data = json.load(f)

    logger.debug(f"Loaded JSON from {path}")
    return data


def save_yaml(
    data: Dict,
    path: Union[str, Path],
    default_flow_style: bool = False
) -> None:
    """
    Save data to YAML file.

    Args:
        data: Dictionary to save
        path: Output file path
        default_flow_style: Use flow style (compact)

    Example:
        >>> config = {'model': 'cnn1d', 'lr': 0.001}
        >>> save_yaml(config, 'config.yaml')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=default_flow_style, sort_keys=False)

    logger.debug(f"Saved YAML to {path}")


def load_yaml(path: Union[str, Path]) -> Dict:
    """
    Load data from YAML file.

    Args:
        path: YAML file path

    Returns:
        Loaded dictionary

    Example:
        >>> config = load_yaml('config.yaml')
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    logger.debug(f"Loaded YAML from {path}")
    return data


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Path object

    Example:
        >>> output_dir = ensure_dir('outputs/experiment_1')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(
    directory: Union[str, Path],
    pattern: str = '*',
    recursive: bool = False
) -> List[Path]:
    """
    List files in directory matching pattern.

    Args:
        directory: Directory to search
        pattern: Glob pattern (e.g., '*.pkl', '**/*.txt')
        recursive: Search recursively

    Returns:
        List of file paths

    Example:
        >>> pkl_files = list_files('data/', pattern='*.pkl')
        >>> all_txt = list_files('logs/', pattern='**/*.txt', recursive=True)
    """
    directory = Path(directory)
    if not directory.exists():
        logger.warning(f"Directory not found: {directory}")
        return []

    if recursive and not pattern.startswith('**'):
        pattern = f'**/{pattern}'

    if recursive:
        files = list(directory.glob(pattern))
    else:
        files = list(directory.glob(pattern))

    # Filter only files (not directories)
    files = [f for f in files if f.is_file()]

    return sorted(files)


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    Copy file from source to destination.

    Args:
        src: Source file path
        dst: Destination file path

    Example:
        >>> copy_file('model.pt', 'backup/model.pt')
    """
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    logger.debug(f"Copied {src} -> {dst}")


def move_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """
    Move file from source to destination.

    Args:
        src: Source file path
        dst: Destination file path

    Example:
        >>> move_file('temp.pkl', 'archive/temp.pkl')
    """
    src = Path(src)
    dst = Path(dst)

    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))

    logger.debug(f"Moved {src} -> {dst}")


def delete_file(path: Union[str, Path]) -> None:
    """
    Delete file if it exists.

    Args:
        path: File path to delete

    Example:
        >>> delete_file('temp.pkl')
    """
    path = Path(path)
    if path.exists():
        path.unlink()
        logger.debug(f"Deleted {path}")


def delete_directory(path: Union[str, Path], confirm: bool = True) -> None:
    """
    Delete directory and all contents.

    Args:
        path: Directory path to delete
        confirm: Require confirmation (safety check)

    Example:
        >>> delete_directory('temp_data/', confirm=False)
    """
    path = Path(path)

    if not path.exists():
        logger.warning(f"Directory not found: {path}")
        return

    if confirm:
        logger.warning(f"Deleting directory: {path}")

    shutil.rmtree(path)
    logger.info(f"Deleted directory: {path}")


def get_file_size(path: Union[str, Path]) -> int:
    """
    Get file size in bytes.

    Args:
        path: File path

    Returns:
        File size in bytes

    Example:
        >>> size = get_file_size('model.pt')
        >>> print(f"Size: {size / (1024**2):.2f} MB")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return path.stat().st_size


def get_directory_size(path: Union[str, Path]) -> int:
    """
    Get total size of directory in bytes.

    Args:
        path: Directory path

    Returns:
        Total size in bytes

    Example:
        >>> size = get_directory_size('checkpoints/')
        >>> print(f"Total: {size / (1024**3):.2f} GB")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {path}")

    total_size = 0
    for item in path.rglob('*'):
        if item.is_file():
            total_size += item.stat().st_size

    return total_size


def save_text(text: str, path: Union[str, Path]) -> None:
    """
    Save text to file.

    Args:
        text: Text content
        path: Output file path

    Example:
        >>> save_text("Hello, world!", "output.txt")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        f.write(text)

    logger.debug(f"Saved text to {path}")


def load_text(path: Union[str, Path]) -> str:
    """
    Load text from file.

    Args:
        path: File path

    Returns:
        File contents as string

    Example:
        >>> text = load_text("input.txt")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, 'r') as f:
        text = f.read()

    return text


def append_to_file(text: str, path: Union[str, Path]) -> None:
    """
    Append text to file.

    Args:
        text: Text to append
        path: File path

    Example:
        >>> append_to_file("New line\n", "log.txt")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'a') as f:
        f.write(text)


def safe_save(
    obj: Any,
    path: Union[str, Path],
    format: str = 'pickle',
    backup: bool = True
) -> None:
    """
    Safely save object with atomic write and optional backup.

    Writes to temporary file first, then renames to avoid corruption.

    Args:
        obj: Object to save
        path: Output file path
        format: File format ('pickle', 'json', 'yaml')
        backup: Create backup of existing file

    Example:
        >>> safe_save(model.state_dict(), 'model.pt', backup=True)
    """
    path = Path(path)
    temp_path = path.with_suffix(path.suffix + '.tmp')

    # Create backup if file exists
    if backup and path.exists():
        backup_path = path.with_suffix(path.suffix + '.bak')
        shutil.copy2(path, backup_path)
        logger.debug(f"Created backup: {backup_path}")

    # Save to temporary file
    try:
        if format == 'pickle':
            save_pickle(obj, temp_path)
        elif format == 'json':
            save_json(obj, temp_path)
        elif format == 'yaml':
            save_yaml(obj, temp_path)
        else:
            raise ValueError(f"Unknown format: {format}")

        # Atomic rename
        temp_path.replace(path)
        logger.debug(f"Safely saved to {path}")

    except Exception as e:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise e


def safe_load(
    path: Union[str, Path],
    format: str = 'pickle',
    fallback_to_backup: bool = True
) -> Any:
    """
    Safely load object with fallback to backup.

    Args:
        path: File path
        format: File format ('pickle', 'json', 'yaml')
        fallback_to_backup: Try backup file if main file fails

    Returns:
        Loaded object

    Example:
        >>> state_dict = safe_load('model.pt', fallback_to_backup=True)
    """
    path = Path(path)

    try:
        if format == 'pickle':
            return load_pickle(path)
        elif format == 'json':
            return load_json(path)
        elif format == 'yaml':
            return load_yaml(path)
        else:
            raise ValueError(f"Unknown format: {format}")

    except Exception as e:
        if fallback_to_backup:
            backup_path = path.with_suffix(path.suffix + '.bak')
            if backup_path.exists():
                logger.warning(f"Main file failed, trying backup: {backup_path}")
                if format == 'pickle':
                    return load_pickle(backup_path)
                elif format == 'json':
                    return load_json(backup_path)
                elif format == 'yaml':
                    return load_yaml(backup_path)
        raise e
