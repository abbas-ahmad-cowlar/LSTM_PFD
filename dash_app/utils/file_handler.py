"""
File handler utilities for MAT file uploads.
Handles storage, validation, and cleanup of uploaded files.
"""
import base64
import os
from pathlib import Path
from typing import List, Dict, Tuple
import tempfile
import shutil

from utils.logger import setup_logger

logger = setup_logger(__name__)


def save_uploaded_mat_files(uploaded_files: List[Dict], output_dir: str = None) -> Tuple[str, List[str]]:
    """
    Save base64-encoded uploaded files to disk.

    Args:
        uploaded_files: List of dicts with {filename, content, size}
                       content is base64-encoded string from dcc.Upload
        output_dir: Directory to save files (creates temp dir if None)

    Returns:
        Tuple of (directory_path, list_of_file_paths)

    Example:
        >>> files = [{'filename': 'signal_001.mat', 'content': 'data:...;base64,ABC123', 'size': 1024}]
        >>> temp_dir, file_paths = save_uploaded_mat_files(files)
        >>> print(f"Saved {len(file_paths)} files to {temp_dir}")
    """
    if output_dir is None:
        # Create temporary directory
        output_dir = tempfile.mkdtemp(prefix='mat_upload_')
        logger.info(f"Created temporary directory: {output_dir}")
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = str(output_dir)

    saved_files = []

    for file_info in uploaded_files:
        filename = file_info['filename']
        content = file_info['content']

        # Parse base64 content from dcc.Upload format
        # Format: 'data:application/octet-stream;base64,<base64_data>'
        if ',' in content:
            content_type, content_string = content.split(',', 1)
        else:
            content_string = content

        # Decode base64
        try:
            decoded = base64.b64decode(content_string)
        except Exception as e:
            logger.error(f"Failed to decode file {filename}: {e}")
            continue

        # Save to disk
        file_path = os.path.join(output_dir, filename)
        try:
            with open(file_path, 'wb') as f:
                f.write(decoded)
            saved_files.append(file_path)
            logger.info(f"Saved uploaded file: {filename} ({len(decoded)} bytes)")
        except Exception as e:
            logger.error(f"Failed to save file {filename}: {e}")
            continue

    logger.info(f"Saved {len(saved_files)}/{len(uploaded_files)} files to {output_dir}")
    return output_dir, saved_files


def validate_mat_file(file_path: str) -> Dict:
    """
    Validate a MAT file before processing.

    Args:
        file_path: Path to MAT file

    Returns:
        Dictionary with validation results:
            - valid: bool
            - errors: list of error messages
            - warnings: list of warning messages
            - file_size: int (bytes)
    """
    result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'file_size': 0
    }

    file_path = Path(file_path)

    # Check file exists
    if not file_path.exists():
        result['valid'] = False
        result['errors'].append(f"File not found: {file_path}")
        return result

    # Check file size
    file_size = file_path.stat().st_size
    result['file_size'] = file_size

    if file_size == 0:
        result['valid'] = False
        result['errors'].append("File is empty")
        return result

    if file_size > 100 * 1024 * 1024:  # 100 MB
        result['warnings'].append(f"Large file: {file_size / 1024 / 1024:.1f} MB")

    # Check file extension
    if file_path.suffix.lower() != '.mat':
        result['warnings'].append(f"Unexpected extension: {file_path.suffix}")

    # Try to load with scipy
    try:
        import scipy.io as sio
        mat_data = sio.loadmat(str(file_path), squeeze_me=True, struct_as_record=False)

        # Check if file has expected fields
        data_fields = [k for k in mat_data.keys() if not k.startswith('__')]
        if not data_fields:
            result['valid'] = False
            result['errors'].append("No data fields found in MAT file")
        else:
            result['warnings'].append(f"Found fields: {', '.join(data_fields[:5])}")

    except Exception as e:
        result['valid'] = False
        result['errors'].append(f"Failed to load MAT file: {str(e)}")

    return result


def cleanup_temp_files(temp_dir: str):
    """
    Remove temporary uploaded files after processing.

    Args:
        temp_dir: Directory to clean up
    """
    if not temp_dir:
        return

    temp_path = Path(temp_dir)

    if not temp_path.exists():
        logger.warning(f"Temp directory doesn't exist: {temp_dir}")
        return

    try:
        shutil.rmtree(temp_dir)
        logger.info(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        logger.error(f"Failed to cleanup {temp_dir}: {e}")


def get_file_summary(file_paths: List[str]) -> Dict:
    """
    Get summary statistics for a list of files.

    Args:
        file_paths: List of file paths

    Returns:
        Dictionary with:
            - total_files: int
            - total_size_bytes: int
            - total_size_mb: float
            - files: list of {name, size_bytes, size_mb}
    """
    summary = {
        'total_files': len(file_paths),
        'total_size_bytes': 0,
        'total_size_mb': 0.0,
        'files': []
    }

    for file_path in file_paths:
        path = Path(file_path)
        if path.exists():
            size_bytes = path.stat().st_size
            summary['total_size_bytes'] += size_bytes
            summary['files'].append({
                'name': path.name,
                'size_bytes': size_bytes,
                'size_mb': round(size_bytes / 1024 / 1024, 2)
            })

    summary['total_size_mb'] = round(summary['total_size_bytes'] / 1024 / 1024, 2)

    return summary


def parse_upload_contents(contents: List[str], filenames: List[str]) -> List[Dict]:
    """
    Parse dcc.Upload contents into structured format.

    Args:
        contents: List of base64-encoded file contents from dcc.Upload
        filenames: List of original filenames

    Returns:
        List of dicts with {filename, content, size}
    """
    if not contents or not filenames:
        return []

    # Ensure lists
    if not isinstance(contents, list):
        contents = [contents]
    if not isinstance(filenames, list):
        filenames = [filenames]

    uploaded_files = []

    for content, filename in zip(contents, filenames):
        # Estimate size from base64
        if ',' in content:
            _, content_string = content.split(',', 1)
        else:
            content_string = content

        # Base64 is ~33% larger than original, so decoded size ≈ len * 0.75
        estimated_size = int(len(content_string) * 0.75)

        uploaded_files.append({
            'filename': filename,
            'content': content,
            'size': estimated_size
        })

    return uploaded_files
