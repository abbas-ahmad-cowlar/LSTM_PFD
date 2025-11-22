"""Utility functions and helpers for LSTM_PFD pipeline."""

from .constants import (
    # Signal parameters
    SIGNAL_LENGTH,
    SAMPLING_RATE,
    SIGNAL_DURATION,
    # Classification
    NUM_CLASSES,
    FAULT_TYPES,
    FAULT_TYPE_TO_ID,
    get_fault_id,
    get_fault_name,
    get_fault_display_name,
    # Defaults
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_DROPOUT,
    DEFAULT_RANDOM_SEED,
)
from .reproducibility import set_seed, make_deterministic, get_random_state, restore_random_state
from .logging import get_logger, setup_logging, log_system_info
from .device_manager import (
    get_device,
    get_available_gpus,
    get_gpu_info,
    move_to_device,
    get_gpu_memory_usage,
    clear_gpu_memory,
    log_device_info,
    DeviceManager
)
from .file_io import (
    save_pickle,
    load_pickle,
    save_json,
    load_json,
    save_yaml,
    load_yaml,
    ensure_dir,
    list_files,
    safe_save,
    safe_load
)
from .timer import (
    Timer,
    TimingStats,
    Profiler,
    time_function,
    benchmark,
    format_time,
    get_global_profiler
)
from .visualization_utils import (
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE
    set_plot_style,
    save_figure,
    create_figure,
    get_color_palette,
    plot_time_series,
    plot_spectrum,
    plot_confusion_matrix,
    plot_training_history
)

__all__ = [
    # Constants
    'SIGNAL_LENGTH',
    'SAMPLING_RATE',
    'SIGNAL_DURATION',
    'NUM_CLASSES',
    'FAULT_TYPES',
    'FAULT_TYPE_TO_ID',
    'get_fault_id',
    'get_fault_name',
    'get_fault_display_name',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_LEARNING_RATE',
    'DEFAULT_DROPOUT',
    'DEFAULT_RANDOM_SEED',
    # Reproducibility
    'set_seed',
    'make_deterministic',
    'get_random_state',
    'restore_random_state',
    # Logging
    'get_logger',
    'setup_logging',
    'log_system_info',
    # Device management
    'get_device',
    'get_available_gpus',
    'get_gpu_info',
    'move_to_device',
    'get_gpu_memory_usage',
    'clear_gpu_memory',
    'log_device_info',
    'DeviceManager',
    # File I/O
    'save_pickle',
    'load_pickle',
    'save_json',
    'load_json',
    'save_yaml',
    'load_yaml',
    'ensure_dir',
    'list_files',
    'safe_save',
    'safe_load',
    # Timing
    'Timer',
    'TimingStats',
    'Profiler',
    'time_function',
    'benchmark',
    'format_time',
    'get_global_profiler',
    # Visualization
    'set_plot_style',
    'save_figure',
    'create_figure',
    'get_color_palette',
    'plot_time_series',
    'plot_spectrum',
    'plot_confusion_matrix',
    'plot_training_history'
]
