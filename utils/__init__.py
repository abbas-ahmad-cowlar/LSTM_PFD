"""Utility functions and helpers for LSTM_PFD pipeline."""

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
