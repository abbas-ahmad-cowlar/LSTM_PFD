"""
Timing utilities for profiling and performance monitoring.

Purpose:
    Tools for measuring execution time:
    - Context manager for timing code blocks
    - Function decorators
    - Profiling utilities
    - Performance reporting

Author: Author Name
Date: 2025-11-19
"""

import time
import functools
from typing import Optional, Callable, Dict, List
from collections import defaultdict
import statistics

from utils.logging import get_logger
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

logger = get_logger(__name__)


class Timer:
    """
    Context manager and decorator for timing code execution.

    Example:
        >>> # As context manager
        >>> with Timer("data_loading"):
        ...     data = load_data()

        >>> # As decorator
        >>> @Timer.decorator("training")
        ... def train_epoch():
        ...     pass
    """

    def __init__(
        self,
        name: str = "timer",
        log_on_exit: bool = True,
        verbose: bool = True
    ):
        """
        Initialize timer.

        Args:
            name: Timer name for logging
            log_on_exit: Whether to log elapsed time on exit
            verbose: Whether to print timing info
        """
        self.name = name
        self.log_on_exit = log_on_exit
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self) -> 'Timer':
        """Start timing."""
        self.start_time = time.time()
        if self.verbose:
            logger.debug(f"[{self.name}] Started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log."""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time

        if self.log_on_exit:
            if self.verbose:
                logger.info(f"[{self.name}] Completed in {self.elapsed:.4f}s")

        return False

    def get_elapsed(self) -> Optional[float]:
        """
        Get elapsed time in seconds.

        Returns:
            Elapsed time or None if not finished
        """
        return self.elapsed

    @staticmethod
    def decorator(name: str = "function", verbose: bool = True) -> Callable:
        """
        Create timing decorator.

        Args:
            name: Timer name
            verbose: Whether to log timing

        Returns:
            Decorator function

        Example:
            >>> @Timer.decorator("my_function")
            ... def my_function():
            ...     time.sleep(1)
        """
        def decorator_timer(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                timer_name = name if name != "function" else func.__name__
                with Timer(timer_name, verbose=verbose):
                    result = func(*args, **kwargs)
                return result
            return wrapper
        return decorator_timer


class TimingStats:
    """
    Accumulate timing statistics across multiple runs.

    Useful for measuring average, min, max execution times.

    Example:
        >>> stats = TimingStats("forward_pass")
        >>> for _ in range(100):
        ...     with stats.timer():
        ...         model(data)
        >>> print(stats.summary())
    """

    def __init__(self, name: str = "operation"):
        """
        Initialize timing stats.

        Args:
            name: Operation name
        """
        self.name = name
        self.times: List[float] = []

    def timer(self) -> Timer:
        """
        Create timer that records to this stats object.

        Returns:
            Timer instance
        """
        class RecordingTimer(Timer):
            def __init__(inner_self, stats_obj):
                super().__init__(name=stats_obj.name, log_on_exit=False, verbose=False)
                inner_self.stats = stats_obj

            def __exit__(inner_self, exc_type, exc_val, exc_tb):
                super().__exit__(exc_type, exc_val, exc_tb)
                if inner_self.elapsed is not None:
                    inner_self.stats.record(inner_self.elapsed)
                return False

        return RecordingTimer(self)

    def record(self, elapsed: float) -> None:
        """
        Record timing.

        Args:
            elapsed: Elapsed time in seconds
        """
        self.times.append(elapsed)

    def mean(self) -> float:
        """Get mean execution time."""
        return statistics.mean(self.times) if self.times else 0.0

    def median(self) -> float:
        """Get median execution time."""
        return statistics.median(self.times) if self.times else 0.0

    def std(self) -> float:
        """Get standard deviation."""
        return statistics.stdev(self.times) if len(self.times) > 1 else 0.0

    def min(self) -> float:
        """Get minimum execution time."""
        return min(self.times) if self.times else 0.0

    def max(self) -> float:
        """Get maximum execution time."""
        return max(self.times) if self.times else 0.0

    def count(self) -> int:
        """Get number of recorded timings."""
        return len(self.times)

    def summary(self) -> str:
        """
        Get formatted summary statistics.

        Returns:
            Summary string

        Example:
            >>> print(stats.summary())
            forward_pass: mean=0.123s, median=0.120s, std=0.015s (n=100)
        """
        if not self.times:
            return f"{self.name}: No timings recorded"

        return (
            f"{self.name}: "
            f"mean={self.mean():.4f}s, "
            f"median={self.median():.4f}s, "
            f"std={self.std():.4f}s, "
            f"min={self.min():.4f}s, "
            f"max={self.max():.4f}s "
            f"(n={self.count()})"
        )

    def reset(self) -> None:
        """Clear all recorded timings."""
        self.times.clear()


class Profiler:
    """
    Multi-operation profiler for tracking multiple timing stats.

    Example:
        >>> profiler = Profiler()
        >>>
        >>> with profiler.timer("data_loading"):
        ...     data = load_data()
        >>>
        >>> with profiler.timer("forward_pass"):
        ...     output = model(data)
        >>>
        >>> profiler.print_summary()
    """

    def __init__(self):
        """Initialize profiler."""
        self.stats: Dict[str, TimingStats] = defaultdict(lambda: TimingStats())

    def timer(self, name: str) -> Timer:
        """
        Create timer for named operation.

        Args:
            name: Operation name

        Returns:
            Timer instance
        """
        if name not in self.stats:
            self.stats[name] = TimingStats(name)

        return self.stats[name].timer()

    def get_stats(self, name: str) -> Optional[TimingStats]:
        """
        Get timing stats for named operation.

        Args:
            name: Operation name

        Returns:
            TimingStats or None
        """
        return self.stats.get(name)

    def print_summary(self) -> None:
        """Print summary of all tracked operations."""
        if not self.stats:
            logger.info("No profiling data available")
            return

        logger.info("=" * 80)
        logger.info("PROFILING SUMMARY")
        logger.info("=" * 80)

        for name, stats in sorted(self.stats.items()):
            logger.info(stats.summary())

        logger.info("=" * 80)

    def get_summary_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary as dictionary.

        Returns:
            Dictionary with stats for each operation
        """
        summary = {}
        for name, stats in self.stats.items():
            summary[name] = {
                'mean': stats.mean(),
                'median': stats.median(),
                'std': stats.std(),
                'min': stats.min(),
                'max': stats.max(),
                'count': stats.count()
            }
        return summary

    def reset(self, name: Optional[str] = None) -> None:
        """
        Reset profiler stats.

        Args:
            name: Operation name to reset (None = reset all)
        """
        if name is None:
            self.stats.clear()
        elif name in self.stats:
            self.stats[name].reset()


def time_function(func: Callable, *args, **kwargs) -> tuple:
    """
    Time a single function call.

    Args:
        func: Function to time
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        (result, elapsed_time) tuple

    Example:
        >>> result, elapsed = time_function(train_model, model, data)
        >>> print(f"Training took {elapsed:.2f}s")
    """
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    return result, elapsed


def benchmark(
    func: Callable,
    iterations: int = 100,
    warmup: int = 10,
    *args,
    **kwargs
) -> TimingStats:
    """
    Benchmark function over multiple iterations.

    Args:
        func: Function to benchmark
        iterations: Number of iterations
        warmup: Number of warmup iterations (not counted)
        *args: Function arguments
        **kwargs: Function keyword arguments

    Returns:
        TimingStats with results

    Example:
        >>> stats = benchmark(model.forward, iterations=100, warmup=10, data)
        >>> print(f"Mean: {stats.mean():.4f}s")
    """
    stats = TimingStats(func.__name__)

    # Warmup
    for _ in range(warmup):
        func(*args, **kwargs)

    # Benchmark
    for _ in range(iterations):
        start = time.time()
        func(*args, **kwargs)
        elapsed = time.time() - start
        stats.record(elapsed)

    return stats


def estimate_time_remaining(
    current_iter: int,
    total_iters: int,
    elapsed_so_far: float
) -> float:
    """
    Estimate remaining time based on current progress.

    Args:
        current_iter: Current iteration number
        total_iters: Total iterations
        elapsed_so_far: Time elapsed so far (seconds)

    Returns:
        Estimated remaining time (seconds)

    Example:
        >>> remaining = estimate_time_remaining(25, 100, 60.0)
        >>> print(f"ETA: {remaining:.0f}s")
    """
    if current_iter == 0:
        return float('inf')

    time_per_iter = elapsed_so_far / current_iter
    remaining_iters = total_iters - current_iter
    return time_per_iter * remaining_iters


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")

    Example:
        >>> print(format_time(3661))
        1h 1m 1s
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


# Global profiler instance
_global_profiler = Profiler()


def get_global_profiler() -> Profiler:
    """
    Get global profiler instance.

    Returns:
        Global Profiler

    Example:
        >>> profiler = get_global_profiler()
        >>> with profiler.timer("operation"):
        ...     do_something()
    """
    return _global_profiler
