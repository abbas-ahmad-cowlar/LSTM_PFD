"""
Visualization utilities for plotting and figure management.

Purpose:
    Helper functions for consistent plotting:
    - Matplotlib style configuration
    - Figure saving with standardized settings
    - Common plot types
    - Color palettes

Author: Syed Abbas Ahmad
Date: 2025-11-19
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Optional, Tuple, List, Union
import numpy as np

from utils.logging import get_logger
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

logger = get_logger(__name__)


# Default style settings
DEFAULT_STYLE = {
    'figure.figsize': (10, 6),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
}


def set_plot_style(style: str = 'default') -> None:
    """
    Set consistent plotting style.

    Args:
        style: Style name ('default', 'seaborn', 'ggplot', 'dark')

    Example:
        >>> set_plot_style('default')
        >>> # All subsequent plots use this style
    """
    if style == 'default':
        # Apply custom default style
        mpl.rcParams.update(DEFAULT_STYLE)
    elif style == 'seaborn':
        plt.style.use('seaborn-v0_8-darkgrid')
    elif style == 'ggplot':
        plt.style.use('ggplot')
    elif style == 'dark':
        plt.style.use('dark_background')
    else:
        logger.warning(f"Unknown style: {style}. Using 'default'")
        mpl.rcParams.update(DEFAULT_STYLE)

    logger.debug(f"Set plot style: {style}")


def save_figure(
    fig: plt.Figure,
    path: Union[str, Path],
    dpi: int = 300,
    bbox_inches: str = 'tight',
    transparent: bool = False,
    close_after: bool = True
) -> None:
    """
    Save figure with consistent settings.

    Args:
        fig: Matplotlib figure
        path: Output file path
        dpi: Resolution (dots per inch)
        bbox_inches: Bounding box setting
        transparent: Transparent background
        close_after: Close figure after saving

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot(data)
        >>> save_figure(fig, 'output.png')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches=bbox_inches,
        transparent=transparent
    )

    logger.debug(f"Saved figure to {path}")

    if close_after:
        plt.close(fig)


def create_figure(
    figsize: Optional[Tuple[int, int]] = None,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs
) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """
    Create figure with subplots.

    Args:
        figsize: Figure size (width, height) in inches
        nrows: Number of rows
        ncols: Number of columns
        **kwargs: Additional arguments to plt.subplots()

    Returns:
        (fig, axes) tuple

    Example:
        >>> fig, (ax1, ax2) = create_figure(ncols=2, figsize=(12, 5))
    """
    if figsize is None:
        figsize = (10 * ncols, 6 * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, **kwargs)

    return fig, axes


def get_color_palette(name: str = 'default', n_colors: int = 11) -> List[str]:
    """
    Get color palette for consistent plotting.

    Args:
        name: Palette name ('default', 'vibrant', 'pastel', 'dark')
        n_colors: Number of colors needed

    Returns:
        List of hex color codes

    Example:
        >>> colors = get_color_palette('vibrant', n_colors=5)
        >>> for i, color in enumerate(colors):
        ...     plt.plot(data[i], color=color)
    """
    palettes = {
        'default': plt.cm.tab10.colors[:n_colors],
        'vibrant': plt.cm.Set1.colors[:n_colors],
        'pastel': plt.cm.Pastel1.colors[:n_colors],
        'dark': plt.cm.Dark2.colors[:n_colors],
    }

    if name in palettes:
        colors = palettes[name]
    else:
        logger.warning(f"Unknown palette: {name}. Using 'default'")
        colors = palettes['default']

    # Convert to hex
    hex_colors = [mpl.colors.rgb2hex(c) for c in colors]

    return hex_colors


def add_grid(ax: plt.Axes, alpha: float = 0.3, linestyle: str = '--') -> None:
    """
    Add grid to axes.

    Args:
        ax: Matplotlib axes
        alpha: Grid transparency
        linestyle: Line style

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot(data)
        >>> add_grid(ax)
    """
    ax.grid(True, alpha=alpha, linestyle=linestyle)


def add_legend(
    ax: plt.Axes,
    loc: str = 'best',
    frameon: bool = True,
    **kwargs
) -> None:
    """
    Add legend to axes with consistent styling.

    Args:
        ax: Matplotlib axes
        loc: Legend location
        frameon: Show legend frame
        **kwargs: Additional legend arguments

    Example:
        >>> ax.plot(x, y, label='data')
        >>> add_legend(ax, loc='upper right')
    """
    ax.legend(loc=loc, frameon=frameon, **kwargs)


def set_axis_labels(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: Optional[str] = None
) -> None:
    """
    Set axis labels and title.

    Args:
        ax: Matplotlib axes
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title (optional)

    Example:
        >>> set_axis_labels(ax, 'Time (s)', 'Amplitude', 'Signal Plot')
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def plot_time_series(
    signal: np.ndarray,
    fs: float = 20480.0,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot time-domain signal.

    Args:
        signal: Signal array
        fs: Sampling frequency
        ax: Matplotlib axes (creates new if None)
        title: Plot title
        **kwargs: Additional plot arguments

    Returns:
        Axes object

    Example:
        >>> ax = plot_time_series(signal, fs = SAMPLING_RATE, title='Vibration Signal')
    """
    if ax is None:
        fig, ax = create_figure()

    t = np.arange(len(signal)) / fs
    ax.plot(t, signal, **kwargs)

    set_axis_labels(ax, 'Time (s)', 'Amplitude', title)
    add_grid(ax)

    return ax


def plot_spectrum(
    signal: np.ndarray,
    fs: float = 20480.0,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot frequency spectrum.

    Args:
        signal: Signal array
        fs: Sampling frequency
        ax: Matplotlib axes
        title: Plot title
        xlim: Frequency range to display
        **kwargs: Additional plot arguments

    Returns:
        Axes object

    Example:
        >>> ax = plot_spectrum(signal, fs = SAMPLING_RATE, xlim=(0, 500))
    """
    if ax is None:
        fig, ax = create_figure()

    # Compute FFT
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    magnitude = np.abs(fft)

    ax.plot(freqs, magnitude, **kwargs)

    if xlim:
        ax.set_xlim(xlim)

    set_axis_labels(ax, 'Frequency (Hz)', 'Magnitude', title)
    add_grid(ax)

    return ax


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    ax: Optional[plt.Axes] = None,
    cmap: str = 'Blues',
    normalize: bool = False,
    **kwargs
) -> plt.Axes:
    """
    Plot confusion matrix as heatmap.

    Args:
        cm: Confusion matrix array (n_classes, n_classes)
        class_names: List of class names
        ax: Matplotlib axes
        cmap: Colormap name
        normalize: Normalize to percentages
        **kwargs: Additional imshow arguments

    Returns:
        Axes object

    Example:
        >>> from sklearn.metrics import confusion_matrix
        >>> cm = confusion_matrix(y_true, y_pred)
        >>> plot_confusion_matrix(cm, class_names)
    """
    if ax is None:
        fig, ax = create_figure(figsize=(10, 8))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, **kwargs)
    ax.figure.colorbar(im, ax=ax)

    # Set ticks
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)

    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

    set_axis_labels(ax, 'Predicted', 'True', 'Confusion Matrix')

    return ax


def plot_training_history(
    history: dict,
    metrics: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot training history (loss, accuracy, etc.).

    Args:
        history: Dictionary with metric histories
        metrics: Metrics to plot (None = all)
        ax: Matplotlib axes
        **kwargs: Additional plot arguments

    Returns:
        Axes object

    Example:
        >>> history = {'loss': [...], 'val_loss': [...]}
        >>> plot_training_history(history)
    """
    if ax is None:
        fig, ax = create_figure()

    if metrics is None:
        metrics = list(history.keys())

    for metric in metrics:
        if metric in history:
            ax.plot(history[metric], label=metric, **kwargs)

    set_axis_labels(ax, 'Epoch', 'Value', 'Training History')
    add_legend(ax)
    add_grid(ax)

    return ax


def annotate_max_min(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    annotate_max: bool = True,
    annotate_min: bool = True
) -> None:
    """
    Annotate maximum and minimum points on plot.

    Args:
        ax: Matplotlib axes
        x: X-coordinates
        y: Y-coordinates
        annotate_max: Annotate maximum
        annotate_min: Annotate minimum

    Example:
        >>> ax.plot(x, y)
        >>> annotate_max_min(ax, x, y)
    """
    if annotate_max:
        max_idx = np.argmax(y)
        ax.annotate(f'Max: {y[max_idx]:.3f}',
                   xy=(x[max_idx], y[max_idx]),
                   xytext=(10, 10),
                   textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', color='red'))

    if annotate_min:
        min_idx = np.argmin(y)
        ax.annotate(f'Min: {y[min_idx]:.3f}',
                   xy=(x[min_idx], y[min_idx]),
                   xytext=(10, -20),
                   textcoords='offset points',
                   arrowprops=dict(arrowstyle='->', color='blue'))


# Initialize default style on import
set_plot_style('default')
