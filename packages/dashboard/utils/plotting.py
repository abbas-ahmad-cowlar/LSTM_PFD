"""
Plotly figure helpers with consistent themes and layouts.
"""
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Optional, Dict, Any
import numpy as np
from dashboard_config import PLOT_THEME, COLOR_PALETTE
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


def create_base_layout(title: str, height: int = 400) -> Dict[str, Any]:
    """Create base layout with consistent styling."""
    return {
        "title": {"text": title, "font": {"size": 16}},
        "template": PLOT_THEME,
        "height": height,
        "margin": {"l": 50, "r": 50, "t": 50, "b": 50},
        "hovermode": "closest"
    }


def create_time_series_plot(
    time: np.ndarray,
    signal: np.ndarray,
    title: str = "Time Domain Signal",
    xlabel: str = "Time (s)",
    ylabel: str = "Amplitude"
) -> go.Figure:
    """Create time series plot."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=signal,
        mode='lines',
        name='Signal',
        line=dict(color=COLOR_PALETTE[0], width=1)
    ))

    fig.update_layout(**create_base_layout(title))
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)

    return fig


def create_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    normalize: bool = False
) -> go.Figure:
    """Create confusion matrix heatmap."""
    if normalize:
        cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        text_template = "%{z:.2%}"
    else:
        cm = confusion_matrix
        text_template = "%{z}"

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate=text_template,
        textfont={"size": 10},
        hovertemplate='Predicted: %{x}<br>True: %{y}<br>Count: %{z}<extra></extra>'
    ))

    fig.update_layout(**create_base_layout(title, height=500))
    fig.update_xaxes(title_text="Predicted Class")
    fig.update_yaxes(title_text="True Class")

    return fig


def create_bar_chart(
    x: List[str],
    y: List[float],
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    color: Optional[List[str]] = None
) -> go.Figure:
    """Create bar chart."""
    fig = go.Figure(data=[go.Bar(
        x=x,
        y=y,
        marker_color=color if color else COLOR_PALETTE[0]
    )])

    fig.update_layout(**create_base_layout(title))
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)

    return fig


def create_line_chart(
    x: List,
    y: List[List[float]],
    names: List[str],
    title: str,
    xlabel: str = "",
    ylabel: str = ""
) -> go.Figure:
    """Create multi-line chart."""
    fig = go.Figure()

    for idx, (y_data, name) in enumerate(zip(y, names)):
        fig.add_trace(go.Scatter(
            x=x,
            y=y_data,
            mode='lines+markers',
            name=name,
            line=dict(color=COLOR_PALETTE[idx % len(COLOR_PALETTE)])
        ))

    fig.update_layout(**create_base_layout(title))
    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=ylabel)

    return fig


def create_pie_chart(
    values: List[float],
    labels: List[str],
    title: str
) -> go.Figure:
    """Create pie chart."""
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=COLOR_PALETTE[:len(labels)]
    )])

    fig.update_layout(**create_base_layout(title))

    return fig
