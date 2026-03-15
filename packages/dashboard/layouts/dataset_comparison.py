"""
Dataset Comparison Dashboard Layout.

Provides side-by-side comparison of model performance across
different dataset versions (e.g., basic physics vs advanced physics).

Features:
- Dataset version selector
- Side-by-side accuracy table  
- Accuracy delta heatmap
- Best model leaderboard per config
- Config diff viewer
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

try:
    import dash
    from dash import html, dcc, dash_table
    from dash.dependencies import Input, Output, State
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_DASH = True
except ImportError:
    HAS_DASH = False


RESULTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "results"


def create_comparison_layout():
    """Create the dataset comparison dashboard layout."""
    if not HAS_DASH:
        return html.Div("Dash not installed. Run: pip install dash plotly")

    return html.Div([
        html.H2("Dataset Comparison", className="page-title"),
        html.P(
            "Compare model performance across different dataset versions "
            "(e.g., basic physics vs advanced physics).",
            className="page-description"
        ),

        # Version selectors
        html.Div([
            html.Div([
                html.Label("Dataset V1:"),
                dcc.Dropdown(
                    id="comparison-v1-selector",
                    placeholder="Select V1 results directory...",
                    className="version-dropdown",
                ),
            ], style={"width": "45%", "display": "inline-block"}),
            html.Div([
                html.Label("Dataset V2:"),
                dcc.Dropdown(
                    id="comparison-v2-selector",
                    placeholder="Select V2 results directory...",
                    className="version-dropdown",
                ),
            ], style={"width": "45%", "display": "inline-block", "marginLeft": "5%"}),
        ], className="version-selector-row"),

        html.Button(
            "Compare",
            id="comparison-run-btn",
            n_clicks=0,
            className="btn-primary",
            style={"marginTop": "10px"},
        ),

        html.Hr(),

        # Summary cards
        html.Div(id="comparison-summary", className="summary-cards"),

        # Comparison table
        html.Div(id="comparison-table", className="comparison-table"),

        # Delta chart
        dcc.Graph(id="comparison-delta-chart", className="comparison-chart"),

        # Leaderboard
        html.Div(id="comparison-leaderboard", className="leaderboard"),
    ], className="comparison-page")


def get_available_versions() -> List[Dict]:
    """Scan results directory for available dataset versions."""
    versions = []
    if RESULTS_DIR.exists():
        # Check for batch result files in root results dir
        batch_files = list(RESULTS_DIR.glob("batch_*.json"))
        if batch_files:
            versions.append({
                "label": f"results/ ({len(batch_files)} batches)",
                "value": str(RESULTS_DIR),
            })
        # Check subdirectories
        for subdir in sorted(RESULTS_DIR.iterdir()):
            if subdir.is_dir():
                sf = list(subdir.glob("batch_*.json")) + list(subdir.glob("*_results.json"))
                if sf:
                    versions.append({
                        "label": f"{subdir.name}/ ({len(sf)} results)",
                        "value": str(subdir),
                    })
    return versions


def create_delta_chart(comparison_data: Dict) -> go.Figure:
    """Create a bar chart showing accuracy deltas per model."""
    models = []
    deltas = []
    colors = []

    for model_key, data in sorted(
        comparison_data.items(),
        key=lambda x: x[1].get("delta", 0),
        reverse=True,
    ):
        if data.get("status") == "error":
            continue
        models.append(model_key)
        deltas.append(data.get("delta", 0))
        colors.append(
            "#2ecc71" if data.get("delta", 0) > 0.001
            else "#e74c3c" if data.get("delta", 0) < -0.001
            else "#95a5a6"
        )

    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=deltas,
            marker_color=colors,
            text=[f"{d:+.4f}" for d in deltas],
            textposition="outside",
        )
    ])
    fig.update_layout(
        title="Accuracy Delta per Model (V2 - V1)",
        xaxis_title="Model",
        yaxis_title="Accuracy Delta",
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=50, t=50, b=120),
        xaxis=dict(tickangle=45),
    )
    return fig
