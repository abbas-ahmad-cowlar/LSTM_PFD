"""
Data Explorer callbacks.
"""
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
import plotly.express as px

from services.data_service import DataService
from utils.logger import setup_logger
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

logger = setup_logger(__name__)


def register_data_explorer_callbacks(app):
    """Register data explorer callbacks."""

    @app.callback(
        Output('dataset-selector', 'options'),
        Input('url', 'pathname')
    )
    def load_datasets(pathname):
        """Load available datasets."""
        if pathname != '/data-explorer':
            raise PreventUpdate

        try:
            datasets = DataService.list_datasets()
            return [{"label": d["name"], "value": d["id"]} for d in datasets]
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            return []

    @app.callback(
        Output('class-distribution-chart', 'figure'),
        Input('dataset-selector', 'value')
    )
    def update_class_distribution(dataset_id):
        """Update class distribution chart."""
        if not dataset_id:
            raise PreventUpdate

        try:
            stats = DataService.get_dataset_stats(dataset_id)
            dist = stats.get("class_distribution", {})

            fig = px.bar(
                x=list(dist.keys()),
                y=list(dist.values()),
                labels={"x": "Fault Class", "y": "Count"},
                title="Class Distribution"
            )
            return fig
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            return {}
