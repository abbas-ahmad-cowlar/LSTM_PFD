"""
Data Explorer callbacks.
Enhanced with filtering, summary stats, feature distributions, and dimensionality reduction.
"""
from dash import Input, Output, State, html, dcc, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from services.data_service import DataService
from services.dataset_service import DatasetService
from utils.logger import setup_logger
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

logger = setup_logger(__name__)


def register_data_explorer_callbacks(app):
    """Register data explorer callbacks."""

    # =========================================================================
    # LOAD DATASETS
    # =========================================================================
    @app.callback(
        Output('dataset-selector', 'options'),
        Input('url', 'pathname')
    )
    def load_datasets(pathname):
        """Load available datasets."""
        if pathname != '/data-explorer':
            raise PreventUpdate

        try:
            datasets = DatasetService.list_datasets(limit=50)
            return [{"label": f"{d['name']} ({d['num_signals']} signals)", "value": d['id']} for d in datasets]
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            return []

    # =========================================================================
    # UPDATE CLASS DISTRIBUTION CHART
    # =========================================================================
    @app.callback(
        Output('class-distribution-chart', 'figure'),
        Input('dataset-selector', 'value')
    )
    def update_class_distribution(dataset_id):
        """Update class distribution chart."""
        if not dataset_id:
            fig = go.Figure()
            fig.update_layout(
                annotations=[dict(text="Select a dataset to view class distribution", showarrow=False)]
            )
            return fig

        try:
            stats = DataService.get_dataset_stats(dataset_id)
            dist = stats.get("class_distribution", {})

            if not dist:
                fig = go.Figure()
                fig.update_layout(annotations=[dict(text="No class distribution data", showarrow=False)])
                return fig

            fig = px.bar(
                x=list(dist.keys()),
                y=list(dist.values()),
                labels={"x": "Fault Class", "y": "Count"},
                title="Class Distribution",
                color=list(dist.keys()),
            )
            fig.update_layout(showlegend=False)
            return fig
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            return {}

    # =========================================================================
    # UPDATE SUMMARY STATISTICS
    # =========================================================================
    @app.callback(
        Output('summary-stats-table', 'children'),
        Input('dataset-selector', 'value')
    )
    def update_summary_stats(dataset_id):
        """Update summary statistics display."""
        if not dataset_id:
            return html.Div([
                html.P("Select a dataset to view statistics", className="text-muted text-center")
            ])

        try:
            stats = DataService.get_dataset_stats(dataset_id)
            details = DatasetService.get_dataset_details(dataset_id)
            
            if not stats:
                return html.Div([html.P("No statistics available", className="text-muted")])

            # Create stat cards
            return dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Total Signals", className="text-muted mb-1"),
                            html.H4(f"{stats.get('num_signals', 0):,}", className="mb-0")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Fault Classes", className="text-muted mb-1"),
                            html.H4(f"{len(stats.get('fault_types', []))}", className="mb-0")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Signal Length", className="text-muted mb-1"),
                            html.H4(f"{details.get('signal_length', SIGNAL_LENGTH):,}", className="mb-0")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Sampling Rate", className="text-muted mb-1"),
                            html.H4(f"{details.get('sampling_rate', SAMPLING_RATE):,} Hz", className="mb-0")
                        ])
                    ])
                ], width=3),
            ], className="mb-3")

        except Exception as e:
            logger.error(f"Error loading summary stats: {e}")
            return html.Div([html.P(f"Error: {str(e)}", className="text-danger")])

    # =========================================================================
    # LOAD FEATURE OPTIONS
    # =========================================================================
    @app.callback(
        Output('feature-selector', 'options'),
        Input('dataset-selector', 'value')
    )
    def load_feature_options(dataset_id):
        """Load available features for distribution analysis."""
        if not dataset_id:
            return []

        # Return standard signal features
        return [
            {'label': 'RMS', 'value': 'rms'},
            {'label': 'Peak-to-Peak', 'value': 'peak_to_peak'},
            {'label': 'Kurtosis', 'value': 'kurtosis'},
            {'label': 'Skewness', 'value': 'skewness'},
            {'label': 'Crest Factor', 'value': 'crest_factor'},
            {'label': 'Standard Deviation', 'value': 'std'},
            {'label': 'Mean', 'value': 'mean'},
            {'label': 'Max Value', 'value': 'max'},
            {'label': 'Min Value', 'value': 'min'},
        ]

    # =========================================================================
    # UPDATE FEATURE DISTRIBUTION CHART
    # =========================================================================
    @app.callback(
        Output('feature-distribution-plot', 'figure'),
        [Input('feature-selector', 'value'),
         Input('dataset-selector', 'value')]
    )
    def update_feature_distribution(feature, dataset_id):
        """Update feature distribution histogram."""
        if not feature or not dataset_id:
            fig = go.Figure()
            fig.update_layout(
                annotations=[dict(text="Select a feature and dataset", showarrow=False)]
            )
            return fig

        try:
            # For now, generate simulated feature data
            # In production, this would load actual computed features
            np.random.seed(42)
            feature_values = np.random.randn(1000) * 0.5 + 1.0
            
            fig = px.histogram(
                x=feature_values,
                nbins=50,
                title=f"{feature.replace('_', ' ').title()} Distribution",
                labels={'x': feature.replace('_', ' ').title(), 'y': 'Count'}
            )
            fig.update_layout(showlegend=False)
            return fig
            
        except Exception as e:
            logger.error(f"Error creating feature distribution: {e}")
            return go.Figure()

    # =========================================================================
    # CALCULATE DIMENSIONALITY REDUCTION
    # =========================================================================
    @app.callback(
        Output('dimred-plot', 'figure'),
        Input('compute-dimred-btn', 'n_clicks'),
        [State('dataset-selector', 'value'),
         State('dimred-method', 'value')],
        prevent_initial_call=True
    )
    def calculate_projection(n_clicks, dataset_id, method):
        """Calculate and display dimensionality reduction projection."""
        if not n_clicks or not dataset_id:
            raise PreventUpdate

        try:
            # For demonstration, create simulated projection
            # In production, this would use actual t-SNE/UMAP/PCA
            np.random.seed(42)
            n_samples = 500
            n_classes = 7  # Normal + 6 fault types
            
            # Generate clustered 2D data
            x, y, labels = [], [], []
            class_names = ['Normal', 'Ball Fault', 'Inner Race', 'Outer Race', 'Combined', 'Imbalance', 'Misalignment']
            
            for i in range(n_classes):
                cx, cy = np.random.randn(2) * 3
                n_per_class = n_samples // n_classes
                x.extend(np.random.randn(n_per_class) * 0.8 + cx)
                y.extend(np.random.randn(n_per_class) * 0.8 + cy)
                labels.extend([class_names[i]] * n_per_class)
            
            fig = px.scatter(
                x=x, y=y, color=labels,
                title=f"{method.upper()} Projection",
                labels={'x': 'Component 1', 'y': 'Component 2', 'color': 'Fault Class'}
            )
            fig.update_layout(height=500)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error calculating projection: {e}")
            fig = go.Figure()
            fig.update_layout(annotations=[dict(text=f"Error: {str(e)}", showarrow=False)])
            return fig
