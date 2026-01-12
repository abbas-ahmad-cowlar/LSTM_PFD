"""
Feature Engineering callbacks.
Handle feature extraction, selection, and importance analysis.
"""
import numpy as np
from dash import Input, Output, State, html, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from tasks.feature_tasks import (
    extract_features_task,
    compute_importance_task,
    select_features_task,
    compute_correlation_task
)
from services.dataset_service import DatasetService
from utils.logger import setup_logger

logger = setup_logger(__name__)


def register_feature_callbacks(app):
    """Register feature engineering callbacks."""

    # Load datasets into all dropdowns
    @app.callback(
        [Output('fe-dataset-select', 'options'),
         Output('fi-dataset-select', 'options'),
         Output('fs-dataset-select', 'options'),
         Output('corr-dataset-select', 'options')],
        Input('url', 'pathname')
    )
    def load_datasets(pathname):
        """Load datasets for all tabs."""
        if pathname != '/feature-engineering':
            raise PreventUpdate

        try:
            datasets = DatasetService.list_datasets(limit=100)

            options = [
                {
                    "label": f"{ds['name']} ({ds['num_signals']} signals)",
                    "value": ds['id']
                }
                for ds in datasets
            ]

            # Return same options for all 4 dropdowns
            return options, options, options, options

        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            return [], [], [], []

    # Feature Extraction
    @app.callback(
        Output('fe-extraction-results', 'children'),
        Input('extract-features-btn', 'n_clicks'),
        [State('fe-dataset-select', 'value'),
         State('fe-domain', 'value')],
        prevent_initial_call=True
    )
    def extract_features(n_clicks, dataset_id, domain):
        """Extract features from dataset."""
        if not n_clicks or not dataset_id:
            raise PreventUpdate

        try:
            logger.info(f"Launching feature extraction: dataset={dataset_id}, domain={domain}")

            # Launch Celery task - check if connection is available
            try:
                task = extract_features_task.delay(dataset_id, domain)
                
                return dbc.Alert([
                    html.H5("✅ Feature Extraction Started", className="alert-heading"),
                    html.P(f"Task ID: {task.id}"),
                    html.P([
                        html.Strong("Dataset ID: "), f"{dataset_id}", html.Br(),
                        html.Strong("Domain: "), domain, html.Br(),
                        html.Strong("Status: "), "Processing..."
                    ]),
                    dbc.Spinner(size="sm", className="mt-2"),
                    html.P("This may take several minutes depending on dataset size.", className="mt-3 text-muted small")
                ], color="info")
                
            except Exception as celery_error:
                # Celery/Redis not available - provide user-friendly message
                logger.warning(f"Celery task submission failed: {celery_error}")
                return dbc.Alert([
                    html.H5("⚠️ Background Worker Not Available", className="alert-heading"),
                    html.P([
                        "Feature extraction requires the Celery background worker to be running.",
                        html.Br(),
                        "Please start Redis and the Celery worker to enable this feature."
                    ]),
                    html.Hr(),
                    html.P([
                        html.Strong("To start the worker:"), html.Br(),
                        html.Code("celery -A tasks worker --loglevel=info", className="d-block mt-2 p-2 bg-dark")
                    ]),
                    html.P([
                        html.Strong("Configuration: "), html.Br(),
                        f"Dataset: {dataset_id}", html.Br(),
                        f"Domain: {domain}"
                    ], className="mt-3 small text-muted")
                ], color="warning")

        except Exception as e:
            logger.error(f"Failed to start feature extraction: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger")

    # Feature Importance
    @app.callback(
        Output('fi-results', 'children'),
        Input('compute-importance-btn', 'n_clicks'),
        [State('fi-dataset-select', 'value'),
         State('fi-domain', 'value'),
         State('fi-method', 'value')],
        prevent_initial_call=True
    )
    def compute_importance(n_clicks, dataset_id, domain, method):
        """Compute feature importance."""
        if not n_clicks or not dataset_id:
            raise PreventUpdate

        try:
            logger.info(f"Launching importance computation: dataset={dataset_id}, method={method}")

            # Launch Celery task
            task = compute_importance_task.delay(dataset_id, domain, method)

            return dbc.Alert([
                html.H5("Feature Importance Computation Started", className="alert-heading"),
                html.P(f"Task ID: {task.id}"),
                html.P([
                    html.Strong("Dataset ID: "), f"{dataset_id}", html.Br(),
                    html.Strong("Domain: "), domain, html.Br(),
                    html.Strong("Method: "), method, html.Br(),
                    html.Strong("Status: "), "Processing..."
                ]),
                dbc.Spinner(size="sm", className="mt-2"),
                html.P("Computing feature importance scores...", className="mt-3 text-muted small")
            ], color="info")

        except Exception as e:
            logger.error(f"Failed to start importance computation: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger")

    # Feature Selection
    @app.callback(
        Output('fs-results', 'children'),
        Input('select-features-btn', 'n_clicks'),
        [State('fs-dataset-select', 'value'),
         State('fs-domain', 'value'),
         State('fs-method', 'value'),
         State('fs-num-features', 'value')],
        prevent_initial_call=True
    )
    def select_features(n_clicks, dataset_id, domain, method, num_features):
        """Select features."""
        if not n_clicks or not dataset_id:
            raise PreventUpdate

        try:
            logger.info(f"Launching feature selection: dataset={dataset_id}, method={method}, n={num_features}")

            # Launch Celery task
            task = select_features_task.delay(dataset_id, domain, method, num_features)

            return dbc.Alert([
                html.H5("Feature Selection Started", className="alert-heading"),
                html.P(f"Task ID: {task.id}"),
                html.P([
                    html.Strong("Dataset ID: "), f"{dataset_id}", html.Br(),
                    html.Strong("Domain: "), domain, html.Br(),
                    html.Strong("Method: "), method, html.Br(),
                    html.Strong("Target Features: "), f"{num_features}", html.Br(),
                    html.Strong("Status: "), "Processing..."
                ]),
                dbc.Spinner(size="sm", className="mt-2"),
                html.P("Selecting most informative features...", className="mt-3 text-muted small")
            ], color="info")

        except Exception as e:
            logger.error(f"Failed to start feature selection: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger")

    # Correlation Analysis
    @app.callback(
        Output('corr-results', 'children'),
        Input('compute-correlation-btn', 'n_clicks'),
        [State('corr-dataset-select', 'value'),
         State('corr-domain', 'value')],
        prevent_initial_call=True
    )
    def compute_correlation(n_clicks, dataset_id, domain):
        """Compute feature correlation."""
        if not n_clicks or not dataset_id:
            raise PreventUpdate

        try:
            logger.info(f"Launching correlation computation: dataset={dataset_id}, domain={domain}")

            # Launch Celery task
            task = compute_correlation_task.delay(dataset_id, domain)

            return dbc.Alert([
                html.H5("Correlation Analysis Started", className="alert-heading"),
                html.P(f"Task ID: {task.id}"),
                html.P([
                    html.Strong("Dataset ID: "), f"{dataset_id}", html.Br(),
                    html.Strong("Domain: "), domain, html.Br(),
                    html.Strong("Status: "), "Processing..."
                ]),
                dbc.Spinner(size="sm", className="mt-2"),
                html.P("Computing feature correlations...", className="mt-3 text-muted small")
            ], color="info")

        except Exception as e:
            logger.error(f"Failed to start correlation computation: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger")


def create_importance_chart(importances_dict: dict, method: str) -> go.Figure:
    """
    Create feature importance bar chart.

    Args:
        importances_dict: Dictionary mapping feature names to importance scores
        method: Importance method used

    Returns:
        Plotly figure
    """
    # Sort by importance
    sorted_features = sorted(importances_dict.items(), key=lambda x: x[1], reverse=True)

    # Take top 20
    top_features = sorted_features[:20]

    feature_names = [f[0] for f in top_features]
    importance_scores = [f[1] for f in top_features]

    fig = go.Figure(data=[
        go.Bar(
            y=feature_names,
            x=importance_scores,
            orientation='h',
            marker=dict(
                color=importance_scores,
                colorscale='Viridis',
                showscale=True
            )
        )
    ])

    fig.update_layout(
        title=f"Top 20 Features by {method.replace('_', ' ').title()} Importance",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=600,
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig


def create_selected_features_table(selected_names: list) -> html.Div:
    """
    Create table of selected features.

    Args:
        selected_names: List of selected feature names

    Returns:
        Dash HTML table
    """
    table_rows = []
    for idx, name in enumerate(selected_names, 1):
        table_rows.append(
            html.Tr([
                html.Td(idx),
                html.Td(name),
            ])
        )

    return dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("#"),
                html.Th("Feature Name"),
            ])
        ]),
        html.Tbody(table_rows)
    ], bordered=True, striped=True, hover=True)


def create_correlation_heatmap(corr_matrix: list, feature_names: list) -> go.Figure:
    """
    Create correlation heatmap.

    Args:
        corr_matrix: Correlation matrix as 2D list
        feature_names: List of feature names

    Returns:
        Plotly figure
    """
    corr_matrix = np.array(corr_matrix)

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=feature_names,
        y=feature_names,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 8},
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title="Feature Correlation Matrix",
        xaxis_title="Features",
        yaxis_title="Features",
        height=max(600, len(feature_names) * 20),
        width=max(800, len(feature_names) * 20)
    )

    return fig


def create_high_correlation_table(high_corr_pairs: list) -> html.Div:
    """
    Create table of highly correlated feature pairs.

    Args:
        high_corr_pairs: List of dicts with feature1, feature2, correlation

    Returns:
        Dash HTML table
    """
    if not high_corr_pairs:
        return dbc.Alert("No highly correlated feature pairs found (|r| > 0.8)", color="info")

    table_rows = []
    for pair in high_corr_pairs:
        table_rows.append(
            html.Tr([
                html.Td(pair['feature1']),
                html.Td(pair['feature2']),
                html.Td(f"{pair['correlation']:.3f}"),
            ])
        )

    return html.Div([
        html.H5("Highly Correlated Features (|r| > 0.8)", className="mb-3"),
        dbc.Table([
            html.Thead([
                html.Tr([
                    html.Th("Feature 1"),
                    html.Th("Feature 2"),
                    html.Th("Correlation"),
                ])
            ]),
            html.Tbody(table_rows)
        ], bordered=True, striped=True, hover=True)
    ])
