"""
Enhanced Evaluation callbacks.
ROC analysis, error analysis, and architecture comparison.
"""
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from tasks.evaluation_tasks import generate_roc_analysis_task, error_analysis_task, architecture_comparison_task
from utils.logger import setup_logger
from database.connection import get_db_session
from models.experiment import Experiment, ExperimentStatus
import plotly.graph_objs as go
import numpy as np

logger = setup_logger(__name__)


def register_evaluation_callbacks(app):
    """Register enhanced evaluation callbacks."""

    @app.callback(
        [
            Output('eval-experiment-select', 'options'),
            Output('eval-compare-experiments', 'options')
        ],
        Input('url', 'pathname')
    )
    def load_experiments(pathname):
        """Load completed experiments for evaluation."""
        if pathname != '/evaluation':
            raise PreventUpdate

        try:
            with get_db_session() as session:
                experiments = session.query(Experiment).filter_by(
                    status=ExperimentStatus.COMPLETED
                ).order_by(Experiment.created_at.desc()).limit(50).all()

                options = [
                    {
                        "label": f"{exp.name} ({exp.model_type}) - {exp.metrics.get('test_accuracy', 0):.2%}",
                        "value": exp.id
                    }
                    for exp in experiments
                ]

                return options, options

        except Exception as e:
            logger.error(f"Failed to load experiments: {e}", exc_info=True)
            return [], []

    @app.callback(
        Output('roc-analysis-content', 'children'),
        Input('generate-roc-btn', 'n_clicks'),
        State('eval-experiment-select', 'value'),
        prevent_initial_call=True
    )
    def generate_roc_analysis(n_clicks, experiment_id):
        """Generate ROC analysis."""
        if not n_clicks or not experiment_id:
            raise PreventUpdate

        try:
            logger.info(f"Launching ROC analysis for experiment {experiment_id}")

            # Launch Celery task
            task = generate_roc_analysis_task.delay(experiment_id)

            return dbc.Alert([
                html.H5("ROC Analysis Started", className="alert-heading"),
                html.P(f"Task ID: {task.id}"),
                html.P("Generating ROC curves and AUC scores..."),
                dbc.Spinner(size="sm")
            ], color="info")

        except Exception as e:
            logger.error(f"Failed to start ROC analysis: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger")

    @app.callback(
        Output('error-analysis-content', 'children'),
        Input('analyze-errors-btn', 'n_clicks'),
        State('eval-experiment-select', 'value'),
        prevent_initial_call=True
    )
    def analyze_errors(n_clicks, experiment_id):
        """Analyze errors."""
        if not n_clicks or not experiment_id:
            raise PreventUpdate

        try:
            logger.info(f"Launching error analysis for experiment {experiment_id}")

            # Launch Celery task
            task = error_analysis_task.delay(experiment_id)

            return dbc.Alert([
                html.H5("Error Analysis Started", className="alert-heading"),
                html.P(f"Task ID: {task.id}"),
                html.P("Analyzing misclassifications and confusion patterns..."),
                dbc.Spinner(size="sm")
            ], color="info")

        except Exception as e:
            logger.error(f"Failed to start error analysis: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger")

    @app.callback(
        Output('architecture-comparison-content', 'children'),
        Input('compare-architectures-btn', 'n_clicks'),
        State('eval-compare-experiments', 'value'),
        prevent_initial_call=True
    )
    def compare_architectures(n_clicks, experiment_ids):
        """Compare architectures."""
        if not n_clicks or not experiment_ids:
            raise PreventUpdate

        if len(experiment_ids) < 2:
            return dbc.Alert("Please select at least 2 experiments to compare", color="warning")

        try:
            logger.info(f"Comparing {len(experiment_ids)} architectures")

            # Launch Celery task
            task = architecture_comparison_task.delay(experiment_ids)

            return dbc.Alert([
                html.H5("Architecture Comparison Started", className="alert-heading"),
                html.P(f"Task ID: {task.id}"),
                html.P(f"Comparing {len(experiment_ids)} architectures..."),
                dbc.Spinner(size="sm")
            ], color="info")

        except Exception as e:
            logger.error(f"Failed to start architecture comparison: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger")


def create_roc_curves_plot(roc_data: dict, class_names: list):
    """
    Create ROC curves plot.

    Args:
        roc_data: ROC data from service
        class_names: List of class names

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    roc_curves = roc_data.get('roc_curves', {})
    auc_scores = roc_data.get('auc_scores', {})

    for class_name in class_names:
        if class_name in roc_curves:
            fpr, tpr = roc_curves[class_name]
            auc = auc_scores.get(class_name, 0)

            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'{class_name} (AUC={auc:.3f})',
                line=dict(width=2)
            ))

    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random (AUC=0.5)',
        line=dict(dash='dash', color='gray')
    ))

    fig.update_layout(
        title="ROC Curves (One-vs-Rest)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        hovermode='closest',
        height=600
    )

    return fig


def create_confusion_heatmap(confusion_matrix: list, class_names: list):
    """
    Create confusion matrix heatmap.

    Args:
        confusion_matrix: Confusion matrix
        class_names: List of class names

    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=confusion_matrix,
        texttemplate='%{text}',
        textfont={"size": 10}
    ))

    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted",
        yaxis_title="True",
        height=600
    )

    return fig
