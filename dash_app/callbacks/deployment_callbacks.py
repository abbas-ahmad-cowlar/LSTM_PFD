"""
Deployment callbacks (Phase 11).
Callbacks for model deployment dashboard.
"""
from dash import Input, Output, State, html, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from layouts.deployment import create_model_info_card, create_deployment_result_card
from services.deployment_service import DeploymentService
from tasks.deployment_tasks import quantize_model_task, export_onnx_task, optimize_model_task, benchmark_models_task
from utils.logger import setup_logger
from database.connection import get_db_session
from models.experiment import Experiment, ExperimentStatus
from dash import dcc
import plotly.graph_objs as go
from utils.constants import (
    DEFAULT_PAGE_SIZE,
    PERCENT_DIVISOR,
)

logger = setup_logger(__name__)


def register_deployment_callbacks(app):
    """Register deployment dashboard callbacks."""

    @app.callback(
        Output('deployment-experiment-select', 'options'),
        Input('url', 'pathname')
    )
    def load_completed_experiments(pathname):
        """Load completed experiments for deployment."""
        if pathname != '/deployment':
            raise PreventUpdate

        try:
            with get_db_session() as session:
                experiments = session.query(Experiment).filter_by(
                    status=ExperimentStatus.COMPLETED
                ).order_by(Experiment.created_at.desc()).limit(DEFAULT_PAGE_SIZE).all()

                options = [
                    {
                        "label": f"{exp.name} ({exp.model_type}) - {exp.metrics.get('test_accuracy', 0):.2%}",
                        "value": exp.id
                    }
                    for exp in experiments
                ]

                return options

        except Exception as e:
            logger.error(f"Failed to load experiments: {e}", exc_info=True)
            return []

    @app.callback(
        Output('deployment-model-info', 'children'),
        Input('deployment-experiment-select', 'value')
    )
    def display_model_info(experiment_id):
        """Display selected model information."""
        if not experiment_id:
            raise PreventUpdate

        try:
            with get_db_session() as session:
                exp = session.query(Experiment).filter_by(id=experiment_id).first()
                if not exp:
                    return dbc.Alert("Experiment not found", color="danger")

                experiment_data = {
                    'name': exp.name,
                    'model_type': exp.model_type,
                    'test_accuracy': exp.metrics.get('test_accuracy', 0) if exp.metrics else 0,
                    'status': exp.status.value,
                    'created_at': exp.created_at.isoformat() if exp.created_at else 'N/A'
                }

                return create_model_info_card(experiment_data)

        except Exception as e:
            logger.error(f"Failed to display model info: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger")

    @app.callback(
        Output('deployment-results', 'children'),
        Input('quantize-btn', 'n_clicks'),
        [
            State('deployment-experiment-select', 'value'),
            State('quantization-type', 'value')
        ],
        prevent_initial_call=True
    )
    def quantize_model(n_clicks, experiment_id, quantization_type):
        """Trigger model quantization."""
        if not n_clicks or not experiment_id:
            raise PreventUpdate

        try:
            logger.info(f"Launching quantization task for experiment {experiment_id}")

            # Launch Celery task
            task = quantize_model_task.delay(experiment_id, quantization_type)

            return dbc.Alert(
                [
                    html.H5("Quantization Started", className="alert-heading"),
                    html.P(f"Task ID: {task.id}"),
                    html.P("The model is being quantized in the background. This may take a few minutes."),
                    dbc.Spinner(size="sm")
                ],
                color="info"
            )

        except Exception as e:
            logger.error(f"Failed to start quantization: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger")

    @app.callback(
        Output('deployment-results', 'children', allow_duplicate=True),
        Input('onnx-export-btn', 'n_clicks'),
        [
            State('deployment-experiment-select', 'value'),
            State('onnx-opset-version', 'value'),
            State('onnx-options', 'value')
        ],
        prevent_initial_call=True
    )
    def export_onnx(n_clicks, experiment_id, opset_version, options):
        """Trigger ONNX export."""
        if not n_clicks or not experiment_id:
            raise PreventUpdate

        try:
            optimize = 'optimize' in (options or [])
            dynamic_axes = 'dynamic_axes' in (options or [])

            logger.info(f"Launching ONNX export task for experiment {experiment_id}")

            # Launch Celery task
            task = export_onnx_task.delay(experiment_id, opset_version, optimize, dynamic_axes)

            return dbc.Alert(
                [
                    html.H5("ONNX Export Started", className="alert-heading"),
                    html.P(f"Task ID: {task.id}"),
                    html.P("The model is being exported to ONNX format."),
                    dbc.Spinner(size="sm")
                ],
                color="info"
            )

        except Exception as e:
            logger.error(f"Failed to start ONNX export: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger")

    @app.callback(
        Output('deployment-results', 'children', allow_duplicate=True),
        Input('optimize-btn', 'n_clicks'),
        [
            State('deployment-experiment-select', 'value'),
            State('pruning-method', 'value'),
            State('pruning-amount', 'value'),
            State('optimization-options', 'value')
        ],
        prevent_initial_call=True
    )
    def optimize_model(n_clicks, experiment_id, pruning_method, pruning_amount, options):
        """Trigger model optimization."""
        if not n_clicks or not experiment_id:
            raise PreventUpdate

        try:
            apply_fusion = 'fusion' in (options or [])

            logger.info(f"Launching optimization task for experiment {experiment_id}")

            # Launch Celery task
            task = optimize_model_task.delay(
                experiment_id,
                pruning_method,
                pruning_amount / PERCENT_DIVISOR,  # Convert percentage to fraction
                apply_fusion
            )

            return dbc.Alert(
                [
                    html.H5("Optimization Started", className="alert-heading"),
                    html.P(f"Task ID: {task.id}"),
                    html.P("The model is being optimized."),
                    dbc.Spinner(size="sm")
                ],
                color="info"
            )

        except Exception as e:
            logger.error(f"Failed to start optimization: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger")

    @app.callback(
        Output('benchmark-results', 'children'),
        Input('benchmark-btn', 'n_clicks'),
        [
            State('deployment-experiment-select', 'value'),
            State('benchmark-runs', 'value'),
            State('benchmark-options', 'value')
        ],
        prevent_initial_call=True
    )
    def run_benchmark(n_clicks, experiment_id, num_runs, model_types):
        """Trigger model benchmarking."""
        if not n_clicks or not experiment_id:
            raise PreventUpdate

        try:
            logger.info(f"Launching benchmark task for experiment {experiment_id}")

            # Launch Celery task
            task = benchmark_models_task.delay(experiment_id, num_runs, model_types or ['original'])

            return dbc.Alert(
                [
                    html.H5("Benchmarking Started", className="alert-heading"),
                    html.P(f"Task ID: {task.id}"),
                    html.P(f"Running {num_runs} benchmark iterations..."),
                    dbc.Spinner(size="sm")
                ],
                color="info"
            )

        except Exception as e:
            logger.error(f"Failed to start benchmark: {e}", exc_info=True)
            return dbc.Alert(f"Error: {str(e)}", color="danger")
