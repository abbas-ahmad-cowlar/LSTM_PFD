"""
XAI Dashboard Callbacks.
Connects XAI dashboard UI to backend explanation services.
"""
import numpy as np
from dash import Input, Output, State, html, dcc, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from typing import Optional, List, Dict, Any

from database.connection import get_db_session
from models.experiment import Experiment, ExperimentStatus
from models.signal import Signal
from models.dataset import Dataset
from models.explanation import Explanation
from services.xai_service import XAIService
from utils.signal_loader import SignalLoader
from services.explanation_cache import ExplanationCache
from utils.xai_visualization import XAIVisualization
from tasks.xai_tasks import generate_explanation_task
from utils.logger import setup_logger
from utils.constants import NUM_CLASSES

logger = setup_logger(__name__)


def register_xai_callbacks(app):
    """Register all XAI dashboard callbacks."""

    @app.callback(
        Output('xai-model-dropdown', 'options'),
        Input('url', 'pathname')
    )
    def load_available_models(pathname):
        """
        Load list of completed experiments with trained models.

        Args:
            pathname: Current URL path

        Returns:
            List of dropdown options
        """
        if pathname != '/xai':
            raise PreventUpdate

        try:
            with get_db_session() as session:
                experiments = session.query(Experiment)\
                    .filter(Experiment.status == ExperimentStatus.COMPLETED)\
                    .filter(Experiment.model_path.isnot(None))\
                    .order_by(Experiment.created_at.desc())\
                    .limit(50)\
                    .all()

                options = []
                for exp in experiments:
                    # Get accuracy from metrics
                    test_acc = exp.metrics.get('test_accuracy', 0) if exp.metrics else 0

                    label = f"{exp.name} - {exp.model_type}"
                    if test_acc > 0:
                        label += f" ({test_acc*100:.1f}% acc)"

                    options.append({
                        'label': label,
                        'value': exp.id
                    })

                logger.info(f"Loaded {len(options)} available models")
                return options

        except Exception as e:
            logger.error(f"Failed to load models: {e}", exc_info=True)
            return []

    @app.callback(
        Output('xai-signal-dropdown', 'options'),
        Input('xai-model-dropdown', 'value')
    )
    def load_signals_for_model(experiment_id):
        """
        Load test signals from experiment's dataset.

        Args:
            experiment_id: Selected experiment ID

        Returns:
            List of signal dropdown options
        """
        if not experiment_id:
            raise PreventUpdate

        try:
            with get_db_session() as session:
                experiment = session.query(Experiment).filter_by(id=experiment_id).first()

                if not experiment or not experiment.dataset_id:
                    return []

                # Load signals from dataset
                signals = session.query(Signal)\
                    .filter_by(dataset_id=experiment.dataset_id)\
                    .limit(100)\
                    .all()

                options = []
                for sig in signals:
                    label = f"{sig.signal_id} - {sig.fault_class}"
                    if sig.severity:
                        label += f" ({sig.severity})"

                    options.append({
                        'label': label,
                        'value': sig.id
                    })

                logger.info(f"Loaded {len(options)} signals for experiment {experiment_id}")
                return options

        except Exception as e:
            logger.error(f"Failed to load signals: {e}", exc_info=True)
            return []

    @app.callback(
        [
            Output('xai-prediction-display', 'children'),
            Output('xai-results-store', 'data'),
        ],
        Input('generate-xai-btn', 'n_clicks'),
        [
            State('xai-model-dropdown', 'value'),
            State('xai-signal-dropdown', 'value'),
            State('xai-method-dropdown', 'value'),
            State('xai-num-features', 'value'),
            State('xai-background-samples', 'value'),
            State('xai-perturbations', 'value'),
        ],
        prevent_initial_call=True
    )
    def generate_explanation(n_clicks, experiment_id, signal_id, method,
                           num_features, bg_samples, perturbations):
        """
        Generate XAI explanation (check cache first, then launch async task).

        Args:
            n_clicks: Button click count
            experiment_id: Selected experiment ID
            signal_id: Selected signal ID
            method: XAI method ('shap', 'lime', 'integrated_gradients', 'gradcam')
            num_features: Number of features to show
            bg_samples: Number of background samples (SHAP)
            perturbations: Number of perturbations (LIME)

        Returns:
            Tuple of (prediction display, results store data)
        """
        if not n_clicks or not experiment_id or not signal_id:
            raise PreventUpdate

        try:
            # Check cache first
            cached = ExplanationCache.get_explanation(experiment_id, signal_id, method)

            if cached and cached.get('success'):
                logger.info(f"Using cached {method} explanation")

                # Create prediction display from cached data
                prediction_display = _create_prediction_display(cached)

                # Return cached data with ready status
                cached['status'] = 'ready'
                cached['from_cache'] = True

                return prediction_display, cached

            # Not in cache - launch async task
            logger.info(f"Launching async {method} explanation task")

            # Build task configuration
            config = {
                'experiment_id': experiment_id,
                'signal_id': signal_id,
                'method': method,
                'params': {}
            }

            # Add method-specific parameters
            if method == 'shap':
                config['params']['shap_method'] = 'gradient'  # Default to gradient
                config['params']['num_samples'] = bg_samples or 100
            elif method == 'lime':
                config['params']['num_segments'] = 20  # Fixed for now
                config['params']['num_perturbations'] = perturbations or 1000
            elif method == 'integrated_gradients':
                config['params']['ig_steps'] = 50

            # Launch Celery task
            task = generate_explanation_task.delay(config)

            # Quick prediction while explanation generates
            model = XAIService.load_model(experiment_id)
            signal = SignalLoader.load_signal_by_id(signal_id)

            if model and signal:
                prediction_result = XAIService.get_model_prediction(model, signal)
                if prediction_result.get('success'):
                    prediction_display = _create_prediction_display(prediction_result)
                else:
                    prediction_display = html.P("Failed to get prediction", className="text-danger")
            else:
                prediction_display = html.P("Loading prediction...", className="text-muted")

            # Return task info for polling
            return prediction_display, {
                'task_id': task.id,
                'status': 'generating',
                'experiment_id': experiment_id,
                'signal_id': signal_id,
                'method': method
            }

        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}", exc_info=True)
            return (
                html.Div([
                    html.I(className="fas fa-exclamation-triangle text-danger me-2"),
                    html.Span(f"Error: {str(e)}", className="text-danger")
                ]),
                {'status': 'error', 'error': str(e)}
            )

    @app.callback(
        [
            Output('xai-signal-plot', 'figure'),
            Output('xai-importance-plot', 'figure'),
            Output('xai-explanation-details', 'children'),
        ],
        Input('xai-results-store', 'data'),
        prevent_initial_call=True
    )
    def update_visualizations(results):
        """
        Update visualization panels when explanation is ready.

        Args:
            results: Results data from store

        Returns:
            Tuple of (signal plot, importance plot, details panel)
        """
        if not results or results.get('status') != 'ready':
            # Return empty figures while generating
            empty_fig = {
                'data': [],
                'layout': {
                    'title': 'Generating explanation...',
                    'template': 'plotly_white',
                    'height': 400
                }
            }
            return empty_fig, empty_fig, html.P("Waiting for explanation...", className="text-muted")

        try:
            method = results.get('method')
            predicted_class = results.get('predicted_class')

            if method == 'shap':
                # SHAP visualizations
                signal_fig = XAIVisualization.create_shap_signal_plot(
                    signal=results['signal'],
                    shap_values=results['shap_values'],
                    time_labels=results.get('time_labels'),
                    predicted_class=predicted_class
                )

                importance_fig = XAIVisualization.create_shap_waterfall(
                    shap_values=results['shap_values'],
                    base_value=results.get('base_value', 0),
                    predicted_value=results.get('confidence', 0),
                    time_labels=results.get('time_labels'),
                    top_k=20
                )

                details = _create_shap_details(results)

            elif method == 'lime':
                # LIME visualizations
                signal_fig = XAIVisualization.create_lime_segment_plot(
                    signal=results['signal'],
                    segment_weights=results['segment_weights'],
                    segment_boundaries=results['segment_boundaries'],
                    predicted_class=predicted_class
                )

                importance_fig = XAIVisualization.create_lime_bar_chart(
                    segment_weights=results['segment_weights'],
                    top_k=15,
                    predicted_class=predicted_class
                )

                details = _create_lime_details(results)

            elif method == 'integrated_gradients' or method == 'gradcam':
                # Attribution-based visualizations
                signal_fig = XAIVisualization.create_attribution_plot(
                    signal=results['signal'],
                    attributions=results['attributions'],
                    method=method,
                    predicted_class=predicted_class
                )

                importance_fig = XAIVisualization.create_feature_importance_summary(
                    attributions=results['attributions'],
                    top_k=20
                )

                details = _create_attribution_details(results, method)

            else:
                raise ValueError(f"Unknown method: {method}")

            return signal_fig, importance_fig, details

        except Exception as e:
            logger.error(f"Failed to create visualizations: {e}", exc_info=True)
            error_fig = {
                'data': [],
                'layout': {
                    'title': f'Error creating visualization: {str(e)}',
                    'template': 'plotly_white'
                }
            }
            error_details = html.Div([
                html.I(className="fas fa-exclamation-triangle text-danger me-2"),
                html.Span(f"Visualization error: {str(e)}", className="text-danger")
            ])
            return error_fig, error_fig, error_details

    @app.callback(
        Output('cached-explanations-list', 'children'),
        Input('url', 'pathname')
    )
    def load_cached_explanations(pathname):
        """
        Display recently cached explanations.

        Args:
            pathname: Current URL path

        Returns:
            List of cached explanation items
        """
        if pathname != '/xai':
            raise PreventUpdate

        try:
            recent = ExplanationCache.get_recent_explanations(limit=10)

            if not recent:
                return html.P("No cached explanations", className="text-muted")

            items = []
            for exp in recent:
                # Format timestamp
                created_at = exp.get('created_at', 'Unknown')
                if created_at and 'T' in created_at:
                    created_at = created_at.split('T')[0]  # Just date

                # Method badge
                method_colors = {
                    'shap': 'primary',
                    'lime': 'success',
                    'integrated_gradients': 'info',
                    'gradcam': 'warning'
                }
                badge_color = method_colors.get(exp['method'], 'secondary')

                items.append(
                    dbc.ListGroupItem([
                        dbc.Row([
                            dbc.Col([
                                dbc.Badge(exp['method'].upper(), color=badge_color, className="me-2"),
                                html.Span(f"Experiment {exp['experiment_id']}, Signal {exp['signal_id']}",
                                         className="text-muted small")
                            ], width=9),
                            dbc.Col([
                                dbc.Button(
                                    "Load",
                                    id={'type': 'load-cached-btn', 'index': exp['id']},
                                    size="sm",
                                    color="primary",
                                    outline=True
                                )
                            ], width=3, className="text-end")
                        ]),
                        html.Small(created_at, className="text-muted")
                    ])
                )

            return items

        except Exception as e:
            logger.error(f"Failed to load cached explanations: {e}", exc_info=True)
            return html.P("Error loading cache", className="text-danger")


# Helper Functions

def _create_prediction_display(result: Dict[str, Any]) -> html.Div:
    """Create prediction display card."""
    predicted_class = result.get('predicted_class', 0)
    confidence = result.get('confidence', 0)
    probabilities = result.get('probabilities', [])

    # Fault class names
    fault_classes = XAIVisualization.FAULT_CLASSES
    if predicted_class < len(fault_classes):
        predicted_label = fault_classes[predicted_class]
    else:
        predicted_label = f"Class {predicted_class}"

    # Create probability bars for top 5 classes
    if probabilities:
        top_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:5]
        prob_bars = []

        for idx in top_indices:
            prob = probabilities[idx]
            label = fault_classes[idx] if idx < len(fault_classes) else f"Class {idx}"

            prob_bars.append(
                dbc.Row([
                    dbc.Col(html.Small(label), width=4),
                    dbc.Col(
                        dbc.Progress(
                            value=prob * 100,
                            label=f"{prob*100:.1f}%",
                            color="success" if idx == predicted_class else "secondary",
                            style={"height": "20px"}
                        ),
                        width=8
                    )
                ], className="mb-2")
            )
    else:
        prob_bars = []

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H6("Predicted Class", className="text-muted mb-1"),
                html.H4(predicted_label, className="text-primary mb-2"),
                dbc.Progress(
                    value=confidence * 100,
                    label=f"{confidence*100:.1f}%",
                    color="success" if confidence > 0.8 else "warning" if confidence > 0.5 else "danger",
                    style={"height": "25px"}
                )
            ], width=12)
        ], className="mb-4"),

        html.Hr(),

        html.H6("Top 5 Predictions", className="mb-3"),
        html.Div(prob_bars)
    ])


def _create_shap_details(result: Dict[str, Any]) -> html.Div:
    """Create SHAP explanation details panel."""
    shap_values = np.array(result.get('shap_values', []))
    base_value = result.get('base_value', 0)

    # Calculate statistics
    mean_abs_shap = np.abs(shap_values).mean() if len(shap_values) > 0 else 0
    max_pos_shap = shap_values.max() if len(shap_values) > 0 else 0
    max_neg_shap = shap_values.min() if len(shap_values) > 0 else 0

    return html.Div([
        html.H6("SHAP Explanation Details", className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.Small("Method:", className="text-muted"),
                html.P(result.get('shap_method', 'gradient').upper(), className="mb-2")
            ], width=6),
            dbc.Col([
                html.Small("Base Value:", className="text-muted"),
                html.P(f"{base_value:.4f}", className="mb-2")
            ], width=6),
        ]),

        dbc.Row([
            dbc.Col([
                html.Small("Mean |SHAP|:", className="text-muted"),
                html.P(f"{mean_abs_shap:.4f}", className="mb-2")
            ], width=6),
            dbc.Col([
                html.Small("Signal Length:", className="text-muted"),
                html.P(f"{result.get('signal_length', 0):,} samples", className="mb-2")
            ], width=6),
        ]),

        dbc.Row([
            dbc.Col([
                html.Small("Max Positive:", className="text-muted"),
                html.P(f"+{max_pos_shap:.4f}", className="text-success mb-2")
            ], width=6),
            dbc.Col([
                html.Small("Max Negative:", className="text-muted"),
                html.P(f"{max_neg_shap:.4f}", className="text-danger mb-2")
            ], width=6),
        ]),

        html.Hr(),

        html.Small([
            html.I(className="fas fa-info-circle me-2"),
            "SHAP values show the contribution of each time step to the prediction. ",
            html.Span("Green", className="text-success"),
            " indicates positive contribution, ",
            html.Span("red", className="text-danger"),
            " indicates negative contribution."
        ], className="text-muted")
    ])


def _create_lime_details(result: Dict[str, Any]) -> html.Div:
    """Create LIME explanation details panel."""
    segment_weights = np.array(result.get('segment_weights', []))
    num_segments = result.get('num_segments', 0)

    # Calculate statistics
    mean_abs_weight = np.abs(segment_weights).mean() if len(segment_weights) > 0 else 0
    max_pos_weight = segment_weights.max() if len(segment_weights) > 0 else 0
    max_neg_weight = segment_weights.min() if len(segment_weights) > 0 else 0

    return html.Div([
        html.H6("LIME Explanation Details", className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.Small("Number of Segments:", className="text-muted"),
                html.P(f"{num_segments}", className="mb-2")
            ], width=6),
            dbc.Col([
                html.Small("Signal Length:", className="text-muted"),
                html.P(f"{result.get('signal_length', 0):,} samples", className="mb-2")
            ], width=6),
        ]),

        dbc.Row([
            dbc.Col([
                html.Small("Mean |Weight|:", className="text-muted"),
                html.P(f"{mean_abs_weight:.4f}", className="mb-2")
            ], width=6),
            dbc.Col([
                html.Small("Samples/Segment:", className="text-muted"),
                html.P(f"~{result.get('signal_length', 0) // max(num_segments, 1)}", className="mb-2")
            ], width=6),
        ]),

        dbc.Row([
            dbc.Col([
                html.Small("Strongest Positive:", className="text-muted"),
                html.P(f"+{max_pos_weight:.4f}", className="text-success mb-2")
            ], width=6),
            dbc.Col([
                html.Small("Strongest Negative:", className="text-muted"),
                html.P(f"{max_neg_weight:.4f}", className="text-danger mb-2")
            ], width=6),
        ]),

        html.Hr(),

        html.Small([
            html.I(className="fas fa-info-circle me-2"),
            "LIME segments show which parts of the signal contribute most to the prediction. ",
            "Higher absolute weights indicate more important segments."
        ], className="text-muted")
    ])


def _create_attribution_details(result: Dict[str, Any], method: str) -> html.Div:
    """Create attribution-based explanation details."""
    attributions = np.array(result.get('attributions', []))

    # Calculate statistics
    mean_abs_attr = np.abs(attributions).mean() if len(attributions) > 0 else 0
    max_pos_attr = attributions.max() if len(attributions) > 0 else 0
    max_neg_attr = attributions.min() if len(attributions) > 0 else 0

    method_name = method.replace('_', ' ').title()

    return html.Div([
        html.H6(f"{method_name} Details", className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.Small("Method:", className="text-muted"),
                html.P(method_name, className="mb-2")
            ], width=6),
            dbc.Col([
                html.Small("Signal Length:", className="text-muted"),
                html.P(f"{result.get('signal_length', 0):,} samples", className="mb-2")
            ], width=6),
        ]),

        dbc.Row([
            dbc.Col([
                html.Small("Mean |Attribution|:", className="text-muted"),
                html.P(f"{mean_abs_attr:.4f}", className="mb-2")
            ], width=6),
            dbc.Col([
                html.Small("Steps (if IG):", className="text-muted"),
                html.P(f"{result.get('steps', 'N/A')}", className="mb-2")
            ], width=6),
        ]),

        dbc.Row([
            dbc.Col([
                html.Small("Max Positive:", className="text-muted"),
                html.P(f"+{max_pos_attr:.4f}", className="text-success mb-2")
            ], width=6),
            dbc.Col([
                html.Small("Max Negative:", className="text-muted"),
                html.P(f"{max_neg_attr:.4f}", className="text-danger mb-2")
            ], width=6),
        ]),

        html.Hr(),

        html.Small([
            html.I(className="fas fa-info-circle me-2"),
            f"{method_name} shows which input features are most important for the prediction."
        ], className="text-muted")
    ])
