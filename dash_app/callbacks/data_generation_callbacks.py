"""
Data Generation callbacks - Phase 0 integration.
Handles user interactions for dataset generation and MAT file import.
"""
from dash import Input, Output, State, html, callback_context
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from datetime import datetime
import json

from database.connection import get_db_session
from models.dataset_generation import DatasetGeneration, DatasetGenerationStatus
from tasks.data_generation_tasks import generate_dataset_task
from utils.logger import setup_logger
from utils.auth_utils import get_current_user_id

logger = setup_logger(__name__)


def register_data_generation_callbacks(app):
    """Register all data generation and import callbacks."""

    # Register MAT import callbacks
    from callbacks.mat_import_callbacks import register_mat_import_callbacks
    register_mat_import_callbacks(app)

    @app.callback(
        Output('config-summary', 'children'),
        [
            Input('dataset-name-input', 'value'),
            Input('num-signals-slider', 'value'),
            Input('fault-types-checklist', 'value'),
            Input('severity-levels-checklist', 'value'),
            Input('noise-layers-checklist', 'value'),
            Input('augmentation-enabled-check', 'value'),
            Input('augmentation-ratio-slider', 'value'),
            Input('output-format-radio', 'value'),
        ]
    )
    def update_config_summary(dataset_name, num_signals, fault_types, severity_levels,
                             noise_layers, aug_enabled, aug_ratio, output_format):
        """Update configuration summary panel."""
        if not fault_types:
            fault_types = []
        if not severity_levels:
            severity_levels = []

        num_faults = len(fault_types)
        base_signals = num_faults * num_signals if num_signals else 0

        aug_enabled_bool = 'enabled' in (aug_enabled or [])
        if aug_enabled_bool and aug_ratio:
            total_signals = base_signals + int(base_signals * (aug_ratio / 100))
        else:
            total_signals = base_signals

        return [
            html.H5(dataset_name or "Untitled Dataset", className="mb-3"),

            html.Div([
                html.Strong("Total Signals: "),
                html.Span(f"{total_signals:,}", className="text-primary")
            ], className="mb-2"),

            html.Div([
                html.Strong("Fault Types: "),
                html.Span(f"{num_faults}", className="text-info")
            ], className="mb-2"),

            html.Div([
                html.Strong("Signals per Fault: "),
                html.Span(f"{num_signals}", className="text-success")
            ], className="mb-2"),

            html.Div([
                html.Strong("Severity Levels: "),
                html.Span(f"{len(severity_levels)}", className="text-warning")
            ], className="mb-2"),

            html.Hr(),

            html.Div([
                html.Strong("Noise Layers: "),
                html.Span(f"{len(noise_layers or [])}/7", className="text-secondary")
            ], className="mb-2"),

            html.Div([
                html.Strong("Augmentation: "),
                html.Span(
                    f"Enabled ({aug_ratio}% additional)" if aug_enabled_bool else "Disabled",
                    className="text-info"
                )
            ], className="mb-2"),

            html.Div([
                html.Strong("Output Format: "),
                html.Span(output_format.upper() if output_format else "N/A")
            ], className="mb-2"),

            html.Hr(),

            html.Div([
                html.I(className="fas fa-info-circle me-2"),
                html.Small(
                    f"Estimated generation time: {_estimate_generation_time(total_signals)} minutes",
                    className="text-muted"
                )
            ])
        ]

    @app.callback(
        [
            Output('active-generation-id', 'data'),
            Output('generation-poll-interval', 'disabled'),
            Output('start-generation-btn', 'disabled'),
        ],
        Input('start-generation-btn', 'n_clicks'),
        [
            State('dataset-name-input', 'value'),
            State('output-dir-input', 'value'),
            State('num-signals-slider', 'value'),
            State('fault-types-checklist', 'value'),
            State('severity-levels-checklist', 'value'),
            State('temporal-evolution-check', 'value'),
            State('noise-layers-checklist', 'value'),
            State('speed-variation-slider', 'value'),
            State('load-range-slider', 'value'),
            State('temp-range-slider', 'value'),
            State('augmentation-enabled-check', 'value'),
            State('augmentation-ratio-slider', 'value'),
            State('augmentation-methods-checklist', 'value'),
            State('output-format-radio', 'value'),
            State('random-seed-input', 'value'),
        ],
        prevent_initial_call=True
    )
    def start_generation(n_clicks, dataset_name, output_dir, num_signals, fault_types,
                        severity_levels, temporal_evolution, noise_layers, speed_variation,
                        load_range, temp_range, aug_enabled, aug_ratio, aug_methods,
                        output_format, random_seed):
        """Launch dataset generation task."""
        if not n_clicks:
            raise PreventUpdate

        # Validate inputs
        if not dataset_name:
            logger.error("Dataset name is required")
            return None, True, False

        if not fault_types:
            logger.error("At least one fault type must be selected")
            return None, True, False

        # Build configuration dictionary
        config = {
            'name': dataset_name,
            'output_dir': output_dir or 'data/generated',
            'num_signals_per_fault': num_signals or 100,
            'fault_types': fault_types,
            'severity_levels': severity_levels or ['incipient', 'mild', 'moderate', 'severe'],
            'temporal_evolution': 'enabled' in (temporal_evolution or []),
            'noise_layers': {
                'measurement': 'measurement' in (noise_layers or []),
                'emi': 'emi' in (noise_layers or []),
                'pink': 'pink' in (noise_layers or []),
                'drift': 'drift' in (noise_layers or []),
                'quantization': 'quantization' in (noise_layers or []),
                'sensor_drift': 'sensor_drift' in (noise_layers or []),
                'impulse': 'impulse' in (noise_layers or []),
            },
            'speed_variation': (speed_variation or 10) / 100,  # Convert to decimal
            'load_range': [(load_range[0] or 30) / 100, (load_range[1] or 100) / 100],
            'temp_range': temp_range or [40, 80],
            'augmentation': {
                'enabled': 'enabled' in (aug_enabled or []),
                'ratio': (aug_ratio or 30) / 100,
                'methods': aug_methods or ['time_shift', 'amplitude_scale', 'noise_injection']
            },
            'output_format': output_format or 'both',
            'random_seed': random_seed or 42,
        }

        try:
            # Create database record
            with get_db_session() as session:
                generation = DatasetGeneration(
                    name=dataset_name,
                    config=config,
                    status=DatasetGenerationStatus.PENDING,
                    num_signals=len(fault_types) * num_signals,
                    num_faults=len(fault_types),
                )
                session.add(generation)
                session.commit()
                generation_id = generation.id

            # Launch Celery task
            config['generation_id'] = generation_id
            config['user_id'] = get_current_user_id()
            task = generate_dataset_task.delay(config)

            # Update database with task ID
            with get_db_session() as session:
                generation = session.query(DatasetGeneration).filter_by(id=generation_id).first()
                if generation:
                    generation.celery_task_id = task.id
                    generation.status = DatasetGenerationStatus.RUNNING
                    session.commit()

            logger.info(f"Started dataset generation task {task.id} for generation {generation_id}")

            # Enable polling, return generation ID, disable button
            return generation_id, False, True

        except Exception as e:
            logger.error(f"Failed to start generation: {e}", exc_info=True)
            return None, True, False

    @app.callback(
        [
            Output('generation-status', 'children'),
            Output('generation-progress', 'value'),
            Output('generation-progress', 'style'),
            Output('progress-divider', 'style'),
            Output('generation-stats', 'children'),
            Output('generation-stats', 'style'),
        ],
        Input('generation-poll-interval', 'n_intervals'),
        State('active-generation-id', 'data'),
        prevent_initial_call=True
    )
    def poll_generation_status(n_intervals, generation_id):
        """Poll generation status and update progress."""
        if not generation_id:
            raise PreventUpdate

        try:
            with get_db_session() as session:
                generation = session.query(DatasetGeneration).filter_by(id=generation_id).first()

                if not generation:
                    return (
                        html.P("Generation not found", className="text-danger"),
                        0,
                        {"display": "none"},
                        {"display": "none"},
                        [],
                        {"display": "none"}
                    )

                if generation.status == DatasetGenerationStatus.RUNNING:
                    progress = generation.progress or 0
                    return (
                        html.Div([
                            html.I(className="fas fa-spinner fa-spin me-2"),
                            html.Span(f"Generating... {progress}% complete")
                        ]),
                        progress,
                        {"display": "block"},
                        {"display": "block"},
                        _create_generation_stats(generation),
                        {"display": "block"}
                    )

                elif generation.status == DatasetGenerationStatus.COMPLETED:
                    return (
                        html.Div([
                            html.I(className="fas fa-check-circle me-2 text-success"),
                            html.Span("Generation completed successfully!", className="text-success")
                        ]),
                        100,
                        {"display": "block"},
                        {"display": "block"},
                        _create_generation_stats(generation),
                        {"display": "block"}
                    )

                elif generation.status == DatasetGenerationStatus.FAILED:
                    return (
                        html.Div([
                            html.I(className="fas fa-exclamation-triangle me-2 text-danger"),
                            html.Span("Generation failed", className="text-danger")
                        ]),
                        0,
                        {"display": "none"},
                        {"display": "none"},
                        [],
                        {"display": "none"}
                    )

                else:  # PENDING
                    return (
                        html.P("Waiting to start...", className="text-muted"),
                        0,
                        {"display": "none"},
                        {"display": "none"},
                        [],
                        {"display": "none"}
                    )

        except Exception as e:
            logger.error(f"Error polling generation status: {e}")
            raise PreventUpdate

    @app.callback(
        Output('recent-generations-list', 'children'),
        Input('url', 'pathname')
    )
    def load_recent_generations(pathname):
        """Load list of recent generations."""
        if pathname != '/data-generation':
            raise PreventUpdate

        try:
            with get_db_session() as session:
                generations = session.query(DatasetGeneration)\
                    .order_by(DatasetGeneration.created_at.desc())\
                    .limit(5)\
                    .all()

                if not generations:
                    return html.P("No recent generations", className="text-muted")

                items = []
                for gen in generations:
                    status_icon = _get_status_icon(gen.status)
                    status_color = _get_status_color(gen.status)

                    items.append(
                        html.Div([
                            html.Div([
                                html.I(className=f"{status_icon} me-2 text-{status_color}"),
                                html.Strong(gen.name),
                            ]),
                            html.Small(
                                f"{gen.num_signals} signals • {gen.created_at.strftime('%Y-%m-%d %H:%M')}",
                                className="text-muted"
                            ),
                        ], className="mb-2 pb-2 border-bottom")
                    )

                return items

        except Exception as e:
            logger.error(f"Error loading recent generations: {e}")
            return html.P("Error loading generations", className="text-danger")

    @app.callback(
        Output('augmentation-settings', 'style'),
        Input('augmentation-enabled-check', 'value')
    )
    def toggle_augmentation_settings(aug_enabled):
        """Show/hide augmentation settings based on checkbox."""
        if 'enabled' in (aug_enabled or []):
            return {"display": "block"}
        return {"display": "none"}


def _estimate_generation_time(num_signals):
    """Estimate generation time based on number of signals."""
    # Rough estimate: 50 signals per minute
    return max(1, round(num_signals / 50))


def _create_generation_stats(generation):
    """Create statistics display for generation."""
    return [
        html.Div([
            html.Strong("Signals: "),
            html.Span(f"{generation.num_signals:,}")
        ], className="mb-1"),
        html.Div([
            html.Strong("Fault Types: "),
            html.Span(f"{generation.num_faults}")
        ], className="mb-1"),
        html.Div([
            html.Strong("Output: "),
            html.Span(generation.output_path or "Generating...")
        ], className="mb-1"),
    ]


def _get_status_icon(status):
    """Get Font Awesome icon for status."""
    icons = {
        DatasetGenerationStatus.PENDING: "fas fa-clock",
        DatasetGenerationStatus.RUNNING: "fas fa-spinner fa-spin",
        DatasetGenerationStatus.COMPLETED: "fas fa-check-circle",
        DatasetGenerationStatus.FAILED: "fas fa-exclamation-triangle",
    }
    return icons.get(status, "fas fa-question-circle")


def _get_status_color(status):
    """Get Bootstrap color class for status."""
    colors = {
        DatasetGenerationStatus.PENDING: "secondary",
        DatasetGenerationStatus.RUNNING: "info",
        DatasetGenerationStatus.COMPLETED: "success",
        DatasetGenerationStatus.FAILED: "danger",
    }
    return colors.get(status, "dark")
