"""
Data Generation callbacks - Phase 0 integration.
Handles user interactions for dataset generation and MAT file import.
"""
from dash import Input, Output, State, html, callback_context, dcc, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime
import json
import base64

from database.connection import get_db_session
from models.dataset_generation import DatasetGeneration, DatasetGenerationStatus
from tasks.data_generation_tasks import generate_dataset_task
from utils.logger import setup_logger
from utils.auth_utils import get_current_user_id
from utils.constants import (
    DEFAULT_NUM_SIGNALS_PER_FAULT,
    DEFAULT_SPEED_VARIATION_PERCENT,
    DEFAULT_LOAD_RANGE_MIN_PERCENT,
    DEFAULT_LOAD_RANGE_MAX_PERCENT,
    DEFAULT_TEMP_RANGE_MIN,
    DEFAULT_TEMP_RANGE_MAX,
    DEFAULT_AUGMENTATION_RATIO_PERCENT,
    DEFAULT_RANDOM_SEED,
    PERCENT_DIVISOR,
    PERCENT_MULTIPLIER,
    SIGNALS_PER_MINUTE_GENERATION,
    DEFAULT_RECENT_ITEMS_LIMIT,
    TOTAL_NOISE_LAYERS,
)

logger = setup_logger(__name__)


def register_data_generation_callbacks(app):
    """Register all data generation and import callbacks."""

    # Register MAT import callbacks
    from callbacks.mat_import_callbacks import register_mat_import_callbacks
    register_mat_import_callbacks(app)

    # =========================================================================
    # SAVE CONFIGURATION CALLBACK
    # =========================================================================
    @app.callback(
        Output('download-config', 'data'),
        Input('save-config-btn', 'n_clicks'),
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
    def save_configuration(n_clicks, dataset_name, output_dir, num_signals, fault_types,
                          severity_levels, temporal_evolution, noise_layers, speed_variation,
                          load_range, temp_range, aug_enabled, aug_ratio, aug_methods,
                          output_format, random_seed):
        """Save current configuration to a downloadable JSON file."""
        if not n_clicks:
            raise PreventUpdate

        config = {
            'version': '1.0',
            'saved_at': datetime.now().isoformat(),
            'dataset_name': dataset_name or '',
            'output_dir': output_dir or 'data/generated',
            'num_signals': num_signals or DEFAULT_NUM_SIGNALS_PER_FAULT,
            'fault_types': fault_types or [],
            'severity_levels': severity_levels or [],
            'temporal_evolution': temporal_evolution or [],
            'noise_layers': noise_layers or [],
            'speed_variation': speed_variation or DEFAULT_SPEED_VARIATION_PERCENT,
            'load_range': load_range or [DEFAULT_LOAD_RANGE_MIN_PERCENT, DEFAULT_LOAD_RANGE_MAX_PERCENT],
            'temp_range': temp_range or [DEFAULT_TEMP_RANGE_MIN, DEFAULT_TEMP_RANGE_MAX],
            'augmentation_enabled': aug_enabled or [],
            'augmentation_ratio': aug_ratio or DEFAULT_AUGMENTATION_RATIO_PERCENT,
            'augmentation_methods': aug_methods or [],
            'output_format': output_format or 'both',
            'random_seed': random_seed or DEFAULT_RANDOM_SEED,
        }

        filename = f"{dataset_name or 'config'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        logger.info(f"Saving configuration to {filename}")

        return dict(content=json.dumps(config, indent=2), filename=filename)

    # =========================================================================
    # LOAD CONFIGURATION CALLBACK - Toggle Modal
    # =========================================================================
    @app.callback(
        Output('load-config-modal', 'is_open'),
        [Input('load-config-btn', 'n_clicks'), Input('close-load-modal-btn', 'n_clicks')],
        State('load-config-modal', 'is_open'),
        prevent_initial_call=True
    )
    def toggle_load_config_modal(open_clicks, close_clicks, is_open):
        """Toggle the load configuration modal."""
        return not is_open

    # =========================================================================
    # LOAD CONFIGURATION CALLBACK - Apply Config
    # =========================================================================
    @app.callback(
        [
            Output('dataset-name-input', 'value'),
            Output('output-dir-input', 'value'),
            Output('num-signals-slider', 'value'),
            Output('fault-types-checklist', 'value'),
            Output('severity-levels-checklist', 'value'),
            Output('temporal-evolution-check', 'value'),
            Output('noise-layers-checklist', 'value'),
            Output('speed-variation-slider', 'value'),
            Output('load-range-slider', 'value'),
            Output('temp-range-slider', 'value'),
            Output('augmentation-enabled-check', 'value'),
            Output('augmentation-ratio-slider', 'value'),
            Output('augmentation-methods-checklist', 'value'),
            Output('output-format-radio', 'value'),
            Output('random-seed-input', 'value'),
            Output('load-config-modal', 'is_open', allow_duplicate=True),
            Output('load-config-error', 'children'),
        ],
        Input('upload-config-file', 'contents'),
        State('upload-config-file', 'filename'),
        prevent_initial_call=True
    )
    def load_configuration(contents, filename):
        """Load configuration from uploaded JSON file."""
        if not contents:
            raise PreventUpdate

        try:
            # Decode the uploaded file
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string).decode('utf-8')
            config = json.loads(decoded)

            logger.info(f"Loaded configuration from {filename}")

            return (
                config.get('dataset_name', ''),
                config.get('output_dir', 'data/generated'),
                config.get('num_signals', DEFAULT_NUM_SIGNALS_PER_FAULT),
                config.get('fault_types', []),
                config.get('severity_levels', []),
                config.get('temporal_evolution', []),
                config.get('noise_layers', []),
                config.get('speed_variation', DEFAULT_SPEED_VARIATION_PERCENT),
                config.get('load_range', [DEFAULT_LOAD_RANGE_MIN_PERCENT, DEFAULT_LOAD_RANGE_MAX_PERCENT]),
                config.get('temp_range', [DEFAULT_TEMP_RANGE_MIN, DEFAULT_TEMP_RANGE_MAX]),
                config.get('augmentation_enabled', []),
                config.get('augmentation_ratio', DEFAULT_AUGMENTATION_RATIO_PERCENT),
                config.get('augmentation_methods', []),
                config.get('output_format', 'both'),
                config.get('random_seed', DEFAULT_RANDOM_SEED),
                False,  # Close modal
                '',  # Clear error
            )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            return (
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update,
                True,  # Keep modal open
                html.Div([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    f"Invalid JSON file: {str(e)}"
                ], className="text-danger")
            )
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            return (
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
                no_update, no_update, no_update,
                True,  # Keep modal open
                html.Div([
                    html.I(className="fas fa-exclamation-triangle me-2"),
                    f"Error loading file: {str(e)}"
                ], className="text-danger")
            )

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
            total_signals = base_signals + int(base_signals * (aug_ratio / PERCENT_DIVISOR))
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
                html.Span(f"{len(noise_layers or [])}/{TOTAL_NOISE_LAYERS}", className="text-secondary")
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
            Output('start-generation-btn', 'children'),
            Output('generation-status', 'children', allow_duplicate=True),
            Output('generation-progress', 'value', allow_duplicate=True),
            Output('generation-progress', 'style', allow_duplicate=True),
            Output('progress-divider', 'style', allow_duplicate=True),
            Output('generation-stats', 'children', allow_duplicate=True),
            Output('generation-stats', 'style', allow_duplicate=True),
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
        """Launch dataset generation task with immediate visual feedback."""
        if not n_clicks:
            raise PreventUpdate

        # Default button content
        default_btn = [html.I(className="fas fa-play me-2"), "Generate Dataset"]
        generating_btn = [html.I(className="fas fa-spinner fa-spin me-2"), "Generating..."]
        
        # Validate inputs
        if not dataset_name:
            logger.error("Dataset name is required")
            return (
                None, True, False, default_btn,
                html.Div([
                    html.I(className="fas fa-exclamation-triangle me-2 text-warning"),
                    html.Span("Please enter a dataset name", className="text-warning")
                ]),
                0, {"display": "none"}, {"display": "none"}, [], {"display": "none"}
            )

        if not fault_types:
            logger.error("At least one fault type must be selected")
            return (
                None, True, False, default_btn,
                html.Div([
                    html.I(className="fas fa-exclamation-triangle me-2 text-warning"),
                    html.Span("Please select at least one fault type", className="text-warning")
                ]),
                0, {"display": "none"}, {"display": "none"}, [], {"display": "none"}
            )

        # Build configuration dictionary
        config = {
            'name': dataset_name,
            'output_dir': output_dir or 'data/generated',
            'num_signals_per_fault': num_signals or DEFAULT_NUM_SIGNALS_PER_FAULT,
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
            'speed_variation': (speed_variation or DEFAULT_SPEED_VARIATION_PERCENT) / PERCENT_DIVISOR,
            'load_range': [
                (load_range[0] or DEFAULT_LOAD_RANGE_MIN_PERCENT) / PERCENT_DIVISOR,
                (load_range[1] or DEFAULT_LOAD_RANGE_MAX_PERCENT) / PERCENT_DIVISOR
            ],
            'temp_range': temp_range or [DEFAULT_TEMP_RANGE_MIN, DEFAULT_TEMP_RANGE_MAX],
            'augmentation': {
                'enabled': 'enabled' in (aug_enabled or []),
                'ratio': (aug_ratio or DEFAULT_AUGMENTATION_RATIO_PERCENT) / PERCENT_DIVISOR,
                'methods': aug_methods or ['time_shift', 'amplitude_scale', 'noise_injection']
            },
            'output_format': output_format or 'both',
            'random_seed': random_seed or DEFAULT_RANDOM_SEED,
        }

        total_signals = len(fault_types) * num_signals

        try:
            # Create database record
            with get_db_session() as session:
                generation = DatasetGeneration(
                    name=dataset_name,
                    config=config,
                    status=DatasetGenerationStatus.PENDING,
                    num_signals=total_signals,
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

            # Immediate visual feedback
            status_content = html.Div([
                html.I(className="fas fa-spinner fa-spin me-2 text-info"),
                html.Span(f"Starting generation of '{dataset_name}'...", className="text-info"),
                html.Br(),
                html.Small(f"Task ID: {task.id[:8]}...", className="text-muted")
            ])
            
            initial_stats = [
                html.Div([
                    html.Strong("Dataset: "),
                    html.Span(dataset_name)
                ], className="mb-1"),
                html.Div([
                    html.Strong("Signals: "),
                    html.Span(f"{total_signals:,}")
                ], className="mb-1"),
                html.Div([
                    html.Strong("Fault Types: "),
                    html.Span(f"{len(fault_types)}")
                ], className="mb-1"),
            ]

            # Return with immediate feedback - button disabled, progress visible
            return (
                generation_id,       # active-generation-id
                False,               # generation-poll-interval disabled=False (enable polling)
                True,                # start-generation-btn disabled=True
                generating_btn,      # start-generation-btn children
                status_content,      # generation-status
                5,                   # generation-progress value (start at 5%)
                {"display": "block"},  # generation-progress style
                {"display": "block"},  # progress-divider style
                initial_stats,       # generation-stats
                {"display": "block"}   # generation-stats style
            )

        except Exception as e:
            logger.error(f"Failed to start generation: {e}", exc_info=True)
            return (
                None, True, False, default_btn,
                html.Div([
                    html.I(className="fas fa-exclamation-triangle me-2 text-danger"),
                    html.Span(f"Error: {str(e)}", className="text-danger")
                ]),
                0, {"display": "none"}, {"display": "none"}, [], {"display": "none"}
            )


    @app.callback(
        [
            Output('generation-status', 'children'),
            Output('generation-progress', 'value'),
            Output('generation-progress', 'style'),
            Output('progress-divider', 'style'),
            Output('generation-stats', 'children'),
            Output('generation-stats', 'style'),
            Output('start-generation-btn', 'disabled', allow_duplicate=True),
            Output('start-generation-btn', 'children', allow_duplicate=True),
            Output('generation-poll-interval', 'disabled', allow_duplicate=True),
            Output('recent-generations-list', 'children', allow_duplicate=True),
        ],
        Input('generation-poll-interval', 'n_intervals'),
        State('active-generation-id', 'data'),
        prevent_initial_call=True
    )
    def poll_generation_status(n_intervals, generation_id):
        """Poll generation status and update progress with full UI feedback."""
        if not generation_id:
            raise PreventUpdate

        default_btn = [html.I(className="fas fa-play me-2"), "Generate Dataset"]
        generating_btn = [html.I(className="fas fa-spinner fa-spin me-2"), "Generating..."]

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
                        {"display": "none"},
                        False,  # Re-enable button
                        default_btn,
                        True,  # Stop polling
                        no_update
                    )

                if generation.status == DatasetGenerationStatus.RUNNING:
                    progress = generation.progress or 0
                    return (
                        html.Div([
                            html.I(className="fas fa-spinner fa-spin me-2 text-info"),
                            html.Span(f"Generating... {progress}% complete", className="text-info")
                        ]),
                        progress,
                        {"display": "block"},
                        {"display": "block"},
                        _create_generation_stats(generation),
                        {"display": "block"},
                        True,  # Keep button disabled while running
                        generating_btn,
                        False,  # Keep polling
                        no_update  # Don't refresh recent list yet
                    )

                elif generation.status == DatasetGenerationStatus.COMPLETED:
                    # Fetch updated recent generations list
                    recent_list = _get_recent_generations_list(session)
                    return (
                        html.Div([
                            html.I(className="fas fa-check-circle me-2 text-success"),
                            html.Span("Generation completed successfully!", className="text-success"),
                            html.Br(),
                            html.Small(f"Output: {generation.output_path or 'data/generated'}", className="text-muted")
                        ]),
                        PERCENT_MULTIPLIER,
                        {"display": "block"},
                        {"display": "block"},
                        _create_generation_stats(generation),
                        {"display": "block"},
                        False,  # Re-enable button
                        default_btn,
                        True,  # Stop polling
                        recent_list  # Update recent list
                    )

                elif generation.status == DatasetGenerationStatus.FAILED:
                    # Fetch updated recent generations list
                    recent_list = _get_recent_generations_list(session)
                    error_msg = generation.error_message if hasattr(generation, 'error_message') and generation.error_message else "Unknown error"
                    return (
                        html.Div([
                            html.I(className="fas fa-exclamation-triangle me-2 text-danger"),
                            html.Span("Generation failed", className="text-danger"),
                            html.Br(),
                            html.Small(error_msg, className="text-muted")
                        ]),
                        0,
                        {"display": "none"},
                        {"display": "none"},
                        [],
                        {"display": "none"},
                        False,  # Re-enable button
                        default_btn,
                        True,  # Stop polling
                        recent_list  # Update recent list
                    )

                else:  # PENDING
                    return (
                        html.Div([
                            html.I(className="fas fa-clock me-2 text-secondary"),
                            html.Span("Waiting to start...", className="text-muted")
                        ]),
                        0,
                        {"display": "none"},
                        {"display": "none"},
                        [],
                        {"display": "none"},
                        True,  # Keep button disabled
                        generating_btn,
                        False,  # Keep polling
                        no_update
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
                    .limit(DEFAULT_RECENT_ITEMS_LIMIT)\
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
    return max(1, round(num_signals / SIGNALS_PER_MINUTE_GENERATION))


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


def _get_recent_generations_list(session):
    """Get recent generations list for UI display (called within session)."""
    try:
        generations = session.query(DatasetGeneration)\
            .order_by(DatasetGeneration.created_at.desc())\
            .limit(DEFAULT_RECENT_ITEMS_LIMIT)\
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
        logger.error(f"Error fetching recent generations: {e}")
        return html.P("Error loading generations", className="text-danger")


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
