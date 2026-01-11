"""
MAT File Import callbacks.
Handles file upload, validation, and import job management.
"""
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
import json

from database.connection import get_db_session
from models.dataset_import import DatasetImport, DatasetImportStatus
from tasks.mat_import_tasks import import_mat_dataset_task
from utils.logger import setup_logger
from utils.file_handler import parse_upload_contents, save_uploaded_mat_files, get_file_summary
from utils.auth_utils import get_current_user_id
from utils.constants import (
    BYTES_PER_MB,
    FILES_PER_MINUTE_IMPORT,
    SIGNAL_LENGTH,
    DEFAULT_RECENT_ITEMS_LIMIT,
    PERCENT_MULTIPLIER,
)

logger = setup_logger(__name__)


def register_mat_import_callbacks(app):
    """Register all MAT file import callbacks."""

    @app.callback(
        [
            Output('uploaded-files-list', 'children'),
            Output('uploaded-files-store', 'data'),
            Output('start-import-btn', 'disabled'),
        ],
        Input('upload-mat-files', 'contents'),
        [
            State('upload-mat-files', 'filename'),
            State('uploaded-files-store', 'data'),
        ],
        prevent_initial_call=True
    )
    def handle_file_upload(contents, filenames, existing_files):
        """Handle MAT file upload and update file list."""
        if not contents:
            raise PreventUpdate

        # Parse uploaded files
        new_files = parse_upload_contents(contents, filenames)

        # Merge with existing files (avoid duplicates)
        all_files = existing_files or []
        existing_names = {f['filename'] for f in all_files}

        for new_file in new_files:
            if new_file['filename'] not in existing_names:
                all_files.append(new_file)

        # Create file list display
        if not all_files:
            return html.P("No files uploaded yet", className="text-muted"), [], True

        file_items = []
        for i, file_info in enumerate(all_files):
            size_mb = file_info['size'] / BYTES_PER_MB
            file_items.append(
                html.Div([
                    html.I(className="fas fa-file me-2"),
                    html.Strong(file_info['filename']),
                    html.Small(f" ({size_mb:.2f} MB)", className="text-muted ms-2")
                ], className="mb-2")
            )

        # Enable import button if files are uploaded
        return file_items, all_files, False

    @app.callback(
        Output('import-summary', 'children'),
        Input('uploaded-files-store', 'data'),
        [
            State('import-dataset-name-input', 'value'),
            State('signal-length-input', 'value'),
        ]
    )
    def update_import_summary(uploaded_files, dataset_name, signal_length):
        """Update import summary based on uploaded files."""
        if not uploaded_files:
            return html.P("Upload files to see summary", className="text-muted")

        num_files = len(uploaded_files)
        total_size_mb = sum(f['size'] for f in uploaded_files) / BYTES_PER_MB

        return [
            html.Div([
                html.Strong("Dataset Name: "),
                html.Span(dataset_name or "Not set", className="text-primary" if dataset_name else "text-danger")
            ], className="mb-2"),

            html.Div([
                html.Strong("Total Files: "),
                html.Span(f"{num_files}", className="text-success")
            ], className="mb-2"),

            html.Div([
                html.Strong("Total Size: "),
                html.Span(f"{total_size_mb:.2f} MB", className="text-info")
            ], className="mb-2"),

            html.Div([
                html.Strong("Signal Length: "),
                html.Span(f"{signal_length:,} samples" if signal_length else "Not set")
            ], className="mb-2"),

            html.Hr(),

            html.Div([
                html.I(className="fas fa-info-circle me-2"),
                html.Small(
                    f"Estimated processing time: {max(1, int(num_files / FILES_PER_MINUTE_IMPORT))} minutes",
                    className="text-muted"
                )
            ])
        ]

    @app.callback(
        [
            Output('uploaded-files-list', 'children', allow_duplicate=True),
            Output('uploaded-files-store', 'data', allow_duplicate=True),
            Output('start-import-btn', 'disabled', allow_duplicate=True),
        ],
        Input('clear-upload-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def clear_uploaded_files(n_clicks):
        """Clear all uploaded files."""
        if not n_clicks:
            raise PreventUpdate

        return html.P("No files uploaded yet", className="text-muted"), [], True

    @app.callback(
        [
            Output('active-import-id', 'data'),
            Output('import-poll-interval', 'disabled'),
        ],
        Input('start-import-btn', 'n_clicks'),
        [
            State('import-dataset-name-input', 'value'),
            State('import-output-dir-input', 'value'),
            State('signal-length-input', 'value'),
            State('import-validate-check', 'value'),
            State('import-auto-normalize-check', 'value'),
            State('import-output-format-radio', 'value'),
            State('uploaded-files-store', 'data'),
        ],
        prevent_initial_call=True
    )
    def start_import(n_clicks, dataset_name, output_dir, signal_length, validate_check,
                    normalize_check, output_format, uploaded_files):
        """Launch MAT file import job."""
        if not n_clicks or not uploaded_files:
            raise PreventUpdate

        # Validate inputs
        if not dataset_name:
            logger.error("Dataset name is required")
            return None, True

        # Save uploaded files to disk
        try:
            temp_dir, file_paths = save_uploaded_mat_files(uploaded_files)
            logger.info(f"Saved {len(file_paths)} files to {temp_dir}")
        except Exception as e:
            logger.error(f"Failed to save uploaded files: {e}")
            return None, True

        # Build configuration
        config = {
            'name': dataset_name,
            'output_dir': output_dir or 'data/imported',
            'signal_length': signal_length or SIGNAL_LENGTH,
            'validate': 'enabled' in (validate_check or []),
            'auto_normalize': 'enabled' in (normalize_check or []),
            'output_format': output_format or 'hdf5',
        }

        try:
            # Create database record
            with get_db_session() as session:
                import_job = DatasetImport(
                    name=dataset_name,
                    config=config,
                    status=DatasetImportStatus.PENDING,
                    num_files=len(file_paths),
                )
                session.add(import_job)
                session.commit()
                import_id = import_job.id

            # Launch Celery task
            config['import_id'] = import_id
            config['user_id'] = get_current_user_id()
            task = import_mat_dataset_task.delay(config, file_paths)

            # Update database with task ID
            with get_db_session() as session:
                import_job = session.query(DatasetImport).filter_by(id=import_id).first()
                if import_job:
                    import_job.celery_task_id = task.id
                    import_job.status = DatasetImportStatus.RUNNING
                    session.commit()

            logger.info(f"Started MAT import task {task.id} for import {import_id}")

            # Enable polling, return import ID
            return import_id, False

        except Exception as e:
            logger.error(f"Failed to start import: {e}", exc_info=True)
            return None, True

    @app.callback(
        [
            Output('import-status', 'children'),
            Output('import-progress', 'value'),
            Output('import-progress', 'style'),
            Output('import-progress-divider', 'style'),
            Output('import-stats', 'children'),
            Output('import-stats', 'style'),
        ],
        Input('import-poll-interval', 'n_intervals'),
        State('active-import-id', 'data'),
        prevent_initial_call=True
    )
    def poll_import_status(n_intervals, import_id):
        """Poll import status and update progress."""
        if not import_id:
            raise PreventUpdate

        try:
            with get_db_session() as session:
                import_job = session.query(DatasetImport).filter_by(id=import_id).first()

                if not import_job:
                    return (
                        html.P("Import not found", className="text-danger"),
                        0,
                        {"display": "none"},
                        {"display": "none"},
                        [],
                        {"display": "none"}
                    )

                if import_job.status == DatasetImportStatus.RUNNING:
                    progress = import_job.progress or 0
                    return (
                        html.Div([
                            html.I(className="fas fa-spinner fa-spin me-2"),
                            html.Span(f"Importing... {progress}% complete")
                        ]),
                        progress,
                        {"display": "block"},
                        {"display": "block"},
                        _create_import_stats(import_job),
                        {"display": "block"}
                    )

                elif import_job.status == DatasetImportStatus.COMPLETED:
                    return (
                        html.Div([
                            html.I(className="fas fa-check-circle me-2 text-success"),
                            html.Span("Import completed successfully!", className="text-success")
                        ]),
                        PERCENT_MULTIPLIER,
                        {"display": "block"},
                        {"display": "block"},
                        _create_import_stats(import_job),
                        {"display": "block"}
                    )

                elif import_job.status == DatasetImportStatus.FAILED:
                    return (
                        html.Div([
                            html.I(className="fas fa-exclamation-triangle me-2 text-danger"),
                            html.Span("Import failed", className="text-danger")
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
            logger.error(f"Error polling import status: {e}")
            raise PreventUpdate

    @app.callback(
        Output('recent-imports-list', 'children'),
        Input('url', 'pathname')
    )
    def load_recent_imports(pathname):
        """Load list of recent imports."""
        if pathname != '/data-generation':
            raise PreventUpdate

        try:
            with get_db_session() as session:
                imports = session.query(DatasetImport)\
                    .order_by(DatasetImport.created_at.desc())\
                    .limit(DEFAULT_RECENT_ITEMS_LIMIT)\
                    .all()

                if not imports:
                    return html.P("No recent imports", className="text-muted")

                items = []
                for imp in imports:
                    status_icon = _get_status_icon(imp.status)
                    status_color = _get_status_color(imp.status)

                    items.append(
                        html.Div([
                            html.Div([
                                html.I(className=f"{status_icon} me-2 text-{status_color}"),
                                html.Strong(imp.name),
                            ]),
                            html.Small(
                                f"{imp.num_files} files • {imp.created_at.strftime('%Y-%m-%d %H:%M')}",
                                className="text-muted"
                            ),
                        ], className="mb-2 pb-2 border-bottom")
                    )

                return items

        except Exception as e:
            logger.error(f"Error loading recent imports: {e}")
            return html.P("Error loading imports", className="text-danger")


def _create_import_stats(import_job):
    """Create statistics display for import."""
    return [
        html.Div([
            html.Strong("Files: "),
            html.Span(f"{import_job.num_files or 0}")
        ], className="mb-1"),
        html.Div([
            html.Strong("Signals: "),
            html.Span(f"{import_job.num_signals or 0:,}")
        ], className="mb-1"),
        html.Div([
            html.Strong("Output: "),
            html.Span(import_job.output_path or "Processing...")
        ], className="mb-1"),
    ]


def _get_status_icon(status):
    """Get Font Awesome icon for status."""
    icons = {
        DatasetImportStatus.PENDING: "fas fa-clock",
        DatasetImportStatus.RUNNING: "fas fa-spinner fa-spin",
        DatasetImportStatus.COMPLETED: "fas fa-check-circle",
        DatasetImportStatus.FAILED: "fas fa-exclamation-triangle",
    }
    return icons.get(status, "fas fa-question-circle")


def _get_status_color(status):
    """Get Bootstrap color class for status."""
    colors = {
        DatasetImportStatus.PENDING: "secondary",
        DatasetImportStatus.RUNNING: "info",
        DatasetImportStatus.COMPLETED: "success",
        DatasetImportStatus.FAILED: "danger",
    }
    return colors.get(status, "dark")
