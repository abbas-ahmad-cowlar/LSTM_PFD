"""
Dataset Management callbacks.
Handle dataset CRUD operations and visualization.
"""
from dash import Input, Output, State, html, dash_table, ALL, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from services.dataset_service import DatasetService
from utils.logger import setup_logger

logger = setup_logger(__name__)


def register_datasets_callbacks(app):
    """Register dataset management callbacks."""

    @app.callback(
        [Output('datasets-table-container', 'children'),
         Output('datasets-data', 'data')],
        [Input('datasets-auto-refresh', 'n_intervals'),
         Input('datasets-refresh-btn', 'n_clicks'),
         Input('dataset-search', 'value')],
        prevent_initial_call=False
    )
    def load_datasets(n_intervals, n_clicks, search_query):
        """Load and display datasets table."""
        try:
            # Get datasets from service
            datasets = DatasetService.list_datasets(limit=100, search_query=search_query)

            if not datasets:
                return html.Div([
                    dbc.Alert([
                        html.I(className="fas fa-info-circle me-2"),
                        "No datasets found. ",
                        html.A("Create one now", href="/data-generation", className="alert-link")
                    ], color="info")
                ], className="text-center"), []

            # Create DataTable
            table = create_datasets_table(datasets)

            return table, datasets

        except Exception as e:
            logger.error(f"Failed to load datasets: {e}", exc_info=True)
            return dbc.Alert(f"Error loading datasets: {str(e)}", color="danger"), []

    @app.callback(
        [Output('dataset-details-modal', 'is_open'),
         Output('dataset-details-title', 'children'),
         Output('dataset-basic-info', 'children'),
         Output('dataset-stats-cards', 'children'),
         Output('dataset-class-distribution', 'figure'),
         Output('dataset-signal-preview', 'figure'),
         Output('selected-dataset-id', 'data')],
        [Input({'type': 'view-dataset-btn', 'index': ALL}, 'n_clicks'),
         Input('close-dataset-modal', 'n_clicks')],
        [State('datasets-data', 'data'),
         State('dataset-details-modal', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_dataset_details(view_clicks, close_click, datasets_data, is_open):
        """Show/hide dataset details modal."""
        if not ctx.triggered:
            raise PreventUpdate

        triggered_id = ctx.triggered[0]['prop_id']

        # Close modal
        if 'close-dataset-modal' in triggered_id:
            return False, "", html.Div(), html.Div(), go.Figure(), go.Figure(), None

        # Open modal - find which dataset was clicked
        if 'view-dataset-btn' in triggered_id:
            # Extract index from triggered_id
            import json
            triggered_dict = json.loads(triggered_id.split('.')[0])
            dataset_id = triggered_dict['index']

            # Get dataset details
            details = DatasetService.get_dataset_details(dataset_id)
            if not details:
                return False, "", html.Div(), html.Div(), go.Figure(), go.Figure(), None

            # Get statistics
            stats = DatasetService.get_dataset_statistics(dataset_id)

            # Get preview
            preview = DatasetService.get_dataset_preview(dataset_id, num_samples=3)

            # Create components
            title = f"Dataset: {details['name']}"
            basic_info = create_basic_info(details)
            stats_cards = create_stats_cards(stats)
            class_dist_fig = create_class_distribution_chart(stats)
            preview_fig = create_signal_preview_chart(preview, stats.get('fault_types', []))

            return True, title, basic_info, stats_cards, class_dist_fig, preview_fig, dataset_id

        raise PreventUpdate

    @app.callback(
        [Output('export-modal', 'is_open'),
         Output('export-status', 'children')],
        [Input('export-dataset-btn', 'n_clicks'),
         Input('cancel-export', 'n_clicks'),
         Input('confirm-export', 'n_clicks')],
        [State('export-modal', 'is_open'),
         State('selected-dataset-id', 'data'),
         State('export-format', 'value')],
        prevent_initial_call=True
    )
    def handle_export(export_click, cancel_click, confirm_click, is_open, dataset_id, format):
        """Handle dataset export."""
        triggered_id = ctx.triggered[0]['prop_id']

        # Open modal
        if 'export-dataset-btn' in triggered_id:
            return True, ""

        # Cancel
        if 'cancel-export' in triggered_id:
            return False, ""

        # Confirm export
        if 'confirm-export' in triggered_id:
            if not dataset_id:
                return True, dbc.Alert("No dataset selected", color="danger")

            # Export dataset
            export_path = DatasetService.export_dataset(dataset_id, format=format)

            if export_path:
                return False, dbc.Alert([
                    html.I(className="fas fa-check-circle me-2"),
                    f"Dataset exported successfully to: {export_path}"
                ], color="success")
            else:
                return True, dbc.Alert("Export failed", color="danger")

        raise PreventUpdate

    @app.callback(
        [Output('delete-modal', 'is_open'),
         Output('datasets-table-container', 'children', allow_duplicate=True),
         Output('dataset-details-modal', 'is_open', allow_duplicate=True)],
        [Input('delete-dataset-btn', 'n_clicks'),
         Input('cancel-delete', 'n_clicks'),
         Input('confirm-delete', 'n_clicks')],
        [State('delete-modal', 'is_open'),
         State('selected-dataset-id', 'data'),
         State('delete-file-checkbox', 'value')],
        prevent_initial_call=True
    )
    def handle_delete(delete_click, cancel_click, confirm_click, is_open, dataset_id, delete_file_option):
        """Handle dataset deletion."""
        triggered_id = ctx.triggered[0]['prop_id']

        # Open delete confirmation modal
        if 'delete-dataset-btn' in triggered_id:
            return True, dash_table.DataTable(), True

        # Cancel deletion
        if 'cancel-delete' in triggered_id:
            return False, dash_table.DataTable(), True

        # Confirm deletion
        if 'confirm-delete' in triggered_id:
            if not dataset_id:
                return False, dash_table.DataTable(), True

            delete_file = 'delete_file' in delete_file_option

            # Delete dataset
            success = DatasetService.delete_dataset(dataset_id, delete_file=delete_file)

            if success:
                # Reload datasets
                datasets = DatasetService.list_datasets(limit=100)
                table = create_datasets_table(datasets)
                return False, table, False
            else:
                return False, dash_table.DataTable(), True

        raise PreventUpdate

    @app.callback(
        Output('datasets-table-container', 'children', allow_duplicate=True),
        Input('archive-dataset-btn', 'n_clicks'),
        State('selected-dataset-id', 'data'),
        prevent_initial_call=True
    )
    def archive_dataset(n_clicks, dataset_id):
        """Archive dataset."""
        if not n_clicks or not dataset_id:
            raise PreventUpdate

        # Archive dataset
        success = DatasetService.archive_dataset(dataset_id)

        if success:
            # Reload datasets
            datasets = DatasetService.list_datasets(limit=100)
            return create_datasets_table(datasets)

        raise PreventUpdate


def create_datasets_table(datasets: list) -> dash_table.DataTable:
    """Create datasets DataTable."""
    if not datasets:
        return html.Div()

    # Format data for table
    data = []
    for ds in datasets:
        fault_types_str = ', '.join(ds['fault_types'][:3])
        if len(ds['fault_types']) > 3:
            fault_types_str += '...'

        data.append({
            'id': ds['id'],
            'name': ds['name'],
            'num_signals': ds['num_signals'],
            'fault_types': fault_types_str,
            'file_size_mb': f"{ds['file_size_mb']:.1f} MB",
            'created_at': ds['created_at'].strftime('%Y-%m-%d %H:%M'),
        })

    # Create table with action buttons
    table_rows = []
    for row in data:
        table_rows.append(
            html.Tr([
                html.Td(row['name']),
                html.Td(row['num_signals'], className="text-center"),
                html.Td(row['fault_types']),
                html.Td(row['file_size_mb'], className="text-end"),
                html.Td(row['created_at'], className="text-center"),
                html.Td([
                    dbc.Button(
                        [html.I(className="fas fa-eye")],
                        id={'type': 'view-dataset-btn', 'index': row['id']},
                        color="primary",
                        size="sm",
                        outline=True
                    )
                ], className="text-center")
            ])
        )

    return dbc.Table([
        html.Thead([
            html.Tr([
                html.Th("Name"),
                html.Th("# Signals", className="text-center"),
                html.Th("Fault Types"),
                html.Th("Size", className="text-end"),
                html.Th("Created", className="text-center"),
                html.Th("Actions", className="text-center")
            ])
        ]),
        html.Tbody(table_rows)
    ], bordered=True, hover=True, striped=True, responsive=True)


def create_basic_info(details: dict) -> html.Div:
    """Create basic dataset information display."""
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Strong("Description:"),
                    html.P(details.get('description', 'No description'))
                ], width=6),
                dbc.Col([
                    html.Strong("File Path:"),
                    html.P(details.get('file_path', 'N/A'), style={'fontSize': '0.9em', 'fontFamily': 'monospace'})
                ], width=6),
            ])
        ])
    ])


def create_stats_cards(stats: dict) -> html.Div:
    """Create statistics cards."""
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Total Signals", className="text-muted mb-2"),
                    html.H3(f"{stats.get('total_signals', 0):,}")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("File Size", className="text-muted mb-2"),
                    html.H3(f"{stats.get('file_size_mb', 0):.1f} MB")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Signal Length", className="text-muted mb-2"),
                    html.H3(f"{stats.get('signal_length', 0):,}")
                ])
            ])
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("Sampling Rate", className="text-muted mb-2"),
                    html.H3(f"{stats.get('sampling_rate', 0):,} Hz")
                ])
            ])
        ], width=3),
    ])


def create_class_distribution_chart(stats: dict) -> go.Figure:
    """Create class distribution pie chart."""
    class_dist = stats.get('class_distribution', {})

    if not class_dist:
        fig = go.Figure()
        fig.add_annotation(
            text="No class distribution data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    labels = [f"Class {k}" for k in class_dist.keys()]
    values = list(class_dist.values())

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.3,
        textinfo='label+percent',
        textposition='auto'
    )])

    fig.update_layout(
        title="Fault Type Distribution",
        height=400,
        showlegend=True
    )

    return fig


def create_signal_preview_chart(preview_data: dict, fault_types: list) -> go.Figure:
    """Create signal preview plot."""
    if not preview_data:
        fig = go.Figure()
        fig.add_annotation(
            text="No signal preview available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig

    fig = go.Figure()

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for idx, (fault_class, signals) in enumerate(preview_data.items()):
        fault_name = fault_types[fault_class] if fault_class < len(fault_types) else f"Class {fault_class}"
        color = colors[idx % len(colors)]

        for i, signal in enumerate(signals[:3]):  # Show first 3 signals per fault
            fig.add_trace(go.Scatter(
                y=signal,
                mode='lines',
                name=f'{fault_name} (sample {i+1})',
                line=dict(width=1, color=color),
                opacity=0.7 if i > 0 else 1.0
            ))

    fig.update_layout(
        title="Signal Preview (First 3 Samples per Fault Type)",
        xaxis_title="Sample Index",
        yaxis_title="Amplitude",
        height=400,
        hovermode='closest',
        showlegend=True
    )

    return fig
