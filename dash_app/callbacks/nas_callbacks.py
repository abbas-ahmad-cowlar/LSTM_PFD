"""
Neural Architecture Search (NAS) Callbacks (Phase 4, Feature 3/3).
Handles all callbacks for the NAS dashboard.
"""
from dash import callback_context, html, no_update, ALL
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import traceback
from datetime import datetime

from utils.logger import setup_logger
from database.connection import get_db_session
from models.dataset import Dataset
from services.nas_service import NASService
from tasks.nas_tasks import run_nas_campaign_task

logger = setup_logger(__name__)


def register_nas_callbacks(app):
    """Register all NAS dashboard callbacks."""

    # ==================== Data Loading ====================

    @app.callback(
        Output('nas-dataset-select', 'options'),
        Input('nas-campaigns-table', 'children')
    )
    def load_datasets(trigger):
        """Load available datasets for NAS."""
        try:
            with get_db_session() as session:
                datasets = session.query(Dataset).order_by(Dataset.created_at.desc()).limit(50).all()

                return [
                    {'label': f"{ds.name} ({ds.signal_count} signals)", 'value': ds.id}
                    for ds in datasets
                ]
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            return []

    # ==================== Campaign Creation ====================

    @app.callback(
        [Output('nas-creation-status', 'children'),
         Output('selected-nas-campaign-id', 'data')],
        Input('launch-nas-btn', 'n_clicks'),
        [State('nas-campaign-name', 'value'),
         State('nas-dataset-select', 'value'),
         State('nas-search-algorithm', 'value'),
         State('nas-num-trials', 'value'),
         State('nas-epochs-per-trial', 'value'),
         State('nas-operations', 'value'),
         State('nas-channel-sizes', 'value'),
         State('nas-min-layers', 'value'),
         State('nas-max-layers', 'value')]
    )
    def launch_nas_campaign(
        n_clicks, name, dataset_id, algorithm, num_trials, epochs_per_trial,
        operations, channel_sizes, min_layers, max_layers
    ):
        """Launch a new NAS campaign."""
        if not n_clicks:
            raise PreventUpdate

        try:
            # Validation
            if not name:
                return dbc.Alert("Please enter a campaign name", color="warning"), None
            if not dataset_id:
                return dbc.Alert("Please select a dataset", color="warning"), None
            if not operations or not channel_sizes:
                return dbc.Alert("Please select at least one operation and channel size", color="warning"), None

            # Create search space config
            search_space_config = {
                'operations': operations,
                'channel_sizes': channel_sizes,
                'min_layers': min_layers or 3,
                'max_layers': max_layers or 10
            }

            # Create campaign
            campaign_id = NASService.create_nas_campaign(
                name=name,
                dataset_id=dataset_id,
                search_space_config=search_space_config,
                search_algorithm=algorithm,
                num_trials=num_trials or 20,
                max_epochs_per_trial=epochs_per_trial or 10
            )

            # Launch background task
            task = run_nas_campaign_task.delay(campaign_id)
            logger.info(f"Launched NAS campaign {campaign_id}, task ID: {task.id}")

            status = dbc.Alert([
                html.I(className="bi bi-check-circle me-2"),
                html.Strong("NAS Campaign launched! "),
                f"Campaign ID: {campaign_id}. ",
                "The campaign is now running in the background. Results will appear in the campaigns table below."
            ], color="success")

            return status, campaign_id

        except Exception as e:
            logger.error(f"Failed to launch NAS campaign: {e}")
            logger.error(traceback.format_exc())
            return dbc.Alert(f"Error: {str(e)}", color="danger"), None

    # ==================== Campaign Listing ====================

    @app.callback(
        Output('nas-campaigns-table', 'children'),
        [Input('nas-auto-refresh', 'n_intervals'),
         Input('reload-nas-campaigns-btn', 'n_clicks'),
         Input('selected-nas-campaign-id', 'data')]
    )
    def load_nas_campaigns(n_intervals, n_clicks, trigger):
        """Load and display NAS campaigns table."""
        try:
            campaigns = NASService.list_campaigns(limit=50)

            if not campaigns:
                return dbc.Alert(
                    "No NAS campaigns yet. Create one above to get started!",
                    color="info"
                )

            # Build table rows
            rows = []
            for campaign in campaigns:
                # Status badge
                if campaign['status'] == 'completed':
                    status_badge = dbc.Badge("Completed", color="success")
                elif campaign['status'] == 'running':
                    status_badge = dbc.Badge("Running", color="primary")
                elif campaign['status'] == 'failed':
                    status_badge = dbc.Badge("Failed", color="danger")
                else:
                    status_badge = dbc.Badge("Pending", color="secondary")

                # Best accuracy
                best_acc = f"{campaign['best_accuracy']:.2%}" if campaign['best_accuracy'] else "N/A"

                # Created date
                created = campaign['created_at'][:10] if campaign['created_at'] else 'N/A'

                rows.append(
                    html.Tr([
                        html.Td(campaign['name']),
                        html.Td(campaign['search_algorithm'].title()),
                        html.Td(campaign['num_trials']),
                        html.Td(status_badge),
                        html.Td(best_acc),
                        html.Td(created),
                        html.Td(
                            dbc.Button(
                                "View",
                                id={'type': 'view-nas-campaign', 'index': campaign['id']},
                                color="primary",
                                size="sm"
                            )
                        ),
                    ])
                )

            # Create table
            table = dbc.Table([
                html.Thead(
                    html.Tr([
                        html.Th("Campaign Name"),
                        html.Th("Algorithm"),
                        html.Th("Trials"),
                        html.Th("Status"),
                        html.Th("Best Accuracy"),
                        html.Th("Created"),
                        html.Th("Actions"),
                    ])
                ),
                html.Tbody(rows)
            ], striped=True, hover=True, responsive=True)

            return table

        except Exception as e:
            logger.error(f"Failed to load campaigns: {e}")
            logger.error(traceback.format_exc())
            return dbc.Alert(f"Error loading campaigns: {str(e)}", color="danger")

    # ==================== Campaign Details Modal ====================

    @app.callback(
        [Output('nas-campaign-modal', 'is_open'),
         Output('nas-campaign-modal-title', 'children'),
         Output('nas-campaign-info', 'children'),
         Output('nas-trials-table', 'children'),
         Output('nas-best-architecture', 'children'),
         Output('nas-architecture-graph', 'figure'),
         Output('selected-nas-campaign-id', 'data', allow_duplicate=True)],
        [Input({'type': 'view-nas-campaign', 'index': ALL}, 'n_clicks'),
         Input('close-nas-modal-btn', 'n_clicks')],
        [State('nas-campaign-modal', 'is_open')],
        prevent_initial_call=True
    )
    def toggle_campaign_modal(view_clicks, close_click, is_open):
        """Open/close campaign details modal."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]['prop_id']

        # Close modal
        if 'close-nas-modal-btn' in trigger_id:
            return False, "", "", "", "", go.Figure(), None

        # Open modal
        if 'view-nas-campaign' in trigger_id:
            # Extract campaign ID from trigger
            import json
            trigger_dict = json.loads(trigger_id.split('.')[0])
            campaign_id = trigger_dict['index']

            # Load campaign details
            campaign = NASService.get_campaign_details(campaign_id)
            if not campaign:
                return no_update, no_update, no_update, no_update, no_update, no_update, None

            # Campaign info
            info = dbc.Row([
                dbc.Col([
                    html.Strong("Dataset ID: "), campaign['dataset_id']
                ], width=3),
                dbc.Col([
                    html.Strong("Algorithm: "), campaign['search_algorithm'].title()
                ], width=3),
                dbc.Col([
                    html.Strong("Status: "), campaign['status'].title()
                ], width=3),
                dbc.Col([
                    html.Strong("Trials: "), f"{len(campaign['trials'])}/{campaign['num_trials']}"
                ], width=3),
            ])

            # Trials table
            if not campaign['trials']:
                trials_table = dbc.Alert("No trials completed yet", color="info")
            else:
                trial_rows = []
                for trial in campaign['trials']:
                    trial_rows.append(
                        html.Tr([
                            html.Td(trial['trial_number']),
                            html.Td(f"{trial['validation_accuracy']:.2%}" if trial['validation_accuracy'] else "N/A"),
                            html.Td(f"{trial['num_parameters']:,}" if trial['num_parameters'] else "N/A"),
                            html.Td(f"{trial['flops']:,}" if trial['flops'] else "N/A"),
                            html.Td(f"{trial['training_time']:.1f}s" if trial['training_time'] else "N/A"),
                        ])
                    )

                trials_table = dbc.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("Trial"),
                            html.Th("Accuracy"),
                            html.Th("Parameters"),
                            html.Th("FLOPs"),
                            html.Th("Time"),
                        ])
                    ),
                    html.Tbody(trial_rows)
                ], striped=True, hover=True, size='sm', style={'maxHeight': '300px', 'overflowY': 'auto'})

            # Best architecture
            best_trial = NASService.get_best_architecture(campaign_id)
            if not best_trial:
                best_arch = dbc.Alert("No trials completed yet", color="info")
            else:
                arch = best_trial['architecture']
                best_arch = dbc.Card([
                    dbc.CardBody([
                        html.P([
                            html.Strong("Accuracy: "), f"{best_trial['validation_accuracy']:.2%}"
                        ]),
                        html.P([
                            html.Strong("Parameters: "), f"{best_trial['num_parameters']:,}"
                        ]),
                        html.P([
                            html.Strong("Layers: "), arch['num_layers']
                        ]),
                        html.P([
                            html.Strong("Operations: "), ", ".join(arch['operations'])
                        ]),
                    ])
                ], color="light")

            # Architecture visualization (simple bar chart of operations)
            if best_trial:
                arch = best_trial['architecture']
                ops = arch['operations']
                channels = arch['channels']

                fig = go.Figure(data=[
                    go.Bar(
                        x=[f"Layer {i+1}" for i in range(len(ops))],
                        y=channels,
                        text=ops,
                        textposition='auto',
                        marker=dict(color=channels, colorscale='Viridis')
                    )
                ])
                fig.update_layout(
                    title="Best Architecture Visualization",
                    xaxis_title="Layer",
                    yaxis_title="Channels",
                    template='plotly_white',
                    height=400
                )
            else:
                fig = go.Figure()

            return True, f"Campaign: {campaign['name']}", info, trials_table, best_arch, fig, campaign_id

        raise PreventUpdate

    # ==================== Architecture Export ====================

    @app.callback(
        [Output('nas-export-modal', 'is_open'),
         Output('nas-exported-code', 'children')],
        [Input('export-nas-architecture-btn', 'n_clicks'),
         Input('close-export-modal-btn', 'n_clicks'),
         Input('nas-export-format', 'value')],
        [State('selected-nas-campaign-id', 'data')],
        prevent_initial_call=True
    )
    def handle_architecture_export(export_click, close_click, export_format, campaign_id):
        """Handle architecture export."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger_id = ctx.triggered[0]['prop_id']

        # Close export modal
        if 'close-export-modal-btn' in trigger_id:
            return False, ""

        # Open export modal
        if 'export-nas-architecture-btn' in trigger_id or 'nas-export-format' in trigger_id:
            if not campaign_id:
                return False, ""

            # Get best trial
            best_trial = NASService.get_best_architecture(campaign_id)
            if not best_trial:
                return True, "No architecture available to export"

            # Export architecture
            code = NASService.export_architecture(best_trial['id'], format=export_format or 'pytorch')

            return True, code or "Failed to export architecture"

        raise PreventUpdate

    logger.info("NAS callbacks registered successfully")
