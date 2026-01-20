"""
HPO Campaign callbacks (Phase 11C).
Callbacks for HPO campaign management and monitoring.
"""
import json
import dash
from dash import Input, Output, State, html, callback_context
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from layouts.hpo_campaigns import create_hpo_campaign_card
from services.hpo_service import HPOService
from tasks.hpo_tasks import run_hpo_campaign_task, stop_hpo_campaign_task
from models.experiment import ExperimentStatus
from utils.logger import setup_logger
from utils.data_objects import CampaignObject
from database.connection import get_db_session
from models.dataset import Dataset
from utils.auth_utils import get_current_user_id

logger = setup_logger(__name__)


def register_hpo_callbacks(app):
    """Register HPO campaign callbacks."""

    @app.callback(
        Output('hpo-modal', 'is_open'),
        [
            Input('new-hpo-campaign-btn', 'n_clicks'),
            Input('hpo-cancel-btn', 'n_clicks'),
            Input('hpo-launch-btn', 'n_clicks')
        ],
        State('hpo-modal', 'is_open')
    )
    def toggle_hpo_modal(new_clicks, cancel_clicks, launch_clicks, is_open):
        """Toggle HPO campaign creation modal."""
        ctx = callback_context
        if not ctx.triggered:
            return False

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id in ['new-hpo-campaign-btn', 'hpo-cancel-btn', 'hpo-launch-btn']:
            return not is_open

        return is_open

    @app.callback(
        Output('hpo-campaigns-list', 'children'),
        Input('url', 'pathname'),
        Input('system-health-interval', 'n_intervals')  # Reuse interval for updates
    )
    def update_campaigns_list(pathname, n_intervals):
        """
        Update HPO campaigns list.

        Args:
            pathname: Current URL path
            n_intervals: Interval counter for auto-refresh

        Returns:
            List of campaign cards
        """
        if pathname != '/hpo/campaigns':
            raise PreventUpdate

        try:
            campaigns = HPOService.get_all_campaigns()

            if not campaigns:
                return dbc.Alert(
                    [
                        html.H5("No HPO Campaigns Yet", className="alert-heading"),
                        html.P("Click 'New Campaign' to create your first hyperparameter optimization campaign."),
                    ],
                    color="info",
                    className="text-center"
                )

            # Create campaign cards
            cards = []
            for campaign in campaigns:
                # Convert campaign dict to object-like structure
                campaign_obj = CampaignObject(campaign)
                cards.append(create_hpo_campaign_card(campaign_obj))

            return cards

        except Exception as e:
            logger.error(f"Failed to update campaigns list: {e}", exc_info=True)
            return dbc.Alert(
                f"Error loading campaigns: {str(e)}",
                color="danger"
            )

    @app.callback(
        Output('hpo-launch-btn', 'children'),
        [
            Input('hpo-campaign-name', 'value'),
            Input('hpo-model-type', 'value'),
            Input('hpo-method', 'value'),
            Input('hpo-num-trials', 'value'),
            Input('hpo-metric', 'value')
        ]
    )
    def launch_hpo_campaign(campaign_name, model_type, method, num_trials, metric):
        """
        Launch HPO campaign (button click handler in separate callback).

        This callback just updates the button text based on form state.
        """
        # Check if all required fields are filled
        if all([campaign_name, model_type, method, num_trials, metric]):
            return [html.I(className="fas fa-rocket me-2"), "Launch Campaign"]
        else:
            return [html.I(className="fas fa-rocket me-2"), "Launch Campaign (Fill All Fields)"]

    @app.callback(
        Output('hpo-modal', 'is_open', allow_duplicate=True),
        Input('hpo-launch-btn', 'n_clicks'),
        [
            State('hpo-campaign-name', 'value'),
            State('hpo-model-type', 'value'),
            State('hpo-method', 'value'),
            State('hpo-num-trials', 'value'),
            State('hpo-metric', 'value')
        ],
        prevent_initial_call=True
    )
    def create_and_launch_campaign(n_clicks, campaign_name, model_type, method, num_trials, metric):
        """
        Create and launch HPO campaign.

        Args:
            n_clicks: Button clicks
            campaign_name: Campaign name
            model_type: Model type
            method: Optimization method
            num_trials: Number of trials
            metric: Optimization metric

        Returns:
            Modal state (closed after launch)
        """
        if not n_clicks:
            raise PreventUpdate

        try:
            # Validate inputs
            if not all([campaign_name, model_type, method, num_trials, metric]):
                logger.warning("Missing required fields for HPO campaign")
                raise PreventUpdate

            # Get a dataset (use first available dataset for now)
            # In production, this should be a dropdown in the modal
            with get_db_session() as session:
                dataset = session.query(Dataset).first()
                if not dataset:
                    logger.error("No datasets available")
                    raise PreventUpdate

                dataset_id = dataset.id

            # Get default search space for model type
            search_space = HPOService.get_default_search_space(model_type)

            # Determine direction
            direction = "maximize" if metric in ["val_accuracy", "f1_score"] else "minimize"

            # Create campaign
            campaign_id = HPOService.create_campaign(
                name=campaign_name,
                method=method,
                base_model_type=model_type,
                dataset_id=dataset_id,
                search_space=search_space,
                num_trials=int(num_trials),
                metric=metric,
                direction=direction,
                created_by=get_current_user_id() or 1  # Fallback to user 1 if no session
            )

            if not campaign_id:
                logger.error("Failed to create HPO campaign")
                raise PreventUpdate

            logger.info(f"Created HPO campaign {campaign_id}, launching task...")

            # Launch Celery task
            task = run_hpo_campaign_task.delay(campaign_id)
            logger.info(f"Launched HPO task {task.id} for campaign {campaign_id}")

            # Close modal
            return False

        except Exception as e:
            logger.error(f"Failed to create/launch HPO campaign: {e}", exc_info=True)
            raise PreventUpdate

    @app.callback(
        Output('hpo-campaigns-list', 'children', allow_duplicate=True),
        Input({'type': 'stop-hpo-btn', 'index': dash.dependencies.ALL}, 'n_clicks'),
        prevent_initial_call=True
    )
    def stop_campaign(n_clicks_list):
        """
        Stop a running HPO campaign.

        Args:
            n_clicks_list: List of button clicks

        Returns:
            Updated campaigns list
        """
        ctx = callback_context
        if not ctx.triggered or not any(n_clicks_list):
            raise PreventUpdate

        # Get campaign ID from button
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        button_data = json.loads(button_id)
        campaign_id = button_data['index']

        logger.info(f"Stopping HPO campaign {campaign_id}")

        try:
            # Stop campaign
            stop_hpo_campaign_task.delay(campaign_id)

            # Return updated list (will be refreshed by interval)
            raise PreventUpdate

        except Exception as e:
            logger.error(f"Failed to stop campaign {campaign_id}: {e}", exc_info=True)
            raise PreventUpdate

    # HPO-5: Resume Campaign Callback
    @app.callback(
        Output('hpo-campaigns-list', 'children', allow_duplicate=True),
        Input({'type': 'resume-hpo-btn', 'index': dash.dependencies.ALL}, 'n_clicks'),
        prevent_initial_call=True
    )
    def resume_campaign(n_clicks_list):
        """Resume a paused/cancelled HPO campaign."""
        ctx = callback_context
        if not ctx.triggered or not any(n_clicks_list):
            raise PreventUpdate

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        button_data = json.loads(button_id)
        campaign_id = button_data['index']

        logger.info(f"Resuming HPO campaign {campaign_id}")

        try:
            # Resume campaign in service
            if HPOService.resume_campaign(campaign_id):
                # Launch resumed task
                run_hpo_campaign_task.delay(campaign_id)
                logger.info(f"Campaign {campaign_id} resumed successfully")
            else:
                logger.warning(f"Failed to resume campaign {campaign_id}")

            raise PreventUpdate

        except Exception as e:
            logger.error(f"Failed to resume campaign {campaign_id}: {e}", exc_info=True)
            raise PreventUpdate

    # HPO-6: Export Modal Toggle
    @app.callback(
        [Output('hpo-export-modal', 'is_open'),
         Output('hpo-selected-campaign-id', 'data')],
        [Input({'type': 'export-hpo-btn', 'index': dash.dependencies.ALL}, 'n_clicks'),
         Input('hpo-export-cancel-btn', 'n_clicks')],
        [State('hpo-export-modal', 'is_open'),
         State('hpo-selected-campaign-id', 'data')],
        prevent_initial_call=True
    )
    def toggle_export_modal(export_clicks, cancel_clicks, is_open, selected_id):
        """Toggle export modal and track selected campaign."""
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate

        trigger = ctx.triggered[0]['prop_id']

        if 'export-hpo-btn' in trigger and any(export_clicks):
            button_id = trigger.split('.')[0]
            button_data = json.loads(button_id)
            campaign_id = button_data['index']
            return True, campaign_id
        elif 'hpo-export-cancel-btn' in trigger:
            return False, selected_id

        return is_open, selected_id

    # HPO-6: Export Preview
    @app.callback(
        Output('hpo-export-preview', 'children'),
        [Input('hpo-export-format', 'value'),
         Input('hpo-selected-campaign-id', 'data')],
        prevent_initial_call=True
    )
    def update_export_preview(format, campaign_id):
        """Show preview of export content."""
        if not campaign_id:
            return html.P("No campaign selected", className="text-muted")

        content = HPOService.export_results(campaign_id, format)
        if not content:
            return html.P("No data available for export", className="text-danger")

        # Truncate preview
        preview = content[:500] + "..." if len(content) > 500 else content

        return html.Pre(preview, className="bg-light p-3", style={"maxHeight": "200px", "overflow": "auto"})

    # HPO-6: Export Download
    @app.callback(
        Output('hpo-download-export', 'data'),
        Input('hpo-export-download-btn', 'n_clicks'),
        [State('hpo-export-format', 'value'),
         State('hpo-selected-campaign-id', 'data')],
        prevent_initial_call=True
    )
    def download_export(n_clicks, format, campaign_id):
        """Download exported campaign data."""
        if not n_clicks or not campaign_id:
            raise PreventUpdate

        content = HPOService.export_results(campaign_id, format)
        if not content:
            raise PreventUpdate

        # Determine file extension and MIME type
        extensions = {"json": ".json", "yaml": ".yaml", "python": ".py"}
        ext = extensions.get(format, ".json")

        campaign = HPOService.get_campaign(campaign_id)
        filename = f"hpo_{campaign['name']}_{format}{ext}" if campaign else f"hpo_export{ext}"

        return dict(content=content, filename=filename)

    # HPO-7: Visualization Card Toggle
    @app.callback(
        Output('hpo-viz-card', 'style'),
        Input({'type': 'viz-hpo-btn', 'index': dash.dependencies.ALL}, 'n_clicks'),
        [State('hpo-selected-campaign-id', 'data')],
        prevent_initial_call=True
    )
    def toggle_viz_card(n_clicks_list, current_id):
        """Show visualization card when visualize button is clicked."""
        ctx = callback_context
        if not ctx.triggered or not any(n_clicks_list):
            raise PreventUpdate

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        button_data = json.loads(button_id)
        campaign_id = button_data['index']

        return {"display": "block"}

    # HPO-7: Parallel Coordinates Chart
    @app.callback(
        Output('hpo-parallel-coords-chart', 'figure'),
        [Input({'type': 'viz-hpo-btn', 'index': dash.dependencies.ALL}, 'n_clicks'),
         Input('hpo-viz-tabs', 'active_tab')],
        prevent_initial_call=True
    )
    def update_parallel_coords(n_clicks_list, active_tab):
        """Update parallel coordinates visualization."""
        import plotly.graph_objects as go

        ctx = callback_context
        if not ctx.triggered or not any(n_clicks_list or []):
            raise PreventUpdate

        # Get campaign ID
        for trigger in ctx.triggered:
            if 'viz-hpo-btn' in trigger['prop_id']:
                button_id = trigger['prop_id'].split('.')[0]
                button_data = json.loads(button_id)
                campaign_id = button_data['index']
                break
        else:
            raise PreventUpdate

        # Get trial data
        trials_data = HPOService.get_trials_dataframe(campaign_id)

        if not trials_data or len(trials_data.get("trial_id", [])) == 0:
            return go.Figure().add_annotation(
                text="No trial data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

        # Build dimensions for parallel coordinates
        dimensions = []

        # Add accuracy as color dimension
        if trials_data.get("test_accuracy"):
            dimensions.append(dict(
                range=[min(trials_data["test_accuracy"]), max(trials_data["test_accuracy"])],
                label="Accuracy",
                values=trials_data["test_accuracy"]
            ))

        # Add hyperparameters
        skip_keys = {"trial_id", "test_accuracy", "test_loss"}
        for param, values in trials_data.items():
            if param in skip_keys:
                continue
            if all(v is None for v in values):
                continue

            # Convert values for visualization
            numeric_values = []
            for v in values:
                if v is None:
                    numeric_values.append(0)
                elif isinstance(v, (int, float)):
                    numeric_values.append(float(v))
                elif isinstance(v, str):
                    numeric_values.append(hash(v) % 100)
                else:
                    numeric_values.append(0)

            dimensions.append(dict(
                range=[min(numeric_values), max(numeric_values)] if numeric_values else [0, 1],
                label=param,
                values=numeric_values
            ))

        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=trials_data.get("test_accuracy", [0]),
                colorscale="Viridis",
                showscale=True
            ),
            dimensions=dimensions
        ))

        fig.update_layout(
            title="Hyperparameter Parallel Coordinates",
            margin=dict(l=100, r=50, t=50, b=50)
        )

        return fig

    # HPO-8: Parameter Importance Chart
    @app.callback(
        Output('hpo-param-importance-chart', 'figure'),
        [Input({'type': 'viz-hpo-btn', 'index': dash.dependencies.ALL}, 'n_clicks'),
         Input('hpo-viz-tabs', 'active_tab')],
        prevent_initial_call=True
    )
    def update_param_importance(n_clicks_list, active_tab):
        """Update parameter importance visualization."""
        import plotly.express as px
        import plotly.graph_objects as go

        ctx = callback_context
        if not ctx.triggered or not any(n_clicks_list or []):
            raise PreventUpdate

        # Get campaign ID
        for trigger in ctx.triggered:
            if 'viz-hpo-btn' in trigger['prop_id']:
                button_id = trigger['prop_id'].split('.')[0]
                button_data = json.loads(button_id)
                campaign_id = button_data['index']
                break
        else:
            raise PreventUpdate

        # Get importance data
        importance = HPOService.get_parameter_importance(campaign_id)

        if not importance:
            return go.Figure().add_annotation(
                text="Not enough trial data for importance analysis (min 3 trials needed)",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

        # Create bar chart
        params = list(importance.keys())
        values = list(importance.values())

        fig = px.bar(
            x=values,
            y=params,
            orientation='h',
            labels={"x": "Importance Score", "y": "Hyperparameter"},
            title="Hyperparameter Importance (Correlation with Accuracy)"
        )

        fig.update_layout(
            yaxis=dict(autorange="reversed"),  # Most important at top
            margin=dict(l=150, r=50, t=50, b=50)
        )

        return fig

    # Optimization History Chart
    @app.callback(
        Output('hpo-optimization-history-chart', 'figure'),
        [Input({'type': 'viz-hpo-btn', 'index': dash.dependencies.ALL}, 'n_clicks'),
         Input('hpo-viz-tabs', 'active_tab')],
        prevent_initial_call=True
    )
    def update_optimization_history(n_clicks_list, active_tab):
        """Update optimization history visualization."""
        import plotly.graph_objects as go

        ctx = callback_context
        if not ctx.triggered or not any(n_clicks_list or []):
            raise PreventUpdate

        # Get campaign ID
        for trigger in ctx.triggered:
            if 'viz-hpo-btn' in trigger['prop_id']:
                button_id = trigger['prop_id'].split('.')[0]
                button_data = json.loads(button_id)
                campaign_id = button_data['index']
                break
        else:
            raise PreventUpdate

        # Get trial data
        trials_data = HPOService.get_trials_dataframe(campaign_id)

        if not trials_data or len(trials_data.get("trial_id", [])) == 0:
            return go.Figure().add_annotation(
                text="No trial data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

        accuracies = trials_data.get("test_accuracy", [])
        trial_nums = list(range(1, len(accuracies) + 1))

        # Calculate running best
        running_best = []
        current_best = 0
        for acc in accuracies:
            current_best = max(current_best, acc)
            running_best.append(current_best)

        fig = go.Figure()

        # Trial accuracy scatter
        fig.add_trace(go.Scatter(
            x=trial_nums,
            y=accuracies,
            mode='markers',
            name='Trial Accuracy',
            marker=dict(size=8, color='blue', opacity=0.6)
        ))

        # Running best line
        fig.add_trace(go.Scatter(
            x=trial_nums,
            y=running_best,
            mode='lines',
            name='Best So Far',
            line=dict(color='green', width=2)
        ))

        fig.update_layout(
            title="Optimization History",
            xaxis_title="Trial Number",
            yaxis_title="Accuracy",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            margin=dict(l=50, r=50, t=50, b=50)
        )

        return fig

