"""
HPO Campaign callbacks (Phase 11C).
Callbacks for HPO campaign management and monitoring.
"""
import json
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
                created_by=get_current_user_id()
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


# Import dash for pattern matching callbacks
import dash
