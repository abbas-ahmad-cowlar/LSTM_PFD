"""
Callback registration.
"""
import urllib.parse as urlparse
from dash import Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate
from flask import request
import re

# Layout imports
from layouts.home import create_home_layout
from layouts.data_generation import create_data_generation_layout
from layouts.data_explorer import create_data_explorer_layout
from layouts.signal_viewer import create_signal_viewer_layout
from layouts.datasets import create_datasets_layout
from layouts.experiments import create_experiments_layout
from layouts.experiment_wizard import create_experiment_wizard_layout
from layouts.training_monitor import create_training_monitor_layout
from layouts.experiment_results import create_experiment_results_layout
from layouts.experiment_comparison import create_experiment_comparison_layout
from layouts.xai_dashboard import create_xai_dashboard_layout
from layouts.system_health import create_system_health_layout
from layouts.hpo_campaigns import create_hpo_campaigns_layout
from layouts.deployment import create_deployment_layout
from layouts.api_monitoring import create_api_monitoring_layout
from layouts.evaluation_dashboard import create_evaluation_dashboard_layout
from layouts.testing_dashboard import create_testing_dashboard_layout
from layouts.feature_engineering import create_feature_engineering_layout
from layouts.visualization import create_visualization_layout
from layouts.nas_dashboard import create_nas_dashboard_layout
from layouts.settings import create_settings_layout



def register_all_callbacks(app):
    """Register all callbacks with the app."""

    # Navigation callback
    @app.callback(
        Output('page-content', 'children'),
        Input('url', 'pathname')
    )
    def display_page(pathname):
        """Route to appropriate page based on URL."""
        if pathname == '/' or pathname is None:
            return create_home_layout()
        elif pathname == '/data-generation':
            return create_data_generation_layout()
        elif pathname == '/data-explorer':
            return create_data_explorer_layout()
        elif pathname == '/signal-viewer':
            return create_signal_viewer_layout()
        elif pathname == '/datasets':
            return create_datasets_layout()
        elif pathname == '/experiments':
            return create_experiments_layout()
        elif pathname == '/experiment/new':
            return create_experiment_wizard_layout()
        elif re.match(r'/experiment/(\d+)/monitor', pathname):
            experiment_id = int(pathname.split('/')[2])
            return create_training_monitor_layout(experiment_id)
        elif re.match(r'/experiment/(\d+)/results', pathname):
            experiment_id = int(pathname.split('/')[2])
            return create_experiment_results_layout(experiment_id)
        elif pathname == '/compare':
            # Parse query string for experiment IDs
            query_params = request.args
            ids_str = query_params.get('ids', '')
            if ids_str:
                try:
                    experiment_ids = [int(id.strip()) for id in ids_str.split(',')]
                    return create_experiment_comparison_layout(experiment_ids)
                except ValueError:
                    return html.Div([
                        html.H2("Invalid Comparison Request"),
                        html.P("Invalid experiment IDs provided."),
                        html.A("Return to Experiments", href="/experiments")
                    ], className="text-center mt-5")
            else:
                return html.Div([
                    html.H2("Invalid Comparison Request"),
                    html.P("No experiment IDs provided. Use: /compare?ids=1,2,3"),
                    html.A("Return to Experiments", href="/experiments")
                ], className="text-center mt-5")
        elif pathname == '/xai':
            return create_xai_dashboard_layout()
        elif pathname == '/system-health':
            return create_system_health_layout()
        elif pathname == '/hpo/campaigns':
            return create_hpo_campaigns_layout()
        elif pathname == '/deployment':
            return create_deployment_layout()
        elif pathname == '/api-monitoring':
            return create_api_monitoring_layout()
        elif pathname == '/evaluation':
            return create_evaluation_dashboard_layout()
        elif pathname == '/testing':
            return create_testing_dashboard_layout()
        elif pathname == '/feature-engineering':
            return create_feature_engineering_layout()
        elif pathname == '/visualization':
            return create_visualization_layout()
        elif pathname == '/nas':
            return create_nas_dashboard_layout()
        elif pathname == '/settings':
            return create_settings_layout()
        else:
            return html.Div([
                html.H2("404: Page Not Found"),
                html.P(f"The page '{pathname}' does not exist."),
                html.A("Return to Home", href="/")
            ], className="text-center mt-5")

    # Register Home Page callbacks
    try:
        from callbacks.home_callbacks import register_home_callbacks
        register_home_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import home_callbacks: {e}")

    # Import and register Phase 0 (data generation) callbacks
    try:
        from callbacks.data_generation_callbacks import register_data_generation_callbacks
        register_data_generation_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import data_generation_callbacks: {e}")

    # Import and register Phase 11A callbacks
    try:
        from callbacks.data_explorer_callbacks import register_data_explorer_callbacks
        register_data_explorer_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import data_explorer_callbacks: {e}")

    try:
        from callbacks.signal_viewer_callbacks import register_signal_viewer_callbacks
        register_signal_viewer_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import signal_viewer_callbacks: {e}")

    # Import and register Dataset Management callbacks
    try:
        from callbacks.datasets_callbacks import register_datasets_callbacks
        register_datasets_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import datasets_callbacks: {e}")

    # Import and register Phase 11B callbacks
    try:
        from callbacks.experiment_wizard_callbacks import register_experiment_wizard_callbacks
        register_experiment_wizard_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import experiment_wizard_callbacks: {e}")

    try:
        from callbacks.training_monitor_callbacks import register_training_monitor_callbacks
        register_training_monitor_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import training_monitor_callbacks: {e}")

    # Import and register comparison callbacks (Feature #2)
    try:
        from callbacks.comparison_callbacks import register_comparison_callbacks
        register_comparison_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import comparison_callbacks: {e}")

    try:
        from callbacks.experiments_callbacks import register_experiments_callbacks
        register_experiments_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import experiments_callbacks: {e}")

    # Import and register Phase 11C XAI callbacks
    try:
        from callbacks.xai_callbacks import register_xai_callbacks
        register_xai_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import xai_callbacks: {e}")

    # Import and register Phase 11D System Health callbacks
    try:
        from callbacks.system_health_callbacks import register_system_health_callbacks
        register_system_health_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import system_health_callbacks: {e}")

    # Import and register Phase 11C HPO callbacks
    try:
        from callbacks.hpo_callbacks import register_hpo_callbacks
        register_hpo_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import hpo_callbacks: {e}")

    # Import and register Deployment callbacks
    try:
        from callbacks.deployment_callbacks import register_deployment_callbacks
        register_deployment_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import deployment_callbacks: {e}")

    # Import and register API Monitoring callbacks
    try:
        from callbacks.api_monitoring_callbacks import register_api_monitoring_callbacks
        register_api_monitoring_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import api_monitoring_callbacks: {e}")

    # Import and register Enhanced Evaluation callbacks
    try:
        from callbacks.evaluation_callbacks import register_evaluation_callbacks
        register_evaluation_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import evaluation_callbacks: {e}")

    # Import and register Testing & QA callbacks
    try:
        from callbacks.testing_callbacks import register_testing_callbacks
        register_testing_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import testing_callbacks: {e}")

    # Import and register Feature Engineering callbacks
    try:
        from callbacks.feature_callbacks import register_feature_callbacks
        register_feature_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import feature_callbacks: {e}")

    # Import and register Notification Management callbacks
    try:
        from callbacks.notification_callbacks import register_notification_callbacks
        register_notification_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import notification_callbacks: {e}")

    # Import and register Email Digest Queue callbacks
    try:
        from callbacks.email_digest_callbacks import register_email_digest_callbacks
        register_email_digest_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import email_digest_callbacks: {e}")

    # Import and register Enhanced Visualization callbacks
    try:
        from callbacks.visualization_callbacks import register_visualization_callbacks
        register_visualization_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import visualization_callbacks: {e}")

    # Import and register NAS Dashboard callbacks
    try:
        from callbacks.nas_callbacks import register_nas_callbacks
        register_nas_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import nas_callbacks: {e}")

    # Import and register Settings (API Keys) callbacks
    try:
        from callbacks.api_key_callbacks import register_api_key_callbacks
        register_api_key_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import api_key_callbacks: {e}")

    # Import and register Webhook Management callbacks
    try:
        from callbacks.webhook_callbacks import register_webhook_callbacks
        register_webhook_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import webhook_callbacks: {e}")

    # Import and register Tag Management callbacks
    try:
        from callbacks.tag_callbacks import register_tag_callbacks
        register_tag_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import tag_callbacks: {e}")

    # Import and register Saved Search callbacks
    try:
        from callbacks.saved_search_callbacks import register_saved_search_callbacks
        register_saved_search_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import saved_search_callbacks: {e}")

    # Import and register User Profile callbacks
    try:
        from callbacks.profile_callbacks import register_profile_callbacks
        register_profile_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import profile_callbacks: {e}")

    # Import and register Security Settings callbacks
    try:
        from callbacks.security_callbacks import register_security_callbacks
        register_security_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import security_callbacks: {e}")
