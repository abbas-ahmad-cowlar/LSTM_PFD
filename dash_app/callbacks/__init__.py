"""
Callback registration.
"""
from dash import Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import re


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
            from layouts.home import create_home_layout
            return create_home_layout()
        elif pathname == '/data-generation':
            from layouts.data_generation import create_data_generation_layout
            return create_data_generation_layout()
        elif pathname == '/data-explorer':
            from layouts.data_explorer import create_data_explorer_layout
            return create_data_explorer_layout()
        elif pathname == '/signal-viewer':
            from layouts.signal_viewer import create_signal_viewer_layout
            return create_signal_viewer_layout()
        elif pathname == '/experiments':
            from layouts.experiments import create_experiments_layout
            return create_experiments_layout()
        elif pathname == '/experiment/new':
            from layouts.experiment_wizard import create_experiment_wizard_layout
            return create_experiment_wizard_layout()
        elif re.match(r'/experiment/(\d+)/monitor', pathname):
            from layouts.training_monitor import create_training_monitor_layout
            experiment_id = int(pathname.split('/')[2])
            return create_training_monitor_layout(experiment_id)
        elif re.match(r'/experiment/(\d+)/results', pathname):
            from layouts.experiment_results import create_experiment_results_layout
            experiment_id = int(pathname.split('/')[2])
            return create_experiment_results_layout(experiment_id)
        elif pathname == '/compare':
            from layouts.experiment_comparison import create_experiment_comparison_layout
            from dash import callback_context
            # Parse query string for experiment IDs
            import urllib.parse as urlparse
            from flask import request
            query_params = request.args
            ids_str = query_params.get('ids', '')
            if ids_str:
                try:
                    experiment_ids = [int(id.strip()) for id in ids_str.split(',')]
                    return create_experiment_comparison_layout(experiment_ids)
                except ValueError:
                    from dash import html
                    return html.Div([
                        html.H2("Invalid Comparison Request"),
                        html.P("Invalid experiment IDs provided."),
                        html.A("Return to Experiments", href="/experiments")
                    ], className="text-center mt-5")
            else:
                from dash import html
                return html.Div([
                    html.H2("Invalid Comparison Request"),
                    html.P("No experiment IDs provided. Use: /compare?ids=1,2,3"),
                    html.A("Return to Experiments", href="/experiments")
                ], className="text-center mt-5")
        elif pathname == '/xai':
            from layouts.xai_dashboard import create_xai_dashboard_layout
            return create_xai_dashboard_layout()
        elif pathname == '/system-health':
            from layouts.system_health import create_system_health_layout
            return create_system_health_layout()
        elif pathname == '/hpo/campaigns':
            from layouts.hpo_campaigns import create_hpo_campaigns_layout
            return create_hpo_campaigns_layout()
        elif pathname == '/deployment':
            from layouts.deployment import create_deployment_layout
            return create_deployment_layout()
        else:
            from dash import html
            return html.Div([
                html.H2("404: Page Not Found"),
                html.P(f"The page '{pathname}' does not exist."),
                html.A("Return to Home", href="/")
            ], className="text-center mt-5")

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
