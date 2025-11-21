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
        else:
            from dash import html
            return html.Div([
                html.H2("404: Page Not Found"),
                html.P(f"The page '{pathname}' does not exist."),
                html.A("Return to Home", href="/")
            ], className="text-center mt-5")

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
