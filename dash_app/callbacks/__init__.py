"""
Callback registration.
"""
from dash import Input, Output, State, callback_context
from dash.exceptions import PreventUpdate


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
        else:
            from dash import html
            return html.Div([
                html.H2("404: Page Not Found"),
                html.P(f"The page '{pathname}' does not exist."),
                html.A("Return to Home", href="/")
            ], className="text-center mt-5")

    # Import and register other callbacks
    try:
        from callbacks.data_explorer_callbacks import register_data_explorer_callbacks
        register_data_explorer_callbacks(app)
    except ImportError:
        pass

    try:
        from callbacks.signal_viewer_callbacks import register_signal_viewer_callbacks
        register_signal_viewer_callbacks(app)
    except ImportError:
        pass
