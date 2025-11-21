"""
Callbacks for experiment comparison page.
"""
from dash import Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from services.comparison_service import ComparisonService
from layouts.experiment_comparison import (
    create_overview_tab,
    create_metrics_tab,
    create_visualizations_tab,
    create_statistical_tab,
    create_configuration_tab
)


def register_comparison_callbacks(app):
    """Register all comparison-related callbacks."""

    @app.callback(
        Output('comparison-tab-content', 'children'),
        [Input('comparison-tabs', 'active_tab'),
         Input('comparison-data-store', 'data')]
    )
    def render_tab_content(active_tab, data_store):
        """Render content based on active tab."""
        if not data_store or 'experiment_ids' not in data_store:
            return dbc.Alert("No experiment data available", color="warning")

        experiment_ids = data_store['experiment_ids']

        # Validate and load comparison data
        valid, error_msg = ComparisonService.validate_comparison_request(experiment_ids)

        if not valid:
            return dbc.Alert([
                html.H4("Comparison Error", className="alert-heading"),
                html.P(error_msg),
                html.Hr(),
                html.A("Return to Experiments", href="/experiments", className="btn btn-primary")
            ], color="danger")

        # Load comparison data
        try:
            comparison_data = ComparisonService.get_comparison_data(experiment_ids)
        except Exception as e:
            return dbc.Alert([
                html.H4("Error Loading Comparison Data", className="alert-heading"),
                html.P(f"An error occurred: {str(e)}"),
                html.A("Return to Experiments", href="/experiments", className="btn btn-primary")
            ], color="danger")

        # Render appropriate tab
        if active_tab == 'overview':
            content = create_overview_tab(comparison_data)
        elif active_tab == 'metrics':
            content = create_metrics_tab(comparison_data)
        elif active_tab == 'visualizations':
            content = create_visualizations_tab(comparison_data)
        elif active_tab == 'statistical':
            content = create_statistical_tab(comparison_data)
        elif active_tab == 'configuration':
            content = create_configuration_tab(comparison_data)
        else:
            content = dbc.Alert("Unknown tab", color="warning")

        return content

    @app.callback(
        Output('key-differences-content', 'children'),
        [Input('comparison-data-store', 'data')]
    )
    def render_key_differences(data_store):
        """Render key differences summary."""
        if not data_store or 'experiment_ids' not in data_store:
            raise PreventUpdate

        experiment_ids = data_store['experiment_ids']

        try:
            comparison_data = ComparisonService.get_comparison_data(experiment_ids)
            differences = ComparisonService.identify_key_differences(comparison_data)

            if not differences:
                return html.P("No significant differences identified.", className="text-muted")

            return html.Ul([
                html.Li(diff) for diff in differences
            ])
        except Exception as e:
            return html.P(f"Error: {str(e)}", className="text-danger")

    @app.callback(
        [Output('share-link-modal', 'is_open'),
         Output('comparison-share-link-input', 'value')],
        [Input('share-comparison-link', 'n_clicks'),
         Input('close-share-modal', 'n_clicks')],
        [State('share-link-modal', 'is_open'),
         State('comparison-data-store', 'data')]
    )
    def toggle_share_modal(share_clicks, close_clicks, is_open, data_store):
        """Toggle share link modal and populate link."""
        if not callback_context.triggered:
            raise PreventUpdate

        trigger_id = callback_context.triggered[0]['prop_id'].split('.')[0]

        if trigger_id == 'share-comparison-link':
            if data_store and 'experiment_ids' in data_store:
                experiment_ids = data_store['experiment_ids']
                ids_str = ','.join(map(str, experiment_ids))
                # Generate full URL (assuming running on localhost:8050, adjust as needed)
                share_link = f"http://localhost:8050/compare?ids={ids_str}"
                return True, share_link
            else:
                return True, ""

        elif trigger_id == 'close-share-modal':
            return False, ""

        raise PreventUpdate

    @app.callback(
        Output('toast-container', 'children', allow_duplicate=True),
        [Input('copy-comparison-link-btn', 'n_clicks')],
        [State('comparison-share-link-input', 'value')],
        prevent_initial_call=True
    )
    def copy_link_to_clipboard(n_clicks, link_value):
        """Show toast notification when link is copied."""
        if not n_clicks:
            raise PreventUpdate

        # Note: Actual clipboard copy needs to be done on client side via JavaScript
        # This just shows a toast notification
        toast = dbc.Toast(
            [html.P("Link copied to clipboard!", className="mb-0")],
            id="copy-success-toast",
            header="Success",
            is_open=True,
            dismissable=True,
            icon="success",
            duration=3000,
            style={"position": "fixed", "top": 66, "right": 10, "width": 350}
        )

        return toast

    @app.callback(
        Output('toast-container', 'children', allow_duplicate=True),
        [Input('export-comparison-pdf', 'n_clicks')],
        [State('comparison-data-store', 'data')],
        prevent_initial_call=True
    )
    def export_comparison_pdf(n_clicks, data_store):
        """Export comparison as PDF (placeholder)."""
        if not n_clicks:
            raise PreventUpdate

        # TODO: Implement PDF export functionality
        toast = dbc.Toast(
            [html.P("PDF export feature coming soon!", className="mb-0")],
            id="export-pdf-toast",
            header="Info",
            is_open=True,
            dismissable=True,
            icon="info",
            duration=3000,
            style={"position": "fixed", "top": 66, "right": 10, "width": 350}
        )

        return toast
