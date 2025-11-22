"""
Callbacks for experiments list page.
"""
from dash import Input, Output, State, callback_context, dcc
from dash.exceptions import PreventUpdate

from database.connection import get_session
from models.experiment import Experiment, ExperimentStatus
from layouts.experiments import create_experiments_table


def register_experiments_callbacks(app):
    """Register all experiments list callbacks."""

    @app.callback(
        Output('experiments-table-container', 'children'),
        [Input('url', 'pathname'),
         Input('experiment-search', 'value'),
         Input('experiment-model-filter', 'value'),
         Input('experiment-status-filter', 'value')]
    )
    def load_experiments(pathname, search_value, model_filter, status_filter):
        """Load and display experiments with filtering."""
        if pathname != '/experiments':
            raise PreventUpdate

        session = get_session()

        # Build query
        query = session.query(Experiment)

        # Apply filters
        if search_value:
            query = query.filter(Experiment.name.ilike(f'%{search_value}%'))

        if model_filter:
            query = query.filter(Experiment.model_type.in_(model_filter))

        if status_filter:
            status_enums = [ExperimentStatus(s) for s in status_filter]
            query = query.filter(Experiment.status.in_(status_enums))

        # Order by created date (newest first)
        query = query.order_by(Experiment.created_at.desc())

        experiments = query.all()
        session.close()

        return create_experiments_table(experiments)

    @app.callback(
        Output('experiments-summary', 'children'),
        [Input('url', 'pathname')]
    )
    def load_experiments_summary(pathname):
        """Load summary statistics for experiments."""
        if pathname != '/experiments':
            raise PreventUpdate

        session = get_session()

        total_experiments = session.query(Experiment).count()
        completed_experiments = session.query(Experiment).filter(
            Experiment.status == ExperimentStatus.COMPLETED
        ).count()
        running_experiments = session.query(Experiment).filter(
            Experiment.status == ExperimentStatus.RUNNING
        ).count()
        failed_experiments = session.query(Experiment).filter(
            Experiment.status == ExperimentStatus.FAILED
        ).count()

        session.close()

        from dash import html
        import dash_bootstrap_components as dbc

        return dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5(str(total_experiments), className="mb-0"),
                    html.P("Total Experiments", className="text-muted small mb-0")
                ], className="text-center")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.H5(str(completed_experiments), className="mb-0 text-success"),
                    html.P("Completed", className="text-muted small mb-0")
                ], className="text-center")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.H5(str(running_experiments), className="mb-0 text-primary"),
                    html.P("Running", className="text-muted small mb-0")
                ], className="text-center")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.H5(str(failed_experiments), className="mb-0 text-danger"),
                    html.P("Failed", className="text-muted small mb-0")
                ], className="text-center")
            ], width=3),
        ])

    @app.callback(
        Output('experiment-model-filter', 'options'),
        [Input('url', 'pathname')]
    )
    def load_model_filter_options(pathname):
        """Load unique model types for filter dropdown."""
        if pathname != '/experiments':
            raise PreventUpdate

        session = get_session()

        # Get unique model types
        model_types = session.query(Experiment.model_type).distinct().all()
        session.close()

        options = [{'label': mt[0], 'value': mt[0]} for mt in model_types if mt[0]]

        return options

    @app.callback(
        [Output('compare-experiments-btn', 'disabled'),
         Output('selected-experiments-store', 'data')],
        [Input('experiments-table', 'selected_rows'),
         Input('experiments-table', 'data')]
    )
    def handle_experiment_selection(selected_rows, table_data):
        """Handle experiment selection and enable/disable compare button."""
        if not selected_rows or not table_data:
            return True, []

        # Get selected experiment IDs
        selected_ids = [table_data[idx]['id'] for idx in selected_rows]

        # Enable button only if 2-3 experiments selected
        if 2 <= len(selected_ids) <= 3:
            return False, selected_ids
        else:
            return True, selected_ids

    @app.callback(
        Output('url', 'pathname', allow_duplicate=True),
        [Input('compare-experiments-btn', 'n_clicks')],
        [State('selected-experiments-store', 'data')],
        prevent_initial_call=True
    )
    def navigate_to_comparison(n_clicks, selected_ids):
        """Navigate to comparison page with selected experiments."""
        if not n_clicks or not selected_ids:
            raise PreventUpdate

        # Build comparison URL
        ids_str = ','.join(map(str, selected_ids))
        return f'/compare?ids={ids_str}'

    @app.callback(
        [Output('comparison-offcanvas', 'is_open'),
         Output('comparison-cart-content', 'children')],
        [Input('compare-experiments-btn', 'n_clicks'),
         Input('view-comparison-btn', 'n_clicks')],
        [State('comparison-offcanvas', 'is_open'),
         State('selected-experiments-store', 'data')]
    )
    def toggle_comparison_cart(compare_clicks, view_clicks, is_open, selected_ids):
        """Toggle comparison cart offcanvas."""
        if not callback_context.triggered:
            raise PreventUpdate

        trigger_id = callback_context.triggered[0]['prop_id'].split('.')[0]

        from dash import html
        import dash_bootstrap_components as dbc
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

        if trigger_id == 'compare-experiments-btn':
            if not selected_ids:
                raise PreventUpdate

            # Load selected experiments
            session = get_session()
            experiments = session.query(Experiment).filter(
                Experiment.id.in_(selected_ids)
            ).all()
            session.close()

            # Build cart content
            cart_items = []
            for exp in experiments:
                cart_items.append(
                    dbc.Card([
                        dbc.CardBody([
                            html.H6(f"{exp.name}", className="mb-1"),
                            html.P(f"ID: {exp.id} | Type: {exp.model_type}", className="small text-muted mb-0")
                        ])
                    ], className="mb-2")
                )

            return True, html.Div(cart_items)

        elif trigger_id == 'view-comparison-btn':
            return False, []

        raise PreventUpdate
