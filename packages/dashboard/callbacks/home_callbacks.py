"""
Home page callbacks.
Populates the dashboard overview with real data.
"""
from dash import Input, Output, html, dcc
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
from sqlalchemy import func

from database.connection import get_db_session
from models.experiment import Experiment, ExperimentStatus
from models.dataset import Dataset
from utils.logger import setup_logger

logger = setup_logger(__name__)


def register_home_callbacks(app):
    """Register callbacks for the home page."""

    @app.callback(
        [
            Output('recent-experiments-list', 'children'),
            Output('dataset-distribution-chart', 'figure'),
            Output('system-health-gauge', 'figure')
        ],
        [Input('url', 'pathname')]
    )
    def update_home_dashboard(pathname):
        """Update home dashboard content."""
        if pathname != '/' and pathname is not None:
            raise PreventUpdate

        try:
            with get_db_session() as session:

                # 1. Fetch Recent Experiments
                experiments = session.query(Experiment).order_by(
                    Experiment.created_at.desc()
                ).limit(5).all()

                if experiments:
                    experiments_list = []
                    for exp in experiments:
                        # Determine color based on status
                        status_colors = {
                            'completed': 'success',
                            'running': 'primary',
                            'failed': 'danger',
                            'pending': 'warning'
                        }
                        color = status_colors.get(exp.status.value, 'secondary')

                        item = dbc.ListGroupItem([
                            html.Div([
                                html.H6(exp.name, className="mb-1"),
                                html.Small(exp.model_type, className="text-muted")
                            ], className="d-flex w-100 justify-content-between"),
                            html.P(f"Status: {exp.status.value}", className=f"mb-1 text-{color}"),
                            html.Small(
                                exp.created_at.strftime('%Y-%m-%d %H:%M') if exp.created_at else "",
                                className="text-muted"
                            )
                        ], action=True, href=f"/experiment/{exp.id}/monitor")
                        experiments_list.append(item)
                    
                    recent_experiments = dbc.ListGroup(experiments_list, flush=True)
                else:
                    recent_experiments = html.P("No experiments found. Start training!", className="text-muted text-center py-3")

            # 2. Dataset Distribution Chart
            # Count signals per class across all datasets
            # For simplicity in this specialized query, we might just count datasets by type or class
            # Let's count experiments by model_type for this chart instead, as it's more interesting for now
            # OR count samples per class if available.
            # Let's do Experiments by Status
            status_counts = session.query(
                Experiment.status, func.count(Experiment.id)
            ).group_by(Experiment.status).all()
            
            if status_counts:
                data = {'Status': [s[0].value for s in status_counts], 'Count': [s[1] for s in status_counts]}
                fig_dist = px.pie(
                    data, 
                    values='Count', 
                    names='Status', 
                    title='Experiment Status Distribution',
                    hole=0.4,
                    color='Status',
                    color_discrete_map={
                        'completed': '#28a745',
                        'running': '#007bff',
                        'failed': '#dc3545',
                        'pending': '#ffc107',
                        'cancelled': '#6c757d'
                    }
                )
                fig_dist.update_layout(margin=dict(t=30, b=0, l=0, r=0), height=250)
            else:
                fig_dist = {}

            # 3. System Health Gauge (Mock or Real)
            # Since we have the system health callbacks, we could reuse or just show a simple static one
            # Ideally this should be updated by the interval, but let's just initialize it
            fig_gauge = {} # Let the specific component callback handle real-time updates if we had one.
            # But wait, layouts/home.py defined 'system-health-gauge'.
            # system_health_callbacks.py updates 'cpu-usage-gauge' etc.
            # So home.py's gauge is currently orphan.
            # Let's explicitly return a placeholder or simple figure
            
            fig_gauge = {
                "data": [{
                    "type": "indicator",
                    "mode": "gauge+number",
                    "value": 100, 
                    "title": {"text": "System Online"},
                    "gauge": {"axis": {"range": [0, 100]}, "bar": {"color": "green"}}
                }],
                "layout": {"height": 200, "margin": dict(t=0, b=0, l=0, r=0)}
            }



            return recent_experiments, fig_dist, fig_gauge

        except Exception as e:
            logger.error(f"Error updating home dashboard: {e}", exc_info=True)
            return html.P("Error loading data"), {}, {}
