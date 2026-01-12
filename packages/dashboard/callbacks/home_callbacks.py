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
            Output('system-health-gauge', 'figure'),
            # Quick stats outputs
            Output('home-total-signals', 'children'),
            Output('home-fault-classes', 'children'),
            Output('home-best-accuracy', 'children'),
            Output('home-total-experiments', 'children'),
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

                # 2. Quick Stats - Total Signals
                total_signals = session.query(
                    func.sum(Dataset.num_signals)
                ).scalar() or 0
                signals_display = f"{total_signals:,}" if total_signals else "0"

                # 3. Quick Stats - Fault Classes
                datasets = session.query(Dataset).all()
                all_fault_types = set()
                for ds in datasets:
                    if ds.fault_types:
                        all_fault_types.update(ds.fault_types)
                fault_classes = len(all_fault_types) if all_fault_types else 11

                # 4. Quick Stats - Best Accuracy from completed experiments' metrics
                completed_experiments = session.query(Experiment).filter(
                    Experiment.status == ExperimentStatus.COMPLETED
                ).all()
                
                best_accuracy = "N/A"
                max_acc = 0.0
                for exp in completed_experiments:
                    if exp.metrics and isinstance(exp.metrics, dict):
                        acc = exp.metrics.get('accuracy', exp.metrics.get('test_accuracy', 0))
                        if acc and acc > max_acc:
                            max_acc = acc
                if max_acc > 0:
                    best_accuracy = f"{max_acc * 100:.1f}%" if max_acc <= 1 else f"{max_acc:.1f}%"

                # 5. Quick Stats - Total Experiments
                total_experiments = session.query(func.count(Experiment.id)).scalar() or 0

                # 6. Dataset Distribution Chart - Experiments by Status
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

                # 7. System Health Gauge
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

                logger.info(f"Home stats: {signals_display} signals, {fault_classes} classes, "
                           f"{best_accuracy} best, {total_experiments} experiments")

            return (
                recent_experiments,
                fig_dist,
                fig_gauge,
                signals_display,
                str(fault_classes),
                best_accuracy,
                str(total_experiments)
            )

        except Exception as e:
            logger.error(f"Error updating home dashboard: {e}", exc_info=True)
            return (
                html.P("Error loading data"),
                {},
                {},
                "Error",
                "Error", 
                "Error",
                "Error"
            )

    logger.info("Home page callbacks registered")
