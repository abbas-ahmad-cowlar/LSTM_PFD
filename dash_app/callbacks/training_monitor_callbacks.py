"""
Training monitor callbacks (Phase 11B).
Real-time updates for training progress.
"""
from dash import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import html

from database.connection import get_db_session
from models.experiment import Experiment, ExperimentStatus
from models.training_run import TrainingRun
from layouts.training_monitor import format_time_duration, create_metric_display
from tasks.training_tasks import train_model_task
from celery.result import AsyncResult
from utils.logger import setup_logger
import time

logger = setup_logger(__name__)


def register_training_monitor_callbacks(app):
    """Register all training monitor callbacks."""

    @app.callback(
        [Output("experiment-title", "children"),
         Output("experiment-subtitle", "children")],
        Input("experiment-id-store", "data")
    )
    def load_experiment_info(experiment_id):
        """Load and display experiment information."""
        if not experiment_id:
            raise PreventUpdate

        try:
            with get_db_session() as session:
                experiment = session.query(Experiment).filter_by(id=experiment_id).first()
                if not experiment:
                    return "Experiment Not Found", ""

                title = f"Experiment: {experiment.name}"
                subtitle = f"Model: {experiment.model_type.upper()} | Status: {experiment.status.value.upper()}"
                return title, subtitle
        except Exception as e:
            logger.error(f"Error loading experiment info: {e}")
            return "Error Loading Experiment", str(e)

    @app.callback(
        Output("training-data-store", "data"),
        Input("training-update-interval", "n_intervals"),
        State("experiment-id-store", "data")
    )
    def fetch_training_data(n_intervals, experiment_id):
        """Fetch latest training data from database and Celery."""
        if not experiment_id:
            raise PreventUpdate

        try:
            with get_db_session() as session:
                experiment = session.query(Experiment).filter_by(id=experiment_id).first()
                if not experiment:
                    return {}

                # Get all training runs (per-epoch data)
                training_runs = session.query(TrainingRun).filter_by(
                    experiment_id=experiment_id
                ).order_by(TrainingRun.epoch).all()

                # Get Celery task status if running
                task_status = {}
                if experiment.config.get("celery_task_id"):
                    task = AsyncResult(experiment.config["celery_task_id"])
                    task_status = {
                        "state": task.state,
                        "info": task.info if task.info else {}
                    }

                return {
                    "experiment": {
                        "id": experiment.id,
                        "name": experiment.name,
                        "status": experiment.status.value,
                        "model_type": experiment.model_type,
                        "total_epochs": experiment.total_epochs or experiment.config.get("num_epochs", 100),
                        "best_epoch": experiment.best_epoch,
                        "duration_seconds": experiment.duration_seconds,
                    },
                    "training_runs": [
                        {
                            "epoch": run.epoch,
                            "train_loss": run.train_loss,
                            "val_loss": run.val_loss,
                            "train_accuracy": run.train_accuracy,
                            "val_accuracy": run.val_accuracy,
                            "learning_rate": run.learning_rate,
                            "duration_seconds": run.duration_seconds,
                        }
                        for run in training_runs
                    ],
                    "task_status": task_status,
                    "timestamp": time.time()
                }
        except Exception as e:
            logger.error(f"Error fetching training data: {e}")
            return {}

    @app.callback(
        [Output("epoch-progress-text", "children"),
         Output("epoch-progress-bar", "value"),
         Output("overall-progress-text", "children"),
         Output("overall-progress-bar", "value"),
         Output("status-badge", "children"),
         Output("status-badge", "color"),
         Output("time-elapsed-badge", "children"),
         Output("eta-badge", "children")],
        Input("training-data-store", "data")
    )
    def update_progress_indicators(data):
        """Update progress bars and status badges."""
        if not data or "experiment" not in data:
            raise PreventUpdate

        exp = data["experiment"]
        runs = data.get("training_runs", [])
        task_status = data.get("task_status", {})

        # Current epoch
        current_epoch = len(runs)
        total_epochs = exp["total_epochs"]

        # Epoch progress (within current epoch, from task status)
        epoch_progress = task_status.get("info", {}).get("epoch_progress", 100)
        epoch_text = f"Epoch {current_epoch}/{total_epochs}"

        # Overall progress
        overall_progress = (current_epoch / total_epochs * 100) if total_epochs > 0 else 0
        overall_text = f"Overall: {overall_progress:.1f}%"

        # Status badge
        status = exp["status"]
        status_colors = {
            "pending": "secondary",
            "running": "primary",
            "paused": "warning",
            "completed": "success",
            "failed": "danger",
            "cancelled": "dark"
        }
        status_badge_color = status_colors.get(status, "secondary")
        status_badge_text = status.upper()

        # Time elapsed
        duration = exp.get("duration_seconds", 0)
        time_elapsed = f"Elapsed: {format_time_duration(duration)}"

        # ETA calculation
        if current_epoch > 0 and status == "running":
            avg_epoch_time = duration / current_epoch
            remaining_epochs = total_epochs - current_epoch
            eta_seconds = avg_epoch_time * remaining_epochs
            eta_text = f"ETA: {format_time_duration(eta_seconds)}"
        else:
            eta_text = "ETA: N/A"

        return (
            epoch_text, epoch_progress,
            overall_text, overall_progress,
            status_badge_text, status_badge_color,
            time_elapsed, eta_text
        )

    @app.callback(
        [Output("current-metrics-display", "children"),
         Output("best-metrics-display", "children")],
        Input("training-data-store", "data")
    )
    def update_metrics_display(data):
        """Update current and best metrics displays."""
        if not data or "training_runs" not in data or not data["training_runs"]:
            empty = html.P("No data yet", className="text-muted")
            return empty, empty

        runs = data["training_runs"]
        current_run = runs[-1]
        best_epoch = data["experiment"].get("best_epoch", 1)
        best_run = runs[best_epoch - 1] if best_epoch <= len(runs) else current_run

        # Current metrics
        current_metrics = html.Div([
            create_metric_display("Train Loss", current_run["train_loss"], ".4f", "danger"),
            create_metric_display("Val Loss", current_run["val_loss"], ".4f", "warning"),
            create_metric_display("Train Acc", current_run["train_accuracy"], ".2%", "info"),
            create_metric_display("Val Acc", current_run["val_accuracy"], ".2%", "success"),
        ])

        # Best metrics
        best_metrics = html.Div([
            create_metric_display("Best Val Loss", best_run["val_loss"], ".4f", "warning"),
            create_metric_display("Best Val Acc", best_run["val_accuracy"], ".2%", "success"),
            create_metric_display("Best Epoch", best_epoch, "d", "primary"),
        ])

        return current_metrics, best_metrics

    @app.callback(
        Output("loss-curve", "figure"),
        Input("training-data-store", "data")
    )
    def update_loss_curve(data):
        """Update loss curves plot."""
        if not data or "training_runs" not in data or not data["training_runs"]:
            return go.Figure()

        runs = data["training_runs"]
        epochs = [r["epoch"] for r in runs]
        train_loss = [r["train_loss"] for r in runs]
        val_loss = [r["val_loss"] for r in runs]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode="lines+markers",
                                 name="Train Loss", line=dict(color="#dc3545")))
        fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines+markers",
                                 name="Val Loss", line=dict(color="#ffc107")))

        fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Loss",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=40, b=40),
            height=300
        )

        return fig

    @app.callback(
        Output("accuracy-curve", "figure"),
        Input("training-data-store", "data")
    )
    def update_accuracy_curve(data):
        """Update accuracy curves plot."""
        if not data or "training_runs" not in data or not data["training_runs"]:
            return go.Figure()

        runs = data["training_runs"]
        epochs = [r["epoch"] for r in runs]
        train_acc = [r["train_accuracy"] * 100 for r in runs]
        val_acc = [r["val_accuracy"] * 100 for r in runs]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=train_acc, mode="lines+markers",
                                 name="Train Accuracy", line=dict(color="#17a2b8")))
        fig.add_trace(go.Scatter(x=epochs, y=val_acc, mode="lines+markers",
                                 name="Val Accuracy", line=dict(color="#28a745")))

        fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Accuracy (%)",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=40, b=40),
            height=300
        )

        return fig

    @app.callback(
        Output("lr-schedule", "figure"),
        Input("training-data-store", "data")
    )
    def update_lr_schedule(data):
        """Update learning rate schedule plot."""
        if not data or "training_runs" not in data or not data["training_runs"]:
            return go.Figure()

        runs = data["training_runs"]
        epochs = [r["epoch"] for r in runs]
        lrs = [r.get("learning_rate", 0) for r in runs]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=lrs, mode="lines+markers",
                                 line=dict(color="#6f42c1")))

        fig.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Learning Rate",
            yaxis_type="log",
            hovermode="x",
            margin=dict(l=40, r=40, t=40, b=40),
            height=300,
            showlegend=False
        )

        return fig

    @app.callback(
        Output("training-logs", "children"),
        Input("training-data-store", "data")
    )
    def update_training_logs(data):
        """Update training logs display."""
        if not data or "training_runs" not in data or not data["training_runs"]:
            return "Waiting for training to start..."

        runs = data["training_runs"]
        log_lines = []

        for run in runs[-10:]:  # Show last 10 epochs
            log_lines.append(
                f"[Epoch {run['epoch']:3d}] "
                f"Train Loss: {run['train_loss']:.4f} | "
                f"Val Loss: {run['val_loss']:.4f} | "
                f"Val Acc: {run['val_accuracy']:.2%} | "
                f"LR: {run.get('learning_rate', 0):.2e}\n"
            )

        return "".join(log_lines)
