"""
Experiment Comparison — Plotly visualization helpers.
Pure functions that create Plotly figure objects for comparison charts.
"""
import plotly.graph_objects as go
from dashboard_config import FAULT_CLASSES


def create_overall_metrics_chart(experiments):
    """Create bar chart comparing overall metrics."""
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    fig = go.Figure()

    for exp in experiments:
        values = [exp['metrics'][m] * 100 for m in metrics]
        fig.add_trace(go.Bar(
            name=f"Exp {exp['id']}: {exp['name']}",
            x=[m.replace('_', ' ').title() for m in metrics],
            y=values,
            text=[f"{v:.2f}%" for v in values],
            textposition='auto',
        ))

    fig.update_layout(
        title="Overall Performance Metrics",
        xaxis_title="Metric",
        yaxis_title="Score (%)",
        barmode='group',
        yaxis=dict(range=[0, 105]),
        height=400,
        hovermode='x unified'
    )

    return fig


def create_per_class_f1_chart(experiments):
    """Create grouped bar chart for per-class F1 scores."""
    fig = go.Figure()

    for exp in experiments:
        f1_scores = []
        for fault_class in FAULT_CLASSES:
            class_metrics = exp['per_class_metrics'].get(fault_class, {})
            f1 = class_metrics.get('f1', 0) * 100
            f1_scores.append(f1)

        fig.add_trace(go.Bar(
            name=f"Exp {exp['id']}",
            x=[fc.replace('_', ' ').title() for fc in FAULT_CLASSES],
            y=f1_scores,
            text=[f"{v:.1f}%" for v in f1_scores],
            textposition='auto',
        ))

    fig.update_layout(
        title="Per-Class F1 Scores",
        xaxis_title="Fault Class",
        yaxis_title="F1 Score (%)",
        barmode='group',
        yaxis=dict(range=[0, 105]),
        height=500,
        hovermode='x unified',
        xaxis={'tickangle': -45}
    )

    return fig


def create_confusion_matrix_heatmap(experiment):
    """Create confusion matrix heatmap for an experiment."""
    cm = experiment['confusion_matrix']

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[fc.replace('_', ' ').title() for fc in FAULT_CLASSES],
        y=[fc.replace('_', ' ').title() for fc in FAULT_CLASSES],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 10},
        hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
    ))

    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        xaxis={'tickangle': -45, 'side': 'bottom'},
        yaxis={'autorange': 'reversed'}
    )

    return fig


def create_training_curves_comparison(experiments, metric_type='loss'):
    """
    Create training curves comparison.

    Args:
        experiments: List of experiment data
        metric_type: 'loss' or 'accuracy'
    """
    fig = go.Figure()

    for exp in experiments:
        history = exp['training_history']
        epochs = history['epochs']

        if metric_type == 'loss':
            # Plot train and val loss
            fig.add_trace(go.Scatter(
                x=epochs,
                y=history['train_loss'],
                mode='lines',
                name=f"Exp {exp['id']} (Train)",
                line=dict(dash='solid')
            ))
            fig.add_trace(go.Scatter(
                x=epochs,
                y=history['val_loss'],
                mode='lines',
                name=f"Exp {exp['id']} (Val)",
                line=dict(dash='dash')
            ))
        else:  # accuracy
            fig.add_trace(go.Scatter(
                x=epochs,
                y=[v * 100 for v in history['val_accuracy']],
                mode='lines',
                name=f"Exp {exp['id']}",
            ))

    title = "Training & Validation Loss" if metric_type == 'loss' else "Validation Accuracy"
    yaxis_title = "Loss" if metric_type == 'loss' else "Accuracy (%)"

    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title=yaxis_title,
        height=400,
        hovermode='x unified'
    )

    return fig
