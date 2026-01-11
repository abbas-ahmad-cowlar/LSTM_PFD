"""
Experiment Comparison Dashboard (Feature #2).
Side-by-side comparison of 2-3 experiments with statistical analysis.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dashboard_config import FAULT_CLASSES
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


def create_experiment_comparison_layout(experiment_ids):
    """
    Create comparison page layout.

    Args:
        experiment_ids: List of experiment IDs from URL (e.g., [1234, 1567, 1890])
    """
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🔍 Experiment Comparison", className="mb-2"),
                html.P(
                    f"Comparing {len(experiment_ids)} experiments",
                    className="text-muted"
                )
            ], width=8),
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button(
                        [html.I(className="fas fa-file-pdf me-2"), "Export PDF"],
                        id='export-comparison-pdf',
                        color="secondary",
                        size="sm"
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-link me-2"), "Share Link"],
                        id='share-comparison-link',
                        color="info",
                        size="sm"
                    ),
                ], className="float-end")
            ], width=4)
        ], className="mb-4"),

        # Breadcrumb navigation
        dbc.Row([
            dbc.Col([
                dbc.Breadcrumb(items=[
                    {"label": "Experiments", "href": "/experiments", "active": False},
                    {"label": "Comparison", "active": True},
                ])
            ])
        ], className="mb-3"),

        # Loading indicator
        dcc.Loading(
            id="comparison-loading",
            type="default",
            children=[
                # Hidden store for comparison data
                dcc.Store(id='comparison-data-store', data={'experiment_ids': experiment_ids}),

                # Main content tabs
                dbc.Tabs(id='comparison-tabs', active_tab='overview', children=[
                    dbc.Tab(label="📊 Overview", tab_id="overview"),
                    dbc.Tab(label="📈 Metrics", tab_id="metrics"),
                    dbc.Tab(label="📉 Visualizations", tab_id="visualizations"),
                    dbc.Tab(label="🔬 Statistical Tests", tab_id="statistical"),
                    dbc.Tab(label="⚙️ Configuration", tab_id="configuration")
                ], className="mb-4"),

                # Tab content container
                html.Div(id='comparison-tab-content', className="mt-4")
            ]
        ),

        # Modals
        create_share_link_modal(),

    ], fluid=True, className="py-4")


def create_overview_tab(comparison_data):
    """
    Overview tab: High-level summary of compared experiments.

    Args:
        comparison_data: Dictionary from ComparisonService.get_comparison_data()
    """
    experiments = comparison_data['experiments']

    if not experiments:
        return dbc.Alert("No experiment data available", color="warning")

    # Determine winner (highest accuracy)
    best_exp = max(experiments, key=lambda e: e['metrics']['accuracy'])

    return html.Div([
        # Winner announcement
        dbc.Alert([
            html.H4("🏆 Winner", className="alert-heading"),
            html.P(f"Experiment #{best_exp['id']}: {best_exp['name']}", className="mb-1"),
            html.P(
                f"Accuracy: {best_exp['metrics']['accuracy']:.2%}",
                className="mb-0",
                style={'fontSize': '1.2em', 'fontWeight': 'bold'}
            )
        ], color="success", className="mb-4"),

        # Summary cards (one per experiment)
        dbc.Row([
            dbc.Col([
                create_experiment_summary_card(exp, rank=idx+1)
            ], width=12 // len(experiments))
            for idx, exp in enumerate(
                sorted(experiments, key=lambda e: e['metrics']['accuracy'], reverse=True)
            )
        ], className="mb-4"),

        # Quick metrics comparison table
        html.H4("Quick Metrics Comparison", className="mb-3"),
        create_metrics_comparison_table(experiments),

        # Key differences summary
        html.H4("Key Differences", className="mt-4 mb-3"),
        html.Div(id='key-differences-content')
    ])


def create_experiment_summary_card(experiment, rank):
    """
    Card showing summary of single experiment.

    Args:
        experiment: Experiment data dictionary
        rank: Position in ranking (1, 2, 3)
    """
    # Medal emoji based on rank
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    medal = medals.get(rank, "")

    # Badge color based on rank
    badge_colors = {1: "success", 2: "info", 3: "warning"}
    badge_color = badge_colors.get(rank, "secondary")

    duration_str = format_duration(experiment['duration_seconds']) if experiment['duration_seconds'] else "N/A"

    return dbc.Card([
        dbc.CardHeader([
            html.Span(medal, className="me-2", style={'fontSize': '1.5em'}),
            dbc.Badge(f"Rank #{rank}", color=badge_color, className="float-end")
        ]),
        dbc.CardBody([
            html.H5(f"{experiment['name']}", className="card-title"),
            html.P(
                f"ID: {experiment['id']} | Type: {experiment['model_type']}",
                className="text-muted small"
            ),
            html.Hr(),
            html.Div([
                create_metric_row("Accuracy", experiment['metrics']['accuracy'], format_pct=True),
                create_metric_row("F1-Score", experiment['metrics']['f1_score'], format_pct=True),
                create_metric_row("Precision", experiment['metrics']['precision'], format_pct=True),
                create_metric_row("Recall", experiment['metrics']['recall'], format_pct=True),
                create_metric_row("Duration", duration_str)
            ])
        ])
    ], className="mb-3 shadow-sm")


def create_metric_row(label, value, format_pct=False):
    """Helper to create a metric row in card."""
    if format_pct and isinstance(value, (int, float)):
        value_str = f"{value:.2%}"
    else:
        value_str = str(value)

    return html.Div([
        html.Span(label, className="text-muted"),
        html.Span(value_str, className="float-end", style={'fontWeight': '500'})
    ], className="mb-2")


def create_metrics_comparison_table(experiments):
    """
    Table comparing all metrics side-by-side.
    """
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score']

    table_header = [
        html.Thead(html.Tr([
            html.Th("Metric"),
            *[html.Th(f"Exp {exp['id']}", style={'textAlign': 'center'}) for exp in experiments],
            html.Th("Best", style={'textAlign': 'center'})
        ]))
    ]

    table_rows = []
    for metric in metrics_to_compare:
        values = [exp['metrics'][metric] for exp in experiments]
        best_value = max(values)

        row = html.Tr([
            html.Td(metric.replace('_', ' ').title()),
            *[
                html.Td(
                    f"{val:.2%}",
                    className="font-weight-bold text-success" if val == best_value else "",
                    style={'textAlign': 'center'}
                )
                for val in values
            ],
            html.Td(f"{best_value:.2%}", className="text-success font-weight-bold", style={'textAlign': 'center'})
        ])
        table_rows.append(row)

    table_body = [html.Tbody(table_rows)]

    return dbc.Table(
        table_header + table_body,
        bordered=True,
        hover=True,
        responsive=True,
        striped=True,
        className="shadow-sm"
    )


def create_metrics_tab(comparison_data):
    """
    Metrics tab: Detailed per-class metrics comparison.
    """
    experiments = comparison_data['experiments']

    return html.Div([
        html.H4("Overall Metrics Comparison", className="mb-3"),

        # Overall metrics chart
        dcc.Graph(
            id='overall-metrics-chart',
            figure=create_overall_metrics_chart(experiments)
        ),

        html.Hr(className="my-4"),

        html.H4("Per-Class Performance", className="mb-3"),

        # Per-class metrics table
        create_per_class_metrics_table(experiments),

        html.Hr(className="my-4"),

        # Per-class F1 scores chart
        dcc.Graph(
            id='per-class-f1-chart',
            figure=create_per_class_f1_chart(experiments)
        )
    ])


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


def create_per_class_metrics_table(experiments):
    """Create table showing per-class metrics for all experiments."""
    # Build table data
    table_data = []

    for fault_class in FAULT_CLASSES:
        row = {'fault_class': fault_class.replace('_', ' ').title()}

        for exp in experiments:
            class_metrics = exp['per_class_metrics'].get(fault_class, {})
            f1 = class_metrics.get('f1', 0)
            recall = class_metrics.get('recall', 0)
            precision = class_metrics.get('precision', 0)

            row[f"exp_{exp['id']}_f1"] = f"{f1:.2%}" if f1 else "N/A"
            row[f"exp_{exp['id']}_recall"] = f"{recall:.2%}" if recall else "N/A"
            row[f"exp_{exp['id']}_precision"] = f"{precision:.2%}" if precision else "N/A"

        table_data.append(row)

    # Build columns
    columns = [{"name": "Fault Class", "id": "fault_class"}]

    for exp in experiments:
        columns.extend([
            {"name": f"Exp {exp['id']} F1", "id": f"exp_{exp['id']}_f1"},
            {"name": f"Exp {exp['id']} Recall", "id": f"exp_{exp['id']}_recall"},
            {"name": f"Exp {exp['id']} Precision", "id": f"exp_{exp['id']}_precision"},
        ])

    return dash_table.DataTable(
        data=table_data,
        columns=columns,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontSize': '14px',
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'column_id': 'fault_class'},
                'fontWeight': 'bold'
            }
        ]
    )


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


def create_visualizations_tab(comparison_data):
    """
    Visualizations tab: Confusion matrices and training curves.
    """
    experiments = comparison_data['experiments']

    return html.Div([
        html.H4("Confusion Matrices", className="mb-3"),

        # Confusion matrices side-by-side
        dbc.Row([
            dbc.Col([
                html.H6(f"Exp {exp['id']}: {exp['name']}", className="text-center mb-2"),
                dcc.Graph(
                    figure=create_confusion_matrix_heatmap(exp)
                )
            ], width=12 // len(experiments))
            for exp in experiments
        ], className="mb-4"),

        html.Hr(className="my-4"),

        html.H4("Training History", className="mb-3"),

        # Training curves comparison
        dbc.Row([
            dbc.Col([
                dcc.Graph(
                    id='training-loss-comparison',
                    figure=create_training_curves_comparison(experiments, 'loss')
                )
            ], width=6),
            dbc.Col([
                dcc.Graph(
                    id='training-accuracy-comparison',
                    figure=create_training_curves_comparison(experiments, 'accuracy')
                )
            ], width=6),
        ])
    ])


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


def create_statistical_tab(comparison_data):
    """
    Statistical tests tab: McNemar's or Friedman test results.
    """
    statistical_tests = comparison_data.get('statistical_tests', {})
    experiments = comparison_data['experiments']

    if not statistical_tests:
        return dbc.Alert("No statistical tests available", color="info")

    # Check which test was run
    if 'mcnemar' in statistical_tests:
        return create_mcnemar_results(statistical_tests['mcnemar'], experiments)
    elif 'friedman' in statistical_tests:
        return create_friedman_results(statistical_tests['friedman'], experiments)
    else:
        return dbc.Alert("No statistical tests performed", color="warning")


def create_mcnemar_results(mcnemar_data, experiments):
    """Create McNemar's test results display."""
    if 'error' in mcnemar_data:
        return dbc.Alert(
            [
                html.H4("Statistical Test Unavailable", className="alert-heading"),
                html.P(mcnemar_data['error']),
                html.P(mcnemar_data['interpretation'])
            ],
            color="warning"
        )

    # Determine alert color based on significance
    alert_color = "success" if mcnemar_data['significant'] else "info"

    return html.Div([
        dbc.Alert([
            html.H4("McNemar's Test", className="alert-heading"),
            html.P(mcnemar_data['interpretation'], className="mb-0")
        ], color=alert_color, className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Test Statistics"),
                    dbc.CardBody([
                        create_metric_row("Test Statistic (χ²)", f"{mcnemar_data['test_statistic']:.4f}"),
                        create_metric_row("P-value", f"{mcnemar_data['p_value']:.4f}"),
                        create_metric_row("Significant (α=0.05)?", "Yes" if mcnemar_data['significant'] else "No"),
                    ])
                ], className="shadow-sm")
            ], width=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Contingency Table"),
                    dbc.CardBody([
                        create_contingency_table(mcnemar_data['contingency_table'], experiments)
                    ])
                ], className="shadow-sm")
            ], width=6),
        ], className="mb-4"),

        # Explanation
        dbc.Card([
            dbc.CardHeader("What is McNemar's Test?"),
            dbc.CardBody([
                html.P([
                    "McNemar's test is a statistical test used to determine if two paired models have significantly different error rates. ",
                    "It analyzes the cases where the models disagree (one correct, one wrong)."
                ]),
                html.P([
                    html.Strong("Interpretation: "),
                    "A p-value < 0.05 indicates a statistically significant difference between the models. ",
                    "This means the performance difference is unlikely due to random chance."
                ], className="mb-0")
            ])
        ], className="shadow-sm")
    ])


def create_contingency_table(contingency_table, experiments):
    """Create contingency table visualization."""
    if not contingency_table:
        return html.P("N/A")

    [[a, b], [c, d]] = contingency_table

    table_header = [
        html.Thead(html.Tr([
            html.Th(""),
            html.Th(f"Exp {experiments[1]['id']} Correct", style={'textAlign': 'center'}),
            html.Th(f"Exp {experiments[1]['id']} Wrong", style={'textAlign': 'center'}),
        ]))
    ]

    table_body = [
        html.Tbody([
            html.Tr([
                html.Th(f"Exp {experiments[0]['id']} Correct"),
                html.Td(str(a), style={'textAlign': 'center'}),
                html.Td(str(b), style={'textAlign': 'center', 'fontWeight': 'bold'}),
            ]),
            html.Tr([
                html.Th(f"Exp {experiments[0]['id']} Wrong"),
                html.Td(str(c), style={'textAlign': 'center', 'fontWeight': 'bold'}),
                html.Td(str(d), style={'textAlign': 'center'}),
            ]),
        ])
    ]

    return dbc.Table(
        table_header + table_body,
        bordered=True,
        className="mb-0"
    )


def create_friedman_results(friedman_data, experiments):
    """Create Friedman test results display."""
    if 'error' in friedman_data:
        return dbc.Alert(
            [
                html.H4("Statistical Test Unavailable", className="alert-heading"),
                html.P(friedman_data['error']),
                html.P(friedman_data['interpretation'])
            ],
            color="warning"
        )

    alert_color = "success" if friedman_data['significant'] else "info"

    return html.Div([
        dbc.Alert([
            html.H4("Friedman Test", className="alert-heading"),
            html.P(friedman_data['interpretation'], className="mb-0")
        ], color=alert_color, className="mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Test Statistics"),
                    dbc.CardBody([
                        create_metric_row("Test Statistic (χ²)", f"{friedman_data['test_statistic']:.4f}"),
                        create_metric_row("P-value", f"{friedman_data['p_value']:.4f}"),
                        create_metric_row("Significant (α=0.05)?", "Yes" if friedman_data['significant'] else "No"),
                    ])
                ], className="shadow-sm")
            ], width=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Rankings"),
                    dbc.CardBody([
                        create_rankings_table(friedman_data['rankings'], experiments)
                    ])
                ], className="shadow-sm")
            ], width=6),
        ], className="mb-4"),

        # Explanation
        dbc.Card([
            dbc.CardHeader("What is Friedman Test?"),
            dbc.CardBody([
                html.P([
                    "The Friedman test is a non-parametric test used to detect differences in performance across three or more paired models. ",
                    "It ranks the models for each test sample and compares average rankings."
                ]),
                html.P([
                    html.Strong("Interpretation: "),
                    "A p-value < 0.05 indicates at least one model performs significantly differently from the others. ",
                    "Lower average rank = better performance (1 is best)."
                ], className="mb-0")
            ])
        ], className="shadow-sm")
    ])


def create_rankings_table(rankings, experiments):
    """Create rankings table."""
    table_rows = []

    for idx, exp in enumerate(experiments):
        rank_value = rankings[idx]

        table_rows.append(html.Tr([
            html.Td(f"Exp {exp['id']}: {exp['name']}"),
            html.Td(f"{rank_value:.2f}", style={'textAlign': 'center', 'fontWeight': 'bold'}),
        ]))

    return dbc.Table([
        html.Thead(html.Tr([
            html.Th("Experiment"),
            html.Th("Avg Rank", style={'textAlign': 'center'}),
        ])),
        html.Tbody(table_rows)
    ], bordered=True, className="mb-0")


def create_configuration_tab(comparison_data):
    """
    Configuration tab: Compare hyperparameters and config settings.
    """
    experiments = comparison_data['experiments']

    return html.Div([
        html.H4("Configuration Comparison", className="mb-3"),

        # Config comparison table
        create_config_comparison_table(experiments)
    ])


def create_config_comparison_table(experiments):
    """Create table comparing configurations."""
    # Extract all unique config keys
    all_keys = set()
    for exp in experiments:
        all_keys.update(exp['config'].keys())

    # Build table data
    table_data = []
    for key in sorted(all_keys):
        row = {'config_key': key}

        for exp in experiments:
            value = exp['config'].get(key, 'N/A')
            # Convert to string representation
            if isinstance(value, (dict, list)):
                value_str = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
            else:
                value_str = str(value)

            row[f"exp_{exp['id']}"] = value_str

        table_data.append(row)

    # Build columns
    columns = [{"name": "Configuration", "id": "config_key"}]
    for exp in experiments:
        columns.append({"name": f"Exp {exp['id']}", "id": f"exp_{exp['id']}"})

    return dash_table.DataTable(
        data=table_data,
        columns=columns,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontSize': '14px',
            'whiteSpace': 'normal',
            'height': 'auto',
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        },
        style_data_conditional=[
            {
                'if': {'column_id': 'config_key'},
                'fontWeight': 'bold',
                'width': '30%'
            }
        ],
        page_size=20,
    )


def create_share_link_modal():
    """Create modal for sharing comparison link."""
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Share Comparison")),
        dbc.ModalBody([
            html.P("Copy this link to share the comparison:"),
            dbc.Input(
                id='comparison-share-link-input',
                type='text',
                readonly=True,
                className="mb-3"
            ),
            dbc.Button(
                [html.I(className="fas fa-copy me-2"), "Copy to Clipboard"],
                id='copy-comparison-link-btn',
                color="primary",
                className="w-100"
            )
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-share-modal", className="ms-auto")
        ),
    ], id="share-link-modal", is_open=False)


def format_duration(seconds):
    """Format duration in human-readable form."""
    if not seconds:
        return "N/A"

    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"
