"""
Hyperparameter Optimization Campaigns (Phase 11C).
UI for creating and monitoring HPO campaigns.

Enhanced with:
- Resume functionality (HPO-5)
- Export functionality (HPO-6)
- Parallel Coordinates visualization (HPO-7)
- Parameter Importance chart (HPO-8)
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


def create_hpo_campaigns_layout():
    """Create HPO campaigns management layout."""
    return dbc.Container([
        html.H2("Hyperparameter Optimization Campaigns", className="mb-4"),

        # Actions row
        dbc.Row([
            dbc.Col([
                html.H5("Active Campaigns", className="text-muted"),
            ], width=6),
            dbc.Col([
                dbc.Button([html.I(className="fas fa-plus me-2"), "New Campaign"],
                           id="new-hpo-campaign-btn", color="primary", className="float-end"),
            ], width=6),
        ], className="mb-4"),

        # Campaigns list
        dbc.Row([
            dbc.Col([
                html.Div(id="hpo-campaigns-list")
            ])
        ], className="mb-4"),

        # Visualization Section (shown when a campaign is selected)
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([html.I(className="fas fa-chart-line me-2"), "Campaign Visualizations"], className="mb-0")
                    ]),
                    dbc.CardBody([
                        dbc.Tabs([
                            dbc.Tab(label="Parallel Coordinates", tab_id="parallel-coords-tab", children=[
                                html.Div([
                                    html.P("Visualize hyperparameter relationships across all trials", className="text-muted mt-2"),
                                    dcc.Graph(id="hpo-parallel-coords-chart", style={"height": "400px"})
                                ])
                            ]),
                            dbc.Tab(label="Parameter Importance", tab_id="param-importance-tab", children=[
                                html.Div([
                                    html.P("See which hyperparameters have the most impact on performance", className="text-muted mt-2"),
                                    dcc.Graph(id="hpo-param-importance-chart", style={"height": "400px"})
                                ])
                            ]),
                            dbc.Tab(label="Optimization History", tab_id="opt-history-tab", children=[
                                html.Div([
                                    html.P("Track the optimization progress over trials", className="text-muted mt-2"),
                                    dcc.Graph(id="hpo-optimization-history-chart", style={"height": "400px"})
                                ])
                            ]),
                        ], id="hpo-viz-tabs", active_tab="parallel-coords-tab")
                    ])
                ], id="hpo-viz-card", className="shadow-sm", style={"display": "none"})
            ])
        ], className="mb-4"),

        # Campaign creation modal
        dbc.Modal([
            dbc.ModalHeader("Create HPO Campaign"),
            dbc.ModalBody([
                dbc.Label("Campaign Name"),
                dbc.Input(id="hpo-campaign-name", placeholder="e.g., resnet_optimization_v1", className="mb-3"),

                dbc.Label("Model Type"),
                dcc.Dropdown(
                    id="hpo-model-type",
                    options=[
                        {"label": "ResNet-18", "value": "resnet18"},
                        {"label": "ResNet-34", "value": "resnet34"},
                        {"label": "Transformer", "value": "transformer"},
                        {"label": "EfficientNet", "value": "efficientnet"},
                    ],
                    className="mb-3"
                ),

                dbc.Label("Optimization Method"),
                dcc.Dropdown(
                    id="hpo-method",
                    options=[
                        {"label": "Bayesian Optimization (Recommended)", "value": "bayesian"},
                        {"label": "Random Search", "value": "random"},
                        {"label": "Grid Search", "value": "grid"},
                        {"label": "Hyperband", "value": "hyperband"},
                    ],
                    value="bayesian",
                    className="mb-3"
                ),

                dbc.Label("Number of Trials"),
                dbc.Input(type="number", id="hpo-num-trials", value=50, min=10, max=200, className="mb-3"),

                dbc.Label("Optimization Metric"),
                dcc.Dropdown(
                    id="hpo-metric",
                    options=[
                        {"label": "Validation Accuracy (maximize)", "value": "val_accuracy"},
                        {"label": "Validation Loss (minimize)", "value": "val_loss"},
                        {"label": "F1 Score (maximize)", "value": "f1_score"},
                    ],
                    value="val_accuracy",
                    className="mb-3"
                ),

                html.Hr(),

                html.H6("Search Space"),
                html.Div(id="hpo-search-space-inputs"),
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="hpo-cancel-btn", color="secondary"),
                dbc.Button("Launch Campaign", id="hpo-launch-btn", color="success"),
            ])
        ], id="hpo-modal", size="lg", is_open=False),

        # Export modal
        dbc.Modal([
            dbc.ModalHeader("Export Campaign Results"),
            dbc.ModalBody([
                html.P("Select export format for campaign results:"),
                dbc.RadioItems(
                    id="hpo-export-format",
                    options=[
                        {"label": "JSON - Full trial data", "value": "json"},
                        {"label": "YAML - Configuration format", "value": "yaml"},
                        {"label": "Python - Best params as dict", "value": "python"},
                    ],
                    value="json",
                    className="mb-3"
                ),
                html.Div(id="hpo-export-preview", className="mt-3")
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="hpo-export-cancel-btn", color="secondary"),
                dbc.Button([html.I(className="fas fa-download me-2"), "Download"], 
                          id="hpo-export-download-btn", color="primary"),
            ])
        ], id="hpo-export-modal", size="lg", is_open=False),

        # Hidden stores
        dcc.Store(id="hpo-selected-campaign-id", data=None),
        dcc.Download(id="hpo-download-export"),

    ], fluid=True)


def create_hpo_campaign_card(campaign):
    """Create a card for an HPO campaign with enhanced controls."""
    # Determine status color and badge
    status_colors = {
        "running": "primary",
        "completed": "success",
        "failed": "danger",
        "paused": "warning",
        "cancelled": "secondary",
        "pending": "info",
    }
    status_color = status_colors.get(campaign.status, "secondary")
    
    # Build action buttons based on status
    action_buttons = []
    
    # View Details - always available
    action_buttons.append(
        dbc.Button([html.I(className="fas fa-eye me-1"), "Details"], 
                   id={"type": "view-hpo-btn", "index": campaign.id}, 
                   color="info", size="sm", className="me-1")
    )
    
    # Visualizations - available for completed campaigns with trials
    if campaign.status == "completed" and campaign.completed_trials > 0:
        action_buttons.append(
            dbc.Button([html.I(className="fas fa-chart-bar me-1"), "Visualize"], 
                       id={"type": "viz-hpo-btn", "index": campaign.id}, 
                       color="primary", size="sm", className="me-1")
        )
    
    # Export - available for completed campaigns
    if campaign.status == "completed":
        action_buttons.append(
            dbc.Button([html.I(className="fas fa-download me-1"), "Export"], 
                       id={"type": "export-hpo-btn", "index": campaign.id}, 
                       color="success", size="sm", className="me-1")
        )
    
    # Resume - available for paused/cancelled campaigns
    if campaign.status in ["paused", "cancelled"]:
        action_buttons.append(
            dbc.Button([html.I(className="fas fa-play me-1"), "Resume"], 
                       id={"type": "resume-hpo-btn", "index": campaign.id}, 
                       color="warning", size="sm", className="me-1")
        )
    
    # Stop - available for running campaigns
    if campaign.status == "running":
        action_buttons.append(
            dbc.Button([html.I(className="fas fa-stop me-1"), "Stop"], 
                       id={"type": "stop-hpo-btn", "index": campaign.id}, 
                       color="danger", size="sm")
        )
    
    # Format best score safely
    best_score_str = f"{campaign.best_score:.4f}" if campaign.best_score is not None else "N/A"
    
    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col(html.H5(campaign.name, className="mb-0"), width=8),
                dbc.Col([
                    dbc.Badge(campaign.status.upper(), color=status_color, className="float-end")
                ], width=4),
            ])
        ]),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.P([html.Strong("Model: "), campaign.model_type], className="mb-1"),
                    html.P([html.Strong("Method: "), campaign.optimization_method], className="mb-1"),
                    html.P([html.Strong("Trials: "), f"{campaign.completed_trials}/{campaign.total_trials}"],
                           className="mb-1"),
                ], width=6),
                dbc.Col([
                    html.P([html.Strong("Best Score: "), best_score_str], className="mb-1"),
                    html.P([html.Strong("Started: "), campaign.created_at.strftime("%Y-%m-%d %H:%M")],
                           className="mb-1"),
                ], width=6),
            ]),

            dbc.Progress(
                value=(campaign.completed_trials / campaign.total_trials * 100) if campaign.total_trials > 0 else 0,
                className="mt-3",
                color=status_color
            ),
        ]),
        dbc.CardFooter([
            html.Div(action_buttons, className="d-flex flex-wrap")
        ])
    ], className="mb-3 shadow-sm")

