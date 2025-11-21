"""
Hyperparameter Optimization Campaigns (Phase 11C).
UI for creating and monitoring HPO campaigns.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc


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

    ], fluid=True)


def create_hpo_campaign_card(campaign):
    """Create a card for an HPO campaign."""
    return dbc.Card([
        dbc.CardHeader([
            dbc.Row([
                dbc.Col(html.H5(campaign.name, className="mb-0"), width=6),
                dbc.Col([
                    dbc.Badge(campaign.status, color="primary" if campaign.status == "running" else "secondary",
                              className="float-end")
                ], width=6),
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
                    html.P([html.Strong("Best Score: "), f"{campaign.best_score:.4f}"], className="mb-1"),
                    html.P([html.Strong("Started: "), campaign.created_at.strftime("%Y-%m-%d %H:%M")],
                           className="mb-1"),
                ], width=6),
            ]),

            dbc.Progress(
                value=(campaign.completed_trials / campaign.total_trials * 100) if campaign.total_trials > 0 else 0,
                className="mt-3"
            ),
        ]),
        dbc.CardFooter([
            dbc.ButtonGroup([
                dbc.Button("View Details", href=f"/hpo/{campaign.id}", color="info", size="sm"),
                dbc.Button("Stop", id={"type": "stop-hpo-btn", "index": campaign.id}, color="danger", size="sm")
                if campaign.status == "running" else None,
            ])
        ])
    ], className="mb-3 shadow-sm")
