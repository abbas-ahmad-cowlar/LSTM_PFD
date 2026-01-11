"""
XAI (Explainable AI) Dashboard (Phase 11C).
Interactive dashboard for SHAP, LIME, and Grad-CAM explanations.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE
from dashboard_config import FAULT_CLASSES


def create_xai_dashboard_layout():
    """Create XAI dashboard layout."""
    return dbc.Container([
        html.H2("Explainable AI Dashboard", className="mb-4"),

        # Model and signal selection
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Select Model & Signal", className="card-title mb-3"),

                        dbc.Label("Trained Model"),
                        dcc.Dropdown(
                            id="xai-model-dropdown",
                            placeholder="Select a trained model...",
                            options=[],  # Populated by callback
                            className="mb-3"
                        ),

                        dbc.Label("Signal to Explain"),
                        dcc.Dropdown(
                            id="xai-signal-dropdown",
                            placeholder="Select a signal...",
                            options=[],  # Populated by callback
                            className="mb-3"
                        ),

                        dbc.Label("Explanation Method"),
                        dcc.Dropdown(
                            id="xai-method-dropdown",
                            options=[
                                {"label": "SHAP (SHapley Additive exPlanations)", "value": "shap"},
                                {"label": "LIME (Local Interpretable Model-agnostic Explanations)", "value": "lime"},
                                {"label": "Integrated Gradients", "value": "integrated_gradients"},
                                {"label": "Grad-CAM (Gradient-weighted Class Activation Mapping)", "value": "gradcam"},
                            ],
                            value="shap",
                            className="mb-3"
                        ),

                        dbc.Button([
                            html.I(className="fas fa-play me-2"),
                            "Generate Explanation"
                        ], id="generate-xai-btn", color="primary", className="w-100"),
                    ])
                ], className="shadow-sm")
            ], width=4),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Prediction"),
                    dbc.CardBody([
                        html.Div(id="xai-prediction-display", children=[
                            html.P("Select a model and signal to see predictions", className="text-muted")
                        ])
                    ])
                ], className="shadow-sm")
            ], width=8),
        ], className="mb-4"),

        # Explanation visualizations
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Signal with Attribution"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="xai-signal-loading",
                            type="default",
                            children=[dcc.Graph(id="xai-signal-plot")]
                        )
                    ])
                ], className="shadow-sm")
            ], width=12),
        ], className="mb-4"),

        # Feature importance and details
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Feature Importance"),
                    dbc.CardBody([
                        dcc.Loading(
                            id="xai-importance-loading",
                            type="default",
                            children=[dcc.Graph(id="xai-importance-plot")]
                        )
                    ])
                ], className="shadow-sm")
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Explanation Details"),
                    dbc.CardBody([
                        html.Div(id="xai-explanation-details")
                    ])
                ], className="shadow-sm")
            ], width=6),
        ], className="mb-4"),

        # Advanced XAI options
        dbc.Accordion([
            dbc.AccordionItem([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Number of Features to Show"),
                        dbc.Input(type="number", id="xai-num-features", value=20, min=5, max=100),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Background Samples (SHAP)"),
                        dbc.Input(type="number", id="xai-background-samples", value=100, min=10, max=500),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Perturbations (LIME)"),
                        dbc.Input(type="number", id="xai-perturbations", value=1000, min=100, max=5000),
                    ], width=4),
                ])
            ], title="Advanced Options", item_id="advanced"),
        ], id="xai-advanced-options", flush=True, className="mb-4"),

        # Cached explanations
        dbc.Card([
            dbc.CardHeader("Cached Explanations"),
            dbc.CardBody([
                html.Div(id="cached-explanations-list")
            ])
        ], className="shadow-sm mb-4"),

        # Hidden stores
        dcc.Store(id="xai-results-store", data={}),

    ], fluid=True)


def create_xai_prediction_card(prediction, probabilities, true_label=None):
    """Create prediction display card."""
    predicted_class = FAULT_CLASSES[prediction] if prediction < len(FAULT_CLASSES) else f"Class {prediction}"
    confidence = probabilities[prediction] if probabilities else 0

    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H6("Predicted Class"),
                html.H4(predicted_class, className="text-primary"),
                dbc.Progress(value=confidence * 100, label=f"{confidence:.1%}", className="mb-3"),
            ], width=6),
            dbc.Col([
                html.H6("True Label" if true_label is not None else "Confidence"),
                html.H4(
                    FAULT_CLASSES[true_label] if true_label is not None else f"{confidence:.2%}",
                    className="text-success" if true_label == prediction else "text-danger"
                ),
                html.P(
                    "✓ Correct" if true_label == prediction else "✗ Incorrect",
                    className="text-muted"
                ) if true_label is not None else None,
            ], width=6),
        ]),

        html.Hr(),

        html.H6("Top 5 Predictions", className="mb-3"),
        html.Div([
            dbc.Row([
                dbc.Col(f"{i+1}. {FAULT_CLASSES[idx]}", width=6),
                dbc.Col(dbc.Progress(value=prob * 100, label=f"{prob:.1%}", size="sm"), width=6),
            ], className="mb-2")
            for i, (idx, prob) in enumerate(sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)[:5])
        ])
    ])
