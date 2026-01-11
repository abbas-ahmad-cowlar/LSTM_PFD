"""
Deployment Dashboard (Phase 11).
Model quantization, ONNX export, and optimization for production deployment.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_deployment_layout():
    """Create deployment dashboard layout."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="fas fa-rocket me-3"),
                    "Model Deployment"
                ], className="mb-1"),
                html.P("Optimize and export models for production deployment", className="text-muted mb-4")
            ])
        ]),

        # Model Selection
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Select Model"),
                    dbc.CardBody([
                        dbc.Label("Experiment"),
                        dcc.Dropdown(
                            id="deployment-experiment-select",
                            placeholder="Select a completed experiment..."
                        ),
                        html.Div(id="deployment-model-info", className="mt-3")
                    ])
                ], className="shadow-sm")
            ])
        ], className="mb-4"),

        # Deployment Options Tabs
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Deployment Options"),
                    dbc.CardBody([
                        dbc.Tabs([
                            # Quantization Tab
                            dbc.Tab(
                                label="Quantization",
                                tab_id="quantization-tab",
                                children=[
                                    create_quantization_tab()
                                ]
                            ),

                            # ONNX Export Tab
                            dbc.Tab(
                                label="ONNX Export",
                                tab_id="onnx-tab",
                                children=[
                                    create_onnx_export_tab()
                                ]
                            ),

                            # Model Optimization Tab
                            dbc.Tab(
                                label="Optimization",
                                tab_id="optimization-tab",
                                children=[
                                    create_optimization_tab()
                                ]
                            ),

                            # Benchmarking Tab
                            dbc.Tab(
                                label="Benchmarking",
                                tab_id="benchmark-tab",
                                children=[
                                    create_benchmark_tab()
                                ]
                            ),

                        ], id="deployment-tabs", active_tab="quantization-tab")
                    ])
                ], className="shadow-sm")
            ])
        ], className="mb-4"),

        # Results Area
        dbc.Row([
            dbc.Col([
                html.Div(id="deployment-results")
            ])
        ])

    ], fluid=True)


def create_quantization_tab():
    """Create quantization tab content."""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H5("Model Quantization", className="mt-3 mb-3"),
                html.P("Reduce model size and improve inference speed using quantization."),

                dbc.Label("Quantization Type"),
                dcc.Dropdown(
                    id="quantization-type",
                    options=[
                        {"label": "Dynamic INT8 (Recommended for most models)", "value": "dynamic"},
                        {"label": "Static INT8 (Requires calibration data)", "value": "static"},
                        {"label": "FP16 (Half precision)", "value": "fp16"},
                        {"label": "Quantization-Aware Training (QAT)", "value": "qat"},
                    ],
                    value="dynamic",
                    className="mb-3"
                ),

                html.Div(id="quantization-options", className="mb-3"),

                dbc.Button(
                    [html.I(className="fas fa-compress me-2"), "Quantize Model"],
                    id="quantize-btn",
                    color="primary",
                    className="mb-3"
                ),

                html.Div(id="quantization-status"),

            ], width=12)
        ])
    ], className="p-3")


def create_onnx_export_tab():
    """Create ONNX export tab content."""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H5("ONNX Export", className="mt-3 mb-3"),
                html.P("Export PyTorch model to ONNX format for cross-platform deployment."),

                dbc.Checklist(
                    id="onnx-options",
                    options=[
                        {"label": "Optimize for inference", "value": "optimize"},
                        {"label": "Use dynamic axes", "value": "dynamic_axes"},
                        {"label": "Enable verbose logging", "value": "verbose"},
                    ],
                    value=["optimize"],
                    className="mb-3"
                ),

                dbc.Label("ONNX Opset Version"),
                dbc.Input(
                    type="number",
                    id="onnx-opset-version",
                    value=14,
                    min=9,
                    max=16,
                    className="mb-3"
                ),

                dbc.Button(
                    [html.I(className="fas fa-file-export me-2"), "Export to ONNX"],
                    id="onnx-export-btn",
                    color="success",
                    className="mb-3"
                ),

                html.Div(id="onnx-export-status"),

            ], width=12)
        ])
    ], className="p-3")


def create_optimization_tab():
    """Create model optimization tab content."""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H5("Model Optimization", className="mt-3 mb-3"),
                html.P("Apply pruning and layer fusion to reduce model size."),

                dbc.Label("Pruning Method"),
                dcc.Dropdown(
                    id="pruning-method",
                    options=[
                        {"label": "L1 Unstructured Pruning", "value": "l1_unstructured"},
                        {"label": "Random Unstructured Pruning", "value": "random_unstructured"},
                        {"label": "Structured Pruning", "value": "structured"},
                        {"label": "No Pruning", "value": "none"},
                    ],
                    value="none",
                    className="mb-3"
                ),

                dbc.Label("Pruning Amount (%)"),
                dbc.Input(
                    type="number",
                    id="pruning-amount",
                    value=30,
                    min=0,
                    max=90,
                    step=5,
                    className="mb-3"
                ),

                dbc.Checklist(
                    id="optimization-options",
                    options=[
                        {"label": "Apply layer fusion", "value": "fusion"},
                        {"label": "Remove unused parameters", "value": "cleanup"},
                    ],
                    value=["fusion"],
                    className="mb-3"
                ),

                dbc.Button(
                    [html.I(className="fas fa-tools me-2"), "Optimize Model"],
                    id="optimize-btn",
                    color="warning",
                    className="mb-3"
                ),

                html.Div(id="optimization-status"),

            ], width=12)
        ])
    ], className="p-3")


def create_benchmark_tab():
    """Create benchmarking tab content."""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H5("Model Benchmarking", className="mt-3 mb-3"),
                html.P("Compare model size, inference speed, and accuracy."),

                dbc.Label("Number of Benchmark Runs"),
                dbc.Input(
                    type="number",
                    id="benchmark-runs",
                    value=100,
                    min=10,
                    max=1000,
                    className="mb-3"
                ),

                dbc.Checklist(
                    id="benchmark-options",
                    options=[
                        {"label": "Benchmark original model", "value": "original"},
                        {"label": "Benchmark quantized model", "value": "quantized"},
                        {"label": "Benchmark ONNX model", "value": "onnx"},
                        {"label": "Benchmark optimized model", "value": "optimized"},
                    ],
                    value=["original"],
                    className="mb-3"
                ),

                dbc.Button(
                    [html.I(className="fas fa-tachometer-alt me-2"), "Run Benchmark"],
                    id="benchmark-btn",
                    color="info",
                    className="mb-3"
                ),

                html.Div(id="benchmark-results"),

            ], width=12)
        ])
    ], className="p-3")


def create_model_info_card(experiment):
    """
    Create model information card.

    Args:
        experiment: Experiment dictionary

    Returns:
        Dash component
    """
    return dbc.Card([
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H6("Model Information"),
                    html.P([html.Strong("Experiment: "), experiment.get('name', 'N/A')], className="mb-1"),
                    html.P([html.Strong("Model Type: "), experiment.get('model_type', 'N/A')], className="mb-1"),
                    html.P([html.Strong("Test Accuracy: "), f"{experiment.get('test_accuracy', 0):.2%}"], className="mb-1"),
                ], width=6),
                dbc.Col([
                    html.P([html.Strong("Status: "), experiment.get('status', 'N/A')], className="mb-1"),
                    html.P([html.Strong("Created: "), experiment.get('created_at', 'N/A')[:10]], className="mb-1"),
                ], width=6),
            ])
        ])
    ], className="mb-3", color="light")


def create_deployment_result_card(title, results, color="success"):
    """
    Create deployment result card.

    Args:
        title: Result title
        results: Results dictionary
        color: Card color

    Returns:
        Dash component
    """
    return dbc.Card([
        dbc.CardHeader(title),
        dbc.CardBody([
            html.Div([
                html.P([html.Strong(f"{key}: "), str(value)])
                for key, value in results.items()
            ])
        ])
    ], color=color, outline=True, className="mb-3")
