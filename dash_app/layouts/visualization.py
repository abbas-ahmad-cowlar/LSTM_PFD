"""
Advanced Visualization Dashboard Layout (Phase 4, Feature 2/3).
Provides access to advanced visualization capabilities:
- Embeddings (t-SNE, UMAP, PCA)
- Signal Analysis (Bispectrum, Wavelet, Spectrogram)
- Feature Analysis (Importance, Correlation, Distributions)
- Model Analysis (Saliency Maps, Activation Maps, Counterfactuals)
"""
from dash import html, dcc
import dash_bootstrap_components as dbc


def create_visualization_layout():
    """
    Create the advanced visualization dashboard layout.

    Features:
        - Embeddings tab: t-SNE/UMAP/PCA dimensionality reduction
        - Signal Analysis tab: Bispectrum, Wavelet, Spectrogram
        - Feature Analysis tab: Importance, Correlation, Distributions
        - Model Analysis tab: Saliency maps, Activation maps, Counterfactuals
    """
    return dbc.Container([
        # Page Header
        dbc.Row([
            dbc.Col([
                html.H2("📊 Advanced Visualizations", className="mb-3"),
                html.P(
                    "Explore your data and models with advanced visualization techniques.",
                    className="text-muted"
                )
            ])
        ], className="mb-4"),

        # Dataset/Experiment Selector
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Select Dataset:", className="fw-bold"),
                        dcc.Dropdown(
                            id="viz-dataset-select",
                            placeholder="Choose a dataset...",
                            className="mb-2"
                        )
                    ], width=6),
                    dbc.Col([
                        html.Label("Select Experiment (optional):", className="fw-bold"),
                        dcc.Dropdown(
                            id="viz-experiment-select",
                            placeholder="Choose an experiment...",
                            className="mb-2"
                        )
                    ], width=6),
                ])
            ])
        ], className="mb-4"),

        # Tabs for different visualization types
        dbc.Tabs([
            # Tab 1: Embeddings
            dbc.Tab(
                label="🗺️ Embeddings",
                tab_id="embeddings",
                children=create_embeddings_tab()
            ),

            # Tab 2: Signal Analysis
            dbc.Tab(
                label="📡 Signal Analysis",
                tab_id="signal-analysis",
                children=create_signal_analysis_tab()
            ),

            # Tab 3: Feature Analysis
            dbc.Tab(
                label="📈 Feature Analysis",
                tab_id="feature-analysis",
                children=create_feature_analysis_tab()
            ),

            # Tab 4: Model Analysis
            dbc.Tab(
                label="🧠 Model Analysis",
                tab_id="model-analysis",
                children=create_model_analysis_tab()
            ),
        ], id='viz-tabs', active_tab='embeddings', className="mb-4"),

        # Auto-refresh interval (hidden)
        dcc.Interval(
            id='viz-auto-refresh',
            interval=30*1000,  # 30 seconds
            n_intervals=0,
            disabled=True
        ),

    ], fluid=True, className="py-4")


def create_embeddings_tab():
    """Create embeddings visualization tab (t-SNE, UMAP, PCA)."""
    return dbc.Container([
        html.H4("Dimensionality Reduction Embeddings", className="mt-3 mb-3"),
        html.P([
            "Visualize high-dimensional feature spaces in 2D using dimensionality reduction. ",
            "This helps identify clusters and patterns in your fault data."
        ], className="text-muted mb-4"),

        dbc.Row([
            # Controls Column
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Settings", className="mb-0")),
                    dbc.CardBody([
                        # Method Selection
                        html.Label("Reduction Method:", className="fw-bold"),
                        dbc.RadioItems(
                            id="embedding-method",
                            options=[
                                {"label": "t-SNE (t-Distributed Stochastic Neighbor Embedding)", "value": "tsne"},
                                {"label": "UMAP (Uniform Manifold Approximation)", "value": "umap"},
                                {"label": "PCA (Principal Component Analysis)", "value": "pca"}
                            ],
                            value="tsne",
                            className="mb-3"
                        ),

                        # t-SNE Parameters
                        html.Div([
                            html.Label("Perplexity:", className="fw-bold"),
                            dcc.Slider(
                                id="tsne-perplexity",
                                min=5,
                                max=50,
                                value=30,
                                step=5,
                                marks={5: '5', 15: '15', 30: '30', 50: '50'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.Small("Controls balance between local and global structure", className="text-muted"),
                        ], id="tsne-params", style={'display': 'block'}, className="mb-3"),

                        # UMAP Parameters
                        html.Div([
                            html.Label("Number of Neighbors:", className="fw-bold"),
                            dcc.Slider(
                                id="umap-neighbors",
                                min=5,
                                max=50,
                                value=15,
                                step=5,
                                marks={5: '5', 15: '15', 30: '30', 50: '50'},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.Small("Controls local vs global structure preservation", className="text-muted"),
                        ], id="umap-params", style={'display': 'none'}, className="mb-3"),

                        # Generate Button
                        dbc.Button(
                            [html.I(className="bi bi-play-circle me-2"), "Generate Embedding"],
                            id="generate-embedding-btn",
                            color="primary",
                            size="lg",
                            className="w-100 mt-3"
                        ),

                        # Status message
                        html.Div(id='embedding-status', className="mt-3")
                    ])
                ])
            ], width=3),

            # Plot Column
            dbc.Col([
                dcc.Loading(
                    id="loading-embedding",
                    children=[dcc.Graph(id="embedding-plot", style={'height': '600px'})],
                    type="default"
                )
            ], width=9)
        ])
    ], className="py-4", fluid=True)


def create_signal_analysis_tab():
    """Create signal analysis visualization tab."""
    return dbc.Container([
        html.H4("Signal Analysis Visualizations", className="mt-3 mb-3"),
        html.P([
            "Analyze signals using advanced time-frequency representations. ",
            "These visualizations reveal patterns not visible in raw signal or standard FFT."
        ], className="text-muted mb-4"),

        dbc.Row([
            # Controls Column
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Settings", className="mb-0")),
                    dbc.CardBody([
                        # Visualization Type
                        html.Label("Visualization Type:", className="fw-bold"),
                        dbc.RadioItems(
                            id="signal-viz-type",
                            options=[
                                {"label": "Bispectrum (Higher-order spectral analysis)", "value": "bispectrum"},
                                {"label": "Wavelet Scalogram (Time-frequency)", "value": "wavelet"},
                                {"label": "Spectrogram (STFT)", "value": "spectrogram"}
                            ],
                            value="bispectrum",
                            className="mb-3"
                        ),

                        # Signal Selection
                        html.Label("Select Signal:", className="fw-bold"),
                        dcc.Dropdown(
                            id="signal-select",
                            placeholder="Choose a signal...",
                            className="mb-3"
                        ),

                        # Fault Type Filter
                        html.Label("Filter by Fault Type (optional):", className="fw-bold"),
                        dcc.Dropdown(
                            id="signal-fault-filter",
                            placeholder="All fault types...",
                            className="mb-3"
                        ),

                        # Generate Button
                        dbc.Button(
                            [html.I(className="bi bi-play-circle me-2"), "Generate Plot"],
                            id="generate-signal-viz-btn",
                            color="primary",
                            size="lg",
                            className="w-100 mt-3"
                        ),

                        # Status message
                        html.Div(id='signal-viz-status', className="mt-3")
                    ])
                ])
            ], width=3),

            # Plot Column
            dbc.Col([
                dcc.Loading(
                    id="loading-signal-viz",
                    children=[dcc.Graph(id="signal-viz-plot", style={'height': '600px'})],
                    type="default"
                )
            ], width=9)
        ])
    ], className="py-4", fluid=True)


def create_feature_analysis_tab():
    """Create feature analysis visualization tab."""
    return dbc.Container([
        html.H4("Feature Analysis Visualizations", className="mt-3 mb-3"),
        html.P([
            "Analyze extracted features to understand which features are most important ",
            "for fault classification and how features correlate with each other."
        ], className="text-muted mb-4"),

        dbc.Row([
            # Controls Column
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Settings", className="mb-0")),
                    dbc.CardBody([
                        # Visualization Type
                        html.Label("Visualization Type:", className="fw-bold"),
                        dbc.RadioItems(
                            id="feature-viz-type",
                            options=[
                                {"label": "Feature Importance", "value": "importance"},
                                {"label": "Feature Correlation", "value": "correlation"},
                                {"label": "Feature Distributions", "value": "distributions"}
                            ],
                            value="importance",
                            className="mb-3"
                        ),

                        # Top N Features
                        html.Label("Top N Features:", className="fw-bold"),
                        dcc.Slider(
                            id="top-n-features",
                            min=5,
                            max=50,
                            value=15,
                            step=5,
                            marks={5: '5', 15: '15', 30: '30', 50: '50'},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className="mb-3"
                        ),

                        # Feature Domain Filter
                        html.Label("Feature Domain (optional):", className="fw-bold"),
                        dcc.Dropdown(
                            id="feature-domain-filter",
                            options=[
                                {'label': 'All Domains', 'value': 'all'},
                                {'label': 'Time Domain', 'value': 'time'},
                                {'label': 'Frequency Domain', 'value': 'frequency'},
                                {'label': 'Wavelet Domain', 'value': 'wavelet'},
                                {'label': 'Bispectrum', 'value': 'bispectrum'},
                            ],
                            value='all',
                            className="mb-3"
                        ),

                        # Generate Button
                        dbc.Button(
                            [html.I(className="bi bi-play-circle me-2"), "Generate Plot"],
                            id="generate-feature-viz-btn",
                            color="primary",
                            size="lg",
                            className="w-100 mt-3"
                        ),

                        # Status message
                        html.Div(id='feature-viz-status', className="mt-3")
                    ])
                ])
            ], width=3),

            # Plot Column
            dbc.Col([
                dcc.Loading(
                    id="loading-feature-viz",
                    children=[dcc.Graph(id="feature-viz-plot", style={'height': '600px'})],
                    type="default"
                )
            ], width=9)
        ])
    ], className="py-4", fluid=True)


def create_model_analysis_tab():
    """Create model analysis visualization tab."""
    return dbc.Container([
        html.H4("Model Interpretation Visualizations", className="mt-3 mb-3"),
        html.P([
            "Understand what your model is learning by visualizing attention, activations, and explanations. ",
            "These visualizations help debug and improve model performance."
        ], className="text-muted mb-4"),

        dbc.Row([
            # Controls Column
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Settings", className="mb-0")),
                    dbc.CardBody([
                        # Visualization Type
                        html.Label("Visualization Type:", className="fw-bold"),
                        dbc.RadioItems(
                            id="model-viz-type",
                            options=[
                                {"label": "Saliency Maps (Gradient-based)", "value": "saliency"},
                                {"label": "Activation Maps (Layer outputs)", "value": "activation"},
                                {"label": "Counterfactual Explanations", "value": "counterfactual"}
                            ],
                            value="saliency",
                            className="mb-3"
                        ),

                        # Layer Selection (for activation maps)
                        html.Div([
                            html.Label("Select Layer:", className="fw-bold"),
                            dcc.Dropdown(
                                id="layer-select",
                                placeholder="Choose a layer...",
                                className="mb-3"
                            ),
                        ], id="layer-selection-div", style={'display': 'none'}),

                        # Sample Selection
                        html.Label("Select Sample:", className="fw-bold"),
                        dcc.Dropdown(
                            id="sample-select",
                            placeholder="Choose a sample...",
                            className="mb-3"
                        ),

                        # Generate Button
                        dbc.Button(
                            [html.I(className="bi bi-play-circle me-2"), "Generate Visualization"],
                            id="generate-model-viz-btn",
                            color="primary",
                            size="lg",
                            className="w-100 mt-3"
                        ),

                        # Status message
                        html.Div(id='model-viz-status', className="mt-3")
                    ])
                ])
            ], width=3),

            # Plot Column
            dbc.Col([
                dcc.Loading(
                    id="loading-model-viz",
                    children=[dcc.Graph(id="model-viz-plot", style={'height': '600px'})],
                    type="default"
                )
            ], width=9)
        ])
    ], className="py-4", fluid=True)
