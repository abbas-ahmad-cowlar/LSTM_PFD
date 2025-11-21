"""
Signal Viewer layout.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_signal_viewer_layout():
    """Create signal viewer layout."""
    return dbc.Container([
        html.H2("Signal Viewer", className="mb-4"),

        # Signal selection
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(id="signal-id-selector", placeholder="Select or enter signal ID"),
            ], width=6),
            dbc.Col([
                dbc.Button("Random Signal", id="random-signal-btn", color="secondary"),
                dbc.Button("Upload Signal", id="upload-signal-btn", color="primary", className="ms-2"),
            ], width=6, className="text-end"),
        ], className="mb-4"),

        dbc.Row([
            # Visualizations
            dbc.Col([
                # Time domain
                dbc.Card([
                    dbc.CardHeader("Time Domain"),
                    dbc.CardBody([
                        dcc.Loading(dcc.Graph(id="time-domain-plot"))
                    ])
                ], className="mb-3 shadow-sm"),

                # Frequency domain
                dbc.Card([
                    dbc.CardHeader("Frequency Domain"),
                    dbc.CardBody([
                        dcc.Loading(dcc.Graph(id="frequency-domain-plot"))
                    ])
                ], className="mb-3 shadow-sm"),

                # Spectrogram
                dbc.Card([
                    dbc.CardHeader([
                        "Time-Frequency Spectrogram",
                        dbc.Button("⚙️", id="spectrogram-settings-btn", size="sm",
                                   color="link", className="float-end")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(dcc.Graph(id="spectrogram-plot"))
                    ])
                ], className="shadow-sm"),
            ], width=9),

            # Metadata panel
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Signal Information"),
                    dbc.CardBody([
                        html.Div(id="signal-metadata")
                    ])
                ], className="mb-3 shadow-sm"),

                dbc.Card([
                    dbc.CardHeader("Extracted Features"),
                    dbc.CardBody([
                        html.Div(id="signal-features")
                    ])
                ], className="mb-3 shadow-sm"),

                dbc.Card([
                    dbc.CardHeader("Actions"),
                    dbc.CardBody([
                        dbc.Button("Predict Fault", id="predict-fault-btn",
                                   color="primary", className="w-100 mb-2"),
                        dbc.Button("Export Signal", id="export-signal-btn",
                                   color="secondary", className="w-100 mb-2"),
                        dbc.Button("Add to Comparison", id="add-comparison-btn",
                                   color="info", className="w-100"),
                    ])
                ], className="shadow-sm"),
            ], width=3),
        ])
    ], fluid=True)
