"""
Data Generation layout - Phase 0 integration.
Allows users to generate synthetic vibration datasets through the dashboard.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from utils.constants import FAULT_CLASSES


def create_data_generation_layout():
    """Create data generation layout with configuration form."""
    return dbc.Container([
        html.H2("Data Generation", className="mb-4"),
        html.P("Generate synthetic vibration signals using physics-based fault models.",
               className="text-muted mb-4"),

        dbc.Row([
            # Configuration panel
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-cog me-2"),
                        "Generation Configuration"
                    ]),
                    dbc.CardBody([
                        # Basic settings
                        html.H6("Basic Settings", className="mb-3"),

                        html.Label("Dataset Name"),
                        dbc.Input(
                            id="dataset-name-input",
                            type="text",
                            placeholder="e.g., bearing_faults_v1",
                            className="mb-3"
                        ),

                        html.Label("Output Directory"),
                        dbc.Input(
                            id="output-dir-input",
                            type="text",
                            value="data/generated",
                            className="mb-3"
                        ),

                        html.Label(f"Signals per Fault Class (Current: 100)"),
                        dcc.Slider(
                            id="num-signals-slider",
                            min=10,
                            max=1000,
                            step=10,
                            value=100,
                            marks={10: '10', 100: '100', 500: '500', 1000: '1000'},
                            tooltip={"placement": "bottom", "always_visible": False},
                            className="mb-4"
                        ),

                        html.Hr(),

                        # Fault types selection
                        html.H6("Fault Types", className="mb-3"),
                        dbc.Checklist(
                            id="fault-types-checklist",
                            options=[
                                {"label": " Normal (Healthy)", "value": "sain"},
                                {"label": " Misalignment", "value": "desalignement"},
                                {"label": " Imbalance", "value": "desequilibre"},
                                {"label": " Bearing Clearance", "value": "jeu"},
                                {"label": " Lubrication Issues", "value": "lubrification"},
                                {"label": " Cavitation", "value": "cavitation"},
                                {"label": " Wear", "value": "usure"},
                                {"label": " Oil Whirl", "value": "oilwhirl"},
                                {"label": " Mixed: Misalign + Imbalance", "value": "mixed_misalign_imbalance"},
                                {"label": " Mixed: Wear + Lube", "value": "mixed_wear_lube"},
                                {"label": " Mixed: Cavitation + Clearance", "value": "mixed_cavit_jeu"},
                            ],
                            value=["sain", "desalignement", "desequilibre", "jeu", "lubrification",
                                   "cavitation", "usure", "oilwhirl"],
                            className="mb-3",
                            switch=True
                        ),

                        html.Hr(),

                        # Severity levels
                        html.H6("Severity Levels", className="mb-3"),
                        dbc.Checklist(
                            id="severity-levels-checklist",
                            options=[
                                {"label": " Incipient (20-45%)", "value": "incipient"},
                                {"label": " Mild (45-70%)", "value": "mild"},
                                {"label": " Moderate (70-90%)", "value": "moderate"},
                                {"label": " Severe (90-100%)", "value": "severe"},
                            ],
                            value=["incipient", "mild", "moderate", "severe"],
                            className="mb-3",
                            switch=True
                        ),

                        dbc.Checklist(
                            id="temporal-evolution-check",
                            options=[{"label": " Enable Temporal Evolution (30% of signals)", "value": "enabled"}],
                            value=["enabled"],
                            className="mb-3",
                            switch=True
                        ),
                    ])
                ], className="shadow-sm mb-3"),

                # Advanced settings card
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-sliders-h me-2"),
                        "Advanced Settings"
                    ]),
                    dbc.CardBody([
                        # Noise layers
                        html.H6("7-Layer Noise Model", className="mb-3"),
                        dbc.Checklist(
                            id="noise-layers-checklist",
                            options=[
                                {"label": " Measurement Noise (sensor electronics)", "value": "measurement"},
                                {"label": " EMI (50/60 Hz interference)", "value": "emi"},
                                {"label": " Pink Noise (1/f environmental)", "value": "pink"},
                                {"label": " Environmental Drift", "value": "drift"},
                                {"label": " Quantization (ADC resolution)", "value": "quantization"},
                                {"label": " Sensor Drift", "value": "sensor_drift"},
                                {"label": " Impulse Noise (mechanical impacts)", "value": "impulse"},
                            ],
                            value=["measurement", "emi", "pink", "drift", "quantization", "sensor_drift", "impulse"],
                            className="mb-3",
                            switch=True
                        ),

                        html.Hr(),

                        # Operating conditions
                        html.H6("Operating Conditions", className="mb-3"),

                        html.Label("Speed Variation (±%)"),
                        dcc.Slider(
                            id="speed-variation-slider",
                            min=0,
                            max=20,
                            step=1,
                            value=10,
                            marks={0: '0%', 10: '10%', 20: '20%'},
                            tooltip={"placement": "bottom", "always_visible": False},
                            className="mb-3"
                        ),

                        html.Label("Load Range"),
                        dcc.RangeSlider(
                            id="load-range-slider",
                            min=0,
                            max=100,
                            step=5,
                            value=[30, 100],
                            marks={0: '0%', 50: '50%', 100: '100%'},
                            tooltip={"placement": "bottom", "always_visible": False},
                            className="mb-3"
                        ),

                        html.Label("Temperature Range (°C)"),
                        dcc.RangeSlider(
                            id="temp-range-slider",
                            min=20,
                            max=100,
                            step=5,
                            value=[40, 80],
                            marks={20: '20°C', 60: '60°C', 100: '100°C'},
                            tooltip={"placement": "bottom", "always_visible": False},
                            className="mb-4"
                        ),

                        html.Hr(),

                        # Data augmentation
                        html.H6("Data Augmentation", className="mb-3"),
                        dbc.Checklist(
                            id="augmentation-enabled-check",
                            options=[{"label": " Enable Data Augmentation", "value": "enabled"}],
                            value=["enabled"],
                            className="mb-3",
                            switch=True
                        ),

                        html.Div(id="augmentation-settings", children=[
                            html.Label("Augmentation Ratio (% additional samples)"),
                            dcc.Slider(
                                id="augmentation-ratio-slider",
                                min=0,
                                max=100,
                                step=5,
                                value=30,
                                marks={0: '0%', 30: '30%', 50: '50%', 100: '100%'},
                                tooltip={"placement": "bottom", "always_visible": False},
                                className="mb-3"
                            ),

                            dbc.Checklist(
                                id="augmentation-methods-checklist",
                                options=[
                                    {"label": " Time Shift", "value": "time_shift"},
                                    {"label": " Amplitude Scale", "value": "amplitude_scale"},
                                    {"label": " Noise Injection", "value": "noise_injection"},
                                ],
                                value=["time_shift", "amplitude_scale", "noise_injection"],
                                className="mb-3"
                            ),
                        ]),

                        html.Hr(),

                        # Output format
                        html.H6("Output Format", className="mb-3"),
                        dbc.RadioItems(
                            id="output-format-radio",
                            options=[
                                {"label": "MATLAB (.mat files)", "value": "mat"},
                                {"label": "HDF5 (.h5 file)", "value": "hdf5"},
                                {"label": "Both (MAT + HDF5)", "value": "both"},
                            ],
                            value="both",
                            className="mb-3"
                        ),

                        # Random seed
                        html.Label("Random Seed (for reproducibility)"),
                        dbc.Input(
                            id="random-seed-input",
                            type="number",
                            value=42,
                            className="mb-3"
                        ),
                    ])
                ], className="shadow-sm")
            ], width=5),

            # Preview and controls panel
            dbc.Col([
                # Configuration summary
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-eye me-2"),
                        "Configuration Summary"
                    ]),
                    dbc.CardBody([
                        html.Div(id="config-summary", children=[
                            html.P("Configure generation settings to see summary", className="text-muted")
                        ])
                    ])
                ], className="shadow-sm mb-3"),

                # Action buttons
                dbc.Card([
                    dbc.CardHeader("Actions"),
                    dbc.CardBody([
                        dbc.Button([
                            html.I(className="fas fa-play me-2"),
                            "Generate Dataset"
                        ], id="start-generation-btn", color="success", size="lg", className="w-100 mb-2"),

                        dbc.Button([
                            html.I(className="fas fa-save me-2"),
                            "Save Configuration"
                        ], id="save-config-btn", color="primary", className="w-100 mb-2", outline=True),

                        dbc.Button([
                            html.I(className="fas fa-upload me-2"),
                            "Load Configuration"
                        ], id="load-config-btn", color="secondary", className="w-100", outline=True),
                    ])
                ], className="shadow-sm mb-3"),

                # Progress tracker
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-tasks me-2"),
                        "Generation Progress"
                    ]),
                    dbc.CardBody([
                        html.Div(id="generation-status", children=[
                            html.P("No generation in progress", className="text-muted mb-0")
                        ]),
                        html.Hr(id="progress-divider", style={"display": "none"}),
                        dbc.Progress(id="generation-progress", value=0, style={"display": "none"}, className="mb-2"),
                        html.Div(id="generation-stats", style={"display": "none"}),
                    ])
                ], className="shadow-sm mb-3"),

                # Recent generations
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-history me-2"),
                        "Recent Generations"
                    ]),
                    dbc.CardBody([
                        html.Div(id="recent-generations-list", children=[
                            html.P("No recent generations", className="text-muted")
                        ])
                    ])
                ], className="shadow-sm")
            ], width=7),
        ], className="mb-4"),

        # Hidden components for state management
        dcc.Interval(id='generation-poll-interval', interval=2000, n_intervals=0, disabled=True),
        dcc.Store(id='active-generation-id', data=None),
        dcc.Store(id='generation-config-store', data={}),
    ], fluid=True)
