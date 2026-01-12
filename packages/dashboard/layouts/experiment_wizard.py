"""
Training experiment configuration wizard (Phase 11B).
Multi-step form for configuring and launching training experiments.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE


def create_experiment_wizard_layout():
    """Create multi-step experiment configuration wizard."""
    return dbc.Container([
        html.H2("New Training Experiment", className="mb-4"),

        # Progress steps indicator
        dbc.Row([
            dbc.Col([
                dbc.Progress(
                    id="wizard-progress",
                    value=25,
                    className="mb-4",
                    striped=True,
                    animated=True
                ),
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("1. Model Selection", active=True, id="step1-nav")),
                    dbc.NavItem(dbc.NavLink("2. Dataset & Hyperparameters", disabled=True, id="step2-nav")),
                    dbc.NavItem(dbc.NavLink("3. Training Options", disabled=True, id="step3-nav")),
                    dbc.NavItem(dbc.NavLink("4. Review & Launch", disabled=True, id="step4-nav")),
                ], pills=True, className="mb-4")
            ])
        ]),

        # Wizard content (dynamic based on current step)
        html.Div(id="wizard-content"),

        # Navigation buttons
        dbc.Row([
            dbc.Col([
                dbc.Button("Previous", id="wizard-prev-btn", color="secondary", className="me-2", disabled=True),
                dbc.Button("Next", id="wizard-next-btn", color="primary", className="me-2"),
                dbc.Button("Cancel", id="wizard-cancel-btn", color="danger", outline=True, href="/experiments"),
            ], className="d-flex justify-content-between mt-4")
        ]),

        # Hidden stores for wizard state
        dcc.Store(id="wizard-step", data=1),
        dcc.Store(id="wizard-config", data={}),
        dcc.Store(id="wizard-validation", data={}),

        # Hidden placeholder components for callback inputs (components render dynamically in steps)
        # These ensure callbacks don't error on missing components
        html.Div([
            # Step 2 components
            dcc.Dropdown(id="dataset-dropdown", style={"display": "none"}),
            dbc.Checkbox(id="show-advanced-options", value=False, style={"display": "none"}),
            dbc.Input(id="random-seed", type="number", value=42, style={"display": "none"}),
            dcc.Dropdown(id="device-dropdown", value="auto", style={"display": "none"}),
            
            # Step 3 components
            dbc.Input(id="num-epochs", type="number", value=100, style={"display": "none"}),
            dcc.Dropdown(id="batch-size-dropdown", value=32, style={"display": "none"}),
            dcc.Dropdown(id="optimizer-dropdown", value="adam", style={"display": "none"}),
            dbc.Input(id="learning-rate", type="number", value=0.001, style={"display": "none"}),
            dcc.Dropdown(id="scheduler-dropdown", value="plateau", style={"display": "none"}),
            dbc.Checklist(id="augmentation-checklist", value=[], style={"display": "none"}),
            dbc.Checkbox(id="enable-distillation", value=False, style={"display": "none"}),
            dcc.Dropdown(id="teacher-model-select", style={"display": "none"}),
            dbc.Input(id="distillation-temperature", type="number", value=3.0, style={"display": "none"}),
            dbc.Input(id="distillation-alpha", type="number", value=0.5, style={"display": "none"}),
            dbc.RadioItems(id="mixed-precision-mode", value="fp32", style={"display": "none"}),
            dbc.Checklist(id="enable-advanced-aug", value=[], style={"display": "none"}),
            dbc.Input(id="aug-magnitude", type="number", value=9, style={"display": "none"}),
            dbc.Input(id="aug-probability", type="number", value=0.5, style={"display": "none"}),
            dbc.Checkbox(id="enable-progressive", value=False, style={"display": "none"}),
            dbc.Input(id="progressive-start-size", type="number", value=51200, style={"display": "none"}),
            dbc.Input(id="progressive-end-size", type="number", value=102400, style={"display": "none"}),
            
            # Step 4 components  
            dbc.Input(id="experiment-name", type="text", style={"display": "none"}),
            dbc.Input(id="experiment-tags", type="text", style={"display": "none"}),
            dbc.Textarea(id="experiment-notes", style={"display": "none"}),
        ], style={"display": "none"}),

    ], fluid=True)


def create_step1_model_selection():
    """Step 1: Model type selection."""
    return dbc.Card([
        dbc.CardHeader(html.H4("Step 1: Select Model Architecture")),
        dbc.CardBody([
            html.P("Choose the model architecture for your training experiment:", className="mb-3"),

            # Model categories
            dbc.Accordion([
                dbc.AccordionItem([
                    _create_model_cards([
                        ("Random Forest", "rf", "Classical ML baseline with feature engineering", "95-96%", "Phase 1"),
                        ("SVM", "svm", "Support Vector Machine with RBF kernel", "94-95%", "Phase 1"),
                        ("Gradient Boosting", "gbm", "Ensemble of decision trees", "95-96%", "Phase 1"),
                    ])
                ], title="Classical Machine Learning (Phase 1)", item_id="classical"),

                dbc.AccordionItem([
                    _create_model_cards([
                        ("1D CNN", "cnn1d", "Multi-scale 1D convolutional network", "93-95%", "Phase 2"),
                        ("ResNet-18", "resnet18", "18-layer residual network", "96-97%", "Phase 3"),
                        ("ResNet-34", "resnet34", "34-layer residual network", "96-98%", "Phase 3"),
                        ("EfficientNet", "efficientnet", "Compound scaled efficient network", "96-97%", "Phase 3"),
                    ])
                ], title="Deep Convolutional Networks (Phases 2-3)", item_id="cnn"),

                dbc.AccordionItem([
                    _create_model_cards([
                        ("Transformer", "transformer", "Self-attention temporal model", "96-97%", "Phase 4"),
                        ("CNN-Transformer Hybrid", "hybrid_transformer", "Combined CNN + Transformer", "97-98%", "Phase 4"),
                    ])
                ], title="Transformer Models (Phase 4)", item_id="transformer"),

                dbc.AccordionItem([
                    _create_model_cards([
                        ("STFT ResNet", "stft_resnet", "ResNet on STFT spectrograms", "96-98%", "Phase 5"),
                        ("CWT ResNet", "cwt_resnet", "ResNet on CWT spectrograms", "97-98%", "Phase 5"),
                        ("WVD ResNet", "wvd_resnet", "ResNet on WVD spectrograms", "96-98%", "Phase 5"),
                        ("Dual-Stream", "dual_stream", "Multi-representation fusion", "97-99%", "Phase 5"),
                    ])
                ], title="Time-Frequency Models (Phase 5)", item_id="timefreq"),

                dbc.AccordionItem([
                    _create_model_cards([
                        ("PINN", "pinn", "Physics-Informed Neural Network", "97-98%", "Phase 6"),
                        ("Hybrid PINN", "hybrid_pinn", "PINN with base model backbone", "98-99%", "Phase 6"),
                    ])
                ], title="Physics-Informed Networks (Phase 6)", item_id="pinn"),

                dbc.AccordionItem([
                    _create_model_cards([
                        ("Voting Ensemble", "voting_ensemble", "Soft voting of multiple models", "98-99%", "Phase 8"),
                        ("Stacked Ensemble", "stacked_ensemble", "Meta-learner stacking", "98-99%", "Phase 8"),
                        ("Mixture of Experts", "moe", "Gated mixture of experts", "98-99%", "Phase 8"),
                    ])
                ], title="Ensemble Methods (Phase 8)", item_id="ensemble"),
            ], id="model-accordion", active_item="cnn", flush=True),

            # Selected model display
            html.Hr(),
            html.Div(id="selected-model-summary", className="mt-3")
        ])
    ], className="shadow-sm")


def _create_model_cards(models):
    """Helper to create model selection cards."""
    cards = []
    for name, model_id, description, accuracy, phase in models:
        cards.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(name, className="card-title"),
                        html.P(description, className="card-text text-muted small"),
                        dbc.Badge(accuracy, color="success", className="me-2"),
                        dbc.Badge(phase, color="info"),
                    ]),
                    dbc.CardFooter([
                        dbc.Button("Select", id={"type": "model-select-btn", "index": model_id},
                                   color="primary", size="sm", className="w-100")
                    ])
                ], className="h-100 shadow-sm model-card")
            ], width=6, lg=4, className="mb-3")
        )
    return dbc.Row(cards)


def create_step2_dataset_hyperparams(model_type=None):
    """Step 2: Dataset selection and hyperparameters."""
    return dbc.Card([
        dbc.CardHeader(html.H4("Step 2: Dataset & Hyperparameters")),
        dbc.CardBody([
            # Dataset selection
            html.H5("Dataset Selection", className="mb-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Select Dataset"),
                    dcc.Dropdown(
                        id="dataset-dropdown",
                        placeholder="Choose a dataset...",
                        options=[],  # Populated by callback
                        className="mb-3"
                    ),
                ], width=8),
                dbc.Col([
                    dbc.Button([html.I(className="fas fa-eye me-2"), "Preview"],
                               id="dataset-preview-btn", color="info", outline=True, className="mt-4")
                ], width=4),
            ]),

            html.Div(id="dataset-info", className="mb-4"),

            html.Hr(),

            # Hyperparameters
            html.H5("Hyperparameters", className="mb-3"),
            html.Div(id="hyperparameters-form"),  # Dynamic based on model type

            html.Hr(),

            # Advanced options toggle
            dbc.Checkbox(
                id="show-advanced-options",
                label="Show advanced options",
                value=False,
                className="mb-3"
            ),
            dbc.Collapse(
                id="advanced-options-collapse",
                is_open=False,
                children=[
                    html.H6("Advanced Training Options"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Random Seed"),
                            dbc.Input(type="number", id="random-seed", value=42, min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Device"),
                            dcc.Dropdown(
                                id="device-dropdown",
                                options=[
                                    {"label": "Auto (GPU if available)", "value": "auto"},
                                    {"label": "CPU", "value": "cpu"},
                                    {"label": "CUDA GPU", "value": "cuda"},
                                ],
                                value="auto"
                            ),
                        ], width=6),
                    ], className="mb-3"),
                ]
            )
        ])
    ], className="shadow-sm")


def create_step3_training_options():
    """Step 3: Training configuration."""
    return dbc.Card([
        dbc.CardHeader(html.H4("Step 3: Training Options")),
        dbc.CardBody([
            dbc.Row([
                # Training parameters
                dbc.Col([
                    html.H5("Training Parameters"),
                    dbc.Label("Number of Epochs"),
                    dbc.Input(type="number", id="num-epochs", value=100, min=1, max=1000, className="mb-3"),

                    dbc.Label("Batch Size"),
                    dcc.Dropdown(
                        id="batch-size-dropdown",
                        options=[
                            {"label": "16 (Small GPU)", "value": 16},
                            {"label": "32 (Recommended)", "value": 32},
                            {"label": "64 (Large GPU)", "value": 64},
                            {"label": "128 (Very Large GPU)", "value": 128},
                        ],
                        value=32,
                        className="mb-3"
                    ),

                    dbc.Label("Early Stopping Patience"),
                    dbc.Input(type="number", id="early-stopping-patience", value=15, min=0, className="mb-3"),

                ], width=6),

                # Optimizer & scheduler
                dbc.Col([
                    html.H5("Optimization"),
                    dbc.Label("Optimizer"),
                    dcc.Dropdown(
                        id="optimizer-dropdown",
                        options=[
                            {"label": "Adam (Recommended)", "value": "adam"},
                            {"label": "AdamW (with weight decay)", "value": "adamw"},
                            {"label": "SGD", "value": "sgd"},
                            {"label": "RMSprop", "value": "rmsprop"},
                        ],
                        value="adam",
                        className="mb-3"
                    ),

                    dbc.Label("Learning Rate"),
                    dbc.Input(type="number", id="learning-rate", value=0.001, step=0.0001, min=0, className="mb-3"),

                    dbc.Label("LR Scheduler"),
                    dcc.Dropdown(
                        id="scheduler-dropdown",
                        options=[
                            {"label": "None", "value": "none"},
                            {"label": "ReduceLROnPlateau", "value": "plateau"},
                            {"label": "CosineAnnealing", "value": "cosine"},
                            {"label": "StepLR", "value": "step"},
                        ],
                        value="plateau",
                        className="mb-3"
                    ),
                ], width=6),
            ]),

            html.Hr(),

            # Data augmentation
            html.H5("Data Augmentation"),
            dbc.Checklist(
                id="augmentation-checklist",
                options=[
                    {"label": "Random noise injection", "value": "noise"},
                    {"label": "Time shifting", "value": "time_shift"},
                    {"label": "Amplitude scaling", "value": "amplitude_scale"},
                    {"label": "Time warping", "value": "time_warp"},
                ],
                value=["noise", "time_shift"],
                inline=True,
                className="mb-3"
            ),

            html.Hr(),

            # Advanced Options (NEW)
            html.H5([
                html.I(className="fas fa-graduation-cap me-2"),
                "Advanced Training Techniques"
            ]),
            html.P("Enable advanced ML techniques for improved performance", className="text-muted small mb-3"),

            dbc.Accordion([
                # Knowledge Distillation
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([
                            dbc.Checkbox(
                                id="enable-distillation",
                                label="Enable Knowledge Distillation",
                                value=False
                            ),
                            html.P("Train student model using knowledge from pre-trained teacher", className="text-muted small")
                        ], width=12, className="mb-3"),
                    ]),
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Teacher Model"),
                                dcc.Dropdown(
                                    id="teacher-model-select",
                                    placeholder="Select completed experiment as teacher...",
                                    className="mb-3"
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Label([
                                    "Temperature ",
                                    html.I(className="fas fa-info-circle ms-1", id="temp-info")
                                ]),
                                dbc.Tooltip("Higher temperature = softer probability distributions", target="temp-info"),
                                dbc.Input(
                                    id="distillation-temperature",
                                    type="number",
                                    value=3.0,
                                    min=1.0,
                                    max=10.0,
                                    step=0.5,
                                    className="mb-3"
                                )
                            ], width=3),
                            dbc.Col([
                                dbc.Label([
                                    "Alpha (student weight) ",
                                    html.I(className="fas fa-info-circle ms-1", id="alpha-info")
                                ]),
                                dbc.Tooltip("0.0 = all teacher loss, 1.0 = all student loss", target="alpha-info"),
                                dbc.Input(
                                    id="distillation-alpha",
                                    type="number",
                                    value=0.5,
                                    min=0.0,
                                    max=1.0,
                                    step=0.1,
                                    className="mb-3"
                                )
                            ], width=3),
                        ])
                    ], id="distillation-config", style={'display': 'none'})
                ], title="Knowledge Distillation", item_id="distillation"),

                # Mixed Precision
                dbc.AccordionItem([
                    dbc.Label("Precision Mode"),
                    dbc.RadioItems(
                        id="mixed-precision-mode",
                        options=[
                            {"label": "FP32 (Full Precision - Default)", "value": "fp32"},
                            {"label": "FP16 (Half Precision - 2x faster, less memory)", "value": "fp16"},
                            {"label": "BF16 (Brain Float16 - Better stability)", "value": "bf16"}
                        ],
                        value="fp32",
                        className="mb-3"
                    ),
                    html.P([
                        html.I(className="fas fa-info-circle me-2"),
                        "FP16/BF16 can speed up training 2-3x and reduce memory usage by ~50%"
                    ], className="text-info small")
                ], title="Mixed Precision Training", item_id="mixed-precision"),

                # Advanced Augmentation
                dbc.AccordionItem([
                    dbc.Checklist(
                        id="enable-advanced-aug",
                        options=[
                            {"label": "RandAugment (random policy selection)", "value": "randaugment"},
                            {"label": "CutMix (mix two samples)", "value": "cutmix"},
                            {"label": "MixUp (blend two samples)", "value": "mixup"}
                        ],
                        value=[],
                        className="mb-3"
                    ),
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Augmentation Magnitude"),
                                dbc.Input(
                                    id="aug-magnitude",
                                    type="number",
                                    value=9,
                                    min=1,
                                    max=20,
                                    className="mb-2"
                                ),
                                html.P("Higher = stronger augmentations", className="text-muted small")
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Probability"),
                                dbc.Input(
                                    id="aug-probability",
                                    type="number",
                                    value=0.5,
                                    min=0.0,
                                    max=1.0,
                                    step=0.1,
                                    className="mb-2"
                                ),
                                html.P("Chance to apply per batch", className="text-muted small")
                            ], width=6),
                        ])
                    ], id="advanced-aug-config", style={'display': 'none'})
                ], title="Advanced Augmentation", item_id="advanced-aug"),

                # Progressive Resizing
                dbc.AccordionItem([
                    dbc.Row([
                        dbc.Col([
                            dbc.Checkbox(
                                id="enable-progressive",
                                label="Enable Progressive Resizing",
                                value=False
                            ),
                            html.P("Start training with smaller signals, gradually increase size", className="text-muted small mb-3")
                        ], width=12),
                    ]),
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Start Size"),
                                dbc.Input(
                                    id="progressive-start-size",
                                    type="number",
                                    value=51200,  # Half of 102400
                                    min=10240,
                                    max=102400,
                                    step=10240,
                                    className="mb-2"
                                ),
                                html.P(f"Default: {SIGNAL_LENGTH // 2}", className="text-muted small")
                            ], width=6),
                            dbc.Col([
                                dbc.Label("End Size"),
                                dbc.Input(
                                    id="progressive-end-size",
                                    type="number",
                                    value=102400,
                                    min=10240,
                                    max=102400,
                                    step=10240,
                                    className="mb-2"
                                ),
                                html.P(f"Default: {SIGNAL_LENGTH}", className="text-muted small")
                            ], width=6),
                        ])
                    ], id="progressive-config", style={'display': 'none'})
                ], title="Progressive Resizing", item_id="progressive"),

            ], start_collapsed=True, className="mb-3"),

            html.Hr(),

            # Checkpointing & logging
            html.H5("Checkpointing & Logging"),
            dbc.Row([
                dbc.Col([
                    dbc.Checkbox(id="save-checkpoints", label="Save model checkpoints", value=True),
                    dbc.Checkbox(id="save-best-only", label="Save best model only", value=True),
                ], width=6),
                dbc.Col([
                    dbc.Checkbox(id="enable-tensorboard", label="Enable TensorBoard logging", value=True),
                    dbc.Checkbox(id="log-gradients", label="Log gradient statistics", value=False),
                ], width=6),
            ]),
        ])
    ], className="shadow-sm")


def create_step4_review_launch(config):
    """Step 4: Review configuration and launch."""
    return dbc.Card([
        dbc.CardHeader(html.H4("Step 4: Review & Launch")),
        dbc.CardBody([
            html.H5("Experiment Summary", className="mb-3"),

            # Configuration review
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Model Configuration", className="card-title"),
                            html.Hr(),
                            html.Div(id="review-model-config")
                        ])
                    ], className="mb-3")
                ], width=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Training Configuration", className="card-title"),
                            html.Hr(),
                            html.Div(id="review-training-config")
                        ])
                    ], className="mb-3")
                ], width=6),
            ]),

            # Experiment name
            html.Hr(),
            dbc.Label("Experiment Name", html_for="experiment-name"),
            dbc.Input(
                type="text",
                id="experiment-name",
                placeholder="e.g., resnet34_adam_lr0.001_run1",
                className="mb-3"
            ),

            # Tags
            dbc.Label("Tags (optional)"),
            dbc.Input(
                type="text",
                id="experiment-tags",
                placeholder="Comma-separated tags, e.g., baseline, production, hyperopt",
                className="mb-3"
            ),

            # Notes
            dbc.Label("Notes (optional)"),
            dbc.Textarea(
                id="experiment-notes",
                placeholder="Add any notes about this experiment...",
                rows=3,
                className="mb-3"
            ),

            html.Hr(),

            # Launch button
            dbc.Alert([
                html.I(className="fas fa-info-circle me-2"),
                "Training will run in the background. You can close this page and monitor progress from the Experiments page."
            ], color="info", className="mb-3"),

            dbc.Button([
                html.I(className="fas fa-rocket me-2"),
                "Launch Training"
            ], id="launch-training-btn", color="success", size="lg", className="w-100"),

        ])
    ], className="shadow-sm")
