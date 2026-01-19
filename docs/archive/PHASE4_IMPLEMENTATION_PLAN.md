> [!WARNING]
> **Archived Document**
> This document is historical and may be outdated.
> For current information, see the main documentation.
>
> *Archived on: 2026-01-20*
> *Reason: Superseded by consolidated documentation*
# Phase 4 Features Implementation Plan

**Created**: 2025-11-22
**Phase**: Phase 4 - Polish & Advanced Features
**Target**: Complete remaining 3 features to achieve 100% dashboard coverage
**Total Estimated Effort**: 6 days

---

## 📋 Overview

This document provides detailed implementation plans for the final three features:

1. **Notification Management** (1 day) - User-configurable notification preferences
2. **Enhanced Visualization** (2 days) - Advanced signal analysis and embeddings
3. **NAS Dashboard** (3 days) - Neural Architecture Search interface

After these features, the dashboard will have **100% feature coverage**.

---

## ✅ What's Already Complete

### Phase 1: Production Critical (3 features)
- System Monitoring, HPO Campaigns, Deployment Dashboard

### Phase 2: Production Completeness (3 features)
- API Monitoring, Enhanced Evaluation, Testing & QA

### Phase 3: Workflow Enhancements (3 features)
- Dataset Management, Feature Engineering, Advanced Training Options

**Current Coverage**: 75% (9/12 features)

---

## 🎯 Phase 4 Features (Priority Order)

### Priority Rationale

1. **Notification Management** - Quick win (1 day), fixes technical debt, improves UX
2. **Enhanced Visualization** - Medium complexity (2 days), enhances analysis capabilities
3. **NAS Dashboard** - Most complex (3 days), advanced ML feature for power users

---

## 1️⃣ Feature: Notification Management

### Status
- **Backend**: ✅ Complete (`notification_service.py`, `email_provider.py`)
- **Models**: ✅ Complete (`NotificationPreference`, `EmailLog`, `WebhookConfiguration`)
- **UI**: ❌ Missing (no settings interface)

### Problem Statement
Users cannot configure their notification preferences via UI. The backend supports multi-channel notifications (email, webhooks, Slack, Teams), but all settings are hardcoded or database-only. This creates poor UX and makes the notification system unusable for most users.

### Technical Debt Impact
- `notification_preference` database model exists but no UI
- Email provider configured but SMTP settings not manageable
- Webhook configurations exist but not editable

### Implementation Plan

#### Files to Create/Modify

**1. Enhance `/settings` page** - `packages/dashboard/layouts/settings.py`
```python
# Add new tab: "Notifications"
# Structure:
# - Event Preferences Table
#   - Columns: Event Type, Email, In-App, Frequency, Actions
#   - Rows: All EventType constants (training.complete, hpo.campaign_complete, etc.)
# - Email Configuration Card
#   - SMTP Server, Port, Username, Password (masked), TLS toggle
#   - Test Email button
# - Notification History Card
#   - Recent notifications table (last 50)
#   - Columns: Timestamp, Event, Channel, Status, Message preview
# - Webhook Configuration Card (Future - optional)
```

**Changes Required**: Add notification tab after existing tabs (~200 lines)

**2. Create notification callbacks** - `packages/dashboard/callbacks/notification_callbacks.py` (~150 lines)
```python
# Callbacks to implement:

@app.callback(...)
def load_notification_preferences(user_id):
    """Load user's notification preferences from DB."""
    # Query NotificationPreference table
    # Create preferences table with toggles
    # Return table component

@app.callback(...)
def update_notification_preference(n_clicks, event_type, channel, value):
    """Update a single notification preference."""
    # Update NotificationPreference in DB
    # Return success toast

@app.callback(...)
def load_email_config(user_id):
    """Load current email configuration."""
    # Return SMTP settings (password masked)

@app.callback(...)
def update_email_config(smtp_server, port, username, password, use_tls):
    """Update email provider configuration."""
    # Update user email settings
    # Reinitialize NotificationService
    # Return success/error

@app.callback(...)
def send_test_email(n_clicks, email_address):
    """Send test notification email."""
    # Use NotificationService to send test email
    # Return status (success/failure with error message)

@app.callback(...)
def load_notification_history(limit=50):
    """Load recent notification history."""
    # Query EmailLog table
    # Return table with recent notifications

@app.callback(...)
def export_notification_history(format):
    """Export notification history to CSV/JSON."""
    # Query all EmailLog entries
    # Export to selected format
```

**3. Register callbacks** - `packages/dashboard/callbacks/__init__.py`
```python
# Add import and registration:
try:
    from callbacks.notification_callbacks import register_notification_callbacks
    register_notification_callbacks(app)
except ImportError as e:
    print(f"Warning: Could not import notification_callbacks: {e}")
```

#### Database Schema (Already Exists)

```sql
-- notification_preferences table
user_id INTEGER FK → users.id
event_type VARCHAR(50)  -- EventType constant
email_enabled BOOLEAN
in_app_enabled BOOLEAN
slack_enabled BOOLEAN
webhook_enabled BOOLEAN
email_frequency VARCHAR(20)  -- 'immediate', 'digest_daily', 'digest_weekly'

-- email_log table
user_id INTEGER FK → users.id
recipient_email VARCHAR(255)
subject TEXT
body TEXT
status VARCHAR(20)  -- 'pending', 'sent', 'failed'
error_message TEXT
sent_at TIMESTAMP
```

#### UI Components

**Notification Preferences Table**:
| Event Type | Email | In-App | Frequency | Actions |
|------------|-------|--------|-----------|---------|
| Training Complete | ☑️ | ☑️ | Immediate | Edit |
| Training Failed | ☑️ | ☑️ | Immediate | Edit |
| HPO Campaign Complete | ☑️ | ☐ | Daily Digest | Edit |

**Email Configuration Form**:
```
SMTP Server:     smtp.gmail.com
Port:            587
Username:        user@example.com
Password:        ••••••••
Use TLS:         ☑️
[Test Email] [Save Configuration]
```

**Notification History Table**:
| Timestamp | Event | Channel | Status | Message |
|-----------|-------|---------|--------|---------|
| 2025-11-22 14:30 | training.complete | Email | Sent | Experiment "ResNet-50" completed... |
| 2025-11-22 10:15 | hpo.campaign_complete | Email | Sent | HPO campaign "Bayesian-Search-1" found... |

#### Implementation Steps

1. **Day 1 Morning** (2-3 hours):
   - Add notification tab to `settings.py`
   - Create preferences table UI with event list
   - Add email configuration form
   - Add notification history table

2. **Day 1 Afternoon** (3-4 hours):
   - Implement `load_notification_preferences()` callback
   - Implement `update_notification_preference()` callback
   - Implement `load_email_config()` and `update_email_config()`
   - Implement `send_test_email()` callback
   - Implement `load_notification_history()` callback

3. **Testing** (1 hour):
   - Test loading preferences (should show defaults from EventType.get_default_preferences())
   - Test updating preferences (toggle email/in-app, change frequency)
   - Test email configuration (update SMTP settings)
   - Test sending test email (should receive email)
   - Test notification history (should show recent emails)

#### Success Criteria

- ✅ Users can view all notification event types in settings
- ✅ Users can toggle email/in-app notifications per event
- ✅ Users can configure email frequency (immediate/daily digest/weekly digest)
- ✅ Users can update SMTP email settings
- ✅ Users can send test emails to verify configuration
- ✅ Users can view notification history (last 50 notifications)
- ✅ Changes persist to database and take effect immediately
- ✅ Technical debt resolved: notification_preference model has UI

#### Estimated Effort: 1 day

---

## 2️⃣ Feature: Enhanced Visualization

### Status
- **Backend**: ✅ Complete (`visualization/*.py` - 11 visualization modules)
- **UI**: ❌ Missing (no advanced visualization dashboard)

### Problem Statement
The codebase has extensive visualization capabilities (t-SNE, UMAP, bispectrum, wavelet scalograms, saliency maps, feature maps) but they're only accessible via code. Users cannot create advanced visualizations through the dashboard, limiting exploratory analysis.

### Existing Visualization Modules

1. `signal_plots.py` - Basic signal plotting
2. `spectrogram_plots.py` - Time-frequency analysis
3. `feature_visualization.py` - Feature importance and distributions
4. `performance_plots.py` - Training curves, confusion matrices
5. `cnn_visualizer.py` - CNN layer visualizations
6. `activation_maps_2d.py` - 2D activation heatmaps
7. `saliency_maps.py` - Gradient-based saliency
8. `counterfactual_explanations.py` - What-if analysis
9. `xai_dashboard.py` - XAI dashboard utilities
10. `cnn_analysis.py` - CNN architecture analysis

### Implementation Plan

#### Files to Create

**1. Visualization dashboard layout** - `packages/dashboard/layouts/visualization.py` (~300 lines)
```python
def create_visualization_layout():
    """
    Advanced visualization dashboard with tabs.

    Tabs:
    1. Embeddings (t-SNE / UMAP)
    2. Signal Analysis (Bispectrum, Wavelet, Spectrogram)
    3. Feature Analysis (Importance, Correlation, Distributions)
    4. Model Analysis (Saliency, Activation Maps, Counterfactuals)
    """
    return dbc.Container([
        html.H2("Advanced Visualizations"),

        # Dataset/Experiment Selector
        dbc.Row([
            dbc.Col([
                html.Label("Select Dataset:"),
                dcc.Dropdown(id="viz-dataset-select")
            ], width=4),
            dbc.Col([
                html.Label("Select Experiment:"),
                dcc.Dropdown(id="viz-experiment-select")
            ], width=4),
        ]),

        # Tabs for different visualization types
        dbc.Tabs([
            # Tab 1: Embeddings
            dbc.Tab(label="Embeddings", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.RadioItems(
                            id="embedding-method",
                            options=[
                                {"label": "t-SNE", "value": "tsne"},
                                {"label": "UMAP", "value": "umap"},
                                {"label": "PCA", "value": "pca"}
                            ],
                            value="tsne"
                        ),
                        html.Label("Perplexity (t-SNE):", id="tsne-perplexity-label"),
                        dcc.Slider(id="tsne-perplexity", min=5, max=50, value=30, step=5),
                        html.Label("n_neighbors (UMAP):", id="umap-neighbors-label"),
                        dcc.Slider(id="umap-neighbors", min=5, max=50, value=15, step=5),
                        dbc.Button("Generate Embedding", id="generate-embedding-btn", color="primary")
                    ], width=3),
                    dbc.Col([
                        dcc.Loading(
                            dcc.Graph(id="embedding-plot")
                        )
                    ], width=9)
                ])
            ]),

            # Tab 2: Signal Analysis
            dbc.Tab(label="Signal Analysis", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.RadioItems(
                            id="signal-viz-type",
                            options=[
                                {"label": "Bispectrum", "value": "bispectrum"},
                                {"label": "Wavelet Scalogram", "value": "wavelet"},
                                {"label": "Spectrogram", "value": "spectrogram"}
                            ],
                            value="bispectrum"
                        ),
                        html.Label("Select Signal:"),
                        dcc.Dropdown(id="signal-select"),
                        dbc.Button("Generate Plot", id="generate-signal-viz-btn", color="primary")
                    ], width=3),
                    dbc.Col([
                        dcc.Loading(
                            dcc.Graph(id="signal-viz-plot")
                        )
                    ], width=9)
                ])
            ]),

            # Tab 3: Feature Analysis
            dbc.Tab(label="Feature Analysis", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.RadioItems(
                            id="feature-viz-type",
                            options=[
                                {"label": "Feature Importance", "value": "importance"},
                                {"label": "Feature Correlation", "value": "correlation"},
                                {"label": "Feature Distributions", "value": "distributions"}
                            ],
                            value="importance"
                        ),
                        html.Label("Top N Features:"),
                        dcc.Slider(id="top-n-features", min=5, max=50, value=15, step=5),
                        dbc.Button("Generate Plot", id="generate-feature-viz-btn", color="primary")
                    ], width=3),
                    dbc.Col([
                        dcc.Loading(
                            dcc.Graph(id="feature-viz-plot")
                        )
                    ], width=9)
                ])
            ]),

            # Tab 4: Model Analysis
            dbc.Tab(label="Model Analysis", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.RadioItems(
                            id="model-viz-type",
                            options=[
                                {"label": "Saliency Maps", "value": "saliency"},
                                {"label": "Activation Maps", "value": "activation"},
                                {"label": "Counterfactual", "value": "counterfactual"}
                            ],
                            value="saliency"
                        ),
                        html.Label("Select Layer:"),
                        dcc.Dropdown(id="layer-select"),
                        html.Label("Select Sample:"),
                        dcc.Dropdown(id="sample-select"),
                        dbc.Button("Generate Viz", id="generate-model-viz-btn", color="primary")
                    ], width=3),
                    dbc.Col([
                        dcc.Loading(
                            dcc.Graph(id="model-viz-plot")
                        )
                    ], width=9)
                ])
            ])
        ])
    ])
```

**2. Visualization callbacks** - `packages/dashboard/callbacks/visualization_callbacks.py` (~250 lines)
```python
def register_visualization_callbacks(app):
    """Register all visualization dashboard callbacks."""

    # Dataset/Experiment loaders
    @app.callback(...)
    def load_datasets():
        """Load available datasets."""
        # Query Dataset table
        # Return dropdown options

    @app.callback(...)
    def load_experiments(dataset_id):
        """Load experiments for selected dataset."""
        # Query Experiment table filtered by dataset
        # Return dropdown options

    # Tab 1: Embeddings
    @app.callback(...)
    def generate_embedding(n_clicks, dataset_id, method, perplexity, neighbors):
        """Generate t-SNE/UMAP/PCA embedding visualization."""
        # Load dataset features
        # Apply dimensionality reduction (sklearn.manifold)
        # Create scatter plot colored by fault type
        # Return plotly figure

    # Tab 2: Signal Analysis
    @app.callback(...)
    def load_signals(dataset_id):
        """Load signal options for selected dataset."""
        # Load signal indices from dataset
        # Return dropdown options

    @app.callback(...)
    def generate_signal_visualization(n_clicks, dataset_id, signal_idx, viz_type):
        """Generate bispectrum/wavelet/spectrogram plot."""
        if viz_type == 'bispectrum':
            # Use existing code from visualization/spectrogram_plots.py
            # Generate bispectrum heatmap
        elif viz_type == 'wavelet':
            # Generate wavelet scalogram
        elif viz_type == 'spectrogram':
            # Generate spectrogram
        # Return plotly figure

    # Tab 3: Feature Analysis
    @app.callback(...)
    def generate_feature_visualization(n_clicks, dataset_id, viz_type, top_n):
        """Generate feature importance/correlation/distribution plots."""
        if viz_type == 'importance':
            # Use visualization/feature_visualization.py
            # Generate importance bar chart
        elif viz_type == 'correlation':
            # Generate correlation heatmap
        elif viz_type == 'distributions':
            # Generate feature distribution plots
        # Return plotly figure

    # Tab 4: Model Analysis
    @app.callback(...)
    def load_layers(experiment_id):
        """Load model layers for selected experiment."""
        # Load model architecture
        # Return layer names

    @app.callback(...)
    def load_samples(dataset_id):
        """Load sample indices."""
        # Return sample options

    @app.callback(...)
    def generate_model_visualization(n_clicks, experiment_id, sample_idx, layer, viz_type):
        """Generate saliency/activation/counterfactual visualization."""
        if viz_type == 'saliency':
            # Use visualization/saliency_maps.py
            # Generate saliency map
        elif viz_type == 'activation':
            # Use visualization/activation_maps_2d.py
            # Generate activation heatmap
        elif viz_type == 'counterfactual':
            # Use visualization/counterfactual_explanations.py
            # Generate counterfactual explanation
        # Return plotly figure
```

**3. Add route** - `packages/dashboard/callbacks/__init__.py`
```python
elif pathname == '/visualization':
    from layouts.visualization import create_visualization_layout
    return create_visualization_layout()

# Register callbacks
try:
    from callbacks.visualization_callbacks import register_visualization_callbacks
    register_visualization_callbacks(app)
except ImportError as e:
    print(f"Warning: Could not import visualization_callbacks: {e}")
```

**4. Add sidebar link** - `packages/dashboard/components/sidebar.py`
```python
dbc.NavLink([
    html.I(className="fas fa-chart-area me-2"),
    "Visualizations"
], href="/visualization", active="exact"),
```

#### Implementation Steps

**Day 1** (4-5 hours):
- Create `visualization.py` layout with 4 tabs
- Add dataset/experiment selectors
- Build Embeddings tab (t-SNE/UMAP controls)
- Build Signal Analysis tab (bispectrum/wavelet/spectrogram)

**Day 2 Morning** (3-4 hours):
- Build Feature Analysis tab
- Build Model Analysis tab
- Add sidebar link and route

**Day 2 Afternoon** (3-4 hours):
- Implement all callbacks in `visualization_callbacks.py`
- Integrate with existing visualization modules
- Test all visualization types

#### Success Criteria

- ✅ Users can generate t-SNE/UMAP embeddings for datasets
- ✅ Users can create bispectrum plots for signals
- ✅ Users can create wavelet scalograms
- ✅ Users can view feature importance/correlation/distributions
- ✅ Users can generate saliency maps for model predictions
- ✅ Users can view activation maps for CNN layers
- ✅ All visualizations are interactive Plotly charts
- ✅ Visualizations can be exported (PNG/PDF/HTML via Plotly)

#### Estimated Effort: 2 days

---

## 3️⃣ Feature: NAS (Neural Architecture Search) Dashboard

### Status
- **Backend**: ⚠️ Partial (`models/nas/search_space.py` defines search space)
- **Search Algorithm**: ❌ Missing (no NAS implementation)
- **UI**: ❌ Missing (no NAS dashboard)

### Problem Statement
NAS is a cutting-edge ML technique for automatically discovering optimal architectures. The codebase defines a search space (conv operations, pooling, skip connections) but has no NAS algorithm implementation or UI. This prevents users from leveraging automated architecture discovery.

### Existing Code

**Search Space** (`models/nas/search_space.py`):
- `OperationType` enum: Conv kernels (3, 5, 7), separable conv, dilated conv, pooling, skip connections
- `SearchSpaceConfig`: Configures operations, nodes, cells, channel sizes
- Search space designed for 1D signal processing

### Implementation Plan

#### Architecture Decision: Search Algorithm

**Option 1: Random Search** (Recommended for MVP)
- Simple to implement (~200 lines)
- Proven effective baseline
- Fast convergence for small search spaces
- No complex dependencies

**Option 2: Bayesian Optimization**
- More sample-efficient
- Requires Optuna integration (already used for HPO)
- Medium complexity (~300 lines)

**Option 3: DARTS (Differentiable Architecture Search)**
- State-of-the-art
- Complex implementation (~1000+ lines)
- Requires gradient-based optimization
- Overkill for MVP

**Decision**: Implement **Random Search** for MVP, with architecture to support Bayesian later.

#### Files to Create

**1. NAS service layer** - `packages/dashboard/services/nas_service.py` (~300 lines)
```python
class NASService:
    """Service for Neural Architecture Search operations."""

    @staticmethod
    def create_nas_campaign(
        name: str,
        dataset_id: int,
        search_space_config: Dict,
        search_algorithm: str = 'random',
        num_trials: int = 20,
        max_epochs_per_trial: int = 10
    ) -> int:
        """
        Create a new NAS campaign.

        Args:
            name: Campaign name
            dataset_id: Dataset to use
            search_space_config: SearchSpaceConfig parameters
            search_algorithm: 'random', 'bayesian', 'evolution'
            num_trials: Number of architectures to try
            max_epochs_per_trial: Training epochs per architecture

        Returns:
            Campaign ID
        """
        # Create NASCampaign database record
        # Return campaign_id

    @staticmethod
    def sample_architecture(search_space_config: SearchSpaceConfig) -> Dict:
        """
        Sample a random architecture from search space.

        Returns:
            Architecture dict with:
            - operations: List of operation types per edge
            - connections: List of (from_node, to_node) tuples
            - channels: List of channel sizes per layer
        """
        # Randomly sample operations
        # Randomly sample connections (ensuring DAG)
        # Randomly sample channel sizes
        # Return architecture specification

    @staticmethod
    def get_campaign_details(campaign_id: int) -> Dict:
        """Get NAS campaign details with all trials."""
        # Load NASCampaign from DB
        # Load all associated NASTrials
        # Return campaign info + trials

    @staticmethod
    def get_best_architecture(campaign_id: int) -> Dict:
        """Get best performing architecture from campaign."""
        # Query NASTrials ordered by validation_accuracy DESC
        # Return top architecture

    @staticmethod
    def export_architecture(trial_id: int, format: str = 'pytorch') -> str:
        """
        Export discovered architecture as code.

        Args:
            trial_id: NAS trial ID
            format: 'pytorch', 'onnx', 'config_json'

        Returns:
            Architecture code/config as string
        """
        # Load architecture from NASTrial
        # Generate PyTorch model code
        # Return code string
```

**2. NAS Celery tasks** - `packages/dashboard/tasks/nas_tasks.py` (~300 lines)
```python
@celery_app.task(bind=True)
def run_nas_campaign_task(self, campaign_id: int):
    """
    Run NAS campaign in background.

    For each trial:
    1. Sample architecture from search space
    2. Build PyTorch model
    3. Train for max_epochs_per_trial
    4. Evaluate on validation set
    5. Save architecture + metrics
    6. Update campaign progress
    """
    # Load campaign config
    campaign = session.query(NASCampaign).get(campaign_id)

    for trial_idx in range(campaign.num_trials):
        # Update progress
        self.update_state(
            state='PROGRESS',
            meta={
                'current_trial': trial_idx + 1,
                'total_trials': campaign.num_trials,
                'status': f'Training architecture {trial_idx + 1}/{campaign.num_trials}'
            }
        )

        # Sample architecture
        architecture = NASService.sample_architecture(campaign.search_space_config)

        # Build model
        model = build_model_from_architecture(architecture)

        # Train briefly
        train_results = train_model_quick(
            model=model,
            dataset_id=campaign.dataset_id,
            num_epochs=campaign.max_epochs_per_trial
        )

        # Save trial
        trial = NASTrial(
            campaign_id=campaign_id,
            architecture=architecture,
            validation_accuracy=train_results['val_accuracy'],
            training_time=train_results['time'],
            num_parameters=count_parameters(model),
            flops=calculate_flops(model)
        )
        session.add(trial)
        session.commit()

    # Mark campaign complete
    campaign.status = 'completed'
    session.commit()

    return {
        "success": True,
        "num_trials": campaign.num_trials,
        "best_accuracy": max([t.validation_accuracy for t in campaign.trials])
    }


def build_model_from_architecture(architecture: Dict) -> nn.Module:
    """Build PyTorch model from NAS architecture specification."""
    # Parse architecture dict
    # Build sequential model with specified operations
    # Return model


def train_model_quick(model, dataset_id, num_epochs) -> Dict:
    """Quick training for NAS trial."""
    # Load dataset
    # Train for num_epochs
    # Return validation accuracy and time
```

**3. NAS database models** - `packages/dashboard/models/nas_campaign.py` (~100 lines)
```python
class NASCampaign(BaseModel):
    """NAS campaign tracking."""
    __tablename__ = 'nas_campaigns'

    name = Column(String(200), nullable=False)
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    search_algorithm = Column(String(50))  # 'random', 'bayesian', 'evolution'
    num_trials = Column(Integer)
    max_epochs_per_trial = Column(Integer)
    search_space_config = Column(JSON)
    status = Column(String(50))  # 'running', 'completed', 'failed'

    # Relationships
    trials = relationship("NASTrial", backref="campaign")


class NASTrial(BaseModel):
    """Individual NAS trial (one architecture evaluation)."""
    __tablename__ = 'nas_trials'

    campaign_id = Column(Integer, ForeignKey('nas_campaigns.id'))
    architecture = Column(JSON)  # Full architecture specification
    validation_accuracy = Column(Float)
    training_time = Column(Float)  # seconds
    num_parameters = Column(Integer)
    flops = Column(BigInteger)  # FLOPs count
```

**4. NAS dashboard layout** - `packages/dashboard/layouts/nas_dashboard.py` (~400 lines)
```python
def create_nas_dashboard_layout():
    """NAS campaign dashboard."""
    return dbc.Container([
        html.H2("Neural Architecture Search"),

        # Campaign creation card
        dbc.Card([
            dbc.CardHeader("Create NAS Campaign"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Campaign Name:"),
                        dbc.Input(id="nas-campaign-name", placeholder="e.g., CNN-Search-1")
                    ], width=6),
                    dbc.Col([
                        html.Label("Dataset:"),
                        dcc.Dropdown(id="nas-dataset-select")
                    ], width=6)
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Label("Search Algorithm:"),
                        dbc.RadioItems(
                            id="nas-search-algorithm",
                            options=[
                                {"label": "Random Search", "value": "random"},
                                {"label": "Bayesian Optimization", "value": "bayesian", "disabled": True},
                            ],
                            value="random"
                        )
                    ], width=4),
                    dbc.Col([
                        html.Label("Number of Trials:"),
                        dbc.Input(id="nas-num-trials", type="number", value=20, min=5, max=100)
                    ], width=4),
                    dbc.Col([
                        html.Label("Epochs per Trial:"),
                        dbc.Input(id="nas-epochs-per-trial", type="number", value=10, min=5, max=50)
                    ], width=4)
                ]),
                dbc.Button("Launch NAS Campaign", id="launch-nas-btn", color="primary")
            ])
        ], className="mb-4"),

        # Campaigns list
        html.H4("NAS Campaigns"),
        html.Div(id="nas-campaigns-table"),

        # Campaign details modal
        dbc.Modal([
            dbc.ModalHeader("NAS Campaign Details"),
            dbc.ModalBody([
                # Campaign info
                html.Div(id="nas-campaign-info"),

                # Trials table
                html.H5("Trials"),
                html.Div(id="nas-trials-table"),

                # Best architecture
                html.H5("Best Architecture"),
                html.Div(id="nas-best-architecture"),

                # Architecture visualization
                dcc.Graph(id="nas-architecture-graph"),

                # Export button
                dbc.Button("Export Best Architecture", id="export-nas-architecture-btn")
            ])
        ], id="nas-campaign-modal", size="xl")
    ])
```

**5. NAS callbacks** - `packages/dashboard/callbacks/nas_callbacks.py` (~350 lines)
```python
def register_nas_callbacks(app):
    """Register NAS dashboard callbacks."""

    @app.callback(...)
    def load_datasets():
        """Load datasets for NAS."""
        # Query Dataset table
        # Return dropdown options

    @app.callback(...)
    def launch_nas_campaign(n_clicks, name, dataset_id, algorithm, num_trials, epochs_per_trial):
        """Launch NAS campaign."""
        # Create NASCampaign in DB
        # Launch run_nas_campaign_task.delay(campaign_id)
        # Return success toast

    @app.callback(...)
    def load_nas_campaigns():
        """Load all NAS campaigns."""
        # Query NASCampaign table
        # Return campaigns table

    @app.callback(...)
    def view_campaign_details(campaign_id):
        """Load campaign details for modal."""
        # Load NASCampaign + trials
        # Return campaign info, trials table, best architecture

    @app.callback(...)
    def visualize_architecture(trial_id):
        """Create network graph visualization of architecture."""
        # Load architecture from NASTrial
        # Create networkx graph
        # Convert to plotly network diagram
        # Return figure

    @app.callback(...)
    def export_architecture(trial_id):
        """Export architecture as PyTorch code."""
        # Call NASService.export_architecture()
        # Trigger download
```

**6. Add route and sidebar**
- Add `/nas` route to callbacks/__init__.py
- Add "NAS" link to sidebar

#### Implementation Steps

**Day 1** (6-7 hours):
- Create NAS database models (NASCampaign, NASTrial)
- Run migrations
- Create `nas_service.py` with architecture sampling
- Create `nas_dashboard.py` layout

**Day 2** (6-7 hours):
- Implement `nas_tasks.py` (background NAS execution)
- Implement `build_model_from_architecture()` helper
- Implement `train_model_quick()` helper
- Test architecture sampling and model building

**Day 3** (6-7 hours):
- Implement all NAS callbacks
- Add architecture visualization (network graph)
- Add architecture export (PyTorch code generation)
- End-to-end testing (launch campaign, monitor trials, view results)

#### Success Criteria

- ✅ Users can create NAS campaigns with custom search space
- ✅ NAS runs in background via Celery
- ✅ Users can monitor NAS progress (current trial, best architecture so far)
- ✅ Users can view all trials with accuracy/params/FLOPs
- ✅ Users can visualize discovered architectures as network graphs
- ✅ Users can export best architecture as PyTorch code
- ✅ Random search effectively explores search space
- ✅ Best architecture achieves competitive accuracy

#### Estimated Effort: 3 days

---

## 📊 Summary

### Total Scope

| Feature | Files Created | Files Modified | Lines of Code | Estimated Days |
|---------|---------------|----------------|---------------|----------------|
| Notification Management | 1 | 2 | ~350 | 1 |
| Enhanced Visualization | 2 | 2 | ~550 | 2 |
| NAS Dashboard | 5 | 2 | ~1450 | 3 |
| **Total** | **8** | **6** | **~2350** | **6** |

### Implementation Order

1. **Notification Management** (Day 1) - Quick win, fixes technical debt
2. **Enhanced Visualization** (Days 2-3) - Leverages existing code
3. **NAS Dashboard** (Days 4-6) - Most complex, requires new backend logic

### Risk Assessment

**Low Risk**:
- Notification Management (simple CRUD UI over existing backend)

**Medium Risk**:
- Enhanced Visualization (integration with 11 existing visualization modules)

**High Risk**:
- NAS Dashboard (new NAS algorithm, architecture building, complex UI)

### Mitigation Strategies

1. **Visualization**: Start with 1-2 viz types, expand incrementally
2. **NAS**: Implement simple random search first, defer Bayesian to future
3. **Testing**: Test each feature independently before moving to next

---

## 🎯 Success Metrics

After Phase 4 completion:

- **Dashboard Feature Coverage**: 100% (12/12 features)
- **Technical Debt**: Zero (all database models have UIs)
- **User Capabilities**: Complete ML workflow from data generation to architecture search
- **Production Readiness**: Full monitoring, testing, deployment, and notification system

---

## 📞 Next Steps

Once you approve this plan:

1. Commit `PHASE4_IMPLEMENTATION_PLAN.md`
2. Update `REMAINING_FEATURES.md` (already done)
3. Begin implementation:
   - Day 1: Notification Management
   - Days 2-3: Enhanced Visualization
   - Days 4-6: NAS Dashboard
4. Create final completion report

**Ready to proceed?** 🚀
