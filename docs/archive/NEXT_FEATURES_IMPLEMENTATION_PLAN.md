# Next Features Implementation Plan

**Features**: Dataset Management, Feature Engineering, Advanced Training Options
**Estimated Total Effort**: 6 days
**Generated**: 2025-11-22

---

## 🎯 Quick Start: Fix Broken Sidebar Links (5 minutes)

### Before We Begin: Clean Up Broken Navigation

**Problem**: `/statistics/compare` and `/analytics` sidebar links lead to 404

**Solution**: Remove from sidebar until implemented

**File to Modify**: `packages/dashboard/components/sidebar.py`

**Changes**:
```python
# REMOVE these lines (62-69):
            dbc.NavLink([
                html.I(className="fas fa-chart-bar me-2"),
                "Statistics"
            ], href="/statistics/compare", active="exact"),
            dbc.NavLink([
                html.I(className="fas fa-chart-line me-2"),
                "Analytics"
            ], href="/analytics", active="exact"),
```

**Commit**: `fix: Remove unimplemented sidebar links (statistics, analytics)`

---

# Feature 1: Dataset Management Page (1 day)

## Overview
Fix the broken `/datasets` link by implementing a complete dataset management interface.

## Architecture

### Database Schema (Already Exists ✅)
**File**: `packages/dashboard/models/dataset.py`

```python
class Dataset(BaseModel):
    name: String(255) - Dataset name (unique)
    description: String(1000) - Description
    num_signals: Integer - Number of signals
    fault_types: JSON - List of fault types
    severity_levels: JSON - List of severity levels
    file_path: String(500) - Path to HDF5 file
    metadata: JSON - Additional metadata
    created_by: Integer - Foreign key to users
```

**Status**: ✅ Model exists, no changes needed

---

## Implementation Plan

### File 1: `packages/dashboard/services/dataset_service.py` (~200 lines)

**Purpose**: Business logic for dataset operations

**Methods to Implement**:

```python
class DatasetService:
    @staticmethod
    def list_datasets(limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Get list of all datasets with pagination.

        Returns:
            List of dataset dictionaries with:
            - id, name, description
            - num_signals, fault_types, severity_levels
            - file_path, created_at
            - file_size_mb (calculated from file_path)
        """
        pass

    @staticmethod
    def get_dataset_details(dataset_id: int) -> Optional[Dict]:
        """
        Get detailed dataset information.

        Returns:
            - Basic info (from database)
            - File statistics (from HDF5 file)
            - Signal statistics (mean, std, min, max per fault type)
            - Class distribution (count per fault type)
        """
        pass

    @staticmethod
    def get_dataset_preview(dataset_id: int, num_samples: int = 5) -> Dict:
        """
        Get preview of dataset signals.

        Returns:
            - Sample signals (first N signals per fault type)
            - Signal plots (time-domain preview)
        """
        pass

    @staticmethod
    def delete_dataset(dataset_id: int, delete_file: bool = False) -> bool:
        """
        Delete dataset from database (and optionally file).

        Args:
            dataset_id: Dataset ID
            delete_file: If True, also delete HDF5 file

        Returns:
            True if successful
        """
        pass

    @staticmethod
    def archive_dataset(dataset_id: int) -> bool:
        """
        Archive dataset (mark as archived in metadata).
        """
        pass

    @staticmethod
    def export_dataset(dataset_id: int, format: str = 'hdf5') -> str:
        """
        Export dataset to different format.

        Args:
            dataset_id: Dataset ID
            format: 'hdf5', 'mat', 'csv'

        Returns:
            Path to exported file
        """
        pass

    @staticmethod
    def get_dataset_statistics(dataset_id: int) -> Dict:
        """
        Compute dataset statistics.

        Returns:
            - Total signals
            - Signals per fault type
            - Signal length statistics
            - Sampling rate
            - Fault type distribution (%)
        """
        pass
```

**Dependencies**:
- `h5py` - Read HDF5 files
- `numpy` - Signal statistics
- `pathlib` - File operations

**Estimated Time**: 3 hours

---

### File 2: `packages/dashboard/layouts/datasets.py` (~350 lines)

**Purpose**: Dataset management UI

**Structure**:

```python
def create_datasets_layout():
    """Create datasets management page."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="fas fa-folder me-3"),
                    "Dataset Management"
                ]),
                html.P("Manage and explore your datasets")
            ], width=8),
            dbc.Col([
                dbc.Button(
                    [html.I(className="fas fa-plus me-2"), "New Dataset"],
                    id="new-dataset-btn",
                    color="primary",
                    href="/data-generation"  # Redirect to data generation
                )
            ], width=4, className="text-end")
        ], className="mb-4"),

        # Dataset Table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col([
                                html.H5("Datasets", className="mb-0")
                            ], width=6),
                            dbc.Col([
                                dbc.Input(
                                    id="dataset-search",
                                    placeholder="Search datasets...",
                                    type="text"
                                )
                            ], width=6)
                        ])
                    ]),
                    dbc.CardBody([
                        html.Div(id="datasets-table")
                    ])
                ])
            ])
        ], className="mb-4"),

        # Dataset Details Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(id="dataset-details-title")),
            dbc.ModalBody([
                # Statistics
                html.H5("Statistics"),
                html.Div(id="dataset-stats-cards"),

                # Class Distribution Chart
                html.H5("Class Distribution", className="mt-3"),
                dcc.Graph(id="dataset-class-distribution"),

                # Signal Preview
                html.H5("Signal Preview", className="mt-3"),
                dcc.Graph(id="dataset-signal-preview"),

                # Actions
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="fas fa-download me-2"), "Export"],
                            id="export-dataset-btn",
                            color="info",
                            className="w-100"
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="fas fa-archive me-2"), "Archive"],
                            id="archive-dataset-btn",
                            color="warning",
                            className="w-100"
                        )
                    ], width=4),
                    dbc.Col([
                        dbc.Button(
                            [html.I(className="fas fa-trash me-2"), "Delete"],
                            id="delete-dataset-btn",
                            color="danger",
                            className="w-100"
                        )
                    ], width=4),
                ])
            ]),
            dbc.ModalFooter([
                dbc.Button("Close", id="close-dataset-modal", className="ms-auto")
            ])
        ], id="dataset-details-modal", size="xl", is_open=False),

        # Export Format Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Export Dataset")),
            dbc.ModalBody([
                dbc.Label("Select Export Format"),
                dbc.RadioItems(
                    id="export-format",
                    options=[
                        {"label": "HDF5 (.h5)", "value": "hdf5"},
                        {"label": "MAT (.mat)", "value": "mat"},
                        {"label": "CSV (.csv)", "value": "csv"}
                    ],
                    value="hdf5"
                )
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="cancel-export", color="secondary"),
                dbc.Button("Export", id="confirm-export", color="primary")
            ])
        ], id="export-modal", is_open=False),

        # Storage
        dcc.Store(id='selected-dataset-id'),

        # Refresh interval
        dcc.Interval(id='datasets-refresh', interval=30*1000, n_intervals=0)

    ], fluid=True)
```

**Components**:
1. **Header**: Title + "New Dataset" button (redirects to data generation)
2. **Dataset Table**: List all datasets with search/filter
3. **Details Modal**: Show dataset statistics, class distribution, signal preview
4. **Export Modal**: Select format (HDF5, MAT, CSV) and export
5. **Action Buttons**: Export, Archive, Delete

**Estimated Time**: 4 hours

---

### File 3: `packages/dashboard/callbacks/datasets_callbacks.py` (~300 lines)

**Purpose**: Handle dataset UI interactions

**Callbacks to Implement**:

```python
def register_datasets_callbacks(app):
    """Register dataset management callbacks."""

    @app.callback(
        Output('datasets-table', 'children'),
        [Input('datasets-refresh', 'n_intervals'),
         Input('dataset-search', 'value')]
    )
    def load_datasets(n_intervals, search_query):
        """Load and display datasets table."""
        # Get datasets from service
        datasets = DatasetService.list_datasets()

        # Filter by search query
        if search_query:
            datasets = [d for d in datasets if search_query.lower() in d['name'].lower()]

        # Create DataTable
        table = create_datasets_table(datasets)
        return table

    @app.callback(
        [Output('dataset-details-modal', 'is_open'),
         Output('dataset-details-title', 'children'),
         Output('dataset-stats-cards', 'children'),
         Output('dataset-class-distribution', 'figure'),
         Output('dataset-signal-preview', 'figure'),
         Output('selected-dataset-id', 'data')],
        [Input({'type': 'dataset-row', 'index': ALL}, 'n_clicks')],
        [State('dataset-details-modal', 'is_open')],
        prevent_initial_call=True
    )
    def show_dataset_details(n_clicks, is_open):
        """Show dataset details modal."""
        # Get clicked dataset ID from ctx
        # Load dataset details
        # Create stats cards
        # Create class distribution pie chart
        # Create signal preview plots
        pass

    @app.callback(
        Output('dataset-details-modal', 'is_open', allow_duplicate=True),
        Input('close-dataset-modal', 'n_clicks'),
        prevent_initial_call=True
    )
    def close_modal(n_clicks):
        """Close dataset details modal."""
        return False

    @app.callback(
        Output('export-modal', 'is_open'),
        [Input('export-dataset-btn', 'n_clicks'),
         Input('cancel-export', 'n_clicks'),
         Input('confirm-export', 'n_clicks')],
        [State('export-modal', 'is_open'),
         State('selected-dataset-id', 'data'),
         State('export-format', 'value')],
        prevent_initial_call=True
    )
    def handle_export(export_click, cancel_click, confirm_click, is_open, dataset_id, format):
        """Handle dataset export."""
        # If confirm clicked, export dataset
        # Return success/error message
        # Toggle modal
        pass

    @app.callback(
        Output('datasets-table', 'children', allow_duplicate=True),
        Input('delete-dataset-btn', 'n_clicks'),
        [State('selected-dataset-id', 'data')],
        prevent_initial_call=True
    )
    def delete_dataset(n_clicks, dataset_id):
        """Delete dataset."""
        # Confirm deletion (could use dbc.Modal for confirmation)
        # Delete from database
        # Refresh table
        pass

    @app.callback(
        Output('datasets-table', 'children', allow_duplicate=True),
        Input('archive-dataset-btn', 'n_clicks'),
        [State('selected-dataset-id', 'data')],
        prevent_initial_call=True
    )
    def archive_dataset(n_clicks, dataset_id):
        """Archive dataset."""
        # Mark as archived
        # Refresh table
        pass


def create_datasets_table(datasets: List[Dict]) -> dash_table.DataTable:
    """Create datasets DataTable."""
    columns = [
        {'name': 'Name', 'id': 'name'},
        {'name': '# Signals', 'id': 'num_signals'},
        {'name': 'Fault Types', 'id': 'fault_types'},
        {'name': 'Size (MB)', 'id': 'file_size_mb'},
        {'name': 'Created', 'id': 'created_at'},
        {'name': 'Actions', 'id': 'actions'}
    ]

    # Format data for table
    data = []
    for ds in datasets:
        data.append({
            'name': ds['name'],
            'num_signals': ds['num_signals'],
            'fault_types': ', '.join(ds['fault_types'][:3]),  # First 3
            'file_size_mb': f"{ds['file_size_mb']:.1f}",
            'created_at': ds['created_at'].strftime('%Y-%m-%d'),
            'actions': 'View'  # Button to view details
        })

    return dash_table.DataTable(
        data=data,
        columns=columns,
        page_size=20,
        style_cell={'textAlign': 'left'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        row_selectable='single'
    )


def create_class_distribution_chart(stats: Dict) -> go.Figure:
    """Create class distribution pie chart."""
    fig = go.Figure(data=[go.Pie(
        labels=list(stats['class_counts'].keys()),
        values=list(stats['class_counts'].values()),
        hole=0.3
    )])

    fig.update_layout(
        title="Fault Type Distribution",
        height=400
    )

    return fig


def create_signal_preview_chart(preview_data: Dict) -> go.Figure:
    """Create signal preview plot."""
    fig = go.Figure()

    for fault_type, signals in preview_data.items():
        for i, signal in enumerate(signals[:3]):  # First 3 signals per fault
            fig.add_trace(go.Scatter(
                y=signal,
                mode='lines',
                name=f'{fault_type} (sample {i+1})',
                line=dict(width=1)
            ))

    fig.update_layout(
        title="Signal Preview (First 3 per Fault Type)",
        xaxis_title="Sample Index",
        yaxis_title="Amplitude",
        height=400,
        hovermode='closest'
    )

    return fig
```

**Estimated Time**: 4 hours

---

### File 4: Update `packages/dashboard/callbacks/__init__.py`

**Add route**:

```python
        elif pathname == '/datasets':
            from layouts.datasets import create_datasets_layout
            return create_datasets_layout()
```

**Register callbacks**:

```python
    # Import and register dataset callbacks
    try:
        from callbacks.datasets_callbacks import register_datasets_callbacks
        register_datasets_callbacks(app)
    except ImportError as e:
        print(f"Warning: Could not import datasets_callbacks: {e}")
```

**Estimated Time**: 5 minutes

---

## Testing Checklist

- [ ] Navigate to `/datasets` - page loads without errors
- [ ] Dataset table displays all datasets
- [ ] Search functionality filters datasets
- [ ] Click on dataset row opens details modal
- [ ] Statistics cards show correct data
- [ ] Class distribution chart displays correctly
- [ ] Signal preview plots render correctly
- [ ] Export button triggers export modal
- [ ] Export to HDF5/MAT/CSV works
- [ ] Archive button marks dataset as archived
- [ ] Delete button removes dataset
- [ ] Auto-refresh updates table every 30 seconds

---

## Total Effort: Dataset Management
- Service layer: 3 hours
- Layout: 4 hours
- Callbacks: 4 hours
- Route registration: 5 minutes
- Testing: 1 hour

**Total: ~1 day (8 hours)**

---

# Feature 2: Feature Engineering Dashboard (3 days)

## Overview
Create a comprehensive feature engineering interface leveraging the extensive feature extraction library.

## Architecture

### Existing Code (Already Available ✅)

**Feature Extraction**:
- `features/feature_extractor.py` - Base feature extraction
- `features/advanced_features.py` - Statistical features
- `features/time_domain.py` - Time-domain features
- `features/frequency_domain.py` - Frequency-domain features
- `features/wavelet_features.py` - Wavelet features
- `features/bispectrum_features.py` - Bispectrum features

**Feature Selection**:
- `features/feature_selector.py` - Selection algorithms
- `features/feature_importance.py` - SHAP, permutation importance

---

## Implementation Plan

### File 1: `packages/dashboard/services/feature_service.py` (~300 lines)

**Purpose**: Business logic for feature engineering

**Methods**:

```python
class FeatureService:
    @staticmethod
    def extract_features(
        dataset_id: int,
        domain: str,  # 'time', 'frequency', 'wavelet', 'bispectrum'
        config: Dict
    ) -> Dict:
        """
        Extract features from dataset.

        Returns:
            - features: np.ndarray [N, num_features]
            - feature_names: List[str]
            - extraction_time: float (seconds)
        """
        pass

    @staticmethod
    def compute_feature_importance(
        features: np.ndarray,
        targets: np.ndarray,
        method: str = 'shap'  # 'shap', 'permutation', 'mutual_info'
    ) -> Dict:
        """
        Compute feature importance.

        Returns:
            - importance_scores: List[float]
            - feature_names: List[str]
            - method: str
        """
        pass

    @staticmethod
    def select_features(
        features: np.ndarray,
        targets: np.ndarray,
        method: str,  # 'variance', 'mutual_info', 'rfe'
        num_features: int
    ) -> Dict:
        """
        Select top features.

        Returns:
            - selected_indices: List[int]
            - selected_names: List[str]
            - scores: List[float]
        """
        pass

    @staticmethod
    def compute_feature_correlation(features: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Compute feature correlation matrix.

        Returns:
            - correlation_matrix: np.ndarray [num_features, num_features]
            - feature_names: List[str]
        """
        pass

    @staticmethod
    def create_feature_pipeline(steps: List[Dict]) -> Dict:
        """
        Create feature engineering pipeline.

        Args:
            steps: List of pipeline steps
                [
                    {"type": "extract", "domain": "time", "config": {...}},
                    {"type": "select", "method": "variance", "threshold": 0.1}
                ]

        Returns:
            - pipeline: Serializable pipeline config
            - num_features_out: int
        """
        pass

    @staticmethod
    def save_features(
        dataset_id: int,
        features: np.ndarray,
        feature_names: List[str],
        pipeline_config: Dict
    ) -> int:
        """
        Save extracted features for use in experiments.

        Returns:
            feature_set_id: int
        """
        pass
```

**Estimated Time**: 1 day

---

### File 2: `packages/dashboard/tasks/feature_tasks.py` (~250 lines)

**Purpose**: Background tasks for feature extraction

```python
@celery_app.task(bind=True)
def extract_features_task(self, dataset_id: int, domain: str, config: Dict):
    """Extract features in background."""
    # Update progress
    self.update_state(state='PROGRESS', meta={'progress': 0.1, 'status': 'Loading dataset...'})

    # Load dataset
    # Extract features
    # Compute statistics

    self.update_state(state='PROGRESS', meta={'progress': 0.5, 'status': 'Extracting features...'})

    # Return results
    return {
        "success": True,
        "features": features.tolist(),
        "feature_names": feature_names,
        "extraction_time": time_elapsed
    }


@celery_app.task(bind=True)
def compute_importance_task(self, features: List, targets: List, method: str):
    """Compute feature importance in background."""
    pass


@celery_app.task(bind=True)
def select_features_task(self, features: List, targets: List, method: str, num_features: int):
    """Select features in background."""
    pass
```

**Estimated Time**: 4 hours

---

### File 3: `packages/dashboard/layouts/feature_engineering.py` (~500 lines)

**Purpose**: Feature engineering UI

**Structure**:

```python
def create_feature_engineering_layout():
    """Create feature engineering dashboard."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="fas fa-magic me-3"),
                    "Feature Engineering"
                ]),
                html.P("Extract, select, and analyze features")
            ])
        ], className="mb-4"),

        # Tabs
        dbc.Tabs([
            # Feature Extraction Tab
            dbc.Tab(
                label="Feature Extraction",
                children=[
                    html.Div([
                        # Dataset Selection
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Select Dataset"),
                                dcc.Dropdown(
                                    id="fe-dataset-select",
                                    placeholder="Choose a dataset..."
                                )
                            ], width=6),
                        ], className="mb-3"),

                        # Domain Selection
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Feature Domain"),
                                dbc.RadioItems(
                                    id="fe-domain",
                                    options=[
                                        {"label": "Time Domain", "value": "time"},
                                        {"label": "Frequency Domain", "value": "frequency"},
                                        {"label": "Wavelet Domain", "value": "wavelet"},
                                        {"label": "Bispectrum", "value": "bispectrum"}
                                    ],
                                    value="time"
                                )
                            ], width=6),
                        ], className="mb-3"),

                        # Feature Configuration
                        html.Div(id="fe-config-panel"),

                        # Extract Button
                        dbc.Button(
                            [html.I(className="fas fa-play me-2"), "Extract Features"],
                            id="extract-features-btn",
                            color="primary",
                            className="mt-3"
                        ),

                        # Results
                        html.Div(id="fe-extraction-results", className="mt-3")

                    ], className="p-3")
                ]
            ),

            # Feature Importance Tab
            dbc.Tab(
                label="Feature Importance",
                children=[
                    html.Div([
                        # Method Selection
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Importance Method"),
                                dcc.Dropdown(
                                    id="fi-method",
                                    options=[
                                        {"label": "SHAP Values", "value": "shap"},
                                        {"label": "Permutation Importance", "value": "permutation"},
                                        {"label": "Mutual Information", "value": "mutual_info"}
                                    ],
                                    value="shap"
                                )
                            ], width=6),
                        ], className="mb-3"),

                        # Compute Button
                        dbc.Button(
                            [html.I(className="fas fa-chart-bar me-2"), "Compute Importance"],
                            id="compute-importance-btn",
                            color="primary"
                        ),

                        # Importance Chart
                        dcc.Graph(id="feature-importance-chart"),

                    ], className="p-3")
                ]
            ),

            # Feature Selection Tab
            dbc.Tab(
                label="Feature Selection",
                children=[
                    html.Div([
                        # Selection Method
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Selection Method"),
                                dcc.Dropdown(
                                    id="fs-method",
                                    options=[
                                        {"label": "Variance Threshold", "value": "variance"},
                                        {"label": "Mutual Information", "value": "mutual_info"},
                                        {"label": "Recursive Feature Elimination", "value": "rfe"},
                                        {"label": "L1-based Selection", "value": "l1"}
                                    ],
                                    value="variance"
                                )
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Number of Features"),
                                dbc.Input(
                                    id="fs-num-features",
                                    type="number",
                                    value=50,
                                    min=1
                                )
                            ], width=6),
                        ], className="mb-3"),

                        # Select Button
                        dbc.Button(
                            [html.I(className="fas fa-filter me-2"), "Select Features"],
                            id="select-features-btn",
                            color="primary"
                        ),

                        # Selected Features Table
                        html.Div(id="selected-features-table", className="mt-3")

                    ], className="p-3")
                ]
            ),

            # Feature Correlation Tab
            dbc.Tab(
                label="Correlation Matrix",
                children=[
                    html.Div([
                        # Correlation Heatmap
                        dcc.Graph(id="feature-correlation-heatmap", style={'height': '700px'}),

                    ], className="p-3")
                ]
            ),

            # Pipeline Builder Tab
            dbc.Tab(
                label="Pipeline Builder",
                children=[
                    html.Div([
                        html.P("Build a feature engineering pipeline by adding steps"),

                        # Pipeline Steps
                        html.Div(id="pipeline-steps", children=[]),

                        # Add Step Button
                        dbc.Button(
                            [html.I(className="fas fa-plus me-2"), "Add Step"],
                            id="add-pipeline-step-btn",
                            color="info"
                        ),

                        # Save Pipeline Button
                        dbc.Button(
                            [html.I(className="fas fa-save me-2"), "Save Pipeline"],
                            id="save-pipeline-btn",
                            color="success",
                            className="ms-2"
                        ),

                    ], className="p-3")
                ]
            ),

        ], id="feature-tabs", active_tab="tab-0")

    ], fluid=True)
```

**Estimated Time**: 1 day

---

### File 4: `packages/dashboard/callbacks/feature_callbacks.py` (~450 lines)

**Purpose**: Handle feature engineering interactions

**Key Callbacks**:

1. Load datasets dropdown
2. Extract features (launch Celery task)
3. Compute feature importance (launch Celery task)
4. Select features (launch Celery task)
5. Generate correlation heatmap
6. Build and save pipeline
7. Display results (charts, tables)

**Estimated Time**: 1 day

---

## Total Effort: Feature Engineering
- Service layer: 1 day
- Celery tasks: 4 hours
- Layout: 1 day
- Callbacks: 1 day
- Route registration: 5 minutes
- Testing: 4 hours

**Total: ~3 days**

---

# Feature 3: Advanced Training Options (2 days)

## Overview
Add advanced training options to the experiment wizard without creating new pages.

## Architecture

### Existing Code (Already Available ✅)

- `training/knowledge_distillation.py` - Teacher-student training
- `training/mixed_precision.py` - FP16/BF16 training
- `training/advanced_augmentation.py` - RandAugment, CutMix
- `training/progressive_resizing.py` - Progressive training

---

## Implementation Plan

### File 1: Enhance `packages/dashboard/layouts/experiment_wizard.py`

**Add new tab**: "Advanced Options"

**Insert after line ~300 (existing tabs)**:

```python
            # Advanced Options Tab (NEW)
            dbc.Tab(
                label="Advanced Options",
                children=[
                    html.Div([
                        html.H5("Advanced Training Techniques", className="mb-3"),

                        # Knowledge Distillation Section
                        dbc.Card([
                            dbc.CardHeader("Knowledge Distillation"),
                            dbc.CardBody([
                                dbc.Checklist(
                                    id="enable-distillation",
                                    options=[{"label": "Enable Knowledge Distillation", "value": "enabled"}],
                                    value=[]
                                ),
                                html.Div([
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Teacher Model"),
                                            dcc.Dropdown(
                                                id="teacher-model-select",
                                                placeholder="Select teacher model (pre-trained)..."
                                            )
                                        ], width=6),
                                        dbc.Col([
                                            dbc.Label("Temperature"),
                                            dbc.Input(
                                                id="distillation-temperature",
                                                type="number",
                                                value=3.0,
                                                min=1.0,
                                                max=10.0,
                                                step=0.5
                                            ),
                                            dbc.FormText("Higher = softer probabilities")
                                        ], width=3),
                                        dbc.Col([
                                            dbc.Label("Alpha (student weight)"),
                                            dbc.Input(
                                                id="distillation-alpha",
                                                type="number",
                                                value=0.5,
                                                min=0.0,
                                                max=1.0,
                                                step=0.1
                                            ),
                                            dbc.FormText("0.0 = all teacher, 1.0 = all student")
                                        ], width=3),
                                    ])
                                ], id="distillation-config", style={'display': 'none'})
                            ])
                        ], className="mb-3"),

                        # Mixed Precision Section
                        dbc.Card([
                            dbc.CardHeader("Mixed Precision Training"),
                            dbc.CardBody([
                                dbc.RadioItems(
                                    id="mixed-precision-mode",
                                    options=[
                                        {"label": "Disabled (FP32)", "value": "fp32"},
                                        {"label": "FP16 (Half Precision)", "value": "fp16"},
                                        {"label": "BF16 (Brain Float)", "value": "bf16"}
                                    ],
                                    value="fp32"
                                ),
                                dbc.FormText(
                                    "FP16/BF16 can speed up training and reduce memory usage"
                                )
                            ])
                        ], className="mb-3"),

                        # Advanced Augmentation Section
                        dbc.Card([
                            dbc.CardHeader("Advanced Augmentation"),
                            dbc.CardBody([
                                dbc.Checklist(
                                    id="enable-advanced-aug",
                                    options=[
                                        {"label": "Enable RandAugment", "value": "randaugment"},
                                        {"label": "Enable CutMix", "value": "cutmix"},
                                        {"label": "Enable MixUp", "value": "mixup"}
                                    ],
                                    value=[]
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
                                                max=20
                                            )
                                        ], width=6),
                                        dbc.Col([
                                            dbc.Label("Probability"),
                                            dbc.Input(
                                                id="aug-probability",
                                                type="number",
                                                value=0.5,
                                                min=0.0,
                                                max=1.0,
                                                step=0.1
                                            )
                                        ], width=6),
                                    ])
                                ], id="advanced-aug-config", style={'display': 'none'})
                            ])
                        ], className="mb-3"),

                        # Progressive Resizing Section
                        dbc.Card([
                            dbc.CardHeader("Progressive Resizing"),
                            dbc.CardBody([
                                dbc.Checklist(
                                    id="enable-progressive",
                                    options=[{"label": "Enable Progressive Resizing", "value": "enabled"}],
                                    value=[]
                                ),
                                html.Div([
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Start Size"),
                                            dbc.Input(
                                                id="progressive-start-size",
                                                type="number",
                                                value=51200  # Half of 102400
                                            )
                                        ], width=6),
                                        dbc.Col([
                                            dbc.Label("End Size"),
                                            dbc.Input(
                                                id="progressive-end-size",
                                                type="number",
                                                value=102400
                                            )
                                        ], width=6),
                                    ])
                                ], id="progressive-config", style={'display': 'none'})
                            ])
                        ], className="mb-3"),

                    ], className="p-3")
                ]
            ),
```

**Estimated Time**: 4 hours

---

### File 2: Enhance `packages/dashboard/callbacks/experiment_wizard_callbacks.py`

**Add callbacks for advanced options**:

```python
    @app.callback(
        Output('distillation-config', 'style'),
        Input('enable-distillation', 'value')
    )
    def toggle_distillation_config(enabled):
        """Show/hide distillation config."""
        if 'enabled' in enabled:
            return {'display': 'block', 'marginTop': '10px'}
        return {'display': 'none'}

    @app.callback(
        Output('advanced-aug-config', 'style'),
        Input('enable-advanced-aug', 'value')
    )
    def toggle_aug_config(enabled):
        """Show/hide augmentation config."""
        if len(enabled) > 0:
            return {'display': 'block', 'marginTop': '10px'}
        return {'display': 'none'}

    @app.callback(
        Output('progressive-config', 'style'),
        Input('enable-progressive', 'value')
    )
    def toggle_progressive_config(enabled):
        """Show/hide progressive config."""
        if 'enabled' in enabled:
            return {'display': 'block', 'marginTop': '10px'}
        return {'display': 'none'}

    @app.callback(
        Output('teacher-model-select', 'options'),
        Input('url', 'pathname')
    )
    def load_teacher_models(pathname):
        """Load available teacher models (completed experiments)."""
        if pathname != '/experiment/new':
            raise PreventUpdate

        try:
            with get_db_session() as session:
                experiments = session.query(Experiment).filter_by(
                    status=ExperimentStatus.COMPLETED
                ).order_by(Experiment.created_at.desc()).limit(20).all()

                return [
                    {
                        "label": f"{exp.name} ({exp.model_type}) - {exp.metrics.get('test_accuracy', 0):.2%}",
                        "value": exp.id
                    }
                    for exp in experiments
                ]
        except Exception as e:
            logger.error(f"Failed to load teacher models: {e}")
            return []
```

**Update existing `launch_experiment` callback** to include advanced options in config:

```python
    @app.callback(
        # ... existing outputs
    )
    def launch_experiment(
        # ... existing parameters,
        enable_distillation,
        teacher_model_id,
        distillation_temp,
        distillation_alpha,
        mixed_precision_mode,
        enable_advanced_aug,
        aug_magnitude,
        aug_probability,
        enable_progressive,
        progressive_start,
        progressive_end
    ):
        """Launch experiment with advanced options."""

        # ... existing code ...

        # Add advanced options to config
        advanced_config = {}

        # Knowledge Distillation
        if 'enabled' in enable_distillation:
            advanced_config['knowledge_distillation'] = {
                'enabled': True,
                'teacher_model_id': teacher_model_id,
                'temperature': distillation_temp,
                'alpha': distillation_alpha
            }

        # Mixed Precision
        if mixed_precision_mode != 'fp32':
            advanced_config['mixed_precision'] = {
                'enabled': True,
                'dtype': mixed_precision_mode
            }

        # Advanced Augmentation
        if len(enable_advanced_aug) > 0:
            advanced_config['advanced_augmentation'] = {
                'enabled': True,
                'methods': enable_advanced_aug,
                'magnitude': aug_magnitude,
                'probability': aug_probability
            }

        # Progressive Resizing
        if 'enabled' in enable_progressive:
            advanced_config['progressive_resizing'] = {
                'enabled': True,
                'start_size': progressive_start,
                'end_size': progressive_end
            }

        # Merge into experiment config
        config['advanced_options'] = advanced_config

        # ... rest of existing code ...
```

**Estimated Time**: 1 day

---

### File 3: Update Training Pipeline

**Modify**: `tasks/training_tasks.py` to use advanced options

**Add logic**:
```python
def train_model_task(self, experiment_id: int):
    # ... existing code ...

    # Check for advanced options
    advanced = config.get('advanced_options', {})

    # Knowledge Distillation
    if advanced.get('knowledge_distillation', {}).get('enabled'):
        from training.knowledge_distillation import DistillationTrainer
        teacher_exp_id = advanced['knowledge_distillation']['teacher_model_id']
        # Load teacher model
        # Use DistillationTrainer instead of regular trainer

    # Mixed Precision
    if advanced.get('mixed_precision', {}).get('enabled'):
        from training.mixed_precision import enable_mixed_precision
        dtype = advanced['mixed_precision']['dtype']
        scaler = enable_mixed_precision(dtype)
        # Use scaler in training loop

    # Advanced Augmentation
    if advanced.get('advanced_augmentation', {}).get('enabled'):
        from training.advanced_augmentation import get_advanced_augmentation
        aug_methods = advanced['advanced_augmentation']['methods']
        # Apply advanced augmentation to data loader

    # Progressive Resizing
    if advanced.get('progressive_resizing', {}).get('enabled'):
        from training.progressive_resizing import ProgressiveResizer
        # Use progressive resizing schedule

    # ... continue with training ...
```

**Estimated Time**: 4 hours

---

## Total Effort: Advanced Training Options
- UI changes (new tab): 4 hours
- Callback enhancements: 1 day
- Training pipeline updates: 4 hours
- Testing: 4 hours

**Total: ~2 days**

---

# Summary: Complete Roadmap

## Week 1: Critical Fixes + Dataset Management

**Days 1-2**:
1. Fix broken sidebar links (5 min)
2. Implement Dataset Management (1 day)
3. Test dataset CRUD operations (2 hours)

**Deliverable**: No broken links, fully functional dataset management

---

## Week 2-3: Feature Engineering

**Days 3-5**:
1. Feature Service + Tasks (1.5 days)
2. Feature UI + Callbacks (1.5 days)
3. Test all feature operations (4 hours)

**Deliverable**: Complete feature engineering workflow

---

## Week 3-4: Advanced Training

**Days 6-7**:
1. Advanced Options UI (0.5 days)
2. Advanced Options Callbacks (1 day)
3. Training Pipeline Integration (0.5 days)
4. Test advanced training (4 hours)

**Deliverable**: Experiment wizard with advanced options

---

## Total Timeline: ~7 days (1.5 weeks)

**Feature Coverage After Completion**: ~75% (up from 65%)

**Remaining Optional Features**:
- Notification Management (1 day)
- NAS Dashboard (3 days)
- Enhanced Visualization (2 days)

---

## Next Steps

1. ✅ Review this plan
2. ✅ Confirm priorities
3. ✅ Start with fixing broken sidebar links (5 min)
4. ✅ Implement Dataset Management (1 day)
5. ✅ Move to Feature Engineering (3 days)
6. ✅ Complete Advanced Training (2 days)

**Ready to begin implementation!**
