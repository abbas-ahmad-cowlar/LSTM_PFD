"""
Dataset Management Layout (Phase 11A).
List, view, and manage datasets.

Note: This implements the 'dataset_manager' functionality described in Phase_11A.md.
The file is named 'datasets.py' instead of 'dataset_manager.py' for consistency
with other layout files (data_explorer.py, signal_viewer.py, etc.).
"""
import dash_bootstrap_components as dbc
from dash import html, dcc, dash_table


def create_datasets_layout():
    """
    Create datasets management page.

    This page provides CRUD operations for datasets:
    - Create new datasets via Phase 0 data generation
    - Upload existing datasets
    - View dataset details and statistics
    - Delete datasets
    """
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2([
                    html.I(className="fas fa-folder me-3"),
                    "Dataset Management"
                ], className="mb-1"),
                html.P("Manage and explore your datasets", className="text-muted mb-4")
            ], width=8),
            dbc.Col([
                dbc.Button(
                    [html.I(className="fas fa-plus me-2"), "New Dataset"],
                    id="new-dataset-btn",
                    color="primary",
                    href="/data-generation"
                )
            ], width=4, className="text-end")
        ], className="mb-4"),

        # Search and Filters
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Input(
                                    id="dataset-search",
                                    placeholder="Search datasets by name...",
                                    type="text",
                                    debounce=True
                                )
                            ], width=8),
                            dbc.Col([
                                dbc.Button(
                                    [html.I(className="fas fa-sync me-2"), "Refresh"],
                                    id="datasets-refresh-btn",
                                    color="secondary",
                                    outline=True,
                                    className="w-100"
                                )
                            ], width=4),
                        ])
                    ])
                ], className="shadow-sm")
            ])
        ], className="mb-4"),

        # Dataset Table
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.H5("Datasets", className="mb-0")
                    ]),
                    dbc.CardBody([
                        html.Div(id="datasets-table-container")
                    ])
                ], className="shadow-sm")
            ])
        ], className="mb-4"),

        # Dataset Details Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle(id="dataset-details-title")),
            dbc.ModalBody([
                # Basic Info
                html.Div(id="dataset-basic-info"),

                # Statistics Cards
                html.H5("Statistics", className="mt-4 mb-3"),
                html.Div(id="dataset-stats-cards"),

                # Class Distribution Chart
                html.H5("Class Distribution", className="mt-4 mb-3"),
                dcc.Graph(id="dataset-class-distribution"),

                # Signal Preview
                html.H5("Signal Preview", className="mt-4 mb-3"),
                dcc.Graph(id="dataset-signal-preview"),

                # Actions
                html.Hr(className="mt-4"),
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
                        {"label": "HDF5 (.h5) - Original format", "value": "hdf5"},
                        {"label": "MATLAB (.mat) - For MATLAB users", "value": "mat"},
                        {"label": "CSV (.csv) - For spreadsheet software", "value": "csv"}
                    ],
                    value="hdf5"
                ),
                html.Div(id="export-status", className="mt-3")
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="cancel-export", color="secondary"),
                dbc.Button("Export", id="confirm-export", color="primary")
            ])
        ], id="export-modal", is_open=False),

        # Delete Confirmation Modal
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle("Confirm Deletion")),
            dbc.ModalBody([
                html.P("Are you sure you want to delete this dataset?"),
                html.P([
                    html.Strong("Warning: "),
                    "This action cannot be undone."
                ], className="text-danger"),
                dbc.Checklist(
                    id="delete-file-checkbox",
                    options=[
                        {"label": "Also delete the HDF5 file from disk", "value": "delete_file"}
                    ],
                    value=[]
                )
            ]),
            dbc.ModalFooter([
                dbc.Button("Cancel", id="cancel-delete", color="secondary"),
                dbc.Button("Delete", id="confirm-delete", color="danger")
            ])
        ], id="delete-modal", is_open=False),

        # Storage
        dcc.Store(id='selected-dataset-id'),
        dcc.Store(id='datasets-data'),

        # Auto-refresh interval (30 seconds)
        dcc.Interval(id='datasets-auto-refresh', interval=30*1000, n_intervals=0)

    ], fluid=True)
