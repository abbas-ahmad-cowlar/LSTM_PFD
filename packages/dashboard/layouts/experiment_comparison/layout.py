"""
Experiment Comparison — main layout and shared utilities.
Provides the top-level comparison page layout, share modal, and formatting helpers.
"""
import dash_bootstrap_components as dbc
from dash import html, dcc


def create_experiment_comparison_layout(experiment_ids):
    """
    Create comparison page layout.

    Args:
        experiment_ids: List of experiment IDs from URL (e.g., [1234, 1567, 1890])
    """
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🔍 Experiment Comparison", className="mb-2"),
                html.P(
                    f"Comparing {len(experiment_ids)} experiments",
                    className="text-muted"
                )
            ], width=8),
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button(
                        [html.I(className="fas fa-file-pdf me-2"), "Export PDF"],
                        id='export-comparison-pdf',
                        color="secondary",
                        size="sm",
                        **{"aria-label": "Export comparison as PDF"}
                    ),
                    dbc.Button(
                        [html.I(className="fas fa-link me-2"), "Share Link"],
                        id='share-comparison-link',
                        color="info",
                        size="sm",
                        **{"aria-label": "Share comparison link"}
                    ),
                ], className="float-end")
            ], width=4)
        ], className="mb-4"),

        # Breadcrumb navigation
        dbc.Row([
            dbc.Col([
                dbc.Breadcrumb(items=[
                    {"label": "Experiments", "href": "/experiments", "active": False},
                    {"label": "Comparison", "active": True},
                ])
            ])
        ], className="mb-3"),

        # Loading indicator
        dcc.Loading(
            id="comparison-loading",
            type="default",
            children=[
                # Hidden store for comparison data
                dcc.Store(id='comparison-data-store', data={'experiment_ids': experiment_ids}),

                # Main content tabs
                dbc.Tabs(id='comparison-tabs', active_tab='overview', children=[
                    dbc.Tab(label="📊 Overview", tab_id="overview"),
                    dbc.Tab(label="📈 Metrics", tab_id="metrics"),
                    dbc.Tab(label="📉 Visualizations", tab_id="visualizations"),
                    dbc.Tab(label="🔬 Statistical Tests", tab_id="statistical"),
                    dbc.Tab(label="⚙️ Configuration", tab_id="configuration")
                ], className="mb-4", role="tablist"),

                # Tab content container
                html.Div(id='comparison-tab-content', className="mt-4",
                         **{"aria-live": "polite"})
            ]
        ),

        # Modals
        _create_share_link_modal(),

    ], fluid=True, className="py-4")


def _create_share_link_modal():
    """Create modal for sharing comparison link."""
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Share Comparison")),
        dbc.ModalBody([
            html.P("Copy this link to share the comparison:"),
            dbc.Input(
                id='comparison-share-link-input',
                type='text',
                readonly=True,
                className="mb-3",
                **{"aria-label": "Comparison share URL"}
            ),
            dbc.Button(
                [html.I(className="fas fa-copy me-2"), "Copy to Clipboard"],
                id='copy-comparison-link-btn',
                color="primary",
                className="w-100",
                **{"aria-label": "Copy share link to clipboard"}
            )
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-share-modal", className="ms-auto")
        ),
    ], id="share-link-modal", is_open=False,
       **{"aria-modal": "true"})


def format_duration(seconds):
    """Format duration in human-readable form."""
    if not seconds:
        return "N/A"

    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"
