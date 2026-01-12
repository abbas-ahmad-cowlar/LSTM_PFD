"""
Signal Viewer callbacks.
"""
from dash import Input, Output, State, html, ctx
from dash.exceptions import PreventUpdate
import numpy as np
import random

from services.signal_service import SignalService
from services.dataset_service import DatasetService
from utils.plotting import create_time_series_plot
from utils.logger import setup_logger
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

logger = setup_logger(__name__)


def register_signal_viewer_callbacks(app):
    """Register signal viewer callbacks."""

    # =========================================================================
    # LOAD DATASETS FOR SELECTOR
    # =========================================================================
    @app.callback(
        Output('signal-viewer-dataset-selector', 'options'),
        Input('url', 'pathname')
    )
    def load_datasets_for_signal_viewer(pathname):
        """Load available datasets for the dataset selector."""
        try:
            datasets = DatasetService.list_datasets(limit=50)
            return [{'label': f"{ds['name']} ({ds['num_signals']} signals)", 'value': ds['id']} for ds in datasets]
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            return []

    # =========================================================================
    # LOAD SIGNAL IDS FOR SELECTED DATASET
    # =========================================================================
    @app.callback(
        Output('signal-id-selector', 'options'),
        Input('signal-viewer-dataset-selector', 'value')
    )
    def load_signal_ids(dataset_id):
        """Load available signal IDs for the selected dataset."""
        if not dataset_id:
            return []
        
        try:
            # Get dataset details to know how many signals we have
            details = DatasetService.get_dataset_details(dataset_id)
            if not details:
                return []
            
            num_signals = details.get('num_signals', 0)
            # Return options for signal indices (0 to num_signals-1)
            return [{'label': f"Signal {i}", 'value': i} for i in range(min(num_signals, 100))]  # Limit to first 100
        except Exception as e:
            logger.error(f"Error loading signal IDs: {e}")
            return []

    # =========================================================================
    # RANDOM SIGNAL BUTTON
    # =========================================================================
    @app.callback(
        Output('signal-id-selector', 'value', allow_duplicate=True),
        Input('random-signal-btn', 'n_clicks'),
        State('signal-viewer-dataset-selector', 'value'),
        prevent_initial_call=True
    )
    def select_random_signal(n_clicks, dataset_id):
        """Select a random signal from the dataset."""
        if not n_clicks or not dataset_id:
            raise PreventUpdate
        
        try:
            details = DatasetService.get_dataset_details(dataset_id)
            if not details:
                raise PreventUpdate
            
            num_signals = details.get('num_signals', 0)
            if num_signals == 0:
                raise PreventUpdate
            
            random_id = random.randint(0, min(num_signals - 1, 99))
            return random_id
        except Exception as e:
            logger.error(f"Error selecting random signal: {e}")
            raise PreventUpdate

    # =========================================================================
    # UPDATE SIGNAL VISUALIZATIONS
    # =========================================================================
    @app.callback(
        [Output('time-domain-plot', 'figure'),
         Output('frequency-domain-plot', 'figure'),
         Output('signal-metadata', 'children')],
        Input('signal-id-selector', 'value'),
        State('signal-viewer-dataset-selector', 'value')
    )
    def update_signal_views(signal_id, dataset_id):
        """Update all signal visualizations."""
        if signal_id is None or not dataset_id:
            # Return empty figures
            import plotly.graph_objects as go
            empty_fig = go.Figure()
            empty_fig.update_layout(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                annotations=[dict(text="Select a signal to view", showarrow=False, font=dict(size=16))]
            )
            return empty_fig, empty_fig, html.P("No signal selected", className="text-muted")

        try:
            # Load signal with features
            signal_info = SignalService.get_signal_with_features(dataset_id, signal_id)
            signal_data = np.array(signal_info["signal_data"])
            features = signal_info.get("features", {})

            # Time domain plot
            time = np.arange(len(signal_data)) / SAMPLING_RATE
            time_fig = create_time_series_plot(time, signal_data)

            # Frequency domain plot
            freq, fft = SignalService.compute_fft(signal_data)
            freq_fig = create_time_series_plot(
                freq, fft,
                title="Frequency Domain",
                xlabel="Frequency (Hz)",
                ylabel="Magnitude"
            )

            # Metadata
            metadata = html.Div([
                html.P([html.Strong("Signal ID: "), f"{signal_id}"]),
                html.P([html.Strong("Length: "), f"{len(signal_data)} samples"]),
                html.P([html.Strong("Duration: "), f"{len(signal_data)/SAMPLING_RATE:.2f} s"]),
                html.P([html.Strong("RMS: "), f"{features.get('rms', 'N/A')}"]),
                html.P([html.Strong("Kurtosis: "), f"{features.get('kurtosis', 'N/A')}"]),
            ])

            return time_fig, freq_fig, metadata

        except Exception as e:
            logger.error(f"Error loading signal: {e}", exc_info=True)
            import plotly.graph_objects as go
            error_fig = go.Figure()
            error_fig.update_layout(
                annotations=[dict(text=f"Error: {str(e)[:50]}...", showarrow=False)]
            )
            return error_fig, error_fig, html.P(f"Error loading signal: {str(e)}", className="text-danger")
