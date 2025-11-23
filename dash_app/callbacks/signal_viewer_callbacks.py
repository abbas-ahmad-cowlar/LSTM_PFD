"""
Signal Viewer callbacks.
"""
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
import numpy as np

from services.signal_service import SignalService
from utils.plotting import create_time_series_plot
from utils.logger import setup_logger
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

logger = setup_logger(__name__)


def register_signal_viewer_callbacks(app):
    """Register signal viewer callbacks."""

    @app.callback(
        [Output('time-domain-plot', 'figure'),
         Output('frequency-domain-plot', 'figure'),
         Output('signal-metadata', 'children')],
        Input('signal-id-selector', 'value'),
        State('dataset-selector', 'value')
    )
    def update_signal_views(signal_id, dataset_id):
        """Update all signal visualizations."""
        if not signal_id or not dataset_id:
            raise PreventUpdate

        try:
            # Load signal with features
            signal_info = SignalService.get_signal_with_features(dataset_id, signal_id)
            signal_data = np.array(signal_info["signal_data"])
            features = signal_info["features"]

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
                html.P([html.Strong("Signal ID: "), signal_id]),
                html.P([html.Strong("Length: "), f"{len(signal_data)} samples"]),
                html.P([html.Strong("Duration: "), f"{len(signal_data)/SAMPLING_RATE:.2f} s"]),
                html.P([html.Strong("RMS: "), f"{features['rms']:.4f}"]),
                html.P([html.Strong("Kurtosis: "), f"{features['kurtosis']:.4f}"]),
            ])

            return time_fig, freq_fig, metadata

        except Exception as e:
            logger.error(f"Error loading signal: {e}")
            raise PreventUpdate
