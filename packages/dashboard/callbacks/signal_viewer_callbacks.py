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

    # =========================================================================
    # PREDICT FAULT BUTTON
    # =========================================================================
    @app.callback(
        Output('signal-features', 'children', allow_duplicate=True),
        Input('predict-fault-btn', 'n_clicks'),
        [State('signal-id-selector', 'value'),
         State('signal-viewer-dataset-selector', 'value')],
        prevent_initial_call=True
    )
    def predict_fault(n_clicks, signal_id, dataset_id):
        """Predict fault class for the selected signal."""
        if not n_clicks or signal_id is None or not dataset_id:
            raise PreventUpdate
        
        try:
            # Load the signal data
            signal_info = SignalService.get_signal_with_features(dataset_id, signal_id)
            signal_data = np.array(signal_info["signal_data"])
            features = signal_info.get("features", {})
            
            # Check for available trained models
            from database.connection import get_db_session
            from models.experiment import Experiment, ExperimentStatus
            
            with get_db_session() as session:
                completed_experiments = session.query(Experiment).filter(
                    Experiment.status == ExperimentStatus.COMPLETED
                ).limit(5).all()
                
                if completed_experiments:
                    # We have trained models - try to use one
                    exp = completed_experiments[0]
                    model_path = exp.model_path
                    
                    if model_path:
                        # Try to load and run prediction
                        try:
                            from services.xai_service import XAIService
                            import torch
                            
                            # Load model
                            model = XAIService._load_model(model_path)
                            if model:
                                # Prepare signal for inference
                                signal_tensor = torch.tensor(signal_data, dtype=torch.float32)
                                if signal_tensor.dim() == 1:
                                    signal_tensor = signal_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len)
                                
                                # Get prediction
                                prediction = XAIService.get_model_prediction(model, signal_tensor)
                                
                                pred_class = prediction.get('predicted_class', 'Unknown')
                                confidence = prediction.get('confidence', 0) * 100
                                
                                return html.Div([
                                    html.H5("🔮 Fault Prediction", className="text-primary mb-3"),
                                    html.Div([
                                        html.P([
                                            html.Strong("Predicted Fault: "),
                                            html.Span(f"{pred_class}", className="badge bg-warning text-dark fs-6")
                                        ]),
                                        html.P([
                                            html.Strong("Confidence: "),
                                            html.Span(f"{confidence:.1f}%", className="text-success")
                                        ]),
                                        html.P([
                                            html.Strong("Model: "),
                                            html.Span(exp.name, className="text-muted")
                                        ]),
                                    ], className="alert alert-success"),
                                    html.Hr(),
                                    html.H6("Extracted Features"),
                                    html.P([html.Strong("RMS: "), f"{features.get('rms', 'N/A')}"]),
                                    html.P([html.Strong("Kurtosis: "), f"{features.get('kurtosis', 'N/A')}"]),
                                ])
                        except Exception as model_error:
                            logger.warning(f"Could not use trained model: {model_error}")
            
            # Fallback: Feature-based heuristic classification
            rms = features.get('rms', 0.5)
            kurtosis = features.get('kurtosis', 3.0)
            
            # Simple heuristic based on signal characteristics
            if rms > 0.8 or abs(kurtosis) > 10:
                predicted_class = "Outer Race Fault"
                confidence = 75.0
            elif rms > 0.6 or abs(kurtosis) > 6:
                predicted_class = "Inner Race Fault"
                confidence = 70.0
            elif rms > 0.4 or abs(kurtosis) > 4:
                predicted_class = "Ball Fault"
                confidence = 65.0
            else:
                predicted_class = "Normal"
                confidence = 80.0
            
            return html.Div([
                html.H5("🔮 Fault Prediction", className="text-primary mb-3"),
                html.Div([
                    html.P([
                        html.Strong("Predicted Fault: "),
                        html.Span(f"{predicted_class}", className="badge bg-warning text-dark fs-6")
                    ]),
                    html.P([
                        html.Strong("Confidence: "),
                        html.Span(f"{confidence:.1f}%", className="text-info")
                    ]),
                    html.Small("ℹ️ Feature-based heuristic (no trained model available)", className="text-muted"),
                ], className="alert alert-info"),
                html.Hr(),
                html.H6("Extracted Features"),
                html.P([html.Strong("RMS: "), f"{rms:.4f}" if isinstance(rms, float) else f"{rms}"]),
                html.P([html.Strong("Kurtosis: "), f"{kurtosis:.4f}" if isinstance(kurtosis, float) else f"{kurtosis}"]),
            ])
            
        except Exception as e:
            logger.error(f"Error predicting fault: {e}", exc_info=True)
            return html.Div([
                html.H5("⚠️ Prediction Error", className="text-danger"),
                html.P(f"Could not predict: {str(e)}", className="text-danger"),
            ])

    # =========================================================================
    # EXPORT SIGNAL BUTTON
    # =========================================================================
    @app.callback(
        Output('signal-download', 'data'),
        Input('export-signal-btn', 'n_clicks'),
        [State('signal-id-selector', 'value'),
         State('signal-viewer-dataset-selector', 'value')],
        prevent_initial_call=True
    )
    def export_signal(n_clicks, signal_id, dataset_id):
        """Export the selected signal as a CSV file."""
        if not n_clicks or signal_id is None or not dataset_id:
            return None
        
        try:
            # Load the signal data
            signal_info = SignalService.get_signal_with_features(dataset_id, signal_id)
            signal_data = np.array(signal_info["signal_data"])
            features = signal_info.get("features", {})
            
            # Create CSV content using pandas for efficiency
            import pandas as pd
            
            # Create time array
            time_values = np.arange(len(signal_data)) / SAMPLING_RATE
            
            # Create dataframe
            df = pd.DataFrame({
                'time_seconds': time_values,
                'amplitude': signal_data
            })
            
            # Add metadata as comment in filename
            filename = f"signal_ds{dataset_id}_sig{signal_id}_len{len(signal_data)}.csv"
            
            logger.info(f"Exporting signal {signal_id} from dataset {dataset_id} as {filename}")
            
            # Use dcc.send_data_frame for efficient export
            from dash import dcc
            return dcc.send_data_frame(df.to_csv, filename, index=False)
            
        except Exception as e:
            logger.error(f"Error exporting signal: {e}", exc_info=True)
            return None

    logger.info("Signal viewer callbacks registered")

