"""
Advanced Visualization Callbacks (Phase 4, Feature 2/3).
Handles all callbacks for the advanced visualization dashboard.
"""
from dash import callback_context, html, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import h5py
from pathlib import Path
import traceback
from typing import List, Dict, Optional
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from utils.logger import setup_logger
from database.connection import get_db_session
from models.dataset import Dataset
from models.experiment import Experiment

logger = setup_logger(__name__)


def register_visualization_callbacks(app):
    """Register all visualization dashboard callbacks."""

    # ==================== Data Loading Callbacks ====================

    @app.callback(
        Output('viz-dataset-select', 'options'),
        Input('viz-tabs', 'active_tab')
    )
    def load_datasets(active_tab):
        """Load available datasets for visualization."""
        try:
            with get_db_session() as session:
                datasets = session.query(Dataset).order_by(Dataset.created_at.desc()).limit(50).all()

                return [
                    {'label': f"{ds.name} ({ds.signal_count} signals)", 'value': ds.id}
                    for ds in datasets
                ]
        except Exception as e:
            logger.error(f"Failed to load datasets: {e}")
            return []

    @app.callback(
        Output('viz-experiment-select', 'options'),
        Input('viz-dataset-select', 'value')
    )
    def load_experiments(dataset_id):
        """Load experiments for selected dataset."""
        if not dataset_id:
            return []

        try:
            with get_db_session() as session:
                experiments = session.query(Experiment).filter(
                    Experiment.dataset_id == dataset_id
                ).order_by(Experiment.created_at.desc()).limit(20).all()

                return [
                    {'label': f"{exp.name} ({exp.model_type}) - {exp.status}", 'value': exp.id}
                    for exp in experiments
                ]
        except Exception as e:
            logger.error(f"Failed to load experiments: {e}")
            return []

    # ==================== Tab 1: Embeddings ====================

    @app.callback(
        [Output('tsne-params', 'style'),
         Output('umap-params', 'style')],
        Input('embedding-method', 'value')
    )
    def toggle_embedding_params(method):
        """Show/hide parameter controls based on selected method."""
        tsne_style = {'display': 'block', 'marginBottom': '1rem'} if method == 'tsne' else {'display': 'none'}
        umap_style = {'display': 'block', 'marginBottom': '1rem'} if method == 'umap' else {'display': 'none'}
        return tsne_style, umap_style

    @app.callback(
        [Output('embedding-plot', 'figure'),
         Output('embedding-status', 'children')],
        Input('generate-embedding-btn', 'n_clicks'),
        [State('viz-dataset-select', 'value'),
         State('embedding-method', 'value'),
         State('tsne-perplexity', 'value'),
         State('umap-neighbors', 'value')]
    )
    def generate_embedding(n_clicks, dataset_id, method, perplexity, neighbors):
        """Generate dimensionality reduction embedding visualization."""
        if not n_clicks or not dataset_id:
            return go.Figure(), ""

        try:
            try:
                import umap
            except ImportError:
                umap = None

            with get_db_session() as session:
                dataset = session.query(Dataset).get(dataset_id)
                if not dataset:
                    return go.Figure(), html.Div("Dataset not found", className="text-danger")

                # Load dataset
                dataset_path = Path(dataset.file_path)
                if not dataset_path.exists():
                    return go.Figure(), html.Div("Dataset file not found", className="text-danger")

                with h5py.File(dataset_path, 'r') as f:
                    signals = f['signals'][:]
                    labels = f['labels'][:]
                    fault_names = [name.decode('utf-8') if isinstance(name, bytes) else name
                                 for name in f['fault_types'][:]]

                # Flatten signals for embedding (keep only limited samples for speed)
                max_samples = 1000
                if len(signals) > max_samples:
                    indices = np.random.choice(len(signals), max_samples, replace=False)
                    signals = signals[indices]
                    labels = labels[indices]

                # Apply dimensionality reduction
                if method == 'tsne':
                    reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                    embedding = reducer.fit_transform(signals)
                    title = f"t-SNE Embedding (perplexity={perplexity})"
                elif method == 'umap':
                    if umap is None:
                        return go.Figure(), html.Div("UMAP not installed. Install with: pip install umap-learn", className="text-warning")
                    reducer = umap.UMAP(n_neighbors=neighbors, random_state=42)
                    embedding = reducer.fit_transform(signals)
                    title = f"UMAP Embedding (n_neighbors={neighbors})"
                else:  # PCA
                    reducer = PCA(n_components=2, random_state=42)
                    embedding = reducer.fit_transform(signals)
                    title = "PCA Embedding"

                # Create scatter plot
                fig = go.Figure()
                for label_idx, fault_name in enumerate(fault_names):
                    mask = labels == label_idx
                    fig.add_trace(go.Scatter(
                        x=embedding[mask, 0],
                        y=embedding[mask, 1],
                        mode='markers',
                        name=fault_name,
                        marker=dict(size=6, opacity=0.7)
                    ))

                fig.update_layout(
                    title=title,
                    xaxis_title="Component 1",
                    yaxis_title="Component 2",
                    hovermode='closest',
                    template='plotly_white',
                    height=600
                )

                status = html.Div([
                    html.I(className="bi bi-check-circle text-success me-2"),
                    f"Successfully generated {method.upper()} embedding for {len(signals)} samples"
                ])

                return fig, status

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            logger.error(traceback.format_exc())
            return go.Figure(), html.Div(f"Error: {str(e)}", className="text-danger")

    # ==================== Tab 2: Signal Analysis ====================

    @app.callback(
        [Output('signal-select', 'options'),
         Output('signal-fault-filter', 'options')],
        Input('viz-dataset-select', 'value')
    )
    def load_signal_options(dataset_id):
        """Load signal indices and fault types for selected dataset."""
        if not dataset_id:
            return [], []

        try:
            with get_db_session() as session:
                dataset = session.query(Dataset).get(dataset_id)
                if not dataset:
                    return [], []

                # Signal options (show first 100)
                signal_options = [
                    {'label': f"Signal {i}", 'value': i}
                    for i in range(min(100, dataset.signal_count))
                ]

                # Fault type options
                dataset_path = Path(dataset.file_path)
                if dataset_path.exists():
                    with h5py.File(dataset_path, 'r') as f:
                        fault_names = [name.decode('utf-8') if isinstance(name, bytes) else name
                                     for name in f['fault_types'][:]]
                        fault_options = [
                            {'label': fault_name, 'value': idx}
                            for idx, fault_name in enumerate(fault_names)
                        ]
                else:
                    fault_options = []

                return signal_options, fault_options

        except Exception as e:
            logger.error(f"Failed to load signal options: {e}")
            return [], []

    @app.callback(
        [Output('signal-viz-plot', 'figure'),
         Output('signal-viz-status', 'children')],
        Input('generate-signal-viz-btn', 'n_clicks'),
        [State('viz-dataset-select', 'value'),
         State('signal-viz-type', 'value'),
         State('signal-select', 'value')]
    )
    def generate_signal_visualization(n_clicks, dataset_id, viz_type, signal_idx):
        """Generate signal visualization (bispectrum/wavelet/spectrogram)."""
        if not n_clicks or not dataset_id or signal_idx is None:
            return go.Figure(), ""

        try:
            with get_db_session() as session:
                dataset = session.query(Dataset).get(dataset_id)
                if not dataset:
                    return go.Figure(), html.Div("Dataset not found", className="text-danger")

                # Load signal
                dataset_path = Path(dataset.file_path)
                with h5py.File(dataset_path, 'r') as f:
                    signal = f['signals'][signal_idx]
                    label = f['labels'][signal_idx]
                    fault_names = [name.decode('utf-8') if isinstance(name, bytes) else name
                                 for name in f['fault_types'][:]]
                    fault_name = fault_names[label]

                # Generate visualization based on type
                if viz_type == 'spectrogram':
                    from scipy import signal as sp_signal
                    sampling_rate = dataset.sampling_rate or 12800
                    f, t, Sxx = sp_signal.spectrogram(signal, fs=sampling_rate)

                    fig = go.Figure(data=go.Heatmap(
                        z=10 * np.log10(Sxx + 1e-10),
                        x=t,
                        y=f,
                        colorscale='Viridis',
                        colorbar=dict(title="Power (dB)")
                    ))
                    fig.update_layout(
                        title=f"Spectrogram - Signal {signal_idx} ({fault_name})",
                        xaxis_title="Time (s)",
                        yaxis_title="Frequency (Hz)",
                        template='plotly_white',
                        height=600
                    )

                elif viz_type == 'wavelet':
                    try:
                        import pywt
                    except ImportError:
                        return go.Figure(), html.Div(
                            "PyWavelets (pywt) library is required for wavelet visualization. Install it with: pip install PyWavelets",
                            className="text-danger"
                        )
                    # Continuous Wavelet Transform
                    scales = np.arange(1, 128)
                    coefficients, frequencies = pywt.cwt(signal, scales, 'morl')

                    fig = go.Figure(data=go.Heatmap(
                        z=np.abs(coefficients),
                        x=np.arange(len(signal)),
                        y=frequencies,
                        colorscale='Jet',
                        colorbar=dict(title="Magnitude")
                    ))
                    fig.update_layout(
                        title=f"Wavelet Scalogram - Signal {signal_idx} ({fault_name})",
                        xaxis_title="Time (samples)",
                        yaxis_title="Frequency Scale",
                        template='plotly_white',
                        height=600
                    )

                else:  # bispectrum
                    # Simple bispectrum approximation using FFT
                    fft_signal = np.fft.fft(signal)
                    power_spectrum = np.abs(fft_signal[:len(fft_signal)//2])**2

                    fig = go.Figure(data=go.Scatter(
                        y=power_spectrum[:500],
                        mode='lines',
                        name='Power Spectrum'
                    ))
                    fig.update_layout(
                        title=f"Frequency Analysis - Signal {signal_idx} ({fault_name})",
                        xaxis_title="Frequency Bin",
                        yaxis_title="Power",
                        template='plotly_white',
                        height=600
                    )

                status = html.Div([
                    html.I(className="bi bi-check-circle text-success me-2"),
                    f"Successfully generated {viz_type} for signal {signal_idx}"
                ])

                return fig, status

        except Exception as e:
            logger.error(f"Failed to generate signal visualization: {e}")
            logger.error(traceback.format_exc())
            return go.Figure(), html.Div(f"Error: {str(e)}", className="text-danger")

    # ==================== Tab 3: Feature Analysis ====================

    @app.callback(
        [Output('feature-viz-plot', 'figure'),
         Output('feature-viz-status', 'children')],
        Input('generate-feature-viz-btn', 'n_clicks'),
        [State('viz-dataset-select', 'value'),
         State('feature-viz-type', 'value'),
         State('top-n-features', 'value')]
    )
    def generate_feature_visualization(n_clicks, dataset_id, viz_type, top_n):
        """Generate feature visualization (importance/correlation/distributions)."""
        if not n_clicks or not dataset_id:
            return go.Figure(), ""

        try:
            # Placeholder: In production, load actual features from feature extraction results
            # For now, generate synthetic feature importance as demonstration

            if viz_type == 'importance':
                feature_names = [f"Feature_{i}" for i in range(top_n)]
                importances = np.random.random(top_n)
                importances = np.sort(importances)[::-1]

                fig = go.Figure(data=go.Bar(
                    x=importances,
                    y=feature_names,
                    orientation='h',
                    marker=dict(color=importances, colorscale='Viridis')
                ))
                fig.update_layout(
                    title=f"Top {top_n} Feature Importance",
                    xaxis_title="Importance Score",
                    yaxis_title="Feature",
                    template='plotly_white',
                    height=600
                )

            elif viz_type == 'correlation':
                # Generate random correlation matrix
                n_features = top_n
                corr_matrix = np.random.rand(n_features, n_features)
                corr_matrix = (corr_matrix + corr_matrix.T) / 2
                np.fill_diagonal(corr_matrix, 1.0)

                feature_names = [f"F{i}" for i in range(n_features)]

                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix,
                    x=feature_names,
                    y=feature_names,
                    colorscale='RdBu_r',
                    zmid=0,
                    colorbar=dict(title="Correlation")
                ))
                fig.update_layout(
                    title=f"Feature Correlation Matrix ({n_features} features)",
                    template='plotly_white',
                    height=600
                )

            else:  # distributions
                # Generate random distributions
                fig = go.Figure()
                for i in range(min(5, top_n)):
                    data = np.random.normal(loc=i, scale=1, size=100)
                    fig.add_trace(go.Box(
                        y=data,
                        name=f"Feature_{i}",
                        boxmean='sd'
                    ))

                fig.update_layout(
                    title="Feature Distributions (Sample)",
                    yaxis_title="Value",
                    template='plotly_white',
                    height=600
                )

            status = html.Div([
                html.I(className="bi bi-info-circle text-info me-2"),
                "Note: Feature analysis requires feature extraction to be run first. ",
                "This is a demonstration with synthetic data."
            ])

            return fig, status

        except Exception as e:
            logger.error(f"Failed to generate feature visualization: {e}")
            logger.error(traceback.format_exc())
            return go.Figure(), html.Div(f"Error: {str(e)}", className="text-danger")

    # ==================== Tab 4: Model Analysis ====================

    @app.callback(
        [Output('layer-selection-div', 'style'),
         Output('layer-select', 'options')],
        [Input('model-viz-type', 'value'),
         Input('viz-experiment-select', 'value')]
    )
    def update_layer_options(viz_type, experiment_id):
        """Show/hide layer selection and populate layer options."""
        if viz_type == 'activation' and experiment_id:
            # In production, load actual model layers
            layer_options = [
                {'label': 'Conv1', 'value': 'conv1'},
                {'label': 'Conv2', 'value': 'conv2'},
                {'label': 'Conv3', 'value': 'conv3'},
                {'label': 'FC1', 'value': 'fc1'},
            ]
            return {'display': 'block', 'marginBottom': '1rem'}, layer_options
        else:
            return {'display': 'none'}, []

    @app.callback(
        Output('sample-select', 'options'),
        Input('viz-dataset-select', 'value')
    )
    def load_sample_options(dataset_id):
        """Load sample options for model visualization."""
        if not dataset_id:
            return []

        # Show first 50 samples
        return [
            {'label': f"Sample {i}", 'value': i}
            for i in range(50)
        ]

    @app.callback(
        [Output('model-viz-plot', 'figure'),
         Output('model-viz-status', 'children')],
        Input('generate-model-viz-btn', 'n_clicks'),
        [State('viz-experiment-select', 'value'),
         State('model-viz-type', 'value'),
         State('sample-select', 'value'),
         State('layer-select', 'value')]
    )
    def generate_model_visualization(n_clicks, experiment_id, viz_type, sample_idx, layer):
        """Generate model interpretation visualization."""
        if not n_clicks or sample_idx is None:
            return go.Figure(), ""

        try:
            # Placeholder: In production, load actual model and generate saliency/activation maps
            # For now, generate synthetic visualization as demonstration

            if viz_type == 'saliency':
                # Generate synthetic saliency map
                saliency = np.random.random(1000) * np.exp(-np.linspace(0, 5, 1000))

                fig = go.Figure(data=go.Scatter(
                    y=saliency,
                    mode='lines',
                    fill='tozeroy',
                    name='Saliency'
                ))
                fig.update_layout(
                    title=f"Saliency Map - Sample {sample_idx}",
                    xaxis_title="Time Step",
                    yaxis_title="Gradient Magnitude",
                    template='plotly_white',
                    height=600
                )

            elif viz_type == 'activation':
                # Generate synthetic activation heatmap
                activations = np.random.random((32, 100))

                fig = go.Figure(data=go.Heatmap(
                    z=activations,
                    colorscale='Viridis',
                    colorbar=dict(title="Activation")
                ))
                fig.update_layout(
                    title=f"Activation Map - {layer or 'Layer'} - Sample {sample_idx}",
                    xaxis_title="Time",
                    yaxis_title="Channel",
                    template='plotly_white',
                    height=600
                )

            else:  # counterfactual
                # Generate synthetic counterfactual
                original = np.sin(np.linspace(0, 10, 1000))
                counterfactual = original + 0.3 * np.random.random(1000)

                fig = go.Figure()
                fig.add_trace(go.Scatter(y=original, mode='lines', name='Original'))
                fig.add_trace(go.Scatter(y=counterfactual, mode='lines', name='Counterfactual'))

                fig.update_layout(
                    title=f"Counterfactual Explanation - Sample {sample_idx}",
                    xaxis_title="Time Step",
                    yaxis_title="Amplitude",
                    template='plotly_white',
                    height=600
                )

            status = html.Div([
                html.I(className="bi bi-info-circle text-info me-2"),
                "Note: Model analysis requires a trained model. ",
                "This is a demonstration with synthetic data."
            ])

            return fig, status

        except Exception as e:
            logger.error(f"Failed to generate model visualization: {e}")
            logger.error(traceback.format_exc())
            return go.Figure(), html.Div(f"Error: {str(e)}", className="text-danger")

    logger.info("Visualization callbacks registered successfully")
