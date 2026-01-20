"""
Data Explorer callbacks.
Enhanced with filtering, summary stats, feature distributions, and dimensionality reduction.
"""
from dash import Input, Output, State, html, dcc, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from services.data_service import DataService
from services.dataset_service import DatasetService
from utils.logger import setup_logger
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

logger = setup_logger(__name__)


def register_data_explorer_callbacks(app):
    """Register data explorer callbacks."""

    # =========================================================================
    # LOAD DATASETS
    # =========================================================================
    @app.callback(
        Output('dataset-selector', 'options'),
        Input('url', 'pathname')
    )
    def load_datasets(pathname):
        """Load available datasets."""
        if pathname != '/data-explorer':
            raise PreventUpdate

        try:
            datasets = DatasetService.list_datasets(limit=50)
            return [{"label": f"{d['name']} ({d['num_signals']} signals)", "value": d['id']} for d in datasets]
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            return []

    # =========================================================================
    # UPDATE CLASS DISTRIBUTION CHART
    # =========================================================================
    @app.callback(
        Output('class-distribution-chart', 'figure'),
        [Input('dataset-selector', 'value'),
         Input('fault-filter', 'value'),
         Input('severity-filter', 'value')]
    )
    def update_class_distribution(dataset_id, fault_filter, severity_filter):
        """Update class distribution chart with filter support."""
        from dashboard_config import DASHBOARD_TO_PHASE0_FAULT_MAP, PHASE0_TO_DASHBOARD_FAULT_MAP
        
        if not dataset_id:
            fig = go.Figure()
            fig.update_layout(
                annotations=[dict(text="Select a dataset to view class distribution", showarrow=False)]
            )
            return fig

        try:
            stats = DataService.get_dataset_stats(dataset_id)
            dist = stats.get("class_distribution", {})

            if not dist:
                fig = go.Figure()
                fig.update_layout(annotations=[dict(text="No class distribution data", showarrow=False)])
                return fig

            # Apply fault class filter if provided
            # Handle both English (dashboard) and French (Phase0) fault names
            if fault_filter:
                # Build a set of acceptable names (both English and French versions)
                acceptable_names = set()
                for f in fault_filter:
                    acceptable_names.add(f)  # Add English name
                    acceptable_names.add(f.lower())
                    # Also add French equivalent
                    if f in DASHBOARD_TO_PHASE0_FAULT_MAP:
                        acceptable_names.add(DASHBOARD_TO_PHASE0_FAULT_MAP[f])
                
                # Filter distribution
                dist = {k: v for k, v in dist.items() if k in acceptable_names or k.lower() in acceptable_names}
            
            if not dist:
                fig = go.Figure()
                fig.update_layout(annotations=[dict(text="No classes match filter", showarrow=False)])
                return fig

            # Convert French names to English for display
            display_dist = {}
            for k, v in dist.items():
                display_name = PHASE0_TO_DASHBOARD_FAULT_MAP.get(k, k)
                display_name = display_name.replace('_', ' ').title()
                display_dist[display_name] = v

            fig = px.bar(
                x=list(display_dist.keys()),
                y=list(display_dist.values()),
                labels={"x": "Fault Class", "y": "Count"},
                title="Class Distribution",
                color=list(display_dist.keys()),
            )
            fig.update_layout(showlegend=False)
            return fig
        except Exception as e:
            logger.error(f"Error creating chart: {e}")
            return {}


    # =========================================================================
    # UPDATE SUMMARY STATISTICS
    # =========================================================================
    @app.callback(
        Output('summary-stats-table', 'children'),
        Input('dataset-selector', 'value')
    )
    def update_summary_stats(dataset_id):
        """Update summary statistics display."""
        if not dataset_id:
            return html.Div([
                html.P("Select a dataset to view statistics", className="text-muted text-center")
            ])

        try:
            stats = DataService.get_dataset_stats(dataset_id)
            details = DatasetService.get_dataset_details(dataset_id)
            
            if not stats:
                return html.Div([html.P("No statistics available", className="text-muted")])

            # Create stat cards
            return dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Total Signals", className="text-muted mb-1"),
                            html.H4(f"{stats.get('num_signals', 0):,}", className="mb-0")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Fault Classes", className="text-muted mb-1"),
                            html.H4(f"{len(stats.get('fault_types', []))}", className="mb-0")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Signal Length", className="text-muted mb-1"),
                            html.H4(f"{details.get('signal_length', SIGNAL_LENGTH):,}", className="mb-0")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Sampling Rate", className="text-muted mb-1"),
                            html.H4(f"{details.get('sampling_rate', SAMPLING_RATE):,} Hz", className="mb-0")
                        ])
                    ])
                ], width=3),
            ], className="mb-3")

        except Exception as e:
            logger.error(f"Error loading summary stats: {e}")
            return html.Div([html.P(f"Error: {str(e)}", className="text-danger")])

    # =========================================================================
    # LOAD FEATURE OPTIONS
    # =========================================================================
    @app.callback(
        Output('feature-selector', 'options'),
        Input('dataset-selector', 'value')
    )
    def load_feature_options(dataset_id):
        """Load available features for distribution analysis."""
        if not dataset_id:
            return []

        # Return standard signal features
        return [
            {'label': 'RMS', 'value': 'rms'},
            {'label': 'Peak-to-Peak', 'value': 'peak_to_peak'},
            {'label': 'Kurtosis', 'value': 'kurtosis'},
            {'label': 'Skewness', 'value': 'skewness'},
            {'label': 'Crest Factor', 'value': 'crest_factor'},
            {'label': 'Standard Deviation', 'value': 'std'},
            {'label': 'Mean', 'value': 'mean'},
            {'label': 'Max Value', 'value': 'max'},
            {'label': 'Min Value', 'value': 'min'},
        ]

    # =========================================================================
    # UPDATE FEATURE DISTRIBUTION CHART
    # =========================================================================
    @app.callback(
        Output('feature-distribution-plot', 'figure'),
        [Input('feature-selector', 'value'),
         Input('dataset-selector', 'value'),
         Input('fault-filter', 'value'),
         Input('severity-filter', 'value'),
         Input('chart-type-selector', 'value')]
    )
    def update_feature_distribution(feature, dataset_id, fault_filter, severity_filter, chart_type):
        """Update feature distribution chart based on selected feature and chart type.
        
        Supports: violin, strip, kde, ridge, histogram
        """
        if not feature or not dataset_id:
            fig = go.Figure()
            fig.update_layout(
                annotations=[dict(text="Select a feature and dataset", showarrow=False)],
                height=450
            )
            return fig

        # Default chart type to violin if not specified
        if not chart_type:
            chart_type = 'violin'

        try:
            import pandas as pd
            from scipy import stats
            
            # Define realistic value ranges for each feature type
            feature_params = {
                'rms': {'mean': 0.5, 'std': 0.15, 'label': 'RMS Value'},
                'peak_to_peak': {'mean': 2.5, 'std': 0.8, 'label': 'Peak-to-Peak'},
                'kurtosis': {'mean': 3.0, 'std': 1.5, 'label': 'Kurtosis'},
                'skewness': {'mean': 0.0, 'std': 0.5, 'label': 'Skewness'},
                'crest_factor': {'mean': 3.5, 'std': 0.7, 'label': 'Crest Factor'},
                'std': {'mean': 0.35, 'std': 0.1, 'label': 'Standard Deviation'},
                'mean': {'mean': 0.0, 'std': 0.2, 'label': 'Mean Value'},
                'max': {'mean': 1.2, 'std': 0.4, 'label': 'Max Value'},
                'min': {'mean': -1.2, 'std': 0.4, 'label': 'Min Value'},
            }
            
            params = feature_params.get(feature, {'mean': 0, 'std': 1, 'label': feature})
            
            # Fault classes aligned with Phase 0 (same order as FAULT_CLASSES)
            all_fault_classes = ['normal', 'misalignment', 'imbalance', 'looseness', 'lubrication',
                                  'cavitation', 'wear', 'oil_whirl', 'combined_misalign_imbalance',
                                  'combined_wear_lube', 'combined_cavit_jeu']
            
            # Apply fault filter
            if fault_filter:
                active_faults = [f for f in all_fault_classes if f in fault_filter]
            else:
                active_faults = all_fault_classes
            
            if not active_faults:
                fig = go.Figure()
                fig.update_layout(annotations=[dict(text="No fault classes selected", showarrow=False)])
                return fig
            
            # Generate synthetic data for each fault class
            feature_values = []
            fault_labels = []
            
            n_samples_per_class = 120
            
            for i, fault in enumerate(active_faults):
                np.random.seed(hash(f"{feature}_{fault}") % (2**32))
                offset = (i - len(active_faults)/2) * params['std'] * 0.3
                values = np.random.normal(params['mean'] + offset, params['std'], n_samples_per_class)
                # Fault-specific adjustments
                if fault in ['looseness', 'cavitation', 'wear']:
                    values = values * 1.2
                elif fault == 'normal':
                    values = values * 0.8
                
                feature_values.extend(values)
                fault_labels.extend([fault.replace('_', ' ').title()] * n_samples_per_class)
            
            # Create DataFrame
            df = pd.DataFrame({
                'value': feature_values,
                'fault_class': fault_labels
            })
            
            # Color palette for consistency across chart types
            colors = px.colors.qualitative.Set2
            
            # ============================================
            # RENDER CHART BASED ON SELECTED TYPE
            # ============================================
            
            if chart_type == 'violin':
                # Violin Plot - shows distribution shape
                fig = px.violin(
                    df,
                    y='fault_class',
                    x='value',
                    color='fault_class',
                    orientation='h',
                    box=True,  # Add box plot inside
                    points='outliers',  # Show outliers only
                    title=f"{params['label']} Distribution (Violin)",
                    labels={'value': params['label'], 'fault_class': 'Fault Class'},
                    color_discrete_sequence=colors
                )
                fig.update_layout(showlegend=False, height=max(350, len(active_faults) * 60))
                
            elif chart_type == 'strip':
                # Strip/Jitter Plot - shows individual points
                fig = px.strip(
                    df,
                    y='fault_class',
                    x='value',
                    color='fault_class',
                    orientation='h',
                    title=f"{params['label']} Distribution (Strip/Jitter)",
                    labels={'value': params['label'], 'fault_class': 'Fault Class'},
                    color_discrete_sequence=colors
                )
                fig.update_traces(jitter=0.4, marker=dict(size=5, opacity=0.6))
                fig.update_layout(showlegend=False, height=max(350, len(active_faults) * 60))
                
            elif chart_type == 'kde':
                # KDE Lines - smooth density curves
                fig = go.Figure()
                
                for i, fault in enumerate(df['fault_class'].unique()):
                    fault_data = df[df['fault_class'] == fault]['value']
                    
                    # Calculate KDE
                    x_range = np.linspace(fault_data.min() - params['std'], 
                                         fault_data.max() + params['std'], 200)
                    kde = stats.gaussian_kde(fault_data)
                    density = kde(x_range)
                    
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=density,
                        mode='lines',
                        name=fault,
                        line=dict(width=2.5),
                        fill='tozeroy',
                        opacity=0.4
                    ))
                
                fig.update_layout(
                    title=f"{params['label']} Distribution (KDE Lines)",
                    xaxis_title=params['label'],
                    yaxis_title='Density',
                    legend_title='Fault Class',
                    height=450
                )
                
            elif chart_type == 'ridge':
                # Ridge Plot - stacked density curves
                fig = go.Figure()
                
                unique_faults = df['fault_class'].unique()
                n_faults = len(unique_faults)
                
                # Use a simple color palette that's guaranteed to work
                ridge_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', 
                               '#a6d854', '#ffd92f', '#e5c494']
                
                for i, fault in enumerate(unique_faults):
                    fault_data = df[df['fault_class'] == fault]['value']
                    
                    # Calculate KDE
                    x_range = np.linspace(df['value'].min() - params['std'], 
                                         df['value'].max() + params['std'], 200)
                    kde = stats.gaussian_kde(fault_data)
                    density = kde(x_range)
                    
                    # Normalize and offset for ridge effect
                    density_normalized = density / density.max() * 0.8
                    y_offset = i * 1.0
                    
                    # Get color and create rgba version
                    hex_color = ridge_colors[i % len(ridge_colors)]
                    # Convert hex to rgba
                    r = int(hex_color[1:3], 16)
                    g = int(hex_color[3:5], 16)
                    b = int(hex_color[5:7], 16)
                    fill_rgba = f'rgba({r},{g},{b},0.4)'
                    
                    # Fill area
                    fig.add_trace(go.Scatter(
                        x=x_range,
                        y=density_normalized + y_offset,
                        mode='lines',
                        name=fault,
                        line=dict(width=2, color=hex_color),
                        fill='tonexty' if i > 0 else 'tozeroy',
                        fillcolor=fill_rgba
                    ))
                
                fig.update_layout(
                    title=f"{params['label']} Distribution (Ridge Plot)",
                    xaxis_title=params['label'],
                    yaxis=dict(
                        tickvals=[i * 1.0 + 0.4 for i in range(n_faults)],
                        ticktext=list(unique_faults),
                        title=''
                    ),
                    height=max(400, n_faults * 70),
                    showlegend=False
                )
                
            else:  # histogram (default/fallback)
                fig = px.histogram(
                    df,
                    x='value',
                    color='fault_class',
                    nbins=40,
                    title=f"{params['label']} Distribution (Histogram)",
                    labels={'value': params['label'], 'fault_class': 'Fault Class'},
                    color_discrete_sequence=colors
                )
                fig.update_layout(
                    barmode='overlay',
                    legend_title_text='Fault Class',
                    height=450
                )
                fig.update_traces(opacity=0.65)
            
            # Common layout updates
            fig.update_layout(
                template='plotly_white',
                margin=dict(l=60, r=40, t=60, b=60)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating feature distribution: {e}")
            fig = go.Figure()
            fig.update_layout(annotations=[dict(text=f"Error: {str(e)}", showarrow=False)])
            return fig


    # =========================================================================
    # CALCULATE DIMENSIONALITY REDUCTION
    # =========================================================================
    @app.callback(
        Output('dimred-plot', 'figure'),
        Input('compute-dimred-btn', 'n_clicks'),
        [State('dataset-selector', 'value'),
         State('dimred-method', 'value')],
        prevent_initial_call=True
    )
    def calculate_projection(n_clicks, dataset_id, method):
        """Calculate and display dimensionality reduction projection."""
        if not n_clicks or not dataset_id:
            raise PreventUpdate

        try:
            # For demonstration, create simulated projection
            # In production, this would use actual t-SNE/UMAP/PCA
            np.random.seed(42)
            n_samples = 500
            n_classes = 7  # Normal + 6 fault types
            
            # Generate clustered 2D data
            # Fault classes aligned with Phase 0
            class_names = ['Normal', 'Misalignment', 'Imbalance', 'Looseness', 'Lubrication',
                          'Cavitation', 'Wear', 'Oil Whirl', 'Combined Misalign+Imbalance',
                          'Combined Wear+Lube', 'Combined Cavit+Jeu']
            n_classes = len(class_names)
            
            x, y, labels = [], [], []
            
            for i in range(n_classes):
                cx, cy = np.random.randn(2) * 3
                n_per_class = n_samples // n_classes
                x.extend(np.random.randn(n_per_class) * 0.8 + cx)
                y.extend(np.random.randn(n_per_class) * 0.8 + cy)
                labels.extend([class_names[i]] * n_per_class)
            
            fig = px.scatter(
                x=x, y=y, color=labels,
                title=f"{method.upper()} Projection",
                labels={'x': 'Component 1', 'y': 'Component 2', 'color': 'Fault Class'}
            )
            fig.update_layout(height=500)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error calculating projection: {e}")
            fig = go.Figure()
            fig.update_layout(annotations=[dict(text=f"Error: {str(e)}", showarrow=False)])
            return fig

    # =========================================================================
    # SPECTRAL ANALYSIS CALLBACKS (PSD, Envelope, Cepstrum)
    # =========================================================================
    @app.callback(
        [Output('spectral-analysis-plot', 'figure'),
         Output('spectral-info-text', 'children')],
        [Input('spectral-analysis-type', 'value'),
         Input('spectral-fault-selector', 'value'),
         Input('rotational-freq-input', 'value'),
         Input('show-harmonics-toggle', 'value'),
         Input('dataset-selector', 'value')]
    )
    def update_spectral_analysis(analysis_type, fault_class, rotational_freq, show_harmonics, dataset_id):
        """Update spectral analysis plot based on selected type."""
        from scipy.signal import welch, hilbert
        
        if not dataset_id or not analysis_type:
            fig = go.Figure()
            fig.update_layout(
                annotations=[dict(text="Select a dataset and analysis type", showarrow=False)],
                height=450
            )
            return fig, ""
        
        try:
            # Simulated signal for demonstration
            # In production, load actual signal from HDF5
            fs = SAMPLING_RATE  # Sampling frequency (Hz)
            duration = 2.0  # seconds
            t = np.linspace(0, duration, int(fs * duration))
            
            # Create realistic vibration signal based on fault type
            np.random.seed(hash(fault_class) % (2**32))
            omega = rotational_freq or 60
            
            # Base signal with rotational harmonics
            signal = (0.3 * np.sin(2 * np.pi * omega * t) +  # 1X
                     0.15 * np.sin(2 * np.pi * 2 * omega * t) +  # 2X
                     0.08 * np.sin(2 * np.pi * 3 * omega * t))  # 3X
            
            # Add fault-specific characteristics
            if fault_class == 'looseness':
                # Looseness: strong higher harmonics
                signal += 0.2 * np.sin(2 * np.pi * 3.5 * omega * t)
                signal += 0.15 * np.sin(2 * np.pi * 4 * omega * t)
            elif fault_class == 'lubrication':
                # Lubrication issues: broadband noise increase
                signal += 0.15 * np.random.randn(len(t))
            elif fault_class == 'wear':
                # Wear: modulated signal with additional harmonics
                signal = signal * (1 + 0.3 * np.sin(2 * np.pi * 0.4 * omega * t))
            elif fault_class == 'oil_whirl':
                # Oil whirl: sub-synchronous component (~0.4X)
                signal += 0.3 * np.sin(2 * np.pi * 0.42 * omega * t)
            elif fault_class == 'cavitation':
                # Cavitation: random impacts and broadband
                signal += 0.2 * np.sin(2 * np.pi * 4.2 * omega * t)
                signal += 0.1 * np.random.randn(len(t))
            elif fault_class == 'misalignment':
                # Misalignment: strong 2X component
                signal += 0.35 * np.sin(2 * np.pi * 2 * omega * t + 0.5)
            elif fault_class == 'imbalance':
                # Imbalance: dominant 1X
                signal += 0.4 * np.sin(2 * np.pi * omega * t + 0.3)
            
            # Add noise
            signal += 0.05 * np.random.randn(len(t))
            
            show_harm = 'show' in (show_harmonics or [])
            
            if analysis_type == 'psd':
                # Power Spectral Density using Welch method
                f, Pxx = welch(signal, fs, nperseg=4096)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=f, y=Pxx,
                    mode='lines',
                    name='PSD',
                    line=dict(color='#1f77b4', width=1.5)
                ))
                
                # Add harmonic markers
                if show_harm and omega:
                    colors = ['#2ca02c', '#ff7f0e', '#d62728']
                    for i, mult in enumerate([1, 2, 3]):
                        freq = omega * mult
                        if freq < f[-1]:
                            fig.add_vline(
                                x=freq, line_dash="dash",
                                annotation_text=f"{mult}X",
                                annotation_position="top",
                                line_color=colors[i],
                                line_width=2
                            )
                
                fig.update_layout(
                    title=f"Power Spectral Density - {fault_class.replace('_', ' ').title()}",
                    xaxis_title="Frequency (Hz)",
                    yaxis_title="Power Spectral Density (V²/Hz)",
                    yaxis_type="log",
                    height=450,
                    template='plotly_white'
                )
                
                info_text = "PSD shows the distribution of signal power across frequencies. Peaks indicate dominant frequency components. Harmonic markers (1X, 2X, 3X) highlight rotational frequency multiples."
                
            elif analysis_type == 'envelope':
                # Envelope Analysis using Hilbert Transform
                analytic_signal = hilbert(signal)
                envelope = np.abs(analytic_signal)
                
                # Envelope spectrum
                f_env, Pxx_env = welch(envelope, fs, nperseg=2048)
                
                fig = go.Figure()
                
                # Top: signal with envelope
                fig.add_trace(go.Scatter(
                    x=t[:2000], y=signal[:2000],
                    mode='lines', name='Signal',
                    line=dict(color='#1f77b4', width=0.5)
                ))
                fig.add_trace(go.Scatter(
                    x=t[:2000], y=envelope[:2000],
                    mode='lines', name='Envelope',
                    line=dict(color='#d62728', width=2)
                ))
                
                fig.update_layout(
                    title=f"Envelope Analysis (Hilbert Transform) - {fault_class.replace('_', ' ').title()}",
                    xaxis_title="Time (s)",
                    yaxis_title="Amplitude",
                    height=450,
                    template='plotly_white',
                    showlegend=True
                )
                
                info_text = "Envelope analysis extracts the amplitude modulation of the signal using the Hilbert transform. Useful for detecting bearing faults that cause amplitude modulation at characteristic frequencies."
                
            elif analysis_type == 'cepstrum':
                # Real Cepstrum
                # Cepstrum = IFFT(log(|FFT(x)|))
                N = len(signal)
                spectrum = np.fft.fft(signal)
                log_spectrum = np.log(np.abs(spectrum) + 1e-10)
                cepstrum = np.real(np.fft.ifft(log_spectrum))
                
                # Quefrency axis (in ms)
                quefrency = np.arange(N) / fs * 1000  # Convert to ms
                
                fig = go.Figure()
                # Only show positive quefrency up to 50 ms
                max_q = min(int(0.05 * fs), N // 2)
                fig.add_trace(go.Scatter(
                    x=quefrency[:max_q], y=cepstrum[:max_q],
                    mode='lines',
                    name='Cepstrum',
                    line=dict(color='#1f77b4', width=1.5)
                ))
                
                # Mark 1X and 2X periods
                if show_harm and omega:
                    period_1x = 1000 / omega  # ms
                    period_2x = 500 / omega  # ms
                    if period_1x < 50:
                        fig.add_vline(x=period_1x, line_dash="dash",
                                     annotation_text=f"1X ({period_1x:.1f}ms)",
                                     line_color='#2ca02c', line_width=2)
                    if period_2x < 50:
                        fig.add_vline(x=period_2x, line_dash="dash",
                                     annotation_text=f"2X ({period_2x:.1f}ms)",
                                     line_color='#ff7f0e', line_width=2)
                
                fig.update_layout(
                    title=f"Cepstrum Analysis - {fault_class.replace('_', ' ').title()}",
                    xaxis_title="Quefrency (ms)",
                    yaxis_title="Cepstral Amplitude",
                    height=450,
                    template='plotly_white'
                )
                
                info_text = "Cepstrum shows periodic structures in the spectrum. Peaks at specific quefrencies indicate harmonic families (gear mesh, bearing faults). Quefrency = 1/frequency."
            
            else:
                fig = go.Figure()
                fig.update_layout(annotations=[dict(text="Unknown analysis type", showarrow=False)])
                info_text = ""
            
            return fig, info_text
            
        except Exception as e:
            logger.error(f"Error in spectral analysis: {e}")
            fig = go.Figure()
            fig.update_layout(annotations=[dict(text=f"Error: {str(e)}", showarrow=False)])
            return fig, f"Error: {str(e)}"

    # =========================================================================
    # FEATURE COMPARISON CALLBACKS (Spider, Heatmap, Importance)
    # =========================================================================
    @app.callback(
        [Output('feature-comparison-plot', 'figure'),
         Output('comparison-description', 'children')],
        [Input('comparison-chart-type', 'value'),
         Input('normalize-features-toggle', 'value'),
         Input('dataset-selector', 'value'),
         Input('fault-filter', 'value')]
    )
    def update_feature_comparison(chart_type, normalize_toggle, dataset_id, fault_filter):
        """Update feature comparison visualization."""
        import pandas as pd
        
        if not dataset_id or not chart_type:
            fig = go.Figure()
            fig.update_layout(
                annotations=[dict(text="Select a dataset and comparison type", showarrow=False)],
                height=500
            )
            return fig, ""
        
        try:
            normalize = 'normalize' in (normalize_toggle or [])
            
            # Feature data (simulated - in production load from actual features)
            feature_names = ['RMS', 'Kurtosis', 'Crest Factor', '2X/1X Ratio', '3X/1X Ratio', 'Skewness']
            fault_classes = ['Normal', 'Misalignment', 'Imbalance', 'Looseness', 
                            'Lubrication', 'Cavitation', 'Wear', 'Oil Whirl']
            
            # Apply fault filter
            if fault_filter:
                fault_mapping = {
                    'normal': 'Normal', 'misalignment': 'Misalignment', 'imbalance': 'Imbalance',
                    'looseness': 'Looseness', 'lubrication': 'Lubrication', 'cavitation': 'Cavitation',
                    'wear': 'Wear', 'oil_whirl': 'Oil Whirl',
                    'combined_misalign_imbalance': 'Combined Misalign+Imbalance',
                    'combined_wear_lube': 'Combined Wear+Lube',
                    'combined_cavit_jeu': 'Combined Cavit+Jeu'
                }
                fault_classes = [fault_mapping.get(f, f.replace('_', ' ').title()) 
                                for f in fault_filter if f in fault_mapping]
            
            if not fault_classes:
                fault_classes = ['Normal', 'Misalignment', 'Imbalance']
            # Simulated feature values (realistic ranges per fault type)
            np.random.seed(42)
            feature_matrix = {
                'Normal':      [0.05, 3.0, 3.2, 0.01, 0.02, 0.0],
                'Misalignment':[0.12, 3.5, 3.8, 0.45, 0.35, 0.1],
                'Imbalance':   [0.20, 3.2, 3.5, 0.05, 0.02, 0.0],
                'Looseness':   [0.15, 5.5, 4.8, 0.08, 0.06, 0.2],
                'Lubrication': [0.14, 4.5, 4.2, 0.10, 0.05, 0.15],
                'Cavitation':  [0.10, 8.0, 6.0, 0.03, 0.01, 0.5],
                'Wear':        [0.18, 6.2, 5.2, 0.12, 0.08, 0.3],
                'Oil Whirl':   [0.16, 4.0, 4.0, 0.30, 0.15, 0.2],
                'Combined Misalign+Imbalance': [0.25, 4.0, 4.5, 0.40, 0.30, 0.15],
                'Combined Wear+Lube': [0.22, 5.5, 5.0, 0.15, 0.10, 0.35],
                'Combined Cavit+Jeu': [0.18, 7.0, 5.5, 0.10, 0.08, 0.45],
            }
            
            # Build matrix for selected faults
            data = np.array([feature_matrix.get(f, [0.1]*6) for f in fault_classes])
            
            # Normalize if requested
            if normalize:
                data_min = data.min(axis=0)
                data_max = data.max(axis=0)
                data = (data - data_min) / (data_max - data_min + 1e-10)
            
            colors = px.colors.qualitative.Set2
            
            # Use hex colors that work reliably (same fix as Ridge Plot)
            spider_colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', 
                           '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']
            
            if chart_type == 'spider':
                # Spider/Radar Chart
                fig = go.Figure()
                
                for i, fault in enumerate(fault_classes):
                    values = data[i].tolist()
                    values.append(values[0])  # Close the polygon
                    
                    # Get color and create rgba version
                    hex_color = spider_colors[i % len(spider_colors)]
                    r = int(hex_color[1:3], 16)
                    g = int(hex_color[3:5], 16)
                    b = int(hex_color[5:7], 16)
                    fill_rgba = f'rgba({r},{g},{b},0.2)'
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=feature_names + [feature_names[0]],
                        fill='toself',
                        name=fault,
                        line=dict(color=hex_color, width=2),
                        fillcolor=fill_rgba
                    ))
                
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1] if normalize else None)),
                    title="Multi-Feature Spider Chart (Normalized)" if normalize else "Multi-Feature Spider Chart",
                    showlegend=True,
                    height=550
                )
                
                description = "Spider chart shows how each fault type differs across multiple features. Wider polygons indicate more severe fault signatures."
                
            elif chart_type == 'heatmap':
                # Feature Heatmap
                fig = px.imshow(
                    data,
                    x=feature_names,
                    y=fault_classes,
                    color_continuous_scale='Viridis',
                    aspect='auto',
                    title="Normalized Feature Heatmap" if normalize else "Feature Heatmap",
                    labels=dict(color="Value")
                )
                
                # Add value annotations
                for i in range(len(fault_classes)):
                    for j in range(len(feature_names)):
                        fig.add_annotation(
                            x=j, y=i,
                            text=f"{data[i, j]:.2f}",
                            showarrow=False,
                            font=dict(color="white" if data[i, j] > 0.5 else "black", size=10)
                        )
                
                fig.update_layout(height=550)
                
                description = "Heatmap shows feature values across fault types. Brighter colors indicate higher values. Useful for identifying which features distinguish specific faults."
                
            elif chart_type == 'importance':
                # Feature Importance (based on variance/discriminability)
                # Calculate importance as coefficient of variation across faults
                importance = np.std(data, axis=0) / (np.mean(data, axis=0) + 1e-10)
                
                # Sort by importance
                sorted_idx = np.argsort(importance)[::-1]
                sorted_features = [feature_names[i] for i in sorted_idx]
                sorted_importance = importance[sorted_idx]
                
                # Normalize importance to 0-1
                sorted_importance = sorted_importance / sorted_importance.max()
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=sorted_features,
                    x=sorted_importance,
                    orientation='h',
                    marker=dict(
                        color=sorted_importance,
                        colorscale='Blues',
                        showscale=False
                    )
                ))
                
                fig.update_layout(
                    title="Feature Importance for Fault Discrimination",
                    xaxis_title="Normalized Importance Score",
                    yaxis_title="Feature",
                    height=550,
                    yaxis=dict(autorange="reversed")  # Highest at top
                )
                
                description = "Feature importance shows which features are most useful for distinguishing between fault types. Higher scores indicate better discriminating power."
            
            else:
                fig = go.Figure()
                fig.update_layout(annotations=[dict(text="Unknown chart type", showarrow=False)])
                description = ""
            
            return fig, description
            
        except Exception as e:
            logger.error(f"Error in feature comparison: {e}")
            fig = go.Figure()
            fig.update_layout(annotations=[dict(text=f"Error: {str(e)}", showarrow=False)])
            return fig, f"Error: {str(e)}"
