"""
XAI Visualization Utilities for Plotly.
Converts explainability results to interactive Plotly visualizations for Dash.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Dict, Any, Tuple
from dashboard_config import FAULT_CLASSES as FAULT_CLASSES_CONFIG


class XAIVisualization:
    """Utility class for creating XAI visualizations in Plotly."""

    # Fault class names for labeling (imported from config)
    FAULT_CLASSES = FAULT_CLASSES_CONFIG

    @staticmethod
    def create_shap_signal_plot(
        signal: List[float],
        shap_values: List[float],
        time_labels: Optional[List[str]] = None,
        predicted_class: Optional[int] = None
    ) -> go.Figure:
        """
        Create interactive Plotly visualization of signal with SHAP attribution overlay.

        Args:
            signal: Original signal values
            shap_values: SHAP attribution values
            time_labels: Time step labels (default: indices)
            predicted_class: Predicted fault class

        Returns:
            Plotly Figure with dual-axis plot
        """
        signal_np = np.array(signal)
        shap_np = np.array(shap_values)

        if time_labels is None:
            time_labels = list(range(len(signal)))

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add signal trace (primary y-axis)
        fig.add_trace(
            go.Scatter(
                x=time_labels,
                y=signal_np,
                name='Signal',
                line=dict(color='#1f77b4', width=1.5),
                hovertemplate='<b>Signal</b><br>Time: %{x}<br>Value: %{y:.4f}<extra></extra>'
            ),
            secondary_y=False
        )

        # Add SHAP attribution bars (secondary y-axis)
        colors = ['rgba(220, 53, 69, 0.6)' if v < 0 else 'rgba(40, 167, 69, 0.6)' for v in shap_np]

        fig.add_trace(
            go.Bar(
                x=time_labels,
                y=shap_np,
                name='SHAP Attribution',
                marker_color=colors,
                marker_line_width=0,
                hovertemplate='<b>SHAP</b><br>Time: %{x}<br>Attribution: %{y:.4f}<extra></extra>'
            ),
            secondary_y=True
        )

        # Update layout
        title = "Signal with SHAP Attribution"
        if predicted_class is not None and predicted_class < len(XAIVisualization.FAULT_CLASSES):
            title += f" (Predicted: {XAIVisualization.FAULT_CLASSES[predicted_class]})"

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16, color='#2c3e50')
            ),
            xaxis_title="Time Steps",
            hovermode='x unified',
            template='plotly_white',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=60, r=60, t=80, b=60)
        )

        # Set y-axes titles
        fig.update_yaxes(title_text="Signal Amplitude", secondary_y=False)
        fig.update_yaxes(title_text="SHAP Value", secondary_y=True)

        return fig

    @staticmethod
    def create_shap_waterfall(
        shap_values: List[float],
        base_value: float,
        predicted_value: float,
        time_labels: Optional[List[str]] = None,
        top_k: int = 20
    ) -> go.Figure:
        """
        Create SHAP waterfall plot showing top feature contributions.

        Args:
            shap_values: SHAP attribution values
            base_value: Base value (expected output)
            predicted_value: Actual prediction
            time_labels: Feature names
            top_k: Number of top features to display

        Returns:
            Plotly Figure with waterfall visualization
        """
        shap_np = np.array(shap_values)

        # Get top-k by absolute value
        abs_shap = np.abs(shap_np)
        top_indices = np.argsort(abs_shap)[-top_k:][::-1]
        top_shap = shap_np[top_indices]

        if time_labels is None:
            time_labels = [f"t_{i}" for i in range(len(shap_np))]

        top_labels = [time_labels[i] if i < len(time_labels) else f"Feature {i}" for i in top_indices]

        # Create waterfall chart
        measure = ["relative"] * len(top_shap)
        text = [f"{val:+.4f}" for val in top_shap]

        fig = go.Figure(go.Waterfall(
            name="SHAP",
            orientation="h",
            measure=measure,
            y=top_labels,
            x=top_shap,
            text=text,
            textposition="outside",
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "#dc3545"}},
            increasing={"marker": {"color": "#28a745"}},
            totals={"marker": {"color": "#007bff"}}
        ))

        fig.update_layout(
            title=dict(
                text=f"Top {top_k} SHAP Feature Contributions",
                font=dict(size=16, color='#2c3e50')
            ),
            xaxis_title="SHAP Value (Impact on Prediction)",
            yaxis_title="",
            height=max(400, top_k * 25),
            template='plotly_white',
            margin=dict(l=150, r=60, t=80, b=60),
            showlegend=False
        )

        return fig

    @staticmethod
    def create_lime_segment_plot(
        signal: List[float],
        segment_weights: List[float],
        segment_boundaries: List[Tuple[int, int]],
        predicted_class: Optional[int] = None
    ) -> go.Figure:
        """
        Create LIME segment importance visualization with signal overlay.

        Args:
            signal: Original signal
            segment_weights: Importance weight for each segment
            segment_boundaries: List of (start, end) tuples for segments
            predicted_class: Predicted fault class

        Returns:
            Plotly Figure with segment visualization
        """
        signal_np = np.array(signal)
        time = np.arange(len(signal_np))

        # Normalize weights for coloring
        weights_np = np.array(segment_weights)
        max_abs_weight = np.abs(weights_np).max()

        if max_abs_weight == 0:
            weights_normalized = np.zeros_like(weights_np)
        else:
            weights_normalized = weights_np / max_abs_weight

        # Create figure with subplot
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            subplot_titles=("Signal with LIME Segments", "Segment Importance"),
            vertical_spacing=0.12
        )

        # Plot 1: Signal with colored segments
        fig.add_trace(
            go.Scatter(
                x=time,
                y=signal_np,
                name='Signal',
                line=dict(color='#1f77b4', width=1.5),
                hovertemplate='<b>Signal</b><br>Time: %{x}<br>Value: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Add colored segments as shapes
        for i, (start, end) in enumerate(segment_boundaries):
            weight_norm = weights_normalized[i]
            alpha = min(abs(weight_norm), 1.0) * 0.4

            if weight_norm >= 0:
                color = f'rgba(40, 167, 69, {alpha})'  # Green
            else:
                color = f'rgba(220, 53, 69, {alpha})'  # Red

            fig.add_vrect(
                x0=start, x1=end,
                fillcolor=color,
                layer="below",
                line_width=0,
                row=1, col=1
            )

        # Plot 2: Segment importance bars
        segment_centers = [(start + end) / 2 for start, end in segment_boundaries]
        segment_widths = [end - start for start, end in segment_boundaries]

        colors_bars = ['#dc3545' if w < 0 else '#28a745' for w in segment_weights]

        fig.add_trace(
            go.Bar(
                x=segment_centers,
                y=segment_weights,
                width=segment_widths,
                marker_color=colors_bars,
                marker_line_width=0.5,
                marker_line_color='#2c3e50',
                name='Segment Weight',
                hovertemplate='<b>Segment %{x}</b><br>Weight: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, row=2, col=1)

        # Update layout
        title = "LIME Explanation"
        if predicted_class is not None and predicted_class < len(XAIVisualization.FAULT_CLASSES):
            title += f" (Predicted: {XAIVisualization.FAULT_CLASSES[predicted_class]})"

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16, color='#2c3e50')
            ),
            height=700,
            template='plotly_white',
            showlegend=False,
            margin=dict(l=60, r=60, t=100, b=60)
        )

        # Update axes
        fig.update_xaxes(title_text="Time Steps", row=1, col=1)
        fig.update_xaxes(title_text="Time Steps", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Segment Weight", row=2, col=1)

        return fig

    @staticmethod
    def create_lime_bar_chart(
        segment_weights: List[float],
        top_k: int = 15,
        predicted_class: Optional[int] = None
    ) -> go.Figure:
        """
        Create bar chart of top LIME segment importances.

        Args:
            segment_weights: Segment importance weights
            top_k: Number of top segments to show
            predicted_class: Predicted fault class

        Returns:
            Plotly Figure with bar chart
        """
        weights_np = np.array(segment_weights)

        # Get top-k by absolute value
        abs_weights = np.abs(weights_np)
        top_indices = np.argsort(abs_weights)[-top_k:][::-1]
        top_weights = weights_np[top_indices]

        labels = [f"Segment {i+1}" for i in top_indices]
        colors = ['#dc3545' if w < 0 else '#28a745' for w in top_weights]

        fig = go.Figure(go.Bar(
            y=labels,
            x=top_weights,
            orientation='h',
            marker_color=colors,
            marker_line_color='#2c3e50',
            marker_line_width=0.5,
            text=[f"{w:+.3f}" for w in top_weights],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Weight: %{x:.4f}<extra></extra>'
        ))

        title = f"Top {top_k} LIME Segment Importances"
        if predicted_class is not None and predicted_class < len(XAIVisualization.FAULT_CLASSES):
            title += f"<br><sub>Predicted Class: {XAIVisualization.FAULT_CLASSES[predicted_class]}</sub>"

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16, color='#2c3e50')
            ),
            xaxis_title="LIME Weight (Contribution to Prediction)",
            yaxis_title="",
            height=max(400, top_k * 30),
            template='plotly_white',
            margin=dict(l=120, r=60, t=100, b=60),
            showlegend=False
        )

        # Add zero line
        fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)

        return fig

    @staticmethod
    def create_attribution_plot(
        signal: List[float],
        attributions: List[float],
        method: str,
        predicted_class: Optional[int] = None
    ) -> go.Figure:
        """
        Create generic attribution plot for Integrated Gradients or Grad-CAM.

        Args:
            signal: Original signal
            attributions: Attribution values
            method: Method name ('integrated_gradients' or 'gradcam')
            predicted_class: Predicted fault class

        Returns:
            Plotly Figure with attribution visualization
        """
        signal_np = np.array(signal)
        attr_np = np.array(attributions)
        time = np.arange(len(signal_np))

        # Normalize attributions for coloring
        attr_abs_max = np.abs(attr_np).max()
        if attr_abs_max > 0:
            attr_normalized = attr_np / attr_abs_max
        else:
            attr_normalized = np.zeros_like(attr_np)

        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.6, 0.4],
            subplot_titles=("Original Signal", f"{method.replace('_', ' ').title()} Attribution"),
            vertical_spacing=0.12
        )

        # Plot 1: Original signal
        fig.add_trace(
            go.Scatter(
                x=time,
                y=signal_np,
                name='Signal',
                line=dict(color='#1f77b4', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)',
                hovertemplate='<b>Signal</b><br>Time: %{x}<br>Value: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Plot 2: Attributions with color gradient
        colors = []
        for val in attr_normalized:
            if val > 0:
                # Green gradient
                colors.append(f'rgba(40, 167, 69, {abs(val) * 0.8})')
            else:
                # Red gradient
                colors.append(f'rgba(220, 53, 69, {abs(val) * 0.8})')

        fig.add_trace(
            go.Bar(
                x=time,
                y=attr_np,
                marker_color=colors,
                marker_line_width=0,
                name='Attribution',
                hovertemplate='<b>Attribution</b><br>Time: %{x}<br>Value: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, row=2, col=1)

        # Update layout
        title = f"{method.replace('_', ' ').title()} Explanation"
        if predicted_class is not None and predicted_class < len(XAIVisualization.FAULT_CLASSES):
            title += f" (Predicted: {XAIVisualization.FAULT_CLASSES[predicted_class]})"

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=16, color='#2c3e50')
            ),
            height=650,
            template='plotly_white',
            showlegend=False,
            margin=dict(l=60, r=60, t=100, b=60)
        )

        # Update axes
        fig.update_xaxes(title_text="Time Steps", row=1, col=1)
        fig.update_xaxes(title_text="Time Steps", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Attribution Value", row=2, col=1)

        return fig

    @staticmethod
    def create_feature_importance_summary(
        attributions: List[float],
        top_k: int = 20
    ) -> go.Figure:
        """
        Create feature importance summary plot.

        Args:
            attributions: Attribution values for all features
            top_k: Number of top features to display

        Returns:
            Plotly Figure with importance summary
        """
        attr_np = np.array(attributions)
        abs_attr = np.abs(attr_np)

        # Get top-k
        top_indices = np.argsort(abs_attr)[-top_k:][::-1]
        top_values = attr_np[top_indices]
        labels = [f"Time {i}" for i in top_indices]

        colors = ['#dc3545' if v < 0 else '#28a745' for v in top_values]

        fig = go.Figure(go.Bar(
            y=labels,
            x=abs_attr[top_indices],  # Use absolute values
            orientation='h',
            marker_color=colors,
            marker_line_color='#2c3e50',
            marker_line_width=0.5,
            text=[f"{v:+.3f}" for v in top_values],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))

        fig.update_layout(
            title=dict(
                text=f"Top {top_k} Feature Importances",
                font=dict(size=16, color='#2c3e50')
            ),
            xaxis_title="Absolute Importance",
            yaxis_title="",
            height=max(400, top_k * 25),
            template='plotly_white',
            margin=dict(l=100, r=60, t=80, b=60),
            showlegend=False
        )

        return fig

    @staticmethod
    def create_counterfactual_signal_plot(
        signal: List[float],
        counterfactual: List[float],
        perturbation: List[float],
        predicted_class: Optional[int] = None,
        target_class: Optional[int] = None
    ) -> go.Figure:
        """
        Create interactive Plotly visualization of counterfactual explanation.

        Shows original signal, counterfactual signal, and the perturbation needed.

        Args:
            signal: Original signal values
            counterfactual: Counterfactual signal values
            perturbation: Difference between counterfactual and original
            predicted_class: Original predicted class
            target_class: Target class for counterfactual

        Returns:
            Plotly Figure with 3-panel subplot
        """
        signal_np = np.array(signal)
        cf_np = np.array(counterfactual)
        diff_np = np.array(perturbation)
        time = np.arange(len(signal_np))

        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.4, 0.4, 0.2],
            subplot_titles=(
                "Original vs Counterfactual Signal",
                "Counterfactual Signal",
                "Required Perturbation"
            ),
            vertical_spacing=0.08
        )

        # Plot 1: Original signal
        fig.add_trace(
            go.Scatter(
                x=time, y=signal_np,
                name='Original',
                line=dict(color='#1f77b4', width=1.5),
                hovertemplate='<b>Original</b><br>Time: %{x}<br>Value: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Overlay counterfactual
        fig.add_trace(
            go.Scatter(
                x=time, y=cf_np,
                name='Counterfactual',
                line=dict(color='#dc3545', width=1.5, dash='dot'),
                hovertemplate='<b>Counterfactual</b><br>Time: %{x}<br>Value: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Plot 2: Counterfactual alone
        fig.add_trace(
            go.Scatter(
                x=time, y=cf_np,
                name='Counterfactual Signal',
                line=dict(color='#dc3545', width=1.5),
                showlegend=False,
                fill='tozeroy',
                fillcolor='rgba(220, 53, 69, 0.1)',
                hovertemplate='<b>Counterfactual</b><br>Time: %{x}<br>Value: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1
        )

        # Plot 3: Perturbation (difference)
        colors = ['rgba(220, 53, 69, 0.6)' if v < 0 else 'rgba(40, 167, 69, 0.6)' for v in diff_np]

        fig.add_trace(
            go.Bar(
                x=time, y=diff_np,
                name='Perturbation',
                marker_color=colors,
                marker_line_width=0,
                hovertemplate='<b>Perturbation</b><br>Time: %{x}<br>Δ: %{y:.4f}<extra></extra>'
            ),
            row=3, col=1
        )

        fig.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, row=3, col=1)

        # Highlight significant changes on overlay plot
        threshold = np.abs(diff_np).std() * 2
        significant = np.abs(diff_np) > threshold
        sig_regions = []
        in_region = False
        start = 0
        for i in range(len(significant)):
            if significant[i] and not in_region:
                start = i
                in_region = True
            elif not significant[i] and in_region:
                sig_regions.append((start, i))
                in_region = False
        if in_region:
            sig_regions.append((start, len(significant)))

        for s, e in sig_regions:
            fig.add_vrect(x0=s, x1=e, fillcolor='rgba(255, 193, 7, 0.2)',
                         layer="below", line_width=0, row=1, col=1)

        # Title
        title = "Counterfactual Explanation"
        if predicted_class is not None and predicted_class < len(XAIVisualization.FAULT_CLASSES):
            title += f" ({XAIVisualization.FAULT_CLASSES[predicted_class]}"
            if target_class is not None and target_class < len(XAIVisualization.FAULT_CLASSES):
                title += f" → {XAIVisualization.FAULT_CLASSES[target_class]}"
            title += ")"

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color='#2c3e50')),
            height=750,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=60, r=60, t=100, b=60)
        )

        fig.update_xaxes(title_text="Time Steps", row=3, col=1)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Amplitude", row=2, col=1)
        fig.update_yaxes(title_text="Change (Δ)", row=3, col=1)

        return fig

    @staticmethod
    def create_counterfactual_importance(
        perturbation: List[float],
        target_class: Optional[int] = None,
        top_k: int = 20
    ) -> go.Figure:
        """
        Create bar chart of top perturbation magnitudes for counterfactual.

        Args:
            perturbation: Perturbation values
            target_class: Target class
            top_k: Number of top time steps to show

        Returns:
            Plotly Figure with importance bars
        """
        diff_np = np.abs(np.array(perturbation))
        top_indices = np.argsort(diff_np)[-top_k:][::-1]
        top_values = np.array(perturbation)[top_indices]
        labels = [f"Time {i}" for i in top_indices]
        colors = ['#dc3545' if v < 0 else '#28a745' for v in top_values]

        fig = go.Figure(go.Bar(
            y=labels, x=np.abs(top_values),
            orientation='h',
            marker_color=colors,
            marker_line_color='#2c3e50',
            marker_line_width=0.5,
            text=[f"{v:+.4f}" for v in top_values],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Change: %{x:.4f}<extra></extra>'
        ))

        title = f"Top {top_k} Time Steps with Largest Changes"
        if target_class is not None and target_class < len(XAIVisualization.FAULT_CLASSES):
            title += f"<br><sub>Target: {XAIVisualization.FAULT_CLASSES[target_class]}</sub>"

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color='#2c3e50')),
            xaxis_title="Absolute Perturbation",
            height=max(400, top_k * 25),
            template='plotly_white',
            margin=dict(l=100, r=60, t=100, b=60),
            showlegend=False
        )

        return fig

    @staticmethod
    def create_activation_maps_plot(
        signal: List[float],
        layer_stats: List[Dict[str, Any]],
        predicted_class: Optional[int] = None
    ) -> go.Figure:
        """
        Create activation maps visualization showing layer-by-layer statistics.

        Args:
            signal: Original signal
            layer_stats: Layer activation statistics
            predicted_class: Predicted class

        Returns:
            Plotly Figure with activation analysis
        """
        signal_np = np.array(signal)
        time = np.arange(len(signal_np))

        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.5, 0.5],
            subplot_titles=("Input Signal", "Layer Activation Magnitudes"),
            vertical_spacing=0.12
        )

        # Plot 1: Signal
        fig.add_trace(
            go.Scatter(
                x=time, y=signal_np,
                name='Signal',
                line=dict(color='#1f77b4', width=1.5),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)',
                hovertemplate='<b>Signal</b><br>Time: %{x}<br>Value: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )

        # Plot 2: Layer activation bar chart
        if layer_stats:
            layers = [s['layer'] for s in layer_stats]
            mean_acts = [s['mean_activation'] for s in layer_stats]
            max_acts = [s['max_activation'] for s in layer_stats]
            num_filters = [s['num_filters'] for s in layer_stats]

            fig.add_trace(
                go.Bar(
                    x=layers, y=mean_acts,
                    name='Mean Activation',
                    marker_color='#17a2b8',
                    hovertemplate='<b>%{x}</b><br>Mean: %{y:.4f}<br>'
                                  'Filters: %{customdata}<extra></extra>',
                    customdata=num_filters
                ),
                row=2, col=1
            )

            fig.add_trace(
                go.Bar(
                    x=layers, y=max_acts,
                    name='Max Activation',
                    marker_color='#fd7e14',
                    hovertemplate='<b>%{x}</b><br>Max: %{y:.4f}<extra></extra>'
                ),
                row=2, col=1
            )

        title = "Activation Maps Analysis"
        if predicted_class is not None and predicted_class < len(XAIVisualization.FAULT_CLASSES):
            title += f" (Predicted: {XAIVisualization.FAULT_CLASSES[predicted_class]})"

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color='#2c3e50')),
            height=700,
            template='plotly_white',
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=60, r=60, t=100, b=60)
        )

        fig.update_xaxes(title_text="Time Steps", row=1, col=1)
        fig.update_xaxes(title_text="Layer", row=2, col=1, tickangle=-45)
        fig.update_yaxes(title_text="Amplitude", row=1, col=1)
        fig.update_yaxes(title_text="Activation Value", row=2, col=1)

        return fig

    @staticmethod
    def create_activation_layer_summary(
        layer_stats: List[Dict[str, Any]],
        predicted_class: Optional[int] = None
    ) -> go.Figure:
        """
        Create a summary chart of filter counts and spatial sizes across layers.

        Args:
            layer_stats: Layer activation statistics
            predicted_class: Predicted class

        Returns:
            Plotly Figure with layer summary
        """
        if not layer_stats:
            fig = go.Figure()
            fig.update_layout(
                title="No convolutional layers found",
                template='plotly_white',
                height=400
            )
            return fig

        layers = [s['layer'] for s in layer_stats]
        num_filters = [s['num_filters'] for s in layer_stats]
        spatial_sizes = [str(s['spatial_size']) for s in layer_stats]

        fig = go.Figure(go.Bar(
            x=layers,
            y=num_filters,
            marker_color='#6f42c1',
            marker_line_width=0.5,
            marker_line_color='#2c3e50',
            text=[f"{n} filters\n{s}" for n, s in zip(num_filters, spatial_sizes)],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Filters: %{y}<br>Spatial: %{text}<extra></extra>'
        ))

        title = "Network Architecture: Filter Counts per Layer"
        if predicted_class is not None and predicted_class < len(XAIVisualization.FAULT_CLASSES):
            title += f"<br><sub>Predicted: {XAIVisualization.FAULT_CLASSES[predicted_class]}</sub>"

        fig.update_layout(
            title=dict(text=title, font=dict(size=16, color='#2c3e50')),
            yaxis_title="Number of Filters",
            xaxis_title="Layer",
            height=400,
            template='plotly_white',
            margin=dict(l=60, r=60, t=100, b=100),
            showlegend=False,
            xaxis_tickangle=-45
        )

        return fig
