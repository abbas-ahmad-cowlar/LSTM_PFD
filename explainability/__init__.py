"""
Explainability Package

This package provides comprehensive explainability methods for neural networks:
- Integrated Gradients: Attribution using gradient integration
- SHAP: SHapley Additive exPlanations
- LIME: Local Interpretable Model-agnostic Explanations
- Uncertainty Quantification: Monte Carlo Dropout, calibration

All methods work with time-series bearing fault diagnosis models.
"""

from .integrated_gradients import IntegratedGradientsExplainer, plot_attribution_map
from .shap_explainer import SHAPExplainer, plot_shap_waterfall, plot_shap_summary
from .lime_explainer import LIMEExplainer, plot_lime_explanation
from .uncertainty_quantification import (
    UncertaintyQuantifier,
    calibrate_model,
    plot_calibration_curve,
    plot_uncertainty_distribution,
    plot_prediction_with_uncertainty
)

__all__ = [
    'IntegratedGradientsExplainer',
    'plot_attribution_map',
    'SHAPExplainer',
    'plot_shap_waterfall',
    'plot_shap_summary',
    'LIMEExplainer',
    'plot_lime_explanation',
    'UncertaintyQuantifier',
    'calibrate_model',
    'plot_calibration_curve',
    'plot_uncertainty_distribution',
    'plot_prediction_with_uncertainty'
]
