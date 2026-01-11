"""
Explainability Package

This package provides comprehensive explainability methods for neural networks:
- Integrated Gradients: Attribution using gradient integration
- SHAP: SHapley Additive exPlanations
- LIME: Local Interpretable Model-agnostic Explanations
- Uncertainty Quantification: Monte Carlo Dropout, calibration
- Concept Activation Vectors (CAVs): Concept-based explanations
- Partial Dependence: Feature effect analysis
- Anchors: Rule-based explanations

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

# Advanced XAI methods (Priority 3)
from .concept_activation_vectors import (
    ConceptActivationVector,
    CAVGenerator,
    TCAVAnalyzer,
    plot_tcav_results,
    plot_cav_comparison
)
from .partial_dependence import (
    PartialDependenceAnalyzer,
    plot_partial_dependence,
    plot_ice_curves,
    plot_partial_dependence_2d,
    detect_interactions
)
from .anchors import (
    Predicate,
    Anchor,
    AnchorExplainer,
    plot_anchor_explanation,
    compare_anchors
)

__all__ = [
    # Core XAI methods
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
    'plot_prediction_with_uncertainty',
    # Advanced XAI methods
    'ConceptActivationVector',
    'CAVGenerator',
    'TCAVAnalyzer',
    'plot_tcav_results',
    'plot_cav_comparison',
    'PartialDependenceAnalyzer',
    'plot_partial_dependence',
    'plot_ice_curves',
    'plot_partial_dependence_2d',
    'detect_interactions',
    'Predicate',
    'Anchor',
    'AnchorExplainer',
    'plot_anchor_explanation',
    'compare_anchors'
]
