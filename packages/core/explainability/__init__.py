"""
Explainability Package

Kept XAI surface for contribution C4 (physics-consistent explanations):
- Integrated Gradients: attribution by gradient integration
- SHAP: SHapley Additive exPlanations
- Uncertainty Quantification: MC-Dropout, calibration (contribution C3/C5)

(LIME/anchors/CAVs/counterfactuals/partial-dependence were pruned 2026-06;
recoverable from tag `pre-convergence-2026-06`.)
"""

from .integrated_gradients import IntegratedGradientsExplainer, plot_attribution_map
from .shap_explainer import SHAPExplainer, plot_shap_waterfall, plot_shap_summary
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
    'UncertaintyQuantifier',
    'calibrate_model',
    'plot_calibration_curve',
    'plot_uncertainty_distribution',
    'plot_prediction_with_uncertainty',
]
