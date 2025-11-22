"""
XAI Service (Phase 11C).
Integrates with Phase 7 explainability functionality.
"""
import sys
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.logger import setup_logger

logger = setup_logger(__name__)


class XAIService:
    """Service for generating model explanations."""

    @staticmethod
    def generate_shap_explanation(model, signal, background_data=None, num_samples=100):
        """
        Generate SHAP explanation for a signal.

        Args:
            model: Trained PyTorch model
            signal: Input signal to explain
            background_data: Background dataset for SHAP
            num_samples: Number of background samples

        Returns:
            Dictionary with SHAP values and metadata
        """
        try:
            import shap

            # Prepare background data
            if background_data is None:
                # Use zeros as background if not provided
                background = np.zeros((num_samples, *signal.shape))
            else:
                background = background_data[:num_samples]

            # Create SHAP explainer
            explainer = shap.DeepExplainer(model, torch.FloatTensor(background))

            # Get SHAP values
            shap_values = explainer.shap_values(torch.FloatTensor(signal.reshape(1, -1)))

            return {
                "method": "shap",
                "shap_values": shap_values[0] if isinstance(shap_values, list) else shap_values,
                "base_value": explainer.expected_value,
                "signal": signal,
            }

        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}", exc_info=True)
            return {"error": str(e)}

    @staticmethod
    def generate_lime_explanation(model, signal, num_features=20, num_samples=1000):
        """
        Generate LIME explanation for a signal.

        Args:
            model: Trained PyTorch model
            signal: Input signal to explain
            num_features: Number of top features to show
            num_samples: Number of perturbations

        Returns:
            Dictionary with LIME explanation
        """
        try:
            from lime import lime_tabular

            # Create LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                training_data=np.zeros((100, len(signal))),  # Dummy background
                mode='classification',
                feature_names=[f"t_{i}" for i in range(len(signal))],
            )

            # Prediction function
            def predict_fn(x):
                with torch.no_grad():
                    outputs = model(torch.FloatTensor(x))
                    return torch.softmax(outputs, dim=1).numpy()

            # Generate explanation
            explanation = explainer.explain_instance(
                signal,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples
            )

            # Extract feature importance
            feature_importance = dict(explanation.as_list())

            return {
                "method": "lime",
                "feature_importance": feature_importance,
                "signal": signal,
                "num_features": num_features,
            }

        except Exception as e:
            logger.error(f"LIME explanation failed: {e}", exc_info=True)
            return {"error": str(e)}

    @staticmethod
    def generate_integrated_gradients(model, signal, baseline=None, steps=50):
        """
        Generate Integrated Gradients explanation.

        Args:
            model: Trained PyTorch model
            signal: Input signal to explain
            baseline: Baseline signal (default: zeros)
            steps: Number of integration steps

        Returns:
            Dictionary with attributions
        """
        try:
            from captum.attr import IntegratedGradients

            model.eval()

            # Prepare input
            input_tensor = torch.FloatTensor(signal).unsqueeze(0).requires_grad_(True)

            # Baseline
            if baseline is None:
                baseline = torch.zeros_like(input_tensor)

            # Create IG explainer
            ig = IntegratedGradients(model)

            # Get target class
            with torch.no_grad():
                output = model(input_tensor)
                target_class = output.argmax(dim=1).item()

            # Generate attributions
            attributions = ig.attribute(input_tensor, baseline, target=target_class, n_steps=steps)

            return {
                "method": "integrated_gradients",
                "attributions": attributions.squeeze().detach().numpy(),
                "target_class": target_class,
                "signal": signal,
            }

        except Exception as e:
            logger.error(f"Integrated Gradients explanation failed: {e}", exc_info=True)
            return {"error": str(e)}

    @staticmethod
    def generate_gradcam(model, signal, target_layer=None):
        """
        Generate Grad-CAM explanation for CNN models.

        Args:
            model: Trained CNN model
            signal: Input signal
            target_layer: Layer to use for Grad-CAM

        Returns:
            Dictionary with Grad-CAM heatmap
        """
        try:
            from captum.attr import LayerGradCam

            model.eval()

            # Prepare input
            input_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

            # Find target layer if not specified
            if target_layer is None:
                # Use last convolutional layer
                for name, module in reversed(list(model.named_modules())):
                    if isinstance(module, torch.nn.Conv1d):
                        target_layer = module
                        break

            if target_layer is None:
                return {"error": "No convolutional layer found in model"}

            # Create Grad-CAM explainer
            grad_cam = LayerGradCam(model, target_layer)

            # Get target class
            with torch.no_grad():
                output = model(input_tensor)
                target_class = output.argmax(dim=1).item()

            # Generate Grad-CAM
            attributions = grad_cam.attribute(input_tensor, target=target_class)

            return {
                "method": "gradcam",
                "attributions": attributions.squeeze().detach().numpy(),
                "target_class": target_class,
                "signal": signal,
            }

        except Exception as e:
            logger.error(f"Grad-CAM explanation failed: {e}", exc_info=True)
            return {"error": str(e)}

    @staticmethod
    def load_model(experiment_id):
        """Load trained model from experiment."""
        from database.connection import get_db_session
        from models.experiment import Experiment
        import torch
from utils.constants import NUM_CLASSES, SIGNAL_LENGTH, SAMPLING_RATE

        try:
            with get_db_session() as session:
                experiment = session.query(Experiment).filter_by(id=experiment_id).first()
                if not experiment or not experiment.model_path:
                    return None

                # Load model
                model = torch.load(experiment.model_path)
                model.eval()
                return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None
