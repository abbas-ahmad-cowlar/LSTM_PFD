"""
XAI Service (Phase 11C) - Enhanced Integration.
Integrates dashboard with Phase 7 explainability implementations.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.logger import setup_logger

logger = setup_logger(__name__)


class XAIService:
    """Enhanced service for generating model explanations using Phase 7 implementations."""

    @staticmethod
    def get_device() -> str:
        """Get available device (GPU if available, else CPU)."""
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def generate_shap_explanation(
        model: nn.Module,
        signal: torch.Tensor,
        background_data: Optional[torch.Tensor] = None,
        method: str = 'gradient',
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation using robust Phase 7 SHAPExplainer.

        Args:
            model: Trained PyTorch model
            signal: Input signal to explain [1, C, T] or [C, T]
            background_data: Background dataset for SHAP [N, C, T]
            method: SHAP method ('gradient', 'deep', 'kernel')
            num_samples: Number of samples for approximation

        Returns:
            Dictionary with SHAP values and metadata
        """
        try:
            from explainability.shap_explainer import SHAPExplainer

            device = XAIService.get_device()
            model = model.to(device).eval()

            # Ensure signal has correct dimensions
            if signal.dim() == 1:
                signal = signal.unsqueeze(0).unsqueeze(0)  # [T] -> [1, 1, T]
            elif signal.dim() == 2:
                signal = signal.unsqueeze(0)  # [C, T] -> [1, C, T]

            signal = signal.to(device)

            # Prepare background data
            if background_data is not None:
                if background_data.dim() == 2:
                    background_data = background_data.unsqueeze(1)  # [N, T] -> [N, 1, T]
                background_data = background_data.to(device)

            # Create SHAP explainer
            explainer = SHAPExplainer(
                model=model,
                background_data=background_data,
                device=device,
                use_shap_library=True  # Try to use official SHAP library
            )

            logger.info(f"Generating {method.upper()} SHAP explanation with {num_samples} samples")

            # Generate SHAP values
            shap_values = explainer.explain(
                input_signal=signal,
                method=method,
                n_samples=num_samples
            )

            # Get prediction
            with torch.no_grad():
                output = model(signal)
                probabilities = torch.softmax(output, dim=1)[0]
                predicted_class = output.argmax(dim=1).item()
                confidence = probabilities[predicted_class].item()

            # Compute base value (expected value from background)
            if background_data is not None:
                with torch.no_grad():
                    bg_outputs = model(background_data[:min(100, len(background_data))])
                    base_value = torch.softmax(bg_outputs, dim=1)[:, predicted_class].mean().item()
            else:
                base_value = 1.0 / output.shape[1]  # Uniform prior

            # Convert to list format for JSON serialization
            shap_values_np = shap_values.squeeze().cpu().numpy()
            signal_np = signal.squeeze().cpu().numpy()

            # Generate time labels
            signal_length = signal_np.shape[-1] if signal_np.ndim > 0 else len(signal_np)
            time_labels = [f"t_{i}" for i in range(signal_length)]

            return {
                "success": True,
                "method": "shap",
                "shap_method": method,
                "shap_values": shap_values_np.tolist(),
                "signal": signal_np.tolist(),
                "base_value": base_value,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities.cpu().numpy().tolist(),
                "time_labels": time_labels,
                "signal_length": signal_length,
            }

        except ImportError as e:
            logger.error(f"SHAP dependencies not installed: {e}")
            return {
                "success": False,
                "error": "SHAP library not installed. Run: pip install shap"
            }
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    def generate_lime_explanation(
        model: nn.Module,
        signal: torch.Tensor,
        num_segments: int = 20,
        num_samples: int = 1000,
        target_class: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate LIME explanation using robust Phase 7 LIMEExplainer.

        Args:
            model: Trained PyTorch model
            signal: Input signal to explain [1, C, T] or [C, T]
            num_segments: Number of segments to divide signal into
            num_samples: Number of perturbation samples
            target_class: Target class to explain (if None, uses predicted)

        Returns:
            Dictionary with LIME explanation
        """
        try:
            from explainability.lime_explainer import LIMEExplainer

            device = XAIService.get_device()
            model = model.to(device).eval()

            # Ensure signal has correct dimensions
            if signal.dim() == 1:
                signal = signal.unsqueeze(0).unsqueeze(0)
            elif signal.dim() == 2:
                signal = signal.unsqueeze(0)

            signal = signal.to(device)

            # Create LIME explainer
            explainer = LIMEExplainer(
                model=model,
                device=device,
                num_segments=num_segments,
                kernel_width=0.25
            )

            logger.info(f"Generating LIME explanation with {num_segments} segments, {num_samples} samples")

            # Generate explanation
            segment_weights, segment_boundaries = explainer.explain(
                input_signal=signal,
                target_class=target_class,
                num_samples=num_samples,
                distance_metric='cosine'
            )

            # Get prediction
            with torch.no_grad():
                output = model(signal)
                probabilities = torch.softmax(output, dim=1)[0]
                predicted_class = output.argmax(dim=1).item()
                confidence = probabilities[predicted_class].item()

            # Convert to serializable format
            signal_np = signal.squeeze().cpu().numpy()
            segment_weights_list = segment_weights.tolist()
            segment_boundaries_list = [(int(start), int(end)) for start, end in segment_boundaries]

            return {
                "success": True,
                "method": "lime",
                "segment_weights": segment_weights_list,
                "segment_boundaries": segment_boundaries_list,
                "signal": signal_np.tolist(),
                "num_segments": num_segments,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities.cpu().numpy().tolist(),
                "signal_length": len(signal_np) if signal_np.ndim == 1 else signal_np.shape[-1],
            }

        except Exception as e:
            logger.error(f"LIME explanation failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    def generate_integrated_gradients(
        model: nn.Module,
        signal: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> Dict[str, Any]:
        """
        Generate Integrated Gradients explanation using Captum.

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

            device = XAIService.get_device()
            model = model.to(device).eval()

            # Prepare input
            if signal.dim() == 1:
                signal = signal.unsqueeze(0).unsqueeze(0)
            elif signal.dim() == 2:
                signal = signal.unsqueeze(0)

            input_tensor = signal.to(device).requires_grad_(True)

            # Baseline
            if baseline is None:
                baseline = torch.zeros_like(input_tensor)
            else:
                baseline = baseline.to(device)

            # Create IG explainer
            ig = IntegratedGradients(model)

            # Get target class
            with torch.no_grad():
                output = model(input_tensor)
                target_class = output.argmax(dim=1).item()
                probabilities = torch.softmax(output, dim=1)[0]
                confidence = probabilities[target_class].item()

            logger.info(f"Generating Integrated Gradients with {steps} steps for class {target_class}")

            # Generate attributions
            attributions = ig.attribute(input_tensor, baseline, target=target_class, n_steps=steps)

            # Convert to serializable format
            attributions_np = attributions.squeeze().detach().cpu().numpy()
            signal_np = signal.squeeze().cpu().numpy()

            return {
                "success": True,
                "method": "integrated_gradients",
                "attributions": attributions_np.tolist(),
                "signal": signal_np.tolist(),
                "target_class": target_class,
                "confidence": confidence,
                "probabilities": probabilities.cpu().numpy().tolist(),
                "steps": steps,
                "signal_length": len(signal_np) if signal_np.ndim == 1 else signal_np.shape[-1],
            }

        except ImportError:
            logger.error("Captum library not installed")
            return {
                "success": False,
                "error": "Captum library not installed. Run: pip install captum"
            }
        except Exception as e:
            logger.error(f"Integrated Gradients failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    def generate_gradcam(
        model: nn.Module,
        signal: torch.Tensor,
        target_layer: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Generate Grad-CAM explanation for CNN models using Captum.

        Args:
            model: Trained CNN model
            signal: Input signal
            target_layer: Layer to use for Grad-CAM (auto-detected if None)

        Returns:
            Dictionary with Grad-CAM heatmap
        """
        try:
            from captum.attr import LayerGradCam

            device = XAIService.get_device()
            model = model.to(device).eval()

            # Prepare input
            if signal.dim() == 1:
                signal = signal.unsqueeze(0).unsqueeze(0)
            elif signal.dim() == 2:
                signal = signal.unsqueeze(0)

            input_tensor = signal.to(device)

            # Find target layer if not specified
            if target_layer is None:
                # Use last convolutional layer
                for name, module in reversed(list(model.named_modules())):
                    if isinstance(module, torch.nn.Conv1d):
                        target_layer = module
                        logger.info(f"Using layer: {name}")
                        break

            if target_layer is None:
                return {
                    "success": False,
                    "error": "No convolutional layer found in model"
                }

            # Create Grad-CAM explainer
            grad_cam = LayerGradCam(model, target_layer)

            # Get target class
            with torch.no_grad():
                output = model(input_tensor)
                target_class = output.argmax(dim=1).item()
                probabilities = torch.softmax(output, dim=1)[0]
                confidence = probabilities[target_class].item()

            logger.info(f"Generating Grad-CAM for class {target_class}")

            # Generate Grad-CAM
            attributions = grad_cam.attribute(input_tensor, target=target_class)

            # Upsample attributions to match signal length if needed
            signal_np = signal.squeeze().cpu().numpy()
            attributions_np = attributions.squeeze().detach().cpu().numpy()

            # Interpolate attributions to signal length
            if len(attributions_np) != len(signal_np):
                from scipy.interpolate import interp1d
                x_attr = np.linspace(0, 1, len(attributions_np))
                x_signal = np.linspace(0, 1, len(signal_np))
                f = interp1d(x_attr, attributions_np, kind='linear', fill_value='extrapolate')
                attributions_np = f(x_signal)

            return {
                "success": True,
                "method": "gradcam",
                "attributions": attributions_np.tolist(),
                "signal": signal_np.tolist(),
                "target_class": target_class,
                "confidence": confidence,
                "probabilities": probabilities.cpu().numpy().tolist(),
                "signal_length": len(signal_np),
            }

        except ImportError:
            logger.error("Captum library not installed")
            return {
                "success": False,
                "error": "Captum library not installed. Run: pip install captum"
            }
        except Exception as e:
            logger.error(f"Grad-CAM failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    @staticmethod
    def load_model(experiment_id: int) -> Optional[nn.Module]:
        """
        Load trained model from experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Loaded model or None if failed
        """
        from database.connection import get_db_session
        from models.experiment import Experiment
        import torch

        try:
            with get_db_session() as session:
                experiment = session.query(Experiment).filter_by(id=experiment_id).first()
                if not experiment or not experiment.model_path:
                    logger.error(f"Experiment {experiment_id} has no saved model")
                    return None

                model_path = Path(experiment.model_path)
                if not model_path.exists():
                    logger.error(f"Model file not found: {model_path}")
                    return None

                # Load model
                device = XAIService.get_device()
                model = torch.load(model_path, map_location=device)
                model.eval()

                logger.info(f"Loaded model from {model_path}")
                return model

        except Exception as e:
            logger.error(f"Failed to load model for experiment {experiment_id}: {e}", exc_info=True)
            return None

    @staticmethod
    def get_model_prediction(model: nn.Module, signal: torch.Tensor) -> Dict[str, Any]:
        """
        Get model prediction for a signal.

        Args:
            model: Trained model
            signal: Input signal

        Returns:
            Dictionary with prediction results
        """
        try:
            device = XAIService.get_device()
            model = model.to(device).eval()

            # Ensure signal has correct dimensions
            if signal.dim() == 1:
                signal = signal.unsqueeze(0).unsqueeze(0)
            elif signal.dim() == 2:
                signal = signal.unsqueeze(0)

            signal = signal.to(device)

            with torch.no_grad():
                output = model(signal)
                probabilities = torch.softmax(output, dim=1)[0]
                predicted_class = output.argmax(dim=1).item()
                confidence = probabilities[predicted_class].item()

            return {
                "success": True,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "probabilities": probabilities.cpu().numpy().tolist(),
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
