"""
Deployment Service (Phase 11).
Business logic for model deployment operations (quantization, ONNX export, optimization).
"""
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from pathlib import Path
from utils.logger import setup_logger
from database.connection import get_db_session
from models.experiment import Experiment
import os
from utils.constants import (
    BYTES_PER_MB,
    DEFAULT_ONNX_INPUT_SHAPE,
    DEFAULT_ONNX_OPSET_VERSION,
    DEFAULT_BENCHMARK_RUNS,
    BENCHMARK_WARMUP_RUNS,
    MILLISECONDS_PER_SECOND,
    DEFAULT_PRUNING_AMOUNT,
)

logger = setup_logger(__name__)


class DeploymentService:
    """Service for model deployment operations."""

    @staticmethod
    def get_model_path(experiment_id: int) -> Optional[Path]:
        """
        Get model checkpoint path for an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Path to model checkpoint or None
        """
        try:
            with get_db_session() as session:
                experiment = session.query(Experiment).filter_by(id=experiment_id).first()
                if not experiment:
                    logger.error(f"Experiment {experiment_id} not found")
                    return None

                # Get model path from config
                config = experiment.config or {}
                model_path = config.get('model_path')

                if not model_path:
                    # Try default path
                    model_path = f"checkpoints/experiment_{experiment_id}_best.pth"

                model_path = Path(model_path)

                if not model_path.exists():
                    logger.error(f"Model checkpoint not found: {model_path}")
                    return None

                return model_path

        except Exception as e:
            logger.error(f"Failed to get model path: {e}", exc_info=True)
            return None

    @staticmethod
    def load_model(experiment_id: int) -> Optional[nn.Module]:
        """
        Load model from experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Loaded PyTorch model or None
        """
        try:
            model_path = DeploymentService.get_model_path(experiment_id)
            if not model_path:
                return None

            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')

            # Get model architecture
            with get_db_session() as session:
                experiment = session.query(Experiment).filter_by(id=experiment_id).first()
                if not experiment:
                    return None

                model_type = experiment.model_type

            # Recreate model architecture
            from integrations.deep_learning_adapter import DeepLearningAdapter
            model = DeepLearningAdapter.create_model(model_type, checkpoint.get('config', {}))

            # Load weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.eval()
            logger.info(f"Loaded model from {model_path}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            return None

    @staticmethod
    def get_model_size(model_path: Path) -> float:
        """
        Get model file size in MB.

        Args:
            model_path: Path to model file

        Returns:
            File size in MB
        """
        try:
            size_bytes = model_path.stat().st_size
            size_mb = size_bytes / BYTES_PER_MB
            return size_mb
        except Exception as e:
            logger.error(f"Failed to get model size: {e}")
            return 0.0

    @staticmethod
    def save_model(model: nn.Module, save_path: Path, metadata: Optional[Dict] = None):
        """
        Save model with metadata.

        Args:
            model: PyTorch model
            save_path: Path to save model
            metadata: Optional metadata dictionary
        """
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                'model_state_dict': model.state_dict(),
                'metadata': metadata or {}
            }

            torch.save(checkpoint, save_path)
            logger.info(f"Saved model to {save_path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}", exc_info=True)
            raise

    @staticmethod
    def quantize_model_dynamic(model: nn.Module) -> nn.Module:
        """
        Apply dynamic INT8 quantization.

        Args:
            model: PyTorch model

        Returns:
            Quantized model
        """
        try:
            logger.info("Applying dynamic INT8 quantization...")

            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM, nn.GRU, nn.Conv1d},
                dtype=torch.qint8
            )

            logger.info("Dynamic quantization complete")
            return quantized_model

        except Exception as e:
            logger.error(f"Quantization failed: {e}", exc_info=True)
            raise

    @staticmethod
    def quantize_model_static(model: nn.Module, calibration_data: torch.Tensor) -> nn.Module:
        """
        Apply static INT8 quantization with calibration.

        Args:
            model: PyTorch model
            calibration_data: Calibration dataset

        Returns:
            Quantized model
        """
        try:
            logger.info("Applying static INT8 quantization...")

            # Set quantization config
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

            # Prepare model for quantization
            torch.quantization.prepare(model, inplace=True)

            # Calibrate with sample data
            logger.info("Calibrating model...")
            with torch.no_grad():
                model(calibration_data)

            # Convert to quantized model
            torch.quantization.convert(model, inplace=True)

            logger.info("Static quantization complete")
            return model

        except Exception as e:
            logger.error(f"Static quantization failed: {e}", exc_info=True)
            raise

    @staticmethod
    def convert_to_fp16(model: nn.Module) -> nn.Module:
        """
        Convert model to FP16 (half precision).

        Args:
            model: PyTorch model

        Returns:
            FP16 model
        """
        try:
            logger.info("Converting to FP16...")
            model_fp16 = model.half()
            logger.info("FP16 conversion complete")
            return model_fp16

        except Exception as e:
            logger.error(f"FP16 conversion failed: {e}", exc_info=True)
            raise

    @staticmethod
    def export_to_onnx(
        model: nn.Module,
        save_path: Path,
        input_shape: tuple = DEFAULT_ONNX_INPUT_SHAPE,
        opset_version: int = DEFAULT_ONNX_OPSET_VERSION,
        optimize: bool = True,
        dynamic_axes: bool = True
    ) -> bool:
        """
        Export model to ONNX format.

        Args:
            model: PyTorch model
            save_path: Path to save ONNX model
            input_shape: Input tensor shape
            opset_version: ONNX opset version
            optimize: Apply ONNX optimization
            dynamic_axes: Use dynamic axes for batch dimension

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Exporting model to ONNX (opset {opset_version})...")

            # Create dummy input
            dummy_input = torch.randn(*input_shape)

            # Setup dynamic axes
            axes = None
            if dynamic_axes:
                axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

            # Export
            save_path.parent.mkdir(parents=True, exist_ok=True)

            torch.onnx.export(
                model,
                dummy_input,
                save_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=optimize,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=axes
            )

            # Verify ONNX model
            try:
                import onnx
                onnx_model = onnx.load(str(save_path))
                onnx.checker.check_model(onnx_model)
                logger.info("ONNX model verification passed")
            except ImportError:
                logger.warning("ONNX library not available for verification")

            logger.info(f"ONNX export complete: {save_path}")
            return True

        except Exception as e:
            logger.error(f"ONNX export failed: {e}", exc_info=True)
            return False

    @staticmethod
    def prune_model(model: nn.Module, amount: float = DEFAULT_PRUNING_AMOUNT, method: str = "l1_unstructured") -> nn.Module:
        """
        Apply pruning to model.

        Args:
            model: PyTorch model
            amount: Fraction of parameters to prune (0.0-1.0)
            method: Pruning method (l1_unstructured, random_unstructured, structured)

        Returns:
            Pruned model
        """
        try:
            import torch.nn.utils.prune as prune

            logger.info(f"Applying {method} pruning (amount={amount})...")

            # Apply pruning to all Conv1d and Linear layers
            for name, module in model.named_modules():
                if isinstance(module, (nn.Conv1d, nn.Linear)):
                    if method == "l1_unstructured":
                        prune.l1_unstructured(module, name='weight', amount=amount)
                    elif method == "random_unstructured":
                        prune.random_unstructured(module, name='weight', amount=amount)
                    elif method == "structured":
                        if isinstance(module, nn.Conv1d):
                            prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)

                    # Make pruning permanent
                    prune.remove(module, 'weight')

            logger.info("Pruning complete")
            return model

        except Exception as e:
            logger.error(f"Pruning failed: {e}", exc_info=True)
            raise

    @staticmethod
    def benchmark_model(model: nn.Module, input_shape: tuple = DEFAULT_ONNX_INPUT_SHAPE, num_runs: int = DEFAULT_BENCHMARK_RUNS) -> Dict[str, Any]:
        """
        Benchmark model inference speed.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape
            num_runs: Number of benchmark runs

        Returns:
            Benchmark results dictionary
        """
        try:
            import time

            logger.info(f"Benchmarking model ({num_runs} runs)...")

            model.eval()
            device = next(model.parameters()).device

            # Warmup
            dummy_input = torch.randn(*input_shape).to(device)
            with torch.no_grad():
                for _ in range(BENCHMARK_WARMUP_RUNS):
                    _ = model(dummy_input)

            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start = time.time()
                    _ = model(dummy_input)
                    end = time.time()
                    times.append((end - start) * MILLISECONDS_PER_SECOND)  # Convert to ms

            import statistics

            results = {
                "mean_ms": statistics.mean(times),
                "median_ms": statistics.median(times),
                "std_ms": statistics.stdev(times) if len(times) > 1 else 0,
                "min_ms": min(times),
                "max_ms": max(times),
                "num_runs": num_runs
            }

            logger.info(f"Benchmark complete: {results['mean_ms']:.2f} ms/inference")
            return results

        except Exception as e:
            logger.error(f"Benchmarking failed: {e}", exc_info=True)
            return {}
