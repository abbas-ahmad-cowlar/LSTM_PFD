"""
Deployment tasks for Phase 11.
Celery tasks for model deployment operations.
"""
from tasks import celery_app
from utils.logger import setup_logger
from services.deployment_service import DeploymentService
from pathlib import Path
import traceback

logger = setup_logger(__name__)


@celery_app.task(bind=True)
def quantize_model_task(self, experiment_id: int, quantization_type: str = "dynamic"):
    """
    Celery task for model quantization.

    Args:
        experiment_id: Experiment ID
        quantization_type: Type of quantization (dynamic, static, fp16)

    Returns:
        Quantization results dictionary
    """
    task_id = self.request.id
    logger.info(f"Starting quantization task {task_id} for experiment {experiment_id}")

    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={
            'progress': 0.1,
            'status': 'Loading model...'
        })

        # Load model
        model = DeploymentService.load_model(experiment_id)
        if model is None:
            raise ValueError(f"Failed to load model for experiment {experiment_id}")

        # Get original model size
        model_path = DeploymentService.get_model_path(experiment_id)
        original_size_mb = DeploymentService.get_model_size(model_path)

        self.update_state(state='PROGRESS', meta={
            'progress': 0.3,
            'status': f'Applying {quantization_type} quantization...'
        })

        # Apply quantization
        if quantization_type == "dynamic":
            quantized_model = DeploymentService.quantize_model_dynamic(model)
        elif quantization_type == "fp16":
            quantized_model = DeploymentService.convert_to_fp16(model)
        elif quantization_type == "static":
            # For static quantization, we need calibration data
            # This is simplified - in production, use actual calibration data
            import torch
            calibration_data = torch.randn(10, 1, 102400)
            quantized_model = DeploymentService.quantize_model_static(model, calibration_data)
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")

        self.update_state(state='PROGRESS', meta={
            'progress': 0.7,
            'status': 'Saving quantized model...'
        })

        # Save quantized model
        save_path = Path(f"checkpoints/experiment_{experiment_id}_quantized_{quantization_type}.pth")
        DeploymentService.save_model(
            quantized_model,
            save_path,
            metadata={
                'quantization_type': quantization_type,
                'original_experiment_id': experiment_id
            }
        )

        # Get quantized model size
        quantized_size_mb = DeploymentService.get_model_size(save_path)
        compression_ratio = original_size_mb / quantized_size_mb if quantized_size_mb > 0 else 0

        self.update_state(state='PROGRESS', meta={
            'progress': 0.9,
            'status': 'Benchmarking quantized model...'
        })

        # Benchmark (quick benchmark with fewer runs)
        benchmark_results = DeploymentService.benchmark_model(quantized_model, num_runs=50)

        results = {
            "success": True,
            "quantization_type": quantization_type,
            "original_size_mb": round(original_size_mb, 2),
            "quantized_size_mb": round(quantized_size_mb, 2),
            "compression_ratio": round(compression_ratio, 2),
            "size_reduction_percent": round((1 - quantized_size_mb / original_size_mb) * 100, 1) if original_size_mb > 0 else 0,
            "save_path": str(save_path),
            "benchmark": benchmark_results
        }

        logger.info(f"Quantization task {task_id} completed successfully")
        logger.info(f"Size reduction: {results['size_reduction_percent']}%")

        return results

    except Exception as e:
        logger.error(f"Quantization task {task_id} failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@celery_app.task(bind=True)
def export_onnx_task(self, experiment_id: int, opset_version: int = 14, optimize: bool = True, dynamic_axes: bool = True):
    """
    Celery task for ONNX export.

    Args:
        experiment_id: Experiment ID
        opset_version: ONNX opset version
        optimize: Apply ONNX optimization
        dynamic_axes: Use dynamic axes

    Returns:
        Export results dictionary
    """
    task_id = self.request.id
    logger.info(f"Starting ONNX export task {task_id} for experiment {experiment_id}")

    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={
            'progress': 0.1,
            'status': 'Loading model...'
        })

        # Load model
        model = DeploymentService.load_model(experiment_id)
        if model is None:
            raise ValueError(f"Failed to load model for experiment {experiment_id}")

        # Get original model size
        model_path = DeploymentService.get_model_path(experiment_id)
        original_size_mb = DeploymentService.get_model_size(model_path)

        self.update_state(state='PROGRESS', meta={
            'progress': 0.3,
            'status': 'Exporting to ONNX...'
        })

        # Export to ONNX
        save_path = Path(f"checkpoints/experiment_{experiment_id}.onnx")
        success = DeploymentService.export_to_onnx(
            model,
            save_path,
            input_shape=(1, 1, 102400),
            opset_version=opset_version,
            optimize=optimize,
            dynamic_axes=dynamic_axes
        )

        if not success:
            raise RuntimeError("ONNX export failed")

        # Get ONNX model size
        onnx_size_mb = DeploymentService.get_model_size(save_path)

        results = {
            "success": True,
            "save_path": str(save_path),
            "original_size_mb": round(original_size_mb, 2),
            "onnx_size_mb": round(onnx_size_mb, 2),
            "opset_version": opset_version,
            "optimized": optimize,
            "dynamic_axes": dynamic_axes
        }

        logger.info(f"ONNX export task {task_id} completed successfully")
        logger.info(f"ONNX model saved to: {save_path}")

        return results

    except Exception as e:
        logger.error(f"ONNX export task {task_id} failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@celery_app.task(bind=True)
def optimize_model_task(self, experiment_id: int, pruning_method: str = "l1_unstructured", pruning_amount: float = 0.3, apply_fusion: bool = True):
    """
    Celery task for model optimization.

    Args:
        experiment_id: Experiment ID
        pruning_method: Pruning method
        pruning_amount: Fraction of parameters to prune
        apply_fusion: Apply layer fusion

    Returns:
        Optimization results dictionary
    """
    task_id = self.request.id
    logger.info(f"Starting optimization task {task_id} for experiment {experiment_id}")

    try:
        # Update task state
        self.update_state(state='PROGRESS', meta={
            'progress': 0.1,
            'status': 'Loading model...'
        })

        # Load model
        model = DeploymentService.load_model(experiment_id)
        if model is None:
            raise ValueError(f"Failed to load model for experiment {experiment_id}")

        # Get original model size
        model_path = DeploymentService.get_model_path(experiment_id)
        original_size_mb = DeploymentService.get_model_size(model_path)

        self.update_state(state='PROGRESS', meta={
            'progress': 0.3,
            'status': f'Applying {pruning_method} pruning...'
        })

        # Apply pruning if requested
        if pruning_method != "none":
            model = DeploymentService.prune_model(model, amount=pruning_amount, method=pruning_method)

        self.update_state(state='PROGRESS', meta={
            'progress': 0.6,
            'status': 'Saving optimized model...'
        })

        # Save optimized model
        save_path = Path(f"checkpoints/experiment_{experiment_id}_optimized.pth")
        DeploymentService.save_model(
            model,
            save_path,
            metadata={
                'pruning_method': pruning_method,
                'pruning_amount': pruning_amount,
                'fusion': apply_fusion,
                'original_experiment_id': experiment_id
            }
        )

        # Get optimized model size
        optimized_size_mb = DeploymentService.get_model_size(save_path)

        self.update_state(state='PROGRESS', meta={
            'progress': 0.9,
            'status': 'Benchmarking optimized model...'
        })

        # Benchmark
        benchmark_results = DeploymentService.benchmark_model(model, num_runs=50)

        results = {
            "success": True,
            "pruning_method": pruning_method,
            "pruning_amount": pruning_amount,
            "original_size_mb": round(original_size_mb, 2),
            "optimized_size_mb": round(optimized_size_mb, 2),
            "size_reduction_percent": round((1 - optimized_size_mb / original_size_mb) * 100, 1) if original_size_mb > 0 else 0,
            "save_path": str(save_path),
            "benchmark": benchmark_results
        }

        logger.info(f"Optimization task {task_id} completed successfully")

        return results

    except Exception as e:
        logger.error(f"Optimization task {task_id} failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@celery_app.task(bind=True)
def benchmark_models_task(self, experiment_id: int, num_runs: int = 100, model_types: list = None):
    """
    Celery task for benchmarking multiple model variants.

    Args:
        experiment_id: Experiment ID
        num_runs: Number of benchmark runs
        model_types: List of model types to benchmark (original, quantized, onnx, optimized)

    Returns:
        Benchmark results dictionary
    """
    task_id = self.request.id
    logger.info(f"Starting benchmark task {task_id} for experiment {experiment_id}")

    if model_types is None:
        model_types = ["original"]

    try:
        results = {}

        # Benchmark original model
        if "original" in model_types:
            self.update_state(state='PROGRESS', meta={
                'progress': 0.2,
                'status': 'Benchmarking original model...'
            })

            model = DeploymentService.load_model(experiment_id)
            if model:
                model_path = DeploymentService.get_model_path(experiment_id)
                results["original"] = {
                    "size_mb": round(DeploymentService.get_model_size(model_path), 2),
                    "benchmark": DeploymentService.benchmark_model(model, num_runs=num_runs)
                }

        # Benchmark quantized model
        if "quantized" in model_types:
            self.update_state(state='PROGRESS', meta={
                'progress': 0.5,
                'status': 'Benchmarking quantized model...'
            })

            # Load quantized model if exists
            quant_path = Path(f"checkpoints/experiment_{experiment_id}_quantized_dynamic.pth")
            if quant_path.exists():
                model = DeploymentService.load_model(experiment_id)  # Load and quantize
                model = DeploymentService.quantize_model_dynamic(model)
                results["quantized"] = {
                    "size_mb": round(DeploymentService.get_model_size(quant_path), 2),
                    "benchmark": DeploymentService.benchmark_model(model, num_runs=num_runs)
                }

        logger.info(f"Benchmark task {task_id} completed successfully")

        return {
            "success": True,
            "results": results,
            "num_runs": num_runs
        }

    except Exception as e:
        logger.error(f"Benchmark task {task_id} failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
