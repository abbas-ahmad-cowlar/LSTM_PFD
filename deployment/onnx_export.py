"""
ONNX Export Utilities

Tools for exporting PyTorch models to ONNX format for cross-platform deployment:
- Model export to ONNX
- ONNX validation and testing
- ONNX optimization
- Runtime inference with ONNX Runtime

Author: LSTM_PFD Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ONNXExportConfig:
    """Configuration for ONNX export."""
    opset_version: int = 14  # ONNX opset version
    do_constant_folding: bool = True  # Optimize constant operations
    dynamic_axes: Optional[Dict] = None  # Dynamic batch size
    input_names: List[str] = None  # Input tensor names
    output_names: List[str] = None  # Output tensor names
    export_params: bool = True  # Export trained parameters
    verbose: bool = False  # Print export info


def export_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    save_path: str,
    config: Optional[ONNXExportConfig] = None,
    **kwargs
) -> Path:
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        dummy_input: Example input tensor (used to trace model)
        save_path: Path to save ONNX model
        config: Export configuration
        **kwargs: Additional arguments for torch.onnx.export

    Returns:
        Path to saved ONNX model

    Example:
        >>> model = load_pretrained('checkpoints/best_model.pth')
        >>> dummy_input = torch.randn(1, 1, SIGNAL_LENGTH)
        >>> onnx_path = export_to_onnx(model, dummy_input, 'models/model.onnx')
    """
    if config is None:
        config = ONNXExportConfig()

    model.eval()

    # Default input/output names
    if config.input_names is None:
        config.input_names = ['input']
    if config.output_names is None:
        config.output_names = ['output']

    # Default dynamic axes (batch size)
    if config.dynamic_axes is None:
        config.dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }

    logger.info(f"Exporting model to ONNX (opset version {config.opset_version})...")
    logger.info(f"Input shape: {dummy_input.shape}")

    try:
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            opset_version=config.opset_version,
            do_constant_folding=config.do_constant_folding,
            input_names=config.input_names,
            output_names=config.output_names,
            dynamic_axes=config.dynamic_axes,
            export_params=config.export_params,
            verbose=config.verbose,
            **kwargs
        )

        save_path = Path(save_path)
        size_mb = save_path.stat().st_size / (1024 * 1024)

        logger.info(f"✓ Model exported to {save_path}")
        logger.info(f"✓ ONNX model size: {size_mb:.2f} MB")

        return save_path

    except Exception as e:
        logger.error(f"Failed to export model to ONNX: {e}")
        raise


def validate_onnx_export(
    onnx_path: str,
    pytorch_model: nn.Module,
    test_input: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5
) -> bool:
    """
    Validate ONNX export by comparing outputs with PyTorch model.

    Args:
        onnx_path: Path to ONNX model
        pytorch_model: Original PyTorch model
        test_input: Test input tensor
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if validation passes

    Example:
        >>> is_valid = validate_onnx_export(
        ...     'models/model.onnx',
        ...     pytorch_model,
        ...     torch.randn(1, 1, SIGNAL_LENGTH)
        ... )
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        logger.error("Please install onnx and onnxruntime: pip install onnx onnxruntime")
        return False

    logger.info("Validating ONNX export...")

    # Check ONNX model validity
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info("✓ ONNX model structure is valid")
    except Exception as e:
        logger.error(f"✗ ONNX model validation failed: {e}")
        return False

    # Compare outputs
    try:
        # PyTorch inference
        pytorch_model.eval()
        with torch.no_grad():
            pytorch_output = pytorch_model(test_input).cpu().numpy()

        # ONNX Runtime inference
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        onnx_output = ort_session.run(None, {input_name: test_input.cpu().numpy()})[0]

        # Compare outputs
        if np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol):
            max_diff = np.abs(pytorch_output - onnx_output).max()
            logger.info(f"✓ Outputs match (max diff: {max_diff:.2e})")
            return True
        else:
            max_diff = np.abs(pytorch_output - onnx_output).max()
            logger.error(f"✗ Outputs differ (max diff: {max_diff:.2e})")
            return False

    except Exception as e:
        logger.error(f"✗ Output comparison failed: {e}")
        return False


def optimize_onnx_model(
    onnx_path: str,
    output_path: Optional[str] = None,
    optimization_level: str = 'basic'
) -> Path:
    """
    Optimize ONNX model for faster inference.

    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save optimized model (default: overwrite)
        optimization_level: 'basic', 'extended', or 'all'

    Returns:
        Path to optimized model

    Example:
        >>> optimized_path = optimize_onnx_model(
        ...     'models/model.onnx',
        ...     'models/model_optimized.onnx',
        ...     optimization_level='all'
        ... )
    """
    try:
        import onnx
        from onnxruntime.transformers import optimizer
    except ImportError:
        logger.error("Please install onnx and onnxruntime: pip install onnx onnxruntime")
        raise

    if output_path is None:
        output_path = onnx_path

    logger.info(f"Optimizing ONNX model (level: {optimization_level})...")

    # Load model
    onnx_model = onnx.load(onnx_path)

    # Apply optimizations
    if optimization_level == 'basic':
        # Basic graph optimizations
        from onnx import optimizer as onnx_optimizer
        passes = [
            'eliminate_nop_transpose',
            'eliminate_identity',
            'fuse_consecutive_transposes',
            'fuse_transpose_into_gemm',
        ]
        optimized_model = onnx_optimizer.optimize(onnx_model, passes)

    elif optimization_level == 'extended':
        # Extended optimizations
        from onnx import optimizer as onnx_optimizer
        passes = [
            'eliminate_nop_transpose',
            'eliminate_identity',
            'fuse_consecutive_transposes',
            'fuse_transpose_into_gemm',
            'fuse_add_bias_into_conv',
            'fuse_bn_into_conv',
            'eliminate_unused_initializer',
        ]
        optimized_model = onnx_optimizer.optimize(onnx_model, passes)

    else:  # 'all'
        # All available optimizations
        from onnx import optimizer as onnx_optimizer
        optimized_model = onnx_optimizer.optimize(onnx_model)

    # Save optimized model
    onnx.save(optimized_model, output_path)

    original_size = Path(onnx_path).stat().st_size / (1024 * 1024)
    optimized_size = Path(output_path).stat().st_size / (1024 * 1024)
    reduction = (1 - optimized_size / original_size) * 100

    logger.info(f"✓ Optimized model saved to {output_path}")
    logger.info(f"✓ Original size: {original_size:.2f} MB")
    logger.info(f"✓ Optimized size: {optimized_size:.2f} MB")
    logger.info(f"✓ Size reduction: {reduction:.1f}%")

    return Path(output_path)


class ONNXInferenceSession:
    """
    Wrapper for ONNX Runtime inference session.

    Provides easy-to-use interface for ONNX model inference.

    Example:
        >>> session = ONNXInferenceSession('models/model.onnx')
        >>> output = session.predict(input_data)
    """

    def __init__(
        self,
        onnx_path: str,
        providers: Optional[List[str]] = None
    ):
        """
        Initialize ONNX inference session.

        Args:
            onnx_path: Path to ONNX model
            providers: Execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        try:
            import onnxruntime as ort
        except ImportError:
            logger.error("Please install onnxruntime: pip install onnxruntime")
            raise

        if providers is None:
            # Auto-detect available providers
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(f"ONNX session initialized with providers: {providers}")
        logger.info(f"Input: {self.input_name}, shape: {self.session.get_inputs()[0].shape}")
        logger.info(f"Output: {self.output_name}, shape: {self.session.get_outputs()[0].shape}")

    def predict(
        self,
        input_data: np.ndarray
    ) -> np.ndarray:
        """
        Run inference on input data.

        Args:
            input_data: Input array (numpy)

        Returns:
            Output array (numpy)
        """
        return self.session.run([self.output_name], {self.input_name: input_data})[0]

    def predict_batch(
        self,
        input_data: np.ndarray,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Run inference on batches of data.

        Args:
            input_data: Input array [N, ...]
            batch_size: Batch size

        Returns:
            Output array [N, num_classes]
        """
        outputs = []
        num_samples = len(input_data)

        for i in range(0, num_samples, batch_size):
            batch = input_data[i:i + batch_size]
            output = self.predict(batch)
            outputs.append(output)

        return np.concatenate(outputs, axis=0)

    def get_model_info(self) -> Dict:
        """Get model metadata."""
        inputs = self.session.get_inputs()[0]
        outputs = self.session.get_outputs()[0]

        return {
            'input_name': inputs.name,
            'input_shape': inputs.shape,
            'input_type': inputs.type,
            'output_name': outputs.name,
            'output_shape': outputs.shape,
            'output_type': outputs.type,
        }


def benchmark_onnx_inference(
    onnx_path: str,
    input_shape: Tuple[int, ...],
    num_runs: int = 100,
    warmup_runs: int = 10
) -> Dict[str, float]:
    """
    Benchmark ONNX model inference speed.

    Args:
        onnx_path: Path to ONNX model
        input_shape: Input tensor shape
        num_runs: Number of inference runs
        warmup_runs: Number of warmup runs

    Returns:
        Dictionary with timing statistics

    Example:
        >>> stats = benchmark_onnx_inference('models/model.onnx', (1, 1, SIGNAL_LENGTH))
        >>> print(f"Latency: {stats['mean_latency_ms']:.2f} ms")
    """
    import time

    session = ONNXInferenceSession(onnx_path)
    dummy_input = np.random.randn(*input_shape).astype(np.float32)

    # Warmup
    for _ in range(warmup_runs):
        _ = session.predict(dummy_input)

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.time()
        _ = session.predict(dummy_input)
        latencies.append((time.time() - start) * 1000)

    stats = {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'median_latency_ms': np.median(latencies),
        'throughput_samples_per_sec': 1000 / np.mean(latencies)
    }

    logger.info(f"ONNX Inference Benchmark Results:")
    logger.info(f"  Mean latency: {stats['mean_latency_ms']:.2f} ± {stats['std_latency_ms']:.2f} ms")
    logger.info(f"  Min/Max: {stats['min_latency_ms']:.2f} / {stats['max_latency_ms']:.2f} ms")
    logger.info(f"  Throughput: {stats['throughput_samples_per_sec']:.1f} samples/sec")

    return stats


def convert_and_quantize_onnx(
    pytorch_model: nn.Module,
    dummy_input: torch.Tensor,
    save_path: str,
    quantization_type: str = 'dynamic'
) -> Path:
    """
    Export PyTorch model to ONNX and apply quantization.

    Args:
        pytorch_model: PyTorch model
        dummy_input: Example input
        save_path: Path to save quantized ONNX model
        quantization_type: 'dynamic' or 'static'

    Returns:
        Path to quantized ONNX model

    Example:
        >>> model = load_pretrained('checkpoints/best_model.pth')
        >>> dummy_input = torch.randn(1, 1, SIGNAL_LENGTH)
        >>> quantized_path = convert_and_quantize_onnx(
        ...     model, dummy_input, 'models/model_int8.onnx'
        ... )
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        logger.error("Please install onnxruntime: pip install onnxruntime")
        raise

    # Export to ONNX first
    temp_path = save_path.replace('.onnx', '_fp32.onnx')
    export_to_onnx(pytorch_model, dummy_input, temp_path)

    # Quantize ONNX model
    logger.info(f"Quantizing ONNX model ({quantization_type})...")

    if quantization_type == 'dynamic':
        quantize_dynamic(
            model_input=temp_path,
            model_output=save_path,
            weight_type=QuantType.QUInt8
        )
    else:
        logger.warning("Static quantization not yet implemented for ONNX")
        return Path(temp_path)

    # Clean up temporary file
    Path(temp_path).unlink()

    original_size = Path(temp_path).stat().st_size / (1024 * 1024) if Path(temp_path).exists() else 0
    quantized_size = Path(save_path).stat().st_size / (1024 * 1024)

    logger.info(f"✓ Quantized ONNX model saved to {save_path}")
    logger.info(f"✓ Model size: {quantized_size:.2f} MB")

    return Path(save_path)
