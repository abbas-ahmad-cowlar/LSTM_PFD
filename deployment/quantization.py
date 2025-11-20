"""
Model Quantization Utilities

Provides tools for quantizing PyTorch models to reduce size and improve inference speed:
- Dynamic quantization (INT8)
- Static quantization (INT8 with calibration)
- FP16 conversion
- Quantization-aware training (QAT) preparation

Author: LSTM_PFD Team
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import copy
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    quantization_type: str = 'dynamic'  # 'dynamic', 'static', 'fp16', 'qat'
    backend: str = 'fbgemm'  # 'fbgemm' (x86), 'qnnpack' (ARM)
    dtype: torch.dtype = torch.qint8  # torch.qint8, torch.quint8
    calibration_samples: int = 100  # For static quantization
    qconfig: Optional[str] = 'fbgemm'  # Predefined qconfig
    preserve_sparsity: bool = False  # Preserve pruned weights


def quantize_model_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    qconfig_spec: Optional[Dict] = None,
    inplace: bool = False
) -> nn.Module:
    """
    Apply dynamic quantization to model.

    Dynamic quantization quantizes weights ahead of time and activations on-the-fly.
    Best for LSTM, GRU, and Linear layers. No calibration data needed.

    Args:
        model: PyTorch model to quantize
        dtype: Quantization dtype (qint8 or quint8)
        qconfig_spec: Optional layer-specific quantization config
        inplace: Modify model in place (default: False)

    Returns:
        Quantized model

    Example:
        >>> model = load_pretrained('checkpoints/best_model.pth')
        >>> quantized_model = quantize_model_dynamic(model)
        >>> # Model is now 4x smaller and 2-3x faster
    """
    if not inplace:
        model = copy.deepcopy(model)

    model.eval()

    # Default: quantize Linear and LSTM layers
    if qconfig_spec is None:
        qconfig_spec = {
            nn.Linear: torch.quantization.default_dynamic_qconfig,
            nn.LSTM: torch.quantization.default_dynamic_qconfig,
            nn.GRU: torch.quantization.default_dynamic_qconfig,
        }

    logger.info(f"Applying dynamic quantization (dtype={dtype})")
    logger.info(f"Quantizing layers: {list(qconfig_spec.keys())}")

    quantized_model = torch.quantization.quantize_dynamic(
        model=model,
        qconfig_spec=qconfig_spec,
        dtype=dtype,
        inplace=True
    )

    logger.info("Dynamic quantization complete")
    return quantized_model


def quantize_model_static(
    model: nn.Module,
    calibration_loader: torch.utils.data.DataLoader,
    backend: str = 'fbgemm',
    inplace: bool = False
) -> nn.Module:
    """
    Apply static quantization to model with calibration.

    Static quantization quantizes both weights and activations ahead of time
    using calibration data. Best for CNNs and fully connected networks.

    Args:
        model: PyTorch model to quantize
        calibration_loader: DataLoader with calibration data
        backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM)
        inplace: Modify model in place

    Returns:
        Quantized model

    Example:
        >>> model = load_pretrained('checkpoints/best_model.pth')
        >>> calibration_loader = DataLoader(calibration_dataset, batch_size=32)
        >>> quantized_model = quantize_model_static(model, calibration_loader)
        >>> # Model is now 4x smaller and 3-4x faster
    """
    if not inplace:
        model = copy.deepcopy(model)

    model.eval()

    # Set quantization backend
    torch.backends.quantized.engine = backend
    logger.info(f"Using quantization backend: {backend}")

    # Attach qconfig
    if backend == 'fbgemm':
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    else:
        model.qconfig = torch.quantization.get_default_qconfig('qnnpack')

    logger.info("Preparing model for static quantization...")

    # Prepare model (insert observers)
    model_prepared = torch.quantization.prepare(model, inplace=True)

    # Calibration: run model on calibration data
    logger.info("Calibrating model with sample data...")
    with torch.no_grad():
        for i, batch in enumerate(calibration_loader):
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch

            # Run forward pass to collect statistics
            model_prepared(x)

            if i >= 99:  # Limit calibration samples
                break

    logger.info("Calibration complete")

    # Convert to quantized model
    logger.info("Converting to quantized model...")
    quantized_model = torch.quantization.convert(model_prepared, inplace=True)

    logger.info("Static quantization complete")
    return quantized_model


def quantize_to_fp16(
    model: nn.Module,
    inplace: bool = False
) -> nn.Module:
    """
    Convert model to FP16 (half precision).

    FP16 reduces model size by 2x and can be 2-3x faster on GPUs with Tensor Cores.
    Less aggressive than INT8 quantization.

    Args:
        model: PyTorch model
        inplace: Modify model in place

    Returns:
        FP16 model

    Example:
        >>> model = load_pretrained('checkpoints/best_model.pth')
        >>> fp16_model = quantize_to_fp16(model)
        >>> # Model is now 2x smaller
    """
    if not inplace:
        model = copy.deepcopy(model)

    model.eval()

    logger.info("Converting model to FP16...")
    model = model.half()

    logger.info("FP16 conversion complete")
    return model


def prepare_qat_model(
    model: nn.Module,
    backend: str = 'fbgemm'
) -> nn.Module:
    """
    Prepare model for Quantization-Aware Training (QAT).

    QAT simulates quantization during training for better accuracy.
    Use this before fine-tuning the model.

    Args:
        model: PyTorch model
        backend: Quantization backend

    Returns:
        QAT-prepared model

    Example:
        >>> model = load_pretrained('checkpoints/best_model.pth')
        >>> qat_model = prepare_qat_model(model)
        >>> # Now train qat_model normally for a few epochs
        >>> # Then convert with: quantize_qat_model(qat_model)
    """
    model.train()

    # Set quantization backend
    torch.backends.quantized.engine = backend

    # Attach qconfig for QAT
    if backend == 'fbgemm':
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    else:
        model.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')

    logger.info("Preparing model for QAT...")

    # Prepare model with fake quantization
    qat_model = torch.quantization.prepare_qat(model, inplace=False)

    logger.info("QAT preparation complete. Train this model for a few epochs.")
    return qat_model


def quantize_qat_model(
    qat_model: nn.Module,
    inplace: bool = False
) -> nn.Module:
    """
    Convert QAT model to fully quantized model.

    Call this after training the QAT model.

    Args:
        qat_model: Model prepared with prepare_qat_model and trained
        inplace: Modify model in place

    Returns:
        Quantized model

    Example:
        >>> # After QAT training
        >>> quantized_model = quantize_qat_model(qat_model)
    """
    if not inplace:
        qat_model = copy.deepcopy(qat_model)

    qat_model.eval()

    logger.info("Converting QAT model to quantized model...")
    quantized_model = torch.quantization.convert(qat_model, inplace=True)

    logger.info("QAT quantization complete")
    return quantized_model


def compare_model_sizes(
    original_model: nn.Module,
    quantized_model: nn.Module
) -> Dict[str, float]:
    """
    Compare sizes of original and quantized models.

    Args:
        original_model: Original FP32 model
        quantized_model: Quantized model

    Returns:
        Dictionary with size comparison stats

    Example:
        >>> stats = compare_model_sizes(original_model, quantized_model)
        >>> print(f"Size reduction: {stats['compression_ratio']:.2f}x")
    """
    # Save models to temporary files to measure size
    import tempfile
    import os

    # Use delete=False to avoid Windows file locking issues
    f1 = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
    f1.close()
    f2 = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
    f2.close()

    try:
        torch.save(original_model.state_dict(), f1.name)
        original_size = Path(f1.name).stat().st_size

        torch.save(quantized_model.state_dict(), f2.name)
        quantized_size = Path(f2.name).stat().st_size

        compression_ratio = original_size / quantized_size
        size_reduction_mb = (original_size - quantized_size) / (1024 * 1024)

        stats = {
            'original_size_mb': original_size / (1024 * 1024),
            'quantized_size_mb': quantized_size / (1024 * 1024),
            'compression_ratio': compression_ratio,
            'size_reduction_mb': size_reduction_mb,
            'size_reduction_percent': (1 - quantized_size / original_size) * 100
        }

        logger.info(f"Original model size: {stats['original_size_mb']:.2f} MB")
        logger.info(f"Quantized model size: {stats['quantized_size_mb']:.2f} MB")
        logger.info(f"Compression ratio: {stats['compression_ratio']:.2f}x")
        logger.info(f"Size reduction: {stats['size_reduction_percent']:.1f}%")

        return stats
    finally:
        # Clean up temporary files
        try:
            os.unlink(f1.name)
        except Exception:
            pass
        try:
            os.unlink(f2.name)
        except Exception:
            pass


def benchmark_quantized_model(
    original_model: nn.Module,
    quantized_model: nn.Module,
    input_tensor: torch.Tensor,
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Benchmark inference speed of original vs quantized model.

    Args:
        original_model: Original FP32 model
        quantized_model: Quantized model
        input_tensor: Sample input tensor
        num_runs: Number of inference runs for averaging

    Returns:
        Dictionary with timing comparison

    Example:
        >>> x = torch.randn(1, 1, 102400)
        >>> stats = benchmark_quantized_model(original_model, quantized_model, x)
        >>> print(f"Speedup: {stats['speedup']:.2f}x")
    """
    import time

    original_model.eval()
    quantized_model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = original_model(input_tensor)
            _ = quantized_model(input_tensor)

    # Benchmark original model
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = original_model(input_tensor)
    original_time = (time.time() - start) / num_runs

    # Benchmark quantized model
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = quantized_model(input_tensor)
    quantized_time = (time.time() - start) / num_runs

    speedup = original_time / quantized_time

    stats = {
        'original_latency_ms': original_time * 1000,
        'quantized_latency_ms': quantized_time * 1000,
        'speedup': speedup,
        'latency_reduction_ms': (original_time - quantized_time) * 1000,
        'latency_reduction_percent': (1 - quantized_time / original_time) * 100
    }

    logger.info(f"Original model latency: {stats['original_latency_ms']:.2f} ms")
    logger.info(f"Quantized model latency: {stats['quantized_latency_ms']:.2f} ms")
    logger.info(f"Speedup: {stats['speedup']:.2f}x")

    return stats


def save_quantized_model(
    model: nn.Module,
    save_path: str,
    metadata: Optional[Dict] = None
):
    """
    Save quantized model with metadata.

    Args:
        model: Quantized model
        save_path: Path to save model
        metadata: Optional metadata dictionary

    Example:
        >>> save_quantized_model(
        ...     quantized_model,
        ...     'checkpoints/phase9/model_int8.pth',
        ...     metadata={'quantization_type': 'dynamic', 'accuracy': 0.97}
        ... )
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'quantized': True,
        'metadata': metadata or {}
    }

    torch.save(checkpoint, save_path)
    logger.info(f"Quantized model saved to {save_path}")


def load_quantized_model(
    model_class: type,
    checkpoint_path: str,
    *args,
    **kwargs
) -> Tuple[nn.Module, Dict]:
    """
    Load quantized model from checkpoint.

    Args:
        model_class: Model class to instantiate
        checkpoint_path: Path to checkpoint
        *args, **kwargs: Arguments for model initialization

    Returns:
        Tuple of (model, metadata)

    Example:
        >>> from models import ResNet18
        >>> model, metadata = load_quantized_model(
        ...     ResNet18,
        ...     'checkpoints/phase9/resnet_int8.pth',
        ...     num_classes=11
        ... )
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Create model instance
    model = model_class(*args, **kwargs)

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    metadata = checkpoint.get('metadata', {})
    logger.info(f"Loaded quantized model from {checkpoint_path}")

    return model, metadata
