"""
Deployment utilities for LSTM_PFD models.

This module provides tools for:
- Model quantization (INT8, FP16)
- ONNX export
- Optimized inference
- Model profiling and benchmarking

Author: Syed Abbas Ahmad
Date: 2025-11-20
"""

from .quantization import (
    quantize_model_dynamic,
    quantize_model_static,
    quantize_to_fp16,
    QuantizationConfig
)

from .onnx_export import (
    export_to_onnx,
    validate_onnx_export,
    optimize_onnx_model,
    ONNXExportConfig
)

from .inference import (
    OptimizedInferenceEngine,
    TorchInferenceEngine,
    ONNXInferenceEngine,
    benchmark_inference
)

from .model_optimization import (
    prune_model,
    fuse_model_layers,
    optimize_for_deployment,
    calculate_model_stats
)

__all__ = [
    # Quantization
    'quantize_model_dynamic',
    'quantize_model_static',
    'quantize_to_fp16',
    'QuantizationConfig',

    # ONNX
    'export_to_onnx',
    'validate_onnx_export',
    'optimize_onnx_model',
    'ONNXExportConfig',

    # Inference
    'OptimizedInferenceEngine',
    'TorchInferenceEngine',
    'ONNXInferenceEngine',
    'benchmark_inference',

    # Optimization
    'prune_model',
    'fuse_model_layers',
    'optimize_for_deployment',
    'calculate_model_stats',
]
