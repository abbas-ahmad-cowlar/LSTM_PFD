"""
Unit Tests for Deployment Module

Tests for deployment utilities (quantization, ONNX, inference).

Author: LSTM_PFD Team
Date: 2025-11-20
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import platform
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Check if quantization is available
QUANTIZATION_AVAILABLE = platform.system() != 'Darwin'  # Skip on macOS

from deployment.quantization import (
    quantize_model_dynamic,
    quantize_to_fp16,
    compare_model_sizes
)
from deployment.inference import (
    TorchInferenceEngine,
    InferenceConfig
)
from deployment.model_optimization import (
    calculate_model_stats,
    prune_model
)


@pytest.mark.unit
class TestQuantization:
    """Test suite for model quantization."""

    @pytest.mark.skipif(not QUANTIZATION_AVAILABLE, reason="Quantization not available on macOS")
    def test_dynamic_quantization(self, simple_cnn_model):
        """Test dynamic quantization."""
        model = simple_cnn_model
        model.eval()

        # Quantize
        quantized_model = quantize_model_dynamic(model, inplace=False)

        # Check model still works
        x = torch.randn(1, 1, 1024)
        with torch.no_grad():
            output = quantized_model(x)

        assert output.shape == (1, 11)
        assert torch.all(torch.isfinite(output))

    def test_fp16_conversion(self, simple_cnn_model):
        """Test FP16 conversion."""
        model = simple_cnn_model
        model.eval()

        # Convert to FP16
        fp16_model = quantize_to_fp16(model, inplace=False)

        # Check parameters are FP16
        for param in fp16_model.parameters():
            assert param.dtype == torch.float16

        # Check model still works
        x = torch.randn(1, 1, 1024).half()
        with torch.no_grad():
            output = fp16_model(x)

        assert output.dtype == torch.float16
        assert output.shape == (1, 11)

    @pytest.mark.skipif(not QUANTIZATION_AVAILABLE, reason="Quantization not available on macOS")
    def test_model_size_comparison(self, simple_cnn_model):
        """Test model size comparison."""
        original_model = simple_cnn_model
        quantized_model = quantize_model_dynamic(original_model, inplace=False)

        stats = compare_model_sizes(original_model, quantized_model)

        # Check stats
        assert 'compression_ratio' in stats
        assert 'size_reduction_percent' in stats
        assert stats['compression_ratio'] >= 1.0
        assert 0 <= stats['size_reduction_percent'] <= 100


@pytest.mark.unit
class TestInferenceEngine:
    """Test suite for inference engines."""

    def test_torch_inference_engine_init(self):
        """Test TorchInferenceEngine initialization."""
        config = InferenceConfig(device='cpu', batch_size=32)
        engine = TorchInferenceEngine(config)

        assert engine.config.device == 'cpu'
        assert engine.config.batch_size == 32

    def test_torch_inference_single_prediction(self, simple_cnn_model):
        """Test single prediction."""
        config = InferenceConfig(device='cpu')
        engine = TorchInferenceEngine(config)

        # Directly set the model (avoid pickling issues with local classes)
        engine.model = simple_cnn_model
        engine.model.eval()

        # Test prediction
        input_data = np.random.randn(1, 1024).astype(np.float32)
        output = engine.predict(input_data)

        assert output.shape == (1, 11)
        assert np.all(np.isfinite(output))

    def test_torch_inference_batch_prediction(self, simple_cnn_model):
        """Test batch prediction."""
        config = InferenceConfig(device='cpu', batch_size=8)
        engine = TorchInferenceEngine(config)
        engine.model = simple_cnn_model
        engine.model.eval()

        # Test batch prediction
        input_data = np.random.randn(32, 1, 1024).astype(np.float32)
        outputs = engine.predict_batch(input_data, batch_size=8)

        assert outputs.shape == (32, 11)
        assert np.all(np.isfinite(outputs))


@pytest.mark.unit
class TestModelOptimization:
    """Test suite for model optimization."""

    def test_calculate_model_stats(self, simple_cnn_model):
        """Test model statistics calculation."""
        stats = calculate_model_stats(simple_cnn_model)

        # Check required keys
        assert 'total_params' in stats
        assert 'trainable_params' in stats
        assert 'size_mb' in stats
        assert 'layer_counts' in stats

        # Check values are reasonable
        assert stats['total_params'] > 0
        assert stats['trainable_params'] > 0
        assert stats['size_mb'] > 0

    def test_prune_model(self, simple_cnn_model):
        """Test model pruning."""
        model = simple_cnn_model

        # Get original stats
        original_stats = calculate_model_stats(model)

        # Prune model
        pruned_model = prune_model(model, pruning_amount=0.3, inplace=False)

        # Get pruned stats
        pruned_stats = calculate_model_stats(pruned_model)

        # Check sparsity increased
        assert pruned_stats['sparsity'] > original_stats['sparsity']
        assert pruned_stats['sparsity'] >= 0.25  # Should be around 30%

        # Model should still work
        x = torch.randn(1, 1, 1024)
        with torch.no_grad():
            output = pruned_model(x)

        assert output.shape == (1, 11)


@pytest.mark.unit
class TestONNXExport:
    """Test suite for ONNX export."""

    @pytest.mark.slow
    def test_onnx_export_basic(self, simple_cnn_model):
        """Test basic ONNX export."""
        pytest.importorskip("onnx")

        from deployment.onnx_export import export_to_onnx

        model = simple_cnn_model
        model.eval()

        dummy_input = torch.randn(1, 1, 1024)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        try:
            # Export to ONNX
            result_path = export_to_onnx(model, dummy_input, onnx_path)

            # Check file exists
            assert Path(result_path).exists()
            assert Path(result_path).stat().st_size > 0

        finally:
            if Path(onnx_path).exists():
                Path(onnx_path).unlink()

    @pytest.mark.slow
    def test_onnx_validation(self, simple_cnn_model):
        """Test ONNX export validation."""
        pytest.importorskip("onnx")
        pytest.importorskip("onnxruntime")

        from deployment.onnx_export import export_to_onnx, validate_onnx_export

        model = simple_cnn_model
        model.eval()

        dummy_input = torch.randn(1, 1, 1024)

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name

        try:
            # Export
            export_to_onnx(model, dummy_input, onnx_path)

            # Validate
            test_input = torch.randn(1, 1, 1024)
            is_valid = validate_onnx_export(onnx_path, model, test_input)

            assert is_valid is True

        finally:
            if Path(onnx_path).exists():
                Path(onnx_path).unlink()
