
import unittest
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import shutil
import logging
from packages.deployment.optimization.onnx_export import export_to_onnx, validate_onnx_export, ONNXExportConfig

# Configure logging to dev/null or capture
logging.getLogger('packages.deployment.optimization.onnx_export').setLevel(logging.ERROR)

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 10, 2)
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TestONNXExport(unittest.TestCase):
    """Test ONNX export functionality."""

    def setUp(self):
        """Setup test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.model = MockModel()
        self.dummy_input = torch.randn(1, 1, 10)
        self.save_path = self.temp_dir / "model.onnx"

    def tearDown(self):
        """Cleanup."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_export_to_onnx(self):
        """Test basic export."""
        export_to_onnx(
            self.model,
            self.dummy_input,
            str(self.save_path)
        )
        self.assertTrue(self.save_path.exists())
        self.assertGreater(self.save_path.stat().st_size, 0)

    def test_validate_onnx_export(self):
        """Test export validation."""
        # Check if onnx/onnxruntime installed
        try:
            import onnx
            import onnxruntime
        except ImportError:
            self.skipTest("onnx or onnxruntime not installed")

        export_to_onnx(self.model, self.dummy_input, str(self.save_path))
        
        is_valid = validate_onnx_export(
            str(self.save_path),
            self.model,
            self.dummy_input
        )
        self.assertTrue(is_valid)

    def test_export_config(self):
        """Test export with config."""
        config = ONNXExportConfig(
            opset_version=12,
            input_names=['input_signal'],
            output_names=['class_logits']
        )
        
        export_to_onnx(
            self.model,
            self.dummy_input,
            str(self.save_path),
            config=config
        )
        self.assertTrue(self.save_path.exists())

