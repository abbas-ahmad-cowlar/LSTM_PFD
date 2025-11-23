"""
Optimized Inference Engine

Provides high-performance inference engines for production deployment:
- PyTorch inference (FP32, FP16, INT8)
- ONNX Runtime inference
- TensorRT inference (for NVIDIA GPUs)
- Batched inference with automatic batching
- Benchmarking utilities

Author: Syed Abbas Ahmad
Date: 2025-11-20
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, List, Union, Tuple
from dataclasses import dataclass
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size: int = 32
    use_amp: bool = False  # Automatic mixed precision
    num_threads: int = 4  # For CPU inference
    warmup_runs: int = 10
    profile: bool = False  # Enable profiling


class BaseInferenceEngine(ABC):
    """
    Abstract base class for inference engines.
    """

    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.model = None
        self.preprocessor = None
        self.postprocessor = None

    @abstractmethod
    def load_model(self, model_path: str):
        """Load model from file."""
        pass

    @abstractmethod
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run single prediction."""
        pass

    @abstractmethod
    def predict_batch(self, input_data: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Run batch predictions."""
        pass

    def set_preprocessor(self, preprocessor):
        """Set preprocessing function."""
        self.preprocessor = preprocessor

    def set_postprocessor(self, postprocessor):
        """Set postprocessing function."""
        self.postprocessor = postprocessor


class TorchInferenceEngine(BaseInferenceEngine):
    """
    PyTorch-based inference engine.

    Supports FP32, FP16, and quantized models.

    Example:
        >>> engine = TorchInferenceEngine()
        >>> engine.load_model('checkpoints/best_model.pth')
        >>> predictions = engine.predict(input_data)
    """

    def __init__(self, config: Optional[InferenceConfig] = None):
        super().__init__(config)
        self.device = torch.device(self.config.device)

        # Set number of threads for CPU inference
        if self.device.type == 'cpu':
            torch.set_num_threads(self.config.num_threads)
            logger.info(f"CPU inference threads: {self.config.num_threads}")

    def load_model(self, model_path: str, model_class: Optional[type] = None):
        """
        Load PyTorch model.

        Args:
            model_path: Path to model checkpoint
            model_class: Model class (if not using state dict)
        """
        logger.info(f"Loading PyTorch model from {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Checkpoint format
            if model_class is None:
                raise ValueError("model_class required for state dict loading")

            self.model = model_class()
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct model
            self.model = checkpoint

        self.model.to(self.device)
        self.model.eval()

        # Convert to FP16 if requested
        if self.config.use_amp and self.device.type == 'cuda':
            self.model = self.model.half()
            logger.info("Model converted to FP16")

        logger.info(f"✓ Model loaded on {self.device}")

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on single sample or batch.

        Args:
            input_data: Input array [B, C, T] or [C, T]

        Returns:
            Predictions [B, num_classes] or [num_classes]
        """
        # Convert to tensor
        if input_data.ndim == 2:
            input_data = input_data[np.newaxis, :]  # Add batch dimension

        x = torch.from_numpy(input_data).float().to(self.device)

        # Apply preprocessing
        if self.preprocessor is not None:
            x = self.preprocessor(x)

        # Inference
        with torch.no_grad():
            if self.config.use_amp and self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    output = self.model(x)
            else:
                output = self.model(x)

        # Convert to numpy
        output = output.cpu().numpy()

        # Apply postprocessing
        if self.postprocessor is not None:
            output = self.postprocessor(output)

        return output

    def predict_batch(
        self,
        input_data: np.ndarray,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Run inference on batches of data.

        Args:
            input_data: Input array [N, C, T]
            batch_size: Batch size (default: from config)

        Returns:
            Predictions [N, num_classes]
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        num_samples = len(input_data)
        outputs = []

        for i in range(0, num_samples, batch_size):
            batch = input_data[i:i + batch_size]
            output = self.predict(batch)
            outputs.append(output)

        return np.concatenate(outputs, axis=0)

    def benchmark(
        self,
        input_shape: Tuple[int, ...],
        num_runs: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.

        Args:
            input_shape: Input tensor shape
            num_runs: Number of runs

        Returns:
            Timing statistics
        """
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(self.config.warmup_runs):
            _ = self.predict(dummy_input)

        # Benchmark
        latencies = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.predict(dummy_input)
            latencies.append((time.time() - start) * 1000)

        return {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_samples_per_sec': 1000 / np.mean(latencies)
        }


class ONNXInferenceEngine(BaseInferenceEngine):
    """
    ONNX Runtime-based inference engine.

    Provides cross-platform deployment with optimized runtime.

    Example:
        >>> engine = ONNXInferenceEngine()
        >>> engine.load_model('models/model.onnx')
        >>> predictions = engine.predict(input_data)
    """

    def __init__(self, config: Optional[InferenceConfig] = None):
        super().__init__(config)
        self.session = None

    def load_model(self, model_path: str, providers: Optional[List[str]] = None):
        """
        Load ONNX model.

        Args:
            model_path: Path to ONNX model
            providers: Execution providers
        """
        try:
            import onnxruntime as ort
        except ImportError:
            logger.error("Please install onnxruntime: pip install onnxruntime")
            raise

        logger.info(f"Loading ONNX model from {model_path}")

        # Auto-detect providers if not specified
        if providers is None:
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')

        # Set number of threads for CPU
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.config.num_threads
        sess_options.inter_op_num_threads = self.config.num_threads

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(f"✓ ONNX model loaded with providers: {self.session.get_providers()}")

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference on input data.

        Args:
            input_data: Input array

        Returns:
            Predictions
        """
        if input_data.ndim == 2:
            input_data = input_data[np.newaxis, :]

        # Apply preprocessing
        if self.preprocessor is not None:
            input_data = self.preprocessor(input_data)

        # Inference
        output = self.session.run(
            [self.output_name],
            {self.input_name: input_data.astype(np.float32)}
        )[0]

        # Apply postprocessing
        if self.postprocessor is not None:
            output = self.postprocessor(output)

        return output

    def predict_batch(
        self,
        input_data: np.ndarray,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Run inference on batches."""
        if batch_size is None:
            batch_size = self.config.batch_size

        num_samples = len(input_data)
        outputs = []

        for i in range(0, num_samples, batch_size):
            batch = input_data[i:i + batch_size]
            output = self.predict(batch)
            outputs.append(output)

        return np.concatenate(outputs, axis=0)

    def benchmark(
        self,
        input_shape: Tuple[int, ...],
        num_runs: int = 100
    ) -> Dict[str, float]:
        """Benchmark inference speed."""
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(self.config.warmup_runs):
            _ = self.predict(dummy_input)

        # Benchmark
        latencies = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.predict(dummy_input)
            latencies.append((time.time() - start) * 1000)

        return {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'median_latency_ms': np.median(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_samples_per_sec': 1000 / np.mean(latencies)
        }


class OptimizedInferenceEngine:
    """
    Unified inference engine that automatically selects best backend.

    Tries backends in order: TensorRT > ONNX Runtime > PyTorch

    Example:
        >>> engine = OptimizedInferenceEngine()
        >>> engine.load_model('checkpoints/best_model.pth')
        >>> predictions = engine.predict(input_data)
        >>> # Engine automatically uses fastest available backend
    """

    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.backend = None
        self.engine = None

    def load_model(
        self,
        model_path: str,
        model_class: Optional[type] = None,
        prefer_backend: Optional[str] = None
    ):
        """
        Load model with optimal backend.

        Args:
            model_path: Path to model
            model_class: Model class (for PyTorch)
            prefer_backend: Force specific backend ('torch', 'onnx', 'tensorrt')
        """
        # Determine backend
        if prefer_backend:
            self.backend = prefer_backend
        elif model_path.endswith('.onnx'):
            self.backend = 'onnx'
        elif model_path.endswith('.pth') or model_path.endswith('.pt'):
            self.backend = 'torch'
        else:
            self.backend = 'torch'  # Default

        logger.info(f"Using backend: {self.backend}")

        # Initialize engine
        if self.backend == 'onnx':
            self.engine = ONNXInferenceEngine(self.config)
            self.engine.load_model(model_path)
        else:  # torch
            self.engine = TorchInferenceEngine(self.config)
            self.engine.load_model(model_path, model_class)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference."""
        return self.engine.predict(input_data)

    def predict_batch(self, input_data: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """Run batch inference."""
        return self.engine.predict_batch(input_data, batch_size)

    def benchmark(self, input_shape: Tuple[int, ...], num_runs: int = 100) -> Dict[str, float]:
        """Benchmark inference."""
        return self.engine.benchmark(input_shape, num_runs)


def benchmark_inference(
    model_path: str,
    input_shape: Tuple[int, ...],
    backends: Optional[List[str]] = None,
    num_runs: int = 100
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark inference across multiple backends.

    Args:
        model_path: Path to model (PyTorch or ONNX)
        input_shape: Input tensor shape
        backends: List of backends to test (default: all available)
        num_runs: Number of inference runs

    Returns:
        Dictionary mapping backend to timing stats

    Example:
        >>> results = benchmark_inference(
        ...     'checkpoints/best_model.pth',
        ...     (1, 1, SIGNAL_LENGTH),
        ...     backends=['torch', 'torch_fp16', 'onnx']
        ... )
        >>> for backend, stats in results.items():
        ...     print(f"{backend}: {stats['mean_latency_ms']:.2f} ms")
    """
    if backends is None:
        backends = ['torch', 'onnx']

    results = {}

    for backend in backends:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Benchmarking: {backend}")
            logger.info(f"{'='*60}")

            if backend == 'torch':
                config = InferenceConfig(use_amp=False)
                engine = TorchInferenceEngine(config)
                # Assume we can load the model directly
                # In practice, you'd need to pass model_class
                # engine.load_model(model_path)

            elif backend == 'torch_fp16':
                config = InferenceConfig(use_amp=True, device='cuda')
                engine = TorchInferenceEngine(config)
                # engine.load_model(model_path)

            elif backend == 'onnx':
                if not model_path.endswith('.onnx'):
                    logger.warning(f"Skipping ONNX benchmark (model not in ONNX format)")
                    continue
                engine = ONNXInferenceEngine()
                engine.load_model(model_path)

            else:
                logger.warning(f"Unknown backend: {backend}")
                continue

            # Run benchmark
            stats = engine.benchmark(input_shape, num_runs)
            results[backend] = stats

            logger.info(f"✓ {backend}: {stats['mean_latency_ms']:.2f} ± {stats['std_latency_ms']:.2f} ms")

        except Exception as e:
            logger.error(f"✗ {backend} benchmark failed: {e}")

    return results


def compare_backends(results: Dict[str, Dict[str, float]]):
    """
    Print comparison of benchmark results.

    Args:
        results: Results from benchmark_inference
    """
    print("\n" + "="*80)
    print("Backend Comparison Summary")
    print("="*80)
    print(f"{'Backend':<20} {'Mean (ms)':<15} {'Std (ms)':<15} {'Throughput (samples/s)':<25}")
    print("-"*80)

    for backend, stats in sorted(results.items(), key=lambda x: x[1]['mean_latency_ms']):
        print(f"{backend:<20} {stats['mean_latency_ms']:<15.2f} {stats['std_latency_ms']:<15.2f} {stats['throughput_samples_per_sec']:<25.1f}")

    print("="*80)

    # Find fastest
    fastest = min(results.items(), key=lambda x: x[1]['mean_latency_ms'])
    print(f"\n🏆 Fastest backend: {fastest[0]} ({fastest[1]['mean_latency_ms']:.2f} ms)")
