"""
Comprehensive Benchmarking Suite

Benchmarks for all phases of the project.

Author: Syed Abbas Ahmad
Date: 2025-11-20

Usage:
    python tests/benchmarks/benchmark_suite.py --output results/benchmarks.json
"""

import argparse
import time
import json
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.constants import SAMPLING_RATE, SIGNAL_LENGTH

from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkSuite:
    """Comprehensive benchmarking suite for all project phases."""

    def __init__(self):
        self.results = {}

    def benchmark_feature_extraction(self, num_samples: int = 100) -> Dict:
        """Benchmark feature extraction performance."""
        from packages.core.features.feature_extractor import FeatureExtractor

        logger.info(f"Benchmarking feature extraction ({num_samples} samples)...")

        extractor = FeatureExtractor(fs=SAMPLING_RATE)

        # Generate test signals
        signals = [np.random.randn(102400).astype(np.float32) for _ in range(num_samples)]

        # Benchmark
        start = time.time()
        for signal in signals:
            _ = extractor.extract_features(signal)
        total_time = time.time() - start

        results = {
            'num_samples': num_samples,
            'total_time_sec': total_time,
            'time_per_sample_ms': (total_time / num_samples) * 1000,
            'throughput_samples_per_sec': num_samples / total_time
        }

        logger.info(f"  Time per sample: {results['time_per_sample_ms']:.2f}ms")
        return results

    def benchmark_model_inference(
        self,
        model_path: str,
        num_samples: int = 100,
        batch_size: int = 32
    ) -> Dict:
        """Benchmark model inference performance."""
        from packages.deployment.optimization.inference import TorchInferenceEngine, InferenceConfig

        logger.info(f"Benchmarking model inference ({num_samples} samples)...")

        try:
            # Load model
            config = InferenceConfig(device='cpu', batch_size=batch_size)
            engine = TorchInferenceEngine(config)

            model = torch.load(model_path, map_location='cpu')
            engine.model = model
            engine.model.eval()

            # Generate test data
            test_data = np.random.randn(num_samples, 1, 102400).astype(np.float32)

            # Warmup
            _ = engine.predict(test_data[:1])

            # Benchmark
            start = time.time()
            outputs = engine.predict_batch(test_data, batch_size=batch_size)
            total_time = time.time() - start

            results = {
                'num_samples': num_samples,
                'batch_size': batch_size,
                'total_time_sec': total_time,
                'time_per_sample_ms': (total_time / num_samples) * 1000,
                'throughput_samples_per_sec': num_samples / total_time
            }

            logger.info(f"  Time per sample: {results['time_per_sample_ms']:.2f}ms")
            return results

        except Exception as e:
            logger.error(f"  Benchmark failed: {e}")
            return {'error': str(e)}

    def benchmark_quantized_model(self, model_path: str, num_samples: int = 100) -> Dict:
        """Benchmark quantized model performance."""
        logger.info(f"Benchmarking quantized model...")

        try:
            from packages.deployment.optimization.quantization import quantize_model_dynamic

            # Load original model
            model = torch.load(model_path, map_location='cpu')
            model.eval()

            # Quantize
            quantized_model = quantize_model_dynamic(model, inplace=False)

            # Test data
            test_input = torch.randn(num_samples, 1, 102400)

            # Benchmark original
            start = time.time()
            with torch.no_grad():
                for i in range(num_samples):
                    _ = model(test_input[i:i+1])
            original_time = time.time() - start

            # Benchmark quantized
            start = time.time()
            with torch.no_grad():
                for i in range(num_samples):
                    _ = quantized_model(test_input[i:i+1])
            quantized_time = time.time() - start

            results = {
                'original_time_ms': (original_time / num_samples) * 1000,
                'quantized_time_ms': (quantized_time / num_samples) * 1000,
                'speedup': original_time / quantized_time,
                'speedup_percent': ((original_time - quantized_time) / original_time) * 100
            }

            logger.info(f"  Speedup: {results['speedup']:.2f}x")
            return results

        except Exception as e:
            logger.error(f"  Benchmark failed: {e}")
            return {'error': str(e)}

    def benchmark_api_latency(self, api_url: str = "http://localhost:8000", num_requests: int = 100) -> Dict:
        """Benchmark API latency."""
        logger.info(f"Benchmarking API latency ({num_requests} requests)...")

        try:
            import requests

            # Test signal
            signal = np.random.randn(102400).tolist()

            latencies = []

            # Warmup
            for _ in range(5):
                response = requests.post(
                    f"{api_url}/predict",
                    json={"signal": signal, "return_probabilities": False},
                    timeout=30
                )

            # Benchmark
            for _ in range(num_requests):
                start = time.time()
                response = requests.post(
                    f"{api_url}/predict",
                    json={"signal": signal, "return_probabilities": False},
                    timeout=30
                )
                latency = (time.time() - start) * 1000
                latencies.append(latency)

                if response.status_code != 200:
                    logger.warning(f"  Request failed: {response.status_code}")

            results = {
                'num_requests': num_requests,
                'mean_latency_ms': np.mean(latencies),
                'std_latency_ms': np.std(latencies),
                'min_latency_ms': np.min(latencies),
                'max_latency_ms': np.max(latencies),
                'p50_latency_ms': np.percentile(latencies, 50),
                'p95_latency_ms': np.percentile(latencies, 95),
                'p99_latency_ms': np.percentile(latencies, 99)
            }

            logger.info(f"  Mean latency: {results['mean_latency_ms']:.2f}ms")
            logger.info(f"  P95 latency: {results['p95_latency_ms']:.2f}ms")
            return results

        except Exception as e:
            logger.error(f"  Benchmark failed: {e}")
            return {'error': str(e)}

    def benchmark_memory_usage(self, model_path: str) -> Dict:
        """Benchmark model memory usage."""
        import psutil
        import os

        logger.info("Benchmarking memory usage...")

        try:
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / (1024 ** 2)  # MB

            # Load model
            model = torch.load(model_path, map_location='cpu')

            after_load_memory = process.memory_info().rss / (1024 ** 2)

            # Run inference
            test_input = torch.randn(32, 1, 102400)
            with torch.no_grad():
                _ = model(test_input)

            after_inference_memory = process.memory_info().rss / (1024 ** 2)

            results = {
                'initial_memory_mb': initial_memory,
                'after_load_memory_mb': after_load_memory,
                'after_inference_memory_mb': after_inference_memory,
                'model_memory_mb': after_load_memory - initial_memory,
                'inference_memory_mb': after_inference_memory - after_load_memory
            }

            logger.info(f"  Model memory: {results['model_memory_mb']:.2f}MB")
            return results

        except Exception as e:
            logger.error(f"  Benchmark failed: {e}")
            return {'error': str(e)}

    def run_all_benchmarks(self, model_path: str = None, api_url: str = None) -> Dict:
        """Run all benchmarks."""
        logger.info("="*60)
        logger.info("Running Comprehensive Benchmark Suite")
        logger.info("="*60)

        results = {}

        # Feature extraction
        results['feature_extraction'] = self.benchmark_feature_extraction()

        # Model inference (if model provided)
        if model_path and Path(model_path).exists():
            results['model_inference'] = self.benchmark_model_inference(model_path)
            results['quantized_model'] = self.benchmark_quantized_model(model_path)
            results['memory_usage'] = self.benchmark_memory_usage(model_path)
        else:
            logger.warning("Skipping model benchmarks (no model path provided)")

        # API latency (if API URL provided)
        if api_url:
            results['api_latency'] = self.benchmark_api_latency(api_url)
        else:
            logger.warning("Skipping API benchmarks (no API URL provided)")

        logger.info("="*60)
        logger.info("Benchmark Suite Complete")
        logger.info("="*60)

        self.results = results
        return results

    def save_results(self, output_path: str):
        """Save benchmark results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "="*60)
        print("Benchmark Summary")
        print("="*60)

        if 'feature_extraction' in self.results:
            fe = self.results['feature_extraction']
            print(f"\nFeature Extraction:")
            print(f"  Time per sample: {fe['time_per_sample_ms']:.2f}ms")
            print(f"  Throughput: {fe['throughput_samples_per_sec']:.1f} samples/sec")

        if 'model_inference' in self.results:
            mi = self.results['model_inference']
            print(f"\nModel Inference:")
            print(f"  Time per sample: {mi['time_per_sample_ms']:.2f}ms")
            print(f"  Throughput: {mi['throughput_samples_per_sec']:.1f} samples/sec")

        if 'quantized_model' in self.results:
            qm = self.results['quantized_model']
            print(f"\nQuantized Model:")
            print(f"  Speedup: {qm['speedup']:.2f}x ({qm['speedup_percent']:.1f}%)")

        if 'api_latency' in self.results:
            api = self.results['api_latency']
            print(f"\nAPI Latency:")
            print(f"  Mean: {api['mean_latency_ms']:.2f}ms")
            print(f"  P95: {api['p95_latency_ms']:.2f}ms")
            print(f"  P99: {api['p99_latency_ms']:.2f}ms")

        if 'memory_usage' in self.results:
            mem = self.results['memory_usage']
            print(f"\nMemory Usage:")
            print(f"  Model: {mem['model_memory_mb']:.2f}MB")
            print(f"  Inference: {mem['inference_memory_mb']:.2f}MB")

        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive benchmark suite")

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--api-url',
        type=str,
        default=None,
        help='API URL for latency benchmarks'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_results.json',
        help='Output path for results'
    )

    parser.add_argument(
        '--num-samples',
        type=int,
        default=100,
        help='Number of samples for benchmarks'
    )

    args = parser.parse_args()

    # Run benchmarks
    suite = BenchmarkSuite()
    results = suite.run_all_benchmarks(
        model_path=args.model,
        api_url=args.api_url
    )

    # Save results
    suite.save_results(args.output)

    # Print summary
    suite.print_summary()


if __name__ == '__main__':
    main()
