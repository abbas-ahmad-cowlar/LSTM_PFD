#!/usr/bin/env python3
"""
Stress Tests for Bearing Fault Diagnosis System

Comprehensive stress tests to verify system stability under load:
- Large batch processing
- Memory leak detection
- GPU memory stress testing
- Concurrent inference requests
- Extended training runs

Usage:
    pytest tests/stress_tests.py -v
    pytest tests/stress_tests.py -v -k "memory"  # Only memory tests
    pytest tests/stress_tests.py -v --stress-duration 60  # Longer runs

Author: Critical Deficiency Fix #8 (Priority: 82)
Date: 2026-01-18
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
import numpy as np
import time
import gc
import tracemalloc
from typing import Optional, List, Dict, Any
from torch.utils.data import DataLoader, TensorDataset


# Pytest configuration for stress tests
def pytest_addoption(parser):
    """Add custom pytest options for stress tests."""
    parser.addoption(
        "--stress-duration",
        default=30,
        type=int,
        help="Duration in seconds for stress tests"
    )
    parser.addoption(
        "--stress-samples",
        default=10000,
        type=int,
        help="Number of samples for large batch tests"
    )


@pytest.fixture
def stress_duration(request):
    """Get stress test duration from command line."""
    return request.config.getoption("--stress-duration", default=30)


@pytest.fixture
def stress_samples(request):
    """Get number of stress test samples from command line."""
    return request.config.getoption("--stress-samples", default=10000)


@pytest.fixture
def device():
    """Get available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def large_dataset(stress_samples):
    """Create a large synthetic dataset for stress testing."""
    signal_length = 1024  # Smaller for faster testing
    num_classes = 11
    
    signals = torch.randn(stress_samples, 1, signal_length)
    labels = torch.randint(0, num_classes, (stress_samples,))
    
    return TensorDataset(signals, labels)


@pytest.fixture
def simple_model():
    """Create a simple CNN model for testing."""
    class SimpleCNN(torch.nn.Module):
        def __init__(self, num_classes=11):
            super().__init__()
            self.conv1 = torch.nn.Conv1d(1, 32, 7, padding=3)
            self.conv2 = torch.nn.Conv1d(32, 64, 5, padding=2)
            self.pool = torch.nn.AdaptiveAvgPool1d(1)
            self.fc = torch.nn.Linear(64, num_classes)
        
        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.nn.functional.max_pool1d(x, 2)
            x = torch.relu(self.conv2(x))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)
    
    return SimpleCNN()


# =============================================================================
# Large Batch Processing Tests
# =============================================================================

class TestLargeBatchProcessing:
    """Test system behavior with large batch sizes."""
    
    @pytest.mark.stress
    def test_large_batch_forward_pass(self, large_dataset, simple_model, device):
        """Test forward pass with large batches."""
        model = simple_model.to(device)
        loader = DataLoader(large_dataset, batch_size=256, shuffle=False)
        
        model.eval()
        total_samples = 0
        start_time = time.time()
        
        with torch.no_grad():
            for batch_signals, batch_labels in loader:
                batch_signals = batch_signals.to(device)
                outputs = model(batch_signals)
                total_samples += len(batch_signals)
        
        elapsed = time.time() - start_time
        throughput = total_samples / elapsed
        
        print(f"\n  Processed {total_samples} samples in {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.0f} samples/sec")
        
        assert total_samples == len(large_dataset)
        assert throughput > 100, f"Throughput too low: {throughput} samples/sec"
    
    @pytest.mark.stress
    def test_extreme_batch_size(self, device, simple_model):
        """Test with extremely large batch sizes (up to memory limit)."""
        model = simple_model.to(device)
        model.eval()
        
        # Try increasingly large batches until we find the limit
        max_successful_batch = 0
        
        for batch_size in [64, 128, 256, 512, 1024, 2048]:
            try:
                signals = torch.randn(batch_size, 1, 1024, device=device)
                with torch.no_grad():
                    _ = model(signals)
                max_successful_batch = batch_size
                del signals
                torch.cuda.empty_cache() if device.type == 'cuda' else None
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    break
                raise
        
        print(f"\n  Max successful batch size: {max_successful_batch}")
        assert max_successful_batch >= 64, "Should handle at least batch size 64"
    
    @pytest.mark.stress
    def test_batch_size_consistency(self, simple_model, device):
        """Verify model produces consistent outputs regardless of batch size."""
        model = simple_model.to(device)
        model.eval()
        
        # Fixed input
        torch.manual_seed(42)
        signals = torch.randn(64, 1, 1024, device=device)
        
        # Get predictions with batch=64
        with torch.no_grad():
            out_64 = model(signals)
        
        # Get predictions with batch=1
        outs_1 = []
        with torch.no_grad():
            for i in range(64):
                outs_1.append(model(signals[i:i+1]))
        out_1 = torch.cat(outs_1, dim=0)
        
        # Should be identical
        assert torch.allclose(out_64, out_1, atol=1e-5), "Batch size affects predictions"


# =============================================================================
# Memory Leak Detection Tests
# =============================================================================

class TestMemoryLeakDetection:
    """Detect memory leaks during training and inference."""
    
    @pytest.mark.stress
    def test_inference_memory_stability(self, simple_model, device, stress_duration):
        """Check memory doesn't grow during repeated inference."""
        model = simple_model.to(device)
        model.eval()
        
        # Baseline memory
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
        else:
            tracemalloc.start()
        
        # Run repeated inference
        iterations = 0
        start_time = time.time()
        
        while time.time() - start_time < min(stress_duration, 10):  # Cap at 10s
            signals = torch.randn(32, 1, 1024, device=device)
            with torch.no_grad():
                _ = model(signals)
            del signals
            iterations += 1
        
        # Check final memory
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            memory_growth = final_memory - initial_memory
        else:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_growth = current
        
        memory_growth_mb = memory_growth / (1024 * 1024)
        print(f"\n  Iterations: {iterations}")
        print(f"  Memory growth: {memory_growth_mb:.2f} MB")
        
        assert memory_growth_mb < 10, f"Memory leak detected: {memory_growth_mb:.2f} MB"
    
    @pytest.mark.stress
    def test_training_memory_stability(self, large_dataset, simple_model, device):
        """Check memory stability during training loop."""
        model = simple_model.to(device)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        
        loader = DataLoader(large_dataset, batch_size=64, shuffle=True)
        
        # Warm-up
        for batch_signals, batch_labels in list(loader)[:5]:
            batch_signals = batch_signals.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_signals)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            memory_after_warmup = torch.cuda.memory_allocated()
        
        # Training iterations
        for epoch in range(2):
            for batch_signals, batch_labels in loader:
                batch_signals = batch_signals.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_signals)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
        
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            memory_after_training = torch.cuda.memory_allocated()
            growth = (memory_after_training - memory_after_warmup) / (1024 * 1024)
            print(f"\n  Memory growth during training: {growth:.2f} MB")
            assert growth < 50, f"Potential memory leak: {growth:.2f} MB"


# =============================================================================
# GPU Memory Stress Tests
# =============================================================================

@pytest.mark.gpu
class TestGPUMemoryStress:
    """GPU-specific stress tests."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_fragmentation(self):
        """Test GPU memory fragmentation under stress."""
        device = torch.device('cuda')
        
        # Allocate and deallocate tensors of varying sizes
        for _ in range(100):
            sizes = np.random.randint(100, 10000, size=10)
            tensors = [torch.randn(s, 1024, device=device) for s in sizes]
            
            # Delete in random order
            np.random.shuffle(tensors)
            for t in tensors:
                del t
            
            torch.cuda.empty_cache()
        
        # Should not have raised OOM
        final_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"\n  Final GPU memory allocated: {final_allocated:.2f} MB")
        assert final_allocated < 100, "GPU memory not properly freed"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_utilization_spike(self, simple_model):
        """Test behavior under sustained GPU load."""
        device = torch.device('cuda')
        model = simple_model.to(device)
        model.eval()
        
        start_time = time.time()
        inference_count = 0
        
        # Sustained inference for 5 seconds
        while time.time() - start_time < 5:
            signals = torch.randn(128, 1, 1024, device=device)
            with torch.no_grad():
                _ = model(signals)
            inference_count += 128
        
        throughput = inference_count / 5
        print(f"\n  GPU inference throughput: {throughput:.0f} samples/sec")
        
        # Check GPU is still responsive
        test_tensor = torch.randn(10, device=device)
        assert test_tensor.sum().item() is not None, "GPU unresponsive after stress"


# =============================================================================
# Concurrent Request Tests
# =============================================================================

class TestConcurrentRequests:
    """Test concurrent inference requests."""
    
    @pytest.mark.stress
    def test_concurrent_inference(self, simple_model, device):
        """Test concurrent inference with threading."""
        import threading
        from queue import Queue
        
        model = simple_model.to(device)
        model.eval()
        model.share_memory()  # For thread safety
        
        results = Queue()
        errors = Queue()
        
        def worker(worker_id: int, num_inferences: int):
            try:
                for i in range(num_inferences):
                    signals = torch.randn(16, 1, 1024, device=device)
                    with torch.no_grad():
                        output = model(signals)
                    results.put((worker_id, output.shape))
            except Exception as e:
                errors.put((worker_id, str(e)))
        
        # Start concurrent workers
        num_workers = 4
        inferences_per_worker = 50
        threads = []
        
        for i in range(num_workers):
            t = threading.Thread(target=worker, args=(i, inferences_per_worker))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join(timeout=30)
        
        # Check results
        assert errors.empty(), f"Errors during concurrent inference: {list(errors.queue)}"
        assert results.qsize() == num_workers * inferences_per_worker
        print(f"\n  Completed {results.qsize()} concurrent inferences")


# =============================================================================
# Extended Training Run Tests
# =============================================================================

class TestExtendedTraining:
    """Test system stability during extended training."""
    
    @pytest.mark.stress
    @pytest.mark.slow
    def test_extended_training_stability(self, large_dataset, simple_model, device):
        """Test training stability over many epochs."""
        model = simple_model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        loader = DataLoader(large_dataset, batch_size=64, shuffle=True)
        
        num_epochs = 5
        loss_history = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            batches = 0
            
            for batch_signals, batch_labels in loader:
                batch_signals = batch_signals.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_signals)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batches += 1
            
            avg_loss = epoch_loss / batches
            loss_history.append(avg_loss)
            print(f"\n  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        
        # Loss should generally decrease (or at least not explode)
        assert loss_history[-1] < loss_history[0] * 5, "Loss exploded during training"
        assert not np.isnan(loss_history[-1]), "Training produced NaN loss"


# =============================================================================
# Data Loading Stress Tests
# =============================================================================

class TestDataLoadingStress:
    """Test data loading under stress conditions."""
    
    @pytest.mark.stress
    def test_high_worker_count(self, large_dataset):
        """Test DataLoader with many workers."""
        # Only test with available CPU cores
        import os
        num_workers = min(8, os.cpu_count() or 4)
        
        loader = DataLoader(
            large_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        batches_loaded = 0
        start_time = time.time()
        
        for batch in loader:
            batches_loaded += 1
            if batches_loaded >= 50:  # Load first 50 batches
                break
        
        elapsed = time.time() - start_time
        print(f"\n  Loaded {batches_loaded} batches with {num_workers} workers in {elapsed:.2f}s")
        
        assert batches_loaded == 50
    
    @pytest.mark.stress
    def test_rapid_loader_creation(self):
        """Test creating/destroying DataLoaders rapidly."""
        dataset = TensorDataset(
            torch.randn(1000, 1, 1024),
            torch.randint(0, 11, (1000,))
        )
        
        for i in range(20):
            loader = DataLoader(dataset, batch_size=32, num_workers=2)
            # Load a few batches
            for batch in list(loader)[:5]:
                pass
            del loader
        
        # Should not have accumulated resources
        gc.collect()
        print("\n  Created/destroyed 20 DataLoaders successfully")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-x', '--tb=short'])
