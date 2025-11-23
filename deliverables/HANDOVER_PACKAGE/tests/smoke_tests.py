"""
Smoke Tests for Production Deployment

Quick sanity checks to verify system is working correctly.
Run these tests after deployment to ensure basic functionality.

Author: Syed Abbas Ahmad
Date: 2025-11-23

Usage:
    python smoke_tests.py
"""

import requests
import numpy as np
import sys
import time


def test_api_health():
    """Test 1: API health endpoint responds."""
    print("Test 1: Checking API health...", end=" ")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data["status"] == "healthy", f"Status not healthy: {data}"
        print("✅ PASS")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_model_loaded():
    """Test 2: Model is loaded."""
    print("Test 2: Checking model loaded...", end=" ")
    try:
        response = requests.get("http://localhost:8000/model/info", timeout=5)
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert "model_name" in data, "model_name not in response"
        assert data["num_classes"] == 11, f"Expected 11 classes, got {data['num_classes']}"
        print("✅ PASS")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_single_prediction():
    """Test 3: Single prediction endpoint works."""
    print("Test 3: Testing single prediction...", end=" ")
    try:
        # Generate dummy signal
        signal = np.random.randn(102400).tolist()

        response = requests.post(
            "http://localhost:8000/predict",
            json={"signal": signal, "return_probabilities": True},
            timeout=10
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()

        # Validate response structure
        assert "predicted_class" in data, "predicted_class not in response"
        assert "class_name" in data, "class_name not in response"
        assert "confidence" in data, "confidence not in response"
        assert 0 <= data["predicted_class"] <= 10, f"Invalid class: {data['predicted_class']}"
        assert 0 <= data["confidence"] <= 1, f"Invalid confidence: {data['confidence']}"

        print("✅ PASS")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_batch_prediction():
    """Test 4: Batch prediction endpoint works."""
    print("Test 4: Testing batch prediction...", end=" ")
    try:
        # Generate dummy signals
        signals = [np.random.randn(102400).tolist() for _ in range(3)]

        response = requests.post(
            "http://localhost:8000/predict/batch",
            json={"signals": signals},
            timeout=15
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()

        # Validate response structure
        assert "predictions" in data, "predictions not in response"
        assert len(data["predictions"]) == 3, f"Expected 3 predictions, got {len(data['predictions'])}"
        assert "batch_size" in data, "batch_size not in response"

        print("✅ PASS")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_inference_latency():
    """Test 5: Inference latency is acceptable."""
    print("Test 5: Testing inference latency...", end=" ")
    try:
        signal = np.random.randn(102400).tolist()

        # Measure latency
        start = time.time()
        response = requests.post(
            "http://localhost:8000/predict",
            json={"signal": signal},
            timeout=10
        )
        latency = (time.time() - start) * 1000  # ms

        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        assert latency < 100, f"Latency {latency:.1f}ms exceeds 100ms threshold"

        print(f"✅ PASS (latency: {latency:.1f}ms)")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def test_error_handling():
    """Test 6: API handles invalid input correctly."""
    print("Test 6: Testing error handling...", end=" ")
    try:
        # Send invalid signal (wrong length)
        response = requests.post(
            "http://localhost:8000/predict",
            json={"signal": [0.1] * 100},  # Too short
            timeout=5
        )
        # Should return 400 or 422 error
        assert response.status_code in [400, 422], \
            f"Expected 400 or 422 for invalid input, got {response.status_code}"

        print("✅ PASS")
        return True
    except Exception as e:
        print(f"❌ FAIL: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("="*60)
    print("LSTM_PFD Smoke Tests")
    print("="*60)
    print()

    tests = [
        test_api_health,
        test_model_loaded,
        test_single_prediction,
        test_batch_prediction,
        test_inference_latency,
        test_error_handling
    ]

    results = []
    for test in tests:
        results.append(test())
        time.sleep(0.5)  # Small delay between tests

    print()
    print("="*60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("="*60)

    if all(results):
        print("✅ All smoke tests PASSED - System is ready!")
        return 0
    else:
        print("❌ Some smoke tests FAILED - Check deployment!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
