"""
Prometheus Metrics for LSTM-PFD API

Exposes metrics at /metrics endpoint for monitoring:
- Request counts and latencies
- Model inference times
- System resource usage

Reference: Master Roadmap Chapter 4.5

Usage:
    from packages.dashboard.utils.metrics import (
        REQUEST_COUNT, REQUEST_LATENCY, INFERENCE_TIME
    )
    
    # In request handler
    REQUEST_COUNT.labels(method='POST', endpoint='/predict').inc()
    
    # Time model inference
    with INFERENCE_TIME.labels(model='pinn').time():
        prediction = model.predict(signal)
"""

from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from functools import wraps
import time
import psutil

# Request metrics
REQUEST_COUNT = Counter(
    'lstm_pfd_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'lstm_pfd_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

# Inference metrics
INFERENCE_TIME = Histogram(
    'lstm_pfd_inference_seconds',
    'Model inference time in seconds',
    ['model'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

INFERENCE_COUNT = Counter(
    'lstm_pfd_inferences_total',
    'Total model inferences',
    ['model', 'fault_class']
)

# Model metrics
MODEL_LOADED = Gauge(
    'lstm_pfd_model_loaded',
    'Whether model is loaded (1) or not (0)',
    ['model']
)

MODEL_SIZE_BYTES = Gauge(
    'lstm_pfd_model_size_bytes',
    'Model size in bytes',
    ['model']
)

# System metrics
SYSTEM_CPU_PERCENT = Gauge(
    'lstm_pfd_system_cpu_percent',
    'System CPU usage percentage'
)

SYSTEM_MEMORY_PERCENT = Gauge(
    'lstm_pfd_system_memory_percent',
    'System memory usage percentage'
)

GPU_MEMORY_USED = Gauge(
    'lstm_pfd_gpu_memory_bytes',
    'GPU memory used in bytes',
    ['gpu_id']
)

# Application info
APP_INFO = Info(
    'lstm_pfd',
    'Application information'
)


def track_request(method: str, endpoint: str):
    """Decorator to track request metrics."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                REQUEST_COUNT.labels(method=method, endpoint=endpoint, status='success').inc()
                return result
            except Exception as e:
                REQUEST_COUNT.labels(method=method, endpoint=endpoint, status='error').inc()
                raise
            finally:
                REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(
                    time.time() - start_time
                )
        return wrapper
    return decorator


def track_inference(model_name: str):
    """Decorator to track inference metrics."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with INFERENCE_TIME.labels(model=model_name).time():
                result = func(*args, **kwargs)
            if hasattr(result, 'predicted_class'):
                INFERENCE_COUNT.labels(
                    model=model_name,
                    fault_class=result.predicted_class
                ).inc()
            return result
        return wrapper
    return decorator


def update_system_metrics():
    """Update system resource metrics."""
    SYSTEM_CPU_PERCENT.set(psutil.cpu_percent())
    SYSTEM_MEMORY_PERCENT.set(psutil.virtual_memory().percent)
    
    # Try to get GPU metrics
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            GPU_MEMORY_USED.labels(gpu_id=str(i)).set(mem_info.used)
        pynvml.nvmlShutdown()
    except (ImportError, Exception):
        pass


def set_app_info(version: str, model_version: str = None):
    """Set application info metrics."""
    info = {'version': version}
    if model_version:
        info['model_version'] = model_version
    APP_INFO.info(info)


def get_metrics() -> bytes:
    """Generate Prometheus metrics output."""
    update_system_metrics()
    return generate_latest()


def get_metrics_content_type() -> str:
    """Get content type for metrics endpoint."""
    return CONTENT_TYPE_LATEST


# Flask/FastAPI integration helper
def create_metrics_endpoint(app):
    """
    Create /metrics endpoint for Flask or FastAPI.
    
    Usage:
        from packages.dashboard.utils.metrics import create_metrics_endpoint
        create_metrics_endpoint(app)
    """
    if hasattr(app, 'get'):  # FastAPI
        from fastapi import Response
        
        @app.get('/metrics')
        async def metrics():
            return Response(
                content=get_metrics(),
                media_type=get_metrics_content_type()
            )
    else:  # Flask
        @app.route('/metrics')
        def metrics():
            from flask import Response
            return Response(
                get_metrics(),
                mimetype=get_metrics_content_type()
            )


if __name__ == '__main__':
    # Test metrics
    set_app_info('1.0.0', 'pinn-v1')
    
    # Simulate some requests
    REQUEST_COUNT.labels(method='POST', endpoint='/predict', status='success').inc()
    REQUEST_COUNT.labels(method='GET', endpoint='/health', status='success').inc()
    
    # Simulate inference
    with INFERENCE_TIME.labels(model='pinn').time():
        time.sleep(0.05)  # Simulate 50ms inference
    
    INFERENCE_COUNT.labels(model='pinn', fault_class='wear').inc()
    
    # Print metrics
    print(get_metrics().decode())
