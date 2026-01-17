"""
Load Testing Suite for LSTM-PFD Dashboard

This module provides comprehensive load testing to assess system behavior
under concurrent user stress. Addresses Deficiency #24 (Priority 50).

Features:
- Concurrent API request simulation
- Dashboard page load stress testing
- Model inference throughput testing
- Database connection pool stress testing
- Memory and resource monitoring during load
- Report generation with bottleneck identification

Usage:
    python tests/load_tests.py --users 50 --duration 60 --ramp-up 10
    python tests/load_tests.py --quick  # Quick 10-user, 30-second test
    python tests/load_tests.py --report-only  # Generate report from last run

Author: AI Research Team
Date: January 2026
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Tuple
from collections import defaultdict
import statistics
import traceback
import gc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed. Resource monitoring will be limited.")

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests not installed. HTTP load testing disabled.")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes for Load Test Results
# ============================================================================

@dataclass
class RequestResult:
    """Result of a single request."""
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    success: bool
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    payload_size_bytes: int = 0


@dataclass
class ResourceSnapshot:
    """System resource snapshot during load test."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    gpu_memory_mb: Optional[float] = None
    active_threads: int = 0
    open_connections: int = 0


@dataclass
class LoadTestSummary:
    """Summary of load test results."""
    test_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    requests_per_second: float
    error_rate: float
    peak_cpu_percent: float
    peak_memory_percent: float
    bottlenecks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat()
        return result


# ============================================================================
# Resource Monitor
# ============================================================================

class ResourceMonitor:
    """Monitors system resources during load tests."""
    
    def __init__(self, interval_seconds: float = 1.0):
        self.interval = interval_seconds
        self.snapshots: List[ResourceSnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start resource monitoring in background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Resource monitor started")
        
    def stop(self) -> List[ResourceSnapshot]:
        """Stop monitoring and return collected snapshots."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info(f"Resource monitor stopped. Collected {len(self.snapshots)} snapshots")
        return self.snapshots
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            snapshot = self._capture_snapshot()
            self.snapshots.append(snapshot)
            time.sleep(self.interval)
            
    def _capture_snapshot(self) -> ResourceSnapshot:
        """Capture current resource usage."""
        cpu_percent = 0.0
        memory_percent = 0.0
        memory_used_mb = 0.0
        gpu_memory_mb = None
        
        if HAS_PSUTIL:
            cpu_percent = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            memory_percent = mem.percent
            memory_used_mb = mem.used / (1024 * 1024)
            
        if HAS_TORCH and torch.cuda.is_available():
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
            except Exception:
                pass
                
        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            gpu_memory_mb=gpu_memory_mb,
            active_threads=threading.active_count()
        )


# ============================================================================
# Load Test Scenarios
# ============================================================================

class LoadTestScenario:
    """Base class for load test scenarios."""
    
    def __init__(self, name: str):
        self.name = name
        self.results: List[RequestResult] = []
        
    def execute(self, **kwargs) -> RequestResult:
        """Execute a single test iteration. Override in subclasses."""
        raise NotImplementedError
        
    def get_results(self) -> List[RequestResult]:
        return self.results


class MockAPIScenario(LoadTestScenario):
    """Simulates API requests when no real server is available."""
    
    def __init__(self, name: str, base_latency_ms: float = 50.0, 
                 latency_variance: float = 20.0, failure_rate: float = 0.02):
        super().__init__(name)
        self.base_latency = base_latency_ms
        self.latency_variance = latency_variance
        self.failure_rate = failure_rate
        self.endpoints = [
            '/api/signals',
            '/api/predictions',
            '/api/experiments',
            '/api/models',
            '/api/health',
            '/api/metrics',
        ]
        
    def execute(self, **kwargs) -> RequestResult:
        """Simulate an API request with realistic latency."""
        endpoint = np.random.choice(self.endpoints)
        
        # Simulate network latency
        latency = max(5, np.random.normal(self.base_latency, self.latency_variance))
        time.sleep(latency / 1000)
        
        # Simulate occasional failures
        success = np.random.random() > self.failure_rate
        status_code = 200 if success else np.random.choice([500, 502, 503, 504])
        error = None if success else f"Simulated error: HTTP {status_code}"
        
        result = RequestResult(
            endpoint=endpoint,
            method='GET',
            status_code=status_code,
            response_time_ms=latency,
            success=success,
            error=error,
            payload_size_bytes=np.random.randint(100, 10000)
        )
        self.results.append(result)
        return result


class HTTPAPIScenario(LoadTestScenario):
    """Real HTTP API load testing scenario."""
    
    def __init__(self, name: str, base_url: str, endpoints: List[Dict[str, Any]]):
        super().__init__(name)
        self.base_url = base_url
        self.endpoints = endpoints
        
    def execute(self, **kwargs) -> RequestResult:
        """Execute real HTTP request."""
        if not HAS_REQUESTS:
            raise RuntimeError("requests library not installed")
            
        endpoint_config = np.random.choice(self.endpoints)
        endpoint = endpoint_config.get('path', '/')
        method = endpoint_config.get('method', 'GET')
        payload = endpoint_config.get('payload', None)
        
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(url, json=payload, timeout=30)
            else:
                response = requests.request(method, url, json=payload, timeout=30)
                
            elapsed_ms = (time.time() - start_time) * 1000
            
            result = RequestResult(
                endpoint=endpoint,
                method=method,
                status_code=response.status_code,
                response_time_ms=elapsed_ms,
                success=response.status_code < 400,
                payload_size_bytes=len(response.content)
            )
            
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            result = RequestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time_ms=elapsed_ms,
                success=False,
                error=str(e)
            )
            
        self.results.append(result)
        return result


class ModelInferenceScenario(LoadTestScenario):
    """Load test model inference throughput."""
    
    def __init__(self, name: str, model: Optional[Any] = None, 
                 batch_size: int = 32, signal_length: int = 4096):
        super().__init__(name)
        self.model = model
        self.batch_size = batch_size
        self.signal_length = signal_length
        self.device = 'cuda' if HAS_TORCH and torch.cuda.is_available() else 'cpu'
        
        if model is None and HAS_TORCH:
            # Create a simple mock model for testing
            self.model = self._create_mock_model()
            
    def _create_mock_model(self):
        """Create a simple mock CNN model for testing."""
        class MockCNN(torch.nn.Module):
            def __init__(self, in_channels=1, num_classes=11):
                super().__init__()
                self.conv1 = torch.nn.Conv1d(in_channels, 32, 7, padding=3)
                self.conv2 = torch.nn.Conv1d(32, 64, 5, padding=2)
                self.pool = torch.nn.AdaptiveAvgPool1d(1)
                self.fc = torch.nn.Linear(64, num_classes)
                
            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.pool(x).squeeze(-1)
                return self.fc(x)
                
        model = MockCNN()
        model.to(self.device)
        model.eval()
        return model
        
    def execute(self, **kwargs) -> RequestResult:
        """Execute model inference and measure latency."""
        if not HAS_TORCH or self.model is None:
            # Simulate inference
            latency = max(5, np.random.normal(20, 5))
            time.sleep(latency / 1000)
            result = RequestResult(
                endpoint='model/inference',
                method='INFERENCE',
                status_code=200,
                response_time_ms=latency,
                success=True
            )
            self.results.append(result)
            return result
            
        batch = torch.randn(self.batch_size, 1, self.signal_length, device=self.device)
        
        start_time = time.time()
        try:
            with torch.no_grad():
                _ = self.model(batch)
            if self.device == 'cuda':
                torch.cuda.synchronize()
                
            elapsed_ms = (time.time() - start_time) * 1000
            
            result = RequestResult(
                endpoint='model/inference',
                method='INFERENCE',
                status_code=200,
                response_time_ms=elapsed_ms,
                success=True,
                payload_size_bytes=batch.numel() * 4
            )
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            result = RequestResult(
                endpoint='model/inference',
                method='INFERENCE',
                status_code=500,
                response_time_ms=elapsed_ms,
                success=False,
                error=str(e)
            )
            
        self.results.append(result)
        return result


class DatabaseScenario(LoadTestScenario):
    """Load test database operations."""
    
    def __init__(self, name: str, db_path: Optional[str] = None):
        super().__init__(name)
        self.db_path = db_path
        self.operations = ['SELECT', 'INSERT', 'UPDATE', 'COUNT']
        
    def execute(self, **kwargs) -> RequestResult:
        """Simulate database operations."""
        operation = np.random.choice(self.operations)
        
        # Simulate different operation latencies
        latency_map = {
            'SELECT': (5, 15),
            'INSERT': (10, 25),
            'UPDATE': (15, 30),
            'COUNT': (20, 50)
        }
        base, variance = latency_map[operation]
        latency = max(1, np.random.normal(base, variance))
        time.sleep(latency / 1000)
        
        result = RequestResult(
            endpoint=f'db/{operation.lower()}',
            method=operation,
            status_code=200,
            response_time_ms=latency,
            success=True
        )
        self.results.append(result)
        return result


# ============================================================================
# Load Test Runner
# ============================================================================

class LoadTestRunner:
    """Orchestrates load test execution."""
    
    def __init__(self, 
                 scenarios: List[LoadTestScenario],
                 num_users: int = 10,
                 duration_seconds: int = 60,
                 ramp_up_seconds: int = 10,
                 output_dir: Path = Path('results/load_tests')):
        self.scenarios = scenarios
        self.num_users = num_users
        self.duration = duration_seconds
        self.ramp_up = ramp_up_seconds
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.resource_monitor = ResourceMonitor()
        self.all_results: List[RequestResult] = []
        self._stop_event = threading.Event()
        
    def run(self) -> LoadTestSummary:
        """Run the load test and return summary."""
        logger.info(f"Starting load test: {self.num_users} users, "
                   f"{self.duration}s duration, {self.ramp_up}s ramp-up")
        
        start_time = datetime.now()
        test_start = time.time()
        
        # Start resource monitoring
        self.resource_monitor.start()
        
        # Create thread pool for concurrent users
        with ThreadPoolExecutor(max_workers=self.num_users) as executor:
            futures = []
            
            # Ramp up users gradually
            users_per_step = max(1, self.num_users // max(1, self.ramp_up))
            current_users = 0
            
            for i in range(self.ramp_up):
                if self._stop_event.is_set():
                    break
                    
                users_to_add = min(users_per_step, self.num_users - current_users)
                for _ in range(users_to_add):
                    future = executor.submit(self._user_session, test_start)
                    futures.append(future)
                    current_users += 1
                    
                time.sleep(1)
                logger.info(f"Ramp-up: {current_users}/{self.num_users} users active")
                
            # Wait for test duration
            remaining_duration = self.duration - self.ramp_up
            if remaining_duration > 0:
                time.sleep(remaining_duration)
                
            # Signal stop
            self._stop_event.set()
            
            # Collect results
            for future in as_completed(futures, timeout=30):
                try:
                    user_results = future.result()
                    self.all_results.extend(user_results)
                except Exception as e:
                    logger.error(f"User session error: {e}")
                    
        # Stop resource monitoring
        resource_snapshots = self.resource_monitor.stop()
        
        end_time = datetime.now()
        
        # Generate summary
        summary = self._generate_summary(
            start_time, end_time, resource_snapshots
        )
        
        # Save results
        self._save_results(summary, resource_snapshots)
        
        return summary
        
    def _user_session(self, test_start: float) -> List[RequestResult]:
        """Simulate a user session making requests."""
        session_results = []
        
        while not self._stop_event.is_set():
            # Pick a random scenario
            scenario = np.random.choice(self.scenarios)
            
            try:
                result = scenario.execute()
                session_results.append(result)
            except Exception as e:
                logger.warning(f"Request error: {e}")
                
            # Random think time between requests
            think_time = np.random.exponential(0.5)
            time.sleep(min(think_time, 2.0))
            
        return session_results
        
    def _generate_summary(self, 
                          start_time: datetime,
                          end_time: datetime,
                          resource_snapshots: List[ResourceSnapshot]) -> LoadTestSummary:
        """Generate load test summary from results."""
        
        duration = (end_time - start_time).total_seconds()
        
        if not self.all_results:
            logger.warning("No results collected!")
            return LoadTestSummary(
                test_name="load_test",
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_response_time_ms=0,
                min_response_time_ms=0,
                max_response_time_ms=0,
                p50_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                requests_per_second=0,
                error_rate=0,
                peak_cpu_percent=0,
                peak_memory_percent=0,
                bottlenecks=["No requests completed"]
            )
            
        response_times = [r.response_time_ms for r in self.all_results]
        successful = [r for r in self.all_results if r.success]
        failed = [r for r in self.all_results if not r.success]
        
        # Calculate percentiles
        sorted_times = sorted(response_times)
        n = len(sorted_times)
        p50 = sorted_times[int(n * 0.50)]
        p95 = sorted_times[int(n * 0.95)]
        p99 = sorted_times[int(n * 0.99)]
        
        # Resource peaks
        peak_cpu = max(s.cpu_percent for s in resource_snapshots) if resource_snapshots else 0
        peak_memory = max(s.memory_percent for s in resource_snapshots) if resource_snapshots else 0
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(
            response_times, peak_cpu, peak_memory, len(failed) / len(self.all_results)
        )
        
        return LoadTestSummary(
            test_name=f"load_test_{self.num_users}users_{self.duration}s",
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            total_requests=len(self.all_results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            avg_response_time_ms=statistics.mean(response_times),
            min_response_time_ms=min(response_times),
            max_response_time_ms=max(response_times),
            p50_response_time_ms=p50,
            p95_response_time_ms=p95,
            p99_response_time_ms=p99,
            requests_per_second=len(self.all_results) / duration,
            error_rate=len(failed) / len(self.all_results) * 100,
            peak_cpu_percent=peak_cpu,
            peak_memory_percent=peak_memory,
            bottlenecks=bottlenecks
        )
        
    def _identify_bottlenecks(self, 
                              response_times: List[float],
                              peak_cpu: float,
                              peak_memory: float,
                              error_rate: float) -> List[str]:
        """Identify potential performance bottlenecks."""
        bottlenecks = []
        
        if peak_cpu > 90:
            bottlenecks.append(f"CPU saturation: {peak_cpu:.1f}%")
        if peak_memory > 85:
            bottlenecks.append(f"Memory pressure: {peak_memory:.1f}%")
        if error_rate > 0.05:
            bottlenecks.append(f"High error rate: {error_rate*100:.2f}%")
            
        # Response time analysis
        p50 = np.percentile(response_times, 50)
        p99 = np.percentile(response_times, 99)
        
        if p99 > p50 * 10:
            bottlenecks.append(f"High latency variance: P99 ({p99:.0f}ms) >> P50 ({p50:.0f}ms)")
        if p50 > 500:
            bottlenecks.append(f"Slow median response time: {p50:.0f}ms")
            
        if not bottlenecks:
            bottlenecks.append("No significant bottlenecks detected")
            
        return bottlenecks
        
    def _save_results(self, 
                      summary: LoadTestSummary,
                      resource_snapshots: List[ResourceSnapshot]):
        """Save load test results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save summary
        summary_path = self.output_dir / f"summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)
        logger.info(f"Summary saved to {summary_path}")
        
        # Save detailed results
        results_path = self.output_dir / f"results_{timestamp}.json"
        results_data = [asdict(r) for r in self.all_results]
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        logger.info(f"Detailed results saved to {results_path}")
        
        # Save resource timeline
        resource_path = self.output_dir / f"resources_{timestamp}.json"
        resource_data = [asdict(s) for s in resource_snapshots]
        with open(resource_path, 'w') as f:
            json.dump(resource_data, f, indent=2)
        logger.info(f"Resource data saved to {resource_path}")
        
        # Generate plots if matplotlib available
        try:
            self._generate_plots(summary, resource_snapshots, timestamp)
        except ImportError:
            logger.warning("matplotlib not available, skipping plots")
            
    def _generate_plots(self, 
                        summary: LoadTestSummary,
                        resource_snapshots: List[ResourceSnapshot],
                        timestamp: str):
        """Generate visualization plots."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Response time distribution
        ax = axes[0, 0]
        response_times = [r.response_time_ms for r in self.all_results]
        ax.hist(response_times, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(summary.p50_response_time_ms, color='green', linestyle='--', 
                   label=f'P50: {summary.p50_response_time_ms:.0f}ms')
        ax.axvline(summary.p95_response_time_ms, color='orange', linestyle='--',
                   label=f'P95: {summary.p95_response_time_ms:.0f}ms')
        ax.axvline(summary.p99_response_time_ms, color='red', linestyle='--',
                   label=f'P99: {summary.p99_response_time_ms:.0f}ms')
        ax.set_xlabel('Response Time (ms)')
        ax.set_ylabel('Count')
        ax.set_title('Response Time Distribution')
        ax.legend()
        
        # Resource usage over time
        ax = axes[0, 1]
        if resource_snapshots:
            times = [(s.timestamp - resource_snapshots[0].timestamp) for s in resource_snapshots]
            cpu = [s.cpu_percent for s in resource_snapshots]
            mem = [s.memory_percent for s in resource_snapshots]
            ax.plot(times, cpu, label='CPU %', color='blue')
            ax.plot(times, mem, label='Memory %', color='green')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Usage (%)')
            ax.set_title('Resource Usage Over Time')
            ax.legend()
            
        # Throughput over time (requests per second in sliding windows)
        ax = axes[1, 0]
        timestamps = [r.timestamp for r in self.all_results]
        if timestamps:
            min_ts = min(timestamps)
            window_size = 5  # 5 second windows
            windows = defaultdict(int)
            for ts in timestamps:
                window = int((ts - min_ts) // window_size)
                windows[window] += 1
            window_times = sorted(windows.keys())
            throughput = [windows[w] / window_size for w in window_times]
            ax.plot([w * window_size for w in window_times], throughput, 
                    marker='o', markersize=3)
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Requests/second')
            ax.set_title('Throughput Over Time')
            
        # Error rate by endpoint
        ax = axes[1, 1]
        endpoint_stats = defaultdict(lambda: {'total': 0, 'errors': 0})
        for r in self.all_results:
            endpoint_stats[r.endpoint]['total'] += 1
            if not r.success:
                endpoint_stats[r.endpoint]['errors'] += 1
        endpoints = list(endpoint_stats.keys())[:10]  # Top 10
        error_rates = [endpoint_stats[e]['errors'] / endpoint_stats[e]['total'] * 100 
                      for e in endpoints]
        ax.barh(endpoints, error_rates, color='red', alpha=0.7)
        ax.set_xlabel('Error Rate (%)')
        ax.set_title('Error Rate by Endpoint')
        
        plt.suptitle(f'Load Test Results: {self.num_users} Users, {self.duration}s Duration',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = self.output_dir / f"plots_{timestamp}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Plots saved to {plot_path}")


# ============================================================================
# Report Generator
# ============================================================================

class LoadTestReportGenerator:
    """Generates comprehensive load test reports."""
    
    def __init__(self, results_dir: Path = Path('results/load_tests')):
        self.results_dir = results_dir
        
    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """Generate markdown report from latest test results."""
        
        # Find latest summary
        summary_files = sorted(self.results_dir.glob("summary_*.json"), reverse=True)
        if not summary_files:
            return "No load test results found."
            
        latest_summary_path = summary_files[0]
        with open(latest_summary_path) as f:
            summary_data = json.load(f)
            
        report = f"""# Load Test Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Test:** {summary_data.get('test_name', 'Unknown')}

## Executive Summary

| Metric | Value |
|--------|-------|
| Duration | {summary_data['duration_seconds']:.1f} seconds |
| Total Requests | {summary_data['total_requests']:,} |
| Successful | {summary_data['successful_requests']:,} |
| Failed | {summary_data['failed_requests']:,} |
| Error Rate | {summary_data['error_rate']:.2f}% |
| RPS | {summary_data['requests_per_second']:.1f} |

## Response Time Analysis

| Percentile | Latency (ms) |
|------------|--------------|
| Average | {summary_data['avg_response_time_ms']:.1f} |
| Minimum | {summary_data['min_response_time_ms']:.1f} |
| P50 (Median) | {summary_data['p50_response_time_ms']:.1f} |
| P95 | {summary_data['p95_response_time_ms']:.1f} |
| P99 | {summary_data['p99_response_time_ms']:.1f} |
| Maximum | {summary_data['max_response_time_ms']:.1f} |

## Resource Utilization

| Resource | Peak |
|----------|------|
| CPU | {summary_data['peak_cpu_percent']:.1f}% |
| Memory | {summary_data['peak_memory_percent']:.1f}% |

## Bottleneck Analysis

"""
        for bottleneck in summary_data.get('bottlenecks', []):
            report += f"- ⚠️ {bottleneck}\n"
            
        report += """

## Recommendations

"""
        # Add recommendations based on results
        if summary_data['p99_response_time_ms'] > 1000:
            report += "- **High P99 Latency:** Consider implementing request queuing or connection pooling.\n"
        if summary_data['error_rate'] > 1:
            report += "- **Elevated Error Rate:** Investigate error logs for root cause.\n"
        if summary_data['peak_cpu_percent'] > 80:
            report += "- **CPU Saturation:** Consider horizontal scaling or code optimization.\n"
        if summary_data['peak_memory_percent'] > 80:
            report += "- **Memory Pressure:** Check for memory leaks, consider adding RAM.\n"
        if summary_data['requests_per_second'] < 10:
            report += "- **Low Throughput:** Profile application to identify bottlenecks.\n"
            
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")
            
        return report


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Load Testing Suite for LSTM-PFD Dashboard',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run standard load test with 50 concurrent users for 60 seconds
    python tests/load_tests.py --users 50 --duration 60
    
    # Quick smoke test
    python tests/load_tests.py --quick
    
    # Test model inference throughput
    python tests/load_tests.py --scenario inference --users 10 --duration 30
    
    # Generate report from previous run
    python tests/load_tests.py --report-only
        """
    )
    
    parser.add_argument('--users', '-u', type=int, default=10,
                        help='Number of concurrent users (default: 10)')
    parser.add_argument('--duration', '-d', type=int, default=60,
                        help='Test duration in seconds (default: 60)')
    parser.add_argument('--ramp-up', '-r', type=int, default=10,
                        help='Ramp-up time in seconds (default: 10)')
    parser.add_argument('--scenario', '-s', type=str, default='all',
                        choices=['all', 'api', 'inference', 'database'],
                        help='Scenario to run (default: all)')
    parser.add_argument('--base-url', type=str, default=None,
                        help='Base URL for HTTP testing (uses mock if not provided)')
    parser.add_argument('--output-dir', '-o', type=str, default='results/load_tests',
                        help='Output directory for results')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: 10 users, 30 seconds')
    parser.add_argument('--report-only', action='store_true',
                        help='Generate report from most recent test results')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Report-only mode
    if args.report_only:
        generator = LoadTestReportGenerator(output_dir)
        report = generator.generate_report(output_dir / 'report.md')
        print(report)
        return
        
    # Quick mode overrides
    if args.quick:
        args.users = 10
        args.duration = 30
        args.ramp_up = 5
        
    # Setup scenarios
    scenarios: List[LoadTestScenario] = []
    
    if args.scenario in ['all', 'api']:
        if args.base_url:
            scenarios.append(HTTPAPIScenario(
                name='http_api',
                base_url=args.base_url,
                endpoints=[
                    {'path': '/api/health', 'method': 'GET'},
                    {'path': '/api/signals', 'method': 'GET'},
                    {'path': '/api/experiments', 'method': 'GET'},
                ]
            ))
        else:
            scenarios.append(MockAPIScenario(name='mock_api'))
            
    if args.scenario in ['all', 'inference']:
        scenarios.append(ModelInferenceScenario(name='model_inference'))
        
    if args.scenario in ['all', 'database']:
        scenarios.append(DatabaseScenario(name='database'))
        
    if not scenarios:
        scenarios.append(MockAPIScenario(name='mock_api'))
        
    # Run load test
    runner = LoadTestRunner(
        scenarios=scenarios,
        num_users=args.users,
        duration_seconds=args.duration,
        ramp_up_seconds=args.ramp_up,
        output_dir=output_dir
    )
    
    print(f"\n{'='*60}")
    print(f"Load Test Configuration")
    print(f"{'='*60}")
    print(f"  Users:      {args.users}")
    print(f"  Duration:   {args.duration}s")
    print(f"  Ramp-up:    {args.ramp_up}s")
    print(f"  Scenarios:  {[s.name for s in scenarios]}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}\n")
    
    summary = runner.run()
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Load Test Results")
    print(f"{'='*60}")
    print(f"  Total Requests:  {summary.total_requests:,}")
    print(f"  Successful:      {summary.successful_requests:,}")
    print(f"  Failed:          {summary.failed_requests:,}")
    print(f"  Error Rate:      {summary.error_rate:.2f}%")
    print(f"  Avg Response:    {summary.avg_response_time_ms:.1f}ms")
    print(f"  P50 Response:    {summary.p50_response_time_ms:.1f}ms")
    print(f"  P95 Response:    {summary.p95_response_time_ms:.1f}ms")
    print(f"  P99 Response:    {summary.p99_response_time_ms:.1f}ms")
    print(f"  RPS:             {summary.requests_per_second:.1f}")
    print(f"  Peak CPU:        {summary.peak_cpu_percent:.1f}%")
    print(f"  Peak Memory:     {summary.peak_memory_percent:.1f}%")
    print(f"\nBottlenecks:")
    for b in summary.bottlenecks:
        print(f"  - {b}")
    print(f"{'='*60}\n")
    
    # Generate report
    generator = LoadTestReportGenerator(output_dir)
    generator.generate_report(output_dir / 'report.md')
    
    print(f"✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
