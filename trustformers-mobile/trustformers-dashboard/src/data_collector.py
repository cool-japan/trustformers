"""
Data collection module for TrustformeRS performance metrics.
Collects real-time metrics from models and system resources.
"""

import psutil
import threading
import time
import json
import os
from datetime import datetime
from collections import deque, defaultdict
import numpy as np
import redis
import sqlite3
from typing import Dict, List, Any, Optional
import requests

try:
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False

class DataCollector:
    def __init__(self, db_path: str = "data/metrics.db", 
                 redis_url: str = "redis://localhost:6379",
                 retention_hours: int = 24):
        self.db_path = db_path
        self.redis_client = redis.from_url(redis_url)
        self.retention_hours = retention_hours
        self.running = False
        self.thread = None
        
        # In-memory buffers
        self.latency_buffer = deque(maxlen=1000)
        self.throughput_buffer = deque(maxlen=1000)
        self.memory_buffer = deque(maxlen=1000)
        self.device_buffer = deque(maxlen=1000)
        
        # Metrics aggregation
        self.model_metrics = defaultdict(lambda: {
            'latencies': deque(maxlen=100),
            'throughputs': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'error_count': 0,
            'total_requests': 0
        })
        
        # Initialize database
        self._init_database()
        
        # Initialize GPU monitoring if available
        if NVIDIA_AVAILABLE:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
        else:
            self.gpu_count = 0
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    model_name TEXT,
                    device TEXT,
                    metric_type TEXT,
                    value REAL,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS benchmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    benchmark_name TEXT,
                    model_name TEXT,
                    device TEXT,
                    batch_size INTEGER,
                    sequence_length INTEGER,
                    latency_mean REAL,
                    latency_p95 REAL,
                    latency_p99 REAL,
                    throughput REAL,
                    memory_used REAL,
                    memory_peak REAL,
                    metadata TEXT
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_timestamp 
                ON metrics(timestamp)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_metrics_model 
                ON metrics(model_name, metric_type)
            ''')
    
    def start(self):
        """Start the data collection thread."""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._collection_loop)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self):
        """Stop the data collection thread."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _collection_loop(self):
        """Main collection loop running in background thread."""
        while self.running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect GPU metrics
                if self.gpu_count > 0:
                    self._collect_gpu_metrics()
                
                # Collect model metrics from endpoints
                self._collect_model_metrics()
                
                # Clean old data
                self._cleanup_old_data()
                
                # Sleep for collection interval
                time.sleep(1)
                
            except Exception as e:
                print(f"Error in collection loop: {e}")
                time.sleep(5)
    
    def _collect_system_metrics(self):
        """Collect CPU and memory metrics."""
        timestamp = datetime.now()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        
        # Memory metrics
        mem = psutil.virtual_memory()
        
        # Store in buffer
        system_metrics = {
            'timestamp': timestamp.isoformat(),
            'cpu_percent': cpu_percent,
            'cpu_freq_current': cpu_freq.current if cpu_freq else 0,
            'memory_percent': mem.percent,
            'memory_used_gb': mem.used / (1024**3),
            'memory_available_gb': mem.available / (1024**3)
        }
        
        # Store in Redis for real-time access
        self.redis_client.setex(
            'system_metrics:latest',
            60,  # 60 second TTL
            json.dumps(system_metrics)
        )
    
    def _collect_gpu_metrics(self):
        """Collect GPU metrics using nvidia-ml-py."""
        timestamp = datetime.now()
        
        for i in range(self.gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Power draw
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            except:
                power = 0
            
            gpu_metrics = {
                'timestamp': timestamp.isoformat(),
                'gpu_id': i,
                'gpu_name': pynvml.nvmlDeviceGetName(handle).decode('utf-8'),
                'utilization_percent': util.gpu,
                'memory_percent': util.memory,
                'memory_used_gb': mem_info.used / (1024**3),
                'memory_total_gb': mem_info.total / (1024**3),
                'temperature_c': temp,
                'power_w': power
            }
            
            # Store in buffer
            self.device_buffer.append(gpu_metrics)
            
            # Store in Redis
            self.redis_client.setex(
                f'gpu_metrics:{i}:latest',
                60,
                json.dumps(gpu_metrics)
            )
    
    def _collect_model_metrics(self):
        """Collect metrics from model serving endpoints."""
        # List of model endpoints to monitor
        endpoints = self._get_model_endpoints()
        
        for endpoint in endpoints:
            try:
                # Query metrics endpoint
                response = requests.get(
                    f"{endpoint['url']}/metrics",
                    timeout=5
                )
                
                if response.status_code == 200:
                    metrics = response.json()
                    self._process_model_metrics(endpoint['model_name'], metrics)
                    
            except Exception as e:
                print(f"Error collecting from {endpoint['url']}: {e}")
    
    def _get_model_endpoints(self) -> List[Dict[str, str]]:
        """Get list of model serving endpoints to monitor."""
        # This could be from configuration or service discovery
        endpoints_json = self.redis_client.get('model_endpoints')
        if endpoints_json:
            return json.loads(endpoints_json)
        
        # Default endpoints
        return [
            {'model_name': 'bert-base', 'url': 'http://localhost:8001'},
            {'model_name': 'gpt2', 'url': 'http://localhost:8002'},
            {'model_name': 'mobilenet', 'url': 'http://localhost:8003'},
        ]
    
    def _process_model_metrics(self, model_name: str, metrics: Dict[str, Any]):
        """Process and store model metrics."""
        timestamp = datetime.now()
        
        # Extract relevant metrics
        latency = metrics.get('inference_latency_ms', 0)
        throughput = metrics.get('throughput_samples_per_sec', 0)
        memory = metrics.get('memory_usage_mb', 0)
        errors = metrics.get('error_count', 0)
        
        # Update model metrics
        model_data = self.model_metrics[model_name]
        model_data['latencies'].append(latency)
        model_data['throughputs'].append(throughput)
        model_data['memory_usage'].append(memory)
        model_data['error_count'] = errors
        model_data['total_requests'] += metrics.get('request_count', 0)
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO metrics (timestamp, model_name, device, metric_type, value)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, model_name, metrics.get('device', 'cpu'), 'latency', latency))
            
            conn.execute('''
                INSERT INTO metrics (timestamp, model_name, device, metric_type, value)
                VALUES (?, ?, ?, ?, ?)
            ''', (timestamp, model_name, metrics.get('device', 'cpu'), 'throughput', throughput))
    
    def _cleanup_old_data(self):
        """Remove data older than retention period."""
        cutoff_time = datetime.now().timestamp() - (self.retention_hours * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                DELETE FROM metrics 
                WHERE timestamp < datetime(?, 'unixepoch')
            ''', (cutoff_time,))
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current aggregate metrics."""
        active_models = len(self.model_metrics)
        
        # Calculate averages
        all_latencies = []
        all_throughputs = []
        
        for model_data in self.model_metrics.values():
            if model_data['latencies']:
                all_latencies.extend(list(model_data['latencies']))
            if model_data['throughputs']:
                all_throughputs.extend(list(model_data['throughputs']))
        
        avg_latency = np.mean(all_latencies) if all_latencies else 0
        total_throughput = sum(all_throughputs) if all_throughputs else 0
        
        # Get GPU memory usage
        gpu_memory = 0
        if self.gpu_count > 0:
            gpu_metrics = self.redis_client.get('gpu_metrics:0:latest')
            if gpu_metrics:
                gpu_data = json.loads(gpu_metrics)
                gpu_memory = gpu_data.get('memory_percent', 0)
        
        return {
            'active_models': active_models,
            'avg_latency': avg_latency,
            'throughput': total_throughput,
            'gpu_memory': gpu_memory
        }
    
    def get_latency_history(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get latency history for the specified time period."""
        cutoff_time = datetime.now().timestamp() - (minutes * 60)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT timestamp, model_name, value 
                FROM metrics 
                WHERE metric_type = 'latency' 
                AND timestamp > datetime(?, 'unixepoch')
                ORDER BY timestamp
            ''', (cutoff_time,))
            
            return [
                {
                    'timestamp': row[0],
                    'model': row[1],
                    'latency': row[2]
                }
                for row in cursor
            ]
    
    def get_throughput_history(self, minutes: int = 5) -> List[Dict[str, Any]]:
        """Get throughput history for the specified time period."""
        cutoff_time = datetime.now().timestamp() - (minutes * 60)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT timestamp, model_name, value 
                FROM metrics 
                WHERE metric_type = 'throughput' 
                AND timestamp > datetime(?, 'unixepoch')
                ORDER BY timestamp
            ''', (cutoff_time,))
            
            return [
                {
                    'timestamp': row[0],
                    'model': row[1],
                    'throughput': row[2]
                }
                for row in cursor
            ]
    
    def get_memory_usage(self) -> List[Dict[str, Any]]:
        """Get current memory usage across devices."""
        devices = []
        
        # System memory
        mem = psutil.virtual_memory()
        devices.append({
            'name': 'System RAM',
            'allocated': mem.used / (1024**3),
            'reserved': mem.total / (1024**3),
            'free': mem.available / (1024**3)
        })
        
        # GPU memory
        for i in range(self.gpu_count):
            gpu_metrics = self.redis_client.get(f'gpu_metrics:{i}:latest')
            if gpu_metrics:
                gpu_data = json.loads(gpu_metrics)
                devices.append({
                    'name': f"GPU {i} ({gpu_data['gpu_name']})",
                    'allocated': gpu_data['memory_used_gb'],
                    'reserved': gpu_data['memory_total_gb'],
                    'free': gpu_data['memory_total_gb'] - gpu_data['memory_used_gb']
                })
        
        return devices
    
    def record_benchmark(self, benchmark_data: Dict[str, Any]):
        """Record benchmark results to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO benchmarks (
                    benchmark_name, model_name, device, batch_size, sequence_length,
                    latency_mean, latency_p95, latency_p99, throughput,
                    memory_used, memory_peak, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                benchmark_data['benchmark_name'],
                benchmark_data['model_name'],
                benchmark_data['device'],
                benchmark_data['batch_size'],
                benchmark_data['sequence_length'],
                benchmark_data['latency_mean'],
                benchmark_data['latency_p95'],
                benchmark_data['latency_p99'],
                benchmark_data['throughput'],
                benchmark_data['memory_used'],
                benchmark_data['memory_peak'],
                json.dumps(benchmark_data.get('metadata', {}))
            ))