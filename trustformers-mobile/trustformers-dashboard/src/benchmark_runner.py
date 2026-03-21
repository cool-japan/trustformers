"""
Benchmark runner for TrustformeRS models.
Executes standardized benchmarks and collects performance metrics.
"""

import time
import json
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import psutil
import threading
from dataclasses import dataclass
import yaml

try:
    import pynvml
    NVIDIA_AVAILABLE = True
except ImportError:
    NVIDIA_AVAILABLE = False

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""
    model_name: str
    device: str
    batch_size: int
    sequence_length: int
    num_iterations: int = 100
    warmup_iterations: int = 10
    measure_memory: bool = True
    measure_power: bool = True
    timeout_seconds: int = 300

class BenchmarkRunner:
    def __init__(self):
        self.benchmark_configs = self._load_benchmark_configs()
        self.running_benchmarks = {}
        
        if NVIDIA_AVAILABLE:
            pynvml.nvmlInit()
    
    def _load_benchmark_configs(self) -> Dict[str, Dict]:
        """Load predefined benchmark configurations."""
        return {
            'bert_base': {
                'model': 'bert-base-uncased',
                'task': 'classification',
                'default_batch_size': 32,
                'default_seq_length': 128,
                'script': 'benchmarks/bert_inference.rs'
            },
            'gpt2_gen': {
                'model': 'gpt2',
                'task': 'generation',
                'default_batch_size': 8,
                'default_seq_length': 512,
                'script': 'benchmarks/gpt2_generation.rs'
            },
            'vit': {
                'model': 'vit-base-patch16-224',
                'task': 'image_classification',
                'default_batch_size': 64,
                'default_seq_length': None,  # Image size instead
                'script': 'benchmarks/vit_inference.rs'
            },
            'mobile_bert': {
                'model': 'mobilebert-uncased',
                'task': 'classification',
                'default_batch_size': 128,
                'default_seq_length': 128,
                'script': 'benchmarks/mobile_bert.rs'
            }
        }
    
    def run(self, benchmark_type: str, batch_size: int = None, 
            sequence_length: int = None, device: str = 'cpu') -> Dict[str, Any]:
        """Run a benchmark and return results."""
        
        if benchmark_type not in self.benchmark_configs:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")
        
        config = self.benchmark_configs[benchmark_type]
        
        # Use provided values or defaults
        batch_size = batch_size or config['default_batch_size']
        sequence_length = sequence_length or config['default_seq_length']
        
        # Create benchmark config
        benchmark = BenchmarkConfig(
            model_name=config['model'],
            device=device,
            batch_size=batch_size,
            sequence_length=sequence_length
        )
        
        # Run the benchmark
        return self._execute_benchmark(benchmark, config['script'])
    
    def _execute_benchmark(self, config: BenchmarkConfig, 
                          script_path: str) -> Dict[str, Any]:
        """Execute a benchmark script and collect metrics."""
        
        # Prepare environment
        env = os.environ.copy()
        env['TRUSTFORMERS_DEVICE'] = config.device
        env['TRUSTFORMERS_BATCH_SIZE'] = str(config.batch_size)
        env['TRUSTFORMERS_SEQ_LENGTH'] = str(config.sequence_length)
        env['TRUSTFORMERS_NUM_ITERATIONS'] = str(config.num_iterations)
        env['TRUSTFORMERS_WARMUP_ITERATIONS'] = str(config.warmup_iterations)
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name
            env['TRUSTFORMERS_OUTPUT_FILE'] = output_file
        
        # Start monitoring threads
        memory_monitor = MemoryMonitor(config.device)
        power_monitor = PowerMonitor(config.device) if config.measure_power else None
        
        memory_monitor.start()
        if power_monitor:
            power_monitor.start()
        
        try:
            # Run benchmark script
            start_time = time.time()
            
            # Build and run the Rust benchmark
            cmd = [
                'cargo', 'run', '--release', '--bin', 
                os.path.basename(script_path).replace('.rs', '')
            ]
            
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for completion with timeout
            stdout, stderr = process.communicate(timeout=config.timeout_seconds)
            
            if process.returncode != 0:
                raise RuntimeError(f"Benchmark failed: {stderr}")
            
            end_time = time.time()
            
            # Stop monitoring
            memory_monitor.stop()
            if power_monitor:
                power_monitor.stop()
            
            # Read results from output file
            with open(output_file, 'r') as f:
                benchmark_results = json.load(f)
            
            # Calculate additional metrics
            total_time = end_time - start_time
            memory_stats = memory_monitor.get_stats()
            power_stats = power_monitor.get_stats() if power_monitor else {}
            
            # Combine results
            results = {
                'model': config.model_name,
                'device': config.device,
                'batch_size': config.batch_size,
                'sequence_length': config.sequence_length,
                'total_time': total_time,
                'iterations': config.num_iterations,
                
                # Latency metrics (from benchmark output)
                'latency': benchmark_results.get('mean_latency_ms', 0),
                'latency_std': benchmark_results.get('std_latency_ms', 0),
                'p50_latency': benchmark_results.get('p50_latency_ms', 0),
                'p95_latency': benchmark_results.get('p95_latency_ms', 0),
                'p99_latency': benchmark_results.get('p99_latency_ms', 0),
                
                # Throughput
                'throughput': benchmark_results.get('throughput_samples_per_sec', 0),
                
                # Memory metrics
                'memory': memory_stats['avg_usage_gb'],
                'peak_memory': memory_stats['peak_usage_gb'],
                'memory_bandwidth_gb_s': memory_stats.get('bandwidth_gb_s', 0),
                
                # Power metrics
                'avg_power_w': power_stats.get('avg_power_w', 0),
                'peak_power_w': power_stats.get('peak_power_w', 0),
                'energy_j': power_stats.get('total_energy_j', 0),
                
                # Efficiency metrics
                'samples_per_joule': (
                    (config.batch_size * config.num_iterations) / power_stats.get('total_energy_j', 1)
                    if power_stats else 0
                ),
                
                # Raw data for analysis
                'latency_distribution': benchmark_results.get('latencies', []),
                'timestamp': time.time()
            }
            
            return results
            
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Benchmark timed out after {config.timeout_seconds}s")
        finally:
            # Cleanup
            if os.path.exists(output_file):
                os.remove(output_file)
    
    def run_comparison(self, models: List[str], devices: List[str], 
                      batch_sizes: List[int]) -> Dict[str, Any]:
        """Run benchmarks across multiple configurations for comparison."""
        results = []
        
        for model in models:
            for device in devices:
                for batch_size in batch_sizes:
                    try:
                        result = self.run(
                            benchmark_type=model,
                            batch_size=batch_size,
                            device=device
                        )
                        results.append(result)
                    except Exception as e:
                        print(f"Failed to run {model} on {device}: {e}")
        
        return self._analyze_comparison(results)
    
    def _analyze_comparison(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze comparison results."""
        comparison = {
            'results': results,
            'summary': {},
            'winners': {}
        }
        
        # Group by metric
        metrics = ['latency', 'throughput', 'memory', 'energy_j']
        
        for metric in metrics:
            values = {}
            for result in results:
                key = f"{result['model']}_{result['device']}"
                values[key] = result.get(metric, 0)
            
            # Find best performer
            if metric in ['latency', 'memory', 'energy_j']:
                # Lower is better
                best = min(values.items(), key=lambda x: x[1])
            else:
                # Higher is better
                best = max(values.items(), key=lambda x: x[1])
            
            comparison['winners'][metric] = best[0]
            comparison['summary'][metric] = values
        
        return comparison

class MemoryMonitor:
    """Monitor memory usage during benchmark execution."""
    
    def __init__(self, device: str):
        self.device = device
        self.running = False
        self.thread = None
        self.measurements = []
        self.gpu_handle = None
        
        if device.startswith('cuda') and NVIDIA_AVAILABLE:
            gpu_id = int(device.split(':')[1]) if ':' in device else 0
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    
    def start(self):
        """Start monitoring."""
        self.running = True
        self.measurements = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _monitor_loop(self):
        """Monitoring loop."""
        while self.running:
            if self.gpu_handle:
                # GPU memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                self.measurements.append({
                    'timestamp': time.time(),
                    'used_gb': mem_info.used / (1024**3),
                    'total_gb': mem_info.total / (1024**3)
                })
            else:
                # System memory
                mem = psutil.virtual_memory()
                self.measurements.append({
                    'timestamp': time.time(),
                    'used_gb': mem.used / (1024**3),
                    'total_gb': mem.total / (1024**3)
                })
            
            time.sleep(0.1)  # 100ms sampling
    
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        if not self.measurements:
            return {'avg_usage_gb': 0, 'peak_usage_gb': 0}
        
        used_values = [m['used_gb'] for m in self.measurements]
        
        # Calculate bandwidth (approximate)
        if len(self.measurements) > 10:
            deltas = []
            for i in range(1, len(self.measurements)):
                dt = self.measurements[i]['timestamp'] - self.measurements[i-1]['timestamp']
                dm = abs(self.measurements[i]['used_gb'] - self.measurements[i-1]['used_gb'])
                if dt > 0:
                    deltas.append(dm / dt)
            
            bandwidth = np.mean(deltas) if deltas else 0
        else:
            bandwidth = 0
        
        return {
            'avg_usage_gb': np.mean(used_values),
            'peak_usage_gb': np.max(used_values),
            'min_usage_gb': np.min(used_values),
            'bandwidth_gb_s': bandwidth
        }

class PowerMonitor:
    """Monitor power consumption during benchmark execution."""
    
    def __init__(self, device: str):
        self.device = device
        self.running = False
        self.thread = None
        self.measurements = []
        self.gpu_handle = None
        
        if device.startswith('cuda') and NVIDIA_AVAILABLE:
            gpu_id = int(device.split(':')[1]) if ':' in device else 0
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    
    def start(self):
        """Start monitoring."""
        self.running = True
        self.measurements = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join()
    
    def _monitor_loop(self):
        """Power monitoring loop."""
        while self.running:
            if self.gpu_handle:
                try:
                    # GPU power in milliwatts
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                    self.measurements.append({
                        'timestamp': time.time(),
                        'power_w': power_mw / 1000.0
                    })
                except:
                    # Power measurement not supported
                    pass
            else:
                # CPU power (if available via psutil or other means)
                # This is platform-specific and may not be available
                pass
            
            time.sleep(0.1)  # 100ms sampling
    
    def get_stats(self) -> Dict[str, float]:
        """Get power statistics."""
        if not self.measurements:
            return {'avg_power_w': 0, 'peak_power_w': 0, 'total_energy_j': 0}
        
        power_values = [m['power_w'] for m in self.measurements]
        
        # Calculate total energy (integrate power over time)
        total_energy = 0
        for i in range(1, len(self.measurements)):
            dt = self.measurements[i]['timestamp'] - self.measurements[i-1]['timestamp']
            avg_power = (self.measurements[i]['power_w'] + self.measurements[i-1]['power_w']) / 2
            total_energy += avg_power * dt
        
        return {
            'avg_power_w': np.mean(power_values),
            'peak_power_w': np.max(power_values),
            'min_power_w': np.min(power_values),
            'total_energy_j': total_energy
        }