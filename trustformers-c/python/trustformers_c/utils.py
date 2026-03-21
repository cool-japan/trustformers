"""
Utility functions for TrustformeRS-C Python bindings
"""

import os
import json
import time
import functools
import logging
from typing import Dict, Any, Optional, Union, List, Callable
from pathlib import Path
import numpy as np

from .core import TrustformersC, TrustformersError, get_platform_info, get_memory_usage

# Setup logging
logger = logging.getLogger(__name__)

class TrustformersConfig:
    """Configuration management for TrustformeRS"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path or self._find_config_file()
        self.config = self._load_config()
    
    def _find_config_file(self) -> str:
        """Find configuration file in standard locations"""
        config_names = [
            "trustformers_c.json",
            "trustformers_config.json",
            ".trustformers_c",
        ]
        
        search_paths = [
            Path.cwd(),
            Path.home(),
            Path.home() / ".config" / "trustformers",
            Path("/etc/trustformers"),
        ]
        
        for path in search_paths:
            for name in config_names:
                config_file = path / name
                if config_file.exists():
                    return str(config_file)
        
        # Return default path if not found
        return str(Path.home() / ".trustformers_c")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not os.path.exists(self.config_path):
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "library_path": None,
            "auto_init": True,
            "performance": {
                "enable_simd": True,
                "enable_dynamic_batching": True,
                "enable_kernel_fusion": True,
                "max_batch_size": 32,
                "target_latency_ms": 100,
                "num_threads": 0,  # Auto-detect
            },
            "logging": {
                "level": "INFO",
                "enable_performance_logging": False,
            },
            "memory": {
                "enable_tracking": True,
                "warning_threshold_mb": 1024,
                "max_memory_mb": 4096,
            },
        }
    
    def save_config(self) -> None:
        """Save configuration to file"""
        try:
            # Create directory if it doesn't exist
            config_dir = Path(self.config_path).parent
            config_dir.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_path}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with dictionary"""
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = deep_update(self.config, updates)

# Global configuration
_global_config = None

def get_config() -> TrustformersConfig:
    """Get global configuration"""
    global _global_config
    if _global_config is None:
        _global_config = TrustformersConfig()
    return _global_config

def set_config(config: TrustformersConfig) -> None:
    """Set global configuration"""
    global _global_config
    _global_config = config

class PerformanceTimer:
    """Performance timing utility"""
    
    def __init__(self, name: str = "operation"):
        """
        Initialize performance timer
        
        Args:
            name: Name of the operation being timed
        """
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def start(self) -> None:
        """Start timing"""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop timing and return duration"""
        if self.start_time is None:
            raise ValueError("Timer not started")
        
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time
        return self.duration
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        duration = self.stop()
        if get_config().get("logging.enable_performance_logging", False):
            logger.info(f"{self.name} took {duration:.4f} seconds")

def time_function(func: Callable = None, *, name: Optional[str] = None) -> Callable:
    """
    Decorator to time function execution
    
    Args:
        func: Function to decorate
        name: Custom name for the operation
        
    Returns:
        Decorated function
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            operation_name = name or f.__name__
            with PerformanceTimer(operation_name):
                return f(*args, **kwargs)
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)

class MemoryMonitor:
    """Memory usage monitoring utility"""
    
    def __init__(self, check_interval: float = 1.0):
        """
        Initialize memory monitor
        
        Args:
            check_interval: Interval between memory checks (seconds)
        """
        self.check_interval = check_interval
        self.monitoring = False
        self.memory_history = []
        self.peak_memory = 0
    
    def start_monitoring(self) -> None:
        """Start memory monitoring"""
        self.monitoring = True
        self.memory_history = []
        self.peak_memory = 0
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop memory monitoring and return statistics"""
        self.monitoring = False
        
        if not self.memory_history:
            return {"peak_memory_mb": 0, "average_memory_mb": 0, "samples": 0}
        
        peak_mb = max(self.memory_history) / (1024 * 1024)
        avg_mb = sum(self.memory_history) / len(self.memory_history) / (1024 * 1024)
        
        return {
            "peak_memory_mb": peak_mb,
            "average_memory_mb": avg_mb,
            "samples": len(self.memory_history),
            "history": [m / (1024 * 1024) for m in self.memory_history]
        }
    
    def check_memory(self) -> Dict[str, int]:
        """Check current memory usage"""
        try:
            memory_stats = get_memory_usage()
            current_memory = memory_stats.get("total_memory_bytes", 0)
            
            if self.monitoring:
                self.memory_history.append(current_memory)
                self.peak_memory = max(self.peak_memory, current_memory)
            
            return memory_stats
        except Exception as e:
            logger.warning(f"Failed to check memory usage: {e}")
            return {"total_memory_bytes": 0}

def check_system_compatibility() -> Dict[str, Any]:
    """Check system compatibility with TrustformeRS"""
    try:
        platform_info = get_platform_info()
        
        compatibility = {
            "compatible": True,
            "warnings": [],
            "errors": [],
            "recommendations": [],
            "platform_info": platform_info,
        }
        
        # Check architecture
        arch = platform_info.get("architecture", "unknown")
        if arch not in ["x86_64", "aarch64"]:
            compatibility["warnings"].append(f"Architecture {arch} may not be fully supported")
        
        # Check CPU cores
        cpu_cores = platform_info.get("cpu_cores", 1)
        if cpu_cores < 2:
            compatibility["warnings"].append("Single-core system may have limited performance")
        elif cpu_cores >= 8:
            compatibility["recommendations"].append("Consider enabling multi-threading for better performance")
        
        # Check memory
        memory_mb = platform_info.get("memory_mb", 0)
        if memory_mb < 2048:
            compatibility["warnings"].append("System memory below 2GB may limit model size")
        elif memory_mb >= 8192:
            compatibility["recommendations"].append("Sufficient memory for large models")
        
        # Check GPU
        has_gpu = platform_info.get("has_gpu", False)
        if has_gpu:
            compatibility["recommendations"].append("GPU acceleration available")
        else:
            compatibility["warnings"].append("No GPU detected - CPU-only inference")
        
        return compatibility
        
    except Exception as e:
        return {
            "compatible": False,
            "warnings": [],
            "errors": [f"Failed to check compatibility: {e}"],
            "recommendations": [],
            "platform_info": {},
        }

def validate_input_data(data: Union[np.ndarray, List, Any], 
                       expected_shape: Optional[tuple] = None,
                       expected_dtype: Optional[np.dtype] = None) -> np.ndarray:
    """
    Validate and convert input data to NumPy array
    
    Args:
        data: Input data to validate
        expected_shape: Expected shape (None to skip shape validation)
        expected_dtype: Expected dtype (None to skip dtype validation)
        
    Returns:
        Validated NumPy array
        
    Raises:
        ValueError: If validation fails
    """
    # Convert to numpy array if needed
    if not isinstance(data, np.ndarray):
        try:
            data = np.array(data)
        except Exception as e:
            raise ValueError(f"Cannot convert input to numpy array: {e}")
    
    # Validate shape
    if expected_shape is not None:
        if data.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {data.shape}")
    
    # Validate dtype
    if expected_dtype is not None:
        if data.dtype != expected_dtype:
            logger.warning(f"Converting from {data.dtype} to {expected_dtype}")
            data = data.astype(expected_dtype)
    
    return data

def format_memory_size(size_bytes: int) -> str:
    """Format memory size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information"""
    try:
        platform_info = get_platform_info()
        memory_stats = get_memory_usage()
        
        return {
            "platform": platform_info,
            "memory": {
                "current_usage": format_memory_size(memory_stats.get("total_memory_bytes", 0)),
                "peak_usage": format_memory_size(memory_stats.get("peak_memory_bytes", 0)),
                "allocated_objects": {
                    "models": memory_stats.get("allocated_models", 0),
                    "tokenizers": memory_stats.get("allocated_tokenizers", 0),
                    "pipelines": memory_stats.get("allocated_pipelines", 0),
                    "tensors": memory_stats.get("allocated_tensors", 0),
                }
            },
            "compatibility": check_system_compatibility(),
        }
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return {"error": str(e)}

def setup_logging(level: str = "INFO", 
                 enable_performance_logging: bool = False) -> None:
    """
    Setup logging for TrustformeRS
    
    Args:
        level: Logging level
        enable_performance_logging: Whether to enable performance logging
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Update configuration
    config = get_config()
    config.set("logging.level", level)
    config.set("logging.enable_performance_logging", enable_performance_logging)

def benchmark_operation(operation: Callable, 
                       *args, 
                       num_iterations: int = 10,
                       warmup_iterations: int = 2,
                       **kwargs) -> Dict[str, float]:
    """
    Benchmark an operation
    
    Args:
        operation: Function to benchmark
        *args: Arguments for the function
        num_iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations
        **kwargs: Keyword arguments for the function
        
    Returns:
        Benchmark results
    """
    # Warmup
    for _ in range(warmup_iterations):
        operation(*args, **kwargs)
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        operation(*args, **kwargs)
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    return {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "median_time": np.median(times),
        "iterations": num_iterations,
    }

class DebugContext:
    """Debug context manager for detailed error reporting"""
    
    def __init__(self, context_name: str):
        """
        Initialize debug context
        
        Args:
            context_name: Name of the debug context
        """
        self.context_name = context_name
        self.start_time = None
        self.memory_monitor = MemoryMonitor()
    
    def __enter__(self):
        """Enter debug context"""
        self.start_time = time.perf_counter()
        self.memory_monitor.start_monitoring()
        logger.debug(f"Entering debug context: {self.context_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit debug context"""
        duration = time.perf_counter() - self.start_time
        memory_stats = self.memory_monitor.stop_monitoring()
        
        if exc_type is not None:
            logger.error(f"Error in {self.context_name}: {exc_val}")
            logger.error(f"Context duration: {duration:.4f} seconds")
            logger.error(f"Peak memory: {memory_stats['peak_memory_mb']:.2f} MB")
        else:
            logger.debug(f"Completed {self.context_name} in {duration:.4f} seconds")
            logger.debug(f"Peak memory: {memory_stats['peak_memory_mb']:.2f} MB")

def create_example_config() -> Dict[str, Any]:
    """Create an example configuration file"""
    return {
        "library_path": "/usr/local/lib/libtrustformers_c.so",
        "auto_init": True,
        "performance": {
            "enable_simd": True,
            "enable_dynamic_batching": True,
            "enable_kernel_fusion": True,
            "max_batch_size": 64,
            "target_latency_ms": 50,
            "num_threads": 8,
        },
        "logging": {
            "level": "INFO",
            "enable_performance_logging": True,
        },
        "memory": {
            "enable_tracking": True,
            "warning_threshold_mb": 2048,
            "max_memory_mb": 8192,
        },
    }

def save_example_config(path: str) -> None:
    """Save example configuration to file"""
    config = create_example_config()
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Example configuration saved to {path}")

# Performance utilities
def get_performance_stats() -> Dict[str, Any]:
    """Get comprehensive performance statistics"""
    try:
        # This would integrate with the performance optimizer
        # For now, return basic stats
        return {
            "system_info": get_system_info(),
            "memory_usage": get_memory_usage(),
            "platform_info": get_platform_info(),
        }
    except Exception as e:
        logger.error(f"Failed to get performance stats: {e}")
        return {"error": str(e)}

def optimize_for_throughput() -> Dict[str, Any]:
    """Get configuration optimized for throughput"""
    return {
        "enable_simd": True,
        "enable_dynamic_batching": True,
        "enable_kernel_fusion": True,
        "max_batch_size": 128,
        "target_latency_ms": 200,
        "num_threads": 0,  # Auto-detect all cores
        "memory_bandwidth_level": 3,  # Maximum optimization
    }

def optimize_for_latency() -> Dict[str, Any]:
    """Get configuration optimized for latency"""
    return {
        "enable_simd": True,
        "enable_dynamic_batching": False,  # Disable batching for low latency
        "enable_kernel_fusion": True,
        "max_batch_size": 1,
        "target_latency_ms": 1,
        "num_threads": 1,  # Single thread for consistency
        "memory_bandwidth_level": 2,
    }

def optimize_for_memory() -> Dict[str, Any]:
    """Get configuration optimized for memory usage"""
    return {
        "enable_simd": False,  # Disable SIMD to save memory
        "enable_dynamic_batching": True,
        "enable_kernel_fusion": False,  # Disable fusion to save cache
        "max_batch_size": 16,
        "target_latency_ms": 500,
        "num_threads": 2,  # Limit threads to save memory
        "memory_bandwidth_level": 1,
    }