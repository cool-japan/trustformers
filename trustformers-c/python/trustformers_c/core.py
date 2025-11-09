"""
Core Python bindings for TrustformeRS-C using ctypes
"""

import ctypes
import json
import os
import sys
import platform
from typing import Optional, List, Dict, Any, Union
from enum import IntEnum
from pathlib import Path

# Error handling
class TrustformersError(Exception):
    """Base exception for TrustformeRS errors"""
    pass

class TrustformersErrorCode(IntEnum):
    """Error codes matching the C API"""
    SUCCESS = 0
    NULL_POINTER = 1
    INVALID_PARAMETER = 2
    RUNTIME_ERROR = 3
    SERIALIZATION_ERROR = 4
    MEMORY_ERROR = 5
    IO_ERROR = 6
    NOT_IMPLEMENTED = 7
    UNKNOWN_ERROR = 8

# C structures
class TrustformersMemoryUsage(ctypes.Structure):
    """Memory usage statistics"""
    _fields_ = [
        ("total_memory_bytes", ctypes.c_uint64),
        ("peak_memory_bytes", ctypes.c_uint64),
        ("allocated_models", ctypes.c_uint64),
        ("allocated_tokenizers", ctypes.c_uint64),
        ("allocated_pipelines", ctypes.c_uint64),
        ("allocated_tensors", ctypes.c_uint64),
    ]

class TrustformersBuildInfo(ctypes.Structure):
    """Build information structure"""
    _fields_ = [
        ("version", ctypes.c_char_p),
        ("features", ctypes.c_char_p),
        ("build_date", ctypes.c_char_p),
        ("target", ctypes.c_char_p),
    ]

class PerformanceConfig(ctypes.Structure):
    """Performance optimization configuration"""
    _fields_ = [
        ("enable_simd", ctypes.c_int),
        ("enable_dynamic_batching", ctypes.c_int),
        ("enable_kernel_fusion", ctypes.c_int),
        ("max_batch_size", ctypes.c_int),
        ("target_latency_ms", ctypes.c_int),
        ("memory_bandwidth_level", ctypes.c_int),
        ("enable_multithreading", ctypes.c_int),
        ("num_threads", ctypes.c_int),
        ("fusion_cache_size_mb", ctypes.c_int),
    ]

class PlatformInfo(ctypes.Structure):
    """Platform information"""
    _fields_ = [
        ("architecture", ctypes.c_char_p),
        ("operating_system", ctypes.c_char_p),
        ("cpu_cores", ctypes.c_int),
        ("has_gpu", ctypes.c_int),
        ("memory_mb", ctypes.c_int),
    ]

def find_library():
    """Find the TrustformeRS-C library"""
    lib_names = []
    
    if platform.system() == "Windows":
        lib_names = ["trustformers_c.dll", "libtrustformers_c.dll"]
    elif platform.system() == "Darwin":
        lib_names = ["libtrustformers_c.dylib", "libtrustformers_c.so"]
    else:
        lib_names = ["libtrustformers_c.so", "libtrustformers_c.a"]
    
    # Search paths
    search_paths = [
        # Current directory
        Path.cwd(),
        # Package directory
        Path(__file__).parent,
        # System library paths
        Path("/usr/local/lib"),
        Path("/usr/lib"),
        Path("/opt/local/lib"),
        # Build directory
        Path(__file__).parent.parent.parent / "target" / "release",
        Path(__file__).parent.parent.parent / "target" / "debug",
    ]
    
    # Add environment variable path
    if "TRUSTFORMERS_C_LIB_PATH" in os.environ:
        search_paths.insert(0, Path(os.environ["TRUSTFORMERS_C_LIB_PATH"]))
    
    for path in search_paths:
        if not path.exists():
            continue
            
        for lib_name in lib_names:
            lib_path = path / lib_name
            if lib_path.exists():
                return str(lib_path)
    
    raise TrustformersError(
        f"Could not find TrustformeRS-C library. "
        f"Searched for {lib_names} in {search_paths}. "
        f"Set TRUSTFORMERS_C_LIB_PATH environment variable to specify library location."
    )

# Load the library
_lib_path = find_library()
_lib = ctypes.CDLL(_lib_path)

# Define function signatures
def setup_function_signatures():
    """Setup ctypes function signatures for type safety"""
    
    # Initialization and cleanup
    _lib.trustformers_init.argtypes = []
    _lib.trustformers_init.restype = ctypes.c_int
    
    _lib.trustformers_cleanup.argtypes = []
    _lib.trustformers_cleanup.restype = ctypes.c_int
    
    # Version and build info
    _lib.trustformers_version.argtypes = []
    _lib.trustformers_version.restype = ctypes.c_char_p
    
    _lib.trustformers_build_info.argtypes = [ctypes.POINTER(TrustformersBuildInfo)]
    _lib.trustformers_build_info.restype = ctypes.c_int
    
    # Memory management
    _lib.trustformers_get_memory_usage.argtypes = [ctypes.POINTER(TrustformersMemoryUsage)]
    _lib.trustformers_get_memory_usage.restype = ctypes.c_int
    
    _lib.trustformers_memory_cleanup.argtypes = []
    _lib.trustformers_memory_cleanup.restype = ctypes.c_int
    
    # String management
    _lib.trustformers_free_string.argtypes = [ctypes.c_char_p]
    _lib.trustformers_free_string.restype = None
    
    # Platform information
    _lib.trustformers_get_platform_info.argtypes = []
    _lib.trustformers_get_platform_info.restype = ctypes.POINTER(PlatformInfo)
    
    _lib.trustformers_free_platform_info.argtypes = [ctypes.POINTER(PlatformInfo)]
    _lib.trustformers_free_platform_info.restype = None
    
    # Performance optimization
    _lib.trustformers_create_performance_optimizer.argtypes = [ctypes.POINTER(PerformanceConfig)]
    _lib.trustformers_create_performance_optimizer.restype = ctypes.c_void_p
    
    _lib.trustformers_destroy_performance_optimizer.argtypes = [ctypes.c_void_p]
    _lib.trustformers_destroy_performance_optimizer.restype = None
    
    _lib.trustformers_get_performance_config.argtypes = []
    _lib.trustformers_get_performance_config.restype = PerformanceConfig
    
    _lib.trustformers_get_performance_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char_p)]
    _lib.trustformers_get_performance_stats.restype = ctypes.c_int
    
    # Matrix operations
    _lib.trustformers_optimize_matrix_operations.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), 
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    _lib.trustformers_optimize_matrix_operations.restype = ctypes.c_int
    
    # Feature detection
    _lib.trustformers_has_feature.argtypes = [ctypes.c_char_p]
    _lib.trustformers_has_feature.restype = ctypes.c_int

setup_function_signatures()

def _check_error(error_code: int) -> None:
    """Check error code and raise exception if needed"""
    if error_code != TrustformersErrorCode.SUCCESS:
        error_names = {
            TrustformersErrorCode.NULL_POINTER: "Null pointer",
            TrustformersErrorCode.INVALID_PARAMETER: "Invalid parameter",
            TrustformersErrorCode.RUNTIME_ERROR: "Runtime error",
            TrustformersErrorCode.SERIALIZATION_ERROR: "Serialization error",
            TrustformersErrorCode.MEMORY_ERROR: "Memory error",
            TrustformersErrorCode.IO_ERROR: "I/O error",
            TrustformersErrorCode.NOT_IMPLEMENTED: "Not implemented",
            TrustformersErrorCode.UNKNOWN_ERROR: "Unknown error",
        }
        error_name = error_names.get(error_code, f"Unknown error ({error_code})")
        raise TrustformersError(f"TrustformeRS-C error: {error_name}")

def _free_c_string(ptr: ctypes.c_char_p) -> None:
    """Free a C string allocated by the library"""
    if ptr:
        _lib.trustformers_free_string(ptr)

class TrustformersC:
    """Main TrustformeRS-C interface"""
    
    def __init__(self, auto_init: bool = True):
        """Initialize TrustformeRS-C
        
        Args:
            auto_init: Whether to automatically initialize the library
        """
        self._initialized = False
        if auto_init:
            self.init()
    
    def init(self) -> None:
        """Initialize the TrustformeRS-C library"""
        if self._initialized:
            return
            
        error_code = _lib.trustformers_init()
        _check_error(error_code)
        self._initialized = True
    
    def cleanup(self) -> None:
        """Cleanup the TrustformeRS-C library"""
        if not self._initialized:
            return
            
        error_code = _lib.trustformers_cleanup()
        _check_error(error_code)
        self._initialized = False
    
    def __enter__(self):
        """Context manager entry"""
        self.init()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    def get_version(self) -> str:
        """Get library version"""
        version_ptr = _lib.trustformers_version()
        if version_ptr:
            return version_ptr.decode('utf-8')
        return "unknown"
    
    def get_build_info(self) -> Dict[str, str]:
        """Get build information"""
        build_info = TrustformersBuildInfo()
        error_code = _lib.trustformers_build_info(ctypes.byref(build_info))
        _check_error(error_code)
        
        try:
            return {
                "version": build_info.version.decode('utf-8') if build_info.version else "unknown",
                "features": build_info.features.decode('utf-8') if build_info.features else "",
                "build_date": build_info.build_date.decode('utf-8') if build_info.build_date else "unknown",
                "target": build_info.target.decode('utf-8') if build_info.target else "unknown",
            }
        finally:
            # Free allocated strings
            if build_info.version:
                _free_c_string(build_info.version)
            if build_info.features:
                _free_c_string(build_info.features)
            if build_info.build_date:
                _free_c_string(build_info.build_date)
            if build_info.target:
                _free_c_string(build_info.target)
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage statistics"""
        usage = TrustformersMemoryUsage()
        error_code = _lib.trustformers_get_memory_usage(ctypes.byref(usage))
        _check_error(error_code)
        
        return {
            "total_memory_bytes": usage.total_memory_bytes,
            "peak_memory_bytes": usage.peak_memory_bytes,
            "allocated_models": usage.allocated_models,
            "allocated_tokenizers": usage.allocated_tokenizers,
            "allocated_pipelines": usage.allocated_pipelines,
            "allocated_tensors": usage.allocated_tensors,
        }
    
    def memory_cleanup(self) -> None:
        """Force memory cleanup"""
        error_code = _lib.trustformers_memory_cleanup()
        _check_error(error_code)
    
    def has_feature(self, feature: str) -> bool:
        """Check if a feature is available"""
        feature_bytes = feature.encode('utf-8')
        result = _lib.trustformers_has_feature(feature_bytes)
        return result != 0
    
    def get_platform_info(self) -> Dict[str, Any]:
        """Get platform information"""
        info_ptr = _lib.trustformers_get_platform_info()
        if not info_ptr:
            raise TrustformersError("Failed to get platform info")
        
        try:
            info = info_ptr.contents
            return {
                "architecture": info.architecture.decode('utf-8') if info.architecture else "unknown",
                "operating_system": info.operating_system.decode('utf-8') if info.operating_system else "unknown",
                "cpu_cores": info.cpu_cores,
                "has_gpu": bool(info.has_gpu),
                "memory_mb": info.memory_mb,
            }
        finally:
            _lib.trustformers_free_platform_info(info_ptr)

class PerformanceOptimizer:
    """Performance optimization interface"""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        """Initialize performance optimizer
        
        Args:
            config: Performance configuration, uses default if None
        """
        if config is None:
            config = _lib.trustformers_get_performance_config()
        
        self._handle = _lib.trustformers_create_performance_optimizer(ctypes.byref(config))
        if not self._handle:
            raise TrustformersError("Failed to create performance optimizer")
    
    def __del__(self):
        """Cleanup performance optimizer"""
        if hasattr(self, '_handle') and self._handle:
            _lib.trustformers_destroy_performance_optimizer(self._handle)
            self._handle = None
    
    def optimize_matrix_operations(self, a, b, c, m: int, n: int, k: int) -> None:
        """Optimize matrix multiplication operations
        
        Args:
            a: Input matrix A (m x k)
            b: Input matrix B (k x n)
            c: Output matrix C (m x n)
            m: Number of rows in A
            n: Number of columns in B
            k: Number of columns in A / rows in B
        """
        # Convert to ctypes arrays
        a_array = (ctypes.c_float * (m * k))(*a.flatten())
        b_array = (ctypes.c_float * (k * n))(*b.flatten())
        c_array = (ctypes.c_float * (m * n))()
        
        error_code = _lib.trustformers_optimize_matrix_operations(
            self._handle,
            a_array,
            b_array,
            c_array,
            m, n, k
        )
        _check_error(error_code)
        
        # Copy results back
        for i in range(m * n):
            c.flat[i] = c_array[i]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats_ptr = ctypes.c_char_p()
        error_code = _lib.trustformers_get_performance_stats(
            self._handle,
            ctypes.byref(stats_ptr)
        )
        _check_error(error_code)
        
        try:
            if stats_ptr.value:
                stats_json = stats_ptr.value.decode('utf-8')
                return json.loads(stats_json)
            return {}
        finally:
            if stats_ptr.value:
                _free_c_string(stats_ptr)

# Global instance
_global_trustformers = None

def init_trustformers() -> TrustformersC:
    """Initialize global TrustformeRS-C instance"""
    global _global_trustformers
    if _global_trustformers is None:
        _global_trustformers = TrustformersC()
    return _global_trustformers

def cleanup_trustformers() -> None:
    """Cleanup global TrustformeRS-C instance"""
    global _global_trustformers
    if _global_trustformers is not None:
        _global_trustformers.cleanup()
        _global_trustformers = None

def get_version() -> str:
    """Get library version"""
    return init_trustformers().get_version()

def get_build_info() -> Dict[str, str]:
    """Get build information"""
    return init_trustformers().get_build_info()

def get_platform_info() -> Dict[str, Any]:
    """Get platform information"""
    return init_trustformers().get_platform_info()

def get_memory_usage() -> Dict[str, int]:
    """Get memory usage statistics"""
    return init_trustformers().get_memory_usage()

def has_feature(feature: str) -> bool:
    """Check if feature is available"""
    return init_trustformers().has_feature(feature)

# Performance optimization helpers
def enable_simd() -> PerformanceConfig:
    """Create performance config with SIMD enabled"""
    config = _lib.trustformers_get_performance_config()
    config.enable_simd = 1
    return config

def enable_dynamic_batching(max_batch_size: int = 32, target_latency_ms: int = 100) -> PerformanceConfig:
    """Create performance config with dynamic batching enabled"""
    config = _lib.trustformers_get_performance_config()
    config.enable_dynamic_batching = 1
    config.max_batch_size = max_batch_size
    config.target_latency_ms = target_latency_ms
    return config

def enable_kernel_fusion(cache_size_mb: int = 128) -> PerformanceConfig:
    """Create performance config with kernel fusion enabled"""
    config = _lib.trustformers_get_performance_config()
    config.enable_kernel_fusion = 1
    config.fusion_cache_size_mb = cache_size_mb
    return config

# Model and pipeline placeholders (to be implemented)
class Model:
    """Model interface placeholder"""
    pass

class Pipeline:
    """Pipeline interface placeholder"""
    pass

class Tokenizer:
    """Tokenizer interface placeholder"""
    pass