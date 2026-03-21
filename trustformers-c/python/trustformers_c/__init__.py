"""
TrustformeRS Python bindings
High-performance transformer library with Python integration
"""

from .core import *
from .numpy_integration import *
from .async_support import *
from .utils import *

__version__ = "0.1.0"
__author__ = "Cool Japan"
__email__ = "info@cool-japan.com"
__description__ = "Python bindings for TrustformeRS transformer library"

__all__ = [
    # Core functionality
    "TrustformersC",
    "TrustformersError",
    "init_trustformers",
    "cleanup_trustformers",
    
    # Models and pipelines
    "Model",
    "Pipeline",
    "Tokenizer",
    
    # NumPy integration
    "NumpyTensor",
    "numpy_to_tensor",
    "tensor_to_numpy",
    
    # Async support
    "AsyncPipeline",
    "AsyncInference",
    "BatchProcessor",
    
    # Performance optimization
    "PerformanceConfig",
    "PerformanceOptimizer",
    "enable_simd",
    "enable_dynamic_batching",
    "enable_kernel_fusion",
    
    # Utilities
    "get_version",
    "get_build_info",
    "get_platform_info",
    "get_memory_usage",
    "get_performance_stats",
]