"""
TrustformeRS: Simplified Core Tensor Operations
===============================================

This is a simplified version that only exposes the core tensor operations that are currently working.
"""

__version__ = "0.1.0"

# Import only the working Rust extension components
try:
    from ._trustformers import (
        # Core classes that are available
        Tensor,
        TensorOptimized,

        # Utility functions
        get_device,
        set_seed,
        enable_grad,
        no_grad,
    )
except ImportError as e:
    raise ImportError(
        "Could not import TrustformeRS C extension. "
        "Make sure the package is properly installed. "
        f"Original error: {e}"
    )

# Define what gets imported with "from trustformers import *"
__all__ = [
    # Version
    "__version__",

    # Core
    "Tensor",
    "TensorOptimized",

    # Utilities
    "set_seed",
    "get_device",
    "enable_grad",
    "no_grad",
]