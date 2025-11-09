"""
JAX integration utilities for TrustformeRS.

Provides seamless conversion between TrustformeRS tensors and JAX arrays,
enabling easy integration with JAX-based workflows.
"""

from typing import Any, Dict, List, Optional, Union, Callable
import warnings
import numpy as np
from .utils import logging

try:
    import jax
    import jax.numpy as jnp
    from jax import Array as JAXArray
    _jax_available = True
except ImportError:
    _jax_available = False
    JAXArray = None

logger = logging.get_logger(__name__)


def jax_available() -> bool:
    """Check if JAX is available."""
    return _jax_available


def jax_to_tensor(jax_array: 'JAXArray') -> 'Tensor':
    """
    Convert JAX array to TrustformeRS tensor.
    
    Args:
        jax_array: JAX array to convert
        
    Returns:
        TrustformeRS Tensor
        
    Raises:
        ImportError: If JAX is not available
        ValueError: If input is not a JAX array
    """
    if not _jax_available:
        raise ImportError("JAX is not available. Install with: pip install jax")
    
    if not isinstance(jax_array, JAXArray):
        raise ValueError(f"Expected JAX array, got {type(jax_array)}")
    
    # Convert JAX array to numpy (this brings it to host memory)
    numpy_array = np.array(jax_array, dtype=np.float32)
    
    # Import here to avoid circular imports
    from . import Tensor
    return Tensor(numpy_array)


def tensor_to_jax(tensor: 'Tensor') -> 'JAXArray':
    """
    Convert TrustformeRS tensor to JAX array.
    
    Args:
        tensor: TrustformeRS tensor to convert
        
    Returns:
        JAX array
        
    Raises:
        ImportError: If JAX is not available
    """
    if not _jax_available:
        raise ImportError("JAX is not available. Install with: pip install jax")
    
    # Convert tensor to numpy first
    numpy_array = tensor.numpy()
    
    # Convert to JAX array
    return jnp.array(numpy_array)


def batch_jax_to_tensor(jax_arrays: Union[List['JAXArray'], Dict[str, 'JAXArray']]) -> Union[List['Tensor'], Dict[str, 'Tensor']]:
    """
    Convert a batch of JAX arrays to TrustformeRS tensors.
    
    Args:
        jax_arrays: List or dict of JAX arrays
        
    Returns:
        List or dict of TrustformeRS tensors
    """
    if isinstance(jax_arrays, dict):
        return {key: jax_to_tensor(array) for key, array in jax_arrays.items()}
    elif isinstance(jax_arrays, list):
        return [jax_to_tensor(array) for array in jax_arrays]
    else:
        raise ValueError("Input must be a list or dict of JAX arrays")


def batch_tensor_to_jax(tensors: Union[List['Tensor'], Dict[str, 'Tensor']]) -> Union[List['JAXArray'], Dict[str, 'JAXArray']]:
    """
    Convert a batch of TrustformeRS tensors to JAX arrays.
    
    Args:
        tensors: List or dict of TrustformeRS tensors
        
    Returns:
        List or dict of JAX arrays
    """
    if isinstance(tensors, dict):
        return {key: tensor_to_jax(tensor) for key, tensor in tensors.items()}
    elif isinstance(tensors, list):
        return [tensor_to_jax(tensor) for tensor in tensors]
    else:
        raise ValueError("Input must be a list or dict of tensors")


def ensure_jax_array(data: Any) -> 'JAXArray':
    """
    Ensure the input is a JAX array, converting if necessary.
    
    Args:
        data: Input data (JAX array, numpy array, list, or TrustformeRS tensor)
        
    Returns:
        JAX array
    """
    if not _jax_available:
        raise ImportError("JAX is not available. Install with: pip install jax")
    
    if isinstance(data, JAXArray):
        return data
    elif isinstance(data, np.ndarray):
        return jnp.array(data)
    elif isinstance(data, (list, tuple)):
        return jnp.array(data)
    else:
        # Assume it's a TrustformeRS tensor
        try:
            return tensor_to_jax(data)
        except:
            # Fallback: convert to numpy first
            numpy_array = np.array(data, dtype=np.float32)
            return jnp.array(numpy_array)


class JAXTensorWrapper:
    """
    Wrapper class for seamless JAX-TrustformeRS tensor operations.
    
    This wrapper automatically converts between JAX arrays and TrustformeRS tensors
    based on the operation context.
    """
    
    def __init__(self, prefer_jax: bool = True):
        """
        Initialize the wrapper.
        
        Args:
            prefer_jax: If True, prefer JAX arrays for output when possible
        """
        if not _jax_available:
            raise ImportError("JAX is not available. Install with: pip install jax")
        
        self.prefer_jax = prefer_jax
    
    def jax_to_tensor(self, jax_array: 'JAXArray') -> 'Tensor':
        """Convert JAX array to tensor with wrapper context."""
        return jax_to_tensor(jax_array)
    
    def tensor_to_jax(self, tensor: 'Tensor') -> 'JAXArray':
        """Convert tensor to JAX array with wrapper context."""
        return tensor_to_jax(tensor)
    
    def auto_convert(self, data: Any) -> Union['JAXArray', 'Tensor']:
        """
        Automatically convert data to the preferred format.
        
        Args:
            data: Input data
            
        Returns:
            Converted data in preferred format
        """
        if self.prefer_jax:
            return ensure_jax_array(data)
        else:
            # Convert to tensor
            if isinstance(data, JAXArray):
                return jax_to_tensor(data)
            else:
                from . import Tensor
                return Tensor(data)


# JAX-specific model utilities
def create_jax_dataloader(
    jax_arrays: Union[List['JAXArray'], Dict[str, 'JAXArray']],
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = False
) -> Callable:
    """
    Create a JAX-compatible data loader.
    
    Args:
        jax_arrays: JAX arrays to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        drop_last: Whether to drop last incomplete batch
        
    Returns:
        Data loader function
    """
    if not _jax_available:
        raise ImportError("JAX is not available. Install with: pip install jax")
    
    if isinstance(jax_arrays, dict):
        # Dict of arrays - assume all have same first dimension
        first_key = list(jax_arrays.keys())[0]
        num_samples = jax_arrays[first_key].shape[0]
        
        def dataloader():
            # Create indices
            indices = jnp.arange(num_samples)
            if shuffle:
                key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
                indices = jax.random.permutation(key, indices)
            
            # Create batches
            num_batches = num_samples // batch_size
            if not drop_last and num_samples % batch_size != 0:
                num_batches += 1
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch = {}
                for key, array in jax_arrays.items():
                    batch[key] = array[batch_indices]
                
                yield batch
        
        return dataloader
    
    elif isinstance(jax_arrays, list):
        # List of arrays
        num_samples = jax_arrays[0].shape[0]
        
        def dataloader():
            indices = jnp.arange(num_samples)
            if shuffle:
                key = jax.random.PRNGKey(42)
                indices = jax.random.permutation(key, indices)
            
            num_batches = num_samples // batch_size
            if not drop_last and num_samples % batch_size != 0:
                num_batches += 1
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch = [array[batch_indices] for array in jax_arrays]
                yield batch
        
        return dataloader
    
    else:
        raise ValueError("Input must be a list or dict of JAX arrays")


def jax_model_apply(model_fn: Callable, params: Dict, inputs: 'JAXArray') -> 'JAXArray':
    """
    Apply a JAX model function with automatic differentiation support.
    
    Args:
        model_fn: JAX model function
        params: Model parameters
        inputs: Input JAX array
        
    Returns:
        Model outputs
    """
    if not _jax_available:
        raise ImportError("JAX is not available. Install with: pip install jax")
    
    return model_fn(params, inputs)


def jax_grad_fn(loss_fn: Callable) -> Callable:
    """
    Create a gradient function for JAX optimization.
    
    Args:
        loss_fn: Loss function to compute gradients for
        
    Returns:
        Gradient function
    """
    if not _jax_available:
        raise ImportError("JAX is not available. Install with: pip install jax")
    
    return jax.grad(loss_fn)


def jax_vmap_fn(fn: Callable, in_axes: Union[int, tuple] = 0) -> Callable:
    """
    Create a vectorized JAX function.
    
    Args:
        fn: Function to vectorize
        in_axes: Input axes to vectorize over
        
    Returns:
        Vectorized function
    """
    if not _jax_available:
        raise ImportError("JAX is not available. Install with: pip install jax")
    
    return jax.vmap(fn, in_axes=in_axes)


def jax_jit_fn(fn: Callable) -> Callable:
    """
    JIT compile a JAX function for performance.
    
    Args:
        fn: Function to JIT compile
        
    Returns:
        JIT compiled function
    """
    if not _jax_available:
        raise ImportError("JAX is not available. Install with: pip install jax")
    
    return jax.jit(fn)


# Integration with existing TrustformeRS components
def create_jax_pipeline(
    model: Any,
    task: str = "text-classification",
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a JAX-compatible pipeline.
    
    Args:
        model: TrustformeRS model
        task: Task type
        device: JAX device (if applicable)
        
    Returns:
        JAX pipeline configuration
    """
    if not _jax_available:
        raise ImportError("JAX is not available. Install with: pip install jax")
    
    # This would integrate with the main pipeline system
    # For now, return a basic configuration
    return {
        "model": model,
        "task": task,
        "framework": "jax",
        "device": device,
        "convert_inputs": lambda x: ensure_jax_array(x),
        "convert_outputs": lambda x: tensor_to_jax(x) if hasattr(x, 'numpy') else x
    }


# Utility functions for JAX ecosystem integration
def setup_jax_config():
    """Setup recommended JAX configuration for TrustformeRS."""
    if not _jax_available:
        logger.warning("JAX is not available")
        return
    
    # Enable 64-bit precision if needed
    # jax.config.update("jax_enable_x64", True)
    
    # Set memory preallocation
    # jax.config.update("jax_platform_name", "cpu")  # or "gpu"
    
    logger.info("JAX configuration set up for TrustformeRS")


# Export main functions
__all__ = [
    'jax_available',
    'jax_to_tensor', 
    'tensor_to_jax',
    'batch_jax_to_tensor',
    'batch_tensor_to_jax', 
    'ensure_jax_array',
    'JAXTensorWrapper',
    'create_jax_dataloader',
    'jax_model_apply',
    'jax_grad_fn',
    'jax_vmap_fn', 
    'jax_jit_fn',
    'create_jax_pipeline',
    'setup_jax_config'
]