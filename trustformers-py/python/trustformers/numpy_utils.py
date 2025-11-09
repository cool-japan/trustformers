"""
NumPy conversion utilities for TrustformeRS

Provides seamless conversion between NumPy arrays and TrustformeRS tensors.
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Any, Dict
import warnings

from . import Tensor


def numpy_to_tensor(
    array: np.ndarray,
    dtype: Optional[np.dtype] = None,
    device: str = "cpu",
    requires_grad: bool = False,
) -> Tensor:
    """
    Convert NumPy array to TrustformeRS Tensor.
    
    Args:
        array: NumPy array to convert
        dtype: Target data type (default: float32)
        device: Device to place tensor on
        requires_grad: Whether tensor requires gradients
        
    Returns:
        TrustformeRS Tensor
    """
    # Ensure array is numpy array
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    
    # Convert to float32 by default
    if dtype is None:
        dtype = np.float32
    
    # Convert array to target dtype
    if array.dtype != dtype:
        array = array.astype(dtype)
    
    # Create tensor
    return Tensor(array, device=device, requires_grad=requires_grad)


def tensor_to_numpy(tensor: Tensor, copy: bool = True) -> np.ndarray:
    """
    Convert TrustformeRS Tensor to NumPy array.
    
    Args:
        tensor: TrustformeRS Tensor
        copy: If True, always copy data. If False, try zero-copy view when possible.
        
    Returns:
        NumPy array
    """
    if copy:
        return tensor.numpy()
    else:
        # Try zero-copy first, fallback to copy
        try:
            return tensor.numpy_view()
        except:
            return tensor.numpy()


def stack_numpy_arrays(arrays: List[np.ndarray], axis: int = 0) -> np.ndarray:
    """
    Stack list of NumPy arrays along a new axis.
    
    Args:
        arrays: List of arrays to stack
        axis: Axis along which to stack
        
    Returns:
        Stacked array
    """
    return np.stack(arrays, axis=axis)


def pad_sequence(
    sequences: List[np.ndarray],
    padding_value: float = 0.0,
    max_length: Optional[int] = None,
    padding_side: str = "right",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of sequences (numpy arrays)
        padding_value: Value to use for padding
        max_length: Maximum length to pad to
        padding_side: Side to pad on ("left" or "right")
        
    Returns:
        Tuple of (padded_sequences, attention_mask)
    """
    # Find max length
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded_sequences = []
    attention_masks = []
    
    for seq in sequences:
        seq_len = len(seq)
        
        if seq_len > max_length:
            # Truncate
            if padding_side == "right":
                seq = seq[:max_length]
            else:
                seq = seq[-max_length:]
            seq_len = max_length
        
        # Create attention mask
        attention_mask = np.ones(seq_len, dtype=np.int64)
        
        if seq_len < max_length:
            # Pad
            pad_width = max_length - seq_len
            if padding_side == "right":
                seq = np.pad(seq, (0, pad_width), constant_values=padding_value)
                attention_mask = np.pad(attention_mask, (0, pad_width), constant_values=0)
            else:
                seq = np.pad(seq, (pad_width, 0), constant_values=padding_value)
                attention_mask = np.pad(attention_mask, (pad_width, 0), constant_values=0)
        
        padded_sequences.append(seq)
        attention_masks.append(attention_mask)
    
    return np.stack(padded_sequences), np.stack(attention_masks)


def batch_convert_to_tensors(
    arrays: Union[List[np.ndarray], Dict[str, List[np.ndarray]]],
    device: str = "cpu",
) -> Union[List[Tensor], Dict[str, Tensor]]:
    """
    Convert a batch of NumPy arrays to TrustformeRS Tensors.
    
    Args:
        arrays: List or dict of NumPy arrays
        device: Device to place tensors on
        
    Returns:
        List or dict of Tensors
    """
    if isinstance(arrays, list):
        return [numpy_to_tensor(arr, device=device) for arr in arrays]
    elif isinstance(arrays, dict):
        return {key: numpy_to_tensor(arr, device=device) for key, arr in arrays.items()}
    else:
        raise TypeError(f"Expected list or dict, got {type(arrays)}")


def ensure_numpy_array(
    data: Union[np.ndarray, List, Tensor],
    dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """
    Ensure data is a NumPy array.
    
    Args:
        data: Input data (array, list, or Tensor)
        dtype: Target data type
        
    Returns:
        NumPy array
    """
    if isinstance(data, Tensor):
        array = data.numpy()
    elif isinstance(data, np.ndarray):
        array = data
    else:
        array = np.array(data)
    
    if dtype is not None and array.dtype != dtype:
        array = array.astype(dtype)
    
    return array


def create_attention_mask_from_lengths(
    lengths: List[int],
    max_length: Optional[int] = None,
) -> np.ndarray:
    """
    Create attention mask from sequence lengths.
    
    Args:
        lengths: List of sequence lengths
        max_length: Maximum length for the mask
        
    Returns:
        Attention mask array of shape (batch_size, max_length)
    """
    if max_length is None:
        max_length = max(lengths)
    
    batch_size = len(lengths)
    mask = np.zeros((batch_size, max_length), dtype=np.int64)
    
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    
    return mask


def apply_numpy_function(
    tensor: Tensor,
    func: callable,
    *args,
    **kwargs
) -> Tensor:
    """
    Apply a NumPy function to a Tensor.
    
    Args:
        tensor: Input tensor
        func: NumPy function to apply
        *args: Additional arguments for the function
        **kwargs: Additional keyword arguments
        
    Returns:
        Result as a Tensor
    """
    # Convert to numpy
    array = tensor.numpy()
    
    # Apply function
    result = func(array, *args, **kwargs)
    
    # Convert back to tensor
    return numpy_to_tensor(result, device=tensor.device)


def tensor_stats(tensor: Tensor) -> Dict[str, float]:
    """
    Compute statistics for a tensor using NumPy.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Dictionary with statistics
    """
    array = tensor.numpy()
    
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "median": float(np.median(array)),
        "sum": float(np.sum(array)),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


class NumpyTensorWrapper:
    """
    Wrapper class that provides NumPy-like interface for TrustformeRS Tensors.
    """
    
    def __init__(self, tensor: Tensor):
        self.tensor = tensor
        self._array = None
    
    @property
    def array(self) -> np.ndarray:
        """Get the underlying NumPy array (cached)."""
        if self._array is None:
            self._array = self.tensor.numpy()
        return self._array
    
    def __getattr__(self, name):
        """Forward NumPy method calls to the array."""
        if hasattr(np.ndarray, name):
            attr = getattr(self.array, name)
            if callable(attr):
                def method(*args, **kwargs):
                    result = attr(*args, **kwargs)
                    if isinstance(result, np.ndarray):
                        return numpy_to_tensor(result)
                    return result
                return method
            return attr
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __array__(self):
        """Support NumPy array interface."""
        return self.array
    
    def __repr__(self):
        return f"NumpyTensorWrapper({self.tensor})"


# Monkey-patch Tensor class to add numpy compatibility methods
def _tensor_to_numpy_compat(self) -> NumpyTensorWrapper:
    """Get NumPy-compatible wrapper for tensor."""
    return NumpyTensorWrapper(self)


# Try to add the method to Tensor if it's already imported
try:
    Tensor.as_numpy = _tensor_to_numpy_compat
except:
    pass


# Utility functions for common NumPy operations on Tensors
def concatenate(tensors: List[Tensor], axis: int = 0) -> Tensor:
    """Concatenate tensors along an axis."""
    arrays = [t.numpy() for t in tensors]
    result = np.concatenate(arrays, axis=axis)
    return numpy_to_tensor(result)


def split(tensor: Tensor, indices_or_sections: Union[int, List[int]], axis: int = 0) -> List[Tensor]:
    """Split tensor into multiple tensors."""
    array = tensor.numpy()
    arrays = np.split(array, indices_or_sections, axis=axis)
    return [numpy_to_tensor(arr) for arr in arrays]


def where(condition: Tensor, x: Union[Tensor, float], y: Union[Tensor, float]) -> Tensor:
    """NumPy-style where operation."""
    cond_array = condition.numpy()
    x_array = x.numpy() if isinstance(x, Tensor) else x
    y_array = y.numpy() if isinstance(y, Tensor) else y
    result = np.where(cond_array, x_array, y_array)
    return numpy_to_tensor(result)


# Zero-copy conversion utilities
def numpy_to_tensor_zero_copy(
    array: np.ndarray,
    dtype: Optional[np.dtype] = None,
    device: str = "cpu",
    requires_grad: bool = False,
) -> Tensor:
    """
    Convert NumPy array to TrustformeRS Tensor with zero-copy when possible.
    
    Args:
        array: NumPy array to convert
        dtype: Target data type (default: float32)
        device: Device to place tensor on
        requires_grad: Whether tensor requires gradients
        
    Returns:
        TrustformeRS Tensor
        
    Note:
        Zero-copy conversion is only possible for C-contiguous float32 arrays.
        Non-contiguous or different dtype arrays will be copied.
    """
    # Ensure array is numpy array
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    
    # Convert to float32 by default
    if dtype is None:
        dtype = np.float32
    
    # Convert array to target dtype if needed
    if array.dtype != dtype:
        array = array.astype(dtype)
    
    # Check if zero-copy is possible
    if array.dtype == np.float32 and array.flags.c_contiguous:
        # Zero-copy path
        try:
            return Tensor(array, device=device, requires_grad=requires_grad)
        except:
            # Fallback to regular conversion
            pass
    
    # Regular conversion (may copy)
    return numpy_to_tensor(array, dtype, device, requires_grad)


def tensor_to_numpy_zero_copy(tensor: Tensor) -> np.ndarray:
    """
    Convert TrustformeRS Tensor to NumPy array with zero-copy when possible.
    
    Args:
        tensor: TrustformeRS Tensor
        
    Returns:
        NumPy array (zero-copy view if possible, otherwise a copy)
    """
    return tensor_to_numpy(tensor, copy=False)


def is_zero_copy_possible(array: np.ndarray) -> bool:
    """
    Check if zero-copy conversion is possible for a NumPy array.
    
    Args:
        array: NumPy array to check
        
    Returns:
        True if zero-copy conversion is possible
    """
    return (
        isinstance(array, np.ndarray) and
        array.dtype == np.float32 and
        array.flags.c_contiguous
    )


def memory_usage_comparison(tensor: Tensor) -> Dict[str, Any]:
    """
    Compare memory usage between copy and zero-copy conversions.
    
    Args:
        tensor: TrustformeRS Tensor
        
    Returns:
        Dictionary with memory usage information
    """
    import psutil
    import os
    
    # Get current process
    process = psutil.Process(os.getpid())
    
    # Initial memory
    initial_memory = process.memory_info().rss
    
    # Test copy conversion
    copy_start = process.memory_info().rss
    arr_copy = tensor.numpy()
    copy_end = process.memory_info().rss
    copy_memory_delta = copy_end - copy_start
    
    # Test zero-copy conversion (if possible)
    zero_copy_start = process.memory_info().rss
    try:
        arr_zero_copy = tensor.numpy_view()
        zero_copy_end = process.memory_info().rss
        zero_copy_memory_delta = zero_copy_end - zero_copy_start
        zero_copy_possible = True
    except:
        zero_copy_memory_delta = 0
        zero_copy_possible = False
    
    # Get tensor memory info
    memory_info = tensor.memory_info()
    
    return {
        "tensor_bytes": memory_info["memory_bytes"],
        "copy_memory_delta_bytes": copy_memory_delta,
        "zero_copy_memory_delta_bytes": zero_copy_memory_delta,
        "zero_copy_possible": zero_copy_possible,
        "memory_savings_bytes": copy_memory_delta - zero_copy_memory_delta if zero_copy_possible else 0,
        "contiguous": memory_info["contiguous"],
        "shape": memory_info["shape"],
        "dtype": memory_info["dtype"],
    }


def optimize_array_for_zero_copy(array: np.ndarray) -> np.ndarray:
    """
    Optimize NumPy array for zero-copy conversion to TrustformeRS Tensor.
    
    Args:
        array: Input NumPy array
        
    Returns:
        Optimized array (possibly the same if already optimal)
    """
    # Convert to float32 if needed
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    
    # Make C-contiguous if needed
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    
    return array


class ZeroCopyTensorWrapper:
    """
    Wrapper that automatically chooses between copy and zero-copy conversions.
    """
    
    def __init__(self, prefer_zero_copy: bool = True):
        self.prefer_zero_copy = prefer_zero_copy
    
    def numpy_to_tensor(self, array: np.ndarray, **kwargs) -> Tensor:
        """Convert NumPy array to tensor with optimal memory usage."""
        if self.prefer_zero_copy and is_zero_copy_possible(array):
            return numpy_to_tensor_zero_copy(array, **kwargs)
        else:
            return numpy_to_tensor(array, **kwargs)
    
    def tensor_to_numpy(self, tensor: Tensor) -> np.ndarray:
        """Convert tensor to NumPy array with optimal memory usage."""
        if self.prefer_zero_copy and tensor.is_contiguous():
            return tensor_to_numpy_zero_copy(tensor)
        else:
            return tensor_to_numpy(tensor, copy=True)


# Global wrapper instance
zero_copy_converter = ZeroCopyTensorWrapper(prefer_zero_copy=True)


def enable_zero_copy_by_default():
    """Enable zero-copy conversions by default."""
    global zero_copy_converter
    zero_copy_converter = ZeroCopyTensorWrapper(prefer_zero_copy=True)


def disable_zero_copy_by_default():
    """Disable zero-copy conversions by default (safer but uses more memory)."""
    global zero_copy_converter  
    zero_copy_converter = ZeroCopyTensorWrapper(prefer_zero_copy=False)