"""
Advanced NumPy utilities and operations for TrustformeRS.

Provides enhanced NumPy functionality including advanced indexing,
broadcasting, memory views, structured arrays, and high-performance
tensor operations.
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Sequence
import warnings
import numpy as np
from functools import wraps

from .utils import logging

logger = logging.get_logger(__name__)


class AdvancedArray:
    """
    Enhanced numpy array wrapper with advanced indexing and broadcasting support.
    
    Provides a more sophisticated interface for tensor operations while maintaining
    compatibility with standard numpy arrays.
    """
    
    def __init__(self, data: Union[np.ndarray, List, Tuple], dtype: Optional[np.dtype] = None):
        """
        Initialize AdvancedArray.
        
        Args:
            data: Input data (numpy array, list, or tuple)
            dtype: Target data type
        """
        if isinstance(data, np.ndarray):
            self._array = data.astype(dtype) if dtype is not None else data
        else:
            self._array = np.array(data, dtype=dtype)
        
        # Cache for memory views
        self._views = {}
    
    @property
    def array(self) -> np.ndarray:
        """Get underlying numpy array."""
        return self._array
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Array shape."""
        return self._array.shape
    
    @property
    def dtype(self) -> np.dtype:
        """Array data type."""
        return self._array.dtype
    
    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._array.ndim
    
    @property
    def size(self) -> int:
        """Total number of elements."""
        return self._array.size
    
    def __getitem__(self, key) -> "AdvancedArray":
        """Advanced indexing with broadcasting support."""
        result = self._advanced_getitem(key)
        return AdvancedArray(result)
    
    def __setitem__(self, key, value):
        """Advanced setting with broadcasting support."""
        self._advanced_setitem(key, value)
    
    def _advanced_getitem(self, key) -> np.ndarray:
        """Internal advanced indexing implementation."""
        if isinstance(key, tuple):
            # Multi-dimensional indexing
            return self._multi_index(key)
        elif isinstance(key, (list, np.ndarray)):
            # Fancy indexing
            return self._fancy_index(key)
        elif isinstance(key, slice):
            # Slice indexing
            return self._array[key]
        elif callable(key):
            # Function-based indexing
            mask = key(self._array)
            return self._array[mask]
        else:
            # Standard indexing
            return self._array[key]
    
    def _advanced_setitem(self, key, value):
        """Internal advanced setting implementation."""
        if isinstance(value, AdvancedArray):
            value = value.array
        
        if isinstance(key, tuple):
            # Multi-dimensional setting
            self._multi_set(key, value)
        elif isinstance(key, (list, np.ndarray)):
            # Fancy setting
            self._fancy_set(key, value)
        elif callable(key):
            # Function-based setting
            mask = key(self._array)
            self._array[mask] = value
        else:
            # Standard setting
            self._array[key] = value
    
    def _multi_index(self, indices: Tuple) -> np.ndarray:
        """Handle multi-dimensional indexing."""
        processed_indices = []
        
        for i, idx in enumerate(indices):
            if isinstance(idx, (list, np.ndarray)):
                # Convert to numpy array
                processed_indices.append(np.array(idx))
            elif callable(idx):
                # Function-based indexing for this dimension
                axis_size = self._array.shape[i] if i < self._array.ndim else 1
                axis_indices = np.arange(axis_size)
                mask = idx(axis_indices)
                processed_indices.append(axis_indices[mask])
            else:
                processed_indices.append(idx)
        
        return self._array[tuple(processed_indices)]
    
    def _multi_set(self, indices: Tuple, value):
        """Handle multi-dimensional setting."""
        processed_indices = []
        
        for i, idx in enumerate(indices):
            if isinstance(idx, (list, np.ndarray)):
                processed_indices.append(np.array(idx))
            elif callable(idx):
                axis_size = self._array.shape[i] if i < self._array.ndim else 1
                axis_indices = np.arange(axis_size)
                mask = idx(axis_indices)
                processed_indices.append(axis_indices[mask])
            else:
                processed_indices.append(idx)
        
        self._array[tuple(processed_indices)] = value
    
    def _fancy_index(self, indices: Union[List, np.ndarray]) -> np.ndarray:
        """Handle fancy indexing."""
        indices = np.array(indices)
        
        # Handle negative indices
        indices = np.where(indices < 0, indices + self._array.shape[0], indices)
        
        # Bounds checking
        if np.any(indices >= self._array.shape[0]) or np.any(indices < 0):
            raise IndexError("Index out of bounds")
        
        return self._array[indices]
    
    def _fancy_set(self, indices: Union[List, np.ndarray], value):
        """Handle fancy setting."""
        indices = np.array(indices)
        indices = np.where(indices < 0, indices + self._array.shape[0], indices)
        
        if np.any(indices >= self._array.shape[0]) or np.any(indices < 0):
            raise IndexError("Index out of bounds")
        
        self._array[indices] = value
    
    def view(self, dtype: Optional[np.dtype] = None, shape: Optional[Tuple[int, ...]] = None) -> "AdvancedArray":
        """
        Create a memory view of the array.
        
        Args:
            dtype: New data type for the view
            shape: New shape for the view
            
        Returns:
            New AdvancedArray view
        """
        cache_key = (dtype, shape)
        
        if cache_key not in self._views:
            if dtype is not None and shape is not None:
                view = self._array.view(dtype).reshape(shape)
            elif dtype is not None:
                view = self._array.view(dtype)
            elif shape is not None:
                view = self._array.reshape(shape)
            else:
                view = self._array.view()
            
            self._views[cache_key] = AdvancedArray(view)
        
        return self._views[cache_key]
    
    def broadcast_to(self, shape: Tuple[int, ...]) -> "AdvancedArray":
        """
        Broadcast array to new shape.
        
        Args:
            shape: Target shape
            
        Returns:
            Broadcasted array
        """
        broadcasted = np.broadcast_to(self._array, shape)
        return AdvancedArray(broadcasted)
    
    def broadcast_with(self, other: Union["AdvancedArray", np.ndarray]) -> Tuple["AdvancedArray", "AdvancedArray"]:
        """
        Broadcast this array with another.
        
        Args:
            other: Other array to broadcast with
            
        Returns:
            Tuple of broadcasted arrays
        """
        other_array = other.array if isinstance(other, AdvancedArray) else other
        
        try:
            broadcasted_self, broadcasted_other = np.broadcast_arrays(self._array, other_array)
            return AdvancedArray(broadcasted_self), AdvancedArray(broadcasted_other)
        except ValueError as e:
            raise ValueError(f"Cannot broadcast arrays with shapes {self.shape} and {other_array.shape}: {e}")
    
    def advanced_slice(self, *slices, step_function: Optional[callable] = None) -> "AdvancedArray":
        """
        Advanced slicing with custom step functions.
        
        Args:
            *slices: Slice objects or indices
            step_function: Custom function to determine step sizes
            
        Returns:
            Sliced array
        """
        if step_function is not None:
            # Custom step function slicing
            indices = []
            for i, s in enumerate(slices):
                if isinstance(s, slice):
                    start, stop, step = s.indices(self._array.shape[i])
                    custom_indices = []
                    current = start
                    while current < stop:
                        custom_indices.append(current)
                        step_size = step_function(current, i) if callable(step_function) else step
                        current += step_size
                    indices.append(custom_indices)
                else:
                    indices.append([s])
            
            # Use fancy indexing with custom indices
            return self._multi_dimensional_fancy_index(indices)
        else:
            # Standard slicing
            return AdvancedArray(self._array[slices])
    
    def _multi_dimensional_fancy_index(self, indices_list: List[List[int]]) -> "AdvancedArray":
        """Multi-dimensional fancy indexing."""
        if len(indices_list) == 1:
            return AdvancedArray(self._array[indices_list[0]])
        
        # Create meshgrid for multi-dimensional indexing
        mesh = np.meshgrid(*indices_list, indexing='ij')
        return AdvancedArray(self._array[tuple(mesh)])
    
    def where(self, condition: Union[callable, np.ndarray], x: Any = None, y: Any = None) -> "AdvancedArray":
        """
        Advanced where operation.
        
        Args:
            condition: Condition function or boolean array
            x: Values to use where condition is True
            y: Values to use where condition is False
            
        Returns:
            Result array
        """
        if callable(condition):
            mask = condition(self._array)
        else:
            mask = condition
        
        if x is None and y is None:
            # Return indices where condition is True
            indices = np.where(mask)
            return AdvancedArray(indices)
        else:
            # Return values based on condition
            result = np.where(mask, x, y)
            return AdvancedArray(result)
    
    def roll_axis(self, axis: int, shift: int) -> "AdvancedArray":
        """Roll elements along specified axis."""
        result = np.roll(self._array, shift, axis=axis)
        return AdvancedArray(result)
    
    def compress(self, condition: Union[List[bool], np.ndarray], axis: Optional[int] = None) -> "AdvancedArray":
        """Compress array based on condition."""
        result = np.compress(condition, self._array, axis=axis)
        return AdvancedArray(result)
    
    def take_along_axis(self, indices: np.ndarray, axis: int) -> "AdvancedArray":
        """Take values along axis using indices."""
        result = np.take_along_axis(self._array, indices, axis=axis)
        return AdvancedArray(result)
    
    def put_along_axis(self, indices: np.ndarray, values: Any, axis: int):
        """Put values along axis using indices."""
        np.put_along_axis(self._array, indices, values, axis=axis)
    
    def memory_layout_info(self) -> Dict[str, Any]:
        """Get information about memory layout."""
        return {
            'c_contiguous': self._array.flags.c_contiguous,
            'f_contiguous': self._array.flags.f_contiguous,
            'aligned': self._array.flags.aligned,
            'writeable': self._array.flags.writeable,
            'owndata': self._array.flags.owndata,
            'strides': self._array.strides,
            'itemsize': self._array.itemsize,
            'nbytes': self._array.nbytes
        }
    
    def optimize_memory_layout(self, order: str = 'C') -> "AdvancedArray":
        """
        Optimize memory layout for better cache performance.
        
        Args:
            order: Memory order ('C' for row-major, 'F' for column-major)
            
        Returns:
            Array with optimized layout
        """
        if order == 'C' and not self._array.flags.c_contiguous:
            optimized = np.ascontiguousarray(self._array)
            return AdvancedArray(optimized)
        elif order == 'F' and not self._array.flags.f_contiguous:
            optimized = np.asfortranarray(self._array)
            return AdvancedArray(optimized)
        else:
            return self
    
    def __array__(self) -> np.ndarray:
        """NumPy array interface."""
        return self._array
    
    def __repr__(self) -> str:
        return f"AdvancedArray({self._array})"


class StructuredArrays:
    """
    Utilities for working with structured numpy arrays.
    """
    
    @staticmethod
    def create_structured_array(
        names: List[str],
        formats: List[str],
        data: Optional[List[Tuple]] = None
    ) -> np.ndarray:
        """
        Create a structured array.
        
        Args:
            names: Field names
            formats: Field data types
            data: Optional data to initialize with
            
        Returns:
            Structured numpy array
        """
        dtype = np.dtype(list(zip(names, formats)))
        
        if data is not None:
            return np.array(data, dtype=dtype)
        else:
            return np.empty(0, dtype=dtype)
    
    @staticmethod
    def add_field(
        arr: np.ndarray,
        name: str,
        dtype: str,
        data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Add a field to structured array.
        
        Args:
            arr: Existing structured array
            name: New field name
            dtype: New field data type
            data: Data for new field
            
        Returns:
            New structured array with added field
        """
        # Create new dtype
        new_dtype = arr.dtype.descr + [(name, dtype)]
        new_arr = np.empty(arr.shape, dtype=new_dtype)
        
        # Copy existing data
        for field_name in arr.dtype.names:
            new_arr[field_name] = arr[field_name]
        
        # Add new field data
        if data is not None:
            new_arr[name] = data
        
        return new_arr
    
    @staticmethod
    def remove_field(arr: np.ndarray, field_name: str) -> np.ndarray:
        """
        Remove a field from structured array.
        
        Args:
            arr: Structured array
            field_name: Field to remove
            
        Returns:
            New structured array without the field
        """
        new_dtype = [desc for desc in arr.dtype.descr if desc[0] != field_name]
        new_arr = np.empty(arr.shape, dtype=new_dtype)
        
        for name, _ in new_dtype:
            new_arr[name] = arr[name]
        
        return new_arr
    
    @staticmethod
    def reorder_fields(arr: np.ndarray, field_order: List[str]) -> np.ndarray:
        """
        Reorder fields in structured array.
        
        Args:
            arr: Structured array
            field_order: New field order
            
        Returns:
            Reordered structured array
        """
        new_dtype = [(name, arr.dtype.fields[name][0]) for name in field_order]
        new_arr = np.empty(arr.shape, dtype=new_dtype)
        
        for name in field_order:
            new_arr[name] = arr[name]
        
        return new_arr


class BroadcastingUtils:
    """
    Advanced broadcasting utilities.
    """
    
    @staticmethod
    def safe_broadcast(*arrays: np.ndarray) -> List[np.ndarray]:
        """
        Safely broadcast arrays with detailed error messages.
        
        Args:
            *arrays: Arrays to broadcast
            
        Returns:
            List of broadcasted arrays
        """
        try:
            return list(np.broadcast_arrays(*arrays))
        except ValueError as e:
            shapes = [arr.shape for arr in arrays]
            raise ValueError(f"Cannot broadcast arrays with shapes {shapes}: {e}")
    
    @staticmethod
    def broadcast_shape(*shapes: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Compute the broadcast shape for given shapes.
        
        Args:
            *shapes: Input shapes
            
        Returns:
            Broadcast shape
        """
        # Reverse shapes for easier computation
        rev_shapes = [shape[::-1] for shape in shapes]
        max_len = max(len(shape) for shape in rev_shapes)
        
        # Pad with 1s
        padded_shapes = []
        for shape in rev_shapes:
            padded = shape + (1,) * (max_len - len(shape))
            padded_shapes.append(padded)
        
        # Compute broadcast shape
        broadcast_shape = []
        for i in range(max_len):
            dims = [shape[i] for shape in padded_shapes]
            max_dim = max(dims)
            
            # Check compatibility
            for dim in dims:
                if dim != 1 and dim != max_dim:
                    raise ValueError(f"Cannot broadcast dimensions {dims}")
            
            broadcast_shape.append(max_dim)
        
        return tuple(broadcast_shape[::-1])
    
    @staticmethod
    def broadcast_compatible(*shapes: Tuple[int, ...]) -> bool:
        """
        Check if shapes are broadcast compatible.
        
        Args:
            *shapes: Shapes to check
            
        Returns:
            True if compatible, False otherwise
        """
        try:
            BroadcastingUtils.broadcast_shape(*shapes)
            return True
        except ValueError:
            return False


def memory_efficient_operation(func):
    """
    Decorator for memory-efficient array operations.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check memory usage before operation
        input_memory = sum(
            arg.nbytes for arg in args 
            if isinstance(arg, (np.ndarray, AdvancedArray))
        )
        
        if input_memory > 1e9:  # 1GB threshold
            logger.warning(f"Large memory operation detected: {input_memory / 1e9:.2f} GB")
        
        # Perform operation
        result = func(*args, **kwargs)
        
        # Log memory usage
        if hasattr(result, 'nbytes'):
            logger.debug(f"Operation memory usage: {result.nbytes / 1e6:.2f} MB")
        
        return result
    
    return wrapper


@memory_efficient_operation
def advanced_concatenate(
    arrays: List[Union[np.ndarray, AdvancedArray]],
    axis: int = 0,
    ensure_contiguous: bool = True
) -> AdvancedArray:
    """
    Advanced concatenation with memory optimization.
    
    Args:
        arrays: Arrays to concatenate
        axis: Concatenation axis
        ensure_contiguous: Whether to ensure result is contiguous
        
    Returns:
        Concatenated AdvancedArray
    """
    # Convert to numpy arrays
    np_arrays = []
    for arr in arrays:
        if isinstance(arr, AdvancedArray):
            np_arrays.append(arr.array)
        else:
            np_arrays.append(arr)
    
    # Concatenate
    result = np.concatenate(np_arrays, axis=axis)
    
    # Ensure contiguous if requested
    if ensure_contiguous and not result.flags.c_contiguous:
        result = np.ascontiguousarray(result)
    
    return AdvancedArray(result)


@memory_efficient_operation
def advanced_stack(
    arrays: List[Union[np.ndarray, AdvancedArray]],
    axis: int = 0,
    dtype: Optional[np.dtype] = None
) -> AdvancedArray:
    """
    Advanced stacking with type promotion.
    
    Args:
        arrays: Arrays to stack
        axis: Stacking axis
        dtype: Target data type
        
    Returns:
        Stacked AdvancedArray
    """
    # Convert to numpy arrays
    np_arrays = []
    for arr in arrays:
        if isinstance(arr, AdvancedArray):
            np_arrays.append(arr.array)
        else:
            np_arrays.append(arr)
    
    # Determine common dtype if not specified
    if dtype is None:
        dtype = np.find_common_type([arr.dtype for arr in np_arrays], [])
    
    # Convert arrays to common dtype
    converted_arrays = [arr.astype(dtype) for arr in np_arrays]
    
    # Stack
    result = np.stack(converted_arrays, axis=axis)
    
    return AdvancedArray(result)


# Export main classes and functions
__all__ = [
    'AdvancedArray',
    'StructuredArrays',
    'BroadcastingUtils',
    'memory_efficient_operation',
    'advanced_concatenate',
    'advanced_stack'
]