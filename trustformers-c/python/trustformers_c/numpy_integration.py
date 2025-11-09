"""
NumPy integration for TrustformeRS-C
Provides seamless conversion between NumPy arrays and TrustformeRS tensors
"""

import ctypes
import numpy as np
from typing import Union, Optional, List, Tuple, Any
from .core import TrustformersError, _lib, _check_error, _free_c_string

# NumPy dtype mappings
NUMPY_TO_C_DTYPE = {
    np.float32: ctypes.c_float,
    np.float64: ctypes.c_double,
    np.int32: ctypes.c_int32,
    np.int64: ctypes.c_int64,
    np.uint32: ctypes.c_uint32,
    np.uint64: ctypes.c_uint64,
    np.bool_: ctypes.c_bool,
    np.int8: ctypes.c_int8,
    np.uint8: ctypes.c_uint8,
    np.int16: ctypes.c_int16,
    np.uint16: ctypes.c_uint16,
}

C_TO_NUMPY_DTYPE = {v: k for k, v in NUMPY_TO_C_DTYPE.items()}

class NumpyTensor:
    """
    A wrapper class that provides seamless integration between NumPy arrays
    and TrustformeRS tensors with zero-copy operations where possible.
    """
    
    def __init__(self, array: np.ndarray, copy: bool = False):
        """
        Initialize NumpyTensor from NumPy array
        
        Args:
            array: NumPy array to wrap
            copy: Whether to copy the array data (default: False for zero-copy)
        """
        if not isinstance(array, np.ndarray):
            raise TypeError("Input must be a NumPy array")
        
        if copy:
            self._array = array.copy()
        else:
            self._array = array
            
        self._c_handle = None
        self._setup_c_interface()
    
    def _setup_c_interface(self) -> None:
        """Setup C interface for the tensor"""
        # Ensure array is contiguous
        if not self._array.flags.c_contiguous:
            self._array = np.ascontiguousarray(self._array)
        
        # Get C data type
        if self._array.dtype.type not in NUMPY_TO_C_DTYPE:
            raise TrustformersError(f"Unsupported dtype: {self._array.dtype}")
        
        self._c_dtype = NUMPY_TO_C_DTYPE[self._array.dtype.type]
        
        # Get data pointer
        self._data_ptr = self._array.ctypes.data_as(ctypes.POINTER(self._c_dtype))
        
        # Store shape and strides
        self._shape = self._array.shape
        self._strides = self._array.strides
        self._ndim = self._array.ndim
        self._size = self._array.size
    
    @property
    def array(self) -> np.ndarray:
        """Get the underlying NumPy array"""
        return self._array
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get tensor shape"""
        return self._shape
    
    @property
    def dtype(self) -> np.dtype:
        """Get tensor dtype"""
        return self._array.dtype
    
    @property
    def size(self) -> int:
        """Get tensor size (total number of elements)"""
        return self._size
    
    @property
    def ndim(self) -> int:
        """Get number of dimensions"""
        return self._ndim
    
    @property
    def data_ptr(self) -> ctypes.POINTER:
        """Get C data pointer"""
        return self._data_ptr
    
    def reshape(self, shape: Tuple[int, ...]) -> 'NumpyTensor':
        """Reshape tensor (returns new NumpyTensor)"""
        reshaped_array = self._array.reshape(shape)
        return NumpyTensor(reshaped_array, copy=False)
    
    def transpose(self, axes: Optional[Tuple[int, ...]] = None) -> 'NumpyTensor':
        """Transpose tensor (returns new NumpyTensor)"""
        transposed_array = self._array.transpose(axes)
        return NumpyTensor(transposed_array, copy=False)
    
    def copy(self) -> 'NumpyTensor':
        """Create a copy of the tensor"""
        return NumpyTensor(self._array.copy(), copy=False)
    
    def to_numpy(self) -> np.ndarray:
        """Convert to NumPy array (returns view if possible)"""
        return self._array
    
    def __array__(self) -> np.ndarray:
        """NumPy array interface"""
        return self._array
    
    def __getitem__(self, key) -> 'NumpyTensor':
        """Index into tensor"""
        return NumpyTensor(self._array[key], copy=False)
    
    def __setitem__(self, key, value) -> None:
        """Set tensor values"""
        self._array[key] = value
    
    def __str__(self) -> str:
        """String representation"""
        return f"NumpyTensor(shape={self.shape}, dtype={self.dtype})\n{self._array}"
    
    def __repr__(self) -> str:
        """String representation"""
        return f"NumpyTensor(shape={self.shape}, dtype={self.dtype})"

def numpy_to_tensor(array: np.ndarray, copy: bool = False) -> NumpyTensor:
    """
    Convert NumPy array to NumpyTensor
    
    Args:
        array: NumPy array to convert
        copy: Whether to copy the array data
        
    Returns:
        NumpyTensor wrapper
    """
    return NumpyTensor(array, copy=copy)

def tensor_to_numpy(tensor: NumpyTensor) -> np.ndarray:
    """
    Convert NumpyTensor to NumPy array
    
    Args:
        tensor: NumpyTensor to convert
        
    Returns:
        NumPy array (view if possible)
    """
    return tensor.to_numpy()

def ensure_numpy_array(data: Union[np.ndarray, List, Tuple, Any]) -> np.ndarray:
    """
    Ensure input is a NumPy array
    
    Args:
        data: Input data to convert
        
    Returns:
        NumPy array
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, NumpyTensor):
        return data.to_numpy()
    else:
        return np.array(data)

def ensure_contiguous(array: np.ndarray) -> np.ndarray:
    """
    Ensure array is C-contiguous
    
    Args:
        array: Input array
        
    Returns:
        C-contiguous array
    """
    if not array.flags.c_contiguous:
        return np.ascontiguousarray(array)
    return array

def ensure_dtype(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Ensure array has the specified dtype
    
    Args:
        array: Input array
        dtype: Target dtype
        
    Returns:
        Array with specified dtype
    """
    if array.dtype != dtype:
        return array.astype(dtype)
    return array

class TensorOperations:
    """
    High-level tensor operations with automatic NumPy integration
    """
    
    @staticmethod
    def matrix_multiply(a: Union[np.ndarray, NumpyTensor], 
                       b: Union[np.ndarray, NumpyTensor],
                       optimizer: Optional[Any] = None) -> np.ndarray:
        """
        Optimized matrix multiplication using TrustformeRS
        
        Args:
            a: First matrix (m x k)
            b: Second matrix (k x n)
            optimizer: Performance optimizer instance
            
        Returns:
            Result matrix (m x n)
        """
        # Convert to numpy arrays
        a_np = ensure_numpy_array(a)
        b_np = ensure_numpy_array(b)
        
        # Ensure compatible shapes
        if a_np.ndim != 2 or b_np.ndim != 2:
            raise ValueError("Matrix multiplication requires 2D arrays")
        
        if a_np.shape[1] != b_np.shape[0]:
            raise ValueError(f"Incompatible shapes: {a_np.shape} and {b_np.shape}")
        
        # Ensure float32 and contiguous
        a_np = ensure_contiguous(ensure_dtype(a_np, np.float32))
        b_np = ensure_contiguous(ensure_dtype(b_np, np.float32))
        
        # Get dimensions
        m, k = a_np.shape
        k2, n = b_np.shape
        
        # Create output array
        c_np = np.zeros((m, n), dtype=np.float32)
        
        # Use optimizer if provided
        if optimizer is not None:
            optimizer.optimize_matrix_operations(a_np, b_np, c_np, m, n, k)
        else:
            # Fallback to NumPy
            c_np = np.dot(a_np, b_np)
        
        return c_np
    
    @staticmethod
    def element_wise_add(a: Union[np.ndarray, NumpyTensor],
                        b: Union[np.ndarray, NumpyTensor]) -> np.ndarray:
        """
        Element-wise addition with broadcasting
        
        Args:
            a: First array
            b: Second array
            
        Returns:
            Result array
        """
        a_np = ensure_numpy_array(a)
        b_np = ensure_numpy_array(b)
        
        return a_np + b_np
    
    @staticmethod
    def relu(x: Union[np.ndarray, NumpyTensor]) -> np.ndarray:
        """
        ReLU activation function
        
        Args:
            x: Input array
            
        Returns:
            ReLU output
        """
        x_np = ensure_numpy_array(x)
        return np.maximum(x_np, 0)
    
    @staticmethod
    def softmax(x: Union[np.ndarray, NumpyTensor], axis: int = -1) -> np.ndarray:
        """
        Softmax activation function
        
        Args:
            x: Input array
            axis: Axis along which to compute softmax
            
        Returns:
            Softmax output
        """
        x_np = ensure_numpy_array(x)
        
        # Numerical stability
        x_max = np.max(x_np, axis=axis, keepdims=True)
        x_shifted = x_np - x_max
        
        # Compute softmax
        exp_x = np.exp(x_shifted)
        sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
        
        return exp_x / sum_exp
    
    @staticmethod
    def layer_norm(x: Union[np.ndarray, NumpyTensor],
                   gamma: Optional[Union[np.ndarray, NumpyTensor]] = None,
                   beta: Optional[Union[np.ndarray, NumpyTensor]] = None,
                   eps: float = 1e-5) -> np.ndarray:
        """
        Layer normalization
        
        Args:
            x: Input array
            gamma: Scale parameter
            beta: Shift parameter
            eps: Small constant for numerical stability
            
        Returns:
            Normalized output
        """
        x_np = ensure_numpy_array(x)
        
        # Compute mean and variance
        mean = np.mean(x_np, axis=-1, keepdims=True)
        var = np.var(x_np, axis=-1, keepdims=True)
        
        # Normalize
        x_norm = (x_np - mean) / np.sqrt(var + eps)
        
        # Apply scale and shift
        if gamma is not None:
            gamma_np = ensure_numpy_array(gamma)
            x_norm = x_norm * gamma_np
        
        if beta is not None:
            beta_np = ensure_numpy_array(beta)
            x_norm = x_norm + beta_np
        
        return x_norm

class BatchProcessor:
    """
    Batch processing utilities for efficient tensor operations
    """
    
    def __init__(self, batch_size: int = 32):
        """
        Initialize batch processor
        
        Args:
            batch_size: Size of batches to process
        """
        self.batch_size = batch_size
        self.pending_batches = []
    
    def add_to_batch(self, data: Union[np.ndarray, NumpyTensor]) -> None:
        """
        Add data to current batch
        
        Args:
            data: Data to add to batch
        """
        data_np = ensure_numpy_array(data)
        self.pending_batches.append(data_np)
    
    def process_batch(self, operation: callable) -> List[np.ndarray]:
        """
        Process accumulated batch with given operation
        
        Args:
            operation: Function to apply to batch
            
        Returns:
            List of results
        """
        if not self.pending_batches:
            return []
        
        # Stack batch data
        batch_data = np.stack(self.pending_batches)
        
        # Apply operation
        results = operation(batch_data)
        
        # Clear pending batches
        self.pending_batches.clear()
        
        # Return results as list
        if isinstance(results, np.ndarray):
            return [results[i] for i in range(results.shape[0])]
        else:
            return results
    
    def get_batch_ready(self) -> bool:
        """Check if batch is ready for processing"""
        return len(self.pending_batches) >= self.batch_size
    
    def clear_batch(self) -> None:
        """Clear current batch"""
        self.pending_batches.clear()

# Utility functions for common operations
def create_tensor(shape: Tuple[int, ...], 
                 dtype: np.dtype = np.float32,
                 fill_value: Optional[float] = None) -> NumpyTensor:
    """
    Create a new tensor with specified shape and dtype
    
    Args:
        shape: Shape of the tensor
        dtype: Data type
        fill_value: Value to fill tensor with (default: zeros)
        
    Returns:
        New NumpyTensor
    """
    if fill_value is None:
        array = np.zeros(shape, dtype=dtype)
    else:
        array = np.full(shape, fill_value, dtype=dtype)
    
    return NumpyTensor(array)

def create_random_tensor(shape: Tuple[int, ...],
                        dtype: np.dtype = np.float32,
                        distribution: str = 'normal',
                        **kwargs) -> NumpyTensor:
    """
    Create a random tensor
    
    Args:
        shape: Shape of the tensor
        dtype: Data type
        distribution: Random distribution ('normal', 'uniform', 'xavier', 'he')
        **kwargs: Additional arguments for distribution
        
    Returns:
        Random NumpyTensor
    """
    if distribution == 'normal':
        mean = kwargs.get('mean', 0.0)
        std = kwargs.get('std', 1.0)
        array = np.random.normal(mean, std, shape).astype(dtype)
    elif distribution == 'uniform':
        low = kwargs.get('low', 0.0)
        high = kwargs.get('high', 1.0)
        array = np.random.uniform(low, high, shape).astype(dtype)
    elif distribution == 'xavier':
        # Xavier/Glorot initialization
        fan_in = np.prod(shape[:-1]) if len(shape) > 1 else shape[0]
        fan_out = shape[-1] if len(shape) > 1 else shape[0]
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        array = np.random.uniform(-limit, limit, shape).astype(dtype)
    elif distribution == 'he':
        # He initialization
        fan_in = np.prod(shape[:-1]) if len(shape) > 1 else shape[0]
        std = np.sqrt(2.0 / fan_in)
        array = np.random.normal(0, std, shape).astype(dtype)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    return NumpyTensor(array)

def stack_tensors(tensors: List[Union[np.ndarray, NumpyTensor]], 
                 axis: int = 0) -> NumpyTensor:
    """
    Stack multiple tensors along a new axis
    
    Args:
        tensors: List of tensors to stack
        axis: Axis along which to stack
        
    Returns:
        Stacked tensor
    """
    arrays = [ensure_numpy_array(t) for t in tensors]
    stacked = np.stack(arrays, axis=axis)
    return NumpyTensor(stacked)

def concatenate_tensors(tensors: List[Union[np.ndarray, NumpyTensor]],
                       axis: int = 0) -> NumpyTensor:
    """
    Concatenate multiple tensors along an existing axis
    
    Args:
        tensors: List of tensors to concatenate
        axis: Axis along which to concatenate
        
    Returns:
        Concatenated tensor
    """
    arrays = [ensure_numpy_array(t) for t in tensors]
    concatenated = np.concatenate(arrays, axis=axis)
    return NumpyTensor(concatenated)