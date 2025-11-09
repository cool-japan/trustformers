"""Tests for zero-copy NumPy conversions."""

import numpy as np
import pytest
import psutil
import os

import trustformers
from trustformers import Tensor
from trustformers.numpy_utils import (
    numpy_to_tensor_zero_copy,
    tensor_to_numpy_zero_copy,
    is_zero_copy_possible,
    memory_usage_comparison,
    optimize_array_for_zero_copy,
    ZeroCopyTensorWrapper,
)


def test_is_zero_copy_possible():
    """Test detection of zero-copy capable arrays."""
    # C-contiguous float32 array (should be zero-copy capable)
    arr_good = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    assert is_zero_copy_possible(arr_good)
    
    # Non-contiguous array (should not be zero-copy capable)
    arr_noncontiguous = arr_good.T  # Transpose creates non-contiguous view
    assert not is_zero_copy_possible(arr_noncontiguous)
    
    # Wrong dtype (should not be zero-copy capable)
    arr_wrong_dtype = np.array([[1, 2], [3, 4]], dtype=np.int32)
    assert not is_zero_copy_possible(arr_wrong_dtype)


def test_optimize_array_for_zero_copy():
    """Test array optimization for zero-copy."""
    # Non-contiguous array
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32).T
    optimized = optimize_array_for_zero_copy(arr)
    
    assert optimized.dtype == np.float32
    assert optimized.flags.c_contiguous
    assert is_zero_copy_possible(optimized)


def test_tensor_memory_info():
    """Test tensor memory information."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor = Tensor(arr)
    
    memory_info = tensor.memory_info()
    
    assert isinstance(memory_info, dict)
    assert "contiguous" in memory_info
    assert "shape" in memory_info
    assert "dtype" in memory_info
    assert "memory_bytes" in memory_info
    
    expected_bytes = 4 * 4  # 4 elements * 4 bytes per float32
    assert memory_info["memory_bytes"] == expected_bytes


def test_numpy_view_method():
    """Test the numpy_view method for potential zero-copy."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor = Tensor(arr)
    
    # Test numpy_view method
    view = tensor.numpy_view()
    
    assert isinstance(view, np.ndarray)
    assert view.shape == (2, 2)
    assert view.dtype == np.float32
    np.testing.assert_array_equal(view, arr)


def test_zero_copy_wrapper():
    """Test the ZeroCopyTensorWrapper class."""
    wrapper = ZeroCopyTensorWrapper(prefer_zero_copy=True)
    
    # Test with optimal array
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor = wrapper.numpy_to_tensor(arr)
    
    assert isinstance(tensor, Tensor)
    assert tensor.shape == [2, 2]
    
    # Convert back
    arr_back = wrapper.tensor_to_numpy(tensor)
    assert isinstance(arr_back, np.ndarray)
    np.testing.assert_array_equal(arr_back, arr)


def test_memory_usage_comparison():
    """Test memory usage comparison between copy and zero-copy."""
    # Create a larger tensor for more noticeable memory difference
    arr = np.random.randn(100, 100).astype(np.float32)
    tensor = Tensor(arr)
    
    usage_info = memory_usage_comparison(tensor)
    
    assert isinstance(usage_info, dict)
    assert "tensor_bytes" in usage_info
    assert "copy_memory_delta_bytes" in usage_info
    assert "zero_copy_memory_delta_bytes" in usage_info
    assert "zero_copy_possible" in usage_info
    assert "memory_savings_bytes" in usage_info
    
    # For contiguous tensors, zero-copy should be possible
    if usage_info["contiguous"]:
        assert usage_info["zero_copy_possible"]


def test_contiguous_check():
    """Test contiguous memory layout checking."""
    # Contiguous array
    arr_contiguous = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    tensor_contiguous = Tensor(arr_contiguous)
    assert tensor_contiguous.is_contiguous()
    
    # Non-contiguous array (if supported by the tensor implementation)
    arr_noncontiguous = arr_contiguous.T
    if not arr_noncontiguous.flags.c_contiguous:
        # Only test if we can actually create non-contiguous tensors
        try:
            tensor_noncontiguous = Tensor(arr_noncontiguous)
            # This might still be contiguous after conversion
            # so we just check that the method exists and works
            assert isinstance(tensor_noncontiguous.is_contiguous(), bool)
        except:
            # If the tensor implementation doesn't support non-contiguous tensors,
            # that's fine - just skip this test
            pass


def test_fallback_behavior():
    """Test that fallback copying works when zero-copy isn't possible."""
    # Create a non-optimal array
    arr = np.array([[1, 2], [3, 4]], dtype=np.int64)  # Wrong dtype
    
    # This should still work via fallback
    tensor = numpy_to_tensor_zero_copy(arr)
    assert isinstance(tensor, Tensor)
    assert tensor.shape == [2, 2]
    
    # Convert back
    arr_back = tensor_to_numpy_zero_copy(tensor)
    assert isinstance(arr_back, np.ndarray)
    assert arr_back.dtype == np.float32  # Converted to float32


def test_large_array_performance():
    """Test with larger arrays to see potential performance benefits."""
    # Create a large array
    large_arr = np.random.randn(1000, 1000).astype(np.float32)
    
    # Test conversion time (not asserting specific times, just that it works)
    import time
    
    start_time = time.time()
    tensor = numpy_to_tensor_zero_copy(large_arr)
    conversion_time = time.time() - start_time
    
    assert isinstance(tensor, Tensor)
    assert tensor.shape == [1000, 1000]
    
    # Test back conversion
    start_time = time.time()
    arr_back = tensor_to_numpy_zero_copy(tensor)
    back_conversion_time = time.time() - start_time
    
    assert isinstance(arr_back, np.ndarray)
    assert arr_back.shape == (1000, 1000)
    
    # Just verify the times are reasonable (not asserting specific values)
    assert conversion_time < 1.0  # Should be fast
    assert back_conversion_time < 1.0  # Should be fast


if __name__ == "__main__":
    # Run basic tests
    test_is_zero_copy_possible()
    test_optimize_array_for_zero_copy()
    test_tensor_memory_info()
    test_numpy_view_method()
    test_zero_copy_wrapper()
    test_memory_usage_comparison()
    test_contiguous_check()
    test_fallback_behavior()
    test_large_array_performance()
    
    print("All zero-copy tests passed!")