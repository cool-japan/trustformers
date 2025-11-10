//! Tensor operations C API for TrustformeRS
//!
//! This module provides a comprehensive C API for tensor operations,
//! enabling efficient manipulation of multidimensional arrays from C/C++.

use crate::error::{TrustformersError, TrustformersResult};
use crate::utils::{c_str_to_string, string_to_c_str};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::Arc;
use trustformers_core::tensor::Tensor;

/// Global tensor registry for handle-based access
static TENSOR_REGISTRY: Lazy<RwLock<TensorRegistry>> =
    Lazy::new(|| RwLock::new(TensorRegistry::new()));

/// Internal tensor registry
struct TensorRegistry {
    tensors: HashMap<usize, Arc<Tensor>>,
    next_handle: usize,
}

impl TensorRegistry {
    fn new() -> Self {
        Self {
            tensors: HashMap::new(),
            next_handle: 1,
        }
    }

    fn register(&mut self, tensor: Tensor) -> usize {
        let handle = self.next_handle;
        self.next_handle += 1;
        self.tensors.insert(handle, Arc::new(tensor));
        handle
    }

    fn get(&self, handle: usize) -> Option<Arc<Tensor>> {
        self.tensors.get(&handle).cloned()
    }

    fn remove(&mut self, handle: usize) -> bool {
        self.tensors.remove(&handle).is_some()
    }
}

/// Tensor handle type (opaque pointer from C perspective)
pub type TrustformersTensor = usize;

/// Tensor data types
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustformersDType {
    Float32 = 0,
    Float64 = 1,
    Int8 = 2,
    Int16 = 3,
    Int32 = 4,
    Int64 = 5,
    UInt8 = 6,
    UInt16 = 7,
    UInt32 = 8,
    UInt64 = 9,
    Bool = 10,
    Float16 = 11,
    BFloat16 = 12,
}

/// Tensor reduction operations
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum TrustformersReduceOp {
    Sum = 0,
    Mean = 1,
    Max = 2,
    Min = 3,
    Prod = 4,
    Any = 5,
    All = 6,
}

/// Tensor interpolation modes for resize operations
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum TrustformersInterpolationMode {
    Nearest = 0,
    Linear = 1,
    Bilinear = 2,
    Bicubic = 3,
}

/// Create a new tensor from raw data
///
/// # Safety
/// The data pointer must be valid and contain at least `numel` elements
#[no_mangle]
pub extern "C" fn trustformers_tensor_create_from_data(
    data: *const f32,
    shape: *const usize,
    ndim: usize,
    dtype: TrustformersDType,
    handle: *mut TrustformersTensor,
) -> TrustformersError {
    if data.is_null() || shape.is_null() || handle.is_null() {
        return TrustformersError::NullPointer;
    }

    if ndim == 0 {
        return TrustformersError::InvalidParameter;
    }

    let shape_slice = unsafe { std::slice::from_raw_parts(shape, ndim) };

    // Calculate total number of elements
    let numel: usize = shape_slice.iter().product();
    if numel == 0 {
        return TrustformersError::InvalidParameter;
    }

    let data_slice = unsafe { std::slice::from_raw_parts(data, numel) };

    // Convert shape to Vec<usize>
    let shape_vec: Vec<usize> = shape_slice.to_vec();

    // Create tensor based on dtype
    let tensor_result = match dtype {
        TrustformersDType::Float32 => Tensor::from_slice(data_slice, &shape_vec),
        _ => return TrustformersError::FeatureNotAvailable,
    };

    match tensor_result {
        Ok(tensor) => {
            let tensor_handle = TENSOR_REGISTRY.write().register(tensor);
            unsafe {
                *handle = tensor_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Create a zero-initialized tensor
#[no_mangle]
pub extern "C" fn trustformers_tensor_zeros(
    shape: *const usize,
    ndim: usize,
    dtype: TrustformersDType,
    handle: *mut TrustformersTensor,
) -> TrustformersError {
    if shape.is_null() || handle.is_null() {
        return TrustformersError::NullPointer;
    }

    if ndim == 0 {
        return TrustformersError::InvalidParameter;
    }

    let shape_slice = unsafe { std::slice::from_raw_parts(shape, ndim) };
    let shape_vec: Vec<usize> = shape_slice.to_vec();

    let tensor_result = match dtype {
        TrustformersDType::Float32 => Tensor::zeros(&shape_vec),
        _ => return TrustformersError::FeatureNotAvailable,
    };

    match tensor_result {
        Ok(tensor) => {
            let tensor_handle = TENSOR_REGISTRY.write().register(tensor);
            unsafe {
                *handle = tensor_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Create a one-initialized tensor
#[no_mangle]
pub extern "C" fn trustformers_tensor_ones(
    shape: *const usize,
    ndim: usize,
    dtype: TrustformersDType,
    handle: *mut TrustformersTensor,
) -> TrustformersError {
    if shape.is_null() || handle.is_null() {
        return TrustformersError::NullPointer;
    }

    if ndim == 0 {
        return TrustformersError::InvalidParameter;
    }

    let shape_slice = unsafe { std::slice::from_raw_parts(shape, ndim) };
    let shape_vec: Vec<usize> = shape_slice.to_vec();

    let tensor_result = match dtype {
        TrustformersDType::Float32 => Tensor::ones(&shape_vec),
        _ => return TrustformersError::FeatureNotAvailable,
    };

    match tensor_result {
        Ok(tensor) => {
            let tensor_handle = TENSOR_REGISTRY.write().register(tensor);
            unsafe {
                *handle = tensor_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Create a random tensor with normal distribution
#[no_mangle]
pub extern "C" fn trustformers_tensor_randn(
    shape: *const usize,
    ndim: usize,
    dtype: TrustformersDType,
    handle: *mut TrustformersTensor,
) -> TrustformersError {
    if shape.is_null() || handle.is_null() {
        return TrustformersError::NullPointer;
    }

    if ndim == 0 {
        return TrustformersError::InvalidParameter;
    }

    let shape_slice = unsafe { std::slice::from_raw_parts(shape, ndim) };
    let shape_vec: Vec<usize> = shape_slice.to_vec();

    let tensor_result = match dtype {
        TrustformersDType::Float32 => Tensor::randn(&shape_vec),
        _ => return TrustformersError::FeatureNotAvailable,
    };

    match tensor_result {
        Ok(tensor) => {
            let tensor_handle = TENSOR_REGISTRY.write().register(tensor);
            unsafe {
                *handle = tensor_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Create a random tensor with uniform distribution
/// Note: Currently implemented as randn (normal distribution)
#[no_mangle]
pub extern "C" fn trustformers_tensor_rand(
    shape: *const usize,
    ndim: usize,
    dtype: TrustformersDType,
    handle: *mut TrustformersTensor,
) -> TrustformersError {
    if shape.is_null() || handle.is_null() {
        return TrustformersError::NullPointer;
    }

    if ndim == 0 {
        return TrustformersError::InvalidParameter;
    }

    let shape_slice = unsafe { std::slice::from_raw_parts(shape, ndim) };
    let shape_vec: Vec<usize> = shape_slice.to_vec();

    // Currently using randn as rand is not yet implemented in core
    let tensor_result = match dtype {
        TrustformersDType::Float32 => Tensor::randn(&shape_vec),
        _ => return TrustformersError::FeatureNotAvailable,
    };

    match tensor_result {
        Ok(tensor) => {
            let tensor_handle = TENSOR_REGISTRY.write().register(tensor);
            unsafe {
                *handle = tensor_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Get tensor shape
#[no_mangle]
pub extern "C" fn trustformers_tensor_shape(
    tensor: TrustformersTensor,
    shape: *mut *mut usize,
    ndim: *mut usize,
) -> TrustformersError {
    if shape.is_null() || ndim.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(tensor_arc) = registry.get(tensor) else {
        return TrustformersError::InvalidHandle;
    };

    let tensor_shape = tensor_arc.shape();
    let shape_len = tensor_shape.len();

    // Allocate memory for shape array
    let shape_vec: Vec<usize> = tensor_shape.to_vec();
    let shape_ptr = shape_vec.into_boxed_slice();
    let shape_raw = Box::into_raw(shape_ptr) as *mut usize;

    unsafe {
        *shape = shape_raw;
        *ndim = shape_len;
    }

    TrustformersError::Success
}

/// Get number of elements in tensor
#[no_mangle]
pub extern "C" fn trustformers_tensor_numel(
    tensor: TrustformersTensor,
    numel: *mut usize,
) -> TrustformersError {
    if numel.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(tensor_arc) = registry.get(tensor) else {
        return TrustformersError::InvalidHandle;
    };

    let shape = tensor_arc.shape();
    let total_elements: usize = shape.iter().product();

    unsafe {
        *numel = total_elements;
    }

    TrustformersError::Success
}

/// Get tensor data pointer (raw access)
///
/// # Safety
/// The returned pointer is valid only as long as the tensor exists
#[no_mangle]
pub extern "C" fn trustformers_tensor_get_data_ptr(
    tensor: TrustformersTensor,
    data: *mut *const f32,
) -> TrustformersError {
    if data.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(_tensor_arc) = registry.get(tensor) else {
        return TrustformersError::InvalidHandle;
    };

    // For now, return null since we don't have direct data access
    // This would need to be implemented with proper zero-copy access
    unsafe {
        *data = ptr::null();
    }

    TrustformersError::FeatureNotAvailable
}

/// Copy tensor data to a buffer
#[no_mangle]
pub extern "C" fn trustformers_tensor_copy_data(
    tensor: TrustformersTensor,
    buffer: *mut f32,
    buffer_size: usize,
) -> TrustformersError {
    if buffer.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(_tensor_arc) = registry.get(tensor) else {
        return TrustformersError::InvalidHandle;
    };

    // This would need to be implemented with actual data copying
    TrustformersError::FeatureNotAvailable
}

/// Reshape tensor
#[no_mangle]
pub extern "C" fn trustformers_tensor_reshape(
    tensor: TrustformersTensor,
    new_shape: *const usize,
    ndim: usize,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if new_shape.is_null() || output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(tensor_arc) = registry.get(tensor) else {
        return TrustformersError::InvalidHandle;
    };

    let new_shape_slice = unsafe { std::slice::from_raw_parts(new_shape, ndim) };
    let new_shape_vec: Vec<usize> = new_shape_slice.to_vec();

    match tensor_arc.reshape(&new_shape_vec) {
        Ok(reshaped) => {
            drop(registry);
            let handle = TENSOR_REGISTRY.write().register(reshaped);
            unsafe {
                *output = handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Transpose tensor
#[no_mangle]
pub extern "C" fn trustformers_tensor_transpose(
    tensor: TrustformersTensor,
    dim0: usize,
    dim1: usize,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(tensor_arc) = registry.get(tensor) else {
        return TrustformersError::InvalidHandle;
    };

    match tensor_arc.transpose(dim0, dim1) {
        Ok(transposed) => {
            drop(registry);
            let handle = TENSOR_REGISTRY.write().register(transposed);
            unsafe {
                *output = handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Permute tensor dimensions
#[no_mangle]
pub extern "C" fn trustformers_tensor_permute(
    tensor: TrustformersTensor,
    axes: *const usize,
    num_axes: usize,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if axes.is_null() || output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(tensor_arc) = registry.get(tensor) else {
        return TrustformersError::InvalidHandle;
    };

    let axes_slice = unsafe { std::slice::from_raw_parts(axes, num_axes) };
    let axes_vec: Vec<usize> = axes_slice.to_vec();

    match tensor_arc.permute(&axes_vec) {
        Ok(permuted) => {
            drop(registry);
            let handle = TENSOR_REGISTRY.write().register(permuted);
            unsafe {
                *output = handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Add two tensors
#[no_mangle]
pub extern "C" fn trustformers_tensor_add(
    tensor1: TrustformersTensor,
    tensor2: TrustformersTensor,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(t1) = registry.get(tensor1) else {
        return TrustformersError::InvalidHandle;
    };
    let Some(t2) = registry.get(tensor2) else {
        return TrustformersError::InvalidHandle;
    };

    match t1.add(&t2) {
        Ok(result) => {
            drop(registry);
            let handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Subtract two tensors
#[no_mangle]
pub extern "C" fn trustformers_tensor_sub(
    tensor1: TrustformersTensor,
    tensor2: TrustformersTensor,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(t1) = registry.get(tensor1) else {
        return TrustformersError::InvalidHandle;
    };
    let Some(t2) = registry.get(tensor2) else {
        return TrustformersError::InvalidHandle;
    };

    match t1.sub(&t2) {
        Ok(result) => {
            drop(registry);
            let handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Multiply two tensors element-wise
#[no_mangle]
pub extern "C" fn trustformers_tensor_mul(
    tensor1: TrustformersTensor,
    tensor2: TrustformersTensor,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(t1) = registry.get(tensor1) else {
        return TrustformersError::InvalidHandle;
    };
    let Some(t2) = registry.get(tensor2) else {
        return TrustformersError::InvalidHandle;
    };

    match t1.mul(&t2) {
        Ok(result) => {
            drop(registry);
            let handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Divide two tensors element-wise
#[no_mangle]
pub extern "C" fn trustformers_tensor_div(
    tensor1: TrustformersTensor,
    tensor2: TrustformersTensor,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(t1) = registry.get(tensor1) else {
        return TrustformersError::InvalidHandle;
    };
    let Some(t2) = registry.get(tensor2) else {
        return TrustformersError::InvalidHandle;
    };

    match t1.div(&t2) {
        Ok(result) => {
            drop(registry);
            let handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Matrix multiplication
#[no_mangle]
pub extern "C" fn trustformers_tensor_matmul(
    tensor1: TrustformersTensor,
    tensor2: TrustformersTensor,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(t1) = registry.get(tensor1) else {
        return TrustformersError::InvalidHandle;
    };
    let Some(t2) = registry.get(tensor2) else {
        return TrustformersError::InvalidHandle;
    };

    match t1.matmul(&t2) {
        Ok(result) => {
            drop(registry);
            let handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Apply ReLU activation
#[no_mangle]
pub extern "C" fn trustformers_tensor_relu(
    tensor: TrustformersTensor,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(t) = registry.get(tensor) else {
        return TrustformersError::InvalidHandle;
    };

    match t.relu() {
        Ok(result) => {
            drop(registry);
            let handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Apply GELU activation
#[no_mangle]
pub extern "C" fn trustformers_tensor_gelu(
    tensor: TrustformersTensor,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(t) = registry.get(tensor) else {
        return TrustformersError::InvalidHandle;
    };

    match t.gelu() {
        Ok(result) => {
            drop(registry);
            let handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Apply softmax along a dimension
#[no_mangle]
pub extern "C" fn trustformers_tensor_softmax(
    tensor: TrustformersTensor,
    dim: c_int,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(t) = registry.get(tensor) else {
        return TrustformersError::InvalidHandle;
    };

    match t.softmax(dim) {
        Ok(result) => {
            drop(registry);
            let handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Reduce tensor along dimension
#[no_mangle]
pub extern "C" fn trustformers_tensor_reduce(
    tensor: TrustformersTensor,
    op: TrustformersReduceOp,
    dim: c_int,
    keep_dim: c_int,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(t) = registry.get(tensor) else {
        return TrustformersError::InvalidHandle;
    };

    let axes_opt = if dim < 0 { None } else { Some(vec![dim as usize]) };
    let keep_dims = keep_dim != 0;

    let result = match op {
        TrustformersReduceOp::Sum => t.sum(axes_opt, keep_dims),
        TrustformersReduceOp::Mean => t.mean(),
        _ => return TrustformersError::FeatureNotAvailable,
    };

    match result {
        Ok(result_tensor) => {
            drop(registry);
            let handle = TENSOR_REGISTRY.write().register(result_tensor);
            unsafe {
                *output = handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Clone a tensor
#[no_mangle]
pub extern "C" fn trustformers_tensor_clone(
    tensor: TrustformersTensor,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(t) = registry.get(tensor) else {
        return TrustformersError::InvalidHandle;
    };

    // Clone the tensor
    let cloned = (*t).clone();
    drop(registry);
    let handle = TENSOR_REGISTRY.write().register(cloned);
    unsafe {
        *output = handle;
    }

    TrustformersError::Success
}

/// Free a tensor
#[no_mangle]
pub extern "C" fn trustformers_tensor_free(tensor: TrustformersTensor) -> TrustformersError {
    if tensor == 0 {
        return TrustformersError::InvalidHandle;
    }

    let removed = TENSOR_REGISTRY.write().remove(tensor);
    if removed {
        TrustformersError::Success
    } else {
        TrustformersError::InvalidHandle
    }
}

/// Print tensor information (for debugging)
#[no_mangle]
pub extern "C" fn trustformers_tensor_print_info(tensor: TrustformersTensor) -> TrustformersError {
    let registry = TENSOR_REGISTRY.read();
    let Some(t) = registry.get(tensor) else {
        return TrustformersError::InvalidHandle;
    };

    let shape = t.shape();
    println!("Tensor {{ shape: {:?} }}", shape);

    TrustformersError::Success
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_zeros() {
        let shape = vec![2, 3];
        let mut handle: TrustformersTensor = 0;

        let err = trustformers_tensor_zeros(
            shape.as_ptr(),
            shape.len(),
            TrustformersDType::Float32,
            &mut handle,
        );

        assert_eq!(err, TrustformersError::Success);
        assert_ne!(handle, 0);

        // Free tensor
        let err = trustformers_tensor_free(handle);
        assert_eq!(err, TrustformersError::Success);
    }

    #[test]
    fn test_tensor_ones() {
        let shape = vec![2, 3];
        let mut handle: TrustformersTensor = 0;

        let err = trustformers_tensor_ones(
            shape.as_ptr(),
            shape.len(),
            TrustformersDType::Float32,
            &mut handle,
        );

        assert_eq!(err, TrustformersError::Success);
        assert_ne!(handle, 0);

        let err = trustformers_tensor_free(handle);
        assert_eq!(err, TrustformersError::Success);
    }

    #[test]
    fn test_tensor_add() {
        let shape = vec![2, 3];
        let mut t1: TrustformersTensor = 0;
        let mut t2: TrustformersTensor = 0;
        let mut result: TrustformersTensor = 0;

        // Create two tensors
        trustformers_tensor_ones(
            shape.as_ptr(),
            shape.len(),
            TrustformersDType::Float32,
            &mut t1,
        );
        trustformers_tensor_ones(
            shape.as_ptr(),
            shape.len(),
            TrustformersDType::Float32,
            &mut t2,
        );

        // Add them
        let err = trustformers_tensor_add(t1, t2, &mut result);
        assert_eq!(err, TrustformersError::Success);
        assert_ne!(result, 0);

        // Free all tensors
        trustformers_tensor_free(t1);
        trustformers_tensor_free(t2);
        trustformers_tensor_free(result);
    }

    #[test]
    fn test_tensor_reshape() {
        let shape = vec![2, 3];
        let mut tensor: TrustformersTensor = 0;
        let mut reshaped: TrustformersTensor = 0;

        trustformers_tensor_ones(
            shape.as_ptr(),
            shape.len(),
            TrustformersDType::Float32,
            &mut tensor,
        );

        let new_shape = vec![3, 2];
        let err =
            trustformers_tensor_reshape(tensor, new_shape.as_ptr(), new_shape.len(), &mut reshaped);

        assert_eq!(err, TrustformersError::Success);
        assert_ne!(reshaped, 0);

        trustformers_tensor_free(tensor);
        trustformers_tensor_free(reshaped);
    }
}
