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

// ============================================================================
// Advanced Tensor Operations
// ============================================================================

/// Apply Sigmoid activation function (1 / (1 + exp(-x)))
#[no_mangle]
pub extern "C" fn trustformers_tensor_sigmoid(
    input: TrustformersTensor,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(input_tensor) = registry.get(input) else {
        return TrustformersError::InvalidHandle;
    };

    match input_tensor.sigmoid() {
        Ok(result) => {
            drop(registry);
            let result_handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = result_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Apply Tanh activation function
#[no_mangle]
pub extern "C" fn trustformers_tensor_tanh(
    input: TrustformersTensor,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(input_tensor) = registry.get(input) else {
        return TrustformersError::InvalidHandle;
    };

    match input_tensor.tanh() {
        Ok(result) => {
            drop(registry);
            let result_handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = result_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Apply SiLU/Swish activation function (x * sigmoid(x))
#[no_mangle]
pub extern "C" fn trustformers_tensor_silu(
    input: TrustformersTensor,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(input_tensor) = registry.get(input) else {
        return TrustformersError::InvalidHandle;
    };

    match input_tensor.silu() {
        Ok(result) => {
            drop(registry);
            let result_handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = result_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Apply LeakyReLU activation function
#[no_mangle]
pub extern "C" fn trustformers_tensor_leaky_relu(
    input: TrustformersTensor,
    negative_slope: f32,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(input_tensor) = registry.get(input) else {
        return TrustformersError::InvalidHandle;
    };

    match input_tensor.leaky_relu(negative_slope) {
        Ok(result) => {
            drop(registry);
            let result_handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = result_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Compute exponential (e^x) element-wise
#[no_mangle]
pub extern "C" fn trustformers_tensor_exp(
    input: TrustformersTensor,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(input_tensor) = registry.get(input) else {
        return TrustformersError::InvalidHandle;
    };

    match input_tensor.exp() {
        Ok(result) => {
            drop(registry);
            let result_handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = result_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Compute natural logarithm element-wise
#[no_mangle]
pub extern "C" fn trustformers_tensor_log(
    input: TrustformersTensor,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(input_tensor) = registry.get(input) else {
        return TrustformersError::InvalidHandle;
    };

    match input_tensor.log() {
        Ok(result) => {
            drop(registry);
            let result_handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = result_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Compute square root element-wise
#[no_mangle]
pub extern "C" fn trustformers_tensor_sqrt(
    input: TrustformersTensor,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(input_tensor) = registry.get(input) else {
        return TrustformersError::InvalidHandle;
    };

    match input_tensor.sqrt() {
        Ok(result) => {
            drop(registry);
            let result_handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = result_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Compute absolute value element-wise
#[no_mangle]
pub extern "C" fn trustformers_tensor_abs(
    input: TrustformersTensor,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(input_tensor) = registry.get(input) else {
        return TrustformersError::InvalidHandle;
    };

    match input_tensor.abs() {
        Ok(result) => {
            drop(registry);
            let result_handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = result_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Compute power (x^exponent) element-wise
#[no_mangle]
pub extern "C" fn trustformers_tensor_pow(
    input: TrustformersTensor,
    exponent: f32,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(input_tensor) = registry.get(input) else {
        return TrustformersError::InvalidHandle;
    };

    match input_tensor.pow(exponent) {
        Ok(result) => {
            drop(registry);
            let result_handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = result_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Compute sine element-wise
#[no_mangle]
pub extern "C" fn trustformers_tensor_sin(
    input: TrustformersTensor,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(input_tensor) = registry.get(input) else {
        return TrustformersError::InvalidHandle;
    };

    match input_tensor.sin() {
        Ok(result) => {
            drop(registry);
            let result_handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = result_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Compute cosine element-wise
#[no_mangle]
pub extern "C" fn trustformers_tensor_cos(
    input: TrustformersTensor,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(input_tensor) = registry.get(input) else {
        return TrustformersError::InvalidHandle;
    };

    match input_tensor.cos() {
        Ok(result) => {
            drop(registry);
            let result_handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = result_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

// NOTE: The following tensor operations are commented out because they are not yet
// implemented in the trustformers_core::tensor::Tensor API. They can be uncommented
// once the corresponding methods are added to the Tensor implementation.

/*
/// Clamp tensor values to [min, max] range (NOT YET IMPLEMENTED)
/// TODO: Implement clamp() method in trustformers_core::tensor::Tensor
#[no_mangle]
pub extern "C" fn trustformers_tensor_clamp(
    input: TrustformersTensor,
    min_val: f32,
    max_val: f32,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    TrustformersError::FeatureNotAvailable
}
*/

/// Concatenate tensors along a specified dimension (NOT YET IMPLEMENTED)
///
/// # Parameters
/// - `tensors`: Array of tensor handles to concatenate
/// - `num_tensors`: Number of tensors in the array
/// - `dim`: Dimension along which to concatenate
/// - `output`: Output tensor handle
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_tensor_concat(
    _tensors: *const TrustformersTensor,
    _num_tensors: usize,
    _dim: i64,
    _output: *mut TrustformersTensor,
) -> TrustformersError {
    // Concatenate operation not yet implemented in trustformers_core
    TrustformersError::FeatureNotAvailable
}

/// Stack tensors along a new dimension (NOT YET IMPLEMENTED)
///
/// # Parameters
/// - `tensors`: Array of tensor handles to stack
/// - `num_tensors`: Number of tensors in the array
/// - `dim`: Dimension along which to stack
/// - `output`: Output tensor handle
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_tensor_stack(
    _tensors: *const TrustformersTensor,
    _num_tensors: usize,
    _dim: i64,
    _output: *mut TrustformersTensor,
) -> TrustformersError {
    // Stack operation not yet implemented in trustformers_core
    TrustformersError::FeatureNotAvailable
}

/// Squeeze tensor by removing dimensions of size 1
///
/// # Parameters
/// - `input`: Input tensor handle
/// - `dim`: Optional dimension to squeeze (use -1 for all dimensions)
/// - `output`: Output tensor handle
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_tensor_squeeze(
    input: TrustformersTensor,
    dim: i64,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(input_tensor) = registry.get(input) else {
        return TrustformersError::InvalidHandle;
    };

    let result = if dim < 0 {
        // Squeeze all dimensions of size 1 not supported - return error
        return TrustformersError::FeatureNotAvailable;
    } else {
        // Squeeze specific dimension
        input_tensor.squeeze(dim as usize)
    };

    match result {
        Ok(squeezed) => {
            drop(registry);
            let result_handle = TENSOR_REGISTRY.write().register(squeezed);
            unsafe {
                *output = result_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Unsqueeze tensor by adding a dimension of size 1
///
/// # Parameters
/// - `input`: Input tensor handle
/// - `dim`: Dimension at which to add the new axis
/// - `output`: Output tensor handle
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_tensor_unsqueeze(
    input: TrustformersTensor,
    dim: i64,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(input_tensor) = registry.get(input) else {
        return TrustformersError::InvalidHandle;
    };

    match input_tensor.unsqueeze(dim as usize) {
        Ok(unsqueezed) => {
            drop(registry);
            let result_handle = TENSOR_REGISTRY.write().register(unsqueezed);
            unsafe {
                *output = result_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Gather values along an axis specified by indices
///
/// # Parameters
/// - `input`: Input tensor handle
/// - `dim`: Dimension along which to index
/// - `indices`: Indices tensor handle
/// - `output`: Output tensor handle
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_tensor_gather(
    input: TrustformersTensor,
    dim: i64,
    indices: TrustformersTensor,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(input_tensor) = registry.get(input) else {
        return TrustformersError::InvalidHandle;
    };
    let Some(indices_tensor) = registry.get(indices) else {
        return TrustformersError::InvalidHandle;
    };

    match input_tensor.gather(dim, &indices_tensor) {
        Ok(result) => {
            drop(registry);
            let result_handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = result_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Scatter values along an axis specified by indices (NOT YET IMPLEMENTED)
///
/// # Parameters
/// - `input`: Input tensor handle
/// - `dim`: Dimension along which to scatter
/// - `indices`: Indices tensor handle
/// - `src`: Source tensor handle with values to scatter
/// - `output`: Output tensor handle
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_tensor_scatter(
    _input: TrustformersTensor,
    _dim: i64,
    _indices: TrustformersTensor,
    _src: TrustformersTensor,
    _output: *mut TrustformersTensor,
) -> TrustformersError {
    // Scatter operation not yet implemented in trustformers_core
    TrustformersError::FeatureNotAvailable
}

/// Index select - Select a single element along a dimension
///
/// # Parameters
/// - `input`: Input tensor handle
/// - `dim`: Dimension to index along
/// - `index`: Index value (not a tensor)
/// - `output`: Output tensor handle
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_tensor_index_select(
    input: TrustformersTensor,
    dim: i64,
    index: i64,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(input_tensor) = registry.get(input) else {
        return TrustformersError::InvalidHandle;
    };

    match input_tensor.select(dim as usize, index) {
        Ok(result) => {
            drop(registry);
            let result_handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = result_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Compute mean along specified dimensions
///
/// # Parameters
/// - `input`: Input tensor handle
/// - `dims`: Array of dimensions to reduce
/// - `num_dims`: Number of dimensions in the array
/// - `keepdim`: Whether to keep reduced dimensions
/// - `output`: Output tensor handle
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_tensor_mean_dims(
    input: TrustformersTensor,
    dims: *const i64,
    num_dims: usize,
    keepdim: bool,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() || (num_dims > 0 && dims.is_null()) {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(input_tensor) = registry.get(input) else {
        return TrustformersError::InvalidHandle;
    };

    let dims_slice = if num_dims > 0 {
        unsafe { std::slice::from_raw_parts(dims, num_dims) }
    } else {
        &[]
    };

    // Convert i64 dims to usize
    let dims_usize: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();

    // Note: keepdim parameter is ignored as mean_axes doesn't support it
    match input_tensor.mean_axes(&dims_usize) {
        Ok(result) => {
            drop(registry);
            let result_handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = result_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
}

/// Compute sum along specified dimensions
///
/// # Parameters
/// - `input`: Input tensor handle
/// - `dims`: Array of dimensions to reduce
/// - `num_dims`: Number of dimensions in the array
/// - `keepdim`: Whether to keep reduced dimensions
/// - `output`: Output tensor handle
///
/// # Returns
/// Error code indicating success or failure
#[no_mangle]
pub extern "C" fn trustformers_tensor_sum_dims(
    input: TrustformersTensor,
    dims: *const i64,
    num_dims: usize,
    keepdim: bool,
    output: *mut TrustformersTensor,
) -> TrustformersError {
    if output.is_null() || (num_dims > 0 && dims.is_null()) {
        return TrustformersError::NullPointer;
    }

    let registry = TENSOR_REGISTRY.read();
    let Some(input_tensor) = registry.get(input) else {
        return TrustformersError::InvalidHandle;
    };

    let dims_slice = if num_dims > 0 {
        unsafe { std::slice::from_raw_parts(dims, num_dims) }
    } else {
        &[]
    };

    // Convert i64 dims to usize
    let dims_usize: Vec<usize> = dims_slice.iter().map(|&d| d as usize).collect();

    // Note: sum_dim only supports single dimension at a time
    // We'll just use the first dimension for now
    if dims_usize.is_empty() {
        return TrustformersError::InvalidParameter;
    }

    match input_tensor.sum_dim(dims_usize[0] as i64, keepdim) {
        Ok(result) => {
            drop(registry);
            let result_handle = TENSOR_REGISTRY.write().register(result);
            unsafe {
                *output = result_handle;
            }
            TrustformersError::Success
        },
        Err(_) => TrustformersError::TensorError,
    }
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
