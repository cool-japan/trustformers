//! Core tensor abstraction for TrustformeRS.
//!
//! This module provides the fundamental `Tensor` type that serves as the backbone
//! for all numerical computations in TrustformeRS. It offers a unified interface
//! over different backend implementations (ndarray, PyTorch, Candle) while
//! maintaining high performance through SIMD optimizations.
//!
//! # Overview
//!
//! The `Tensor` enum provides:
//! - Multi-backend support (CPU via ndarray, GPU via PyTorch/Candle)
//! - Common tensor operations (matmul, add, softmax, etc.)
//! - Broadcasting and shape manipulation
//! - Gradient-related operations for training
//! - Serialization support for model persistence
//!
//! # Example
//!
//! ```no_run
//! use trustformers_core::tensor::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create tensors
//! let a = Tensor::randn(&[2, 3])?;
//! let b = Tensor::randn(&[3, 4])?;
//!
//! // Perform operations
//! let c = a.matmul(&b)?;  // Matrix multiplication
//! let d = c.relu()?;       // ReLU activation
//! let e = d.softmax(-1)?;  // Softmax along last dimension
//! # Ok(())
//! # }
//! ```
//!
//! # Performance Notes
//!
//! - SIMD operations are used where available for better performance
//! - Tensor operations are optimized for common transformer patterns
//! - GPU operations are available when compiled with appropriate features

mod activations;
mod complex;
pub mod constructors;
mod conversions;
mod expression;
mod math_ops;
mod sparse;
pub mod transformations;
mod utils;

#[cfg(test)]
mod property_tests;

use crate::errors::Result;
use ndarray::ArrayD;
use num_complex::{Complex32, Complex64};
use serde::{Deserialize, Serialize};

/// Data types supported by tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point
    F16,
    /// Brain floating point 16
    BF16,
    /// 64-bit floating point
    F64,
    /// 32-bit complex number (two 32-bit floats)
    C32,
    /// 64-bit complex number (two 64-bit floats)
    C64,
    /// 16-bit complex number (two 16-bit floats)
    CF16,
    /// Brain floating point 16 complex number (two BF16 floats)
    CBF16,
    /// 8-bit unsigned integer
    U8,
    /// 16-bit unsigned integer
    U16,
    /// 32-bit unsigned integer
    U32,
    /// 64-bit unsigned integer
    U64,
    /// 8-bit signed integer
    I8,
    /// 16-bit signed integer
    I16,
    /// 32-bit signed integer
    I32,
    /// 64-bit signed integer
    I64,
    /// Boolean
    Bool,
}

impl DType {
    /// Returns the size in bytes of an element of this data type
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::F64 => 8,
            DType::C32 => 8,   // Two 32-bit floats
            DType::C64 => 16,  // Two 64-bit floats
            DType::CF16 => 4,  // Two 16-bit floats
            DType::CBF16 => 4, // Two BF16 floats
            DType::U8 => 1,
            DType::U16 => 2,
            DType::U32 => 4,
            DType::U64 => 8,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::Bool => 1,
        }
    }
}

/// Multi-backend tensor representation.
///
/// The `Tensor` enum provides a unified interface over different tensor backends,
/// allowing seamless switching between CPU and GPU computations based on availability
/// and requirements.
///
/// # Variants
///
/// - `F32`: 32-bit floating point tensors (most common for neural networks)
/// - `F64`: 64-bit floating point tensors (for high precision requirements)
/// - `I64`: 64-bit integer tensors (for indices and discrete values)
/// - `Torch`: PyTorch backend (requires `torch` feature)
/// - `Candle`: Candle backend (requires `candle` feature)
///
/// # Backend Selection
///
/// The default backend is ndarray (CPU), which provides good performance for
/// small to medium models. For larger models or when GPU acceleration is needed,
/// enable the `torch` or `candle` features.
///
/// # Example
///
/// ```no_run
/// use trustformers_core::tensor::Tensor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // Create a tensor with default backend
/// let tensor = Tensor::zeros(&[2, 3, 4])?;
/// assert_eq!(tensor.shape(), vec![2, 3, 4]);
/// # Ok(())
/// # }
/// ```

/// Metal GPU buffer wrapper for GPU-resident tensors
#[cfg(feature = "metal")]
#[derive(Debug)]
pub struct MetalTensorData {
    pub buffer_id: crate::gpu_ops::metal::BufferId,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

#[cfg(feature = "metal")]
impl Clone for MetalTensorData {
    fn clone(&self) -> Self {
        // Note: This creates a reference to the same GPU buffer
        // Actual data is not copied - buffer is reference counted
        Self {
            buffer_id: self.buffer_id,
            shape: self.shape.clone(),
            dtype: self.dtype,
        }
    }
}

pub enum Tensor {
    // Standard ndarray types
    F32(ArrayD<f32>),
    F64(ArrayD<f64>),
    F16(ArrayD<half::f16>),
    BF16(ArrayD<half::bf16>),
    I64(ArrayD<i64>),
    // Complex number types
    C32(ArrayD<Complex32>),
    C64(ArrayD<Complex64>),
    CF16(ArrayD<num_complex::Complex<half::f16>>),
    CBF16(ArrayD<num_complex::Complex<half::bf16>>),
    // Sparse tensor variant
    Sparse(crate::sparse_tensor::SparseTensor),
    // GPU support available via hardware acceleration module (CUDA, ROCm, Intel OneAPI, Vulkan, Metal)
    // and backend-specific implementations (Torch, Candle)
    #[cfg(feature = "torch")]
    Torch(tch::Tensor),
    #[cfg(feature = "candle")]
    Candle(candle_core::Tensor),
    // Metal GPU-resident tensor (data lives on GPU)
    #[cfg(feature = "metal")]
    Metal(MetalTensorData),
}

// Manual Clone implementation because tch::Tensor doesn't implement Clone
impl Clone for Tensor {
    fn clone(&self) -> Self {
        match self {
            Tensor::F32(arr) => Tensor::F32(arr.clone()),
            Tensor::F64(arr) => Tensor::F64(arr.clone()),
            Tensor::F16(arr) => Tensor::F16(arr.clone()),
            Tensor::BF16(arr) => Tensor::BF16(arr.clone()),
            Tensor::I64(arr) => Tensor::I64(arr.clone()),
            Tensor::C32(arr) => Tensor::C32(arr.clone()),
            Tensor::C64(arr) => Tensor::C64(arr.clone()),
            Tensor::CF16(arr) => Tensor::CF16(arr.clone()),
            Tensor::CBF16(arr) => Tensor::CBF16(arr.clone()),
            Tensor::Sparse(s) => Tensor::Sparse(s.clone()),
            #[cfg(feature = "torch")]
            Tensor::Torch(t) => Tensor::Torch(t.shallow_clone()),
            #[cfg(feature = "candle")]
            Tensor::Candle(t) => Tensor::Candle(t.clone()),
            #[cfg(feature = "metal")]
            Tensor::Metal(data) => Tensor::Metal(data.clone()),
        }
    }
}

// Manual Debug implementation for Tensor
impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Tensor::F32(_) => write!(f, "Tensor::F32(shape: {:?}, dtype: F32)", self.shape()),
            Tensor::F64(_) => write!(f, "Tensor::F64(shape: {:?}, dtype: F64)", self.shape()),
            Tensor::F16(_) => write!(f, "Tensor::F16(shape: {:?}, dtype: F16)", self.shape()),
            Tensor::BF16(_) => write!(f, "Tensor::BF16(shape: {:?}, dtype: BF16)", self.shape()),
            Tensor::I64(_) => write!(f, "Tensor::I64(shape: {:?}, dtype: I64)", self.shape()),
            Tensor::C32(_) => write!(f, "Tensor::C32(shape: {:?}, dtype: C32)", self.shape()),
            Tensor::C64(_) => write!(f, "Tensor::C64(shape: {:?}, dtype: C64)", self.shape()),
            Tensor::CF16(_) => write!(f, "Tensor::CF16(shape: {:?}, dtype: CF16)", self.shape()),
            Tensor::CBF16(_) => write!(f, "Tensor::CBF16(shape: {:?}, dtype: CBF16)", self.shape()),
            Tensor::Sparse(s) => write!(f, "Tensor::Sparse({:?})", s),
            #[cfg(feature = "torch")]
            Tensor::Torch(_) => write!(f, "Tensor::Torch(shape: {:?})", self.shape()),
            #[cfg(feature = "candle")]
            Tensor::Candle(_) => write!(f, "Tensor::Candle(shape: {:?})", self.shape()),
            #[cfg(feature = "metal")]
            Tensor::Metal(data) => write!(f, "Tensor::Metal(shape: {:?}, dtype: {:?}, buffer_id: {:?})",
                data.shape, data.dtype, data.buffer_id),
        }
    }
}

// Safety: Both PyTorch and Candle backends are internally thread-safe:
// - PyTorch: The tch::Tensor uses reference counting and the underlying data is managed
//   by PyTorch's thread-safe memory allocator. The raw pointer is just an FFI wrapper.
// - Candle: Tensors are designed to be thread-safe with reference-counted storage.
// Multiple threads can safely hold references to the same tensor.
#[cfg(any(feature = "torch", feature = "candle"))]
unsafe impl Sync for Tensor {}

// The implementations are in separate modules but the methods are part of the Tensor impl blocks

impl From<ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>>>
    for Tensor
{
    fn from(arr: ArrayD<f32>) -> Self {
        Tensor::F32(arr)
    }
}

impl From<ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<ndarray::IxDynImpl>>>
    for Tensor
{
    fn from(arr: ArrayD<f64>) -> Self {
        Tensor::F64(arr)
    }
}

// Additional math operations for trait compatibility
impl std::ops::Add for Tensor {
    type Output = Result<Tensor>;

    fn add(self, other: Tensor) -> Self::Output {
        Tensor::add(&self, &other)
    }
}

impl std::ops::Add for &Tensor {
    type Output = Result<Tensor>;

    fn add(self, other: &Tensor) -> Self::Output {
        Tensor::add(self, other)
    }
}

impl std::ops::Add<&&Tensor> for &Tensor {
    type Output = Result<Tensor>;

    fn add(self, other: &&Tensor) -> Self::Output {
        Tensor::add(self, other)
    }
}

impl std::ops::Add<&Tensor> for &&Tensor {
    type Output = Result<Tensor>;

    fn add(self, other: &Tensor) -> Self::Output {
        Tensor::add(self, other)
    }
}

impl std::ops::Sub for Tensor {
    type Output = Result<Tensor>;

    fn sub(self, other: Tensor) -> Self::Output {
        Tensor::sub(&self, &other)
    }
}

// Scalar multiplication operators
impl std::ops::Mul<f32> for Tensor {
    type Output = Result<Tensor>;

    fn mul(self, scalar: f32) -> Self::Output {
        self.scalar_mul(scalar)
    }
}

impl std::ops::Mul<f32> for &Tensor {
    type Output = Result<Tensor>;

    fn mul(self, scalar: f32) -> Self::Output {
        self.scalar_mul(scalar)
    }
}

impl std::ops::Mul<f64> for Tensor {
    type Output = Result<Tensor>;

    fn mul(self, scalar: f64) -> Self::Output {
        self.scalar_mul(scalar as f32)
    }
}

impl std::ops::Mul<f64> for &Tensor {
    type Output = Result<Tensor>;

    fn mul(self, scalar: f64) -> Self::Output {
        self.scalar_mul(scalar as f32)
    }
}

// Element-wise multiplication with another tensor
impl std::ops::Mul<&Tensor> for &Tensor {
    type Output = Result<Tensor>;

    fn mul(self, other: &Tensor) -> Self::Output {
        Tensor::mul(self, other)
    }
}

impl std::ops::Mul<Tensor> for &Tensor {
    type Output = Result<Tensor>;

    fn mul(self, other: Tensor) -> Self::Output {
        Tensor::mul(self, &other)
    }
}

impl std::ops::Mul<&Tensor> for Tensor {
    type Output = Result<Tensor>;

    fn mul(self, other: &Tensor) -> Self::Output {
        Tensor::mul(&self, other)
    }
}

// Scalar division operators
impl std::ops::Div<f32> for Tensor {
    type Output = Result<Tensor>;

    fn div(self, scalar: f32) -> Self::Output {
        self.scalar_div(scalar)
    }
}

impl std::ops::Div<f32> for &Tensor {
    type Output = Result<Tensor>;

    fn div(self, scalar: f32) -> Self::Output {
        self.scalar_div(scalar)
    }
}

impl std::ops::Div<f64> for Tensor {
    type Output = Result<Tensor>;

    fn div(self, scalar: f64) -> Self::Output {
        self.scalar_div(scalar as f32)
    }
}

impl std::ops::Div<f64> for &Tensor {
    type Output = Result<Tensor>;

    fn div(self, scalar: f64) -> Self::Output {
        self.scalar_div(scalar as f32)
    }
}

impl std::ops::Div<f64> for &&Tensor {
    type Output = Result<Tensor>;

    fn div(self, scalar: f64) -> Self::Output {
        (*self).scalar_div(scalar as f32)
    }
}

// Tensor subtraction operators
impl std::ops::Sub for &Tensor {
    type Output = Result<Tensor>;

    fn sub(self, other: &Tensor) -> Self::Output {
        Tensor::sub(self, other)
    }
}

// Type alias for backward compatibility
pub type TensorType = DType;

// Re-export expression template types
pub use expression::{EvalContext, ExprNode, OpType, OptimizationHints, TensorExpr};

// Re-export gradient tracking utilities
pub use utils::{clear_gradients, disable_grad, enable_grad, is_grad_enabled};
