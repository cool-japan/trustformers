//! Utility functions for tensors with SIMD optimizations.
//!
//! This module contains comprehensive utility functions for tensor manipulation,
//! element access, comparison, transformation, and validation. It includes:
//!
//! * **Tensor Manipulation**: clamp, scale, broadcast_to, clip_grad_norm
//! * **Element Access**: get_scalar, set_scalar
//! * **Tensor Comparison**: greater
//! * **Tensor Transformation**: lerp (linear interpolation)
//! * **Tensor Validation**: isnan, isinf, isfinite
//! * **Tensor Statistics**: min_max, sign
//! * **Tensor Arithmetic**: add_scaled, sub_scaled
//!
//! The SIMD functions provide hardware-accelerated implementations for
//! statistical operations when available (AVX2, SSE2), with fallback
//! to scalar implementations on unsupported platforms.

#![allow(unused_variables)] // SIMD utilities with architecture-specific code

use super::super::Tensor;
use crate::errors::{Result, TrustformersError};
use scirs2_core::ndarray::{IxDyn, Zip};
use std::sync::OnceLock;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// Cached CPU features for optimal performance
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // Fields used in feature-gated code
struct CachedCpuFeatures {
    avx2: bool,
    sse2: bool,
}

static CPU_FEATURES: OnceLock<CachedCpuFeatures> = OnceLock::new();

/// Get or initialize cached CPU features
fn get_cpu_features() -> &'static CachedCpuFeatures {
    CPU_FEATURES.get_or_init(|| {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            CachedCpuFeatures {
                avx2: is_x86_feature_detected!("avx2"),
                sse2: is_x86_feature_detected!("sse2"),
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            CachedCpuFeatures {
                avx2: false,
                sse2: false,
            }
        }
    })
}

impl Tensor {
    // ================================
    // TENSOR MANIPULATION
    // ================================

    /// Scaling operation.
    pub fn scale(&self, factor: f32) -> Result<Tensor> {
        self.scalar_mul(factor)
    }

    /// Clamp values to a range.
    pub fn clamp(&self, min_val: f32, max_val: f32) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| x.clamp(min_val, max_val));
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| x.clamp(min_val as f64, max_val as f64));
                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Clamp not supported for this tensor type",
                "clamp",
            )),
        }
    }

    /// Broadcast tensor to a target shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The target shape to broadcast to
    ///
    /// # Returns
    ///
    /// A new tensor with the broadcasted shape.
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a
                    .broadcast(IxDyn(shape))
                    .ok_or_else(|| {
                        TrustformersError::shape_error("Broadcasting failed".to_string())
                    })?
                    .to_owned();
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a
                    .broadcast(IxDyn(shape))
                    .ok_or_else(|| {
                        TrustformersError::shape_error("Broadcasting failed".to_string())
                    })?
                    .to_owned();
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                let result = a
                    .broadcast(IxDyn(shape))
                    .ok_or_else(|| {
                        TrustformersError::shape_error("Broadcasting failed".to_string())
                    })?
                    .to_owned();
                Ok(Tensor::I64(result))
            },
            Tensor::C32(a) => {
                let result = a
                    .broadcast(IxDyn(shape))
                    .ok_or_else(|| {
                        TrustformersError::shape_error("Broadcasting failed".to_string())
                    })?
                    .to_owned();
                Ok(Tensor::C32(result))
            },
            Tensor::C64(a) => {
                let result = a
                    .broadcast(IxDyn(shape))
                    .ok_or_else(|| {
                        TrustformersError::shape_error("Broadcasting failed".to_string())
                    })?
                    .to_owned();
                Ok(Tensor::C64(result))
            },
            Tensor::F16(a) => {
                let result = a
                    .broadcast(IxDyn(shape))
                    .ok_or_else(|| {
                        TrustformersError::shape_error("Broadcasting failed".to_string())
                    })?
                    .to_owned();
                Ok(Tensor::F16(result))
            },
            Tensor::BF16(a) => {
                let result = a
                    .broadcast(IxDyn(shape))
                    .ok_or_else(|| {
                        TrustformersError::shape_error("Broadcasting failed".to_string())
                    })?
                    .to_owned();
                Ok(Tensor::BF16(result))
            },
            Tensor::CF16(a) => {
                let result = a
                    .broadcast(IxDyn(shape))
                    .ok_or_else(|| {
                        TrustformersError::shape_error("Broadcasting failed".to_string())
                    })?
                    .to_owned();
                Ok(Tensor::CF16(result))
            },
            Tensor::CBF16(a) => {
                let result = a
                    .broadcast(IxDyn(shape))
                    .ok_or_else(|| {
                        TrustformersError::shape_error("Broadcasting failed".to_string())
                    })?
                    .to_owned();
                Ok(Tensor::CBF16(result))
            },
            _ => Err(TrustformersError::invalid_operation(
                "broadcast_to operation not implemented for this tensor type".into(),
            )),
        }
    }

    // ================================
    // ELEMENT ACCESS
    // ================================

    /// Get a scalar value at the specified index.
    ///
    /// # Arguments
    ///
    /// * `indices` - The indices to get the scalar value from
    ///
    /// # Returns
    ///
    /// The scalar value at the specified index.
    pub fn get_scalar(&self, indices: &[usize]) -> Result<f32> {
        match self {
            Tensor::F32(a) => {
                if indices.len() != a.ndim() {
                    return Err(TrustformersError::shape_error(format!(
                        "Index dimension mismatch: expected {}, got {}",
                        a.ndim(),
                        indices.len()
                    )));
                }

                // Convert to IxDyn
                let ix = IxDyn(indices);
                match a.get(ix) {
                    Some(val) => Ok(*val),
                    None => Err(TrustformersError::shape_error("Index out of bounds".into())),
                }
            },
            Tensor::F64(a) => {
                if indices.len() != a.ndim() {
                    return Err(TrustformersError::shape_error(format!(
                        "Index dimension mismatch: expected {}, got {}",
                        a.ndim(),
                        indices.len()
                    )));
                }

                let ix = IxDyn(indices);
                match a.get(ix) {
                    Some(val) => Ok(*val as f32),
                    None => Err(TrustformersError::shape_error("Index out of bounds".into())),
                }
            },
            _ => Err(TrustformersError::tensor_op_error(
                "get_scalar not supported for this tensor type",
                "get_scalar",
            )),
        }
    }

    /// Set a scalar value at the specified index.
    ///
    /// # Arguments
    ///
    /// * `indices` - The indices to set the scalar value
    /// * `value` - The value to set
    ///
    /// # Returns
    ///
    /// A new tensor with the value set at the specified index.
    pub fn set_scalar(&self, indices: &[usize], value: f32) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                if indices.len() != a.ndim() {
                    return Err(TrustformersError::shape_error(format!(
                        "Index dimension mismatch: expected {}, got {}",
                        a.ndim(),
                        indices.len()
                    )));
                }

                let mut result = a.clone();
                let ix = IxDyn(indices);

                // Check bounds
                if !indices.iter().zip(a.shape()).all(|(&idx, &dim)| idx < dim) {
                    return Err(TrustformersError::shape_error("Index out of bounds".into()));
                }

                result[ix] = value;
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                if indices.len() != a.ndim() {
                    return Err(TrustformersError::shape_error(format!(
                        "Index dimension mismatch: expected {}, got {}",
                        a.ndim(),
                        indices.len()
                    )));
                }

                let mut result = a.clone();
                let ix = IxDyn(indices);

                // Check bounds
                if !indices.iter().zip(a.shape()).all(|(&idx, &dim)| idx < dim) {
                    return Err(TrustformersError::shape_error("Index out of bounds".into()));
                }

                result[ix] = value as f64;
                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "set_scalar not supported for this tensor type",
                "set_scalar",
            )),
        }
    }

    // ================================
    // TENSOR COMPARISON
    // ================================

    /// Element-wise greater than comparison.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to compare against
    ///
    /// # Returns
    ///
    /// A tensor with 1.0 where self > other, 0.0 otherwise.
    pub fn greater(&self, other: &Tensor) -> Result<Tensor> {
        match (self, other) {
            (Tensor::F32(a), Tensor::F32(b)) => {
                if a.shape() != b.shape() {
                    return Err(TrustformersError::shape_error(
                        "Tensors must have the same shape for greater comparison".to_string(),
                    ));
                }
                let result =
                    Zip::from(a).and(b).map_collect(|&x, &y| if x > y { 1.0f32 } else { 0.0f32 });
                Ok(Tensor::F32(result))
            },
            (Tensor::F64(a), Tensor::F64(b)) => {
                if a.shape() != b.shape() {
                    return Err(TrustformersError::shape_error(
                        "Tensors must have the same shape for greater comparison".to_string(),
                    ));
                }
                let result =
                    Zip::from(a).and(b).map_collect(|&x, &y| if x > y { 1.0f64 } else { 0.0f64 });
                Ok(Tensor::F64(result))
            },
            (Tensor::I64(a), Tensor::I64(b)) => {
                if a.shape() != b.shape() {
                    return Err(TrustformersError::shape_error(
                        "Tensors must have the same shape for greater comparison".to_string(),
                    ));
                }
                let result =
                    Zip::from(a).and(b).map_collect(|&x, &y| if x > y { 1i64 } else { 0i64 });
                Ok(Tensor::I64(result))
            },
            (Tensor::F16(a), Tensor::F16(b)) => {
                if a.shape() != b.shape() {
                    return Err(TrustformersError::shape_error(
                        "Tensors must have the same shape for greater comparison".to_string(),
                    ));
                }
                let result = Zip::from(a).and(b).map_collect(|&x, &y| {
                    if x.to_f32() > y.to_f32() {
                        half::f16::from_f32(1.0)
                    } else {
                        half::f16::from_f32(0.0)
                    }
                });
                Ok(Tensor::F16(result))
            },
            (Tensor::BF16(a), Tensor::BF16(b)) => {
                if a.shape() != b.shape() {
                    return Err(TrustformersError::shape_error(
                        "Tensors must have the same shape for greater comparison".to_string(),
                    ));
                }
                let result = Zip::from(a).and(b).map_collect(|&x, &y| {
                    if x.to_f32() > y.to_f32() {
                        half::bf16::from_f32(1.0)
                    } else {
                        half::bf16::from_f32(0.0)
                    }
                });
                Ok(Tensor::BF16(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Greater operation not implemented for this tensor type combination",
                "greater",
            )),
        }
    }

    // ================================
    // TENSOR TRANSFORMATION
    // ================================

    /// Linear interpolation between two tensors.
    ///
    /// Computes: self * (1 - weight) + other * weight
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to interpolate towards
    /// * `weight` - Interpolation weight (must be between 0.0 and 1.0)
    ///
    /// # Returns
    ///
    /// A tensor interpolated between self and other.
    pub fn lerp(&self, other: &Tensor, weight: f32) -> Result<Tensor> {
        if !(0.0..=1.0).contains(&weight) {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Interpolation weight must be between 0.0 and 1.0, got {}",
                    weight
                ),
                "lerp",
            ));
        }

        match (self, other) {
            (Tensor::F32(a), Tensor::F32(b)) => {
                if a.shape() != b.shape() {
                    return Err(TrustformersError::shape_error(format!(
                        "Shape mismatch for interpolation: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }

                // Compute: a + weight * (b - a) = a * (1 - weight) + b * weight
                let diff = b - a;
                let weighted_diff = &diff * weight;
                let result = a + &weighted_diff;
                Ok(Tensor::F32(result))
            },
            (Tensor::F64(a), Tensor::F64(b)) => {
                if a.shape() != b.shape() {
                    return Err(TrustformersError::shape_error(format!(
                        "Shape mismatch for interpolation: {:?} vs {:?}",
                        a.shape(),
                        b.shape()
                    )));
                }

                let weight_f64 = weight as f64;
                let diff = b - a;
                let weighted_diff = &diff * weight_f64;
                let result = a + &weighted_diff;
                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Linear interpolation only supported for F32 and F64 tensors",
                "lerp",
            )),
        }
    }

    // ================================
    // TENSOR VALIDATION
    // ================================

    // ================================
    // TENSOR ARITHMETIC
    // ================================
}

/// SIMD-optimized min/max for f32 arrays with hardware acceleration.
///
/// Automatically selects the best implementation based on available CPU features:
/// - AVX2 for modern processors (8 elements per cycle)
/// - SSE2 for older x86 processors (4 elements per cycle)
/// - Scalar fallback for other architectures
///
/// # Arguments
///
/// * `data` - Input f32 slice to find min/max values
///
/// # Returns
///
/// A tuple containing (min_value, max_value), or (NaN, NaN) for empty input.
pub fn simd_min_max_f32(data: &[f32]) -> (f32, f32) {
    if data.is_empty() {
        return (f32::NAN, f32::NAN);
    }

    let features = get_cpu_features();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if features.avx2 {
            return unsafe { simd_min_max_f32_avx2(data) };
        } else if features.sse2 {
            return unsafe { simd_min_max_f32_sse2(data) };
        }
    }

    // Fallback to scalar implementation
    let min_val = data.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
    let max_val = data.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    (min_val, max_val)
}

/// SIMD-optimized min/max for f64 arrays with hardware acceleration.
///
/// Automatically selects the best implementation based on available CPU features:
/// - AVX2 for modern processors (4 f64 elements per cycle)
/// - SSE2 for older x86 processors (2 f64 elements per cycle)
/// - Scalar fallback for other architectures
///
/// # Arguments
///
/// * `data` - Input f64 slice to find min/max values
///
/// # Returns
///
/// A tuple containing (min_value, max_value), or (NaN, NaN) for empty input.
pub fn simd_min_max_f64(data: &[f64]) -> (f64, f64) {
    if data.is_empty() {
        return (f64::NAN, f64::NAN);
    }

    let features = get_cpu_features();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if features.avx2 {
            return unsafe { simd_min_max_f64_avx2(data) };
        } else if features.sse2 {
            return unsafe { simd_min_max_f64_sse2(data) };
        }
    }

    // Fallback to scalar implementation
    let min_val = data.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
    let max_val = data.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
    (min_val, max_val)
}

// SIMD implementation functions for x86/x86_64 architectures

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn simd_min_max_f32_avx2(data: &[f32]) -> (f32, f32) {
    let chunk_size = 8; // AVX2 processes 8 f32s at once
    let mut min_vec = _mm256_set1_ps(f32::INFINITY);
    let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);

    // Process chunks of 8 elements
    let chunks = data.len() / chunk_size;
    let ptr = data.as_ptr();

    for i in 0..chunks {
        let values = _mm256_loadu_ps(ptr.add(i * chunk_size));
        min_vec = _mm256_min_ps(min_vec, values);
        max_vec = _mm256_max_ps(max_vec, values);
    }

    // Horizontal min/max reduction
    let min_vals = std::mem::transmute::<__m256, [f32; 8]>(min_vec);
    let max_vals = std::mem::transmute::<__m256, [f32; 8]>(max_vec);

    let mut min_result = min_vals[0];
    let mut max_result = max_vals[0];

    for i in 1..8 {
        min_result = min_result.min(min_vals[i]);
        max_result = max_result.max(max_vals[i]);
    }

    // Handle remaining elements
    for &val in &data[chunks * chunk_size..] {
        min_result = min_result.min(val);
        max_result = max_result.max(val);
    }

    (min_result, max_result)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn simd_min_max_f32_sse2(data: &[f32]) -> (f32, f32) {
    let chunk_size = 4; // SSE2 processes 4 f32s at once
    let mut min_vec = _mm_set1_ps(f32::INFINITY);
    let mut max_vec = _mm_set1_ps(f32::NEG_INFINITY);

    let chunks = data.len() / chunk_size;
    let ptr = data.as_ptr();

    for i in 0..chunks {
        let values = _mm_loadu_ps(ptr.add(i * chunk_size));
        min_vec = _mm_min_ps(min_vec, values);
        max_vec = _mm_max_ps(max_vec, values);
    }

    // Horizontal reduction
    let min_vals = std::mem::transmute::<__m128, [f32; 4]>(min_vec);
    let max_vals = std::mem::transmute::<__m128, [f32; 4]>(max_vec);

    let mut min_result = min_vals[0];
    let mut max_result = max_vals[0];

    for i in 1..4 {
        min_result = min_result.min(min_vals[i]);
        max_result = max_result.max(max_vals[i]);
    }

    // Handle remaining elements
    for &val in &data[chunks * chunk_size..] {
        min_result = min_result.min(val);
        max_result = max_result.max(val);
    }

    (min_result, max_result)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn simd_min_max_f64_avx2(data: &[f64]) -> (f64, f64) {
    let chunk_size = 4; // AVX2 processes 4 f64s at once
    let mut min_vec = _mm256_set1_pd(f64::INFINITY);
    let mut max_vec = _mm256_set1_pd(f64::NEG_INFINITY);

    let chunks = data.len() / chunk_size;
    let ptr = data.as_ptr();

    for i in 0..chunks {
        let values = _mm256_loadu_pd(ptr.add(i * chunk_size));
        min_vec = _mm256_min_pd(min_vec, values);
        max_vec = _mm256_max_pd(max_vec, values);
    }

    // Horizontal reduction
    let min_vals = std::mem::transmute::<__m256d, [f64; 4]>(min_vec);
    let max_vals = std::mem::transmute::<__m256d, [f64; 4]>(max_vec);

    let mut min_result = min_vals[0];
    let mut max_result = max_vals[0];

    for i in 1..4 {
        min_result = min_result.min(min_vals[i]);
        max_result = max_result.max(max_vals[i]);
    }

    // Handle remaining elements
    for &val in &data[chunks * chunk_size..] {
        min_result = min_result.min(val);
        max_result = max_result.max(val);
    }

    (min_result, max_result)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn simd_min_max_f64_sse2(data: &[f64]) -> (f64, f64) {
    let chunk_size = 2; // SSE2 processes 2 f64s at once
    let mut min_vec = _mm_set1_pd(f64::INFINITY);
    let mut max_vec = _mm_set1_pd(f64::NEG_INFINITY);

    let chunks = data.len() / chunk_size;
    let ptr = data.as_ptr();

    for i in 0..chunks {
        let values = _mm_loadu_pd(ptr.add(i * chunk_size));
        min_vec = _mm_min_pd(min_vec, values);
        max_vec = _mm_max_pd(max_vec, values);
    }

    // Horizontal reduction
    let min_vals = std::mem::transmute::<__m128d, [f64; 2]>(min_vec);
    let max_vals = std::mem::transmute::<__m128d, [f64; 2]>(max_vec);

    let mut min_result = min_vals[0].min(min_vals[1]);
    let mut max_result = max_vals[0].max(max_vals[1]);

    // Handle remaining elements
    for &val in &data[chunks * chunk_size..] {
        min_result = min_result.min(val);
        max_result = max_result.max(val);
    }

    (min_result, max_result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::Result;
    use crate::tensor::Tensor;

    #[test]
    fn test_scale() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3])?;
        let s = t.scale(2.0)?;
        let data = s.data()?;
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 4.0).abs() < 1e-6);
        assert!((data[2] - 6.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_clamp() -> Result<()> {
        let t = Tensor::from_data(vec![-5.0, 0.0, 5.0, 10.0], &[4])?;
        let c = t.clamp(0.0, 7.0)?;
        let data = c.data()?;
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[1] - 0.0).abs() < 1e-6);
        assert!((data[2] - 5.0).abs() < 1e-6);
        assert!((data[3] - 7.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_clamp_all_within() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3])?;
        let c = t.clamp(0.0, 10.0)?;
        let data = c.data()?;
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_broadcast_to_f32() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &[1, 3])?;
        let b = t.broadcast_to(&[4, 3])?;
        assert_eq!(b.shape(), vec![4, 3]);
        Ok(())
    }

    #[test]
    fn test_broadcast_to_scalar() -> Result<()> {
        let t = Tensor::from_data(vec![5.0], &[1])?;
        let b = t.broadcast_to(&[4])?;
        assert_eq!(b.shape(), vec![4]);
        let data = b.data()?;
        for val in &data {
            assert!((val - 5.0).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_get_scalar() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let v = t.get_scalar(&[0, 1])?;
        assert!((v - 2.0).abs() < 1e-6);
        let v2 = t.get_scalar(&[1, 0])?;
        assert!((v2 - 3.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_get_scalar_out_of_bounds() {
        let t = Tensor::from_data(vec![1.0, 2.0], &[2]).expect("create failed");
        let result = t.get_scalar(&[5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_scalar_wrong_dims() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("create failed");
        let result = t.get_scalar(&[0]); // need 2 indices
        assert!(result.is_err());
    }

    #[test]
    fn test_set_scalar() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let t2 = t.set_scalar(&[0, 1], 99.0)?;
        let v = t2.get_scalar(&[0, 1])?;
        assert!((v - 99.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_set_scalar_preserves_other() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let t2 = t.set_scalar(&[1, 1], 99.0)?;
        let v = t2.get_scalar(&[0, 0])?;
        assert!((v - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_greater() -> Result<()> {
        let a = Tensor::from_data(vec![1.0, 5.0, 3.0], &[3])?;
        let b = Tensor::from_data(vec![2.0, 4.0, 3.0], &[3])?;
        let result = a.greater(&b)?;
        let data = result.data()?;
        assert!((data[0] - 0.0).abs() < 1e-6); // 1 > 2 = false
        assert!((data[1] - 1.0).abs() < 1e-6); // 5 > 4 = true
        assert!((data[2] - 0.0).abs() < 1e-6); // 3 > 3 = false
        Ok(())
    }

    #[test]
    fn test_lerp() -> Result<()> {
        let a = Tensor::from_data(vec![0.0, 0.0], &[2])?;
        let b = Tensor::from_data(vec![10.0, 20.0], &[2])?;
        let c = a.lerp(&b, 0.5)?;
        let data = c.data()?;
        assert!((data[0] - 5.0).abs() < 1e-5);
        assert!((data[1] - 10.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_lerp_weight_zero() -> Result<()> {
        let a = Tensor::from_data(vec![1.0, 2.0], &[2])?;
        let b = Tensor::from_data(vec![10.0, 20.0], &[2])?;
        let c = a.lerp(&b, 0.0)?;
        let data = c.data()?;
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_lerp_weight_one() -> Result<()> {
        let a = Tensor::from_data(vec![1.0, 2.0], &[2])?;
        let b = Tensor::from_data(vec![10.0, 20.0], &[2])?;
        let c = a.lerp(&b, 1.0)?;
        let data = c.data()?;
        assert!((data[0] - 10.0).abs() < 1e-5);
        assert!((data[1] - 20.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_simd_min_max_f32_basic() {
        let data = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let (min_val, max_val) = simd_min_max_f32(&data);
        assert!((min_val - 1.0).abs() < 1e-6);
        assert!((max_val - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_min_max_f32_single() {
        let data = vec![42.0f32];
        let (min_val, max_val) = simd_min_max_f32(&data);
        assert!((min_val - 42.0).abs() < 1e-6);
        assert!((max_val - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_simd_min_max_f32_negative() {
        let data = vec![-5.0f32, -1.0, -10.0, -3.0];
        let (min_val, max_val) = simd_min_max_f32(&data);
        assert!((min_val - (-10.0)).abs() < 1e-6);
        assert!((max_val - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_simd_min_max_f64_basic() {
        let data = vec![3.0f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let (min_val, max_val) = simd_min_max_f64(&data);
        assert!((min_val - 1.0).abs() < 1e-10);
        assert!((max_val - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_simd_min_max_f64_single() {
        let data = vec![7.0f64];
        let (min_val, max_val) = simd_min_max_f64(&data);
        assert!((min_val - 7.0).abs() < 1e-10);
        assert!((max_val - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_broadcast_to_i64() -> Result<()> {
        let t = Tensor::from_vec_i64(vec![1, 2, 3], &[1, 3])?;
        let b = t.broadcast_to(&[3, 3])?;
        assert_eq!(b.shape(), vec![3, 3]);
        Ok(())
    }

    #[test]
    fn test_scale_negative() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0], &[2])?;
        let s = t.scale(-1.0)?;
        let data = s.data()?;
        assert!((data[0] - (-1.0)).abs() < 1e-6);
        assert!((data[1] - (-2.0)).abs() < 1e-6);
        Ok(())
    }
}
