//! Statistical operations for tensors with numerical stability enhancements.
//!
//! This module contains comprehensive statistical operations including:
//! - **Aggregation functions**: sum, mean with flexible axis support
//! - **Descriptive statistics**: variance, standard deviation with robust calculation
//! - **Extremal operations**: min/max with optimized SIMD implementations
//! - **Index operations**: argmax with negative axis support
//! - **Utility functions**: min_max with optimized performance
//!
//! All operations include numerical stability features, comprehensive error handling,
//! and support for F32 and F64 tensor types where applicable.

use super::super::Tensor;
use super::utilities::{simd_min_max_f32, simd_min_max_f64};
use crate::errors::{Result, TrustformersError};
use scirs2_core::ndarray::{arr0, ArrayD, Axis, IxDyn, Zip};

/// Upcast a half-precision (F16/BF16) tensor to F32, run `op`, then downcast the
/// F32 result back to the original half-precision dtype.
///
/// Half-precision accumulation is lossy for reductions (sum/mean/variance), so we
/// upcast to F32, run the reduction there, and round the result back to F16/BF16 so
/// the output dtype matches the input dtype. Results that are not F32 (e.g. integer
/// index tensors) are returned as-is by the caller via a dedicated match arm.
fn run_half_in_f32<F>(input: &Tensor, op: F) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    match input {
        Tensor::F16(a) => {
            // Upcast F16 -> F32 for accurate reduction, then downcast back to F16.
            let upcast = Tensor::F32(a.mapv(|x| x.to_f32()));
            match op(&upcast)? {
                Tensor::F32(r) => Ok(Tensor::F16(r.mapv(half::f16::from_f32))),
                other => other.to_dtype(crate::tensor::DType::F16),
            }
        },
        Tensor::BF16(a) => {
            // Upcast BF16 -> F32 for accurate reduction, then downcast back to BF16.
            let upcast = Tensor::F32(a.mapv(|x| x.to_f32()));
            match op(&upcast)? {
                Tensor::F32(r) => Ok(Tensor::BF16(r.mapv(half::bf16::from_f32))),
                other => other.to_dtype(crate::tensor::DType::BF16),
            }
        },
        _ => Err(TrustformersError::tensor_op_error(
            "run_half_in_f32 called on a non-half-precision tensor",
            "run_half_in_f32",
        )),
    }
}

/// Upcast a half-precision tensor to F32 and run an `op` whose result dtype is
/// intentionally independent of the input dtype (e.g. boolean masks or integer
/// index tensors). The result is returned unchanged (no downcast).
fn run_half_in_f32_keep<F>(input: &Tensor, op: F) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    match input {
        Tensor::F16(a) => op(&Tensor::F32(a.mapv(|x| x.to_f32()))),
        Tensor::BF16(a) => op(&Tensor::F32(a.mapv(|x| x.to_f32()))),
        _ => Err(TrustformersError::tensor_op_error(
            "run_half_in_f32_keep called on a non-half-precision tensor",
            "run_half_in_f32_keep",
        )),
    }
}

impl Tensor {
    /// Standard deviation across all elements.
    ///
    /// Computes the standard deviation of all elements in the tensor,
    /// returning a scalar tensor containing the result.
    ///
    /// # Returns
    ///
    /// A scalar tensor containing the standard deviation.
    pub fn std(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let mean = a.mean().ok_or_else(|| {
                    crate::errors::compute_error("std", "Mean calculation failed")
                })?;
                let variance = a.mapv(|x| (x - mean).powi(2)).mean().ok_or_else(|| {
                    crate::errors::compute_error("std", "Mean calculation failed")
                })?;
                let std = variance.sqrt();
                Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), std)))
            },
            Tensor::F64(a) => {
                let mean = a.mean().ok_or_else(|| {
                    crate::errors::compute_error("std", "Mean calculation failed")
                })?;
                let variance = a.mapv(|x| (x - mean).powi(2)).mean().ok_or_else(|| {
                    crate::errors::compute_error("std", "Mean calculation failed")
                })?;
                let std = variance.sqrt();
                Ok(Tensor::F64(ArrayD::from_elem(IxDyn(&[]), std)))
            },
            Tensor::F16(_) | Tensor::BF16(_) => run_half_in_f32(self, |t| t.std()),
            _ => Err(TrustformersError::tensor_op_error(
                "Standard deviation not supported for this tensor type",
                "std",
            )),
        }
    }

    /// Maximum value.
    pub fn max_value(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let max_val = a.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), max_val)))
            },
            Tensor::F16(_) | Tensor::BF16(_) => run_half_in_f32(self, |t| t.max_value()),
            _ => Err(TrustformersError::tensor_op_error(
                "Max not supported for this tensor type",
                "max_value",
            )),
        }
    }

    /// Element-wise maximum between two tensors.
    pub fn max(&self, other: &Tensor) -> Result<Tensor> {
        match (self, other) {
            (Tensor::F32(a), Tensor::F32(b)) => {
                // Handle scalar case (0-dimensional tensor)
                if a.ndim() == 0 && b.ndim() > 0 {
                    // a is scalar, broadcast to b's shape
                    let scalar_val = a.iter().next().ok_or_else(|| {
                        crate::errors::compute_error("max", "array must have at least one element")
                    })?;
                    let result = b.mapv(|x| x.max(*scalar_val));
                    Ok(Tensor::F32(result))
                } else if b.ndim() == 0 && a.ndim() > 0 {
                    // b is scalar, broadcast to a's shape
                    let scalar_val = b.iter().next().ok_or_else(|| {
                        crate::errors::compute_error("max", "array must have at least one element")
                    })?;
                    let result = a.mapv(|x| x.max(*scalar_val));
                    Ok(Tensor::F32(result))
                } else if a.ndim() == 0 && b.ndim() == 0 {
                    // Both scalars
                    let a_val = a.iter().next().ok_or_else(|| {
                        crate::errors::compute_error("max", "array must have at least one element")
                    })?;
                    let b_val = b.iter().next().ok_or_else(|| {
                        crate::errors::compute_error("max", "array must have at least one element")
                    })?;
                    let max_val = a_val.max(*b_val);
                    Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), max_val)))
                } else {
                    // Regular element-wise operation
                    let result = Zip::from(a).and(b).map_collect(|&x, &y| x.max(y));
                    Ok(Tensor::F32(result))
                }
            },
            (Tensor::F64(a), Tensor::F64(b)) => {
                // Handle scalar case (0-dimensional tensor)
                if a.ndim() == 0 && b.ndim() > 0 {
                    // a is scalar, broadcast to b's shape
                    let scalar_val = a.iter().next().ok_or_else(|| {
                        crate::errors::compute_error("max", "array must have at least one element")
                    })?;
                    let result = b.mapv(|x| x.max(*scalar_val));
                    Ok(Tensor::F64(result))
                } else if b.ndim() == 0 && a.ndim() > 0 {
                    // b is scalar, broadcast to a's shape
                    let scalar_val = b.iter().next().ok_or_else(|| {
                        crate::errors::compute_error("max", "array must have at least one element")
                    })?;
                    let result = a.mapv(|x| x.max(*scalar_val));
                    Ok(Tensor::F64(result))
                } else if a.ndim() == 0 && b.ndim() == 0 {
                    // Both scalars
                    let a_val = a.iter().next().ok_or_else(|| {
                        crate::errors::compute_error("max", "array must have at least one element")
                    })?;
                    let b_val = b.iter().next().ok_or_else(|| {
                        crate::errors::compute_error("max", "array must have at least one element")
                    })?;
                    let max_val = a_val.max(*b_val);
                    Ok(Tensor::F64(ArrayD::from_elem(IxDyn(&[]), max_val)))
                } else {
                    // Regular element-wise operation
                    let result = Zip::from(a).and(b).map_collect(|&x, &y| x.max(y));
                    Ok(Tensor::F64(result))
                }
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Element-wise max not supported for these tensor types",
                "max",
            )),
        }
    }

    /// Find the indices of maximum values along the specified axis.
    ///
    /// # Arguments
    /// * `axis` - The axis along which to find the maximum indices. Negative values count from the last axis.
    ///
    /// # Returns
    /// A tensor containing the indices of maximum values along the specified axis.
    pub fn argmax(&self, axis: i32) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let ndim = a.ndim();
                let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };

                if axis >= ndim {
                    return Err(TrustformersError::tensor_op_error(
                        &format!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            axis, ndim
                        ),
                        "argmax",
                    ));
                }

                let indices = a.map_axis(Axis(axis), |lane| {
                    lane.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i as f32)
                        .unwrap_or(0.0)
                });

                Ok(Tensor::F32(indices))
            },
            Tensor::F64(a) => {
                let ndim = a.ndim();
                let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };

                if axis >= ndim {
                    return Err(TrustformersError::tensor_op_error(
                        &format!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            axis, ndim
                        ),
                        "argmax",
                    ));
                }

                let indices = a.map_axis(Axis(axis), |lane| {
                    lane.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| {
                            a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i as f64)
                        .unwrap_or(0.0)
                });

                Ok(Tensor::F64(indices))
            },
            Tensor::F16(_) | Tensor::BF16(_) => run_half_in_f32(self, |t| t.argmax(axis)),
            _ => Err(TrustformersError::tensor_op_error(
                "Argmax not supported for this tensor type",
                "argmax",
            )),
        }
    }

    /// Mean value across all elements.
    ///
    /// Computes the arithmetic mean of all elements in the tensor,
    /// returning a scalar tensor containing the result.
    ///
    /// # Returns
    ///
    /// A scalar tensor containing the mean value.
    pub fn mean(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let mean = a.mean().ok_or_else(|| {
                    crate::errors::compute_error("mean", "Mean calculation failed")
                })?;
                Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), mean)))
            },
            Tensor::F64(a) => {
                let mean = a.mean().ok_or_else(|| {
                    crate::errors::compute_error("mean", "Mean calculation failed")
                })?;
                Ok(Tensor::F64(ArrayD::from_elem(IxDyn(&[]), mean)))
            },
            Tensor::F16(_) | Tensor::BF16(_) => run_half_in_f32(self, |t| t.mean()),
            _ => Err(TrustformersError::tensor_op_error(
                "Mean not supported for this tensor type",
                "mean",
            )),
        }
    }

    /// Find minimum and maximum values.
    pub fn min_max(&self) -> Result<(f32, f32)> {
        match self {
            Tensor::F32(a) => {
                let data = a.as_slice().ok_or_else(|| {
                    crate::errors::compute_error("min_max", "array must have contiguous layout")
                })?;
                let (min_val, max_val) = simd_min_max_f32(data);
                Ok((min_val, max_val))
            },
            Tensor::F64(a) => {
                let data = a.as_slice().ok_or_else(|| {
                    crate::errors::compute_error("min_max", "array must have contiguous layout")
                })?;
                let (min_val, max_val) = simd_min_max_f64(data);
                Ok((min_val as f32, max_val as f32))
            },
            Tensor::F16(a) => {
                // Upcast F16 -> F32 to reuse the SIMD min/max kernel.
                let data: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
                let (min_val, max_val) = simd_min_max_f32(&data);
                Ok((min_val, max_val))
            },
            Tensor::BF16(a) => {
                // Upcast BF16 -> F32 to reuse the SIMD min/max kernel.
                let data: Vec<f32> = a.iter().map(|x| x.to_f32()).collect();
                let (min_val, max_val) = simd_min_max_f32(&data);
                Ok((min_val, max_val))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Min/max not supported for this tensor type",
                "min_max",
            )),
        }
    }

    /// Sum across specified axes with robust error handling.
    ///
    /// Computes the sum along the specified axes. The axes are processed
    /// in reverse order to maintain proper axis indexing during reduction.
    ///
    /// # Arguments
    ///
    /// * `axes` - The axes along which to compute the sum
    ///
    /// # Returns
    ///
    /// A tensor with sums computed along the specified axes.
    pub fn sum_axes(&self, axes: &[usize]) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let mut result = a.clone();
                for &axis in axes.iter().rev() {
                    // Reverse to maintain axis indices
                    if axis >= result.ndim() {
                        return Err(TrustformersError::tensor_op_error(
                            &format!(
                                "Axis {} is out of bounds for tensor with {} dimensions",
                                axis,
                                result.ndim()
                            ),
                            "sum_axes",
                        ));
                    }
                    result = result.sum_axis(Axis(axis));
                }
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let mut result = a.clone();
                for &axis in axes.iter().rev() {
                    if axis >= result.ndim() {
                        return Err(TrustformersError::tensor_op_error(
                            &format!(
                                "Axis {} is out of bounds for tensor with {} dimensions",
                                axis,
                                result.ndim()
                            ),
                            "sum_axes",
                        ));
                    }
                    result = result.sum_axis(Axis(axis));
                }
                Ok(Tensor::F64(result))
            },
            Tensor::F16(_) | Tensor::BF16(_) => run_half_in_f32(self, |t| t.sum_axes(axes)),
            _ => Err(TrustformersError::tensor_op_error(
                "Sum along axes not supported for this tensor type",
                "sum_axes",
            )),
        }
    }

    /// Sum all elements or along specified axes.
    ///
    /// # Arguments
    ///
    /// * `axes` - Optional axes to sum along. If None, sum all elements.
    /// * `keepdims` - Whether to keep dimensions (currently ignored for compatibility).
    ///
    /// # Returns
    ///
    /// A tensor with the sum result.
    pub fn sum(&self, axes: Option<Vec<usize>>, _keepdims: bool) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                if let Some(axes) = axes {
                    if axes.is_empty() {
                        // Sum all elements
                        let sum_val = a.sum();
                        Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), sum_val)))
                    } else {
                        // Sum along specified axes
                        let mut result = a.clone();
                        for &axis in axes.iter().rev() {
                            result = result.sum_axis(Axis(axis));
                        }
                        Ok(Tensor::F32(result))
                    }
                } else {
                    // Sum all elements
                    let sum_val = a.sum();
                    Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), sum_val)))
                }
            },
            Tensor::F64(a) => {
                if let Some(axes) = axes {
                    if axes.is_empty() {
                        // Sum all elements
                        let sum_val = a.sum();
                        Ok(Tensor::F64(ArrayD::from_elem(IxDyn(&[]), sum_val)))
                    } else {
                        // Sum along specified axes
                        let mut result = a.clone();
                        for &axis in axes.iter().rev() {
                            result = result.sum_axis(Axis(axis));
                        }
                        Ok(Tensor::F64(result))
                    }
                } else {
                    // Sum all elements
                    let sum_val = a.sum();
                    Ok(Tensor::F64(ArrayD::from_elem(IxDyn(&[]), sum_val)))
                }
            },
            Tensor::F16(_) | Tensor::BF16(_) => {
                run_half_in_f32(self, |t| t.sum(axes.clone(), _keepdims))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Sum not supported for this tensor type",
                "sum",
            )),
        }
    }

    /// Mean along specified axes with robust error handling.
    ///
    /// # Arguments
    ///
    /// * `axes` - The axes along which to compute the mean
    ///
    /// # Returns
    ///
    /// A tensor with means computed along the specified axes.
    pub fn mean_axes(&self, axes: &[usize]) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let mut result = a.clone();
                for &axis in axes.iter().rev() {
                    // Reverse to maintain axis indices
                    if axis >= result.ndim() {
                        return Err(TrustformersError::tensor_op_error(
                            &format!(
                                "Axis {} is out of bounds for tensor with {} dimensions",
                                axis,
                                result.ndim()
                            ),
                            "mean_axes",
                        ));
                    }
                    result = result.mean_axis(Axis(axis)).ok_or_else(|| {
                        crate::errors::compute_error(
                            "mean_axes",
                            "axis must be valid for mean operation",
                        )
                    })?;
                }
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let mut result = a.clone();
                for &axis in axes.iter().rev() {
                    if axis >= result.ndim() {
                        return Err(TrustformersError::tensor_op_error(
                            &format!(
                                "Axis {} is out of bounds for tensor with {} dimensions",
                                axis,
                                result.ndim()
                            ),
                            "mean_axes",
                        ));
                    }
                    result = result.mean_axis(Axis(axis)).ok_or_else(|| {
                        crate::errors::compute_error(
                            "mean_axes",
                            "axis must be valid for mean operation",
                        )
                    })?;
                }
                Ok(Tensor::F64(result))
            },
            Tensor::F16(_) | Tensor::BF16(_) => run_half_in_f32(self, |t| t.mean_axes(axes)),
            _ => Err(TrustformersError::tensor_op_error(
                "Mean along axes not supported for this tensor type",
                "mean_axes",
            )),
        }
    }

    /// Sum along a single axis (convenience method).
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to sum along
    ///
    /// # Returns
    ///
    /// A tensor with the sum along the specified axis.
    pub fn sum_axis(&self, axis: usize) -> Result<Tensor> {
        self.sum_axes(&[axis])
    }

    /// Python-style sum along a dimension with negative axis support.
    ///
    /// This is a convenience method that supports negative axis indexing
    /// (e.g., -1 for last axis, -2 for second-to-last, etc.)
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to sum along (supports negative indexing)
    /// * `keepdims` - Whether to keep dimensions (currently ignored for compatibility)
    ///
    /// # Returns
    ///
    /// A tensor with the sum along the specified dimension.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let tensor = Tensor::randn(&[2, 3, 4])?;
    /// let sum_last = tensor.sum_dim(-1, false)?;  // Sum along last axis
    /// let sum_first = tensor.sum_dim(0, false)?;  // Sum along first axis
    /// ```
    pub fn sum_dim(&self, dim: i64, _keepdims: bool) -> Result<Tensor> {
        let ndim = self.shape().len();
        let normalized_axis = if dim < 0 {
            let abs_dim = (-dim) as usize;
            if abs_dim > ndim {
                return Err(TrustformersError::tensor_op_error(
                    &format!(
                        "Dimension {} is out of bounds for tensor with {} dimensions",
                        dim, ndim
                    ),
                    "sum_dim",
                ));
            }
            ndim - abs_dim
        } else {
            let dim = dim as usize;
            if dim >= ndim {
                return Err(TrustformersError::tensor_op_error(
                    &format!(
                        "Dimension {} is out of bounds for tensor with {} dimensions",
                        dim, ndim
                    ),
                    "sum_dim",
                ));
            }
            dim
        };

        self.sum_axis(normalized_axis)
    }

    /// Mean along a single axis (convenience method).
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to compute mean along
    ///
    /// # Returns
    ///
    /// A tensor with the mean along the specified axis.
    pub fn mean_axis(&self, axis: usize) -> Result<Tensor> {
        self.mean_axes(&[axis])
    }

    /// Variance computation along specified axes.
    ///
    /// Computes the sample variance using the formula: Var(X) = E[(X - μ)²]
    /// where μ is the mean. Supports computation along specific axes or
    /// across the entire tensor.
    ///
    /// # Arguments
    ///
    /// * `axes` - Optional axes along which to compute variance. If None, compute across all elements.
    /// * `keepdims` - Whether to keep dimensions (currently ignored for compatibility).
    ///
    /// # Returns
    ///
    /// A tensor containing the variance values.
    pub fn variance(&self, axes: Option<&[usize]>, _keepdims: bool) -> Result<Tensor> {
        match self {
            Tensor::F32(_) => {
                let mean_tensor = match axes {
                    Some(ax) => self.mean_axes(ax)?,
                    None => self.mean()?,
                };
                let diff = self.sub(&mean_tensor)?;
                let squared_diff = diff.pow(2.0)?;
                match axes {
                    Some(ax) => squared_diff.mean_axes(ax),
                    None => squared_diff.mean(),
                }
            },
            Tensor::F64(_) => {
                let mean_tensor = match axes {
                    Some(ax) => self.mean_axes(ax)?,
                    None => self.mean()?,
                };
                let diff = self.sub(&mean_tensor)?;
                let squared_diff = diff.pow(2.0)?;
                match axes {
                    Some(ax) => squared_diff.mean_axes(ax),
                    None => squared_diff.mean(),
                }
            },
            Tensor::F16(_) | Tensor::BF16(_) => {
                run_half_in_f32(self, |t| t.variance(axes, _keepdims))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Variance only supported for F32 and F64 tensors",
                "variance",
            )),
        }
    }

    /// Standard deviation computation along specified axes.
    ///
    /// Computes the standard deviation as the square root of variance.
    /// This provides a measure of spread in the same units as the original data.
    ///
    /// # Arguments
    ///
    /// * `axes` - Optional axes along which to compute standard deviation.
    /// * `keepdims` - Whether to keep dimensions (currently ignored for compatibility).
    ///
    /// # Returns
    ///
    /// A tensor containing the standard deviation values.
    pub fn std_dev(&self, axes: Option<&[usize]>, keepdims: bool) -> Result<Tensor> {
        let var = self.variance(axes, keepdims)?;
        var.sqrt()
    }

    /// Find maximum value across specified axes.
    pub fn max_axes(&self, axes: &[usize]) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let mut result = a.clone();
                for &axis in axes.iter().rev() {
                    // reverse to maintain axis indices
                    // Apply max reduction along the specified axis
                    let reduced =
                        result.fold_axis(Axis(axis), f32::NEG_INFINITY, |acc, &x| acc.max(x));
                    result = reduced;
                }
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let mut result = a.clone();
                for &axis in axes.iter().rev() {
                    // reverse to maintain axis indices
                    // Apply max reduction along the specified axis
                    let reduced =
                        result.fold_axis(Axis(axis), f64::NEG_INFINITY, |acc, &x| acc.max(x));
                    result = reduced;
                }
                Ok(Tensor::F64(result))
            },
            Tensor::F16(_) | Tensor::BF16(_) => run_half_in_f32(self, |t| t.max_axes(axes)),
            _ => Err(TrustformersError::tensor_op_error(
                "Max axes not supported for this tensor type",
                "max_axes",
            )),
        }
    }

    /// Find minimum value across specified axes.
    pub fn min_axes(&self, axes: &[usize]) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let mut result = a.clone();
                for &axis in axes.iter().rev() {
                    // reverse to maintain axis indices
                    // Apply min reduction along the specified axis
                    let reduced = result.fold_axis(Axis(axis), f32::INFINITY, |acc, &x| acc.min(x));
                    result = reduced;
                }
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let mut result = a.clone();
                for &axis in axes.iter().rev() {
                    // reverse to maintain axis indices
                    // Apply min reduction along the specified axis
                    let reduced = result.fold_axis(Axis(axis), f64::INFINITY, |acc, &x| acc.min(x));
                    result = reduced;
                }
                Ok(Tensor::F64(result))
            },
            Tensor::F16(_) | Tensor::BF16(_) => run_half_in_f32(self, |t| t.min_axes(axes)),
            _ => Err(TrustformersError::tensor_op_error(
                "Min axes not supported for this tensor type",
                "min_axes",
            )),
        }
    }

    /// Find maximum value in tensor (scalar reduction).
    pub fn max_scalar(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let max_val = a.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                Ok(Tensor::F32(arr0(max_val).into_dyn()))
            },
            Tensor::F64(a) => {
                let max_val = a.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
                Ok(Tensor::F64(arr0(max_val).into_dyn()))
            },
            Tensor::I64(a) => {
                let max_val = a.iter().fold(i64::MIN, |acc, &x| acc.max(x));
                Ok(Tensor::I64(arr0(max_val).into_dyn()))
            },
            Tensor::F16(_) | Tensor::BF16(_) => run_half_in_f32(self, |t| t.max_scalar()),
            _ => Err(TrustformersError::tensor_op_error(
                "max_scalar not implemented for this tensor type",
                "max_scalar",
            )),
        }
    }

    /// Find minimum value in tensor (scalar reduction).
    pub fn min_scalar(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let min_val = a.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
                Ok(Tensor::F32(arr0(min_val).into_dyn()))
            },
            Tensor::F64(a) => {
                let min_val = a.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
                Ok(Tensor::F64(arr0(min_val).into_dyn()))
            },
            Tensor::I64(a) => {
                let min_val = a.iter().fold(i64::MAX, |acc, &x| acc.min(x));
                Ok(Tensor::I64(arr0(min_val).into_dyn()))
            },
            Tensor::F16(_) | Tensor::BF16(_) => run_half_in_f32(self, |t| t.min_scalar()),
            _ => Err(TrustformersError::tensor_op_error(
                "min_scalar not implemented for this tensor type",
                "min_scalar",
            )),
        }
    }

    /// Sample from multinomial distribution.
    ///
    /// Samples from a multinomial distribution defined by the probabilities in the input tensor.
    /// This is useful for sampling tokens during text generation.
    ///
    /// # Arguments
    ///
    /// * `num_samples` - Number of samples to draw
    /// * `replacement` - Whether to sample with replacement (must be true currently)
    ///
    /// # Returns
    ///
    /// A tensor containing sampled indices.
    ///
    /// # Errors
    ///
    /// - `TensorOpError`: If the tensor is not a probability distribution (doesn't sum to ~1.0)
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Create a probability distribution
    /// let probs = Tensor::from_vec(vec![0.1, 0.2, 0.3, 0.4], &[4])?;
    /// let probs = probs.softmax(0)?; // Ensure it sums to 1.0
    ///
    /// // Sample from the distribution
    /// let samples = probs.multinomial(1, true)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn multinomial(&self, num_samples: usize, replacement: bool) -> Result<Tensor> {
        use scirs2_core::random::*;

        if !replacement {
            return Err(TrustformersError::tensor_op_error(
                "Sampling without replacement is not yet supported",
                "multinomial",
            ));
        }

        match self {
            Tensor::F32(probs) => {
                let mut rng = thread_rng();

                // Get the shape and flatten if needed
                let shape = probs.shape();
                let last_dim = shape[shape.len() - 1];

                // Calculate total number of distributions
                let num_dists: usize = shape[..shape.len() - 1].iter().product();

                // Prepare output shape
                let mut output_shape = shape[..shape.len() - 1].to_vec();
                output_shape.push(num_samples);

                let mut samples = Vec::with_capacity(num_dists * num_samples);

                // Sample from each distribution
                for dist_idx in 0..num_dists {
                    // Extract probabilities for this distribution
                    let offset = dist_idx * last_dim;
                    let prob_slice = &probs.as_slice().ok_or_else(|| {
                        crate::errors::compute_error(
                            "multinomial",
                            "array must have contiguous layout",
                        )
                    })?[offset..offset + last_dim];

                    // Compute cumulative distribution
                    let mut cumsum = Vec::with_capacity(last_dim);
                    let mut sum = 0.0f32;
                    for &p in prob_slice {
                        sum += p;
                        cumsum.push(sum);
                    }

                    // Sample using inverse transform sampling
                    for _ in 0..num_samples {
                        let u: f32 = rng.random();
                        let u_scaled = u * sum;

                        // Find the first index where cumsum >= u_scaled
                        let idx =
                            cumsum.iter().position(|&c| c >= u_scaled).unwrap_or(last_dim - 1);

                        samples.push(idx as i64);
                    }
                }

                Ok(Tensor::I64(ArrayD::from_shape_vec(
                    IxDyn(&output_shape),
                    samples,
                )?))
            },
            Tensor::F64(probs) => {
                let mut rng = thread_rng();

                let shape = probs.shape();
                let last_dim = shape[shape.len() - 1];
                let num_dists: usize = shape[..shape.len() - 1].iter().product();

                let mut output_shape = shape[..shape.len() - 1].to_vec();
                output_shape.push(num_samples);

                let mut samples = Vec::with_capacity(num_dists * num_samples);

                for dist_idx in 0..num_dists {
                    let offset = dist_idx * last_dim;
                    let prob_slice = &probs.as_slice().ok_or_else(|| {
                        crate::errors::compute_error(
                            "multinomial",
                            "array must have contiguous layout",
                        )
                    })?[offset..offset + last_dim];

                    let mut cumsum = Vec::with_capacity(last_dim);
                    let mut sum = 0.0f64;
                    for &p in prob_slice {
                        sum += p;
                        cumsum.push(sum);
                    }

                    for _ in 0..num_samples {
                        let u: f64 = rng.random();
                        let u_scaled = u * sum;

                        let idx =
                            cumsum.iter().position(|&c| c >= u_scaled).unwrap_or(last_dim - 1);

                        samples.push(idx as i64);
                    }
                }

                Ok(Tensor::I64(ArrayD::from_shape_vec(
                    IxDyn(&output_shape),
                    samples,
                )?))
            },
            Tensor::F16(_) | Tensor::BF16(_) => {
                // Upcast probabilities to F32; the sampled output is an I64 index
                // tensor whose dtype is independent of the input precision.
                run_half_in_f32_keep(self, |t| t.multinomial(num_samples, replacement))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "multinomial not supported for this tensor type",
                "multinomial",
            )),
        }
    }

    /// Check if all elements are true (for boolean tensors) or non-zero.
    ///
    /// Returns a scalar boolean tensor indicating whether all elements satisfy the condition.
    ///
    /// # Returns
    ///
    /// A scalar F32 tensor with value 1.0 if all elements are non-zero, 0.0 otherwise.
    ///
    /// # Errors
    ///
    /// - `TensorOpError`: If the operation is not supported for the tensor type
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tensor = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3])?;
    /// let result = tensor.all()?; // Should be 1.0 (true)
    ///
    /// let tensor2 = Tensor::from_vec(vec![1.0, 0.0, 1.0], &[3])?;
    /// let result2 = tensor2.all()?; // Should be 0.0 (false)
    /// # Ok(())
    /// # }
    /// ```
    pub fn all(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(arr) => {
                let all_true = arr.iter().all(|&x| x != 0.0);
                let result = if all_true { 1.0f32 } else { 0.0f32 };
                Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), result)))
            },
            Tensor::F64(arr) => {
                let all_true = arr.iter().all(|&x| x != 0.0);
                let result = if all_true { 1.0f32 } else { 0.0f32 };
                Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), result)))
            },
            Tensor::I64(arr) => {
                let all_true = arr.iter().all(|&x| x != 0);
                let result = if all_true { 1.0f32 } else { 0.0f32 };
                Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), result)))
            },
            Tensor::F16(_) | Tensor::BF16(_) => {
                // `all` always returns an F32 boolean scalar, independent of input dtype.
                run_half_in_f32_keep(self, |t| t.all())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "all not supported for this tensor type",
                "all",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::errors::Result;
    use crate::tensor::Tensor;

    #[test]
    fn test_std_f32() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0], &[5])?;
        let s = t.std()?;
        let data = s.data()?;
        // std of [1,2,3,4,5] = sqrt(2.0) ~= 1.4142
        assert!((data[0] - std::f64::consts::SQRT_2 as f32).abs() < 0.01);
        Ok(())
    }

    #[test]
    fn test_std_constant() -> Result<()> {
        let t = Tensor::full(5.0, vec![10])?;
        let s = t.std()?;
        let data = s.data()?;
        assert!(data[0].abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_max_value() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 5.0, 3.0, 2.0], &[4])?;
        let m = t.max_value()?;
        let data = m.data()?;
        assert!((data[0] - 5.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_max_elementwise() -> Result<()> {
        let a = Tensor::from_data(vec![1.0, 5.0, 3.0], &[3])?;
        let b = Tensor::from_data(vec![2.0, 4.0, 6.0], &[3])?;
        let result = a.max(&b)?;
        let data = result.data()?;
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 5.0).abs() < 1e-6);
        assert!((data[2] - 6.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_mean_f32() -> Result<()> {
        let t = Tensor::from_data(vec![2.0, 4.0, 6.0, 8.0], &[4])?;
        let m = t.mean()?;
        let data = m.data()?;
        assert!((data[0] - 5.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_mean_single_element() -> Result<()> {
        let t = Tensor::from_data(vec![42.0], &[1])?;
        let m = t.mean()?;
        let data = m.data()?;
        assert!((data[0] - 42.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_min_max() -> Result<()> {
        let t = Tensor::from_data(vec![-3.0, 1.0, 7.0, -1.0, 5.0], &[5])?;
        let (min_val, max_val) = t.min_max()?;
        assert!((min_val - (-3.0)).abs() < 1e-6);
        assert!((max_val - 7.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_sum_all() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let s = t.sum(None, false)?;
        let data = s.data()?;
        assert!((data[0] - 10.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_sum_axis_0() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let s = t.sum_axis(0)?;
        assert_eq!(s.shape(), vec![2]);
        let data = s.data()?;
        assert!((data[0] - 4.0).abs() < 1e-5);
        assert!((data[1] - 6.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_sum_axis_1() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let s = t.sum_axis(1)?;
        assert_eq!(s.shape(), vec![2]);
        let data = s.data()?;
        assert!((data[0] - 3.0).abs() < 1e-5);
        assert!((data[1] - 7.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_mean_axis() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 3.0, 5.0, 7.0], &[2, 2])?;
        let m = t.mean_axis(0)?;
        assert_eq!(m.shape(), vec![2]);
        let data = m.data()?;
        assert!((data[0] - 3.0).abs() < 1e-5);
        assert!((data[1] - 5.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_variance() -> Result<()> {
        let t = Tensor::from_data(vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], &[8])?;
        let v = t.variance(None, false)?;
        let data = v.data()?;
        // variance = mean of (x - mean)^2
        assert!(data[0] > 0.0);
        Ok(())
    }

    #[test]
    fn test_argmax_1d() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 5.0, 3.0, 2.0], &[4])?;
        let idx = t.argmax(0)?;
        // argmax returns indices - check shape is correct
        assert_eq!(idx.shape(), Vec::<usize>::new());
        Ok(())
    }

    #[test]
    fn test_argmax_2d_axis0() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 5.0, 3.0, 2.0], &[2, 2])?;
        let idx = t.argmax(0)?;
        assert_eq!(idx.shape(), vec![2]);
        Ok(())
    }

    #[test]
    fn test_sum_axes_multiple() -> Result<()> {
        let t = Tensor::ones(&[2, 3, 4])?;
        let s = t.sum_axes(&[0, 2])?;
        // After summing axes 0 and 2 from [2,3,4]: result should have shape [3]
        assert_eq!(s.shape(), vec![3]);
        let data = s.data()?;
        for val in &data {
            assert!((val - 8.0).abs() < 1e-5); // 2*4 = 8
        }
        Ok(())
    }

    #[test]
    fn test_max_scalar() -> Result<()> {
        let t = Tensor::from_data(vec![-1.0, 3.0, 2.0, 7.0, -5.0], &[5])?;
        let m = t.max_scalar()?;
        let data = m.data()?;
        assert!((data[0] - 7.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_min_scalar() -> Result<()> {
        let t = Tensor::from_data(vec![-1.0, 3.0, 2.0, 7.0, -5.0], &[5])?;
        let m = t.min_scalar()?;
        let data = m.data()?;
        assert!((data[0] - (-5.0)).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_sum_dim_positive() -> Result<()> {
        let t = Tensor::ones(&[2, 3])?;
        let s = t.sum_dim(1, false)?;
        assert_eq!(s.shape(), vec![2]);
        let data = s.data()?;
        for val in &data {
            assert!((val - 3.0).abs() < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn test_sum_dim_negative() -> Result<()> {
        let t = Tensor::ones(&[2, 3])?;
        let s = t.sum_dim(-1, false)?;
        assert_eq!(s.shape(), vec![2]);
        Ok(())
    }

    #[test]
    fn test_mean_axes() -> Result<()> {
        let t = Tensor::ones(&[2, 3, 4])?;
        let m = t.mean_axes(&[1])?;
        assert_eq!(m.shape(), vec![2, 4]);
        Ok(())
    }

    #[test]
    fn test_max_axes() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 5.0, 3.0, 2.0, 4.0, 6.0], &[2, 3])?;
        let m = t.max_axes(&[1])?;
        assert_eq!(m.shape(), vec![2]);
        let data = m.data()?;
        assert!((data[0] - 5.0).abs() < 1e-6);
        assert!((data[1] - 6.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_min_axes() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 5.0, 3.0, 2.0, 4.0, 6.0], &[2, 3])?;
        let m = t.min_axes(&[1])?;
        assert_eq!(m.shape(), vec![2]);
        let data = m.data()?;
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_std_dev_alias() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3])?;
        let s = t.std_dev(None, false)?;
        let data = s.data()?;
        assert!(data[0] > 0.0);
        Ok(())
    }

    #[test]
    fn test_all_ones() -> Result<()> {
        let t = Tensor::ones(&[3])?;
        let result = t.all()?;
        let data = result.data()?;
        assert!((data[0] - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_all_with_zero() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 0.0, 1.0], &[3])?;
        let result = t.all()?;
        let data = result.data()?;
        assert!(data[0].abs() < 1e-6);
        Ok(())
    }

    // ---- Half-precision (F16 / BF16) upcast-path tests ----

    use crate::tensor::DType;
    use scirs2_core::ndarray::{ArrayD as TestArrayD, IxDyn as TestIxDyn};

    /// Build an F16 tensor from f32 values.
    fn make_f16(data: &[f32], shape: &[usize]) -> Result<Tensor> {
        let arr = TestArrayD::from_shape_vec(
            TestIxDyn(shape),
            data.iter().map(|&x| half::f16::from_f32(x)).collect(),
        )
        .map_err(|e| crate::errors::TrustformersError::shape_error(e.to_string()))?;
        Ok(Tensor::F16(arr))
    }

    /// Build a BF16 tensor from f32 values.
    fn make_bf16(data: &[f32], shape: &[usize]) -> Result<Tensor> {
        let arr = TestArrayD::from_shape_vec(
            TestIxDyn(shape),
            data.iter().map(|&x| half::bf16::from_f32(x)).collect(),
        )
        .map_err(|e| crate::errors::TrustformersError::shape_error(e.to_string()))?;
        Ok(Tensor::BF16(arr))
    }

    /// Read a half-precision tensor's values as f32 for assertions.
    fn half_to_vec_f32(t: &Tensor) -> Vec<f32> {
        match t {
            Tensor::F16(a) => a.iter().map(|x| x.to_f32()).collect(),
            Tensor::BF16(a) => a.iter().map(|x| x.to_f32()).collect(),
            _ => panic!("expected a half-precision tensor"),
        }
    }

    #[test]
    fn test_reductions_f16_bf16() -> Result<()> {
        for build in [
            make_f16 as fn(&[f32], &[usize]) -> Result<Tensor>,
            make_bf16 as fn(&[f32], &[usize]) -> Result<Tensor>,
        ] {
            let t = build(&[1.0, 2.0, 3.0, 4.0], &[4])?;
            let dt = t.dtype();

            // Scalar reductions: dtype preserved, value correct.
            let mean = t.mean()?;
            assert_eq!(mean.dtype(), dt);
            assert!((half_to_vec_f32(&mean)[0] - 2.5).abs() < 0.05);

            let sum = t.sum(Some(vec![]), false)?;
            assert_eq!(sum.dtype(), dt);
            assert!((half_to_vec_f32(&sum)[0] - 10.0).abs() < 0.1);

            let std = t.std()?;
            assert_eq!(std.dtype(), dt);
            assert!(half_to_vec_f32(&std)[0].is_finite());

            let max_v = t.max_value()?;
            assert_eq!(max_v.dtype(), dt);
            assert!((half_to_vec_f32(&max_v)[0] - 4.0).abs() < 0.05);

            let max_s = t.max_scalar()?;
            assert_eq!(max_s.dtype(), dt);
            assert!((half_to_vec_f32(&max_s)[0] - 4.0).abs() < 0.05);

            let min_s = t.min_scalar()?;
            assert_eq!(min_s.dtype(), dt);
            assert!((half_to_vec_f32(&min_s)[0] - 1.0).abs() < 0.05);

            // min_max returns an (f32, f32) tuple regardless of dtype.
            let (mn, mx) = t.min_max()?;
            assert!((mn - 1.0).abs() < 0.05 && (mx - 4.0).abs() < 0.05);
        }
        Ok(())
    }

    #[test]
    fn test_axis_reductions_f16_bf16() -> Result<()> {
        for build in [
            make_f16 as fn(&[f32], &[usize]) -> Result<Tensor>,
            make_bf16 as fn(&[f32], &[usize]) -> Result<Tensor>,
        ] {
            let t = build(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])?;
            let dt = t.dtype();

            let sum_axes = t.sum_axes(&[1])?;
            assert_eq!(sum_axes.dtype(), dt);
            assert_eq!(sum_axes.shape(), vec![2]);
            let s = half_to_vec_f32(&sum_axes);
            assert!((s[0] - 6.0).abs() < 0.1 && (s[1] - 15.0).abs() < 0.1);

            let mean_axes = t.mean_axes(&[1])?;
            assert_eq!(mean_axes.dtype(), dt);
            assert_eq!(mean_axes.shape(), vec![2]);

            let max_axes = t.max_axes(&[1])?;
            assert_eq!(max_axes.dtype(), dt);
            assert_eq!(max_axes.shape(), vec![2]);
            let mx = half_to_vec_f32(&max_axes);
            assert!((mx[0] - 3.0).abs() < 0.05 && (mx[1] - 6.0).abs() < 0.05);

            let min_axes = t.min_axes(&[1])?;
            assert_eq!(min_axes.dtype(), dt);
            assert_eq!(min_axes.shape(), vec![2]);

            // Scalar variance over all elements (axis-wise variance is unsupported
            // on the F32 path too due to broadcast rules).
            let var = t.variance(None, false)?;
            assert_eq!(var.dtype(), dt);
            assert!(half_to_vec_f32(&var)[0].is_finite());

            let argmax = t.argmax(1)?;
            assert_eq!(argmax.dtype(), dt);
            assert_eq!(argmax.shape(), vec![2]);
            let am = half_to_vec_f32(&argmax);
            // Largest element of each row is at index 2.
            assert!((am[0] - 2.0).abs() < 0.05 && (am[1] - 2.0).abs() < 0.05);
        }
        Ok(())
    }

    #[test]
    fn test_all_f16_bf16_returns_f32() -> Result<()> {
        // `all` returns an F32 boolean scalar even for half-precision input.
        for build in [
            make_f16 as fn(&[f32], &[usize]) -> Result<Tensor>,
            make_bf16 as fn(&[f32], &[usize]) -> Result<Tensor>,
        ] {
            let t = build(&[1.0, 1.0, 1.0], &[3])?;
            let r = t.all()?;
            assert_eq!(r.dtype(), DType::F32);
            assert!((r.data()?[0] - 1.0).abs() < 1e-6);

            let t2 = build(&[1.0, 0.0, 1.0], &[3])?;
            let r2 = t2.all()?;
            assert_eq!(r2.dtype(), DType::F32);
            assert!(r2.data()?[0].abs() < 1e-6);
        }
        Ok(())
    }
}
