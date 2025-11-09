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
use ndarray::{ArrayD, Axis, IxDyn};

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
                let mean = a.mean().unwrap();
                let variance = a.mapv(|x| (x - mean).powi(2)).mean().unwrap();
                let std = variance.sqrt();
                Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), std)))
            },
            Tensor::F64(a) => {
                let mean = a.mean().unwrap();
                let variance = a.mapv(|x| (x - mean).powi(2)).mean().unwrap();
                let std = variance.sqrt();
                Ok(Tensor::F64(ArrayD::from_elem(IxDyn(&[]), std)))
            },
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
                    let scalar_val = a.iter().next().unwrap();
                    let result = b.mapv(|x| x.max(*scalar_val));
                    Ok(Tensor::F32(result))
                } else if b.ndim() == 0 && a.ndim() > 0 {
                    // b is scalar, broadcast to a's shape
                    let scalar_val = b.iter().next().unwrap();
                    let result = a.mapv(|x| x.max(*scalar_val));
                    Ok(Tensor::F32(result))
                } else if a.ndim() == 0 && b.ndim() == 0 {
                    // Both scalars
                    let a_val = a.iter().next().unwrap();
                    let b_val = b.iter().next().unwrap();
                    let max_val = a_val.max(*b_val);
                    Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), max_val)))
                } else {
                    // Regular element-wise operation
                    let result = ndarray::Zip::from(a).and(b).map_collect(|&x, &y| x.max(y));
                    Ok(Tensor::F32(result))
                }
            },
            (Tensor::F64(a), Tensor::F64(b)) => {
                // Handle scalar case (0-dimensional tensor)
                if a.ndim() == 0 && b.ndim() > 0 {
                    // a is scalar, broadcast to b's shape
                    let scalar_val = a.iter().next().unwrap();
                    let result = b.mapv(|x| x.max(*scalar_val));
                    Ok(Tensor::F64(result))
                } else if b.ndim() == 0 && a.ndim() > 0 {
                    // b is scalar, broadcast to a's shape
                    let scalar_val = b.iter().next().unwrap();
                    let result = a.mapv(|x| x.max(*scalar_val));
                    Ok(Tensor::F64(result))
                } else if a.ndim() == 0 && b.ndim() == 0 {
                    // Both scalars
                    let a_val = a.iter().next().unwrap();
                    let b_val = b.iter().next().unwrap();
                    let max_val = a_val.max(*b_val);
                    Ok(Tensor::F64(ArrayD::from_elem(IxDyn(&[]), max_val)))
                } else {
                    // Regular element-wise operation
                    let result = ndarray::Zip::from(a).and(b).map_collect(|&x, &y| x.max(y));
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
                let mean = a.mean().unwrap();
                Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), mean)))
            },
            Tensor::F64(a) => {
                let mean = a.mean().unwrap();
                Ok(Tensor::F64(ArrayD::from_elem(IxDyn(&[]), mean)))
            },
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
                let data = a.as_slice().unwrap();
                let (min_val, max_val) = simd_min_max_f32(data);
                Ok((min_val, max_val))
            },
            Tensor::F64(a) => {
                let data = a.as_slice().unwrap();
                let (min_val, max_val) = simd_min_max_f64(data);
                Ok((min_val as f32, max_val as f32))
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
                    result = result.sum_axis(ndarray::Axis(axis));
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
                    result = result.sum_axis(ndarray::Axis(axis));
                }
                Ok(Tensor::F64(result))
            },
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
                            result = result.sum_axis(ndarray::Axis(axis));
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
                            result = result.sum_axis(ndarray::Axis(axis));
                        }
                        Ok(Tensor::F64(result))
                    }
                } else {
                    // Sum all elements
                    let sum_val = a.sum();
                    Ok(Tensor::F64(ArrayD::from_elem(IxDyn(&[]), sum_val)))
                }
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
                    result = result.mean_axis(ndarray::Axis(axis)).unwrap();
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
                    result = result.mean_axis(ndarray::Axis(axis)).unwrap();
                }
                Ok(Tensor::F64(result))
            },
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
                Ok(Tensor::F32(ndarray::arr0(max_val).into_dyn()))
            },
            Tensor::F64(a) => {
                let max_val = a.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
                Ok(Tensor::F64(ndarray::arr0(max_val).into_dyn()))
            },
            Tensor::I64(a) => {
                let max_val = a.iter().fold(i64::MIN, |acc, &x| acc.max(x));
                Ok(Tensor::I64(ndarray::arr0(max_val).into_dyn()))
            },
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
                Ok(Tensor::F32(ndarray::arr0(min_val).into_dyn()))
            },
            Tensor::F64(a) => {
                let min_val = a.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
                Ok(Tensor::F64(ndarray::arr0(min_val).into_dyn()))
            },
            Tensor::I64(a) => {
                let min_val = a.iter().fold(i64::MAX, |acc, &x| acc.min(x));
                Ok(Tensor::I64(ndarray::arr0(min_val).into_dyn()))
            },
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
                    let prob_slice = &probs.as_slice().unwrap()[offset..offset + last_dim];

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
                    let prob_slice = &probs.as_slice().unwrap()[offset..offset + last_dim];

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
            _ => Err(TrustformersError::tensor_op_error(
                "all not supported for this tensor type",
                "all",
            )),
        }
    }
}
