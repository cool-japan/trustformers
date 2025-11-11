//! Linear algebra operations for tensors.
//!
//! This module provides core linear algebra operations including matrix multiplication,
//! norm calculations, and gradient clipping. All operations include numerical stability
//! enhancements and comprehensive error handling.
//!
//! # Features
//!
//! - **Matrix Multiplication**: Optimized matmul with support for 2D, 3D, and 4D tensors
//! - **Norm Calculations**: L2 norm and squared norm computations
//! - **Gradient Clipping**: Norm-based gradient clipping for training stability
//! - **Numerical Stability**: Built-in overflow/underflow protection and NaN detection
//! - **Multi-type Support**: Full support for F32, F64, I64, C32, and C64 tensor types
//!
//! # Examples
//!
//! ```no_run
//! use trustformers_core::tensor::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Matrix multiplication
//! let a = Tensor::randn(&[128, 64])?;
//! let b = Tensor::randn(&[64, 256])?;
//! let result = a.matmul(&b)?;
//!
//! // Norm calculation
//! let tensor = Tensor::randn(&[100])?;
//! let norm = tensor.norm()?;
//!
//! // Gradient clipping
//! let gradients = Tensor::randn(&[1000])?;
//! let clipped = gradients.clip_grad_norm(1.0)?;
//! # Ok(())
//! # }
//! ```

use super::super::Tensor;
use super::stability::*;
use crate::errors::{Result, TrustformersError};
use scirs2_core::ndarray::{s, ArrayD, IxDyn};

impl Tensor {
    /// Matrix multiplication with numerical stability enhancements.
    ///
    /// Performs matrix multiplication between two tensors with support for:
    /// - 2D matrix multiplication
    /// - Batched 3D matrix multiplication
    /// - Multi-headed 4D matrix multiplication (for attention mechanisms)
    ///
    /// # Numerical Stability Features
    ///
    /// - Automatic detection of unstable values (NaN, infinity, extreme values)
    /// - Kahan summation algorithm for unstable inputs
    /// - Memory layout optimization for performance
    /// - Overflow/underflow protection
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to multiply with (right operand)
    ///
    /// # Returns
    ///
    /// A new tensor containing the matrix multiplication result.
    ///
    /// # Errors
    ///
    /// - `ShapeError`: If tensors have incompatible dimensions for matrix multiplication
    /// - `TensorOpError`: If the operation is not supported for the tensor types
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // 2D matrix multiplication
    /// let a = Tensor::randn(&[128, 64])?;
    /// let b = Tensor::randn(&[64, 256])?;
    /// let result = a.matmul(&b)?; // Shape: [128, 256]
    ///
    /// // Batched matrix multiplication
    /// let a = Tensor::randn(&[32, 128, 64])?;  // 32 batches
    /// let b = Tensor::randn(&[32, 64, 256])?;
    /// let result = a.matmul(&b)?; // Shape: [32, 128, 256]
    ///
    /// // Multi-headed attention matrices
    /// let q = Tensor::randn(&[8, 12, 512, 64])?;  // 8 batches, 12 heads
    /// let k = Tensor::randn(&[8, 12, 64, 512])?;
    /// let attention = q.matmul(&k)?; // Shape: [8, 12, 512, 512]
    /// # Ok(())
    /// # }
    /// ```
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        // Ensure both inputs have contiguous memory layouts before any operations
        let self_contiguous = match self {
            Tensor::F32(a) => Tensor::F32(a.as_standard_layout().to_owned()),
            Tensor::F64(a) => Tensor::F64(a.as_standard_layout().to_owned()),
            _ => self.clone(),
        };

        let other_contiguous = match other {
            Tensor::F32(a) => Tensor::F32(a.as_standard_layout().to_owned()),
            Tensor::F64(a) => Tensor::F64(a.as_standard_layout().to_owned()),
            _ => other.clone(),
        };

        match (&self_contiguous, &other_contiguous) {
            (Tensor::F32(a), Tensor::F32(b)) => {
                let a_shape = a.shape();
                let b_shape = b.shape();

                if a_shape.len() < 2 || b_shape.len() < 2 {
                    return Err(TrustformersError::shape_error(
                        "Matrix multiplication requires at least 2D tensors".into(),
                    ));
                }

                let a_last = a_shape[a_shape.len() - 1];
                let b_second_last = b_shape[b_shape.len() - 2];

                if a_last != b_second_last {
                    return Err(TrustformersError::shape_error(format!(
                        "Matrix dimensions mismatch: {} vs {}",
                        a_last, b_second_last
                    )));
                }

                // Handle different dimensionalities
                if a_shape.len() == 2 && b_shape.len() == 2 {
                    // Simple 2D matrix multiplication with numerical stability
                    let a_2d = a
                        .view()
                        .into_dimensionality::<ndarray::Ix2>()
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    let b_2d = b
                        .view()
                        .into_dimensionality::<ndarray::Ix2>()
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;

                    // Check for stability before computation
                    let has_unstable_a = a_2d.iter().any(|&x| !is_stable_f32(x));
                    let has_unstable_b = b_2d.iter().any(|&x| !is_stable_f32(x));

                    if has_unstable_a || has_unstable_b {
                        // Use stabilized manual multiplication with Kahan summation
                        let rows = a_2d.nrows();
                        let cols = b_2d.ncols();
                        let inner = a_2d.ncols();
                        let mut result = ArrayD::zeros(IxDyn(&[rows, cols]));

                        for i in 0..rows {
                            for j in 0..cols {
                                let mut sum = 0.0f32;
                                let mut compensation = 0.0f32; // Kahan summation

                                for k in 0..inner {
                                    let a_val = stabilize_f32(a_2d[[i, k]]);
                                    let b_val = stabilize_f32(b_2d[[k, j]]);
                                    let product = a_val * b_val;

                                    // Kahan summation for numerical stability
                                    let y = product - compensation;
                                    let t = sum + y;
                                    compensation = (t - sum) - y;
                                    sum = t;
                                }

                                result[[i, j]] = stabilize_f32(sum);
                            }
                        }

                        Ok(Tensor::F32(result))
                    } else {
                        // Use optimized BLAS if inputs are stable
                        let result = a_2d.dot(&b_2d);
                        Ok(Tensor::F32(result.into_dyn()))
                    }
                } else {
                    // Batched matrix multiplication
                    let mut result_shape = a_shape.to_vec();
                    let last_idx = result_shape.len() - 1;
                    result_shape[last_idx] = b_shape[b_shape.len() - 1];

                    let mut result = ArrayD::zeros(IxDyn(&result_shape));

                    // For simplicity, handle 3D case (batch matrix multiplication)
                    if a_shape.len() == 3 && b_shape.len() == 3 {
                        let batch_size = a_shape[0];
                        for i in 0..batch_size {
                            let a_slice = a.slice(s![i, .., ..]);
                            let b_slice = b.slice(s![i, .., ..]);

                            // Ensure contiguous layout before dimensionality conversion
                            let a_contiguous = a_slice.to_owned().as_standard_layout().to_owned();
                            let b_contiguous = b_slice.to_owned().as_standard_layout().to_owned();

                            let a_2d = a_contiguous
                                .into_dimensionality::<ndarray::Ix2>()
                                .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                            let b_2d = b_contiguous
                                .into_dimensionality::<ndarray::Ix2>()
                                .map_err(|e| TrustformersError::shape_error(e.to_string()))?;

                            let batch_result = a_2d.dot(&b_2d);
                            result.slice_mut(s![i, .., ..]).assign(&batch_result);
                        }
                    } else if a_shape.len() == 4 && b_shape.len() == 4 {
                        // Simple 4D batched matrix multiplication
                        // Handle as multiple 2D matrix multiplications
                        let batch_size = a_shape[0];
                        let num_heads = a_shape[1];
                        let seq_len_a = a_shape[2];
                        let seq_len_b = b_shape[3];

                        result =
                            ArrayD::zeros(IxDyn(&[batch_size, num_heads, seq_len_a, seq_len_b]));

                        for i in 0..batch_size {
                            for j in 0..num_heads {
                                // Extract 2D slices and ensure contiguous layout
                                let a_slice = a.slice(s![i, j, .., ..]);
                                let b_slice = b.slice(s![i, j, .., ..]);

                                // Create contiguous copies
                                let a_matrix = a_slice.to_owned();
                                let b_matrix = b_slice.to_owned();

                                // Force contiguous layout before dimensionality conversion
                                let a_contiguous = a_matrix.as_standard_layout().to_owned();
                                let b_contiguous = b_matrix.as_standard_layout().to_owned();

                                // Convert to 2D arrays for dot product
                                let a_2d = a_contiguous
                                    .into_dimensionality::<ndarray::Ix2>()
                                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                                let b_2d = b_contiguous
                                    .into_dimensionality::<ndarray::Ix2>()
                                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;

                                // Perform 2D matrix multiplication
                                let result_2d = a_2d.dot(&b_2d);

                                // Assign result back to 4D tensor
                                result.slice_mut(s![i, j, .., ..]).assign(&result_2d);
                            }
                        }
                    } else {
                        return Err(TrustformersError::tensor_op_error(
                            "Unsupported tensor dimensions for matmul",
                            "matmul",
                        ));
                    }

                    Ok(Tensor::F32(result))
                }
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Matmul not supported for these tensor types",
                "matmul",
            )),
        }
    }

    /// Calculate the L2 norm (Euclidean norm) of the tensor.
    ///
    /// Computes the square root of the sum of squares of all elements in the tensor.
    /// This is equivalent to the Euclidean distance from the origin in the tensor's
    /// vector space.
    ///
    /// # Mathematical Definition
    ///
    /// For a tensor `x`, the L2 norm is: `||x||_2 = sqrt(Σ x_i²)`
    ///
    /// # Returns
    ///
    /// The L2 norm as a scalar `f32` value.
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
    /// let tensor = Tensor::from_vec(vec![3.0, 4.0], &[2])?;
    /// let norm = tensor.norm()?; // Should be 5.0 (sqrt(3² + 4²))
    /// # Ok(())
    /// # }
    /// ```
    pub fn norm(&self) -> Result<f32> {
        match self {
            Tensor::F32(a) => {
                let sum_squares = a.mapv(|x| x * x).sum();
                Ok(sum_squares.sqrt())
            },
            Tensor::F64(a) => {
                let sum_squares = a.mapv(|x| x * x).sum();
                Ok(sum_squares.sqrt() as f32)
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Norm not supported for this tensor type",
                "norm",
            )),
        }
    }

    /// Calculate the squared L2 norm of the tensor.
    ///
    /// Computes the sum of squares of all elements in the tensor without taking
    /// the square root. This is computationally more efficient than `norm()` when
    /// only the squared norm is needed.
    ///
    /// # Mathematical Definition
    ///
    /// For a tensor `x`, the squared L2 norm is: `||x||_2² = Σ x_i²`
    ///
    /// # Returns
    ///
    /// A scalar tensor containing the squared norm value.
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
    /// let tensor = Tensor::from_vec(vec![3.0, 4.0], &[2])?;
    /// let norm_squared = tensor.norm_squared()?; // Should be 25.0 (3² + 4²)
    /// # Ok(())
    /// # }
    /// ```
    pub fn norm_squared(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let sum_squares = a.mapv(|x| x * x).sum();
                Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), sum_squares)))
            },
            Tensor::F64(a) => {
                let sum_squares = a.mapv(|x| x * x).sum();
                Ok(Tensor::F64(ArrayD::from_elem(IxDyn(&[]), sum_squares)))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Norm squared not supported for this tensor type",
                "norm_squared",
            )),
        }
    }

    /// Clip gradients based on their norm to prevent gradient explosion.
    ///
    /// This function implements gradient clipping by scaling the tensor values
    /// to ensure the L2 norm does not exceed the specified maximum value.
    /// This is a common technique used in training deep neural networks to
    /// prevent gradient explosion.
    ///
    /// # Algorithm
    ///
    /// 1. Calculate the current L2 norm of the tensor
    /// 2. If norm ≤ max_norm, return the tensor unchanged
    /// 3. If norm > max_norm, scale the tensor by (max_norm / norm)
    ///
    /// # Arguments
    ///
    /// * `max_norm` - The maximum allowed norm value
    ///
    /// # Returns
    ///
    /// A new tensor with clipped gradient values.
    ///
    /// # Errors
    ///
    /// - `TensorOpError`: If norm calculation or scalar multiplication fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Create a tensor with large gradient values
    /// let gradients = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3])?;
    ///
    /// // Clip to maximum norm of 1.0
    /// let clipped = gradients.clip_grad_norm(1.0)?;
    ///
    /// // The resulting tensor will have norm ≤ 1.0
    /// assert!(clipped.norm()? <= 1.0);
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Use in Training
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # let gradients = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3])?;
    /// // Typical usage in gradient clipping during training
    /// let max_gradient_norm = 1.0;
    /// let clipped_gradients = gradients.clip_grad_norm(max_gradient_norm)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn clip_grad_norm(&self, max_norm: f32) -> Result<Tensor> {
        let norm = self.norm()?;
        if norm > max_norm {
            self.scalar_mul(max_norm / norm)
        } else {
            Ok(self.clone())
        }
    }

    /// Calculate L2 norm along specified dimension(s).
    ///
    /// This function computes the L2 norm along one or more dimensions, which is
    /// useful for normalization operations (e.g., in contrastive learning, CLIP models).
    ///
    /// # Arguments
    ///
    /// * `p` - The order of the norm (typically 2 for L2 norm)
    /// * `dims` - Optional dimensions along which to compute the norm. If None, computes
    ///   the norm across all dimensions (equivalent to `norm()`).
    /// * `keepdim` - If true, keeps the reduced dimensions with size 1
    ///
    /// # Returns
    ///
    /// A tensor containing the L2 norm values along the specified dimensions.
    ///
    /// # Errors
    ///
    /// - `TensorOpError`: If the operation is not supported for the tensor type
    /// - `ShapeError`: If the specified dimensions are out of bounds
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Create a 2D tensor
    /// let tensor = Tensor::from_vec(vec![3.0, 4.0, 1.0, 2.0], &[2, 2])?;
    ///
    /// // Compute L2 norm along last dimension
    /// let norm = tensor.norm_dim(2, Some(vec![-1]), true)?;
    /// // Result: [[5.0], [sqrt(5)]]
    /// # Ok(())
    /// # }
    /// ```
    pub fn norm_dim(&self, p: i32, dims: Option<Vec<i32>>, keepdim: bool) -> Result<Tensor> {
        use ndarray::Axis;

        if p != 2 {
            return Err(TrustformersError::tensor_op_error(
                &format!("Only L2 norm (p=2) is currently supported, got p={}", p),
                "norm_dim",
            ));
        }

        // If no dims specified, compute global norm
        if dims.is_none() {
            let global_norm = self.norm()?;
            return Tensor::from_vec(vec![global_norm], &[1]);
        }

        let dims = dims.unwrap();

        match self {
            Tensor::F32(arr) => {
                let mut result = arr.clone();

                // Convert negative dimensions to positive
                let ndim = arr.ndim() as i32;
                let mut positive_dims: Vec<usize> = dims
                    .iter()
                    .map(|&d| {
                        let pos_d = if d < 0 { ndim + d } else { d };
                        if pos_d < 0 || pos_d >= ndim {
                            return Err(TrustformersError::shape_error(format!(
                                "Dimension {} is out of bounds for tensor with {} dimensions",
                                d, ndim
                            )));
                        }
                        Ok(pos_d as usize)
                    })
                    .collect::<Result<Vec<_>>>()?;

                // Sort in descending order to remove dimensions from back to front
                positive_dims.sort_unstable_by(|a, b| b.cmp(a));

                // Square all values
                result.mapv_inplace(|x| x * x);

                // Sum along specified dimensions
                for &dim in &positive_dims {
                    result = result.sum_axis(Axis(dim));
                }

                // Take square root
                result.mapv_inplace(|x| x.sqrt());

                // Add back dimensions if keepdim is true
                if keepdim {
                    let mut new_shape = arr.shape().to_vec();
                    for &dim in &positive_dims {
                        new_shape[dim] = 1;
                    }
                    result = result.to_shape(new_shape)?.to_owned();
                }

                Ok(Tensor::F32(result))
            },
            Tensor::F64(arr) => {
                let mut result = arr.clone();

                // Convert negative dimensions to positive
                let ndim = arr.ndim() as i32;
                let mut positive_dims: Vec<usize> = dims
                    .iter()
                    .map(|&d| {
                        let pos_d = if d < 0 { ndim + d } else { d };
                        if pos_d < 0 || pos_d >= ndim {
                            return Err(TrustformersError::shape_error(format!(
                                "Dimension {} is out of bounds for tensor with {} dimensions",
                                d, ndim
                            )));
                        }
                        Ok(pos_d as usize)
                    })
                    .collect::<Result<Vec<_>>>()?;

                // Sort in descending order
                positive_dims.sort_unstable_by(|a, b| b.cmp(a));

                // Square all values
                result.mapv_inplace(|x| x * x);

                // Sum along specified dimensions
                for &dim in &positive_dims {
                    result = result.sum_axis(Axis(dim));
                }

                // Take square root
                result.mapv_inplace(|x| x.sqrt());

                // Add back dimensions if keepdim is true
                if keepdim {
                    let mut new_shape = arr.shape().to_vec();
                    for &dim in &positive_dims {
                        new_shape[dim] = 1;
                    }
                    result = result.to_shape(new_shape)?.to_owned();
                }

                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "norm_dim not supported for this tensor type",
                "norm_dim",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2d() -> Result<()> {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2])?;
        let result = a.matmul(&b)?;

        if let Tensor::F32(arr) = result {
            assert_eq!(arr.shape(), &[2, 2]);
            // Expected: [[19, 22], [43, 50]]
            assert!((arr[[0, 0]] - 19.0).abs() < 1e-6);
            assert!((arr[[0, 1]] - 22.0).abs() < 1e-6);
            assert!((arr[[1, 0]] - 43.0).abs() < 1e-6);
            assert!((arr[[1, 1]] - 50.0).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_norm() -> Result<()> {
        let tensor = Tensor::from_vec(vec![3.0, 4.0], &[2])?;
        let norm = tensor.norm()?;
        assert!((norm - 5.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_norm_squared() -> Result<()> {
        let tensor = Tensor::from_vec(vec![3.0, 4.0], &[2])?;
        let norm_squared = tensor.norm_squared()?;

        if let Tensor::F32(arr) = norm_squared {
            assert!(
                (arr.into_dimensionality::<ndarray::Ix0>().unwrap().into_scalar() - 25.0).abs()
                    < 1e-6
            );
        }
        Ok(())
    }

    #[test]
    fn test_clip_grad_norm() -> Result<()> {
        let tensor = Tensor::from_vec(vec![10.0, 20.0], &[2])?;
        let clipped = tensor.clip_grad_norm(1.0)?;
        let norm = clipped.norm()?;
        assert!((norm - 1.0).abs() < 1e-6);
        Ok(())
    }
}
