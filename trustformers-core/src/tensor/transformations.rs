//! Tensor transformation operations.
//!
//! This module contains functions for manipulating tensor shapes and structure.

use super::Tensor;
use crate::errors::{Result, TrustformersError};
use scirs2_core::ndarray::{s, ArrayD, Axis, IxDyn};

impl Tensor {
    /// Helper to normalize negative axis indices.
    ///
    /// Converts negative indices (e.g., -1 for last dimension) to positive indices.
    fn normalize_axis(&self, axis: i64, for_insert: bool) -> Result<usize> {
        let ndim = self.shape().len();
        let max_val = if for_insert { ndim + 1 } else { ndim };

        if axis < 0 {
            let normalized = (max_val as i64 + axis) as usize;
            if normalized >= max_val {
                return Err(TrustformersError::shape_error(format!(
                    "Axis {} out of bounds for tensor with {} dimensions",
                    axis, ndim
                )));
            }
            Ok(normalized)
        } else {
            let axis_usize = axis as usize;
            if axis_usize >= max_val {
                return Err(TrustformersError::shape_error(format!(
                    "Axis {} out of bounds for tensor with {} dimensions",
                    axis, ndim
                )));
            }
            Ok(axis_usize)
        }
    }

    /// Transpose two dimensions of the tensor (accepts negative indices).
    ///
    /// # Arguments
    ///
    /// * `dim0` - First dimension to transpose (negative indices count from the end)
    /// * `dim1` - Second dimension to transpose (negative indices count from the end)
    ///
    /// # Returns
    ///
    /// A tensor with the specified dimensions transposed.
    pub fn transpose_i64(&self, dim0: i64, dim1: i64) -> Result<Tensor> {
        let normalized_dim0 = self.normalize_axis(dim0, false)?;
        let normalized_dim1 = self.normalize_axis(dim1, false)?;
        self.transpose(normalized_dim0, normalized_dim1)
    }

    /// Transpose two dimensions of the tensor.
    ///
    /// # Arguments
    ///
    /// * `dim0` - First dimension to transpose
    /// * `dim1` - Second dimension to transpose
    ///
    /// # Returns
    ///
    /// A tensor with the specified dimensions transposed.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                if dim0 >= a.ndim() || dim1 >= a.ndim() {
                    return Err(TrustformersError::shape_error(format!(
                        "Dimension out of bounds: tensor has {} dimensions, tried to transpose dimensions {} and {}",
                        a.ndim(), dim0, dim1
                    )));
                }
                let mut result = a.clone();
                result.swap_axes(dim0, dim1);
                // Ensure contiguous memory layout
                let contiguous_result = result.to_owned();
                Ok(Tensor::F32(contiguous_result))
            },
            Tensor::F64(a) => {
                if dim0 >= a.ndim() || dim1 >= a.ndim() {
                    return Err(TrustformersError::shape_error(format!(
                        "Dimension out of bounds: tensor has {} dimensions, tried to transpose dimensions {} and {}",
                        a.ndim(), dim0, dim1
                    )));
                }
                let mut result = a.clone();
                result.swap_axes(dim0, dim1);
                // Ensure contiguous memory layout
                let contiguous_result = result.to_owned();
                Ok(Tensor::F64(contiguous_result))
            },
            Tensor::I64(a) => {
                if dim0 >= a.ndim() || dim1 >= a.ndim() {
                    return Err(TrustformersError::shape_error(format!(
                        "Dimension out of bounds: tensor has {} dimensions, tried to transpose dimensions {} and {}",
                        a.ndim(), dim0, dim1
                    )));
                }
                let mut result = a.clone();
                result.swap_axes(dim0, dim1);
                // Ensure contiguous memory layout
                let contiguous_result = result.to_owned();
                Ok(Tensor::I64(contiguous_result))
            },
            Tensor::C32(a) => {
                if dim0 >= a.ndim() || dim1 >= a.ndim() {
                    return Err(TrustformersError::shape_error(format!(
                        "Dimension out of bounds: tensor has {} dimensions, tried to transpose dimensions {} and {}",
                        a.ndim(), dim0, dim1
                    )));
                }
                let mut result = a.clone();
                result.swap_axes(dim0, dim1);
                // Ensure contiguous memory layout
                let contiguous_result = result.to_owned();
                Ok(Tensor::C32(contiguous_result))
            },
            Tensor::C64(a) => {
                if dim0 >= a.ndim() || dim1 >= a.ndim() {
                    return Err(TrustformersError::shape_error(format!(
                        "Dimension out of bounds: tensor has {} dimensions, tried to transpose dimensions {} and {}",
                        a.ndim(), dim0, dim1
                    )));
                }
                let mut result = a.clone();
                result.swap_axes(dim0, dim1);
                // Ensure contiguous memory layout
                let contiguous_result = result.to_owned();
                Ok(Tensor::C64(contiguous_result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Transpose not supported for this tensor type",
                "transpose",
            )),
        }
    }

    /// Transpose (convenience method for 2D).
    pub fn t(&self) -> Result<Tensor> {
        self.transpose(0, 1)
    }

    /// Slice the tensor along a specific axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to slice along
    /// * `start` - Start index
    /// * `end` - End index (exclusive)
    ///
    /// # Returns
    ///
    /// A tensor slice.
    pub fn slice(&self, axis: usize, start: usize, end: usize) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let slice = a.slice_axis(Axis(axis), (start..end).into());
                Ok(Tensor::F32(slice.to_owned()))
            },
            Tensor::F64(a) => {
                let slice = a.slice_axis(Axis(axis), (start..end).into());
                Ok(Tensor::F64(slice.to_owned()))
            },
            Tensor::I64(a) => {
                let slice = a.slice_axis(Axis(axis), (start..end).into());
                Ok(Tensor::I64(slice.to_owned()))
            },
            Tensor::C32(a) => {
                let slice = a.slice_axis(Axis(axis), (start..end).into());
                Ok(Tensor::C32(slice.to_owned()))
            },
            Tensor::C64(a) => {
                let slice = a.slice_axis(Axis(axis), (start..end).into());
                Ok(Tensor::C64(slice.to_owned()))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Slice not supported for this tensor type",
                "slice",
            )),
        }
    }

    /// Multi-dimensional slice of the tensor.
    ///
    /// # Arguments
    ///
    /// * `ranges` - Slice of tuples (start, end) for each dimension
    ///
    /// # Returns
    ///
    /// A tensor slice.
    pub fn slice_multi(&self, ranges: &[(usize, usize)]) -> Result<Tensor> {
        let shape = self.shape();
        if ranges.len() != shape.len() {
            return Err(TrustformersError::shape_error(format!(
                "Slice dimensions {} do not match tensor dimensions {}",
                ranges.len(),
                shape.len()
            )));
        }

        match self {
            Tensor::F32(a) => {
                // For now, use a simple approach with chain slicing
                let mut result = a.clone();
                for (i, &(start, end)) in ranges.iter().enumerate() {
                    if end > shape[i] {
                        return Err(TrustformersError::shape_error(format!(
                            "Slice end {} exceeds dimension size {} for axis {}",
                            end, shape[i], i
                        )));
                    }
                    result = result.slice_axis(Axis(i), (start..end).into()).to_owned();
                }
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let mut result = a.clone();
                for (i, &(start, end)) in ranges.iter().enumerate() {
                    if end > shape[i] {
                        return Err(TrustformersError::shape_error(format!(
                            "Slice end {} exceeds dimension size {} for axis {}",
                            end, shape[i], i
                        )));
                    }
                    result = result.slice_axis(Axis(i), (start..end).into()).to_owned();
                }
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                let mut result = a.clone();
                for (i, &(start, end)) in ranges.iter().enumerate() {
                    if end > shape[i] {
                        return Err(TrustformersError::shape_error(format!(
                            "Slice end {} exceeds dimension size {} for axis {}",
                            end, shape[i], i
                        )));
                    }
                    result = result.slice_axis(Axis(i), (start..end).into()).to_owned();
                }
                Ok(Tensor::I64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Multi-dimensional slice not supported for this tensor type",
                "slice_multi",
            )),
        }
    }

    /// Split the tensor into chunks along an axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to split along
    /// * `split_size` - Size of each chunk
    ///
    /// # Returns
    ///
    /// A vector of tensor chunks.
    pub fn split(&self, axis: usize, split_size: usize) -> Result<Vec<Tensor>> {
        match self {
            Tensor::F32(a) => {
                let dim_size = a.shape()[axis];
                let mut chunks = Vec::new();

                for start in (0..dim_size).step_by(split_size) {
                    let end = (start + split_size).min(dim_size);
                    let chunk = a.slice_axis(Axis(axis), (start..end).into());
                    chunks.push(Tensor::F32(chunk.to_owned()));
                }

                Ok(chunks)
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Split not supported for this tensor type",
                "split",
            )),
        }
    }

    /// Reshape the tensor to a new shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The new shape
    ///
    /// # Returns
    ///
    /// A tensor with the new shape.
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                // Ensure contiguous layout before reshaping
                let contiguous = a.to_owned();
                let reshaped = contiguous
                    .into_shape_with_order(IxDyn(shape))
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                Ok(Tensor::F32(reshaped))
            },
            Tensor::F64(a) => {
                // Ensure contiguous layout before reshaping
                let contiguous = a.to_owned();
                let reshaped = contiguous
                    .into_shape_with_order(IxDyn(shape))
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                Ok(Tensor::F64(reshaped))
            },
            Tensor::I64(a) => {
                // Ensure contiguous layout before reshaping
                let contiguous = a.to_owned();
                let reshaped = contiguous
                    .into_shape_with_order(IxDyn(shape))
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                Ok(Tensor::I64(reshaped))
            },
            Tensor::C32(a) => {
                // Ensure contiguous layout before reshaping
                let contiguous = a.to_owned();
                let reshaped = contiguous
                    .into_shape_with_order(IxDyn(shape))
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                Ok(Tensor::C32(reshaped))
            },
            Tensor::C64(a) => {
                // Ensure contiguous layout before reshaping
                let contiguous = a.to_owned();
                let reshaped = contiguous
                    .into_shape_with_order(IxDyn(shape))
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                Ok(Tensor::C64(reshaped))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Reshape not supported for this tensor type",
                "reshape",
            )),
        }
    }

    /// Flatten tensor dimensions from start_dim to end_dim (inclusive).
    ///
    /// # Arguments
    ///
    /// * `start_dim` - Starting dimension to flatten (supports negative indexing)
    /// * `end_dim` - Ending dimension to flatten (supports negative indexing)
    ///
    /// # Returns
    ///
    /// A tensor with flattened dimensions.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let t = Tensor::randn(&[2, 3, 4, 5])?;
    /// let flattened = t.flatten(1, 2)?; // Shape becomes [2, 12, 5]
    /// ```
    pub fn flatten(&self, start_dim: i64, end_dim: i64) -> Result<Tensor> {
        let shape = self.shape();
        let ndim = shape.len();

        // Normalize axes
        let start = self.normalize_axis(start_dim, false)?;
        let end = self.normalize_axis(end_dim, false)?;

        if start > end {
            return Err(TrustformersError::shape_error(format!(
                "start_dim {} must be <= end_dim {}",
                start, end
            )));
        }

        // Calculate new shape
        let mut new_shape = Vec::new();

        // Keep dimensions before start_dim
        new_shape.extend_from_slice(&shape[..start]);

        // Flatten dimensions from start to end (inclusive)
        let flattened_size: usize = shape[start..=end].iter().product();
        new_shape.push(flattened_size);

        // Keep dimensions after end_dim
        if end + 1 < ndim {
            new_shape.extend_from_slice(&shape[end + 1..]);
        }

        self.reshape(&new_shape)
    }

    /// Slice with multiple ranges.
    ///
    /// # Arguments
    ///
    /// * `ranges` - Vector of (start, end) pairs for each dimension
    ///
    /// # Returns
    ///
    /// A tensor slice.
    pub fn slice_ranges(&self, ranges: &[(usize, usize)]) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let mut result = a.clone();
                for (axis, &(start, end)) in ranges.iter().enumerate() {
                    result = result.slice_axis(Axis(axis), (start..end).into()).to_owned();
                }
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let mut result = a.clone();
                for (axis, &(start, end)) in ranges.iter().enumerate() {
                    result = result.slice_axis(Axis(axis), (start..end).into()).to_owned();
                }
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                let mut result = a.clone();
                for (axis, &(start, end)) in ranges.iter().enumerate() {
                    result = result.slice_axis(Axis(axis), (start..end).into()).to_owned();
                }
                Ok(Tensor::I64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Multi-range slice not supported for this tensor type",
                "slice_ranges",
            )),
        }
    }

    /// Concatenate multiple tensors along an axis.
    ///
    /// # Arguments
    ///
    /// * `tensors` - Vector of tensors to concatenate
    /// * `axis` - The axis to concatenate along
    ///
    /// # Returns
    ///
    /// A concatenated tensor.
    pub fn concat(tensors: &[Tensor], axis: usize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(TrustformersError::tensor_op_error(
                "Cannot concatenate empty tensor list",
                "concat",
            ));
        }

        // Check all tensors are the same type
        let first_type = std::mem::discriminant(&tensors[0]);
        for tensor in tensors.iter().skip(1) {
            if std::mem::discriminant(tensor) != first_type {
                return Err(TrustformersError::tensor_op_error(
                    "All tensors must have the same type for concatenation",
                    "concat",
                ));
            }
        }

        match &tensors[0] {
            Tensor::F32(_) => {
                let arrays: Vec<_> = tensors
                    .iter()
                    .map(|t| match t {
                        Tensor::F32(a) => a.view(),
                        _ => unreachable!(),
                    })
                    .collect();

                let result = ndarray::concatenate(Axis(axis), &arrays)
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                Ok(Tensor::F32(result))
            },
            Tensor::F64(_) => {
                let arrays: Vec<_> = tensors
                    .iter()
                    .map(|t| match t {
                        Tensor::F64(a) => a.view(),
                        _ => unreachable!(),
                    })
                    .collect();

                let result = ndarray::concatenate(Axis(axis), &arrays)
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                Ok(Tensor::F64(result))
            },
            Tensor::I64(_) => {
                let arrays: Vec<_> = tensors
                    .iter()
                    .map(|t| match t {
                        Tensor::I64(a) => a.view(),
                        _ => unreachable!(),
                    })
                    .collect();

                let result = ndarray::concatenate(Axis(axis), &arrays)
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                Ok(Tensor::I64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Concatenation not supported for this tensor type",
                "concat",
            )),
        }
    }

    /// Sort the tensor.
    ///
    /// # Returns
    ///
    /// A sorted tensor.
    pub fn sort(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let mut data = a.iter().cloned().collect::<Vec<_>>();
                data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let result = ArrayD::from_shape_vec(a.raw_dim(), data)
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                Ok(Tensor::F32(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Sort not supported for this tensor type",
                "sort",
            )),
        }
    }

    /// Zero-padding for embeddings.
    ///
    /// # Arguments
    ///
    /// * `padding_idx` - Index to zero out
    ///
    /// # Returns
    ///
    /// A tensor with the specified index zeroed.
    pub fn zero_padding_embedding(&self, padding_idx: usize) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let mut result = a.clone();
                if padding_idx < a.shape()[0] {
                    result.slice_mut(s![padding_idx, ..]).fill(0.0);
                }
                Ok(Tensor::F32(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Zero padding not supported for this tensor type",
                "zero_padding_embedding",
            )),
        }
    }

    /// Select along a specific dimension with an index.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension to select along
    /// * `index` - The index to select (can be negative for indexing from the end)
    ///
    /// # Returns
    ///
    /// A tensor with the specified index selected along the given dimension.
    pub fn select(&self, dim: usize, index: i64) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let shape = a.shape();
                if dim >= shape.len() {
                    return Err(TrustformersError::shape_error(format!(
                        "Dimension {} out of bounds for tensor with {} dimensions",
                        dim,
                        shape.len()
                    )));
                }

                let axis_size = shape[dim] as i64;
                let actual_index = if index < 0 { axis_size + index } else { index };

                if actual_index < 0 || actual_index >= axis_size {
                    return Err(TrustformersError::shape_error(format!(
                        "Index {} out of bounds for dimension {} with size {}",
                        index, dim, axis_size
                    )));
                }

                let result = a.index_axis(ndarray::Axis(dim), actual_index as usize).to_owned();
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let shape = a.shape();
                if dim >= shape.len() {
                    return Err(TrustformersError::shape_error(format!(
                        "Dimension {} out of bounds for tensor with {} dimensions",
                        dim,
                        shape.len()
                    )));
                }

                let axis_size = shape[dim] as i64;
                let actual_index = if index < 0 { axis_size + index } else { index };

                if actual_index < 0 || actual_index >= axis_size {
                    return Err(TrustformersError::shape_error(format!(
                        "Index {} out of bounds for dimension {} with size {}",
                        index, dim, axis_size
                    )));
                }

                let result = a.index_axis(ndarray::Axis(dim), actual_index as usize).to_owned();
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                let shape = a.shape();
                if dim >= shape.len() {
                    return Err(TrustformersError::shape_error(format!(
                        "Dimension {} out of bounds for tensor with {} dimensions",
                        dim,
                        shape.len()
                    )));
                }

                let axis_size = shape[dim] as i64;
                let actual_index = if index < 0 { axis_size + index } else { index };

                if actual_index < 0 || actual_index >= axis_size {
                    return Err(TrustformersError::shape_error(format!(
                        "Index {} out of bounds for dimension {} with size {}",
                        index, dim, axis_size
                    )));
                }

                let result = a.index_axis(ndarray::Axis(dim), actual_index as usize).to_owned();
                Ok(Tensor::I64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Select not supported for this tensor type",
                "select",
            )),
        }
    }

    /// Select the first token from a sequence.
    ///
    /// # Returns
    ///
    /// A tensor with the first token selected.
    pub fn select_first_token(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                if a.ndim() < 2 {
                    return Err(TrustformersError::shape_error(
                        "Tensor must have at least 2 dimensions".into(),
                    ));
                }
                let result = a
                    .slice_axis(ndarray::Axis(1), ndarray::Slice::from(0..1))
                    .remove_axis(ndarray::Axis(1))
                    .to_owned();
                Ok(Tensor::F32(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Select first token not supported for this tensor type",
                "select_first_token",
            )),
        }
    }

    /// Ensure tensor has contiguous memory layout.
    ///
    /// # Returns
    ///
    /// A tensor with contiguous memory layout.
    pub fn contiguous(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                if a.is_standard_layout() {
                    // Already contiguous
                    Ok(self.clone())
                } else {
                    // Make contiguous by creating owned copy
                    Ok(Tensor::F32(a.to_owned()))
                }
            },
            Tensor::F64(a) => {
                if a.is_standard_layout() {
                    Ok(self.clone())
                } else {
                    Ok(Tensor::F64(a.to_owned()))
                }
            },
            Tensor::I64(a) => {
                if a.is_standard_layout() {
                    Ok(self.clone())
                } else {
                    Ok(Tensor::I64(a.to_owned()))
                }
            },
            Tensor::C32(a) => {
                if a.is_standard_layout() {
                    Ok(self.clone())
                } else {
                    Ok(Tensor::C32(a.to_owned()))
                }
            },
            Tensor::C64(a) => {
                if a.is_standard_layout() {
                    Ok(self.clone())
                } else {
                    Ok(Tensor::C64(a.to_owned()))
                }
            },
            _ => {
                // For other tensor types, just return a clone since they're typically already contiguous
                Ok(self.clone())
            },
        }
    }

    /// Permute tensor dimensions.
    ///
    /// # Arguments
    ///
    /// * `permutation` - Vector specifying the new order of dimensions
    ///
    /// # Returns
    ///
    /// A tensor with permuted dimensions.
    pub fn permute(&self, permutation: &[usize]) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let mut result = a.clone();
                result = result.permuted_axes(permutation);
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let mut result = a.clone();
                result = result.permuted_axes(permutation);
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                let mut result = a.clone();
                result = result.permuted_axes(permutation);
                Ok(Tensor::I64(result))
            },
            Tensor::C32(a) => {
                let mut result = a.clone();
                result = result.permuted_axes(permutation);
                Ok(Tensor::C32(result))
            },
            Tensor::C64(a) => {
                let mut result = a.clone();
                result = result.permuted_axes(permutation);
                Ok(Tensor::C64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Permute not supported for this tensor type",
                "permute",
            )),
        }
    }

    /// Add a new dimension at the specified axis (accepts negative indices).
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis where to insert the new dimension (negative indices count from the end)
    ///
    /// # Returns
    ///
    /// A tensor with an added dimension.
    pub fn unsqueeze_i64(&self, axis: i64) -> Result<Tensor> {
        let normalized_axis = self.normalize_axis(axis, true)?;
        self.unsqueeze(normalized_axis)
    }

    /// Add a new dimension at the specified axis.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis where to insert the new dimension
    ///
    /// # Returns
    ///
    /// A tensor with an added dimension.
    pub fn unsqueeze(&self, axis: usize) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.clone().insert_axis(Axis(axis));
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.clone().insert_axis(Axis(axis));
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                let result = a.clone().insert_axis(Axis(axis));
                Ok(Tensor::I64(result))
            },
            Tensor::C32(a) => {
                let result = a.clone().insert_axis(Axis(axis));
                Ok(Tensor::C32(result))
            },
            Tensor::C64(a) => {
                let result = a.clone().insert_axis(Axis(axis));
                Ok(Tensor::C64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsqueeze not supported for this tensor type",
                "unsqueeze",
            )),
        }
    }

    /// Removes a single-dimensional entry from the shape of the tensor (accepts negative indices).
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to remove (must have size 1, negative indices count from the end)
    ///
    /// # Returns
    ///
    /// A tensor with the specified dimension removed.
    pub fn squeeze_i64(&self, axis: i64) -> Result<Tensor> {
        let normalized_axis = self.normalize_axis(axis, false)?;
        self.squeeze(normalized_axis)
    }

    /// Removes a single-dimensional entry from the shape of the tensor.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis to remove (must have size 1)
    ///
    /// # Returns
    ///
    /// A tensor with the specified dimension removed.
    pub fn squeeze(&self, axis: usize) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                if a.shape()[axis] != 1 {
                    return Err(TrustformersError::tensor_op_error(
                        &format!("Cannot squeeze axis {} with size {}", axis, a.shape()[axis]),
                        "squeeze",
                    ));
                }
                let result = a.clone().remove_axis(Axis(axis));
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                if a.shape()[axis] != 1 {
                    return Err(TrustformersError::tensor_op_error(
                        &format!("Cannot squeeze axis {} with size {}", axis, a.shape()[axis]),
                        "squeeze",
                    ));
                }
                let result = a.clone().remove_axis(Axis(axis));
                Ok(Tensor::F64(result))
            },
            Tensor::I64(a) => {
                if a.shape()[axis] != 1 {
                    return Err(TrustformersError::tensor_op_error(
                        &format!("Cannot squeeze axis {} with size {}", axis, a.shape()[axis]),
                        "squeeze",
                    ));
                }
                let result = a.clone().remove_axis(Axis(axis));
                Ok(Tensor::I64(result))
            },
            Tensor::C32(a) => {
                if a.shape()[axis] != 1 {
                    return Err(TrustformersError::tensor_op_error(
                        &format!("Cannot squeeze axis {} with size {}", axis, a.shape()[axis]),
                        "squeeze",
                    ));
                }
                let result = a.clone().remove_axis(Axis(axis));
                Ok(Tensor::C32(result))
            },
            Tensor::C64(a) => {
                if a.shape()[axis] != 1 {
                    return Err(TrustformersError::tensor_op_error(
                        &format!("Cannot squeeze axis {} with size {}", axis, a.shape()[axis]),
                        "squeeze",
                    ));
                }
                let result = a.clone().remove_axis(Axis(axis));
                Ok(Tensor::C64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Squeeze not supported for this tensor type",
                "squeeze",
            )),
        }
    }

    /// Extract a scalar value from a 0-dimensional tensor.
    ///
    /// # Returns
    ///
    /// The scalar value as f32.
    pub fn to_scalar(&self) -> Result<f32> {
        match self {
            Tensor::F32(a) => {
                if a.ndim() != 0 {
                    return Err(TrustformersError::tensor_op_error(
                        "Tensor must be 0-dimensional to extract scalar",
                        "to_scalar",
                    ));
                }
                Ok(a[ndarray::IxDyn(&[])])
            },
            Tensor::F64(a) => {
                if a.ndim() != 0 {
                    return Err(TrustformersError::tensor_op_error(
                        "Tensor must be 0-dimensional to extract scalar",
                        "to_scalar",
                    ));
                }
                Ok(a[ndarray::IxDyn(&[])] as f32)
            },
            Tensor::I64(a) => {
                if a.ndim() != 0 {
                    return Err(TrustformersError::tensor_op_error(
                        "Tensor must be 0-dimensional to extract scalar",
                        "to_scalar",
                    ));
                }
                Ok(a[ndarray::IxDyn(&[])] as f32)
            },
            _ => Err(TrustformersError::tensor_op_error(
                "to_scalar not supported for this tensor type",
                "to_scalar",
            )),
        }
    }

    /// Gathers values along an axis specified by an index tensor.
    ///
    /// This is a PyTorch-style gather operation that selects values from the input tensor
    /// along the specified dimension according to the indices in the index tensor.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension along which to gather (supports negative indexing)
    /// * `index` - Tensor containing indices to gather
    ///
    /// # Returns
    ///
    /// A tensor with gathered values.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let tensor = Tensor::randn(&[3, 4, 5])?;
    /// let indices = Tensor::from_vec(vec![0, 2, 1], &[3, 1, 1])?;
    /// let gathered = tensor.gather(-2, &indices)?;
    /// ```
    pub fn gather(&self, dim: i64, index: &Tensor) -> Result<Tensor> {
        let normalized_dim = self.normalize_axis(dim, false)?;

        match (self, index) {
            (Tensor::F32(data), Tensor::I64(idx)) => {
                let data_shape = data.shape();
                let idx_shape = idx.shape();

                // Verify shapes are compatible
                if data_shape.len() != idx_shape.len() {
                    return Err(TrustformersError::tensor_op_error(
                        "Index tensor must have same number of dimensions as input tensor",
                        "gather",
                    ));
                }

                // Create output with same shape as index
                let mut result = ArrayD::zeros(IxDyn(idx_shape));

                // Simplified gather implementation
                // For each element in the index tensor, gather the corresponding value
                for idx_flat in 0..idx.len() {
                    let mut idx_coords = Vec::new();
                    let mut remaining = idx_flat;

                    // Convert flat index to coordinates
                    for &dim_size in idx_shape.iter().rev() {
                        idx_coords.push(remaining % dim_size);
                        remaining /= dim_size;
                    }
                    idx_coords.reverse();

                    // Build data coordinates by replacing the gather dimension with the index value
                    let mut data_coords = idx_coords.clone();
                    let gather_idx = idx[IxDyn(&idx_coords)] as usize;

                    if gather_idx >= data_shape[normalized_dim] {
                        return Err(TrustformersError::tensor_op_error(
                            &format!(
                                "Index {} out of bounds for dimension {} with size {}",
                                gather_idx, normalized_dim, data_shape[normalized_dim]
                            ),
                            "gather",
                        ));
                    }

                    data_coords[normalized_dim] = gather_idx;

                    // Gather the value
                    result[IxDyn(&idx_coords)] = data[IxDyn(&data_coords)];
                }

                Ok(Tensor::F32(result))
            },
            (Tensor::F64(data), Tensor::I64(idx)) => {
                let data_shape = data.shape();
                let idx_shape = idx.shape();

                if data_shape.len() != idx_shape.len() {
                    return Err(TrustformersError::tensor_op_error(
                        "Index tensor must have same number of dimensions as input tensor",
                        "gather",
                    ));
                }

                let mut result = ArrayD::zeros(IxDyn(idx_shape));

                for idx_flat in 0..idx.len() {
                    let mut idx_coords = Vec::new();
                    let mut remaining = idx_flat;

                    for &dim_size in idx_shape.iter().rev() {
                        idx_coords.push(remaining % dim_size);
                        remaining /= dim_size;
                    }
                    idx_coords.reverse();

                    let mut data_coords = idx_coords.clone();
                    let gather_idx = idx[IxDyn(&idx_coords)] as usize;

                    if gather_idx >= data_shape[normalized_dim] {
                        return Err(TrustformersError::tensor_op_error(
                            &format!(
                                "Index {} out of bounds for dimension {} with size {}",
                                gather_idx, normalized_dim, data_shape[normalized_dim]
                            ),
                            "gather",
                        ));
                    }

                    data_coords[normalized_dim] = gather_idx;
                    result[IxDyn(&idx_coords)] = data[IxDyn(&data_coords)];
                }

                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Gather only supports F32/F64 tensors with I64 indices",
                "gather",
            )),
        }
    }

    /// Repeat tensor elements along specified dimensions.
    ///
    /// Repeats the tensor along each dimension according to the specified repetition counts.
    ///
    /// # Arguments
    ///
    /// * `repeats` - Number of times to repeat along each dimension. If the length is less
    ///   than the number of dimensions, repeats are prepended with 1s.
    ///
    /// # Returns
    ///
    /// A new tensor with repeated elements.
    ///
    /// # Errors
    ///
    /// - `TensorOpError`: If the operation fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tensor = Tensor::from_vec(vec![1.0, 2.0], &[2])?;
    ///
    /// // Repeat 3 times along dimension 0
    /// let repeated = tensor.repeat(&[3])?;
    /// // Result: [1.0, 2.0, 1.0, 2.0, 1.0, 2.0] with shape [6]
    /// # Ok(())
    /// # }
    /// ```
    pub fn repeat(&self, repeats: &[usize]) -> Result<Tensor> {
        use ndarray::concatenate;

        match self {
            Tensor::F32(arr) => {
                let mut result = arr.clone();
                let ndim = arr.ndim();

                // Pad repeats with 1s if needed
                let mut full_repeats = vec![1; ndim];
                let offset = ndim.saturating_sub(repeats.len());
                for (i, &r) in repeats.iter().enumerate() {
                    full_repeats[offset + i] = r;
                }

                // Repeat along each dimension
                for (dim, &repeat_count) in full_repeats.iter().enumerate() {
                    if repeat_count > 1 {
                        let views: Vec<_> = (0..repeat_count).map(|_| result.view()).collect();
                        result = concatenate(Axis(dim), &views)?;
                    }
                }

                Ok(Tensor::F32(result))
            },
            Tensor::F64(arr) => {
                let mut result = arr.clone();
                let ndim = arr.ndim();

                let mut full_repeats = vec![1; ndim];
                let offset = ndim.saturating_sub(repeats.len());
                for (i, &r) in repeats.iter().enumerate() {
                    full_repeats[offset + i] = r;
                }

                for (dim, &repeat_count) in full_repeats.iter().enumerate() {
                    if repeat_count > 1 {
                        let views: Vec<_> = (0..repeat_count).map(|_| result.view()).collect();
                        result = concatenate(Axis(dim), &views)?;
                    }
                }

                Ok(Tensor::F64(result))
            },
            Tensor::I64(arr) => {
                let mut result = arr.clone();
                let ndim = arr.ndim();

                let mut full_repeats = vec![1; ndim];
                let offset = ndim.saturating_sub(repeats.len());
                for (i, &r) in repeats.iter().enumerate() {
                    full_repeats[offset + i] = r;
                }

                for (dim, &repeat_count) in full_repeats.iter().enumerate() {
                    if repeat_count > 1 {
                        let views: Vec<_> = (0..repeat_count).map(|_| result.view()).collect();
                        result = concatenate(Axis(dim), &views)?;
                    }
                }

                Ok(Tensor::I64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "repeat not supported for this tensor type",
                "repeat",
            )),
        }
    }

    /// Upsample a 4D tensor using nearest neighbor interpolation.
    ///
    /// This function performs upsampling on a 4D tensor (typically for image data in NCHW format).
    /// Currently supports nearest neighbor interpolation which is simple and efficient.
    ///
    /// # Arguments
    ///
    /// * `scale_factor` - Scaling factor for spatial dimensions (height and width)
    ///
    /// # Returns
    ///
    /// An upsampled tensor with spatial dimensions multiplied by scale_factor.
    ///
    /// # Errors
    ///
    /// - `ShapeError`: If the tensor is not 4D
    /// - `TensorOpError`: If the operation fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// // Create a 4D tensor [batch, channels, height, width]
    /// let tensor = Tensor::zeros(&[1, 3, 8, 8])?;
    ///
    /// // Upsample by factor of 2
    /// let upsampled = tensor.upsample_nearest(2)?;
    /// // Result shape: [1, 3, 16, 16]
    /// # Ok(())
    /// # }
    /// ```
    pub fn upsample_nearest(&self, scale_factor: usize) -> Result<Tensor> {
        let shape = self.shape();

        if shape.len() != 4 {
            return Err(TrustformersError::shape_error(format!(
                "upsample_nearest expects 4D tensor (NCHW), got {}D",
                shape.len()
            )));
        }

        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let new_h = h * scale_factor;
        let new_w = w * scale_factor;

        match self {
            Tensor::F32(arr) => {
                let mut result = ArrayD::zeros(IxDyn(&[n, c, new_h, new_w]));

                for batch in 0..n {
                    for channel in 0..c {
                        for out_h in 0..new_h {
                            for out_w in 0..new_w {
                                // Nearest neighbor: map output pixel to input pixel
                                let in_h = out_h / scale_factor;
                                let in_w = out_w / scale_factor;

                                let value = arr[[batch, channel, in_h, in_w]];
                                result[[batch, channel, out_h, out_w]] = value;
                            }
                        }
                    }
                }

                Ok(Tensor::F32(result))
            },
            Tensor::F64(arr) => {
                let mut result = ArrayD::zeros(IxDyn(&[n, c, new_h, new_w]));

                for batch in 0..n {
                    for channel in 0..c {
                        for out_h in 0..new_h {
                            for out_w in 0..new_w {
                                let in_h = out_h / scale_factor;
                                let in_w = out_w / scale_factor;

                                let value = arr[[batch, channel, in_h, in_w]];
                                result[[batch, channel, out_h, out_w]] = value;
                            }
                        }
                    }
                }

                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "upsample_nearest not supported for this tensor type",
                "upsample_nearest",
            )),
        }
    }

    /// Interpolate (upsample or downsample) a tensor using bilinear interpolation.
    ///
    /// This function performs bilinear interpolation on a 4D tensor (NCHW format).
    /// For upsampling in VAE decoders and other generative models.
    ///
    /// # Arguments
    ///
    /// * `size` - Target size as (height, width)
    ///
    /// # Returns
    ///
    /// An interpolated tensor with the specified spatial dimensions.
    ///
    /// # Errors
    ///
    /// - `ShapeError`: If the tensor is not 4D
    /// - `TensorOpError`: If the operation fails
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use trustformers_core::tensor::Tensor;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let tensor = Tensor::zeros(&[1, 3, 8, 8])?;
    ///
    /// // Interpolate to 16x16
    /// let interpolated = tensor.interpolate((16, 16))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn interpolate(&self, size: (usize, usize)) -> Result<Tensor> {
        let shape = self.shape();

        if shape.len() != 4 {
            return Err(TrustformersError::shape_error(format!(
                "interpolate expects 4D tensor (NCHW), got {}D",
                shape.len()
            )));
        }

        let (n, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let (h_out, w_out) = size;

        // If sizes match, return clone
        if h_in == h_out && w_in == w_out {
            return Ok(self.clone());
        }

        // For now, use nearest neighbor if it's an integer scale factor
        if h_out % h_in == 0 && w_out % w_in == 0 && h_out / h_in == w_out / w_in {
            return self.upsample_nearest(h_out / h_in);
        }

        match self {
            Tensor::F32(arr) => {
                let mut result = ArrayD::zeros(IxDyn(&[n, c, h_out, w_out]));

                let h_scale = (h_in - 1) as f32 / (h_out - 1).max(1) as f32;
                let w_scale = (w_in - 1) as f32 / (w_out - 1).max(1) as f32;

                for batch in 0..n {
                    for channel in 0..c {
                        for out_h in 0..h_out {
                            for out_w in 0..w_out {
                                // Compute source coordinates
                                let src_h = (out_h as f32 * h_scale).min((h_in - 1) as f32);
                                let src_w = (out_w as f32 * w_scale).min((w_in - 1) as f32);

                                let h0 = src_h.floor() as usize;
                                let w0 = src_w.floor() as usize;
                                let h1 = (h0 + 1).min(h_in - 1);
                                let w1 = (w0 + 1).min(w_in - 1);

                                let h_weight = src_h - h0 as f32;
                                let w_weight = src_w - w0 as f32;

                                // Bilinear interpolation
                                let v00 = arr[[batch, channel, h0, w0]];
                                let v01 = arr[[batch, channel, h0, w1]];
                                let v10 = arr[[batch, channel, h1, w0]];
                                let v11 = arr[[batch, channel, h1, w1]];

                                let v0 = v00 * (1.0 - w_weight) + v01 * w_weight;
                                let v1 = v10 * (1.0 - w_weight) + v11 * w_weight;

                                let value = v0 * (1.0 - h_weight) + v1 * h_weight;
                                result[[batch, channel, out_h, out_w]] = value;
                            }
                        }
                    }
                }

                Ok(Tensor::F32(result))
            },
            Tensor::F64(arr) => {
                let mut result = ArrayD::zeros(IxDyn(&[n, c, h_out, w_out]));

                let h_scale = (h_in - 1) as f64 / (h_out - 1).max(1) as f64;
                let w_scale = (w_in - 1) as f64 / (w_out - 1).max(1) as f64;

                for batch in 0..n {
                    for channel in 0..c {
                        for out_h in 0..h_out {
                            for out_w in 0..w_out {
                                let src_h = (out_h as f64 * h_scale).min((h_in - 1) as f64);
                                let src_w = (out_w as f64 * w_scale).min((w_in - 1) as f64);

                                let h0 = src_h.floor() as usize;
                                let w0 = src_w.floor() as usize;
                                let h1 = (h0 + 1).min(h_in - 1);
                                let w1 = (w0 + 1).min(w_in - 1);

                                let h_weight = src_h - h0 as f64;
                                let w_weight = src_w - w0 as f64;

                                let v00 = arr[[batch, channel, h0, w0]];
                                let v01 = arr[[batch, channel, h0, w1]];
                                let v10 = arr[[batch, channel, h1, w0]];
                                let v11 = arr[[batch, channel, h1, w1]];

                                let v0 = v00 * (1.0 - w_weight) + v01 * w_weight;
                                let v1 = v10 * (1.0 - w_weight) + v11 * w_weight;

                                let value = v0 * (1.0 - h_weight) + v1 * h_weight;
                                result[[batch, channel, out_h, out_w]] = value;
                            }
                        }
                    }
                }

                Ok(Tensor::F64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "interpolate not supported for this tensor type",
                "interpolate",
            )),
        }
    }
}
