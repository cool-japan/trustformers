//! Sparse tensor operations.
//!
//! This module contains functions for working with sparse tensors.

use super::Tensor;
use crate::errors::{Result, TrustformersError};
use crate::sparse_tensor::SparseTensor;

impl Tensor {
    /// Convert a dense tensor to sparse format.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Values below this threshold will be considered zero
    ///
    /// # Returns
    ///
    /// A sparse tensor representation.
    pub fn to_sparse(&self, threshold: f32) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let sparse = SparseTensor::from_dense(&Tensor::F32(a.clone()), threshold)?;
                Ok(Tensor::Sparse(sparse))
            },
            Tensor::Sparse(_) => {
                // Already sparse
                Ok(self.clone())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Cannot convert this tensor type to sparse",
                "Tensor::to_sparse",
            )),
        }
    }

    /// Convert a sparse tensor to dense format.
    ///
    /// # Returns
    ///
    /// A dense tensor representation.
    pub fn to_dense(&self) -> Result<Tensor> {
        match self {
            Tensor::Sparse(s) => s.to_dense(),
            Tensor::F32(_) | Tensor::F64(_) | Tensor::I64(_) => {
                // Already dense
                Ok(self.clone())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Cannot convert this tensor type to dense",
                "Tensor::to_dense",
            )),
        }
    }

    /// Check if the tensor is sparse.
    ///
    /// # Returns
    ///
    /// True if the tensor is sparse, false otherwise.
    pub fn is_sparse(&self) -> bool {
        matches!(self, Tensor::Sparse(_))
    }

    /// Get the sparsity ratio of the tensor.
    ///
    /// # Returns
    ///
    /// The ratio of zero elements to total elements.
    pub fn sparsity(&self) -> Result<f32> {
        match self {
            Tensor::Sparse(s) => Ok(s.sparsity()),
            Tensor::F32(a) => {
                let total = a.len() as f32;
                let zeros = a.iter().filter(|&&x| x == 0.0).count() as f32;
                Ok(zeros / total)
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Sparsity calculation not supported for this tensor type",
                "Tensor::sparsity",
            )),
        }
    }

    /// Get the number of non-zero elements.
    ///
    /// # Returns
    ///
    /// The number of non-zero elements.
    pub fn nnz(&self) -> Result<usize> {
        match self {
            Tensor::Sparse(s) => Ok(s.nnz()),
            Tensor::F32(a) => Ok(a.iter().filter(|&&x| x != 0.0).count()),
            _ => Err(TrustformersError::tensor_op_error(
                "NNZ calculation not supported for this tensor type",
                "Tensor::nnz",
            )),
        }
    }

    /// Create a sparse tensor in COO format.
    ///
    /// # Arguments
    ///
    /// * `indices` - Coordinate indices
    /// * `values` - Non-zero values
    /// * `shape` - Tensor shape
    ///
    /// # Returns
    ///
    /// A sparse tensor in COO format.
    pub fn sparse_coo(
        indices: Vec<Vec<usize>>,
        values: Vec<f32>,
        shape: Vec<usize>,
    ) -> Result<Tensor> {
        let sparse = SparseTensor::new_coo(shape, indices[0].clone(), indices[1].clone(), values)?;
        Ok(Tensor::Sparse(sparse))
    }

    /// Create a sparse tensor in CSR format.
    ///
    /// # Arguments
    ///
    /// * `row_ptr` - Row pointers
    /// * `col_indices` - Column indices
    /// * `values` - Non-zero values
    /// * `shape` - Tensor shape
    ///
    /// # Returns
    ///
    /// A sparse tensor in CSR format.
    pub fn sparse_csr(
        row_ptr: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<f32>,
        shape: Vec<usize>,
    ) -> Result<Tensor> {
        let sparse = SparseTensor::new_csr(shape, row_ptr, col_indices, values)?;
        Ok(Tensor::Sparse(sparse))
    }
}
