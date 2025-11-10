//! Advanced sparse tensor operations and structured sparsity
//!
//! This module provides high-performance sparse operations optimized for transformer models,
//! including sparse matrix multiplication, structured sparsity patterns, and pruning utilities.
//!
//! # Features
//!
//! - **Sparse Matrix Multiplication**: SpMM, SpMSpM with various formats
//! - **Structured Sparsity**: N:M sparsity, block sparsity, channel pruning
//! - **Sparse Attention**: Memory-efficient attention for long sequences
//! - **Pruning Utilities**: Magnitude pruning, gradient-based pruning
//! - **Format Conversion**: Efficient COO ↔ CSR ↔ CSC ↔ BSR conversions
//!
//! # Examples
//!
//! ```rust
//! use trustformers_core::sparse_ops::{sparse_matmul, StructuredSparsityPattern, NMSparsity};
//! use trustformers_core::sparse_tensor::SparseTensor;
//! use trustformers_core::tensor::Tensor;
//!
//! // Create structured N:M sparsity
//! let pattern = NMSparsity::new(2, 4); // 2:4 sparsity (50%)
//! let dense = Tensor::randn(&[128, 128])?;
//! let sparse = pattern.apply(&dense)?;
//!
//! // Sparse-dense matrix multiplication
//! let result = sparse_matmul(&sparse, &dense)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::errors::{Result, TrustformersError};
use crate::sparse_tensor::{SparseFormat, SparseIndices, SparseTensor};
use crate::tensor::Tensor;
use scirs2_core::ndarray::{Array1, Array2, ArrayD, Axis, IxDyn, s};
use scirs2_core::parallel_ops::*;
use scirs2_core::random::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Structured sparsity pattern types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StructuredSparsityPattern {
    /// N:M sparsity - N non-zero elements every M elements
    NM { n: usize, m: usize },

    /// Block sparsity - blocks of size (bh, bw) are either all zero or all non-zero
    Block { block_height: usize, block_width: usize },

    /// Channel pruning - entire channels (columns or rows) are pruned
    Channel { dimension: usize, keep_ratio: f32 },

    /// Head pruning - prune entire attention heads
    Head { num_heads: usize, keep_ratio: f32 },

    /// Random sparsity - random elements are pruned with given sparsity
    Random { sparsity: f32 },

    /// Magnitude-based - keep top-k by magnitude
    Magnitude { keep_ratio: f32 },
}

/// N:M structured sparsity implementation
pub struct NMSparsity {
    n: usize,
    m: usize,
}

impl NMSparsity {
    /// Create a new N:M sparsity pattern
    ///
    /// # Arguments
    /// * `n` - Number of non-zero elements to keep
    /// * `m` - Window size (n elements out of every m)
    ///
    /// Common patterns:
    /// - 1:2 = 50% sparsity
    /// - 2:4 = 50% sparsity (better for hardware)
    /// - 1:4 = 75% sparsity
    pub fn new(n: usize, m: usize) -> Self {
        assert!(n <= m, "N must be <= M in N:M sparsity");
        Self { n, m }
    }

    /// Apply N:M sparsity to a dense tensor
    pub fn apply(&self, tensor: &Tensor) -> Result<SparseTensor> {
        let data = tensor.to_vec_f32()?;
        let shape = tensor.shape().to_vec();

        if shape.len() != 2 {
            return Err(TrustformersError::shape_error(
                "N:M sparsity currently supports only 2D tensors".to_string(),
            ));
        }

        let rows = shape[0];
        let cols = shape[1];

        // Check that columns are divisible by M
        if cols % self.m != 0 {
            return Err(TrustformersError::shape_error(format!(
                "Number of columns {} must be divisible by M={}",
                cols, self.m
            )));
        }

        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        // Process each row
        for row in 0..rows {
            let row_start = row * cols;

            // Process in windows of M elements
            for window_start in (0..cols).step_by(self.m) {
                let window_end = (window_start + self.m).min(cols);

                // Collect values in this window with their original indices
                let mut window_vals: Vec<(usize, f32)> = (window_start..window_end)
                    .map(|col| {
                        let idx = row_start + col;
                        (col, data[idx])
                    })
                    .collect();

                // Sort by absolute value (descending)
                window_vals.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

                // Keep top N values
                for (col, val) in window_vals.iter().take(self.n) {
                    row_indices.push(row);
                    col_indices.push(*col);
                    values.push(*val);
                }
            }
        }

        SparseTensor::new_coo(shape, row_indices, col_indices, values)
    }

    /// Get theoretical sparsity ratio
    pub fn sparsity_ratio(&self) -> f32 {
        1.0 - (self.n as f32 / self.m as f32)
    }
}

/// Block sparsity implementation
pub struct BlockSparsity {
    block_height: usize,
    block_width: usize,
    keep_ratio: f32,
}

impl BlockSparsity {
    /// Create a new block sparsity pattern
    pub fn new(block_height: usize, block_width: usize, keep_ratio: f32) -> Self {
        Self {
            block_height,
            block_width,
            keep_ratio,
        }
    }

    /// Apply block sparsity to a dense tensor
    pub fn apply(&self, tensor: &Tensor) -> Result<SparseTensor> {
        let data = tensor.to_vec_f32()?;
        let shape = tensor.shape().to_vec();

        if shape.len() != 2 {
            return Err(TrustformersError::shape_error(
                "Block sparsity currently supports only 2D tensors".to_string(),
            ));
        }

        let rows = shape[0];
        let cols = shape[1];

        let num_block_rows = (rows + self.block_height - 1) / self.block_height;
        let num_block_cols = (cols + self.block_width - 1) / self.block_width;

        // Compute importance score for each block
        let mut block_scores = Vec::new();
        for br in 0..num_block_rows {
            for bc in 0..num_block_cols {
                let row_start = br * self.block_height;
                let row_end = (row_start + self.block_height).min(rows);
                let col_start = bc * self.block_width;
                let col_end = (col_start + self.block_width).min(cols);

                // Compute L2 norm of block
                let mut block_norm = 0.0f32;
                for r in row_start..row_end {
                    for c in col_start..col_end {
                        let val = data[r * cols + c];
                        block_norm += val * val;
                    }
                }
                block_norm = block_norm.sqrt();

                block_scores.push(((br, bc), block_norm));
            }
        }

        // Sort blocks by importance
        block_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Keep top blocks
        let num_blocks_to_keep = ((block_scores.len() as f32) * self.keep_ratio) as usize;
        let blocks_to_keep: HashSet<(usize, usize)> = block_scores
            .iter()
            .take(num_blocks_to_keep)
            .map(|&((br, bc), _)| (br, bc))
            .collect();

        // Build sparse tensor from kept blocks
        let mut row_ptr = vec![0];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for br in 0..num_block_rows {
            let row_start = br * self.block_height;
            let row_end = (row_start + self.block_height).min(rows);

            for r in row_start..row_end {
                let mut row_nnz = 0;

                for bc in 0..num_block_cols {
                    if !blocks_to_keep.contains(&(br, bc)) {
                        continue;
                    }

                    let col_start = bc * self.block_width;
                    let col_end = (col_start + self.block_width).min(cols);

                    for c in col_start..col_end {
                        let val = data[r * cols + c];
                        if val != 0.0 {
                            col_indices.push(c);
                            values.push(val);
                            row_nnz += 1;
                        }
                    }
                }

                row_ptr.push(row_ptr.last().unwrap() + row_nnz);
            }
        }

        SparseTensor::new_csr(shape, row_ptr, col_indices, values)
    }
}

/// Sparse matrix - dense matrix multiplication
pub fn sparse_matmul(sparse: &SparseTensor, dense: &Tensor) -> Result<Tensor> {
    let dense_data = dense.to_vec_f32()?;
    let dense_shape = dense.shape();

    if sparse.shape.len() != 2 || dense_shape.len() != 2 {
        return Err(TrustformersError::shape_error(
            "Sparse matmul requires 2D matrices".to_string(),
        ));
    }

    if sparse.shape[1] != dense_shape[0] {
        return Err(TrustformersError::shape_error(format!(
            "Incompatible shapes for matmul: {:?} x {:?}",
            sparse.shape, dense_shape
        )));
    }

    let m = sparse.shape[0];
    let k = sparse.shape[1];
    let n = dense_shape[1];

    let mut result = vec![0.0f32; m * n];

    match sparse.format {
        SparseFormat::CSR => {
            if let SparseIndices::CSR {
                row_ptr,
                col_indices,
            } = &sparse.indices
            {
                // CSR format is optimal for SpMM
                for row in 0..m {
                    let row_start = row_ptr[row];
                    let row_end = row_ptr[row + 1];

                    for j in row_start..row_end {
                        let col = col_indices[j];
                        let sparse_val = sparse.values[j];

                        // Compute dot product contribution
                        for out_col in 0..n {
                            result[row * n + out_col] +=
                                sparse_val * dense_data[col * n + out_col];
                        }
                    }
                }
            } else {
                return Err(TrustformersError::tensor_op_error(
                    "Invalid indices format",
                    "sparse matmul",
                ));
            }
        }
        SparseFormat::COO => {
            if let SparseIndices::COO {
                row_indices,
                col_indices,
            } = &sparse.indices
            {
                for ((&&row, &&col), &val) in row_indices
                    .iter()
                    .zip(col_indices.iter())
                    .zip(sparse.values.iter())
                {
                    for out_col in 0..n {
                        result[row * n + out_col] += val * dense_data[col * n + out_col];
                    }
                }
            } else {
                return Err(TrustformersError::tensor_op_error(
                    "Invalid indices format",
                    "sparse matmul",
                ));
            }
        }
        _ => {
            return Err(TrustformersError::tensor_op_error(
                "Unsupported sparse format for matmul",
                "sparse matmul",
            ));
        }
    }

    Tensor::from_vec(result, &[m, n])
}

/// Sparse attention utilities
pub mod sparse_attention {
    use super::*;

    /// Block-sparse attention pattern
    pub struct BlockSparseAttention {
        block_size: usize,
        num_random_blocks: usize,
    }

    impl BlockSparseAttention {
        /// Create a new block-sparse attention pattern
        pub fn new(block_size: usize, num_random_blocks: usize) -> Self {
            Self {
                block_size,
                num_random_blocks,
            }
        }

        /// Generate attention mask for block-sparse pattern
        pub fn generate_mask(&self, seq_len: usize) -> Result<SparseTensor> {
            let num_blocks = (seq_len + self.block_size - 1) / self.block_size;

            let mut row_indices = Vec::new();
            let mut col_indices = Vec::new();
            let mut values = Vec::new();

            let mut rng = thread_rng();

            for block_i in 0..num_blocks {
                // Local attention (diagonal blocks)
                for block_j in block_i.saturating_sub(1)..=(block_i + 1).min(num_blocks - 1) {
                    self.add_block(
                        block_i,
                        block_j,
                        seq_len,
                        &mut row_indices,
                        &mut col_indices,
                        &mut values,
                    );
                }

                // Random global attention
                for _ in 0..self.num_random_blocks {
                    let random_block = rng.gen_range(0..num_blocks);
                    self.add_block(
                        block_i,
                        random_block,
                        seq_len,
                        &mut row_indices,
                        &mut col_indices,
                        &mut values,
                    );
                }
            }

            SparseTensor::new_coo(vec![seq_len, seq_len], row_indices, col_indices, values)
        }

        fn add_block(
            &self,
            block_i: usize,
            block_j: usize,
            seq_len: usize,
            row_indices: &mut Vec<usize>,
            col_indices: &mut Vec<usize>,
            values: &mut Vec<f32>,
        ) {
            let row_start = block_i * self.block_size;
            let row_end = (row_start + self.block_size).min(seq_len);
            let col_start = block_j * self.block_size;
            let col_end = (col_start + self.block_size).min(seq_len);

            for r in row_start..row_end {
                for c in col_start..col_end {
                    row_indices.push(r);
                    col_indices.push(c);
                    values.push(1.0); // Attention mask value
                }
            }
        }
    }

    /// Sliding window attention pattern
    pub fn sliding_window_mask(seq_len: usize, window_size: usize) -> Result<SparseTensor> {
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..seq_len {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(seq_len);

            for j in start..end {
                row_indices.push(i);
                col_indices.push(j);
                values.push(1.0);
            }
        }

        SparseTensor::new_coo(vec![seq_len, seq_len], row_indices, col_indices, values)
    }

    /// Dilated sliding window (for longer-range dependencies)
    pub fn dilated_window_mask(
        seq_len: usize,
        window_size: usize,
        dilation: usize,
    ) -> Result<SparseTensor> {
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..seq_len {
            // Local window
            let local_start = i.saturating_sub(window_size / 2);
            let local_end = (i + window_size / 2 + 1).min(seq_len);

            for j in local_start..local_end {
                row_indices.push(i);
                col_indices.push(j);
                values.push(1.0);
            }

            // Dilated positions
            for k in 1..=window_size {
                let dilated_pos = i + k * dilation;
                if dilated_pos < seq_len {
                    row_indices.push(i);
                    col_indices.push(dilated_pos);
                    values.push(1.0);
                }

                if k * dilation <= i {
                    let dilated_pos = i - k * dilation;
                    row_indices.push(i);
                    col_indices.push(dilated_pos);
                    values.push(1.0);
                }
            }
        }

        SparseTensor::new_coo(vec![seq_len, seq_len], row_indices, col_indices, values)
    }
}

/// Format conversion utilities
pub mod conversion {
    use super::*;

    /// Convert COO to CSR format
    pub fn coo_to_csr(sparse: &SparseTensor) -> Result<SparseTensor> {
        if sparse.format != SparseFormat::COO {
            return Err(TrustformersError::tensor_op_error(
                "Input must be in COO format",
                "COO to CSR conversion",
            ));
        }

        if let SparseIndices::COO {
            row_indices,
            col_indices,
        } = &sparse.indices
        {
            let num_rows = sparse.shape[0];

            // Build row_ptr
            let mut row_ptr = vec![0; num_rows + 1];
            for &row in row_indices {
                row_ptr[row + 1] += 1;
            }

            // Cumulative sum
            for i in 0..num_rows {
                row_ptr[i + 1] += row_ptr[i];
            }

            // Sort entries by row, then by column
            let mut entries: Vec<(usize, usize, f32)> = row_indices
                .iter()
                .zip(col_indices.iter())
                .zip(sparse.values.iter())
                .map(|((&r, &c), &v)| (r, c, v))
                .collect();

            entries.sort_by_key(|&(r, c, _)| (r, c));

            let sorted_col_indices: Vec<usize> = entries.iter().map(|&(_, c, _)| c).collect();
            let sorted_values: Vec<f32> = entries.iter().map(|&(_, _, v)| v).collect();

            SparseTensor::new_csr(
                sparse.shape.clone(),
                row_ptr,
                sorted_col_indices,
                sorted_values,
            )
        } else {
            Err(TrustformersError::tensor_op_error(
                "Invalid indices format",
                "COO to CSR conversion",
            ))
        }
    }

    /// Convert CSR to COO format
    pub fn csr_to_coo(sparse: &SparseTensor) -> Result<SparseTensor> {
        if sparse.format != SparseFormat::CSR {
            return Err(TrustformersError::tensor_op_error(
                "Input must be in CSR format",
                "CSR to COO conversion",
            ));
        }

        if let SparseIndices::CSR {
            row_ptr,
            col_indices,
        } = &sparse.indices
        {
            let mut row_indices = Vec::new();

            for (row, window) in row_ptr.windows(2).enumerate() {
                let count = window[1] - window[0];
                row_indices.extend(vec![row; count]);
            }

            SparseTensor::new_coo(
                sparse.shape.clone(),
                row_indices,
                col_indices.clone(),
                sparse.values.clone(),
            )
        } else {
            Err(TrustformersError::tensor_op_error(
                "Invalid indices format",
                "CSR to COO conversion",
            ))
        }
    }
}

/// Pruning utilities
pub mod pruning {
    use super::*;

    /// Magnitude-based pruning
    pub fn magnitude_prune(tensor: &Tensor, keep_ratio: f32) -> Result<SparseTensor> {
        let data = tensor.to_vec_f32()?;
        let shape = tensor.shape().to_vec();

        // Sort by magnitude
        let mut indexed_data: Vec<(usize, f32)> =
            data.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed_data.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        // Keep top-k
        let num_keep = ((data.len() as f32) * keep_ratio) as usize;
        let keep_indices: HashSet<usize> = indexed_data
            .iter()
            .take(num_keep)
            .map(|&(idx, _)| idx)
            .collect();

        // Build sparse tensor
        if shape.len() == 2 {
            let cols = shape[1];
            let mut row_indices = Vec::new();
            let mut col_indices = Vec::new();
            let mut values = Vec::new();

            for idx in keep_indices {
                let row = idx / cols;
                let col = idx % cols;
                row_indices.push(row);
                col_indices.push(col);
                values.push(data[idx]);
            }

            SparseTensor::new_coo(shape, row_indices, col_indices, values)
        } else {
            Err(TrustformersError::shape_error(
                "Pruning currently supports only 2D tensors".to_string(),
            ))
        }
    }

    /// Gradient-based pruning (requires gradient information)
    pub fn gradient_based_prune(
        tensor: &Tensor,
        gradients: &Tensor,
        keep_ratio: f32,
    ) -> Result<SparseTensor> {
        let weights = tensor.to_vec_f32()?;
        let grads = gradients.to_vec_f32()?;
        let shape = tensor.shape().to_vec();

        if weights.len() != grads.len() {
            return Err(TrustformersError::shape_error(
                "Weight and gradient shapes must match".to_string(),
            ));
        }

        // Compute importance score: |weight * gradient|
        let mut scores: Vec<(usize, f32)> = weights
            .iter()
            .zip(grads.iter())
            .enumerate()
            .map(|(i, (&w, &g))| (i, (w * g).abs()))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let num_keep = ((weights.len() as f32) * keep_ratio) as usize;
        let keep_indices: HashSet<usize> = scores.iter().take(num_keep).map(|&(idx, _)| idx).collect();

        // Build sparse tensor
        if shape.len() == 2 {
            let cols = shape[1];
            let mut row_indices = Vec::new();
            let mut col_indices = Vec::new();
            let mut values = Vec::new();

            for idx in keep_indices {
                let row = idx / cols;
                let col = idx % cols;
                row_indices.push(row);
                col_indices.push(col);
                values.push(weights[idx]);
            }

            SparseTensor::new_coo(shape, row_indices, col_indices, values)
        } else {
            Err(TrustformersError::shape_error(
                "Pruning currently supports only 2D tensors".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nm_sparsity() -> Result<()> {
        let nm = NMSparsity::new(2, 4);
        assert_eq!(nm.sparsity_ratio(), 0.5);

        // Create a test tensor
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, &[8, 8])?;

        let sparse = nm.apply(&tensor)?;

        // Check that we have 50% sparsity
        let expected_nnz = 8 * 8 / 2; // 50% of 64
        assert_eq!(sparse.nnz, expected_nnz);

        Ok(())
    }

    #[test]
    fn test_sparse_matmul() -> Result<()> {
        // Create a sparse matrix
        let sparse = SparseTensor::new_coo(
            vec![3, 3],
            vec![0, 0, 1, 2],
            vec![0, 1, 1, 2],
            vec![1.0, 2.0, 3.0, 4.0],
        )?;

        // Create a dense matrix
        let dense_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dense = Tensor::from_vec(dense_data, &[3, 2])?;

        // Multiply
        let result = sparse_matmul(&sparse, &dense)?;

        assert_eq!(result.shape(), &[3, 2]);

        Ok(())
    }

    #[test]
    fn test_block_sparsity() -> Result<()> {
        let block_sparse = BlockSparsity::new(2, 2, 0.5);

        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, &[8, 8])?;

        let sparse = block_sparse.apply(&tensor)?;

        // Should keep approximately 50% of blocks
        assert!(sparse.nnz > 0);
        assert!(sparse.nnz < 64);

        Ok(())
    }

    #[test]
    fn test_sliding_window_mask() -> Result<()> {
        let mask = sparse_attention::sliding_window_mask(100, 10)?;

        // Each position should attend to window_size positions
        assert!(mask.nnz <= 100 * 10);

        Ok(())
    }

    #[test]
    fn test_magnitude_pruning() -> Result<()> {
        let data: Vec<f32> = (0..64).map(|i| (i as f32) - 32.0).collect();
        let tensor = Tensor::from_vec(data, &[8, 8])?;

        let sparse = pruning::magnitude_prune(&tensor, 0.25)?;

        // Should keep 25% of elements
        assert_eq!(sparse.nnz, 16);

        Ok(())
    }

    #[test]
    fn test_coo_to_csr_conversion() -> Result<()> {
        let coo = SparseTensor::new_coo(
            vec![3, 3],
            vec![0, 0, 1, 2],
            vec![0, 1, 1, 2],
            vec![1.0, 2.0, 3.0, 4.0],
        )?;

        let csr = conversion::coo_to_csr(&coo)?;

        assert_eq!(csr.format, SparseFormat::CSR);
        assert_eq!(csr.nnz, 4);

        // Convert back
        let coo2 = conversion::csr_to_coo(&csr)?;
        assert_eq!(coo2.format, SparseFormat::COO);
        assert_eq!(coo2.nnz, 4);

        Ok(())
    }
}
