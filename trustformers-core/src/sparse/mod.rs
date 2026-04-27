//! Sparse tensor formats for efficient ML operations.
//!
//! Provides COO (coordinate), CSR (compressed sparse row), and block-sparse
//! matrix representations. Designed for attention masks, sparse gradients,
//! and MoE routing patterns common in transformer models.
//!
//! # Formats
//!
//! - [`SparseCoo`]: Coordinate format; best for construction and incremental updates.
//! - [`SparseCsr`]: Compressed sparse row; best for row-wise traversal and SpMV.
//! - [`BlockSparse`]: Block-structured sparsity; best for attention masks.
//!
//! # Utilities
//!
//! - [`top_k_sparsify`]: Keep top-k values by magnitude.
//! - [`vector_sparsity`]: Fraction of near-zero elements.
//! - [`top_k_mask`]: Boolean mask selecting top-k scoring indices.
//! - [`sparse_add_vecs`]: Merge two sparse vectors by index.

use std::collections::HashMap;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors that can arise from sparse-tensor operations.
#[derive(Debug, thiserror::Error)]
pub enum SparseError {
    /// Shape mismatch between two sparse structures.
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    /// Index lies outside the declared matrix dimensions.
    #[error("Index out of bounds: ({row}, {col}) for shape ({rows}, {cols})")]
    IndexOutOfBounds {
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    },
    /// A duplicate (row, col) pair was encountered where one is forbidden.
    #[error("Duplicate index: ({row}, {col})")]
    DuplicateIndex { row: usize, col: usize },
    /// CSR requires triples to arrive in row-major sorted order.
    #[error("Not sorted: CSR requires sorted row-major order")]
    NotSorted,
    /// Sparsity value outside [0, 1].
    #[error("Invalid sparsity: {0}")]
    InvalidSparsity(f64),
    /// Operation on a completely empty structure.
    #[error("Empty sparse tensor")]
    Empty,
    /// General conversion failure.
    #[error("Conversion error: {0}")]
    ConversionError(String),
}

// ─── COO (Coordinate) Format ──────────────────────────────────────────────────

/// Sparse matrix in COO (coordinate) format.
///
/// Stores (row, col, value) triples for non-zero elements. Duplicate (row, col)
/// pairs are **accumulated** on [`insert`](SparseCoo::insert).
#[derive(Debug, Clone)]
pub struct SparseCoo {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row index of each stored element.
    pub row_indices: Vec<usize>,
    /// Column index of each stored element.
    pub col_indices: Vec<usize>,
    /// Value of each stored element.
    pub values: Vec<f64>,
    /// Number of stored (possibly accumulated) non-zero entries.
    pub nnz: usize,
}

impl SparseCoo {
    /// Create an empty COO matrix with the given shape.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
            nnz: 0,
        }
    }

    /// Insert a value at `(row, col)`.
    ///
    /// If the position already exists its stored value is **increased** by
    /// `value` (accumulate semantics, matching COO assembly conventions).
    /// Returns [`SparseError::IndexOutOfBounds`] when the coordinate lies
    /// outside the declared shape.
    pub fn insert(&mut self, row: usize, col: usize, value: f64) -> Result<(), SparseError> {
        if row >= self.rows || col >= self.cols {
            return Err(SparseError::IndexOutOfBounds {
                row,
                col,
                rows: self.rows,
                cols: self.cols,
            });
        }
        // Accumulate if duplicate
        for i in 0..self.nnz {
            if self.row_indices[i] == row && self.col_indices[i] == col {
                self.values[i] += value;
                return Ok(());
            }
        }
        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(value);
        self.nnz += 1;
        Ok(())
    }

    /// Return the value stored at `(row, col)`, or `0.0` if absent.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        for i in 0..self.nnz {
            if self.row_indices[i] == row && self.col_indices[i] == col {
                return self.values[i];
            }
        }
        0.0
    }

    /// Fraction of elements that are structurally zero: `1 - density`.
    pub fn sparsity(&self) -> f64 {
        let total = self.rows * self.cols;
        if total == 0 {
            return 1.0;
        }
        1.0 - (self.nnz as f64 / total as f64)
    }

    /// Fraction of elements that are structurally non-zero.
    pub fn density(&self) -> f64 {
        let total = self.rows * self.cols;
        if total == 0 {
            return 0.0;
        }
        self.nnz as f64 / total as f64
    }

    /// Expand to a dense `rows × cols` matrix.
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut out = vec![vec![0.0_f64; self.cols]; self.rows];
        for i in 0..self.nnz {
            out[self.row_indices[i]][self.col_indices[i]] += self.values[i];
        }
        out
    }

    /// Build a COO matrix from a dense matrix, dropping elements whose
    /// absolute value is at or below `threshold`.
    pub fn from_dense(dense: &[Vec<f64>], threshold: f64) -> Self {
        let rows = dense.len();
        let cols = if rows == 0 { 0 } else { dense[0].len() };
        let mut result = Self::new(rows, cols);
        for (r, row_data) in dense.iter().enumerate() {
            for (c, &v) in row_data.iter().enumerate() {
                if v.abs() > threshold {
                    // insert cannot fail here because indices come from the shape
                    let _ = result.insert(r, c, v);
                }
            }
        }
        result
    }

    /// Sparse matrix–vector multiply: `y[i] = Σ_j A[i,j] * x[j]`.
    ///
    /// Returns [`SparseError::ShapeMismatch`] when `x.len() != self.cols`.
    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>, SparseError> {
        if x.len() != self.cols {
            return Err(SparseError::ShapeMismatch(format!(
                "vector length {} does not match cols {}",
                x.len(),
                self.cols
            )));
        }
        let mut y = vec![0.0_f64; self.rows];
        for i in 0..self.nnz {
            y[self.row_indices[i]] += self.values[i] * x[self.col_indices[i]];
        }
        Ok(y)
    }

    /// Element-wise addition with another COO matrix of the same shape.
    pub fn add(&self, other: &SparseCoo) -> Result<SparseCoo, SparseError> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(SparseError::ShapeMismatch(format!(
                "({}, {}) vs ({}, {})",
                self.rows, self.cols, other.rows, other.cols
            )));
        }
        let mut result = SparseCoo::new(self.rows, self.cols);
        for i in 0..self.nnz {
            result
                .insert(self.row_indices[i], self.col_indices[i], self.values[i])
                .map_err(|e| SparseError::ConversionError(e.to_string()))?;
        }
        for i in 0..other.nnz {
            result
                .insert(other.row_indices[i], other.col_indices[i], other.values[i])
                .map_err(|e| SparseError::ConversionError(e.to_string()))?;
        }
        Ok(result)
    }

    /// Return the transpose `A^T` as a new COO matrix.
    pub fn transpose(&self) -> SparseCoo {
        SparseCoo {
            rows: self.cols,
            cols: self.rows,
            row_indices: self.col_indices.clone(),
            col_indices: self.row_indices.clone(),
            values: self.values.clone(),
            nnz: self.nnz,
        }
    }

    /// Convert to CSR format. Duplicate (row, col) entries are summed.
    pub fn to_csr(&self) -> SparseCsr {
        // Accumulate into a sorted list of (row, col, value) triples
        let mut map: HashMap<(usize, usize), f64> = HashMap::new();
        for i in 0..self.nnz {
            *map.entry((self.row_indices[i], self.col_indices[i]))
                .or_insert(0.0) += self.values[i];
        }
        let mut triples: Vec<(usize, usize, f64)> =
            map.into_iter().map(|((r, c), v)| (r, c, v)).collect();
        triples.sort_by_key(|&(r, c, _)| (r, c));

        // Build row_ptr
        let mut row_ptr = vec![0usize; self.rows + 1];
        for &(r, _, _) in &triples {
            row_ptr[r + 1] += 1;
        }
        for i in 0..self.rows {
            row_ptr[i + 1] += row_ptr[i];
        }
        let col_indices: Vec<usize> = triples.iter().map(|&(_, c, _)| c).collect();
        let values: Vec<f64> = triples.iter().map(|&(_, _, v)| v).collect();
        let nnz = triples.len();

        SparseCsr {
            rows: self.rows,
            cols: self.cols,
            row_ptr,
            col_indices,
            values,
            nnz,
        }
    }
}

// ─── CSR (Compressed Sparse Row) Format ──────────────────────────────────────

/// Sparse matrix in CSR format.
///
/// `row_ptr[i]..row_ptr[i+1]` indexes the column-indices and values arrays
/// for row `i`. Efficient for row-wise operations and SpMV.
#[derive(Debug, Clone)]
pub struct SparseCsr {
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
    /// Row pointer array of length `rows + 1`.
    pub row_ptr: Vec<usize>,
    /// Column index of each stored element.
    pub col_indices: Vec<usize>,
    /// Value of each stored element.
    pub values: Vec<f64>,
    /// Total number of stored non-zero elements.
    pub nnz: usize,
}

impl SparseCsr {
    /// Build a CSR matrix from a list of `(row, col, value)` triples.
    ///
    /// The triples **must** arrive in strictly ascending `(row, col)` order.
    /// Returns [`SparseError::NotSorted`] for out-of-order input and
    /// [`SparseError::DuplicateIndex`] for repeated coordinates.
    pub fn from_triples(
        rows: usize,
        cols: usize,
        triples: Vec<(usize, usize, f64)>,
    ) -> Result<Self, SparseError> {
        // Validate sorted order and absence of duplicates
        for w in triples.windows(2) {
            let (r0, c0, _) = w[0];
            let (r1, c1, _) = w[1];
            if (r0, c0) >= (r1, c1) {
                if (r0, c0) == (r1, c1) {
                    return Err(SparseError::DuplicateIndex { row: r0, col: c0 });
                }
                return Err(SparseError::NotSorted);
            }
        }
        // Validate bounds
        for &(r, c, _) in &triples {
            if r >= rows || c >= cols {
                return Err(SparseError::IndexOutOfBounds {
                    row: r,
                    col: c,
                    rows,
                    cols,
                });
            }
        }
        let nnz = triples.len();
        let mut row_ptr = vec![0usize; rows + 1];
        for &(r, _, _) in &triples {
            row_ptr[r + 1] += 1;
        }
        for i in 0..rows {
            row_ptr[i + 1] += row_ptr[i];
        }
        let col_indices = triples.iter().map(|&(_, c, _)| c).collect();
        let values = triples.iter().map(|&(_, _, v)| v).collect();
        Ok(Self {
            rows,
            cols,
            row_ptr,
            col_indices,
            values,
            nnz,
        })
    }

    /// Return the value at `(row, col)` using binary search within the row.
    /// Returns `0.0` if the element is not stored.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row >= self.rows || col >= self.cols {
            return 0.0;
        }
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        let slice = &self.col_indices[start..end];
        match slice.binary_search(&col) {
            Ok(pos) => self.values[start + pos],
            Err(_) => 0.0,
        }
    }

    /// Return the column-index and value slices for a single row.
    ///
    /// Returns empty slices for an out-of-bounds row index rather than
    /// panicking.
    pub fn row(&self, row_idx: usize) -> (&[usize], &[f64]) {
        if row_idx >= self.rows {
            return (&[], &[]);
        }
        let start = self.row_ptr[row_idx];
        let end = self.row_ptr[row_idx + 1];
        (&self.col_indices[start..end], &self.values[start..end])
    }

    /// Sparse matrix–vector multiply: `y[i] = Σ_j A[i,j] * x[j]`.
    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>, SparseError> {
        if x.len() != self.cols {
            return Err(SparseError::ShapeMismatch(format!(
                "vector length {} does not match cols {}",
                x.len(),
                self.cols
            )));
        }
        let mut y = vec![0.0_f64; self.rows];
        for (r, y_r) in y.iter_mut().enumerate() {
            let start = self.row_ptr[r];
            let end = self.row_ptr[r + 1];
            for idx in start..end {
                *y_r += self.values[idx] * x[self.col_indices[idx]];
            }
        }
        Ok(y)
    }

    /// Sparse matrix × dense matrix multiply.
    ///
    /// `dense` is treated as a `(self.cols × k)` matrix stored as a
    /// `Vec<Vec<f64>>` of length `self.cols`. Returns a `(self.rows × k)`
    /// dense result.
    pub fn spmm(&self, dense: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, SparseError> {
        if dense.len() != self.cols {
            return Err(SparseError::ShapeMismatch(format!(
                "dense matrix rows {} does not match sparse cols {}",
                dense.len(),
                self.cols
            )));
        }
        let k = if dense.is_empty() { 0 } else { dense[0].len() };
        let mut out = vec![vec![0.0_f64; k]; self.rows];
        for (r, out_r) in out.iter_mut().enumerate() {
            let start = self.row_ptr[r];
            let end = self.row_ptr[r + 1];
            for idx in start..end {
                let c = self.col_indices[idx];
                let a_val = self.values[idx];
                for j in 0..k {
                    out_r[j] += a_val * dense[c][j];
                }
            }
        }
        Ok(out)
    }

    /// Fraction of elements that are structurally zero.
    pub fn sparsity(&self) -> f64 {
        let total = self.rows * self.cols;
        if total == 0 {
            return 1.0;
        }
        1.0 - (self.nnz as f64 / total as f64)
    }

    /// Fraction of elements that are structurally non-zero.
    pub fn density(&self) -> f64 {
        let total = self.rows * self.cols;
        if total == 0 {
            return 0.0;
        }
        self.nnz as f64 / total as f64
    }

    /// Expand to a dense `rows × cols` matrix.
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut out = vec![vec![0.0_f64; self.cols]; self.rows];
        for (r, out_r) in out.iter_mut().enumerate() {
            let start = self.row_ptr[r];
            let end = self.row_ptr[r + 1];
            for idx in start..end {
                out_r[self.col_indices[idx]] = self.values[idx];
            }
        }
        out
    }

    /// Convert to COO format.
    pub fn to_coo(&self) -> SparseCoo {
        let mut row_indices = Vec::with_capacity(self.nnz);
        let mut col_indices = Vec::with_capacity(self.nnz);
        for r in 0..self.rows {
            let start = self.row_ptr[r];
            let end = self.row_ptr[r + 1];
            for idx in start..end {
                row_indices.push(r);
                col_indices.push(self.col_indices[idx]);
            }
        }
        SparseCoo {
            rows: self.rows,
            cols: self.cols,
            row_indices,
            col_indices,
            values: self.values.clone(),
            nnz: self.nnz,
        }
    }

    /// Transpose: convert to COO, swap indices, convert back to CSR.
    pub fn transpose(&self) -> SparseCsr {
        self.to_coo().transpose().to_csr()
    }
}

// ─── Block-Sparse Format ──────────────────────────────────────────────────────

/// Block-sparse matrix using uniform square blocks.
///
/// The matrix is partitioned into `block_rows × block_cols` blocks, each of
/// size `block_size × block_size`. Only blocks that are explicitly inserted
/// are stored. This is efficient for attention masks where large contiguous
/// regions are either fully present or fully absent.
#[derive(Debug, Clone)]
pub struct BlockSparse {
    /// Total number of rows.
    pub rows: usize,
    /// Total number of columns.
    pub cols: usize,
    /// Side length of each square block.
    pub block_size: usize,
    /// Number of block rows (`rows / block_size`).
    pub block_rows: usize,
    /// Number of block columns (`cols / block_size`).
    pub block_cols: usize,
    /// Stored blocks keyed by `(block_row, block_col)`.
    /// Each value is a flat row-major array of length `block_size²`.
    pub blocks: HashMap<(usize, usize), Vec<f64>>,
}

impl BlockSparse {
    /// Create an empty block-sparse matrix.
    ///
    /// Returns [`SparseError::ShapeMismatch`] if `rows` or `cols` is not
    /// divisible by `block_size`.
    pub fn new(rows: usize, cols: usize, block_size: usize) -> Result<Self, SparseError> {
        if block_size == 0 || rows % block_size != 0 || cols % block_size != 0 {
            return Err(SparseError::ShapeMismatch(format!(
                "rows ({}) and cols ({}) must both be divisible by block_size ({})",
                rows, cols, block_size
            )));
        }
        Ok(Self {
            rows,
            cols,
            block_size,
            block_rows: rows / block_size,
            block_cols: cols / block_size,
            blocks: HashMap::new(),
        })
    }

    /// Insert a block at grid position `(block_row, block_col)`.
    ///
    /// `block_data` must be a flat row-major array of length `block_size²`.
    pub fn insert_block(
        &mut self,
        block_row: usize,
        block_col: usize,
        block_data: Vec<f64>,
    ) -> Result<(), SparseError> {
        if block_row >= self.block_rows || block_col >= self.block_cols {
            return Err(SparseError::IndexOutOfBounds {
                row: block_row,
                col: block_col,
                rows: self.block_rows,
                cols: self.block_cols,
            });
        }
        let expected = self.block_size * self.block_size;
        if block_data.len() != expected {
            return Err(SparseError::ShapeMismatch(format!(
                "block_data length {} != expected {}",
                block_data.len(),
                expected
            )));
        }
        self.blocks.insert((block_row, block_col), block_data);
        Ok(())
    }

    /// Return the element at absolute position `(row, col)`.
    /// Returns `0.0` if the containing block is not stored.
    pub fn get(&self, row: usize, col: usize) -> f64 {
        let br = row / self.block_size;
        let bc = col / self.block_size;
        let local_r = row % self.block_size;
        let local_c = col % self.block_size;
        match self.blocks.get(&(br, bc)) {
            Some(data) => data[local_r * self.block_size + local_c],
            None => 0.0,
        }
    }

    /// Return a reference to the stored block at `(block_row, block_col)`,
    /// or `None` if the block is absent.
    pub fn get_block(&self, block_row: usize, block_col: usize) -> Option<&Vec<f64>> {
        self.blocks.get(&(block_row, block_col))
    }

    /// Number of blocks currently stored.
    pub fn num_stored_blocks(&self) -> usize {
        self.blocks.len()
    }

    /// Total number of structurally non-zero scalar elements.
    pub fn nnz(&self) -> usize {
        self.blocks.len() * self.block_size * self.block_size
    }

    /// Fraction of scalar elements that are structurally zero.
    pub fn sparsity(&self) -> f64 {
        let total = self.rows * self.cols;
        if total == 0 {
            return 1.0;
        }
        1.0 - (self.nnz() as f64 / total as f64)
    }

    /// Expand to a dense `rows × cols` matrix.
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let mut out = vec![vec![0.0_f64; self.cols]; self.rows];
        for (&(br, bc), data) in &self.blocks {
            for lr in 0..self.block_size {
                for lc in 0..self.block_size {
                    let r = br * self.block_size + lr;
                    let c = bc * self.block_size + lc;
                    out[r][c] = data[lr * self.block_size + lc];
                }
            }
        }
        out
    }

    /// Build a causal (lower-triangular) attention mask.
    ///
    /// Block `(i, j)` is stored (filled with `1.0`) iff `j <= i`.
    pub fn causal_mask(seq_len: usize, block_size: usize) -> Result<Self, SparseError> {
        let mut mask = Self::new(seq_len, seq_len, block_size)?;
        let block_rows = seq_len / block_size;
        let full_block: Vec<f64> = vec![1.0; block_size * block_size];
        let lower_tri_block = |br: usize, bc: usize| -> Vec<f64> {
            let mut data = vec![0.0_f64; block_size * block_size];
            for lr in 0..block_size {
                for lc in 0..block_size {
                    let abs_r = br * block_size + lr;
                    let abs_c = bc * block_size + lc;
                    if abs_c <= abs_r {
                        data[lr * block_size + lc] = 1.0;
                    }
                }
            }
            data
        };
        for br in 0..block_rows {
            for bc in 0..=br {
                let data = if bc < br {
                    full_block.clone()
                } else {
                    // br == bc: diagonal block, partially masked
                    lower_tri_block(br, bc)
                };
                mask.insert_block(br, bc, data)?;
            }
        }
        Ok(mask)
    }

    /// Build a sliding-window attention mask.
    ///
    /// Each query at position `r` can attend to positions
    /// `r.saturating_sub(window_size)..=r`. Blocks that contain at least one
    /// valid `(r, c)` pair are stored.
    pub fn sliding_window_mask(
        seq_len: usize,
        block_size: usize,
        window_size: usize,
    ) -> Result<Self, SparseError> {
        let mut mask = Self::new(seq_len, seq_len, block_size)?;
        let block_count = seq_len / block_size;
        for br in 0..block_count {
            for bc in 0..block_count {
                let mut data = vec![0.0_f64; block_size * block_size];
                let mut has_entry = false;
                for lr in 0..block_size {
                    for lc in 0..block_size {
                        let abs_r = br * block_size + lr;
                        let abs_c = bc * block_size + lc;
                        if abs_c <= abs_r && abs_r - abs_c <= window_size {
                            data[lr * block_size + lc] = 1.0;
                            has_entry = true;
                        }
                    }
                }
                if has_entry {
                    mask.insert_block(br, bc, data)?;
                }
            }
        }
        Ok(mask)
    }

    /// Block-sparse matrix–vector multiply.
    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>, SparseError> {
        if x.len() != self.cols {
            return Err(SparseError::ShapeMismatch(format!(
                "vector length {} does not match cols {}",
                x.len(),
                self.cols
            )));
        }
        let mut y = vec![0.0_f64; self.rows];
        for (&(br, bc), data) in &self.blocks {
            for lr in 0..self.block_size {
                let abs_r = br * self.block_size + lr;
                for lc in 0..self.block_size {
                    let abs_c = bc * self.block_size + lc;
                    y[abs_r] += data[lr * self.block_size + lc] * x[abs_c];
                }
            }
        }
        Ok(y)
    }
}

// ─── Utilities ────────────────────────────────────────────────────────────────

/// Keep the top-`k` elements by absolute value, zeroing out the rest.
///
/// If `k >= values.len()` the input is returned unchanged.
pub fn top_k_sparsify(values: &[f64], k: usize) -> Vec<f64> {
    if k >= values.len() {
        return values.to_vec();
    }
    if k == 0 {
        return vec![0.0; values.len()];
    }
    // Collect (abs_value, original_index) and partial-sort
    let mut indexed: Vec<(f64, usize)> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| (v.abs(), i))
        .collect();
    // Find the k-th largest threshold via a simple O(n log n) sort
    indexed.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = indexed[k - 1].0;
    // Build output: keep top-k (ties broken by keeping the first k)
    let mut out = vec![0.0_f64; values.len()];
    let mut kept = 0usize;
    for &(abs_v, idx) in &indexed {
        if kept >= k {
            break;
        }
        if abs_v >= threshold {
            out[idx] = values[idx];
            kept += 1;
        }
    }
    out
}

/// Fraction of elements in `values` whose absolute value is at or below
/// `threshold`.
pub fn vector_sparsity(values: &[f64], threshold: f64) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    let zeros = values.iter().filter(|&&v| v.abs() <= threshold).count();
    zeros as f64 / values.len() as f64
}

/// Return a boolean mask that is `true` for exactly the top-`k` indices by
/// score (ties broken in favor of lower indices when scores are equal).
pub fn top_k_mask(scores: &[f64], k: usize) -> Vec<bool> {
    if scores.is_empty() {
        return Vec::new();
    }
    let effective_k = k.min(scores.len());
    if effective_k == 0 {
        return vec![false; scores.len()];
    }
    // Sort indices by descending score, stable by index for ties
    let mut indexed: Vec<(usize, f64)> = scores
        .iter()
        .enumerate()
        .map(|(i, &s)| (i, s))
        .collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    let mut mask = vec![false; scores.len()];
    for &(idx, _) in indexed.iter().take(effective_k) {
        mask[idx] = true;
    }
    mask
}

/// Merge two sparse vectors by adding values at shared indices.
///
/// `a_indices` / `a_values` and `b_indices` / `b_values` are parallel arrays
/// of (index, value) pairs. `size` is the declared length of the dense vector.
/// Returns a new pair of (sorted indices, values) with no duplicates.
pub fn sparse_add_vecs(
    a_indices: &[usize],
    a_values: &[f64],
    b_indices: &[usize],
    b_values: &[f64],
    size: usize,
) -> (Vec<usize>, Vec<f64>) {
    let mut map: HashMap<usize, f64> = HashMap::new();
    for (&idx, &val) in a_indices.iter().zip(a_values.iter()) {
        if idx < size {
            *map.entry(idx).or_insert(0.0) += val;
        }
    }
    for (&idx, &val) in b_indices.iter().zip(b_values.iter()) {
        if idx < size {
            *map.entry(idx).or_insert(0.0) += val;
        }
    }
    let mut pairs: Vec<(usize, f64)> = map.into_iter().collect();
    pairs.sort_by_key(|&(idx, _)| idx);
    let indices = pairs.iter().map(|&(i, _)| i).collect();
    let values = pairs.iter().map(|&(_, v)| v).collect();
    (indices, values)
}

// ─── BigBird Sparse Attention ─────────────────────────────────────────────────

/// Configuration for BigBird sparse attention patterns.
#[derive(Debug, Clone)]
pub struct BigBirdConfig {
    /// Number of attention heads (informational only; mask is head-agnostic).
    pub num_attention_heads: usize,
    /// Block size in tokens (e.g. 64).
    pub block_size: usize,
    /// Number of random blocks attended by each query block.
    pub num_random_blocks: usize,
    /// Seed for deterministic random block generation.
    pub seed: u64,
}

/// BigBird attention mask: combines global tokens, sliding window, and random blocks.
#[derive(Debug, Clone)]
pub struct BigBirdAttentionMask {
    /// Indices of tokens that are always attended to (global tokens = first block).
    pub global_tokens: Vec<usize>,
    /// Half-width of sliding window in tokens (≈ one block on each side).
    pub window_size: usize,
    /// (query_block_idx, key_block_idx) pairs selected randomly.
    pub random_block_pairs: Vec<(usize, usize)>,
    /// Total sequence length.
    pub seq_len: usize,
    /// Block size.
    pub block_size: usize,
}

impl BigBirdAttentionMask {
    /// Construct a BigBird attention mask.
    ///
    /// Global tokens are the first `block_size` token indices.
    /// Random blocks are generated using a Lehmer-style LCG seeded from `config.seed`.
    pub fn new(seq_len: usize, config: &BigBirdConfig) -> Self {
        let block_size = config.block_size.max(1);
        let num_blocks = (seq_len + block_size - 1) / block_size;

        // Global tokens: first block
        let global_count = block_size.min(seq_len);
        let global_tokens: Vec<usize> = (0..global_count).collect();

        // Sliding window: one block on each side
        let window_size = block_size;

        // Random blocks: deterministic LCG per query block
        let mut random_block_pairs: Vec<(usize, usize)> = Vec::new();
        if num_blocks > 0 && config.num_random_blocks > 0 {
            let mut state: u64 = config.seed.wrapping_add(1);
            for q_blk in 0..num_blocks {
                let mut added = 0usize;
                let mut attempts = 0usize;
                while added < config.num_random_blocks && attempts < num_blocks * 4 {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    let k_blk = (state >> 33) as usize % num_blocks;
                    random_block_pairs.push((q_blk, k_blk));
                    added += 1;
                    attempts += 1;
                }
            }
        }

        Self {
            global_tokens,
            window_size,
            random_block_pairs,
            seq_len,
            block_size,
        }
    }

    /// Returns `true` if position `q_pos` should attend to position `k_pos`.
    ///
    /// Attend when any of the following holds:
    /// 1. `k_pos` is a global token.
    /// 2. `q_pos` is a global token.
    /// 3. `|q_pos - k_pos| <= window_size` (sliding window).
    /// 4. `(q_pos/block_size, k_pos/block_size)` is in `random_block_pairs`.
    pub fn should_attend(&self, q_pos: usize, k_pos: usize) -> bool {
        // 1. k is global
        if self.global_tokens.contains(&k_pos) {
            return true;
        }
        // 2. q is global
        if self.global_tokens.contains(&q_pos) {
            return true;
        }
        // 3. Sliding window
        let diff = if q_pos >= k_pos {
            q_pos - k_pos
        } else {
            k_pos - q_pos
        };
        if diff <= self.window_size {
            return true;
        }
        // 4. Random block pair
        let q_blk = q_pos / self.block_size;
        let k_blk = k_pos / self.block_size;
        self.random_block_pairs.contains(&(q_blk, k_blk))
    }

    /// Build a dense `seq_len × seq_len` attention mask matrix.
    ///
    /// Attended positions have value `0.0`; masked positions have value `-1e9`.
    pub fn to_dense_mask(&self) -> Vec<Vec<f32>> {
        let n = self.seq_len;
        let mut mask = vec![vec![-1e9_f32; n]; n];
        for q in 0..n {
            for k in 0..n {
                if self.should_attend(q, k) {
                    mask[q][k] = 0.0;
                }
            }
        }
        mask
    }

    /// Fraction of position pairs that attend to each other.
    pub fn sparsity_ratio(&self) -> f32 {
        let n = self.seq_len;
        if n == 0 {
            return 0.0;
        }
        let total = n * n;
        let mut attended = 0usize;
        for q in 0..n {
            for k in 0..n {
                if self.should_attend(q, k) {
                    attended += 1;
                }
            }
        }
        attended as f32 / total as f32
    }
}

// ─── Longformer Sliding Window + Global Attention ─────────────────────────────

/// Configuration for Longformer-style attention masks.
#[derive(Debug, Clone)]
pub struct LongformerConfig {
    /// One-sided sliding window size in tokens (e.g. 256).
    pub window_size: usize,
    /// Indices of globally-attending tokens (e.g. `[0]` for `[CLS]`).
    pub global_token_indices: Vec<usize>,
}

/// Build a Longformer attention mask matrix of shape `seq_len × seq_len`.
///
/// Value `0.0` means the pair attends; value `-1e9` means the pair is masked.
/// A position pair `(q, k)` attends when:
/// - `|q - k| <= window_size`, OR
/// - `q` is in `global_token_indices`, OR
/// - `k` is in `global_token_indices`.
pub fn longformer_attention_mask(seq_len: usize, config: &LongformerConfig) -> Vec<Vec<f32>> {
    let mut mask = vec![vec![-1e9_f32; seq_len]; seq_len];
    for q in 0..seq_len {
        for k in 0..seq_len {
            let diff = if q >= k { q - k } else { k - q };
            let within_window = diff <= config.window_size;
            let q_global = config.global_token_indices.contains(&q);
            let k_global = config.global_token_indices.contains(&k);
            if within_window || q_global || k_global {
                mask[q][k] = 0.0;
            }
        }
    }
    mask
}

/// Combine local and global attention outputs.
///
/// For each position `i`, if `i` is a global token (in
/// `config.global_token_indices`) the value from `global_attn` is used;
/// otherwise the value from `local_attn` is used.
///
/// Both slices must have the same length.  If lengths differ the shorter one
/// dictates the output length.
pub fn longformer_combine_local_global(
    local_attn: &[f32],
    global_attn: &[f32],
    config: &LongformerConfig,
) -> Vec<f32> {
    let len = local_attn.len().min(global_attn.len());
    (0..len)
        .map(|i| {
            if config.global_token_indices.contains(&i) {
                global_attn[i]
            } else {
                local_attn[i]
            }
        })
        .collect()
}

// ─── Block Sparse Attention Pattern ──────────────────────────────────────────

/// Configuration for a block-sparse attention mask.
#[derive(Debug, Clone)]
pub struct BlockSparseAttnConfig {
    /// Size of each block in tokens.
    pub block_size: usize,
    /// Number of attention heads (informational; mask is head-agnostic).
    pub num_heads: usize,
    /// Fraction of block pairs to *skip* (0.0 = dense, 1.0 = fully sparse).
    pub sparsity: f32,
    /// Seed for deterministic block selection.
    pub seed: u64,
}

/// Build a block-sparse attention mask.
///
/// Returns a flat `Vec<bool>` of length `n_blocks * n_blocks` where
/// `n_blocks = ceil(seq_len / block_size)`.  `true` means the block pair
/// attends; `false` means it is skipped.
///
/// A block pair `(q_blk, k_blk)` attends when a pseudo-random value in
/// `[0, 1)` drawn from a LCG is `>= config.sparsity`.
pub fn block_sparse_attn_mask(seq_len: usize, config: &BlockSparseAttnConfig) -> Vec<bool> {
    let block_size = config.block_size.max(1);
    let n_blocks = (seq_len + block_size - 1) / block_size;
    let total = n_blocks * n_blocks;
    if total == 0 {
        return Vec::new();
    }

    let mut mask = Vec::with_capacity(total);
    let mut state: u64 = config.seed.wrapping_add(1);

    for _ in 0..total {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let val = (state >> 33) as f32 / (u32::MAX as f32);
        mask.push(val >= config.sparsity);
    }
    mask
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SparseCoo ────────────────────────────────────────────────────────────

    #[test]
    fn coo_insert_and_get() {
        let mut m = SparseCoo::new(3, 3);
        m.insert(0, 1, 2.5).unwrap();
        m.insert(2, 2, -1.0).unwrap();
        assert_eq!(m.get(0, 1), 2.5);
        assert_eq!(m.get(2, 2), -1.0);
        assert_eq!(m.get(0, 0), 0.0);
        assert_eq!(m.nnz, 2);
    }

    #[test]
    fn coo_insert_accumulates_duplicates() {
        let mut m = SparseCoo::new(4, 4);
        m.insert(1, 2, 3.0).unwrap();
        m.insert(1, 2, 7.0).unwrap(); // duplicate → accumulate
        assert_eq!(m.get(1, 2), 10.0);
        assert_eq!(m.nnz, 1, "duplicate should not add a new entry");
    }

    #[test]
    fn coo_insert_out_of_bounds_returns_error() {
        let mut m = SparseCoo::new(2, 2);
        assert!(m.insert(5, 0, 1.0).is_err());
        assert!(m.insert(0, 5, 1.0).is_err());
    }

    #[test]
    fn coo_sparsity_and_density() {
        let mut m = SparseCoo::new(4, 4); // 16 total
        m.insert(0, 0, 1.0).unwrap();
        m.insert(1, 1, 1.0).unwrap(); // 2 nnz
        let expected_density = 2.0 / 16.0;
        let expected_sparsity = 1.0 - expected_density;
        assert!((m.density() - expected_density).abs() < 1e-12);
        assert!((m.sparsity() - expected_sparsity).abs() < 1e-12);
    }

    #[test]
    fn coo_to_dense() {
        let mut m = SparseCoo::new(2, 3);
        m.insert(0, 2, 5.0).unwrap();
        m.insert(1, 0, -3.0).unwrap();
        let d = m.to_dense();
        assert_eq!(d[0], vec![0.0, 0.0, 5.0]);
        assert_eq!(d[1], vec![-3.0, 0.0, 0.0]);
    }

    #[test]
    fn coo_from_dense_with_threshold() {
        let dense = vec![
            vec![0.001, 2.0, 0.0],
            vec![-1.5, 0.0, 0.0005],
        ];
        let coo = SparseCoo::from_dense(&dense, 0.01);
        // Only |x| > 0.01 should be kept: 2.0 and -1.5
        assert_eq!(coo.nnz, 2);
        assert_eq!(coo.get(0, 1), 2.0);
        assert_eq!(coo.get(1, 0), -1.5);
        assert_eq!(coo.get(0, 0), 0.0);
    }

    #[test]
    fn coo_matvec_correctness() {
        // Identity-ish: [[1,0],[0,2]]
        let mut m = SparseCoo::new(2, 2);
        m.insert(0, 0, 1.0).unwrap();
        m.insert(1, 1, 2.0).unwrap();
        let y = m.matvec(&[3.0, 4.0]).unwrap();
        assert!((y[0] - 3.0).abs() < 1e-12);
        assert!((y[1] - 8.0).abs() < 1e-12);
    }

    #[test]
    fn coo_add() {
        let mut a = SparseCoo::new(2, 2);
        a.insert(0, 0, 1.0).unwrap();
        a.insert(0, 1, 2.0).unwrap();
        let mut b = SparseCoo::new(2, 2);
        b.insert(0, 1, 3.0).unwrap();
        b.insert(1, 0, 4.0).unwrap();
        let c = a.add(&b).unwrap();
        assert_eq!(c.get(0, 0), 1.0);
        assert_eq!(c.get(0, 1), 5.0); // 2 + 3
        assert_eq!(c.get(1, 0), 4.0);
    }

    #[test]
    fn coo_transpose() {
        let mut m = SparseCoo::new(2, 3);
        m.insert(0, 2, 7.0).unwrap();
        m.insert(1, 0, -2.0).unwrap();
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(2, 0), 7.0);
        assert_eq!(t.get(0, 1), -2.0);
    }

    #[test]
    fn coo_to_csr_roundtrip() {
        let mut coo = SparseCoo::new(3, 3);
        coo.insert(0, 0, 1.0).unwrap();
        coo.insert(0, 2, 3.0).unwrap();
        coo.insert(1, 1, 5.0).unwrap();
        coo.insert(2, 0, 7.0).unwrap();
        let csr = coo.to_csr();
        assert_eq!(csr.get(0, 0), 1.0);
        assert_eq!(csr.get(0, 2), 3.0);
        assert_eq!(csr.get(1, 1), 5.0);
        assert_eq!(csr.get(2, 0), 7.0);
        assert_eq!(csr.get(0, 1), 0.0);
        assert_eq!(csr.nnz, 4);
    }

    // ── SparseCsr ────────────────────────────────────────────────────────────

    #[test]
    fn csr_from_triples_unsorted_errors() {
        let triples = vec![(1, 0, 1.0), (0, 0, 2.0)]; // out of order
        assert!(SparseCsr::from_triples(2, 2, triples).is_err());
    }

    #[test]
    fn csr_from_triples_duplicate_errors() {
        let triples = vec![(0, 0, 1.0), (0, 0, 2.0)]; // duplicate
        assert!(SparseCsr::from_triples(2, 2, triples).is_err());
    }

    #[test]
    fn csr_get_and_row_slice() {
        let triples = vec![(0, 1, 2.0), (1, 0, 3.0), (1, 2, 4.0), (2, 2, 5.0)];
        let csr = SparseCsr::from_triples(3, 3, triples).unwrap();
        assert_eq!(csr.get(0, 1), 2.0);
        assert_eq!(csr.get(0, 0), 0.0);
        let (cols, vals) = csr.row(1);
        assert_eq!(cols, &[0, 2]);
        assert!((vals[0] - 3.0).abs() < 1e-12);
        assert!((vals[1] - 4.0).abs() < 1e-12);
    }

    #[test]
    fn csr_matvec() {
        // [[2, 0, 0], [0, 3, 0], [0, 0, 4]]
        let triples = vec![(0, 0, 2.0), (1, 1, 3.0), (2, 2, 4.0)];
        let csr = SparseCsr::from_triples(3, 3, triples).unwrap();
        let y = csr.matvec(&[1.0, 2.0, 3.0]).unwrap();
        assert!((y[0] - 2.0).abs() < 1e-12);
        assert!((y[1] - 6.0).abs() < 1e-12);
        assert!((y[2] - 12.0).abs() < 1e-12);
    }

    #[test]
    fn csr_spmm() {
        // A = [[1, 0], [0, 2]]  (2×2 sparse)
        // B = [[3, 4], [5, 6]]  (2×2 dense, stored as rows)
        let triples = vec![(0, 0, 1.0), (1, 1, 2.0)];
        let csr = SparseCsr::from_triples(2, 2, triples).unwrap();
        let dense = vec![vec![3.0, 4.0], vec![5.0, 6.0]];
        let out = csr.spmm(&dense).unwrap();
        // Row 0: 1*row0 = [3, 4]
        assert!((out[0][0] - 3.0).abs() < 1e-12);
        assert!((out[0][1] - 4.0).abs() < 1e-12);
        // Row 1: 2*row1 = [10, 12]
        assert!((out[1][0] - 10.0).abs() < 1e-12);
        assert!((out[1][1] - 12.0).abs() < 1e-12);
    }

    #[test]
    fn csr_transpose() {
        // A = [[1, 0, 2], [0, 3, 0]]  →  A^T = [[1, 0], [0, 3], [2, 0]]
        let triples = vec![(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0)];
        let csr = SparseCsr::from_triples(2, 3, triples).unwrap();
        let t = csr.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        assert_eq!(t.get(0, 0), 1.0);
        assert_eq!(t.get(2, 0), 2.0);
        assert_eq!(t.get(1, 1), 3.0);
        assert_eq!(t.get(0, 1), 0.0);
    }

    #[test]
    fn csr_to_dense_and_sparsity() {
        let triples = vec![(0, 0, 9.0)];
        let csr = SparseCsr::from_triples(2, 2, triples).unwrap();
        let d = csr.to_dense();
        assert_eq!(d[0][0], 9.0);
        assert_eq!(d[0][1], 0.0);
        assert_eq!(d[1][0], 0.0);
        // 1/4 filled
        assert!((csr.density() - 0.25).abs() < 1e-12);
        assert!((csr.sparsity() - 0.75).abs() < 1e-12);
    }

    // ── BlockSparse ──────────────────────────────────────────────────────────

    #[test]
    fn block_sparse_invalid_size_errors() {
        assert!(BlockSparse::new(5, 4, 2).is_err()); // 5 % 2 != 0
        assert!(BlockSparse::new(4, 6, 4).is_err()); // 6 % 4 != 0
        assert!(BlockSparse::new(4, 4, 0).is_err()); // block_size == 0
    }

    #[test]
    fn block_sparse_insert_and_get_element() {
        let mut bs = BlockSparse::new(4, 4, 2).unwrap();
        let data = vec![1.0, 2.0, 3.0, 4.0]; // block (1,0)
        bs.insert_block(1, 0, data).unwrap();
        // Block (1,0) covers rows 2..4, cols 0..2
        assert_eq!(bs.get(2, 0), 1.0);
        assert_eq!(bs.get(2, 1), 2.0);
        assert_eq!(bs.get(3, 0), 3.0);
        assert_eq!(bs.get(3, 1), 4.0);
        // Absent block
        assert_eq!(bs.get(0, 0), 0.0);
    }

    #[test]
    fn block_sparse_get_block_none() {
        let bs = BlockSparse::new(4, 4, 2).unwrap();
        assert!(bs.get_block(0, 0).is_none());
        assert!(bs.get_block(1, 1).is_none());
    }

    #[test]
    fn block_sparse_causal_mask_lower_triangular() {
        let mask = BlockSparse::causal_mask(4, 2).unwrap();
        // Lower triangular: position (r,c) should be 1 iff c <= r
        let dense = mask.to_dense();
        for r in 0..4 {
            for c in 0..4 {
                let expected = if c <= r { 1.0 } else { 0.0 };
                assert_eq!(
                    dense[r][c], expected,
                    "causal_mask({r},{c}) expected {expected} got {}",
                    dense[r][c]
                );
            }
        }
    }

    #[test]
    fn block_sparse_sliding_window_mask() {
        let mask = BlockSparse::sliding_window_mask(6, 2, 1).unwrap();
        let dense = mask.to_dense();
        // Each row r attends to positions max(0, r-1)..=r
        for r in 0..6_usize {
            for c in 0..6_usize {
                let expected = if c <= r && r - c <= 1 { 1.0 } else { 0.0 };
                assert_eq!(
                    dense[r][c], expected,
                    "sliding_window({r},{c}) expected {expected} got {}",
                    dense[r][c]
                );
            }
        }
    }

    #[test]
    fn block_sparse_matvec() {
        // 4×4 matrix, block_size=2. Insert identity-like block at (0,0) and (1,1).
        let mut bs = BlockSparse::new(4, 4, 2).unwrap();
        bs.insert_block(0, 0, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        bs.insert_block(1, 1, vec![2.0, 0.0, 0.0, 2.0]).unwrap();
        let y = bs.matvec(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!((y[0] - 1.0).abs() < 1e-12);
        assert!((y[1] - 2.0).abs() < 1e-12);
        assert!((y[2] - 6.0).abs() < 1e-12);
        assert!((y[3] - 8.0).abs() < 1e-12);
    }

    // ── Utilities ────────────────────────────────────────────────────────────

    #[test]
    fn top_k_sparsify_keeps_largest() {
        let v = vec![1.0, -5.0, 3.0, 0.1, -4.0];
        let out = top_k_sparsify(&v, 2);
        // Top-2 by abs: -5.0 (idx 1) and -4.0 (idx 4)
        assert_eq!(out[0], 0.0);
        assert!((out[1] - (-5.0)).abs() < 1e-12);
        assert_eq!(out[2], 0.0);
        assert_eq!(out[3], 0.0);
        assert!((out[4] - (-4.0)).abs() < 1e-12);
    }

    #[test]
    fn vector_sparsity_correct() {
        let v = vec![0.0, 0.001, 5.0, -3.0, 0.0];
        // threshold 0.01: 0.0, 0.001, 0.0 are zeros → 3/5
        let s = vector_sparsity(&v, 0.01);
        assert!((s - 0.6).abs() < 1e-12);
    }

    #[test]
    fn top_k_mask_selects_correct_indices() {
        let scores = vec![0.1, 0.9, 0.5, 0.8, 0.2];
        let mask = top_k_mask(&scores, 2);
        // Top-2: index 1 (0.9) and index 3 (0.8)
        assert!(mask[1]);
        assert!(mask[3]);
        assert!(!mask[0]);
        assert!(!mask[2]);
        assert!(!mask[4]);
    }

    #[test]
    fn sparse_add_vecs_merges_correctly() {
        let a_idx = vec![0, 2, 4];
        let a_val = vec![1.0, 2.0, 3.0];
        let b_idx = vec![2, 3];
        let b_val = vec![10.0, 20.0];
        let (idx, val) = sparse_add_vecs(&a_idx, &a_val, &b_idx, &b_val, 5);
        // Expected: idx=[0,2,3,4], val=[1,12,20,3]
        assert_eq!(idx, vec![0, 2, 3, 4]);
        assert!((val[0] - 1.0).abs() < 1e-12);
        assert!((val[1] - 12.0).abs() < 1e-12); // 2 + 10
        assert!((val[2] - 20.0).abs() < 1e-12);
        assert!((val[3] - 3.0).abs() < 1e-12);
    }

    // ── BigBird tests ─────────────────────────────────────────────────────────

    #[test]
    fn bigbird_global_token_count_equals_block_size() {
        let config = BigBirdConfig {
            num_attention_heads: 8,
            block_size: 4,
            num_random_blocks: 2,
            seed: 42,
        };
        let mask = BigBirdAttentionMask::new(16, &config);
        assert_eq!(mask.global_tokens.len(), 4, "global tokens = block_size");
        assert_eq!(mask.global_tokens, vec![0, 1, 2, 3]);
    }

    #[test]
    fn bigbird_global_tokens_always_attend() {
        let config = BigBirdConfig {
            num_attention_heads: 4,
            block_size: 2,
            num_random_blocks: 1,
            seed: 7,
        };
        let mask = BigBirdAttentionMask::new(8, &config);
        // Every token should attend to global token 0
        for q in 0..8 {
            assert!(
                mask.should_attend(q, 0),
                "position {q} must attend to global token 0"
            );
        }
        // Global token 0 must attend to every key
        for k in 0..8 {
            assert!(
                mask.should_attend(0, k),
                "global token 0 must attend to position {k}"
            );
        }
    }

    #[test]
    fn bigbird_sliding_window_attends() {
        let config = BigBirdConfig {
            num_attention_heads: 2,
            block_size: 4,
            num_random_blocks: 0,
            seed: 0,
        };
        let mask = BigBirdAttentionMask::new(16, &config);
        // Position 8 should attend to 9 (diff=1 <= window=4) but not to 15 (diff=7 > 4)
        // unless 15 is a global token (which it isn't — globals are 0..4)
        assert!(mask.should_attend(8, 9), "within window: 8→9");
        // diff(8,15) = 7 > window_size(4) and 15 is not global → should NOT attend
        // unless random block pair lands on (2, 3)
        // We just test that within-window attends:
        assert!(mask.should_attend(7, 8), "within window: 7→8");
    }

    #[test]
    fn bigbird_dense_mask_shape() {
        let config = BigBirdConfig {
            num_attention_heads: 4,
            block_size: 2,
            num_random_blocks: 1,
            seed: 1,
        };
        let mask = BigBirdAttentionMask::new(6, &config);
        let dense = mask.to_dense_mask();
        assert_eq!(dense.len(), 6);
        for row in &dense {
            assert_eq!(row.len(), 6);
        }
    }

    #[test]
    fn bigbird_dense_mask_global_token_row_is_zero() {
        let config = BigBirdConfig {
            num_attention_heads: 2,
            block_size: 2,
            num_random_blocks: 0,
            seed: 0,
        };
        let mask = BigBirdAttentionMask::new(6, &config);
        let dense = mask.to_dense_mask();
        // Row 0 = global token → all entries should be 0.0 (attends to everything)
        for &v in &dense[0] {
            assert!((v - 0.0).abs() < 1e-6, "global token row 0 should all be 0.0");
        }
    }

    #[test]
    fn bigbird_sparsity_ratio_between_0_and_1() {
        let config = BigBirdConfig {
            num_attention_heads: 4,
            block_size: 4,
            num_random_blocks: 2,
            seed: 99,
        };
        let mask = BigBirdAttentionMask::new(16, &config);
        let ratio = mask.sparsity_ratio();
        assert!(
            ratio >= 0.0 && ratio <= 1.0,
            "sparsity ratio must be in [0,1]: got {ratio}"
        );
        assert!(ratio > 0.0, "with global tokens + window there must be some attention");
    }

    // ── Longformer tests ──────────────────────────────────────────────────────

    #[test]
    fn longformer_mask_within_window_is_zero() {
        let config = LongformerConfig {
            window_size: 2,
            global_token_indices: vec![],
        };
        let mask = longformer_attention_mask(8, &config);
        // Position 3 attends to positions 1,2,3,4,5 (|diff| <= 2)
        assert!(
            (mask[3][1] - 0.0).abs() < 1e-6,
            "within window should be 0.0"
        );
        assert!(
            (mask[3][5] - 0.0).abs() < 1e-6,
            "within window should be 0.0"
        );
    }

    #[test]
    fn longformer_mask_outside_window_is_large_neg() {
        let config = LongformerConfig {
            window_size: 1,
            global_token_indices: vec![],
        };
        let mask = longformer_attention_mask(8, &config);
        // Position 0 and position 7: diff=7 > window=1 and neither is global
        assert!(
            mask[0][7] < -100.0,
            "outside window and non-global should be -1e9: got {}",
            mask[0][7]
        );
    }

    #[test]
    fn longformer_mask_global_token_attends_everywhere() {
        let config = LongformerConfig {
            window_size: 1,
            global_token_indices: vec![0],
        };
        let mask = longformer_attention_mask(8, &config);
        // Global token 0 attends to all positions
        for k in 0..8 {
            assert!(
                (mask[0][k] - 0.0).abs() < 1e-6,
                "global token 0 should attend to position {k}: got {}",
                mask[0][k]
            );
        }
        // All positions attend to global token 0
        for q in 0..8 {
            assert!(
                (mask[q][0] - 0.0).abs() < 1e-6,
                "position {q} should attend to global token 0: got {}",
                mask[q][0]
            );
        }
    }

    #[test]
    fn longformer_combine_local_global_uses_global_for_global_tokens() {
        let config = LongformerConfig {
            window_size: 1,
            global_token_indices: vec![0, 2],
        };
        let local = vec![1.0_f32, 2.0, 3.0, 4.0];
        let global = vec![10.0_f32, 20.0, 30.0, 40.0];
        let combined = longformer_combine_local_global(&local, &global, &config);
        // index 0 and 2 are global → use global values
        assert!((combined[0] - 10.0).abs() < 1e-6);
        assert!((combined[1] - 2.0).abs() < 1e-6);
        assert!((combined[2] - 30.0).abs() < 1e-6);
        assert!((combined[3] - 4.0).abs() < 1e-6);
    }

    // ── BlockSparseAttn tests ──────────────────────────────────────────────────

    #[test]
    fn block_sparse_attn_mask_correct_length() {
        let config = BlockSparseAttnConfig {
            block_size: 4,
            num_heads: 8,
            sparsity: 0.5,
            seed: 0,
        };
        let mask = block_sparse_attn_mask(16, &config);
        // n_blocks = 16/4 = 4, total = 4*4 = 16
        assert_eq!(mask.len(), 16);
    }

    #[test]
    fn block_sparse_attn_mask_sparsity_zero_all_attend() {
        let config = BlockSparseAttnConfig {
            block_size: 2,
            num_heads: 1,
            sparsity: 0.0, // attend all
            seed: 42,
        };
        let mask = block_sparse_attn_mask(8, &config);
        for (i, &v) in mask.iter().enumerate() {
            assert!(v, "with sparsity=0 all blocks should attend, but block {i} does not");
        }
    }

    #[test]
    fn block_sparse_attn_mask_sparsity_one_none_attend() {
        let config = BlockSparseAttnConfig {
            block_size: 2,
            num_heads: 1,
            sparsity: 1.0, // skip all
            seed: 1,
        };
        let mask = block_sparse_attn_mask(8, &config);
        for (i, &v) in mask.iter().enumerate() {
            assert!(!v, "with sparsity=1 no blocks should attend, but block {i} does");
        }
    }

    #[test]
    fn block_sparse_attn_mask_non_divisible_seq_len() {
        // seq_len=10, block_size=3 → n_blocks=4, total=16
        let config = BlockSparseAttnConfig {
            block_size: 3,
            num_heads: 2,
            sparsity: 0.3,
            seed: 5,
        };
        let mask = block_sparse_attn_mask(10, &config);
        assert_eq!(mask.len(), 16);
    }
}
