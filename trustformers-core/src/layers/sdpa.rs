//! Optimized Scaled Dot-Product Attention (SDPA) kernels.
//!
//! This module provides CPU implementations of `softmax(Q K^T / sqrt(d) + mask) V`:
//!
//! * [`SDPA::attention`] dispatches on sequence length,
//! * a dense kernel that materialises one `seq_q x seq_k` score matrix per
//!   `(batch, head)` pair (fast for short and medium sequences),
//! * a memory-efficient tiled kernel with online softmax for long sequences.
//!
//! Both kernels parallelise over the flattened `(batch, head)` index space via
//! `scirs2_core::parallel_ops` and write directly into disjoint slices of the
//! output buffer.

use crate::errors::{Result, TrustformersError};
use crate::layers::attention::flash_kernel::head_matrix;
use crate::layers::attention::mask::MaskView;
use crate::tensor::Tensor;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayD, ArrayView1, CowArray, Ix2, IxDyn};
use scirs2_core::parallel_ops::*;
use scirs2_core::simd::activation::simd_softmax_f32;
#[cfg(not(target_os = "macos"))]
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Minimum size threshold for BLAS GEMM
const MIN_SIZE_FOR_BLAS: usize = 32;

/// Minimum size threshold for SIMD softmax
const MIN_SIZE_FOR_SIMD_SOFTMAX: usize = 64;

/// Row-major GEMM: `C(m x n) = alpha * A(m x k) * B(k x n) + beta * C`.
///
/// On macOS this bridges to OxiBLAS (pure Rust); elsewhere it falls back to
/// the scirs2-core SIMD GEMM.
#[cfg(target_os = "macos")]
#[inline]
pub(crate) fn blas_sgemm(
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    use oxiblas_blas::level3::gemm;
    use oxiblas_matrix::{MatMut, MatRef};

    // Bridge row-major → col-major via Cᵀ = Bᵀ·Aᵀ identity:
    // Row-major A(m×k) reinterpreted as col-major is Aᵀ(k×m), lda=k.
    // Row-major B(k×n) reinterpreted as col-major is Bᵀ(n×k), lda=n.
    // Row-major C(m×n) reinterpreted as col-major is Cᵀ(n×m), lda=n.
    // gemm(Bᵀ, Aᵀ) → Cᵀ = alpha·Bᵀ·Aᵀ + beta·Cᵀ = (alpha·A·B + beta·C)ᵀ. ✓
    let a_t = MatRef::from_column_major(a, k, m).expect("A slice must hold m*k elements");
    let b_t = MatRef::from_column_major(b, n, k).expect("B slice must hold k*n elements");
    let c_t = MatMut::from_column_major(c, n, m).expect("C slice must hold m*n elements");

    // GEMM: Cᵀ = alpha * Bᵀ * Aᵀ + beta * Cᵀ
    gemm(alpha, b_t, a_t, beta, c_t);
}

/// Fallback for non-macOS: use scirs2-core SIMD GEMM
#[cfg(not(target_os = "macos"))]
#[inline]
pub(crate) fn blas_sgemm(
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    // Borrow the inputs as views — `simd_gemm` only reads `a`/`b`, so no copy is needed.
    // `c` is read only when beta != 0; in the common beta == 0.0 case its prior contents
    // are discarded, so allocate zeros instead of copying `c` in.
    let a_arr =
        scirs2_core::ndarray::ArrayView2::from_shape((m, k), a).expect("BLAS input shape mismatch");
    let b_arr =
        scirs2_core::ndarray::ArrayView2::from_shape((k, n), b).expect("BLAS input shape mismatch");
    let mut c_arr = if beta == 0.0 {
        Array2::zeros((m, n))
    } else {
        Array2::from_shape_vec((m, n), c.to_vec()).expect("BLAS output shape mismatch")
    };
    f32::simd_gemm(alpha, &a_arr, &b_arr, beta, &mut c_arr);
    if let Some(slice) = c_arr.as_slice() {
        c.copy_from_slice(slice);
    } else {
        // Fallback: copy element by element
        for (i, &val) in c_arr.iter().enumerate() {
            c[i] = val;
        }
    }
}

/// Row-major GEMM with a transposed right operand:
/// `C(m x n) = alpha * A(m x k) * B(n x k)^T + beta * C`.
///
/// Attention needs `Q K^T` where both operands are stored as
/// `[sequence, head_dim]`; this variant consumes `K` in place instead of
/// materialising its transpose.
#[cfg(target_os = "macos")]
#[inline]
pub(crate) fn blas_sgemm_nt(
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    use oxiblas_blas::level3::gemm::gemm_transposed;
    use oxiblas_blas::level3::Trans;
    use oxiblas_matrix::{MatMut, MatRef};

    // Row-major A(m×k) reinterpreted as col-major is Aᵀ(k×m), lda=k.
    // Row-major B(n×k) reinterpreted as col-major is Bᵀ(k×n), lda=k.
    // Row-major C(m×n) reinterpreted as col-major is Cᵀ(n×m), lda=n.
    // We need Cᵀ = (A·Bᵀ)ᵀ = B·Aᵀ, and B = (Bᵀ)ᵀ, so the left operand is the
    // *transposed* view of `b_t`.
    //
    // `gemm_transposed` reads the transposed operand with swapped indices while
    // packing (no copy). A plain `MatRef::transpose()` view would NOT work here:
    // the `NoTrans` packing path assumes a unit row-to-row stride, which a
    // transposed view does not have.
    let a_t = MatRef::from_column_major(a, k, m).expect("A slice must hold m*k elements");
    let b_t = MatRef::from_column_major(b, k, n).expect("B slice must hold n*k elements");
    let c_t = MatMut::from_column_major(c, n, m).expect("C slice must hold m*n elements");

    // Cᵀ = alpha * (Bᵀ)ᵀ * Aᵀ + beta * Cᵀ  ⇒  C = alpha * A * Bᵀ + beta * C. ✓
    gemm_transposed(Trans::Trans, Trans::NoTrans, alpha, b_t, a_t, beta, c_t);
}

/// Fallback for non-macOS: materialise `Bᵀ` and defer to the SIMD GEMM.
#[cfg(not(target_os = "macos"))]
#[inline]
pub(crate) fn blas_sgemm_nt(
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let b_arr =
        scirs2_core::ndarray::ArrayView2::from_shape((n, k), b).expect("BLAS input shape mismatch");
    let b_transposed = b_arr.t().to_owned();
    let b_data = b_transposed.as_slice().expect("freshly allocated transpose is contiguous");
    blas_sgemm(alpha, a, b_data, beta, c, m, k, n);
}

/// Borrow a `[batch, heads, seq, head_dim]` F32 tensor, or fail with a
/// structured error.
fn attention_operand<'a>(tensor: &'a Tensor, name: &str) -> Result<&'a ArrayD<f32>> {
    match tensor {
        Tensor::F32(array) if array.ndim() == 4 => Ok(array),
        Tensor::F32(array) => Err(TrustformersError::tensor_op_error(
            &format!(
                "SDPA expects 4-D [batch, heads, seq, head_dim] tensors; {} has shape {:?}",
                name,
                array.shape()
            ),
            "SDPA",
        )),
        other => Err(TrustformersError::tensor_op_error(
            &format!(
                "SDPA supports F32 tensors only; {} has dtype {:?}",
                name,
                other.dtype()
            ),
            "SDPA",
        )),
    }
}

/// Apply post-softmax dropout in place, scaling survivors by `1/(1-p)`.
fn apply_attention_dropout(probabilities: &mut [f32], dropout_prob: f32) {
    use scirs2_core::random::*;

    let mut rng = thread_rng();
    let scale = 1.0 / (1.0 - dropout_prob);
    for value in probabilities.iter_mut() {
        if rng.random::<f32>() < dropout_prob {
            *value = 0.0;
        } else {
            *value *= scale;
        }
    }
}

/// Row-wise softmax over a flat `rows x columns` buffer, tolerating
/// `-inf` entries produced by masking.
fn softmax_rows_in_place(scores: &mut [f32], rows: usize, columns: usize, allow_simd: bool) {
    if allow_simd && columns >= MIN_SIZE_FOR_SIMD_SOFTMAX {
        for row in 0..rows {
            let slice = &mut scores[row * columns..(row + 1) * columns];
            let view = ArrayView1::from(&*slice);
            let softmaxed: Array1<f32> = simd_softmax_f32(&view);
            for (destination, value) in slice.iter_mut().zip(softmaxed.iter()) {
                *destination = *value;
            }
        }
        return;
    }

    for row in 0..rows {
        let slice = &mut scores[row * columns..(row + 1) * columns];
        let max = slice.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        if max == f32::NEG_INFINITY {
            // Fully masked row: no key is reachable, emit an all-zero
            // distribution rather than NaNs.
            slice.fill(0.0);
            continue;
        }
        let mut sum = 0.0f32;
        for value in slice.iter_mut() {
            let exponential = (*value - max).exp();
            *value = exponential;
            sum += exponential;
        }
        let inverse = 1.0 / sum.max(f32::MIN_POSITIVE);
        for value in slice.iter_mut() {
            *value *= inverse;
        }
    }
}

/// Optimized Scaled Dot-Product Attention (SDPA) kernels
///
/// This module provides various optimized implementations of scaled dot-product attention
/// for different hardware and use cases:
/// - Dense SDPA for short and medium sequences
/// - Memory-efficient tiled SDPA with online softmax for long sequences
/// - Optional post-softmax attention dropout
pub struct SDPA;

impl SDPA {
    /// Basic scaled dot-product attention: softmax(QK^T / sqrt(d_k))V
    ///
    /// Args:
    ///   q: Query tensor [batch, heads, seq_q, head_dim]
    ///   k: Key tensor [batch, heads, seq_k, head_dim]
    ///   v: Value tensor [batch, heads, seq_k, head_dim]
    ///   attn_mask: Optional attention mask, see [`crate::layers::attention::mask`]
    ///   causal: Whether to apply causal masking
    pub fn attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
        causal: bool,
    ) -> Result<Tensor> {
        Self::attention_with_dropout(q, k, v, attn_mask, causal, None)
    }

    /// Scaled dot-product attention with optional post-softmax dropout.
    ///
    /// `dropout_prob` must be in `[0, 1)`; `None` disables dropout entirely.
    pub fn attention_with_dropout(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
        causal: bool,
        dropout_prob: Option<f32>,
    ) -> Result<Tensor> {
        if let Some(prob) = dropout_prob {
            if !(0.0..1.0).contains(&prob) {
                return Err(TrustformersError::tensor_op_error(
                    &format!("Attention dropout probability must be in [0, 1), got {prob}"),
                    "SDPA::attention_with_dropout",
                ));
            }
        }
        let dropout_prob = dropout_prob.filter(|&prob| prob > 0.0);

        let seq_q = attention_operand(q, "query")?.shape()[2];
        let seq_k = attention_operand(k, "key")?.shape()[2];

        if seq_q > 2048 || seq_k > 2048 {
            // Memory-efficient tiled attention for long sequences
            Self::tiled_attention(q, k, v, attn_mask, causal, dropout_prob)
        } else {
            Self::dense_attention(q, k, v, attn_mask, causal, dropout_prob)
        }
    }

    /// Validate Q/K/V and return `(batch, heads, seq_q, seq_k, head_dim)`.
    fn attention_dims(
        q: &ArrayD<f32>,
        k: &ArrayD<f32>,
        v: &ArrayD<f32>,
    ) -> Result<(usize, usize, usize, usize, usize)> {
        let q_shape = q.shape();
        let k_shape = k.shape();
        let v_shape = v.shape();

        if k_shape[0] != q_shape[0]
            || v_shape[0] != q_shape[0]
            || k_shape[1] != q_shape[1]
            || v_shape[1] != q_shape[1]
        {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Q/K/V batch and head counts must match, got {:?} / {:?} / {:?}",
                    q_shape, k_shape, v_shape
                ),
                "SDPA",
            ));
        }
        if k_shape[2] != v_shape[2] {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Key and value sequence lengths must match, got {} and {}",
                    k_shape[2], v_shape[2]
                ),
                "SDPA",
            ));
        }
        if k_shape[3] != q_shape[3] || v_shape[3] != q_shape[3] {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Q/K/V head dimensions must match, got {} / {} / {}",
                    q_shape[3], k_shape[3], v_shape[3]
                ),
                "SDPA",
            ));
        }

        Ok((q_shape[0], q_shape[1], q_shape[2], k_shape[2], q_shape[3]))
    }

    /// Dense SDPA: one `seq_q x seq_k` score matrix per `(batch, head)` pair.
    ///
    /// Work is distributed over `(batch, head)` and each worker writes straight
    /// into its own contiguous slice of the output buffer.
    fn dense_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
        causal: bool,
        dropout_prob: Option<f32>,
    ) -> Result<Tensor> {
        let q_arr = attention_operand(q, "query")?;
        let k_arr = attention_operand(k, "key")?;
        let v_arr = attention_operand(v, "value")?;
        let (batch_size, num_heads, seq_q, seq_k, head_dim) =
            Self::attention_dims(q_arr, k_arr, v_arr)?;

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mask_view = match attn_mask {
            Some(mask) => Some(MaskView::new(mask, batch_size, num_heads, seq_q, seq_k)?),
            None => None,
        };

        let mut output = vec![0.0f32; batch_size * num_heads * seq_q * head_dim];
        if output.is_empty() {
            return Self::finish(output, batch_size, num_heads, seq_q, head_dim);
        }

        let use_blas = seq_q >= MIN_SIZE_FOR_BLAS
            && seq_k >= MIN_SIZE_FOR_BLAS
            && head_dim >= MIN_SIZE_FOR_BLAS;

        output.par_chunks_mut(seq_q * head_dim).enumerate().try_for_each(
            |(flat_index, out_chunk)| -> Result<()> {
                let batch_index = flat_index / num_heads;
                let head_index = flat_index % num_heads;

                let q_2d = head_matrix(q_arr, batch_index, head_index)?;
                let k_2d = head_matrix(k_arr, batch_index, head_index)?;
                let v_2d = head_matrix(v_arr, batch_index, head_index)?;

                let mut scores = vec![0.0f32; seq_q * seq_k];
                if use_blas {
                    let q_data = Self::contiguous(&q_2d, "query head")?;
                    let k_data = Self::contiguous(&k_2d, "key head")?;
                    blas_sgemm_nt(
                        scale,
                        q_data,
                        k_data,
                        0.0,
                        &mut scores,
                        seq_q,
                        head_dim,
                        seq_k,
                    );
                } else {
                    let product = q_2d.dot(&k_2d.t());
                    for (destination, &value) in scores.iter_mut().zip(product.iter()) {
                        *destination = value * scale;
                    }
                }

                if causal {
                    for row in 0..seq_q {
                        let allowed = (row + 1).min(seq_k);
                        scores[row * seq_k + allowed..(row + 1) * seq_k].fill(f32::NEG_INFINITY);
                    }
                }
                if let Some(mask) = mask_view.as_ref() {
                    for row in 0..seq_q {
                        for column in 0..seq_k {
                            let penalty = mask.additive(batch_index, head_index, row, column);
                            if penalty != 0.0 {
                                scores[row * seq_k + column] += penalty;
                            }
                        }
                    }
                }

                // The SIMD softmax has no `-inf` handling, so it is only used
                // when nothing has been masked out.
                let allow_simd = !causal && mask_view.is_none();
                softmax_rows_in_place(&mut scores, seq_q, seq_k, allow_simd);

                if let Some(prob) = dropout_prob {
                    apply_attention_dropout(&mut scores, prob);
                }

                if use_blas {
                    let v_data = Self::contiguous(&v_2d, "value head")?;
                    blas_sgemm(1.0, &scores, v_data, 0.0, out_chunk, seq_q, seq_k, head_dim);
                } else {
                    let weights = Array2::from_shape_vec((seq_q, seq_k), scores)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    let product = weights.dot(&v_2d);
                    for (destination, &value) in out_chunk.iter_mut().zip(product.iter()) {
                        *destination = value;
                    }
                }

                Ok(())
            },
        )?;

        Self::finish(output, batch_size, num_heads, seq_q, head_dim)
    }

    /// Memory-efficient tiled SDPA with online softmax for long sequences.
    ///
    /// Never materialises the full `seq_q x seq_k` score matrix: peak extra
    /// memory is one `tile x tile` block per worker.
    fn tiled_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
        causal: bool,
        dropout_prob: Option<f32>,
    ) -> Result<Tensor> {
        let q_arr = attention_operand(q, "query")?;
        let k_arr = attention_operand(k, "key")?;
        let v_arr = attention_operand(v, "value")?;
        let (batch_size, num_heads, seq_q, seq_k, head_dim) =
            Self::attention_dims(q_arr, k_arr, v_arr)?;

        let scale = 1.0 / (head_dim as f32).sqrt();
        let tile_size = 256.min(seq_q.max(1)).max(1);
        let key_tile_size = 256.min(seq_k.max(1)).max(1);

        let mask_view = match attn_mask {
            Some(mask) => Some(MaskView::new(mask, batch_size, num_heads, seq_q, seq_k)?),
            None => None,
        };

        let mut output = vec![0.0f32; batch_size * num_heads * seq_q * head_dim];
        if output.is_empty() {
            return Self::finish(output, batch_size, num_heads, seq_q, head_dim);
        }

        output.par_chunks_mut(seq_q * head_dim).enumerate().try_for_each(
            |(flat_index, out_chunk)| -> Result<()> {
                let batch_index = flat_index / num_heads;
                let head_index = flat_index % num_heads;

                let q_2d = head_matrix(q_arr, batch_index, head_index)?;
                let k_2d = head_matrix(k_arr, batch_index, head_index)?;
                let v_2d = head_matrix(v_arr, batch_index, head_index)?;

                let mut scores = vec![0.0f32; tile_size * key_tile_size];
                let mut accumulator = vec![0.0f32; tile_size * head_dim];
                let mut row_max = vec![f32::NEG_INFINITY; tile_size];
                let mut row_sum = vec![0.0f32; tile_size];

                for q_start in (0..seq_q).step_by(tile_size) {
                    let q_end = (q_start + tile_size).min(seq_q);
                    let q_rows = q_end - q_start;

                    row_max[..q_rows].fill(f32::NEG_INFINITY);
                    row_sum[..q_rows].fill(0.0);
                    accumulator[..q_rows * head_dim].fill(0.0);

                    let q_tile = q_2d.slice(s![q_start..q_end, ..]);
                    let q_data = q_tile.as_slice().ok_or_else(|| {
                        TrustformersError::tensor_op_error(
                            "query tile must be contiguous for GEMM",
                            "SDPA::tiled_attention",
                        )
                    })?;

                    for k_start in (0..seq_k).step_by(key_tile_size) {
                        let k_end = (k_start + key_tile_size).min(seq_k);
                        let k_rows = k_end - k_start;

                        if causal && k_start >= q_end {
                            break;
                        }

                        let k_tile = k_2d.slice(s![k_start..k_end, ..]);
                        let v_tile = v_2d.slice(s![k_start..k_end, ..]);
                        let k_data = k_tile.as_slice().ok_or_else(|| {
                            TrustformersError::tensor_op_error(
                                "key tile must be contiguous for GEMM",
                                "SDPA::tiled_attention",
                            )
                        })?;
                        let v_data = v_tile.as_slice().ok_or_else(|| {
                            TrustformersError::tensor_op_error(
                                "value tile must be contiguous for GEMM",
                                "SDPA::tiled_attention",
                            )
                        })?;

                        let score_tile = &mut scores[..q_rows * k_rows];
                        blas_sgemm_nt(
                            scale, q_data, k_data, 0.0, score_tile, q_rows, head_dim, k_rows,
                        );

                        if causal {
                            for row in 0..q_rows {
                                let allowed =
                                    (q_start + row + 1).saturating_sub(k_start).min(k_rows);
                                if allowed < k_rows {
                                    score_tile[row * k_rows + allowed..(row + 1) * k_rows]
                                        .fill(f32::NEG_INFINITY);
                                }
                            }
                        }
                        if let Some(mask) = mask_view.as_ref() {
                            for row in 0..q_rows {
                                for column in 0..k_rows {
                                    let penalty = mask.additive(
                                        batch_index,
                                        head_index,
                                        q_start + row,
                                        k_start + column,
                                    );
                                    if penalty != 0.0 {
                                        score_tile[row * k_rows + column] += penalty;
                                    }
                                }
                            }
                        }

                        for row in 0..q_rows {
                            let row_slice = &mut score_tile[row * k_rows..(row + 1) * k_rows];
                            let tile_max =
                                row_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                            let previous_max = row_max[row];
                            let combined_max = previous_max.max(tile_max);

                            if combined_max == f32::NEG_INFINITY {
                                row_slice.fill(0.0);
                                continue;
                            }

                            let correction = if previous_max == f32::NEG_INFINITY {
                                0.0
                            } else {
                                (previous_max - combined_max).exp()
                            };

                            let mut tile_sum = 0.0f32;
                            for value in row_slice.iter_mut() {
                                let exponential = (*value - combined_max).exp();
                                *value = exponential;
                                tile_sum += exponential;
                            }

                            // The running denominator uses the *undropped*
                            // weights, so scaling the numerator below yields
                            // exactly post-softmax dropout.
                            row_sum[row] = row_sum[row] * correction + tile_sum;
                            row_max[row] = combined_max;

                            if let Some(prob) = dropout_prob {
                                apply_attention_dropout(row_slice, prob);
                            }

                            if correction != 1.0 {
                                for slot in
                                    accumulator[row * head_dim..(row + 1) * head_dim].iter_mut()
                                {
                                    *slot *= correction;
                                }
                            }
                        }

                        blas_sgemm(
                            1.0,
                            score_tile,
                            v_data,
                            1.0,
                            &mut accumulator[..q_rows * head_dim],
                            q_rows,
                            k_rows,
                            head_dim,
                        );
                    }

                    for row in 0..q_rows {
                        let inverse = if row_sum[row] > 0.0 { 1.0 / row_sum[row] } else { 0.0 };
                        let source = &accumulator[row * head_dim..(row + 1) * head_dim];
                        let destination = &mut out_chunk
                            [(q_start + row) * head_dim..(q_start + row + 1) * head_dim];
                        for (slot, &value) in destination.iter_mut().zip(source.iter()) {
                            *slot = value * inverse;
                        }
                    }
                }

                Ok(())
            },
        )?;

        Self::finish(output, batch_size, num_heads, seq_q, head_dim)
    }

    fn contiguous<'a>(matrix: &'a CowArray<'_, f32, Ix2>, what: &str) -> Result<&'a [f32]> {
        matrix.as_slice().ok_or_else(|| {
            TrustformersError::tensor_op_error(
                &format!("{what} must be contiguous for GEMM"),
                "SDPA",
            )
        })
    }

    fn finish(
        data: Vec<f32>,
        batch_size: usize,
        num_heads: usize,
        seq_q: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let array = ArrayD::from_shape_vec(IxDyn(&[batch_size, num_heads, seq_q, head_dim]), data)
            .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
        Ok(Tensor::F32(array))
    }

    /// Fused SDPA kernel with post-softmax attention dropout.
    ///
    /// Dropout is applied to the attention probabilities (as in the reference
    /// transformer implementations) and only when `training` is true;
    /// at inference the call is exactly [`SDPA::attention`].
    pub fn fused_attention_dropout(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
        causal: bool,
        dropout_prob: f32,
        training: bool,
    ) -> Result<Tensor> {
        let dropout = if training { Some(dropout_prob) } else { None };
        Self::attention_with_dropout(q, k, v, attn_mask, causal, dropout)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    /// Deterministic pseudo-random tensor so assertions are reproducible.
    fn deterministic(shape: &[usize], seed: u32) -> Tensor {
        let count: usize = shape.iter().product();
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12_345);
        let mut data = Vec::with_capacity(count);
        for _ in 0..count {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
            data.push(unit * 2.0 - 1.0);
        }
        Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(shape), data).expect("test tensor shape must be valid"),
        )
    }

    /// Naive `softmax(scale * Q K^T + mask) V` reference.
    fn naive_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        causal: bool,
    ) -> Vec<f32> {
        let (Tensor::F32(q_arr), Tensor::F32(k_arr), Tensor::F32(v_arr)) = (q, k, v) else {
            panic!("reference implementation requires F32 tensors");
        };
        let batch = q_arr.shape()[0];
        let heads = q_arr.shape()[1];
        let seq_q = q_arr.shape()[2];
        let head_dim = q_arr.shape()[3];
        let seq_k = k_arr.shape()[2];
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mask_view =
            mask.map(|m| MaskView::new(m, batch, heads, seq_q, seq_k).expect("valid test mask"));

        let mut out = vec![0.0f32; batch * heads * seq_q * head_dim];
        for b in 0..batch {
            for h in 0..heads {
                for i in 0..seq_q {
                    let mut scores = vec![f32::NEG_INFINITY; seq_k];
                    for (j, score) in scores.iter_mut().enumerate() {
                        if causal && j > i {
                            continue;
                        }
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_arr[[b, h, i, d]] * k_arr[[b, h, j, d]];
                        }
                        let mut value = dot * scale;
                        if let Some(view) = mask_view.as_ref() {
                            value += view.additive(b, h, i, j);
                        }
                        *score = value;
                    }
                    let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                    if max == f32::NEG_INFINITY {
                        continue;
                    }
                    let exponentials: Vec<f32> = scores.iter().map(|&s| (s - max).exp()).collect();
                    let sum: f32 = exponentials.iter().sum();
                    for d in 0..head_dim {
                        let mut acc = 0.0f32;
                        for (j, &weight) in exponentials.iter().enumerate() {
                            acc += weight * v_arr[[b, h, j, d]];
                        }
                        out[((b * heads + h) * seq_q + i) * head_dim + d] = acc / sum;
                    }
                }
            }
        }
        out
    }

    fn max_abs_difference(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "output length mismatch");
        a.iter().zip(b.iter()).fold(0.0f32, |acc, (x, y)| acc.max((x - y).abs()))
    }

    #[test]
    fn blas_sgemm_matches_reference_product() {
        // C(2x3) = 2.0 * A(2x4) * B(4x3) + 1.0 * C
        let a: Vec<f32> = (0..8).map(|x| x as f32 * 0.5 - 1.0).collect();
        let b: Vec<f32> = (0..12).map(|x| 1.0 - x as f32 * 0.25).collect();
        let mut c = vec![0.5f32; 6];

        let a_arr = Array2::from_shape_vec((2, 4), a.clone()).expect("shape");
        let b_arr = Array2::from_shape_vec((4, 3), b.clone()).expect("shape");
        let mut expected = a_arr.dot(&b_arr);
        expected.mapv_inplace(|x| x * 2.0);
        for (slot, previous) in expected.iter_mut().zip(c.iter()) {
            *slot += previous;
        }

        blas_sgemm(2.0, &a, &b, 1.0, &mut c, 2, 4, 3);
        let expected_data: Vec<f32> = expected.iter().copied().collect();
        assert!(
            max_abs_difference(&c, &expected_data) < 1e-5,
            "blas_sgemm mismatch: {c:?} vs {expected_data:?}"
        );
    }

    #[test]
    fn blas_sgemm_nt_matches_transposed_product() {
        // C(3x2) = A(3x5) * B(2x5)^T
        let a: Vec<f32> = (0..15).map(|x| (x as f32).sin()).collect();
        let b: Vec<f32> = (0..10).map(|x| (x as f32).cos()).collect();
        let mut c = vec![0.0f32; 6];

        let a_arr = Array2::from_shape_vec((3, 5), a.clone()).expect("shape");
        let b_arr = Array2::from_shape_vec((2, 5), b.clone()).expect("shape");
        let expected = a_arr.dot(&b_arr.t());

        blas_sgemm_nt(1.0, &a, &b, 0.0, &mut c, 3, 5, 2);
        let expected_data: Vec<f32> = expected.iter().copied().collect();
        assert!(
            max_abs_difference(&c, &expected_data) < 1e-5,
            "blas_sgemm_nt mismatch: {c:?} vs {expected_data:?}"
        );
    }

    #[test]
    fn blas_sgemm_nt_accumulates_with_beta() {
        let a: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..6).map(|x| (x as f32) * 0.5).collect();
        let mut c = vec![1.0f32; 4];

        let a_arr = Array2::from_shape_vec((2, 3), a.clone()).expect("shape");
        let b_arr = Array2::from_shape_vec((2, 3), b.clone()).expect("shape");
        let mut expected = a_arr.dot(&b_arr.t());
        for slot in expected.iter_mut() {
            *slot += 1.0;
        }

        blas_sgemm_nt(1.0, &a, &b, 1.0, &mut c, 2, 3, 2);
        let expected_data: Vec<f32> = expected.iter().copied().collect();
        assert!(
            max_abs_difference(&c, &expected_data) < 1e-5,
            "blas_sgemm_nt beta accumulation mismatch: {c:?} vs {expected_data:?}"
        );
    }

    fn check_dense_against_reference(
        shape: (usize, usize, usize, usize, usize),
        causal: bool,
        mask: Option<Tensor>,
    ) {
        let (batch, heads, seq_q, seq_k, head_dim) = shape;
        let q = deterministic(&[batch, heads, seq_q, head_dim], 1);
        let k = deterministic(&[batch, heads, seq_k, head_dim], 2);
        let v = deterministic(&[batch, heads, seq_k, head_dim], 3);

        let output = SDPA::attention(&q, &k, &v, mask.as_ref(), causal)
            .expect("SDPA attention must succeed");
        assert_eq!(output.shape(), vec![batch, heads, seq_q, head_dim]);

        let expected = naive_attention(&q, &k, &v, mask.as_ref(), causal);
        let actual = output.data().expect("output data must be readable");
        let difference = max_abs_difference(&actual, &expected);
        assert!(
            difference < 1e-4,
            "SDPA deviates from the naive reference by {difference} for shape {shape:?}"
        );
    }

    #[test]
    fn dense_attention_matches_reference() {
        check_dense_against_reference((2, 4, 32, 32, 64), false, None);
    }

    #[test]
    fn dense_attention_matches_reference_small_matrices() {
        // Below MIN_SIZE_FOR_BLAS: exercises the ndarray fallback path.
        check_dense_against_reference((1, 2, 5, 7, 4), false, None);
    }

    #[test]
    fn dense_attention_matches_reference_causal() {
        check_dense_against_reference((1, 2, 16, 16, 32), true, None);
    }

    #[test]
    fn dense_attention_simd_softmax_path_matches_reference() {
        // seq_k >= MIN_SIZE_FOR_SIMD_SOFTMAX, no masking: SIMD softmax path.
        check_dense_against_reference((1, 2, 64, 64, 32), false, None);
    }

    #[test]
    fn dense_attention_matches_reference_with_mask() {
        let seq = 16;
        let mut data = vec![1.0f32; seq];
        data[5] = 0.0;
        data[9] = 0.0;
        let mask = Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, seq]), data).expect("valid mask shape"),
        );
        check_dense_against_reference((1, 2, seq, seq, 32), false, Some(mask));
    }

    #[test]
    fn tiled_attention_matches_reference() {
        let (batch, heads, seq, head_dim) = (1, 2, 300, 32);
        let q = deterministic(&[batch, heads, seq, head_dim], 4);
        let k = deterministic(&[batch, heads, seq, head_dim], 5);
        let v = deterministic(&[batch, heads, seq, head_dim], 6);

        let tiled = SDPA::tiled_attention(&q, &k, &v, None, true, None)
            .expect("tiled attention must succeed");
        assert_eq!(tiled.shape(), vec![batch, heads, seq, head_dim]);

        let expected = naive_attention(&q, &k, &v, None, true);
        let actual = tiled.data().expect("output data must be readable");
        let difference = max_abs_difference(&actual, &expected);
        assert!(
            difference < 1e-4,
            "tiled SDPA deviates from the naive reference by {difference}"
        );
    }

    #[test]
    fn tiled_and_dense_agree() {
        let q = deterministic(&[1, 2, 300, 16], 7);
        let k = deterministic(&[1, 2, 300, 16], 8);
        let v = deterministic(&[1, 2, 300, 16], 9);

        let dense = SDPA::dense_attention(&q, &k, &v, None, false, None).expect("dense");
        let tiled = SDPA::tiled_attention(&q, &k, &v, None, false, None).expect("tiled");
        let difference = max_abs_difference(
            &dense.data().expect("dense data"),
            &tiled.data().expect("tiled data"),
        );
        assert!(
            difference < 1e-4,
            "dense and tiled SDPA disagree by {difference}"
        );
    }

    #[test]
    fn causal_attention_ignores_future_tokens() {
        let seq = 24;
        let head_dim = 16;
        let q = deterministic(&[1, 1, seq, head_dim], 10);
        let k = deterministic(&[1, 1, seq, head_dim], 11);
        let v = deterministic(&[1, 1, seq, head_dim], 12);

        let base = SDPA::attention(&q, &k, &v, None, true).expect("baseline");
        let (Tensor::F32(k_arr), Tensor::F32(v_arr)) = (&k, &v) else {
            panic!("expected F32 tensors");
        };
        let mut k_perturbed = k_arr.clone();
        let mut v_perturbed = v_arr.clone();
        for position in seq / 2..seq {
            for dim in 0..head_dim {
                k_perturbed[[0, 0, position, dim]] += 2.0;
                v_perturbed[[0, 0, position, dim]] -= 3.0;
            }
        }
        let perturbed = SDPA::attention(
            &q,
            &Tensor::F32(k_perturbed),
            &Tensor::F32(v_perturbed),
            None,
            true,
        )
        .expect("perturbed");

        let base_data = base.data().expect("base data");
        let perturbed_data = perturbed.data().expect("perturbed data");
        let prefix = (seq / 2) * head_dim;
        assert!(
            max_abs_difference(&base_data[..prefix], &perturbed_data[..prefix]) < 1e-5,
            "causal SDPA leaked future information"
        );
    }

    #[test]
    fn attention_mask_removes_masked_keys() {
        // Attending over two keys where the second is masked out must return
        // exactly the first value row.
        let q = Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 2]), vec![1.0, 0.0]).expect("shape"),
        );
        let k = Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![1.0, 0.0, 1.0, 0.0]).expect("shape"),
        );
        let v = Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 2, 2]), vec![5.0, 7.0, -11.0, -13.0])
                .expect("shape"),
        );
        let mask = Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, 2]), vec![1.0, 0.0]).expect("shape"),
        );

        let output = SDPA::attention(&q, &k, &v, Some(&mask), false).expect("masked attention");
        let data = output.data().expect("output data");
        assert!((data[0] - 5.0).abs() < 1e-5, "got {data:?}");
        assert!((data[1] - 7.0).abs() < 1e-5, "got {data:?}");
    }

    #[test]
    fn fused_attention_dropout_actually_drops() {
        let q = deterministic(&[1, 2, 64, 32], 13);
        let k = deterministic(&[1, 2, 64, 32], 14);
        let v = deterministic(&[1, 2, 64, 32], 15);

        let inference = SDPA::fused_attention_dropout(&q, &k, &v, None, false, 0.5, false)
            .expect("inference pass");
        let baseline = SDPA::attention(&q, &k, &v, None, false).expect("baseline");
        assert!(
            max_abs_difference(
                &inference.data().expect("data"),
                &baseline.data().expect("data")
            ) < 1e-6,
            "dropout must be inert outside training"
        );

        let training = SDPA::fused_attention_dropout(&q, &k, &v, None, false, 0.5, true)
            .expect("training pass");
        assert!(
            max_abs_difference(
                &training.data().expect("data"),
                &baseline.data().expect("data")
            ) > 1e-3,
            "dropout_prob and training must actually change the result"
        );
    }

    #[test]
    fn fused_attention_dropout_rejects_invalid_probability() {
        let q = deterministic(&[1, 1, 4, 4], 16);
        assert!(SDPA::fused_attention_dropout(&q, &q, &q, None, false, 1.0, true).is_err());
        assert!(SDPA::fused_attention_dropout(&q, &q, &q, None, false, -0.1, true).is_err());
    }

    #[test]
    fn rejects_mismatched_shapes() {
        let q = deterministic(&[1, 2, 4, 8], 17);
        let k = deterministic(&[1, 2, 4, 4], 18);
        assert!(SDPA::attention(&q, &k, &k, None, false).is_err());
    }
}
