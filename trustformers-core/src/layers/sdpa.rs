#![allow(unused_variables)] // SDPA implementation with reserved parameters

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayD, Axis, IxDyn};
use scirs2_core::simd::activation::simd_softmax_f32;
#[cfg(not(target_os = "macos"))]
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Minimum size threshold for BLAS GEMM
const MIN_SIZE_FOR_BLAS: usize = 32;

/// Minimum size threshold for SIMD softmax
const MIN_SIZE_FOR_SIMD_SOFTMAX: usize = 64;

/// Direct BLAS GEMM using cblas_sgemm for maximum performance on macOS (Accelerate)
#[cfg(target_os = "macos")]
#[inline]
fn blas_sgemm(
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    unsafe {
        use cblas_sys::{cblas_sgemm, CblasNoTrans, CblasRowMajor};
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans,
            CblasNoTrans,
            m as i32,
            n as i32,
            k as i32,
            alpha,
            a.as_ptr(),
            k as i32,
            b.as_ptr(),
            n as i32,
            beta,
            c.as_mut_ptr(),
            n as i32,
        );
    }
}

/// Fallback for non-macOS: use scirs2-core SIMD GEMM
#[cfg(not(target_os = "macos"))]
#[inline]
fn blas_sgemm(
    alpha: f32,
    a: &[f32],
    b: &[f32],
    beta: f32,
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let a_arr = Array2::from_shape_vec((m, k), a.to_vec()).unwrap();
    let b_arr = Array2::from_shape_vec((k, n), b.to_vec()).unwrap();
    let mut c_arr = Array2::from_shape_vec((m, n), c.to_vec()).unwrap();
    f32::simd_gemm(alpha, &a_arr.view(), &b_arr.view(), beta, &mut c_arr);
    c.copy_from_slice(c_arr.as_slice().unwrap());
}

/// Optimized Scaled Dot-Product Attention (SDPA) kernels
///
/// This module provides various optimized implementations of scaled dot-product attention
/// for different hardware and use cases:
/// - Basic SDPA for CPU
/// - Memory-efficient SDPA with tiling
/// - Optimized kernels for specific sequence lengths
/// - Fused attention operations
pub struct SDPA;

impl SDPA {
    /// Basic scaled dot-product attention: softmax(QK^T / sqrt(d_k))V
    ///
    /// Args:
    ///   q: Query tensor [batch, heads, seq_q, head_dim]
    ///   k: Key tensor [batch, heads, seq_k, head_dim]
    ///   v: Value tensor [batch, heads, seq_k, head_dim]
    ///   attn_mask: Optional attention mask [batch, heads, seq_q, seq_k]
    ///   causal: Whether to apply causal masking
    pub fn attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
        causal: bool,
    ) -> Result<Tensor> {
        let q_shape = q.shape();
        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let seq_q = q_shape[2];
        let head_dim = q_shape[3];

        let k_shape = k.shape();
        let seq_k = k_shape[2];

        if seq_q <= 512 && seq_k <= 512 {
            // Use optimized kernel for small sequences
            Self::small_sequence_attention(q, k, v, attn_mask, causal)
        } else if seq_q > 2048 || seq_k > 2048 {
            // Use memory-efficient tiled attention for long sequences
            Self::tiled_attention(q, k, v, attn_mask, causal)
        } else {
            // Use standard attention for medium sequences
            Self::standard_attention(q, k, v, attn_mask, causal)
        }
    }

    /// Standard SDPA implementation
    fn standard_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
        causal: bool,
    ) -> Result<Tensor> {
        let q_shape = q.shape();
        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let seq_q = q_shape[2];
        let head_dim = q_shape[3];

        let k_shape = k.shape();
        let seq_k = k_shape[2];

        let scale = 1.0 / (head_dim as f32).sqrt();

        match (q, k, v) {
            (Tensor::F32(q_arr), Tensor::F32(k_arr), Tensor::F32(v_arr)) => {
                let mut output = ArrayD::zeros(IxDyn(&[batch_size, num_heads, seq_q, head_dim]));

                for b in 0..batch_size {
                    for h in 0..num_heads {
                        // Extract matrices for this batch and head
                        let q_batch = q_arr.index_axis(Axis(0), b);
                        let k_batch = k_arr.index_axis(Axis(0), b);
                        let v_batch = v_arr.index_axis(Axis(0), b);
                        let q_bh = q_batch.index_axis(Axis(0), h);
                        let k_bh = k_batch.index_axis(Axis(0), h);
                        let v_bh = v_batch.index_axis(Axis(0), h);

                        // Convert to owned 2D arrays for BLAS operations
                        let q_2d: Array2<f32> = q_bh
                            .to_owned()
                            .into_shape_with_order((seq_q, head_dim))
                            .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                        let k_2d: Array2<f32> = k_bh
                            .to_owned()
                            .into_shape_with_order((seq_k, head_dim))
                            .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                        let v_2d: Array2<f32> = v_bh
                            .to_owned()
                            .into_shape_with_order((seq_k, head_dim))
                            .map_err(|e| TrustformersError::shape_error(e.to_string()))?;

                        // Compute QK^T using BLAS gemm: Q @ K^T
                        // Q: [seq_q, head_dim], K^T: [head_dim, seq_k] => scores: [seq_q, seq_k]
                        let k_t = k_2d.t();
                        let k_t_owned: Array2<f32> = k_t.to_owned();

                        let scores = if seq_q >= MIN_SIZE_FOR_BLAS
                            && seq_k >= MIN_SIZE_FOR_BLAS
                            && head_dim >= MIN_SIZE_FOR_BLAS
                        {
                            // Use direct BLAS (Accelerate on macOS) for larger matrices
                            let q_vec: Vec<f32> = q_2d.iter().copied().collect();
                            let k_t_vec: Vec<f32> = k_t_owned.iter().copied().collect();
                            let mut result_vec = vec![0.0f32; seq_q * seq_k];
                            blas_sgemm(
                                scale,
                                &q_vec,
                                &k_t_vec,
                                0.0,
                                &mut result_vec,
                                seq_q,
                                head_dim,
                                seq_k,
                            );
                            Array2::from_shape_vec((seq_q, seq_k), result_vec).unwrap()
                        } else {
                            // Use ndarray dot for smaller matrices
                            let mut result = q_2d.dot(&k_t_owned);
                            result.mapv_inplace(|x| x * scale);
                            result
                        };

                        let mut scores = scores;

                        // Apply causal mask
                        if causal {
                            for i in 0..seq_q {
                                for j in i + 1..seq_k {
                                    scores[[i, j]] = f32::NEG_INFINITY;
                                }
                            }
                        }

                        // Apply attention mask if provided
                        if let Some(Tensor::F32(mask_arr)) = attn_mask {
                            let mask_batch = mask_arr.index_axis(Axis(0), b);
                            let mask_bh = mask_batch.index_axis(Axis(0), h);
                            for i in 0..seq_q {
                                for j in 0..seq_k {
                                    if mask_bh[[i, j]] == 0.0 {
                                        scores[[i, j]] = f32::NEG_INFINITY;
                                    }
                                }
                            }
                        }

                        // Softmax (row-wise) with SIMD optimization
                        if seq_k >= MIN_SIZE_FOR_SIMD_SOFTMAX && !causal && attn_mask.is_none() {
                            // Fast path: No masking, use SIMD softmax
                            for i in 0..seq_q {
                                let row = scores.row(i);
                                let softmax_row = simd_softmax_f32(&row);
                                for j in 0..seq_k {
                                    scores[[i, j]] = softmax_row[j];
                                }
                            }
                        } else {
                            // Standard path: Handle masking (NEG_INFINITY values)
                            for i in 0..seq_q {
                                let max_score =
                                    scores.row(i).fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                                let mut sum = 0.0f32;
                                for j in 0..seq_k {
                                    let exp_val = (scores[[i, j]] - max_score).exp();
                                    scores[[i, j]] = exp_val;
                                    sum += exp_val;
                                }
                                let inv_sum = 1.0 / sum.max(f32::MIN_POSITIVE);
                                for j in 0..seq_k {
                                    scores[[i, j]] *= inv_sum;
                                }
                            }
                        }

                        // Apply attention to values using BLAS gemm: scores @ V
                        // scores: [seq_q, seq_k], V: [seq_k, head_dim] => output: [seq_q, head_dim]
                        let attn_output = if seq_q >= MIN_SIZE_FOR_BLAS
                            && seq_k >= MIN_SIZE_FOR_BLAS
                            && head_dim >= MIN_SIZE_FOR_BLAS
                        {
                            // Use direct BLAS (Accelerate on macOS) for larger matrices
                            let scores_vec: Vec<f32> = scores.iter().copied().collect();
                            let v_vec: Vec<f32> = v_2d.iter().copied().collect();
                            let mut result_vec = vec![0.0f32; seq_q * head_dim];
                            blas_sgemm(
                                1.0,
                                &scores_vec,
                                &v_vec,
                                0.0,
                                &mut result_vec,
                                seq_q,
                                seq_k,
                                head_dim,
                            );
                            Array2::from_shape_vec((seq_q, head_dim), result_vec).unwrap()
                        } else {
                            // Use ndarray dot for smaller matrices
                            scores.dot(&v_2d)
                        };

                        // Copy to output
                        for i in 0..seq_q {
                            for d in 0..head_dim {
                                output[[b, h, i, d]] = attn_output[[i, d]];
                            }
                        }
                    }
                }

                Ok(Tensor::F32(output))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for SDPA",
                "SDPA::forward",
            )),
        }
    }

    /// Optimized SDPA for small sequences (≤512 tokens)
    /// Uses more aggressive optimizations and better cache locality
    fn small_sequence_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
        causal: bool,
    ) -> Result<Tensor> {
        let q_shape = q.shape();
        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let seq_q = q_shape[2];
        let head_dim = q_shape[3];

        let k_shape = k.shape();
        let seq_k = k_shape[2];

        let scale = 1.0 / (head_dim as f32).sqrt();

        match (q, k, v) {
            (Tensor::F32(q_arr), Tensor::F32(k_arr), Tensor::F32(v_arr)) => {
                let mut output = ArrayD::zeros(IxDyn(&[batch_size, num_heads, seq_q, head_dim]));

                for b in 0..batch_size {
                    for h in 0..num_heads {
                        // Extract and transpose for better cache locality
                        let q_batch = q_arr.index_axis(Axis(0), b);
                        let k_batch = k_arr.index_axis(Axis(0), b);
                        let v_batch = v_arr.index_axis(Axis(0), b);
                        let q_bh = q_batch.index_axis(Axis(0), h);
                        let k_bh = k_batch.index_axis(Axis(0), h);
                        let v_bh = v_batch.index_axis(Axis(0), h);

                        // Convert to owned 2D arrays for BLAS operations
                        let q_2d: Array2<f32> = q_bh
                            .to_owned()
                            .into_shape_with_order((seq_q, head_dim))
                            .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                        let k_2d: Array2<f32> = k_bh
                            .to_owned()
                            .into_shape_with_order((seq_k, head_dim))
                            .map_err(|e| TrustformersError::shape_error(e.to_string()))?;

                        // Compute QK^T using BLAS gemm (faster than blocked impl for any size)
                        let k_t = k_2d.t();
                        let k_t_owned: Array2<f32> = k_t.to_owned();

                        let mut scores = if seq_q >= MIN_SIZE_FOR_BLAS
                            && seq_k >= MIN_SIZE_FOR_BLAS
                            && head_dim >= MIN_SIZE_FOR_BLAS
                        {
                            // Use direct BLAS (Accelerate on macOS)
                            let q_vec: Vec<f32> = q_2d.iter().copied().collect();
                            let k_t_vec: Vec<f32> = k_t_owned.iter().copied().collect();
                            let mut result_vec = vec![0.0f32; seq_q * seq_k];
                            blas_sgemm(
                                scale,
                                &q_vec,
                                &k_t_vec,
                                0.0,
                                &mut result_vec,
                                seq_q,
                                head_dim,
                                seq_k,
                            );
                            Array2::from_shape_vec((seq_q, seq_k), result_vec).unwrap()
                        } else {
                            let mut result = q_2d.dot(&k_t_owned);
                            result.mapv_inplace(|x| x * scale);
                            result
                        };

                        // Apply masks and softmax (same as standard)
                        if causal {
                            for i in 0..seq_q {
                                for j in i + 1..seq_k {
                                    scores[[i, j]] = f32::NEG_INFINITY;
                                }
                            }
                        }

                        if let Some(Tensor::F32(mask_arr)) = attn_mask {
                            let mask_batch = mask_arr.index_axis(Axis(0), b);
                            let mask_bh = mask_batch.index_axis(Axis(0), h);
                            for i in 0..seq_q {
                                for j in 0..seq_k {
                                    if mask_bh[[i, j]] == 0.0 {
                                        scores[[i, j]] = f32::NEG_INFINITY;
                                    }
                                }
                            }
                        }

                        // Softmax (row-wise) with SIMD optimization
                        if seq_k >= MIN_SIZE_FOR_SIMD_SOFTMAX && !causal && attn_mask.is_none() {
                            // Fast path: No masking, use SIMD softmax
                            for i in 0..seq_q {
                                let row = scores.row(i);
                                let softmax_row = simd_softmax_f32(&row);
                                for j in 0..seq_k {
                                    scores[[i, j]] = softmax_row[j];
                                }
                            }
                        } else {
                            // Standard path: Handle masking
                            for i in 0..seq_q {
                                let max_score =
                                    scores.row(i).fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                                let mut sum = 0.0f32;
                                for j in 0..seq_k {
                                    let exp_val = (scores[[i, j]] - max_score).exp();
                                    scores[[i, j]] = exp_val;
                                    sum += exp_val;
                                }
                                let inv_sum = 1.0 / sum.max(f32::MIN_POSITIVE);
                                for j in 0..seq_k {
                                    scores[[i, j]] *= inv_sum;
                                }
                            }
                        }

                        // Apply attention to values using BLAS gemm: scores @ V
                        let v_2d: Array2<f32> = v_bh
                            .to_owned()
                            .into_shape_with_order((seq_k, head_dim))
                            .map_err(|e| TrustformersError::shape_error(e.to_string()))?;

                        let attn_output = if seq_q >= MIN_SIZE_FOR_BLAS
                            && seq_k >= MIN_SIZE_FOR_BLAS
                            && head_dim >= MIN_SIZE_FOR_BLAS
                        {
                            // Use direct BLAS (Accelerate on macOS)
                            let scores_vec: Vec<f32> = scores.iter().copied().collect();
                            let v_vec: Vec<f32> = v_2d.iter().copied().collect();
                            let mut result_vec = vec![0.0f32; seq_q * head_dim];
                            blas_sgemm(
                                1.0,
                                &scores_vec,
                                &v_vec,
                                0.0,
                                &mut result_vec,
                                seq_q,
                                seq_k,
                                head_dim,
                            );
                            Array2::from_shape_vec((seq_q, head_dim), result_vec).unwrap()
                        } else {
                            scores.dot(&v_2d)
                        };

                        // Copy to output
                        for i in 0..seq_q {
                            for d in 0..head_dim {
                                output[[b, h, i, d]] = attn_output[[i, d]];
                            }
                        }
                    }
                }

                Ok(Tensor::F32(output))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for small sequence SDPA",
                "SDPA::small_sequence_attention",
            )),
        }
    }

    /// Memory-efficient tiled SDPA for long sequences (>2048 tokens)
    /// Uses tiling to reduce memory complexity
    fn tiled_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
        causal: bool,
    ) -> Result<Tensor> {
        let q_shape = q.shape();
        let batch_size = q_shape[0];
        let num_heads = q_shape[1];
        let seq_q = q_shape[2];
        let head_dim = q_shape[3];

        let k_shape = k.shape();
        let seq_k = k_shape[2];

        let scale = 1.0 / (head_dim as f32).sqrt();

        // Tile size for memory efficiency
        let tile_size = 256;

        match (q, k, v) {
            (Tensor::F32(q_arr), Tensor::F32(k_arr), Tensor::F32(v_arr)) => {
                let mut output = ArrayD::zeros(IxDyn(&[batch_size, num_heads, seq_q, head_dim]));

                for b in 0..batch_size {
                    for h in 0..num_heads {
                        let q_batch = q_arr.index_axis(Axis(0), b);
                        let k_batch = k_arr.index_axis(Axis(0), b);
                        let v_batch = v_arr.index_axis(Axis(0), b);
                        let q_bh = q_batch.index_axis(Axis(0), h);
                        let k_bh = k_batch.index_axis(Axis(0), h);
                        let v_bh = v_batch.index_axis(Axis(0), h);

                        // Process in tiles to reduce memory usage
                        for q_start in (0..seq_q).step_by(tile_size) {
                            let q_end = (q_start + tile_size).min(seq_q);
                            let q_tile_size = q_end - q_start;

                            // Initialize tile outputs
                            let mut o_tile = Array2::<f32>::zeros((q_tile_size, head_dim));
                            let mut l_tile = Array1::<f32>::zeros(q_tile_size);
                            let mut m_tile =
                                Array1::<f32>::from_elem(q_tile_size, f32::NEG_INFINITY);

                            for k_start in (0..seq_k).step_by(tile_size) {
                                let k_end = (k_start + tile_size).min(seq_k);
                                let k_tile_size = k_end - k_start;

                                // Skip future tiles for causal attention
                                if causal && k_start >= q_end {
                                    break;
                                }

                                // Extract tiles
                                let q_tile = q_bh.slice(s![q_start..q_end, ..]).to_owned();
                                let k_tile = k_bh.slice(s![k_start..k_end, ..]).to_owned();
                                let v_tile = v_bh.slice(s![k_start..k_end, ..]).to_owned();

                                // Compute scores for this tile using BLAS gemm
                                let q_tile_2d: Array2<f32> = q_tile
                                    .into_shape_with_order((q_tile_size, head_dim))
                                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                                let k_tile_2d: Array2<f32> = k_tile
                                    .into_shape_with_order((k_tile_size, head_dim))
                                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                                let k_tile_t = k_tile_2d.t().to_owned();

                                let mut scores_tile = if q_tile_size >= MIN_SIZE_FOR_BLAS
                                    && k_tile_size >= MIN_SIZE_FOR_BLAS
                                    && head_dim >= MIN_SIZE_FOR_BLAS
                                {
                                    // Use direct BLAS (Accelerate on macOS)
                                    let q_vec: Vec<f32> = q_tile_2d.iter().copied().collect();
                                    let k_t_vec: Vec<f32> = k_tile_t.iter().copied().collect();
                                    let mut result_vec = vec![0.0f32; q_tile_size * k_tile_size];
                                    blas_sgemm(
                                        scale,
                                        &q_vec,
                                        &k_t_vec,
                                        0.0,
                                        &mut result_vec,
                                        q_tile_size,
                                        head_dim,
                                        k_tile_size,
                                    );
                                    Array2::from_shape_vec((q_tile_size, k_tile_size), result_vec)
                                        .unwrap()
                                } else {
                                    let mut result = q_tile_2d.dot(&k_tile_t);
                                    result.mapv_inplace(|x| x * scale);
                                    result
                                };

                                // Apply causal mask within tile
                                if causal {
                                    for i in 0..q_tile_size {
                                        for j in 0..k_tile_size {
                                            let global_q = q_start + i;
                                            let global_k = k_start + j;
                                            if global_q < global_k {
                                                scores_tile[[i, j]] = f32::NEG_INFINITY;
                                            }
                                        }
                                    }
                                }

                                // Apply mask if provided
                                if let Some(Tensor::F32(mask_arr)) = attn_mask {
                                    let mask_batch = mask_arr.index_axis(Axis(0), b);
                                    let mask_bh = mask_batch.index_axis(Axis(0), h);
                                    for i in 0..q_tile_size {
                                        for j in 0..k_tile_size {
                                            let global_q = q_start + i;
                                            let global_k = k_start + j;
                                            if mask_bh[[global_q, global_k]] == 0.0 {
                                                scores_tile[[i, j]] = f32::NEG_INFINITY;
                                            }
                                        }
                                    }
                                }

                                // Online softmax update (similar to FlashAttention)
                                let m_new = scores_tile.fold_axis(
                                    Axis(1),
                                    f32::NEG_INFINITY,
                                    |&acc, &x| acc.max(x),
                                );
                                let m_prev = m_tile.clone();
                                let m_combined = Array1::<f32>::from_shape_fn(q_tile_size, |i| {
                                    m_tile[i].max(m_new[i])
                                });

                                let mut exp_scores =
                                    Array2::<f32>::zeros((q_tile_size, k_tile_size));
                                for i in 0..q_tile_size {
                                    for j in 0..k_tile_size {
                                        exp_scores[[i, j]] =
                                            (scores_tile[[i, j]] - m_combined[i]).exp();
                                    }
                                }

                                let exp_prev = Array1::<f32>::from_shape_fn(q_tile_size, |i| {
                                    (m_prev[i] - m_combined[i]).exp()
                                });

                                // Update denominators
                                let l_new = exp_scores.sum_axis(Axis(1));
                                for i in 0..q_tile_size {
                                    l_tile[i] = l_tile[i] * exp_prev[i] + l_new[i];
                                }

                                // Update outputs
                                for i in 0..q_tile_size {
                                    for d in 0..head_dim {
                                        o_tile[[i, d]] *= exp_prev[i];
                                    }
                                }

                                // Add new contribution using BLAS gemm: exp_scores @ V
                                let v_tile_2d: Array2<f32> = v_tile
                                    .into_shape_with_order((k_tile_size, head_dim))
                                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;

                                if q_tile_size >= MIN_SIZE_FOR_BLAS
                                    && k_tile_size >= MIN_SIZE_FOR_BLAS
                                    && head_dim >= MIN_SIZE_FOR_BLAS
                                {
                                    // Use direct BLAS with beta=1.0 to add to existing o_tile
                                    let exp_vec: Vec<f32> = exp_scores.iter().copied().collect();
                                    let v_vec: Vec<f32> = v_tile_2d.iter().copied().collect();
                                    let o_slice = o_tile.as_slice_mut().unwrap();
                                    blas_sgemm(
                                        1.0,
                                        &exp_vec,
                                        &v_vec,
                                        1.0,
                                        o_slice,
                                        q_tile_size,
                                        k_tile_size,
                                        head_dim,
                                    );
                                } else {
                                    // Fallback to ndarray dot for small tiles
                                    let new_contrib = exp_scores.dot(&v_tile_2d);
                                    for i in 0..q_tile_size {
                                        for d in 0..head_dim {
                                            o_tile[[i, d]] += new_contrib[[i, d]];
                                        }
                                    }
                                }

                                m_tile = m_combined;
                            }

                            // Normalize and store tile output
                            for i in 0..q_tile_size {
                                let inv_l = if l_tile[i] > 0.0 { 1.0 / l_tile[i] } else { 0.0 };
                                for d in 0..head_dim {
                                    output[[b, h, q_start + i, d]] = o_tile[[i, d]] * inv_l;
                                }
                            }
                        }
                    }
                }

                Ok(Tensor::F32(output))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for tiled SDPA",
                "SDPA::tiled_attention",
            )),
        }
    }

    /// Fused SDPA kernel that combines attention computation with common post-processing
    pub fn fused_attention_dropout(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attn_mask: Option<&Tensor>,
        causal: bool,
        dropout_prob: f32,
        training: bool,
    ) -> Result<Tensor> {
        // For now, just use standard attention (would add dropout in actual implementation)
        let _ = (dropout_prob, training); // Suppress unused warnings
        Self::attention(q, k, v, attn_mask, causal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_standard_attention() {
        let q = Tensor::randn(&[2, 4, 32, 64]).unwrap();
        let k = Tensor::randn(&[2, 4, 32, 64]).unwrap();
        let v = Tensor::randn(&[2, 4, 32, 64]).unwrap();

        let output = SDPA::attention(&q, &k, &v, None, false);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.shape(), vec![2, 4, 32, 64]);
    }

    #[test]
    fn test_small_sequence_attention() {
        let q = Tensor::randn(&[1, 8, 128, 64]).unwrap();
        let k = Tensor::randn(&[1, 8, 128, 64]).unwrap();
        let v = Tensor::randn(&[1, 8, 128, 64]).unwrap();

        let output = SDPA::small_sequence_attention(&q, &k, &v, None, false);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.shape(), vec![1, 8, 128, 64]);
    }

    #[test]
    fn test_tiled_attention() {
        let q = Tensor::randn(&[1, 4, 512, 64]).unwrap();
        let k = Tensor::randn(&[1, 4, 512, 64]).unwrap();
        let v = Tensor::randn(&[1, 4, 512, 64]).unwrap();

        let output = SDPA::tiled_attention(&q, &k, &v, None, false);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.shape(), vec![1, 4, 512, 64]);
    }

    #[test]
    fn test_causal_attention() {
        let q = Tensor::randn(&[1, 2, 16, 32]).unwrap();
        let k = Tensor::randn(&[1, 2, 16, 32]).unwrap();
        let v = Tensor::randn(&[1, 2, 16, 32]).unwrap();

        let output = SDPA::attention(&q, &k, &v, None, true);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.shape(), vec![1, 2, 16, 32]);
    }

    #[test]
    fn test_attention_with_mask() {
        let q = Tensor::randn(&[1, 2, 16, 32]).unwrap();
        let k = Tensor::randn(&[1, 2, 16, 32]).unwrap();
        let v = Tensor::randn(&[1, 2, 16, 32]).unwrap();
        let mask = Tensor::ones(&[1, 2, 16, 16]).unwrap();

        let output = SDPA::attention(&q, &k, &v, Some(&mask), false);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.shape(), vec![1, 2, 16, 32]);
    }

    #[test]
    fn test_fused_attention_dropout() {
        let q = Tensor::randn(&[1, 4, 64, 32]).unwrap();
        let k = Tensor::randn(&[1, 4, 64, 32]).unwrap();
        let v = Tensor::randn(&[1, 4, 64, 32]).unwrap();

        let output = SDPA::fused_attention_dropout(&q, &k, &v, None, false, 0.1, true);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.shape(), vec![1, 4, 64, 32]);
    }
}
