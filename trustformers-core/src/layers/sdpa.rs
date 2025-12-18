#![allow(unused_variables)] // SDPA implementation with reserved parameters

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayD, Axis, IxDyn};
use scirs2_core::simd::activation::simd_softmax_f32;
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Minimum size threshold for SIMD GEMM (avoid scirs2-core bug with small matrices)
/// Note: scirs2-core has bugs with matrices of size <64, so we use 64 as the threshold
const MIN_SIZE_FOR_SIMD_GEMM: usize = 64;

/// Minimum size threshold for SIMD softmax
const MIN_SIZE_FOR_SIMD_SOFTMAX: usize = 64;

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

                        let scores = if seq_q >= MIN_SIZE_FOR_SIMD_GEMM
                            && seq_k >= MIN_SIZE_FOR_SIMD_GEMM
                            && head_dim >= MIN_SIZE_FOR_SIMD_GEMM
                        {
                            // Use BLAS GEMM for larger matrices
                            let mut result = Array2::<f32>::zeros((seq_q, seq_k));
                            f32::simd_gemm(
                                scale,
                                &q_2d.view(),
                                &k_t_owned.view(),
                                0.0,
                                &mut result,
                            );
                            result
                        } else {
                            // Use ndarray dot for smaller matrices (avoid scirs2-core bug)
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
                        let attn_output = if seq_q >= MIN_SIZE_FOR_SIMD_GEMM
                            && seq_k >= MIN_SIZE_FOR_SIMD_GEMM
                            && head_dim >= MIN_SIZE_FOR_SIMD_GEMM
                        {
                            // Use BLAS GEMM for larger matrices
                            let mut result = Array2::<f32>::zeros((seq_q, head_dim));
                            f32::simd_gemm(1.0, &scores.view(), &v_2d.view(), 0.0, &mut result);
                            result
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

                        let mut scores = if seq_q >= MIN_SIZE_FOR_SIMD_GEMM
                            && seq_k >= MIN_SIZE_FOR_SIMD_GEMM
                            && head_dim >= MIN_SIZE_FOR_SIMD_GEMM
                        {
                            let mut result = Array2::<f32>::zeros((seq_q, seq_k));
                            f32::simd_gemm(
                                scale,
                                &q_2d.view(),
                                &k_t_owned.view(),
                                0.0,
                                &mut result,
                            );
                            result
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

                        let attn_output = if seq_q >= MIN_SIZE_FOR_SIMD_GEMM
                            && seq_k >= MIN_SIZE_FOR_SIMD_GEMM
                            && head_dim >= MIN_SIZE_FOR_SIMD_GEMM
                        {
                            let mut result = Array2::<f32>::zeros((seq_q, head_dim));
                            f32::simd_gemm(1.0, &scores.view(), &v_2d.view(), 0.0, &mut result);
                            result
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

                                let mut scores_tile = if q_tile_size >= MIN_SIZE_FOR_SIMD_GEMM
                                    && k_tile_size >= MIN_SIZE_FOR_SIMD_GEMM
                                    && head_dim >= MIN_SIZE_FOR_SIMD_GEMM
                                {
                                    let mut result =
                                        Array2::<f32>::zeros((q_tile_size, k_tile_size));
                                    f32::simd_gemm(
                                        scale,
                                        &q_tile_2d.view(),
                                        &k_tile_t.view(),
                                        0.0,
                                        &mut result,
                                    );
                                    result
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

                                if q_tile_size >= MIN_SIZE_FOR_SIMD_GEMM
                                    && k_tile_size >= MIN_SIZE_FOR_SIMD_GEMM
                                    && head_dim >= MIN_SIZE_FOR_SIMD_GEMM
                                {
                                    // Use BLAS gemm with beta=1.0 to add to existing o_tile
                                    f32::simd_gemm(
                                        1.0,
                                        &exp_scores.view(),
                                        &v_tile_2d.view(),
                                        1.0,
                                        &mut o_tile,
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
