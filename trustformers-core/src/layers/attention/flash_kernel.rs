//! Block-tiled FlashAttention kernel.
//!
//! This is the single numerical core shared by every FlashAttention entry
//! point in the crate ([`super::flash::FlashAttention`],
//! [`super::multi_head::MultiHeadAttention`]'s memory-efficient path and
//! [`crate::layers::flash_attention::FlashAttention`]).
//!
//! # Algorithm
//!
//! The kernel implements the exact algorithm of *FlashAttention: Fast and
//! Memory-Efficient Exact Attention with IO-Awareness*
//! (<https://arxiv.org/abs/2205.14135>) with the FlashAttention-2 style
//! deferred normalisation (<https://arxiv.org/abs/2307.08691>):
//!
//! For every query block `Q_i` the kernel keeps three running statistics that
//! are **per query row**, never global:
//!
//! * `m_i` - the running row maximum of the scores seen so far,
//! * `l_i` - the running row sum of `exp(score - m_i)`,
//! * `O_i` - the running unnormalised output accumulator.
//!
//! For each key block `K_j`/`V_j` it computes `S = scale * Q_i K_j^T`, applies
//! causal and user masking *inside the block*, then updates
//!
//! ```text
//! m_new = max(m_i, rowmax(S))
//! alpha = exp(m_i - m_new)                    (0 when m_i = -inf)
//! P     = exp(S - m_new)                      (0 for masked entries)
//! l_i   = alpha * l_i + rowsum(P)
//! O_i   = alpha * O_i + P V_j
//! m_i   = m_new
//! ```
//!
//! and divides `O_i` by `l_i` once, after the last key block. The full
//! `seq_q x seq_k` score matrix is never materialised: peak extra memory is
//! `block_q * block_k` per worker.
//!
//! # Parallelism
//!
//! Work is partitioned over the flattened `(batch, head)` index space with
//! `scirs2_core::parallel_ops` (rayon); each worker writes into its own
//! disjoint, contiguous slice of the output buffer.

use scirs2_core::ndarray::{s, ArrayD, ArrayView2, Axis, CowArray, Ix2, IxDyn};
use scirs2_core::parallel_ops::*;

use crate::errors::{Result, TrustformersError};
use crate::layers::sdpa::{blas_sgemm, blas_sgemm_nt};
use crate::tensor::Tensor;

use super::mask::MaskView;

/// Tiling and masking parameters for [`flash_attention`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct FlashParams {
    /// Scaling applied to `Q K^T` (normally `1 / sqrt(head_dim)`).
    pub scale: f32,
    /// Whether future key positions must be masked out.
    pub causal: bool,
    /// Number of query rows processed per tile.
    pub block_q: usize,
    /// Number of key/value rows processed per tile.
    pub block_k: usize,
    /// Post-softmax attention dropout probability, applied only when `Some`.
    ///
    /// Dropout is applied to the attention probabilities of each tile while the
    /// running denominator keeps accumulating the *undropped* weights, which
    /// makes the result identical to dropping the fully normalised attention
    /// matrix.
    pub dropout: Option<f32>,
}

impl FlashParams {
    /// Parameters for a square tiling with the canonical `1/sqrt(d)` scale.
    pub(crate) fn new(head_dim: usize, causal: bool, block_size: usize) -> Self {
        Self {
            scale: 1.0 / (head_dim.max(1) as f32).sqrt(),
            causal,
            block_q: block_size,
            block_k: block_size,
            dropout: None,
        }
    }

    /// Enable post-softmax attention dropout with the given probability.
    ///
    /// A probability outside `[0, 1)` is rejected; `0.0` disables dropout.
    pub(crate) fn with_dropout(mut self, dropout_prob: Option<f32>) -> Result<Self> {
        match dropout_prob {
            Some(prob) if !(0.0..1.0).contains(&prob) => Err(TrustformersError::tensor_op_error(
                &format!("Attention dropout probability must be in [0, 1), got {prob}"),
                "FlashParams::with_dropout",
            )),
            other => {
                self.dropout = other.filter(|&prob| prob > 0.0);
                Ok(self)
            },
        }
    }

    /// Clamp the tile sizes into a usable range for the given problem.
    fn sanitized(self, seq_q: usize, seq_k: usize) -> Self {
        Self {
            block_q: self.block_q.clamp(1, seq_q.max(1)),
            block_k: self.block_k.clamp(1, seq_k.max(1)),
            ..self
        }
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

/// Shapes of a validated attention problem.
struct AttentionShape {
    batch: usize,
    heads: usize,
    seq_q: usize,
    seq_k: usize,
    head_dim: usize,
}

fn f32_array<'a>(tensor: &'a Tensor, name: &str) -> Result<&'a ArrayD<f32>> {
    match tensor {
        Tensor::F32(array) => Ok(array),
        other => Err(TrustformersError::tensor_op_error(
            &format!(
                "FlashAttention currently supports F32 tensors only; {} has dtype {:?}",
                name,
                other.dtype()
            ),
            "flash_attention",
        )),
    }
}

fn validate(q: &ArrayD<f32>, k: &ArrayD<f32>, v: &ArrayD<f32>) -> Result<AttentionShape> {
    for (name, array) in [("query", q), ("key", k), ("value", v)] {
        if array.ndim() != 4 {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "FlashAttention expects 4-D [batch, heads, seq, head_dim] tensors; {} has shape {:?}",
                    name,
                    array.shape()
                ),
                "flash_attention",
            ));
        }
    }

    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();

    if k_shape[0] != q_shape[0] || v_shape[0] != q_shape[0] {
        return Err(TrustformersError::tensor_op_error(
            &format!(
                "Q/K/V batch sizes must match, got {:?} / {:?} / {:?}",
                q_shape, k_shape, v_shape
            ),
            "flash_attention",
        ));
    }
    if k_shape[1] != q_shape[1] || v_shape[1] != q_shape[1] {
        return Err(TrustformersError::tensor_op_error(
            &format!(
                "Q/K/V head counts must match (expand grouped KV heads before calling), got {:?} / {:?} / {:?}",
                q_shape, k_shape, v_shape
            ),
            "flash_attention",
        ));
    }
    if k_shape[2] != v_shape[2] {
        return Err(TrustformersError::tensor_op_error(
            &format!(
                "Key and value sequence lengths must match, got {} and {}",
                k_shape[2], v_shape[2]
            ),
            "flash_attention",
        ));
    }
    if k_shape[3] != q_shape[3] || v_shape[3] != q_shape[3] {
        return Err(TrustformersError::tensor_op_error(
            &format!(
                "Q/K/V head dimensions must match, got {} / {} / {}",
                q_shape[3], k_shape[3], v_shape[3]
            ),
            "flash_attention",
        ));
    }

    Ok(AttentionShape {
        batch: q_shape[0],
        heads: q_shape[1],
        seq_q: q_shape[2],
        seq_k: k_shape[2],
        head_dim: q_shape[3],
    })
}

/// Extract `[seq, head_dim]` for one `(batch, head)` pair of a
/// `[batch, heads, seq, head_dim]` tensor as a contiguous row-major matrix.
///
/// Shared with [`crate::layers::sdpa`]: attention inputs frequently arrive as
/// non-standard-layout views (e.g. after `permuted_axes`), so the per-head
/// matrix has to be materialised before it can be handed to GEMM.
pub(crate) fn head_matrix<'a>(
    array: &'a ArrayD<f32>,
    batch: usize,
    head: usize,
) -> Result<CowArray<'a, f32, Ix2>> {
    let view: ArrayView2<'a, f32> = array
        .index_axis(Axis(0), batch)
        .index_axis_move(Axis(0), head)
        .into_dimensionality::<Ix2>()
        .map_err(|e| TrustformersError::shape_error(format!("Expected a 2-D head slice: {e}")))?;

    if view.is_standard_layout() {
        // Already contiguous row-major - borrow it, no copy at all.
        Ok(CowArray::from(view))
    } else {
        // Permuted views (`split_heads` uses `permuted_axes`) are not
        // contiguous; `map` materialises them in row-major order, which is what
        // makes the later `as_slice()` calls succeed.
        Ok(CowArray::from(view.map(|&value| value)))
    }
}

/// Exact block-tiled scaled dot-product attention.
///
/// * `q` - `[batch, heads, seq_q, head_dim]`
/// * `k`, `v` - `[batch, heads, seq_k, head_dim]`
/// * `mask` - optional attention mask, see [`super::mask`]
///
/// Returns `[batch, heads, seq_q, head_dim]`. Rows whose keys are entirely
/// masked out produce zeros (their softmax is undefined).
pub(crate) fn flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    params: &FlashParams,
) -> Result<Tensor> {
    let q_array = f32_array(q, "query")?;
    let k_array = f32_array(k, "key")?;
    let v_array = f32_array(v, "value")?;

    let shape = validate(q_array, k_array, v_array)?;
    let AttentionShape {
        batch,
        heads,
        seq_q,
        seq_k,
        head_dim,
    } = shape;

    let params = params.sanitized(seq_q, seq_k);

    let mask_view = match mask {
        Some(mask) => Some(MaskView::new(mask, batch, heads, seq_q, seq_k)?),
        None => None,
    };

    let mut output = vec![0.0f32; batch * heads * seq_q * head_dim];
    if output.is_empty() {
        return Ok(Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq_q, head_dim]), output)
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        ));
    }

    let head_stride = seq_q * head_dim;
    output.par_chunks_mut(head_stride).enumerate().try_for_each(
        |(flat_index, out_chunk)| -> Result<()> {
            let batch_index = flat_index / heads;
            let head_index = flat_index % heads;
            let q_head = head_matrix(q_array, batch_index, head_index)?;
            let k_head = head_matrix(k_array, batch_index, head_index)?;
            let v_head = head_matrix(v_array, batch_index, head_index)?;
            flash_attention_head(
                &q_head,
                &k_head,
                &v_head,
                mask_view.as_ref(),
                batch_index,
                head_index,
                &params,
                out_chunk,
            )
        },
    )?;

    Ok(Tensor::F32(
        ArrayD::from_shape_vec(IxDyn(&[batch, heads, seq_q, head_dim]), output)
            .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
    ))
}

/// FlashAttention for a single `(batch, head)` pair.
///
/// `out` is the `[seq_q, head_dim]` row-major destination slice.
#[allow(clippy::too_many_arguments)]
fn flash_attention_head(
    q_head: &CowArray<'_, f32, Ix2>,
    k_head: &CowArray<'_, f32, Ix2>,
    v_head: &CowArray<'_, f32, Ix2>,
    mask: Option<&MaskView<'_>>,
    batch_index: usize,
    head_index: usize,
    params: &FlashParams,
    out: &mut [f32],
) -> Result<()> {
    let seq_q = q_head.shape()[0];
    let seq_k = k_head.shape()[0];
    let head_dim = q_head.shape()[1];

    // Flat, contiguous scratch buffers so every tile can be handed straight to
    // GEMM without a copy. `scores` holds the current `q_rows x k_rows` block,
    // `accumulator` the unnormalised `q_rows x head_dim` output block.
    let mut scores = vec![0.0f32; params.block_q * params.block_k];
    let mut accumulator = vec![0.0f32; params.block_q * head_dim];
    let mut row_max = vec![f32::NEG_INFINITY; params.block_q];
    let mut row_sum = vec![0.0f32; params.block_q];

    for q_start in (0..seq_q).step_by(params.block_q) {
        let q_end = (q_start + params.block_q).min(seq_q);
        let q_rows = q_end - q_start;

        row_max[..q_rows].fill(f32::NEG_INFINITY);
        row_sum[..q_rows].fill(0.0);
        accumulator[..q_rows * head_dim].fill(0.0);

        let q_block = q_head.slice(s![q_start..q_end, ..]);
        let q_data = q_block.as_slice().ok_or_else(|| {
            TrustformersError::tensor_op_error(
                "query block must be contiguous for GEMM",
                "flash_attention",
            )
        })?;

        for k_start in (0..seq_k).step_by(params.block_k) {
            let k_end = (k_start + params.block_k).min(seq_k);
            let k_rows = k_end - k_start;

            // Every key in this block is strictly in the future of every query
            // in the current block: nothing to accumulate.
            if params.causal && k_start >= q_end {
                break;
            }

            let k_block = k_head.slice(s![k_start..k_end, ..]);
            let v_block = v_head.slice(s![k_start..k_end, ..]);
            let k_data = k_block.as_slice().ok_or_else(|| {
                TrustformersError::tensor_op_error(
                    "key block must be contiguous for GEMM",
                    "flash_attention",
                )
            })?;
            let v_data = v_block.as_slice().ok_or_else(|| {
                TrustformersError::tensor_op_error(
                    "value block must be contiguous for GEMM",
                    "flash_attention",
                )
            })?;

            let score_block = &mut scores[..q_rows * k_rows];

            // S = scale * Q_block @ K_block^T, without materialising K^T.
            blas_sgemm_nt(
                params.scale,
                q_data,
                k_data,
                0.0,
                score_block,
                q_rows,
                head_dim,
                k_rows,
            );

            // Masking happens inside the block, on global positions.
            if params.causal {
                for row in 0..q_rows {
                    let allowed = (q_start + row + 1).saturating_sub(k_start).min(k_rows);
                    if allowed < k_rows {
                        score_block[row * k_rows + allowed..(row + 1) * k_rows]
                            .fill(f32::NEG_INFINITY);
                    }
                }
            }
            if let Some(mask) = mask {
                for row in 0..q_rows {
                    let query_pos = q_start + row;
                    for column in 0..k_rows {
                        let penalty =
                            mask.additive(batch_index, head_index, query_pos, k_start + column);
                        if penalty != 0.0 {
                            score_block[row * k_rows + column] += penalty;
                        }
                    }
                }
            }

            // Online softmax: per-row max, per-row sum, rescale the accumulator.
            for row in 0..q_rows {
                let row_slice = &mut score_block[row * k_rows..(row + 1) * k_rows];
                let block_max = row_slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let previous_max = row_max[row];
                let combined_max = previous_max.max(block_max);

                if combined_max == f32::NEG_INFINITY {
                    // Every key seen so far is masked out for this row: the
                    // block contributes nothing and `exp(-inf - -inf)` is NaN,
                    // so zero the probabilities explicitly.
                    row_slice.fill(0.0);
                    continue;
                }

                let correction = if previous_max == f32::NEG_INFINITY {
                    0.0
                } else {
                    (previous_max - combined_max).exp()
                };

                let mut block_sum = 0.0f32;
                for weight in row_slice.iter_mut() {
                    let exponential = (*weight - combined_max).exp();
                    *weight = exponential;
                    block_sum += exponential;
                }

                // The denominator uses the undropped weights so that scaling
                // the numerator below yields exactly post-softmax dropout.
                row_sum[row] = row_sum[row] * correction + block_sum;
                row_max[row] = combined_max;

                if let Some(prob) = params.dropout {
                    apply_attention_dropout(row_slice, prob);
                }

                if correction != 1.0 {
                    for slot in accumulator[row * head_dim..(row + 1) * head_dim].iter_mut() {
                        *slot *= correction;
                    }
                }
            }

            // O_block += P @ V_block
            blas_sgemm(
                1.0,
                score_block,
                v_data,
                1.0,
                &mut accumulator[..q_rows * head_dim],
                q_rows,
                k_rows,
                head_dim,
            );
        }

        // Deferred normalisation: divide by the final per-row sum exactly once.
        for row in 0..q_rows {
            let inverse = if row_sum[row] > 0.0 { 1.0 / row_sum[row] } else { 0.0 };
            let source = &accumulator[row * head_dim..(row + 1) * head_dim];
            let destination = &mut out[(q_start + row) * head_dim..(q_start + row + 1) * head_dim];
            for (slot, &value) in destination.iter_mut().zip(source.iter()) {
                *slot = value * inverse;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::IxDyn;

    /// Straightforward `softmax(scale * Q K^T + mask) V` reference.
    fn naive_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        causal: bool,
        scale: f32,
    ) -> Vec<f32> {
        let (Tensor::F32(q_array), Tensor::F32(k_array), Tensor::F32(v_array)) = (q, k, v) else {
            panic!("reference implementation requires F32 tensors");
        };
        let batch = q_array.shape()[0];
        let heads = q_array.shape()[1];
        let seq_q = q_array.shape()[2];
        let head_dim = q_array.shape()[3];
        let seq_k = k_array.shape()[2];
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
                            dot += q_array[[b, h, i, d]] * k_array[[b, h, j, d]];
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
                            acc += weight * v_array[[b, h, j, d]];
                        }
                        out[((b * heads + h) * seq_q + i) * head_dim + d] = acc / sum;
                    }
                }
            }
        }
        out
    }

    /// Deterministic pseudo-random tensor (no RNG dependency in assertions).
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

    fn max_abs_difference(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "output length mismatch");
        a.iter().zip(b.iter()).fold(0.0f32, |acc, (x, y)| acc.max((x - y).abs()))
    }

    fn check_against_reference(
        batch: usize,
        heads: usize,
        seq_q: usize,
        seq_k: usize,
        head_dim: usize,
        block_size: usize,
        causal: bool,
        mask: Option<Tensor>,
    ) {
        let q = deterministic(&[batch, heads, seq_q, head_dim], 1);
        let k = deterministic(&[batch, heads, seq_k, head_dim], 2);
        let v = deterministic(&[batch, heads, seq_k, head_dim], 3);
        let scale = 1.0 / (head_dim as f32).sqrt();
        let params = FlashParams {
            scale,
            causal,
            block_q: block_size,
            block_k: block_size,
            dropout: None,
        };

        let actual = flash_attention(&q, &k, &v, mask.as_ref(), &params)
            .expect("flash attention must succeed");
        let Tensor::F32(actual_array) = &actual else {
            panic!("flash attention must return F32");
        };
        assert_eq!(
            actual_array.shape(),
            &[batch, heads, seq_q, head_dim],
            "unexpected output shape"
        );

        let expected = naive_attention(&q, &k, &v, mask.as_ref(), causal, scale);
        let actual_data: Vec<f32> = actual_array.iter().copied().collect();
        let difference = max_abs_difference(&actual_data, &expected);
        assert!(
            difference < 1e-4,
            "flash attention deviates from the naive reference by {difference} \
             (batch={batch}, heads={heads}, seq_q={seq_q}, seq_k={seq_k}, \
              head_dim={head_dim}, block={block_size}, causal={causal})"
        );
    }

    #[test]
    fn matches_reference_non_causal_block_aligned() {
        check_against_reference(2, 3, 16, 16, 8, 8, false, None);
    }

    #[test]
    fn matches_reference_non_causal_ragged_blocks() {
        // Neither seq_q nor seq_k is a multiple of the block size.
        check_against_reference(2, 2, 17, 23, 6, 5, false, None);
    }

    #[test]
    fn matches_reference_causal_block_aligned() {
        check_against_reference(1, 2, 12, 12, 4, 4, true, None);
    }

    #[test]
    fn matches_reference_causal_ragged_blocks() {
        check_against_reference(2, 3, 19, 19, 7, 6, true, None);
    }

    #[test]
    fn matches_reference_cross_attention_shapes() {
        // seq_q != seq_k, single block covering everything.
        check_against_reference(1, 1, 5, 11, 4, 64, false, None);
    }

    #[test]
    fn matches_reference_with_large_blocks() {
        check_against_reference(1, 4, 33, 33, 16, 128, true, None);
    }

    #[test]
    fn matches_reference_with_keep_mask() {
        let seq_k = 13;
        let mut data = vec![1.0f32; seq_k];
        data[3] = 0.0;
        data[7] = 0.0;
        data[11] = 0.0;
        let mask = Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(&[1, 1, 1, seq_k]), data).expect("valid mask shape"),
        );
        check_against_reference(2, 2, 9, seq_k, 8, 4, false, Some(mask));
    }

    #[test]
    fn matches_reference_with_additive_mask() {
        let seq_q = 9;
        let seq_k = 9;
        let mut data = vec![0.0f32; seq_q * seq_k];
        for (index, slot) in data.iter_mut().enumerate() {
            if (index / seq_k + index % seq_k) % 4 == 0 {
                *slot = -1.0e9;
            }
        }
        let mask = Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(&[seq_q, seq_k]), data).expect("valid mask shape"),
        );
        check_against_reference(1, 2, seq_q, seq_k, 8, 3, false, Some(mask));
    }

    #[test]
    fn causal_output_ignores_future_tokens() {
        // Regression test for the block causal mask that used to be a no-op:
        // perturbing key/value rows in the future must not change earlier
        // query outputs.
        let seq = 12;
        let head_dim = 4;
        let q = deterministic(&[1, 1, seq, head_dim], 11);
        let k = deterministic(&[1, 1, seq, head_dim], 12);
        let v = deterministic(&[1, 1, seq, head_dim], 13);
        let params = FlashParams::new(head_dim, true, 5);

        let base = flash_attention(&q, &k, &v, None, &params).expect("baseline must succeed");

        let (Tensor::F32(k_array), Tensor::F32(v_array)) = (&k, &v) else {
            panic!("expected F32 tensors");
        };
        let mut k_perturbed = k_array.clone();
        let mut v_perturbed = v_array.clone();
        for position in 6..seq {
            for dim in 0..head_dim {
                k_perturbed[[0, 0, position, dim]] += 3.5;
                v_perturbed[[0, 0, position, dim]] -= 2.25;
            }
        }
        let perturbed = flash_attention(
            &q,
            &Tensor::F32(k_perturbed),
            &Tensor::F32(v_perturbed),
            None,
            &params,
        )
        .expect("perturbed run must succeed");

        let (Tensor::F32(base_array), Tensor::F32(perturbed_array)) = (&base, &perturbed) else {
            panic!("expected F32 outputs");
        };
        for position in 0..6 {
            for dim in 0..head_dim {
                let difference = (base_array[[0, 0, position, dim]]
                    - perturbed_array[[0, 0, position, dim]])
                .abs();
                assert!(
                    difference < 1e-5,
                    "causal attention leaked future information at position {position}"
                );
            }
        }
        // Positions after the perturbation must actually change, otherwise the
        // test would also pass for an implementation that ignores K/V.
        let mut changed = 0.0f32;
        for position in 6..seq {
            for dim in 0..head_dim {
                changed = changed.max(
                    (base_array[[0, 0, position, dim]] - perturbed_array[[0, 0, position, dim]])
                        .abs(),
                );
            }
        }
        assert!(changed > 1e-3, "perturbation had no effect at all");
    }

    #[test]
    fn output_depends_on_the_input() {
        // Regression test for the all-zeros implementation.
        let params = FlashParams::new(8, false, 4);
        let k = deterministic(&[1, 2, 6, 8], 21);
        let v = deterministic(&[1, 2, 6, 8], 22);
        let first = flash_attention(&deterministic(&[1, 2, 6, 8], 23), &k, &v, None, &params)
            .expect("first run must succeed");
        let second = flash_attention(&deterministic(&[1, 2, 6, 8], 24), &k, &v, None, &params)
            .expect("second run must succeed");
        let first_data = first.data().expect("data must be readable");
        let second_data = second.data().expect("data must be readable");
        assert!(
            first_data.iter().any(|&x| x.abs() > 1e-6),
            "output must not be all zeros"
        );
        assert!(
            max_abs_difference(&first_data, &second_data) > 1e-3,
            "output must depend on the query input"
        );
    }

    #[test]
    fn rejects_non_f32_inputs() {
        let q = Tensor::zeros_f64(&[1, 1, 2, 2]).expect("f64 tensor");
        let k = Tensor::zeros_f64(&[1, 1, 2, 2]).expect("f64 tensor");
        let v = Tensor::zeros_f64(&[1, 1, 2, 2]).expect("f64 tensor");
        let params = FlashParams::new(2, false, 2);
        assert!(flash_attention(&q, &k, &v, None, &params).is_err());
    }

    #[test]
    fn rejects_mismatched_shapes() {
        let q = deterministic(&[1, 2, 4, 8], 31);
        let k = deterministic(&[1, 2, 4, 4], 32);
        let v = deterministic(&[1, 2, 4, 4], 33);
        let params = FlashParams::new(8, false, 4);
        assert!(flash_attention(&q, &k, &v, None, &params).is_err());
    }
}
