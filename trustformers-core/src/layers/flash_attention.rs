//! FlashAttention and its multi-query / grouped-query variants.
//!
//! All three layers share the block-tiled kernel in
//! [`crate::layers::attention::flash_kernel`]: the full `seq_q x seq_k`
//! attention matrix is never materialised and the online softmax keeps
//! per-query-row statistics.

use crate::errors::{Result, TrustformersError};
use crate::layers::attention::flash_kernel::{flash_attention, FlashParams};
use crate::layers::Linear;
use crate::tensor::Tensor;
use crate::traits::Layer;
use scirs2_core::ndarray::{ArrayD, Axis, IxDyn};

/// Split `[batch, seq_len, num_heads * head_dim]` into
/// `[batch, num_heads, seq_len, head_dim]`.
fn split_heads(tensor: &Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    let shape = tensor.shape();
    if shape.len() != 3 {
        return Err(TrustformersError::tensor_op_error(
            &format!(
                "Input tensor must have 3 dimensions for split_heads, got {}",
                shape.len()
            ),
            "flash_attention::split_heads",
        ));
    }

    match tensor {
        Tensor::F32(arr) => {
            let batch_size = shape[0];
            let seq_len = shape[1];
            if shape[2] != num_heads * head_dim {
                return Err(TrustformersError::tensor_op_error(
                    &format!(
                        "hidden size {} must equal num_heads * head_dim ({})",
                        shape[2],
                        num_heads * head_dim
                    ),
                    "flash_attention::split_heads",
                ));
            }

            let reshaped = arr
                .to_shape(IxDyn(&[batch_size, seq_len, num_heads, head_dim]))
                .map_err(|_| {
                    TrustformersError::shape_error("Failed to reshape in split_heads".into())
                })?
                .to_owned();

            Ok(Tensor::F32(reshaped.permuted_axes(vec![0, 2, 1, 3])))
        },
        _ => Err(TrustformersError::tensor_op_error(
            "Unsupported tensor type",
            "flash_attention::split_heads",
        )),
    }
}

/// Merge `[batch, num_heads, seq_len, head_dim]` back into
/// `[batch, seq_len, num_heads * head_dim]`.
fn merge_heads(tensor: &Tensor) -> Result<Tensor> {
    let shape = tensor.shape();
    if shape.len() != 4 {
        return Err(TrustformersError::tensor_op_error(
            "Input tensor must have 4 dimensions",
            "flash_attention::merge_heads",
        ));
    }

    match tensor {
        Tensor::F32(arr) => {
            let batch_size = shape[0];
            let seq_len = shape[2];
            let hidden_size = shape[1] * shape[3];

            let transposed = arr.clone().permuted_axes(vec![0, 2, 1, 3]);
            let merged = transposed
                .as_standard_layout()
                .to_shape(IxDyn(&[batch_size, seq_len, hidden_size]))
                .map_err(|_| {
                    TrustformersError::shape_error("Failed to reshape in merge_heads".into())
                })?
                .to_owned();

            Ok(Tensor::F32(merged))
        },
        _ => Err(TrustformersError::tensor_op_error(
            "Unsupported tensor type",
            "flash_attention::merge_heads",
        )),
    }
}

/// Repeat each key/value head so that grouped-query attention can reuse the
/// dense `[batch, num_query_heads, seq, head_dim]` kernel.
fn expand_kv_heads(tensor: &Tensor, num_kv_heads: usize, num_query_heads: usize) -> Result<Tensor> {
    if num_kv_heads == num_query_heads {
        return Ok(tensor.clone());
    }
    if num_kv_heads == 0 || !num_query_heads.is_multiple_of(num_kv_heads) {
        return Err(TrustformersError::tensor_op_error(
            &format!(
                "num_query_heads {} must be a positive multiple of num_key_value_heads {}",
                num_query_heads, num_kv_heads
            ),
            "flash_attention::expand_kv_heads",
        ));
    }

    match tensor {
        Tensor::F32(array) => {
            let shape = array.shape();
            if shape.len() != 4 || shape[1] != num_kv_heads {
                return Err(TrustformersError::tensor_op_error(
                    &format!(
                        "Expected a [batch, {}, seq, head_dim] tensor, got {:?}",
                        num_kv_heads, shape
                    ),
                    "flash_attention::expand_kv_heads",
                ));
            }
            let (batch, seq_len, head_dim) = (shape[0], shape[2], shape[3]);
            let group_size = num_query_heads / num_kv_heads;

            let mut data = Vec::with_capacity(batch * num_query_heads * seq_len * head_dim);
            for b in 0..batch {
                for head in 0..num_query_heads {
                    let kv_head = head / group_size;
                    for position in 0..seq_len {
                        for dim in 0..head_dim {
                            data.push(array[[b, kv_head, position, dim]]);
                        }
                    }
                }
            }

            Ok(Tensor::F32(
                ArrayD::from_shape_vec(IxDyn(&[batch, num_query_heads, seq_len, head_dim]), data)
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            ))
        },
        _ => Err(TrustformersError::tensor_op_error(
            "Unsupported tensor type",
            "flash_attention::expand_kv_heads",
        )),
    }
}

/// Promote a 2-D `[seq_len, hidden]` tensor to `[1, seq_len, hidden]`.
///
/// Returns the tensor together with a flag telling the caller whether the
/// batch dimension has to be removed again on the way out.
fn ensure_batched(hidden_states: Tensor) -> Result<(Tensor, bool)> {
    match &hidden_states {
        Tensor::F32(arr) if arr.ndim() == 2 => {
            let shape = arr.shape();
            let expanded = arr
                .view()
                .into_shape_with_order(IxDyn(&[1, shape[0], shape[1]]))
                .map_err(|e| {
                    TrustformersError::shape_error(format!("Failed to add batch dimension: {e}"))
                })?
                .to_owned();
            Ok((Tensor::F32(expanded), true))
        },
        _ => Ok((hidden_states, false)),
    }
}

fn drop_batch_dimension(result: Tensor) -> Tensor {
    match &result {
        Tensor::F32(arr) if arr.shape()[0] == 1 => {
            Tensor::F32(arr.index_axis(Axis(0), 0).to_owned())
        },
        _ => result,
    }
}

/// Shared projection + attention + output-projection pipeline used by
/// [`FlashAttention`], [`MultiQueryAttention`] and [`GroupedQueryAttention`].
#[allow(clippy::too_many_arguments)]
fn projected_attention(
    hidden_states: &Tensor,
    query: &Linear,
    key: &Linear,
    value: &Linear,
    out_proj: &Linear,
    num_query_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attention_mask: Option<&Tensor>,
    params: &FlashParams,
) -> Result<Tensor> {
    let query_states = split_heads(
        &query.forward(hidden_states.clone())?,
        num_query_heads,
        head_dim,
    )?;
    let key_states = expand_kv_heads(
        &split_heads(&key.forward(hidden_states.clone())?, num_kv_heads, head_dim)?,
        num_kv_heads,
        num_query_heads,
    )?;
    let value_states = expand_kv_heads(
        &split_heads(
            &value.forward(hidden_states.clone())?,
            num_kv_heads,
            head_dim,
        )?,
        num_kv_heads,
        num_query_heads,
    )?;

    let context = flash_attention(
        &query_states,
        &key_states,
        &value_states,
        attention_mask,
        params,
    )?;
    out_proj.forward(merge_heads(&context)?)
}

/// FlashAttention: Memory-efficient attention computation
///
/// This implements the FlashAttention algorithm which reduces memory complexity
/// from O(N²) to O(N) by computing attention in blocks and not materializing
/// the full attention matrix.
///
/// Reference: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
/// <https://arxiv.org/abs/2205.14135>
///
/// `use_flash_attention_2` selects the tile size only: on CPU both settings run
/// the same kernel (which already uses FlashAttention-2's deferred
/// normalisation), with `-2` doubling the tile for long sequences.
#[derive(Debug, Clone)]
pub struct FlashAttention {
    num_heads: usize,
    hidden_size: usize,
    head_dim: usize,
    query: Linear,
    key: Linear,
    value: Linear,
    out_proj: Linear,
    dropout_prob: f32,
    training: bool,
    block_size: usize,
    causal: bool,
    use_flash_attention_2: bool,
}

impl FlashAttention {
    /// Create a FlashAttention layer (FlashAttention-2 tiling by default).
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        dropout_prob: f32,
        bias: bool,
        block_size: Option<usize>,
        causal: bool,
    ) -> Result<Self> {
        Self::new_with_version(
            hidden_size,
            num_heads,
            dropout_prob,
            bias,
            block_size,
            causal,
            true,
        )
    }

    /// Create a FlashAttention layer, choosing the tiling strategy explicitly.
    pub fn new_with_version(
        hidden_size: usize,
        num_heads: usize,
        dropout_prob: f32,
        bias: bool,
        block_size: Option<usize>,
        causal: bool,
        use_flash_attention_2: bool,
    ) -> Result<Self> {
        if !hidden_size.is_multiple_of(num_heads) {
            return Err(TrustformersError::invalid_config(format!(
                "hidden_size {} must be divisible by num_heads {}",
                hidden_size, num_heads
            )));
        }

        let head_dim = hidden_size / num_heads;
        let block_size = block_size.unwrap_or(64).max(1); // Default block size

        Ok(Self {
            num_heads,
            hidden_size,
            head_dim,
            query: Linear::new(hidden_size, hidden_size, bias),
            key: Linear::new(hidden_size, hidden_size, bias),
            value: Linear::new(hidden_size, hidden_size, bias),
            out_proj: Linear::new(hidden_size, hidden_size, bias),
            dropout_prob,
            training: false,
            block_size,
            causal,
            use_flash_attention_2,
        })
    }

    /// Number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Hidden size of the layer.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Dimension of a single attention head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Whether the layer is in training mode (which enables attention dropout).
    pub fn is_training(&self) -> bool {
        self.training
    }

    /// Switch between training and inference mode.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Total number of parameters in this layer.
    pub fn parameter_count(&self) -> usize {
        self.query.parameter_count()
            + self.key.parameter_count()
            + self.value.parameter_count()
            + self.out_proj.parameter_count()
    }

    /// Replace the query/key/value/output projection weights.
    pub fn set_projections(&mut self, query: Linear, key: Linear, value: Linear, out_proj: Linear) {
        self.query = query;
        self.key = key;
        self.value = value;
        self.out_proj = out_proj;
    }

    /// Tiling parameters for the current configuration.
    fn params(&self) -> Result<FlashParams> {
        let block_size = if self.use_flash_attention_2 {
            (self.block_size * 2).min(1024)
        } else {
            self.block_size
        };
        let dropout = if self.training { Some(self.dropout_prob) } else { None };
        FlashParams::new(self.head_dim, self.causal, block_size).with_dropout(dropout)
    }

    /// Run FlashAttention over already-projected `[batch, heads, seq, head_dim]`
    /// tensors.
    pub fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        flash_attention(q, k, v, attention_mask, &self.params()?)
    }
}

/// Input bundle for the attention layers in this module.
#[derive(Debug, Clone)]
pub struct FlashAttentionInput {
    /// `[batch, seq_len, hidden]` (or `[seq_len, hidden]`) hidden states.
    pub hidden_states: Tensor,
    /// Optional attention mask, see [`crate::layers::attention::mask`].
    pub attention_mask: Option<Tensor>,
}

impl Layer for FlashAttention {
    type Input = FlashAttentionInput;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let (hidden_states, was_2d) = ensure_batched(input.hidden_states)?;

        let result = projected_attention(
            &hidden_states,
            &self.query,
            &self.key,
            &self.value,
            &self.out_proj,
            self.num_heads,
            self.num_heads,
            self.head_dim,
            input.attention_mask.as_ref(),
            &self.params()?,
        )?;

        if was_2d {
            Ok(drop_batch_dimension(result))
        } else {
            Ok(result)
        }
    }
}

/// Multi-Query Attention (MQA) - uses a single key/value head shared by all
/// query heads, which shrinks the KV cache by `num_heads`.
///
/// Reference: *Fast Transformer Decoding: One Write-Head is All You Need*
/// (<https://arxiv.org/abs/1911.02150>).
#[derive(Debug, Clone)]
pub struct MultiQueryAttention {
    num_heads: usize,
    hidden_size: usize,
    head_dim: usize,
    query: Linear,
    key: Linear,
    value: Linear,
    out_proj: Linear,
    dropout_prob: f32,
    training: bool,
    block_size: usize,
    causal: bool,
}

impl MultiQueryAttention {
    /// Create a multi-query attention layer.
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        dropout_prob: f32,
        bias: bool,
    ) -> Result<Self> {
        if !hidden_size.is_multiple_of(num_heads) {
            return Err(TrustformersError::invalid_config(format!(
                "hidden_size {} must be divisible by num_heads {}",
                hidden_size, num_heads
            )));
        }

        let head_dim = hidden_size / num_heads;

        Ok(Self {
            num_heads,
            hidden_size,
            head_dim,
            query: Linear::new(hidden_size, hidden_size, bias),
            key: Linear::new(hidden_size, head_dim, bias), // Single head for key
            value: Linear::new(hidden_size, head_dim, bias), // Single head for value
            out_proj: Linear::new(hidden_size, hidden_size, bias),
            dropout_prob,
            training: false,
            block_size: 64,
            causal: false,
        })
    }

    /// Number of query heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Hidden size of the layer.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Dimension of a single attention head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Enable or disable causal masking.
    pub fn set_causal(&mut self, causal: bool) {
        self.causal = causal;
    }

    /// Set the tile size used by the FlashAttention kernel.
    pub fn set_block_size(&mut self, block_size: usize) {
        self.block_size = block_size.max(1);
    }

    /// Switch between training and inference mode.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Total number of parameters in this layer.
    pub fn parameter_count(&self) -> usize {
        self.query.parameter_count()
            + self.key.parameter_count()
            + self.value.parameter_count()
            + self.out_proj.parameter_count()
    }

    /// Replace the query/key/value/output projections.
    pub fn set_projections(&mut self, query: Linear, key: Linear, value: Linear, out_proj: Linear) {
        self.query = query;
        self.key = key;
        self.value = value;
        self.out_proj = out_proj;
    }

    /// Multi-query self-attention over `[batch, seq_len, hidden]` states.
    pub fn forward_self_attention(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let dropout = if self.training { Some(self.dropout_prob) } else { None };
        let params =
            FlashParams::new(self.head_dim, self.causal, self.block_size).with_dropout(dropout)?;
        projected_attention(
            hidden_states,
            &self.query,
            &self.key,
            &self.value,
            &self.out_proj,
            self.num_heads,
            1,
            self.head_dim,
            attention_mask,
            &params,
        )
    }
}

impl Layer for MultiQueryAttention {
    type Input = FlashAttentionInput;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let (hidden_states, was_2d) = ensure_batched(input.hidden_states)?;
        let result = self.forward_self_attention(&hidden_states, input.attention_mask.as_ref())?;
        if was_2d {
            Ok(drop_batch_dimension(result))
        } else {
            Ok(result)
        }
    }
}

/// Grouped-Query Attention (GQA) - groups query heads so that each group shares
/// one key/value head. Sits between MHA and MQA in both cost and quality.
///
/// Reference: *GQA: Training Generalized Multi-Query Transformer Models from
/// Multi-Head Checkpoints* (<https://arxiv.org/abs/2305.13245>).
#[derive(Debug, Clone)]
pub struct GroupedQueryAttention {
    num_query_heads: usize,
    num_key_value_heads: usize,
    hidden_size: usize,
    head_dim: usize,
    query: Linear,
    key: Linear,
    value: Linear,
    out_proj: Linear,
    dropout_prob: f32,
    training: bool,
    block_size: usize,
    causal: bool,
}

impl GroupedQueryAttention {
    /// Create a grouped-query attention layer.
    pub fn new(
        hidden_size: usize,
        num_query_heads: usize,
        num_key_value_heads: usize,
        dropout_prob: f32,
        bias: bool,
    ) -> Result<Self> {
        if !hidden_size.is_multiple_of(num_query_heads) {
            return Err(TrustformersError::invalid_config(format!(
                "hidden_size {} must be divisible by num_query_heads {}",
                hidden_size, num_query_heads
            )));
        }

        if num_key_value_heads == 0 || !num_query_heads.is_multiple_of(num_key_value_heads) {
            return Err(TrustformersError::invalid_config(format!(
                "num_query_heads {} must be divisible by num_key_value_heads {}",
                num_query_heads, num_key_value_heads
            )));
        }

        let head_dim = hidden_size / num_query_heads;
        let kv_hidden_size = num_key_value_heads * head_dim;

        Ok(Self {
            num_query_heads,
            num_key_value_heads,
            hidden_size,
            head_dim,
            query: Linear::new(hidden_size, hidden_size, bias),
            key: Linear::new(hidden_size, kv_hidden_size, bias),
            value: Linear::new(hidden_size, kv_hidden_size, bias),
            out_proj: Linear::new(hidden_size, hidden_size, bias),
            dropout_prob,
            training: false,
            block_size: 64,
            causal: false,
        })
    }

    /// Number of query heads.
    pub fn num_query_heads(&self) -> usize {
        self.num_query_heads
    }

    /// Number of key/value heads.
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }

    /// Hidden size of the layer.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Dimension of a single attention head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Enable or disable causal masking.
    pub fn set_causal(&mut self, causal: bool) {
        self.causal = causal;
    }

    /// Set the tile size used by the FlashAttention kernel.
    pub fn set_block_size(&mut self, block_size: usize) {
        self.block_size = block_size.max(1);
    }

    /// Switch between training and inference mode.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Total number of parameters in this layer.
    pub fn parameter_count(&self) -> usize {
        self.query.parameter_count()
            + self.key.parameter_count()
            + self.value.parameter_count()
            + self.out_proj.parameter_count()
    }

    /// Replace the query/key/value/output projections.
    pub fn set_projections(&mut self, query: Linear, key: Linear, value: Linear, out_proj: Linear) {
        self.query = query;
        self.key = key;
        self.value = value;
        self.out_proj = out_proj;
    }

    /// Grouped-query self-attention over `[batch, seq_len, hidden]` states.
    pub fn forward_self_attention(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let dropout = if self.training { Some(self.dropout_prob) } else { None };
        let params =
            FlashParams::new(self.head_dim, self.causal, self.block_size).with_dropout(dropout)?;
        projected_attention(
            hidden_states,
            &self.query,
            &self.key,
            &self.value,
            &self.out_proj,
            self.num_query_heads,
            self.num_key_value_heads,
            self.head_dim,
            attention_mask,
            &params,
        )
    }
}

impl Layer for GroupedQueryAttention {
    type Input = FlashAttentionInput;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let (hidden_states, was_2d) = ensure_batched(input.hidden_states)?;
        let result = self.forward_self_attention(&hidden_states, input.attention_mask.as_ref())?;
        if was_2d {
            Ok(drop_batch_dimension(result))
        } else {
            Ok(result)
        }
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
        Tensor::from_vec(data, shape).expect("test tensor shape must be valid")
    }

    fn max_abs_difference(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "output length mismatch");
        a.iter().zip(b.iter()).fold(0.0f32, |acc, (x, y)| acc.max((x - y).abs()))
    }

    /// Linear layer with deterministic weights so two layers can be made
    /// numerically identical.
    fn linear(in_features: usize, out_features: usize, seed: u32) -> Linear {
        let mut layer = Linear::new(in_features, out_features, true);
        layer
            .set_weight(deterministic(&[out_features, in_features], seed))
            .expect("weight shape");
        layer.set_bias(deterministic(&[out_features], seed + 1)).expect("bias shape");
        layer
    }

    /// Naive `softmax(scale * Q K^T) V` over `[batch, heads, seq, head_dim]`.
    fn naive_attention(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Vec<f32> {
        let (Tensor::F32(q_arr), Tensor::F32(k_arr), Tensor::F32(v_arr)) = (q, k, v) else {
            panic!("reference implementation requires F32 tensors");
        };
        let batch = q_arr.shape()[0];
        let heads = q_arr.shape()[1];
        let seq_q = q_arr.shape()[2];
        let head_dim = q_arr.shape()[3];
        let seq_k = k_arr.shape()[2];
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut out = vec![0.0f32; batch * heads * seq_q * head_dim];
        for b in 0..batch {
            for h in 0..heads {
                for i in 0..seq_q {
                    let limit = if causal { i + 1 } else { seq_k };
                    let mut scores = Vec::with_capacity(limit);
                    for j in 0..limit {
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_arr[[b, h, i, d]] * k_arr[[b, h, j, d]];
                        }
                        scores.push(dot * scale);
                    }
                    let max = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
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

    #[test]
    fn test_flash_attention_creation() {
        let flash_attn = FlashAttention::new(768, 12, 0.1, true, Some(64), false);
        assert!(flash_attn.is_ok());

        let flash_attn = flash_attn.expect("Failed to create FlashAttention");
        assert_eq!(flash_attn.num_heads(), 12);
        assert_eq!(flash_attn.hidden_size(), 768);
        assert_eq!(flash_attn.head_dim(), 64);
        assert_eq!(flash_attn.block_size, 64);
        assert!(!flash_attn.causal);
    }

    #[test]
    fn test_flash_attention_forward_pass() {
        let flash_attn = FlashAttention::new(256, 8, 0.0, true, Some(32), false)
            .expect("Failed to create FlashAttention");

        let hidden_states = deterministic(&[2, 128, 256], 1);
        let input = FlashAttentionInput {
            hidden_states,
            attention_mask: None,
        };

        let output = flash_attn.forward(input).expect("Forward pass failed");
        assert_eq!(output.shape(), vec![2, 128, 256]);
    }

    #[test]
    fn flash_attention_matches_naive_reference() {
        for (causal, block_size) in [(false, 4usize), (true, 4), (false, 7), (true, 7)] {
            let flash_attn = FlashAttention::new(32, 4, 0.0, false, Some(block_size), causal)
                .expect("Failed to create FlashAttention");
            let q = deterministic(&[2, 4, 13, 8], 2);
            let k = deterministic(&[2, 4, 13, 8], 3);
            let v = deterministic(&[2, 4, 13, 8], 4);

            let actual = flash_attn.attention(&q, &k, &v, None).expect("attention must succeed");
            let expected = naive_attention(&q, &k, &v, causal);
            let difference = max_abs_difference(&actual.data().expect("data"), &expected);
            assert!(
                difference < 1e-4,
                "FlashAttention deviates from the naive reference by {difference} \
                 (causal={causal}, block={block_size})"
            );
        }
    }

    #[test]
    fn test_multi_query_attention_creation() {
        let mqa = MultiQueryAttention::new(768, 12, 0.1, true);
        assert!(mqa.is_ok());

        let mqa = mqa.expect("Failed to create MultiQueryAttention");
        assert_eq!(mqa.num_heads(), 12);
        assert_eq!(mqa.hidden_size(), 768);
        assert_eq!(mqa.head_dim(), 64);
    }

    #[test]
    fn multi_query_attention_forward_shares_one_kv_head() {
        let hidden_size = 32;
        let num_heads = 4;
        let head_dim = hidden_size / num_heads;
        let mut mqa = MultiQueryAttention::new(hidden_size, num_heads, 0.0, true)
            .expect("construction failed");
        mqa.set_block_size(3);
        mqa.set_projections(
            linear(hidden_size, hidden_size, 10),
            linear(hidden_size, head_dim, 12),
            linear(hidden_size, head_dim, 14),
            linear(hidden_size, hidden_size, 16),
        );

        let hidden_states = deterministic(&[1, 9, hidden_size], 5);
        let output = mqa.forward_self_attention(&hidden_states, None).expect("MQA forward failed");
        assert_eq!(output.shape(), vec![1, 9, hidden_size]);

        // The output must actually depend on the input.
        let other = mqa
            .forward_self_attention(&deterministic(&[1, 9, hidden_size], 6), None)
            .expect("MQA forward failed");
        assert!(
            max_abs_difference(&output.data().expect("data"), &other.data().expect("data")) > 1e-3,
            "MQA output must depend on its input"
        );
    }

    #[test]
    fn test_grouped_query_attention_creation() {
        let gqa = GroupedQueryAttention::new(768, 12, 4, 0.1, true);
        assert!(gqa.is_ok());

        let gqa = gqa.expect("Failed to create GroupedQueryAttention");
        assert_eq!(gqa.num_query_heads(), 12);
        assert_eq!(gqa.num_key_value_heads(), 4);
        assert_eq!(gqa.hidden_size(), 768);
        assert_eq!(gqa.head_dim(), 64);
    }

    #[test]
    fn grouped_query_attention_degenerates_to_flash_attention() {
        // With one KV head per query head, GQA must reproduce plain
        // FlashAttention exactly.
        let hidden_size = 32;
        let num_heads = 4;
        let mut gqa = GroupedQueryAttention::new(hidden_size, num_heads, num_heads, 0.0, true)
            .expect("gqa construction failed");
        let mut flash = FlashAttention::new_with_version(
            hidden_size,
            num_heads,
            0.0,
            true,
            Some(64),
            false,
            false,
        )
        .expect("flash construction failed");

        let projections = || {
            (
                linear(hidden_size, hidden_size, 20),
                linear(hidden_size, hidden_size, 22),
                linear(hidden_size, hidden_size, 24),
                linear(hidden_size, hidden_size, 26),
            )
        };
        let (q, k, v, o) = projections();
        gqa.set_projections(q, k, v, o);
        let (q, k, v, o) = projections();
        flash.set_projections(q, k, v, o);

        let hidden_states = deterministic(&[1, 11, hidden_size], 7);
        let gqa_output =
            gqa.forward_self_attention(&hidden_states, None).expect("gqa forward failed");
        let flash_output = flash
            .forward(FlashAttentionInput {
                hidden_states,
                attention_mask: None,
            })
            .expect("flash forward failed");

        let difference = max_abs_difference(
            &gqa_output.data().expect("gqa data"),
            &flash_output.data().expect("flash data"),
        );
        assert!(
            difference < 1e-5,
            "GQA with one KV head per query head must match FlashAttention, got {difference}"
        );
    }

    #[test]
    fn grouped_query_attention_repeats_kv_heads() {
        let kv = deterministic(&[1, 2, 3, 2], 8);
        let expanded = expand_kv_heads(&kv, 2, 6).expect("expansion must succeed");
        assert_eq!(expanded.shape(), vec![1, 6, 3, 2]);

        let (Tensor::F32(source), Tensor::F32(target)) = (&kv, &expanded) else {
            panic!("expected F32 tensors");
        };
        for head in 0..6 {
            for position in 0..3 {
                for dim in 0..2 {
                    assert_eq!(
                        target[[0, head, position, dim]],
                        source[[0, head / 3, position, dim]],
                        "head {head} must mirror kv head {}",
                        head / 3
                    );
                }
            }
        }
    }

    #[test]
    fn test_flash_attention_causal() {
        let flash_attn = FlashAttention::new(256, 8, 0.0, true, Some(32), true)
            .expect("Failed to create FlashAttention");

        let hidden_states = deterministic(&[1, 64, 256], 9);
        let input = FlashAttentionInput {
            hidden_states,
            attention_mask: None,
        };

        let output = flash_attn.forward(input).expect("Forward pass failed");
        assert_eq!(output.shape(), vec![1, 64, 256]);
    }

    #[test]
    fn test_flash_attention_deterministic() {
        let flash_attn = FlashAttention::new(128, 4, 0.0, true, Some(16), false)
            .expect("Failed to create FlashAttention");

        let hidden_states = Tensor::ones(&[1, 32, 128]).expect("Failed to create ones tensor");
        let input = FlashAttentionInput {
            hidden_states: hidden_states.clone(),
            attention_mask: None,
        };

        let output1 = flash_attn.forward(input.clone()).expect("Forward pass failed");
        let output2 = flash_attn.forward(input).expect("Forward pass failed");

        let data1 = output1.data().expect("Failed to get data");
        let data2 = output2.data().expect("Failed to get data");
        assert!(
            max_abs_difference(&data1, &data2) < 1e-6,
            "Outputs should be deterministic"
        );
    }

    #[test]
    fn test_flash_attention_2_creation() {
        let flash_attn_2 =
            FlashAttention::new_with_version(768, 12, 0.1, true, Some(64), false, true);
        assert!(flash_attn_2.is_ok());

        let flash_attn_2 = flash_attn_2.expect("Failed to create FlashAttention-2");
        assert_eq!(flash_attn_2.num_heads(), 12);
        assert_eq!(flash_attn_2.hidden_size(), 768);
        assert_eq!(flash_attn_2.head_dim(), 64);
        assert_eq!(flash_attn_2.block_size, 64);
        assert!(!flash_attn_2.causal);
        assert!(flash_attn_2.use_flash_attention_2);
    }

    #[test]
    fn test_flash_attention_2_forward_pass() {
        let flash_attn_2 =
            FlashAttention::new_with_version(256, 8, 0.0, true, Some(32), false, true)
                .expect("Failed to create FlashAttention-2");

        let hidden_states = deterministic(&[2, 128, 256], 10);
        let input = FlashAttentionInput {
            hidden_states,
            attention_mask: None,
        };

        let output = flash_attn_2.forward(input).expect("Forward pass failed");
        assert_eq!(output.shape(), vec![2, 128, 256]);
    }

    #[test]
    fn test_flash_attention_2_vs_1_consistency() {
        // Both versions run the same exact kernel with different tile sizes, so
        // they must agree to within floating-point noise.
        for causal in [false, true] {
            let hidden_size = 128;
            let mut flash_attn_1 = FlashAttention::new_with_version(
                hidden_size,
                4,
                0.0,
                true,
                Some(16),
                causal,
                false,
            )
            .expect("Failed to create FlashAttention-1");
            let mut flash_attn_2 =
                FlashAttention::new_with_version(hidden_size, 4, 0.0, true, Some(16), causal, true)
                    .expect("Failed to create FlashAttention-2");

            let projections = || {
                (
                    linear(hidden_size, hidden_size, 30),
                    linear(hidden_size, hidden_size, 32),
                    linear(hidden_size, hidden_size, 34),
                    linear(hidden_size, hidden_size, 36),
                )
            };
            let (q, k, v, o) = projections();
            flash_attn_1.set_projections(q, k, v, o);
            let (q, k, v, o) = projections();
            flash_attn_2.set_projections(q, k, v, o);

            let hidden_states = deterministic(&[1, 37, hidden_size], 11);
            let input = FlashAttentionInput {
                hidden_states,
                attention_mask: None,
            };

            let output1 = flash_attn_1.forward(input.clone()).expect("Forward pass failed");
            let output2 = flash_attn_2.forward(input).expect("Forward pass failed");

            let max_diff = max_abs_difference(
                &output1.data().expect("Failed to get data"),
                &output2.data().expect("Failed to get data"),
            );
            assert!(
                max_diff < 1e-4,
                "FlashAttention-2 output differs from FlashAttention-1: max_diff = {max_diff} \
                 (causal={causal})"
            );
        }
    }

    #[test]
    fn test_flash_attention_2_causal() {
        let flash_attn_2 =
            FlashAttention::new_with_version(256, 8, 0.0, true, Some(32), true, true)
                .expect("Failed to create FlashAttention-2");

        let hidden_states = deterministic(&[1, 64, 256], 12);
        let input = FlashAttentionInput {
            hidden_states,
            attention_mask: None,
        };

        let output = flash_attn_2.forward(input).expect("Forward pass failed");
        assert_eq!(output.shape(), vec![1, 64, 256]);
    }

    #[test]
    fn attention_mask_is_actually_applied() {
        // Regression test: the mask argument used to be ignored outright.
        let hidden_size = 32;
        let seq_len = 10;
        let mut flash = FlashAttention::new(hidden_size, 4, 0.0, true, Some(4), false)
            .expect("construction failed");
        flash.set_projections(
            linear(hidden_size, hidden_size, 40),
            linear(hidden_size, hidden_size, 42),
            linear(hidden_size, hidden_size, 44),
            linear(hidden_size, hidden_size, 46),
        );

        let hidden_states = deterministic(&[1, seq_len, hidden_size], 13);
        let mut keep = vec![1.0f32; seq_len];
        keep[3] = 0.0;
        keep[8] = 0.0;
        let mask = Tensor::from_vec(keep, &[1, 1, 1, seq_len]).expect("mask shape");

        let unmasked = flash
            .forward(FlashAttentionInput {
                hidden_states: hidden_states.clone(),
                attention_mask: None,
            })
            .expect("unmasked forward failed");
        let masked = flash
            .forward(FlashAttentionInput {
                hidden_states,
                attention_mask: Some(mask),
            })
            .expect("masked forward failed");

        assert!(
            max_abs_difference(
                &unmasked.data().expect("data"),
                &masked.data().expect("data")
            ) > 1e-3,
            "the attention mask must change the result"
        );
    }

    #[test]
    fn training_mode_enables_attention_dropout() {
        let hidden_size = 32;
        let mut flash = FlashAttention::new(hidden_size, 4, 0.5, true, Some(8), false)
            .expect("construction failed");
        let hidden_states = deterministic(&[1, 16, hidden_size], 14);
        let input = FlashAttentionInput {
            hidden_states,
            attention_mask: None,
        };

        assert!(!flash.is_training());
        let inference = flash.forward(input.clone()).expect("inference forward failed");
        let inference_again = flash.forward(input.clone()).expect("inference forward failed");
        assert!(
            max_abs_difference(
                &inference.data().expect("data"),
                &inference_again.data().expect("data")
            ) < 1e-6,
            "inference must be deterministic"
        );

        flash.set_training(true);
        let training = flash.forward(input).expect("training forward failed");
        assert!(
            max_abs_difference(
                &inference.data().expect("data"),
                &training.data().expect("data")
            ) > 1e-4,
            "training mode must apply attention dropout"
        );
    }

    #[test]
    fn two_dimensional_input_keeps_its_rank() {
        let flash_attn =
            FlashAttention::new(32, 4, 0.0, true, Some(8), false).expect("construction failed");
        let hidden_states = deterministic(&[6, 32], 15);
        let output = flash_attn
            .forward(FlashAttentionInput {
                hidden_states,
                attention_mask: None,
            })
            .expect("forward failed");
        assert_eq!(output.shape(), vec![6, 32]);
    }
}
