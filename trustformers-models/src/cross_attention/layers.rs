use super::config::{
    AdaptiveAttentionConfig, CrossAttentionConfig, GateActivation, GatedAttentionConfig,
    HierarchicalAttentionConfig, SparseAttentionConfig,
};
use super::utils::{
    create_sparse_mask, pool_tensor, reshape_for_multihead, reshape_from_multihead,
    scaled_dot_product_attention, CrossAttentionOutput, PoolingMethod, SparsePattern,
};
use trustformers_core::{
    errors::{invalid_config, tensor_op_error, Result},
    layers::{LayerNorm, Linear},
    ops::activations::{gelu, sigmoid, silu, tanh},
    tensor::Tensor,
    traits::Layer,
};

/// Standard cross-attention layer
pub struct CrossAttention {
    config: CrossAttentionConfig,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    scale: f32,
}

impl CrossAttention {
    pub fn new(config: CrossAttentionConfig) -> Result<Self> {
        config.validate().map_err(|e| invalid_config("config_field", e.to_string()))?;

        let hidden_size = config.hidden_size;
        let q_proj = Linear::new(hidden_size, hidden_size, config.bias);
        let k_proj = Linear::new(hidden_size, hidden_size, config.bias);
        let v_proj = Linear::new(hidden_size, hidden_size, config.bias);
        let out_proj = Linear::new(hidden_size, hidden_size, config.bias);
        let scale = config.get_scale();

        Ok(Self {
            config,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            scale,
        })
    }

    /// Forward pass with separate query, key, and value inputs
    pub fn forward(
        &self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Option<Tensor>,
    ) -> Result<CrossAttentionOutput> {
        let q = self.q_proj.forward(query)?;
        let k = self.k_proj.forward(key)?;
        let v = self.v_proj.forward(value)?;

        let attn_output = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            mask.as_ref(),
            self.scale,
            self.config.attention_dropout,
            true, // training flag - would be configurable
        )?;

        let output = self.out_proj.forward(attn_output.output)?;

        Ok(CrossAttentionOutput {
            output,
            attention_weights: attn_output.attention_weights,
            attention_stats: attn_output.attention_stats,
        })
    }
}

/// Multi-head cross-attention layer
pub struct MultiHeadCrossAttention {
    config: CrossAttentionConfig,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    head_dim: usize,
    scale: f32,
}

impl MultiHeadCrossAttention {
    pub fn new(config: CrossAttentionConfig) -> Result<Self> {
        config.validate().map_err(|e| invalid_config("config_field", e.to_string()))?;

        let hidden_size = config.hidden_size;
        let head_dim = config.get_head_dim();

        let q_proj = Linear::new(hidden_size, hidden_size, config.bias);
        let k_proj = Linear::new(hidden_size, hidden_size, config.bias);
        let v_proj = Linear::new(hidden_size, hidden_size, config.bias);
        let out_proj = Linear::new(hidden_size, hidden_size, config.bias);
        let scale = config.get_scale();

        Ok(Self {
            config,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            head_dim,
            scale,
        })
    }

    pub fn forward(
        &self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Option<Tensor>,
    ) -> Result<CrossAttentionOutput> {
        let batch_size = query.shape()[0];
        let query_len = query.shape()[1];
        let key_len = key.shape()[1];

        // Project to query, key, value
        let q = self.q_proj.forward(query)?;
        let k = self.k_proj.forward(key)?;
        let v = self.v_proj.forward(value)?;

        // Reshape for multi-head attention
        let q = reshape_for_multihead(q, self.config.num_heads, self.head_dim)?;
        let k = reshape_for_multihead(k, self.config.num_heads, self.head_dim)?;
        let v = reshape_for_multihead(v, self.config.num_heads, self.head_dim)?;

        // Expand mask for multiple heads if provided
        let mask = if let Some(mask) = mask {
            Some(mask.unsqueeze(1)?.broadcast_to(&[
                batch_size,
                self.config.num_heads,
                query_len,
                key_len,
            ])?)
        } else {
            None
        };

        // Compute attention
        let attn_output = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            mask.as_ref(),
            self.scale,
            self.config.attention_dropout,
            true,
        )?;

        // Reshape back
        let output = reshape_from_multihead(attn_output.output, self.config.hidden_size)?;
        let output = self.out_proj.forward(output)?;

        Ok(CrossAttentionOutput {
            output,
            attention_weights: attn_output.attention_weights,
            attention_stats: attn_output.attention_stats,
        })
    }
}

/// Sparse cross-attention layer
pub struct SparseCrossAttention {
    #[allow(dead_code)]
    config: CrossAttentionConfig,
    base_attention: MultiHeadCrossAttention,
    sparse_config: SparseAttentionConfig,
}

impl SparseCrossAttention {
    pub fn new(config: CrossAttentionConfig) -> Result<Self> {
        let sparse_config = config.sparse_config.clone().unwrap_or_default();
        let base_attention = MultiHeadCrossAttention::new(config.clone())?;

        Ok(Self {
            config,
            base_attention,
            sparse_config,
        })
    }

    pub fn forward(
        &self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Option<Tensor>,
    ) -> Result<CrossAttentionOutput> {
        let query_len = query.shape()[1];
        let key_len = key.shape()[1];

        // Create sparse mask
        let sparse_pattern = match self.sparse_config.pattern {
            crate::cross_attention::config::SparsePattern::Random => SparsePattern::Random,
            crate::cross_attention::config::SparsePattern::Block => {
                SparsePattern::Block(self.sparse_config.block_size.unwrap_or(64))
            },
            crate::cross_attention::config::SparsePattern::Strided => {
                SparsePattern::Strided(4) // Default stride
            },
            crate::cross_attention::config::SparsePattern::Local => {
                SparsePattern::TopK(self.sparse_config.block_size.unwrap_or(64))
            },
            crate::cross_attention::config::SparsePattern::TopK => {
                SparsePattern::TopK(self.sparse_config.random_connections.unwrap_or(32))
            },
        };

        let sparse_mask = create_sparse_mask(
            query_len,
            key_len,
            self.sparse_config.sparsity_ratio,
            sparse_pattern,
        )?;

        // Combine with existing mask
        let combined_mask = if let Some(mask) = mask {
            Some(mask.add(&sparse_mask)?)
        } else {
            Some(sparse_mask)
        };

        // Apply sparse attention
        self.base_attention.forward(query, key, value, combined_mask)
    }
}

/// Hierarchical cross-attention layer
pub struct HierarchicalCrossAttention {
    #[allow(dead_code)]
    config: CrossAttentionConfig,
    hierarchical_config: HierarchicalAttentionConfig,
    attention_layers: Vec<MultiHeadCrossAttention>,
    pooling_layers: Vec<Linear>,
    output_projection: Linear,
}

impl HierarchicalCrossAttention {
    pub fn new(config: CrossAttentionConfig) -> Result<Self> {
        let hierarchical_config = config.hierarchical_config.clone().unwrap_or_default();

        let mut attention_layers = Vec::new();
        let mut pooling_layers = Vec::new();

        for _ in 0..hierarchical_config.num_levels {
            let layer_config = config.clone();
            attention_layers.push(MultiHeadCrossAttention::new(layer_config)?);

            if hierarchical_config.learnable_pooling {
                pooling_layers.push(Linear::new(config.hidden_size, config.hidden_size, false));
            }
        }

        let output_projection = Linear::new(
            config.hidden_size * hierarchical_config.num_levels,
            config.hidden_size,
            config.bias,
        );

        Ok(Self {
            config,
            hierarchical_config,
            attention_layers,
            pooling_layers,
            output_projection,
        })
    }

    pub fn forward(
        &self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Option<Tensor>,
    ) -> Result<CrossAttentionOutput> {
        let mut level_outputs = Vec::new();
        let mut current_key = key;
        let mut current_value = value;

        for level in 0..self.hierarchical_config.num_levels {
            // Apply attention at current level
            let attn_output = self.attention_layers[level].forward(
                query.clone(),
                current_key.clone(),
                current_value.clone(),
                mask.clone(),
            )?;

            level_outputs.push(attn_output.output);

            // Pool for next level
            if level < self.hierarchical_config.num_levels - 1 {
                let pooling_factor = self.hierarchical_config.pooling_factor;

                current_key = pool_tensor(current_key, pooling_factor, PoolingMethod::Average)?;
                current_value = pool_tensor(current_value, pooling_factor, PoolingMethod::Average)?;

                // Apply learnable pooling if configured
                if self.hierarchical_config.learnable_pooling && level < self.pooling_layers.len() {
                    current_key = self.pooling_layers[level].forward(current_key)?;
                    current_value = self.pooling_layers[level].forward(current_value)?;
                }
            }
        }

        // Aggregate multi-level outputs
        let output = match self.hierarchical_config.aggregation_method {
            crate::cross_attention::config::AggregationMethod::WeightedSum => {
                aggregate_weighted_sum(level_outputs)?
            },
            crate::cross_attention::config::AggregationMethod::Concatenation => {
                let first_output = &level_outputs[0];
                let concat_dim = first_output.shape().len().saturating_sub(1);
                let concatenated = Tensor::concat(&level_outputs, concat_dim)?;
                self.output_projection.forward(concatenated)?
            },
            crate::cross_attention::config::AggregationMethod::MaxPooling => {
                aggregate_max_pooling(level_outputs)?
            },
            crate::cross_attention::config::AggregationMethod::AvgPooling => {
                aggregate_avg_pooling(level_outputs)?
            },
        };

        Ok(CrossAttentionOutput {
            output,
            attention_weights: None,
            attention_stats: None,
        })
    }
}

/// Adaptive cross-attention layer
pub struct AdaptiveCrossAttention {
    #[allow(dead_code)]
    config: CrossAttentionConfig,
    adaptive_config: AdaptiveAttentionConfig,
    base_attention: MultiHeadCrossAttention,
    pattern_embeddings: Linear,
    pattern_selector: Linear,
    attention_patterns: Vec<Linear>,
}

impl AdaptiveCrossAttention {
    pub fn new(config: CrossAttentionConfig) -> Result<Self> {
        let adaptive_config = config.adaptive_config.clone().unwrap_or_default();
        let base_attention = MultiHeadCrossAttention::new(config.clone())?;

        let pattern_embeddings =
            Linear::new(config.hidden_size, adaptive_config.pattern_dim, false);

        let pattern_selector = Linear::new(
            adaptive_config.pattern_dim,
            adaptive_config.num_patterns,
            false,
        );

        let mut attention_patterns = Vec::new();
        for _ in 0..adaptive_config.num_patterns {
            attention_patterns.push(Linear::new(config.hidden_size, config.hidden_size, false));
        }

        Ok(Self {
            config,
            adaptive_config,
            base_attention,
            pattern_embeddings,
            pattern_selector,
            attention_patterns,
        })
    }

    pub fn forward(
        &self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Option<Tensor>,
    ) -> Result<CrossAttentionOutput> {
        // Compute pattern embeddings
        let pattern_emb = self.pattern_embeddings.forward(query.clone())?;
        let seq_len = pattern_emb.shape()[1] as f32;
        let pattern_emb = pattern_emb.sum(Some(vec![1]), false)?.div_scalar(seq_len)?; // Average over sequence length

        // Select attention pattern
        let pattern_logits = self.pattern_selector.forward(pattern_emb)?;
        let pattern_weights = if self.adaptive_config.hard_selection {
            // Hard selection - use argmax for discrete selection
            let _max_indices = pattern_logits.argmax(-1)?;
            // Create one-hot encoding by comparing with max indices
            // For now, return softmax of large logits to approximate one-hot
            let large_logits = pattern_logits.mul_scalar(100.0)?;
            large_logits.softmax(-1)?
        } else {
            // Soft selection with temperature-scaled softmax
            let scaled_logits = pattern_logits.div_scalar(self.adaptive_config.temperature)?;
            scaled_logits.softmax(-1)?
        };

        // Apply selected pattern(s)
        let mut pattern_outputs = Vec::new();
        for i in 0..self.adaptive_config.num_patterns {
            let pattern_query = self.attention_patterns[i].forward(query.clone())?;
            let pattern_output = self.base_attention.forward(
                pattern_query,
                key.clone(),
                value.clone(),
                mask.clone(),
            )?;
            pattern_outputs.push(pattern_output.output);
        }

        // Combine pattern outputs
        let output = combine_pattern_outputs(pattern_outputs, pattern_weights)?;

        Ok(CrossAttentionOutput {
            output,
            attention_weights: None,
            attention_stats: None,
        })
    }
}

/// Gated cross-attention layer
pub struct GatedCrossAttention {
    #[allow(dead_code)]
    config: CrossAttentionConfig,
    gated_config: GatedAttentionConfig,
    base_attention: MultiHeadCrossAttention,
    query_gate: Linear,
    key_gate: Option<Linear>,
    value_gate: Option<Linear>,
    gate_norm: LayerNorm,
}

impl GatedCrossAttention {
    pub fn new(config: CrossAttentionConfig) -> Result<Self> {
        let gated_config = config.gated_config.clone().unwrap_or_default();
        let base_attention = MultiHeadCrossAttention::new(config.clone())?;

        let gate_hidden_dim = gated_config.gate_hidden_dim.unwrap_or(config.hidden_size);

        let query_gate = Linear::new(config.hidden_size, gate_hidden_dim, gated_config.gate_bias);

        let key_gate = if gated_config.separate_gates {
            Some(Linear::new(
                config.hidden_size,
                gate_hidden_dim,
                gated_config.gate_bias,
            ))
        } else {
            None
        };

        let value_gate = if gated_config.separate_gates {
            Some(Linear::new(
                config.hidden_size,
                gate_hidden_dim,
                gated_config.gate_bias,
            ))
        } else {
            None
        };

        let gate_norm = LayerNorm::new(vec![gate_hidden_dim], 1e-5)?;

        Ok(Self {
            config,
            gated_config,
            base_attention,
            query_gate,
            key_gate,
            value_gate,
            gate_norm,
        })
    }

    pub fn forward(
        &self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Option<Tensor>,
    ) -> Result<CrossAttentionOutput> {
        // Compute gates
        let query_gate_out = self.query_gate.forward(query.clone())?;
        let query_gate_out = self.gate_norm.forward(query_gate_out)?;
        let query_gate_out =
            apply_gate_activation(&query_gate_out, &self.gated_config.gate_activation)?;

        let key_gate_out = if let Some(key_gate) = &self.key_gate {
            let gate_out = key_gate.forward(key.clone())?;
            let gate_out = self.gate_norm.forward(gate_out)?;
            apply_gate_activation(&gate_out, &self.gated_config.gate_activation)?
        } else {
            query_gate_out.clone()
        };

        let value_gate_out = if let Some(value_gate) = &self.value_gate {
            let gate_out = value_gate.forward(value.clone())?;
            let gate_out = self.gate_norm.forward(gate_out)?;
            apply_gate_activation(&gate_out, &self.gated_config.gate_activation)?
        } else {
            query_gate_out.clone()
        };

        // Apply gates
        let gated_query = query.mul(&query_gate_out)?;
        let gated_key = key.mul(&key_gate_out)?;
        let gated_value = value.mul(&value_gate_out)?;

        // Apply attention
        self.base_attention.forward(gated_query, gated_key, gated_value, mask)
    }
}

// Helper functions

fn apply_gate_activation(tensor: &Tensor, activation: &GateActivation) -> Result<Tensor> {
    match activation {
        GateActivation::Sigmoid => sigmoid(tensor),
        GateActivation::Tanh => tanh(tensor),
        GateActivation::ReLU => tensor.relu(),
        GateActivation::GELU => gelu(tensor),
        GateActivation::Swish => silu(tensor),
    }
}

fn aggregate_weighted_sum(outputs: Vec<Tensor>) -> Result<Tensor> {
    let mut result = outputs[0].clone();
    let weight = 1.0 / outputs.len() as f32;

    for output in outputs.iter().skip(1) {
        result = result.add(&output.mul_scalar(weight)?)?;
    }

    Ok(result)
}

fn aggregate_max_pooling(outputs: Vec<Tensor>) -> Result<Tensor> {
    if outputs.is_empty() {
        return Err(tensor_op_error(
            "tensor_operation",
            "Cannot perform max pooling on empty outputs".to_string(),
        ));
    }

    // Start with the first tensor and compute element-wise maximum with others
    let mut result = outputs[0].clone();
    for output in outputs.iter().skip(1) {
        result = result.max(output)?;
    }
    Ok(result)
}

fn aggregate_avg_pooling(outputs: Vec<Tensor>) -> Result<Tensor> {
    // Element-wise average across tensors
    let mut result = outputs[0].clone();
    for output in outputs.iter().skip(1) {
        result = result.add(output)?;
    }
    result.div_scalar(outputs.len() as f32)
}

fn combine_pattern_outputs(outputs: Vec<Tensor>, weights: Tensor) -> Result<Tensor> {
    if outputs.is_empty() {
        return Err(tensor_op_error(
            "tensor_operation",
            "Cannot combine empty outputs".to_string(),
        ));
    }

    // Weighted combination of pattern outputs
    // weights should have shape [batch_size, num_patterns]
    // outputs should be a vector of tensors with shape [batch_size, seq_len, hidden_size]
    let mut result = outputs[0].mul_scalar(0.0)?; // Initialize with zeros of the right shape

    for (i, output) in outputs.iter().enumerate() {
        // Extract weight for this pattern and expand to match output dimensions
        let pattern_weight = weights.select(1, i as i64)?; // Select weight for pattern i
        let expanded_weight = pattern_weight.unsqueeze(1)?.unsqueeze(2)?; // Expand to [batch, 1, 1]

        // Apply weight and add to result
        let weighted_output = output.mul(&expanded_weight)?;
        result = result.add(&weighted_output)?;
    }

    Ok(result)
}
