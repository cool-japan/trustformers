use crate::performer::config::PerformerConfig;
use std::io::Read;
use trustformers_core::{
    errors::Result,
    layers::{Embedding, LayerNorm, Linear},
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

/// FAVOR+ attention mechanism for linear complexity
/// Approximates softmax attention using positive random features
pub struct FavorPlusAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,

    num_attention_heads: usize,
    attention_head_size: usize,
    num_random_features: usize,
    kernel_type: String,
    causal: bool,
    normalize_features: bool,
    numerical_stabilizer: f32,

    // Random feature matrices (would be redrawn periodically in training)
    random_features: Option<Tensor>,
}

impl FavorPlusAttention {
    pub fn new(config: &PerformerConfig) -> Result<Self> {
        let attention_head_size = config.head_dim();
        let all_head_size = config.num_attention_heads * attention_head_size;

        let query = Linear::new(config.hidden_size, all_head_size, true);
        let key = Linear::new(config.hidden_size, all_head_size, true);
        let value = Linear::new(config.hidden_size, all_head_size, true);
        let output = Linear::new(all_head_size, config.hidden_size, true);

        Ok(Self {
            query,
            key,
            value,
            output,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            num_random_features: config.num_random_features,
            kernel_type: config.kernel_type.clone(),
            causal: config.causal_attention,
            normalize_features: config.normalize_features,
            numerical_stabilizer: config.numerical_stabilizer,
            random_features: None,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.query.parameter_count()
            + self.key.parameter_count()
            + self.value.parameter_count()
            + self.output.parameter_count()
    }

    /// Generate random features for FAVOR+ approximation
    fn generate_random_features(&self, _device: &str) -> Result<Tensor> {
        // Generate random Gaussian matrix: [head_dim, num_random_features]
        let random_matrix = Tensor::randn(&[self.attention_head_size, self.num_random_features])?;

        if self.normalize_features {
            // Normalize to unit length along the feature dimension
            // Compute L2 norm across the feature dimension (axis 1)
            let squared = random_matrix.mul(&random_matrix)?;
            let sum_squared = squared.sum(None, false)?; // Sum across all dimensions
            let norm = sum_squared.sqrt()?;

            // Add small epsilon for numerical stability
            let eps = Tensor::scalar(1e-8)?;
            let stable_norm = norm.add(&eps)?;

            // Normalize by broadcasting the norm
            random_matrix.div(&stable_norm)
        } else {
            Ok(random_matrix)
        }
    }

    /// Apply feature map function φ(x) based on kernel type
    fn apply_feature_map(&self, x: &Tensor, random_features: &Tensor) -> Result<Tensor> {
        // x: [batch, heads, seq_len, head_dim]
        // random_features: [head_dim, num_random_features]

        let _batch_size = x.shape()[0];
        let _num_heads = x.shape()[1];
        let _seq_len = x.shape()[2];

        // Project: x @ random_features -> [batch, heads, seq_len, num_random_features]
        let projections = x.matmul(random_features)?;

        match self.kernel_type.as_str() {
            "relu" => {
                // ReLU kernel: φ(x) = sqrt(2/m) * max(0, x @ w)
                let scale = (2.0 / self.num_random_features as f32).sqrt();
                let features = projections.relu()?.mul_scalar(scale)?;
                Ok(features)
            },
            "exp" => {
                // Exponential kernel: φ(x) = exp(x @ w - ||x||²/2) / sqrt(m)
                let x_norm_sq = x.pow(2.0)?.sum(Some(vec![x.shape().len() - 1]), true)?; // [batch, heads, seq_len, 1]
                let scaled_proj = projections.sub(&x_norm_sq.mul_scalar(0.5)?)?;
                let features = scaled_proj
                    .exp()?
                    .mul_scalar(1.0 / (self.num_random_features as f32).sqrt())?;
                Ok(features)
            },
            "softmax+" => {
                // Positive features for softmax approximation
                let x_norm_sq = x.pow(2.0)?.sum(Some(vec![x.shape().len() - 1]), true)?;
                let h = self.attention_head_size as f32;

                // φ(x) = exp(x @ w - ||x||²/2) / sqrt(m) for better softmax approximation
                let scaled_proj = projections.sub(&x_norm_sq.mul_scalar(0.5)?)?;
                let features =
                    scaled_proj.exp()?.mul_scalar((h / self.num_random_features as f32).sqrt())?;
                Ok(features)
            },
            _ => {
                // Default to ReLU
                let scale = (2.0 / self.num_random_features as f32).sqrt();
                let features = projections.relu()?.mul_scalar(scale)?;
                Ok(features)
            },
        }
    }

    /// Compute FAVOR+ attention
    fn favor_attention(
        &self,
        query_features: &Tensor,
        key_features: &Tensor,
        values: &Tensor,
    ) -> Result<Tensor> {
        // query_features, key_features: [batch, heads, seq_len, num_random_features]
        // values: [batch, heads, seq_len, head_dim]

        if self.causal {
            // Causal attention: use cumulative sums
            self.causal_favor_attention(query_features, key_features, values)
        } else {
            // Non-causal attention: use matrix multiplication
            self.non_causal_favor_attention(query_features, key_features, values)
        }
    }

    fn non_causal_favor_attention(
        &self,
        query_features: &Tensor,
        key_features: &Tensor,
        values: &Tensor,
    ) -> Result<Tensor> {
        // Compute D = sum(key_features, dim=seq_len)
        // D: [batch, heads, num_random_features]
        let d = key_features.sum(Some(vec![2]), false)?;

        // Compute numerator: query_features @ (key_features^T @ values)
        // key_features^T: [batch, heads, num_random_features, seq_len]
        let key_features_t = key_features.transpose(
            key_features.shape().len() - 2,
            key_features.shape().len() - 1,
        )?;

        // kv: [batch, heads, num_random_features, head_dim]
        let kv = key_features_t.matmul(values)?;

        // numerator: [batch, heads, seq_len, head_dim]
        let numerator = query_features.matmul(&kv)?;

        // Compute denominator: query_features @ D
        // denominator: [batch, heads, seq_len, 1]
        let denominator = query_features.matmul(&d.unsqueeze(d.shape().len())?)?;
        let denominator = denominator.add_scalar(self.numerical_stabilizer)?;

        // Final attention output
        numerator.div(&denominator)
    }

    fn causal_favor_attention(
        &self,
        query_features: &Tensor,
        key_features: &Tensor,
        values: &Tensor,
    ) -> Result<Tensor> {
        let batch_size = query_features.shape()[0];
        let num_heads = query_features.shape()[1];
        let seq_len = query_features.shape()[2];
        let head_dim = values.shape()[3];

        // Initialize output
        let mut output = Tensor::zeros(&[batch_size, num_heads, seq_len, head_dim])?;

        // Running sums for causal attention
        let mut running_kv =
            Tensor::zeros(&[batch_size, num_heads, self.num_random_features, head_dim])?;
        let mut running_k = Tensor::zeros(&[batch_size, num_heads, self.num_random_features])?;

        // Process each position causally
        for i in 0..seq_len {
            // Get current query, key, value using proper tensor slicing
            let q_i = query_features.slice_multi(&[
                (0, batch_size),
                (0, num_heads),
                (i, i + 1),
                (0, self.num_random_features),
            ])?;
            let k_i = key_features.slice_multi(&[
                (0, batch_size),
                (0, num_heads),
                (i, i + 1),
                (0, self.num_random_features),
            ])?;
            let v_i = values.slice_multi(&[
                (0, batch_size),
                (0, num_heads),
                (i, i + 1),
                (0, head_dim),
            ])?;

            // Compute attention output for position i
            let numerator = q_i.matmul(&running_kv)?;
            let denominator = q_i.matmul(&running_k.unsqueeze(running_k.shape().len())?)?;
            let denominator = denominator.add_scalar(self.numerical_stabilizer)?;

            let att_output = numerator.div(&denominator)?;

            // Build output tensor by concatenating position outputs
            if i == 0 {
                output = att_output.clone();
            } else {
                output = Tensor::concat(&[output, att_output], 2)?;
            }

            // Update running sums
            let shape = k_i.shape();
            let dim0 = shape.len().saturating_sub(2);
            let dim1 = shape.len().saturating_sub(1);
            let k_i_t = k_i.transpose(dim0, dim1)?; // [batch, heads, num_random_features, 1]
            let kv_update = k_i_t.matmul(&v_i)?; // [batch, heads, num_random_features, head_dim]
            running_kv = running_kv.add(&kv_update)?;
            let shape = k_i.shape();
            let squeeze_dim = shape.len().saturating_sub(2);
            running_k = running_k.add(&k_i.squeeze(squeeze_dim)?)?;
        }

        Ok(output)
    }

    /// Transpose tensor for multi-head attention
    fn transpose_for_scores(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        // Reshape: [batch, seq, heads * head_dim] -> [batch, seq, heads, head_dim]
        let reshaped = x.reshape(&[
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ])?;

        // Permute: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        reshaped.permute(&[0, 2, 1, 3])
    }
}

impl Layer for FavorPlusAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];

        // Linear projections
        let query_layer = self.query.forward(input.clone())?;
        let key_layer = self.key.forward(input.clone())?;
        let value_layer = self.value.forward(input)?;

        // Transpose for multi-head attention
        let query_layer = self.transpose_for_scores(&query_layer)?;
        let key_layer = self.transpose_for_scores(&key_layer)?;
        let value_layer = self.transpose_for_scores(&value_layer)?;

        // Generate or reuse random features
        let random_features = if let Some(ref features) = self.random_features {
            features.clone()
        } else {
            self.generate_random_features("cpu")?
        };

        // Apply feature maps
        let query_features = self.apply_feature_map(&query_layer, &random_features)?;
        let key_features = self.apply_feature_map(&key_layer, &random_features)?;

        // Compute FAVOR+ attention
        let context_layer = self.favor_attention(&query_features, &key_features, &value_layer)?;

        // Transpose back: [batch, heads, seq_len, head_dim] -> [batch, seq_len, heads, head_dim]
        let context_layer = context_layer.permute(&[0, 2, 1, 3])?;

        // Reshape: [batch, seq_len, heads, head_dim] -> [batch, seq_len, heads * head_dim]
        let context_layer = context_layer.reshape(&[
            batch_size,
            seq_len,
            self.num_attention_heads * self.attention_head_size,
        ])?;

        // Apply output projection
        self.output.forward(context_layer)
    }
}

/// Performer feed-forward network (same as BERT)
pub struct PerformerFeedForward {
    dense1: Linear,
    dense2: Linear,
    activation: String,
    #[allow(dead_code)]
    dropout: f32,
}

impl PerformerFeedForward {
    pub fn new(config: &PerformerConfig) -> Result<Self> {
        let dense1 = Linear::new(config.hidden_size, config.intermediate_size, true);
        let dense2 = Linear::new(config.intermediate_size, config.hidden_size, true);

        Ok(Self {
            dense1,
            dense2,
            activation: config.hidden_act.clone(),
            dropout: config.hidden_dropout_prob,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.dense1.parameter_count() + self.dense2.parameter_count()
    }

    fn apply_activation(&self, x: &Tensor) -> Result<Tensor> {
        match self.activation.as_str() {
            "gelu" => x.gelu(),
            "relu" => x.relu(),
            "silu" | "swish" => x.silu(),
            _ => Ok(x.clone()),
        }
    }
}

impl Layer for PerformerFeedForward {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let hidden = self.dense1.forward(input);
        let hidden = hidden?;
        let hidden = self.apply_activation(&hidden)?;
        self.dense2.forward(hidden)
    }
}

/// Performer encoder layer
pub struct PerformerLayer {
    attention: FavorPlusAttention,
    feed_forward: PerformerFeedForward,
    attention_norm: LayerNorm,
    output_norm: LayerNorm,
}

impl PerformerLayer {
    pub fn new(config: &PerformerConfig) -> Result<Self> {
        let attention = FavorPlusAttention::new(config)?;
        let feed_forward = PerformerFeedForward::new(config)?;
        let attention_norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps);
        let output_norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps);

        Ok(Self {
            attention,
            feed_forward,
            attention_norm: attention_norm?,
            output_norm: output_norm?,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.attention.parameter_count()
            + self.feed_forward.parameter_count()
            + self.attention_norm.parameter_count()
            + self.output_norm.parameter_count()
    }
}

impl Layer for PerformerLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Multi-head attention with residual connection and layer norm
        let attention_output = self.attention.forward(input.clone())?;
        let attention_output = input.add(&attention_output)?;
        let attention_output = self.attention_norm.forward(attention_output)?;

        // Feed-forward with residual connection and layer norm
        let ff_output = self.feed_forward.forward(attention_output.clone())?;
        let output = attention_output.add(&ff_output)?;
        self.output_norm.forward(output)
    }
}

/// Performer embeddings (same as BERT)
pub struct PerformerEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    #[allow(dead_code)]
    dropout: f32,
}

impl PerformerEmbeddings {
    pub fn new(config: &PerformerConfig) -> Result<Self> {
        let word_embeddings = Embedding::new(
            config.vocab_size,
            config.hidden_size,
            Some(config.pad_token_id as usize),
        )?;
        let position_embeddings =
            Embedding::new(config.max_position_embeddings, config.hidden_size, None)?;
        let token_type_embeddings =
            Embedding::new(config.type_vocab_size, config.hidden_size, None)?;
        let layer_norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout: config.hidden_dropout_prob,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.word_embeddings.parameter_count()
            + self.position_embeddings.parameter_count()
            + self.token_type_embeddings.parameter_count()
            + self.layer_norm.parameter_count()
    }
}

impl Layer for PerformerEmbeddings {
    type Input = (Vec<u32>, Option<Vec<u32>>, Option<Vec<u32>>);
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let (input_ids, token_type_ids, position_ids) = input;
        let seq_len = input_ids.len();

        let words_embeddings = self.word_embeddings.forward(input_ids)?;

        let position_ids = position_ids.unwrap_or_else(|| (0..seq_len as u32).collect());
        let position_embeddings = self.position_embeddings.forward(position_ids)?;

        let token_type_ids = token_type_ids.unwrap_or_else(|| vec![0; seq_len]);
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;

        let embeddings = words_embeddings.add(&position_embeddings)?.add(&token_type_embeddings)?;
        let embeddings = self.layer_norm.forward(embeddings)?;

        Ok(embeddings)
    }
}

/// Performer encoder
pub struct PerformerEncoder {
    layers: Vec<PerformerLayer>,
}

impl PerformerEncoder {
    pub fn new(config: &PerformerConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(PerformerLayer::new(config)?);
        }

        Ok(Self { layers })
    }

    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(|layer| layer.parameter_count()).sum()
    }
}

impl Layer for PerformerEncoder {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let mut hidden_states = input;

        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states)?;
        }

        Ok(hidden_states)
    }
}

/// Performer model
pub struct PerformerModel {
    config: PerformerConfig,
    embeddings: PerformerEmbeddings,
    encoder: PerformerEncoder,
}

impl PerformerModel {
    pub fn new(config: PerformerConfig) -> Result<Self> {
        config.validate()?;

        let embeddings = PerformerEmbeddings::new(&config)?;
        let encoder = PerformerEncoder::new(&config)?;

        Ok(Self {
            config,
            embeddings,
            encoder,
        })
    }
}

impl Model for PerformerModel {
    type Config = PerformerConfig;
    type Input = (Vec<u32>, Option<Vec<u32>>, Option<Vec<u32>>);
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let embeddings = self.embeddings.forward(input)?;
        let sequence_output = self.encoder.forward(embeddings)?;
        Ok(sequence_output)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Ok(())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        self.embeddings.parameter_count() + self.encoder.parameter_count()
    }
}

/// Performer for sequence classification
pub struct PerformerForSequenceClassification {
    performer: PerformerModel,
    classifier: Linear,
    #[allow(dead_code)]
    num_labels: usize,
}

impl PerformerForSequenceClassification {
    pub fn new(config: PerformerConfig, num_labels: usize) -> Result<Self> {
        let performer = PerformerModel::new(config.clone())?;
        let classifier = Linear::new(config.hidden_size, num_labels, true);

        Ok(Self {
            performer,
            classifier,
            num_labels,
        })
    }
}

impl Model for PerformerForSequenceClassification {
    type Config = PerformerConfig;
    type Input = (Vec<u32>, Option<Vec<u32>>, Option<Vec<u32>>);
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let sequence_output = self.performer.forward(input)?;
        let cls_output = sequence_output.slice(1, 0, 1)?; // Get first token (CLS) from sequence
        let cls_output = cls_output.squeeze(1)?; // Remove singleton sequence dimension
        self.classifier.forward(cls_output)
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.performer.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.performer.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.performer.num_parameters() + self.classifier.parameter_count()
    }
}

/// Performer for masked language modeling
pub struct PerformerForMaskedLM {
    performer: PerformerModel,
    mlm_head: Linear,
}

impl PerformerForMaskedLM {
    pub fn new(config: PerformerConfig) -> Result<Self> {
        let performer = PerformerModel::new(config.clone())?;
        let mlm_head = Linear::new(config.hidden_size, config.vocab_size, true);

        Ok(Self {
            performer,
            mlm_head,
        })
    }
}

impl Model for PerformerForMaskedLM {
    type Config = PerformerConfig;
    type Input = (Vec<u32>, Option<Vec<u32>>, Option<Vec<u32>>);
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let sequence_output = self.performer.forward(input)?;
        self.mlm_head.forward(sequence_output)
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.performer.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.performer.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.performer.num_parameters() + self.mlm_head.parameter_count()
    }
}
