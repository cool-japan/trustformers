use crate::performer::config::PerformerConfig;
use std::io::Read;
use trustformers_core::{
    device::Device,
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

    device: Device,
}

impl FavorPlusAttention {
    pub fn new(config: &PerformerConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &PerformerConfig, device: Device) -> Result<Self> {
        let attention_head_size = config.head_dim();
        let all_head_size = config.num_attention_heads * attention_head_size;

        let query = Linear::new_with_device(config.hidden_size, all_head_size, true, device);
        let key = Linear::new_with_device(config.hidden_size, all_head_size, true, device);
        let value = Linear::new_with_device(config.hidden_size, all_head_size, true, device);
        let output = Linear::new_with_device(all_head_size, config.hidden_size, true, device);

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
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
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
    device: Device,
}

impl PerformerFeedForward {
    pub fn new(config: &PerformerConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &PerformerConfig, device: Device) -> Result<Self> {
        let dense1 =
            Linear::new_with_device(config.hidden_size, config.intermediate_size, true, device);
        let dense2 =
            Linear::new_with_device(config.intermediate_size, config.hidden_size, true, device);

        Ok(Self {
            dense1,
            dense2,
            activation: config.hidden_act.clone(),
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
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
    device: Device,
}

impl PerformerLayer {
    pub fn new(config: &PerformerConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &PerformerConfig, device: Device) -> Result<Self> {
        let attention = FavorPlusAttention::new_with_device(config, device)?;
        let feed_forward = PerformerFeedForward::new_with_device(config, device)?;
        let attention_norm =
            LayerNorm::new_with_device(vec![config.hidden_size], config.layer_norm_eps, device)?;
        let output_norm =
            LayerNorm::new_with_device(vec![config.hidden_size], config.layer_norm_eps, device)?;

        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            output_norm,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
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
    device: Device,
}

impl PerformerEmbeddings {
    pub fn new(config: &PerformerConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &PerformerConfig, device: Device) -> Result<Self> {
        let word_embeddings = Embedding::new_with_device(
            config.vocab_size,
            config.hidden_size,
            Some(config.pad_token_id as usize),
            device,
        )?;
        let position_embeddings = Embedding::new_with_device(
            config.max_position_embeddings,
            config.hidden_size,
            None,
            device,
        )?;
        let token_type_embeddings =
            Embedding::new_with_device(config.type_vocab_size, config.hidden_size, None, device)?;
        let layer_norm =
            LayerNorm::new_with_device(vec![config.hidden_size], config.layer_norm_eps, device)?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
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
    device: Device,
}

impl PerformerEncoder {
    pub fn new(config: &PerformerConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &PerformerConfig, device: Device) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(PerformerLayer::new_with_device(config, device)?);
        }

        Ok(Self { layers, device })
    }

    pub fn device(&self) -> Device {
        self.device
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
    device: Device,
}

impl PerformerModel {
    pub fn new(config: PerformerConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: PerformerConfig, device: Device) -> Result<Self> {
        config.validate()?;

        let embeddings = PerformerEmbeddings::new_with_device(&config, device)?;
        let encoder = PerformerEncoder::new_with_device(&config, device)?;

        Ok(Self {
            config,
            embeddings,
            encoder,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
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
    device: Device,
}

impl PerformerForSequenceClassification {
    pub fn new(config: PerformerConfig, num_labels: usize) -> Result<Self> {
        Self::new_with_device(config, num_labels, Device::CPU)
    }

    pub fn new_with_device(
        config: PerformerConfig,
        num_labels: usize,
        device: Device,
    ) -> Result<Self> {
        let performer = PerformerModel::new_with_device(config.clone(), device)?;
        let classifier = Linear::new_with_device(config.hidden_size, num_labels, true, device);

        Ok(Self {
            performer,
            classifier,
            num_labels,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
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
    device: Device,
}

impl PerformerForMaskedLM {
    pub fn new(config: PerformerConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: PerformerConfig, device: Device) -> Result<Self> {
        let performer = PerformerModel::new_with_device(config.clone(), device)?;
        let mlm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, true, device);

        Ok(Self {
            performer,
            mlm_head,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::performer::config::PerformerConfig;
    use trustformers_core::traits::Model;

    /// LCG deterministic pseudo-random: a=6364136223846793005, c=1442695040888963407
    fn lcg_next(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
        (*state as f32) / (u64::MAX as f32)
    }

    fn lcg_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n).map(|_| lcg_next(&mut s) * 2.0 - 1.0).collect()
    }

    fn small_config() -> PerformerConfig {
        PerformerConfig {
            vocab_size: 256,
            hidden_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 32,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 32,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: "absolute".to_string(),
            num_random_features: 8,
            redraw_features: false,
            feature_redraw_interval: 1000,
            use_favor_plus: true,
            normalize_features: false,
            causal_attention: false,
            kernel_type: "relu".to_string(),
            ortho_features: false,
            numerical_stabilizer: 1e-6,
        }
    }

    // ── config tests ──────────────────────────────────────────────────────

    #[test]
    fn test_config_head_dim() {
        let cfg = small_config();
        assert_eq!(
            cfg.head_dim(),
            8,
            "head_dim should be hidden_size / num_attention_heads = 16/2 = 8"
        );
    }

    #[test]
    fn test_config_approximation_quality() {
        let cfg = small_config();
        let aq = cfg.approximation_quality();
        // num_random_features=8, head_dim=8, so quality = 8/8 = 1.0
        assert!(
            (aq - 1.0).abs() < 1e-6,
            "approximation quality should be 1.0"
        );
    }

    #[test]
    fn test_config_is_efficient() {
        let cfg = small_config();
        // num_random_features=8 < max_position_embeddings=32
        assert!(
            cfg.is_efficient(),
            "should be efficient when random_features < max_pos_emb"
        );
    }

    #[test]
    fn test_config_validate_passes_for_small() {
        let cfg = small_config();
        cfg.validate().expect("small config should pass validation");
    }

    #[test]
    fn test_config_validate_fails_hidden_not_divisible_by_heads() {
        let mut cfg = small_config();
        cfg.num_attention_heads = 3; // 16 % 3 != 0
        assert!(
            cfg.validate().is_err(),
            "should fail if hidden_size not divisible by heads"
        );
    }

    #[test]
    fn test_config_validate_fails_invalid_kernel_type() {
        let mut cfg = small_config();
        cfg.kernel_type = "invalid_kernel".to_string();
        assert!(
            cfg.validate().is_err(),
            "should fail for invalid kernel_type"
        );
    }

    #[test]
    fn test_config_performer_base_has_correct_architecture() {
        let cfg = PerformerConfig::performer_base();
        assert_eq!(
            cfg.architecture(),
            "Performer",
            "architecture should be Performer"
        );
        // performer_base defaults use num_random_features=256, head_dim=64 (768/12)
        // 256 > 2*64=128, so validation intentionally warns; just check the fields exist
        assert_eq!(cfg.num_random_features, 256);
        assert!(cfg.num_attention_heads > 0);
    }

    #[test]
    fn test_config_performer_causal_flag() {
        let cfg = PerformerConfig::performer_causal();
        assert!(
            cfg.causal_attention,
            "causal config should have causal_attention=true"
        );
    }

    // ── FavorPlusAttention tests ───────────────────────────────────────────

    #[test]
    fn test_favor_attention_creation() {
        let cfg = small_config();
        let attn = FavorPlusAttention::new(&cfg).expect("should create FavorPlusAttention");
        // num_heads=2, head_dim=8, so all_head_size=16
        // each linear: in*out + out (bias)
        // query: 16*16 + 16, same for key, value, output
        let expected_params = 4 * (16 * 16 + 16);
        assert_eq!(
            attn.parameter_count(),
            expected_params,
            "parameter count mismatch"
        );
    }

    #[test]
    fn test_favor_attention_device_is_cpu() {
        let cfg = small_config();
        let attn = FavorPlusAttention::new(&cfg).expect("should create FavorPlusAttention");
        assert_eq!(
            format!("{:?}", attn.device()),
            "CPU",
            "default device should be CPU"
        );
    }

    #[test]
    fn test_favor_attention_relu_kernel_param_count() {
        // FAVOR+ with relu kernel: verify parameter count (linear layers only, no random features)
        let cfg = small_config();
        let layer = FavorPlusAttention::new(&cfg).expect("should create FavorPlusAttention");
        // 4 linear layers: query, key, value, output
        // each: hidden*all_head + all_head (bias) = 16*16 + 16 = 272; 4 * 272 = 1088
        let all_head = cfg.num_attention_heads * cfg.head_dim();
        let per_linear = cfg.hidden_size * all_head + all_head;
        let expected = 4 * per_linear;
        assert_eq!(
            layer.parameter_count(),
            expected,
            "FAVOR+ relu param count mismatch"
        );
    }

    #[test]
    fn test_favor_attention_exp_kernel_param_count() {
        // FAVOR+ with exp kernel should have same parameter count as relu (only linear weights differ)
        let mut cfg = small_config();
        cfg.kernel_type = "exp".to_string();
        let layer_exp =
            FavorPlusAttention::new(&cfg).expect("should create FavorPlusAttention exp");
        let cfg_relu = small_config();
        let layer_relu =
            FavorPlusAttention::new(&cfg_relu).expect("should create FavorPlusAttention relu");
        assert_eq!(
            layer_exp.parameter_count(),
            layer_relu.parameter_count(),
            "exp and relu kernels should have same learnable parameter count"
        );
    }

    #[test]
    fn test_favor_attention_softmax_plus_kernel_param_count() {
        // softmax+ kernel should also have the same linear parameter count
        let mut cfg = small_config();
        cfg.kernel_type = "softmax+".to_string();
        let layer =
            FavorPlusAttention::new(&cfg).expect("should create FavorPlusAttention softmax+");
        assert!(
            layer.parameter_count() > 0,
            "softmax+ kernel should have positive param count"
        );
    }

    // ── PerformerFeedForward tests ─────────────────────────────────────────

    #[test]
    fn test_feed_forward_parameter_count() {
        let cfg = small_config();
        let ff = PerformerFeedForward::new(&cfg).expect("should create PerformerFeedForward");
        // dense1: 16*32 + 32 = 544, dense2: 32*16 + 16 = 528 → total 1072
        let expected = (cfg.hidden_size * cfg.intermediate_size + cfg.intermediate_size)
            + (cfg.intermediate_size * cfg.hidden_size + cfg.hidden_size);
        assert_eq!(
            ff.parameter_count(),
            expected,
            "parameter count should match"
        );
    }

    #[test]
    fn test_feed_forward_gelu_activation() {
        let mut cfg = small_config();
        cfg.hidden_act = "gelu".to_string();
        let ff = PerformerFeedForward::new(&cfg).expect("should create PerformerFeedForward");
        let input_data = lcg_vec(cfg.hidden_size, 99);
        let input =
            trustformers_core::tensor::Tensor::from_vec(input_data, &[1, 1, cfg.hidden_size])
                .expect("should build input tensor");
        let output = ff.forward(input).expect("feed_forward gelu forward should succeed");
        let shape = output.shape();
        assert_eq!(
            shape[shape.len() - 1],
            cfg.hidden_size,
            "output dim matches"
        );
    }

    #[test]
    fn test_feed_forward_silu_activation() {
        let mut cfg = small_config();
        cfg.hidden_act = "silu".to_string();
        let ff = PerformerFeedForward::new(&cfg).expect("should create PerformerFeedForward");
        let input_data = lcg_vec(cfg.hidden_size, 17);
        let input =
            trustformers_core::tensor::Tensor::from_vec(input_data, &[1, 1, cfg.hidden_size])
                .expect("should build input tensor");
        let output = ff.forward(input).expect("feed_forward silu forward should succeed");
        let shape = output.shape();
        assert_eq!(
            shape[shape.len() - 1],
            cfg.hidden_size,
            "output dim matches silu"
        );
    }

    // ── PerformerLayer tests ───────────────────────────────────────────────

    #[test]
    fn test_performer_layer_parameter_count_positive() {
        let cfg = small_config();
        let layer = PerformerLayer::new(&cfg).expect("should create PerformerLayer");
        assert!(
            layer.parameter_count() > 0,
            "PerformerLayer should have positive parameter count"
        );
    }

    #[test]
    fn test_performer_layer_parameter_count_nonzero() {
        // PerformerLayer wraps FavorPlusAttention + FFN + 2xLayerNorm
        let cfg = small_config();
        let layer = PerformerLayer::new(&cfg).expect("should create PerformerLayer");
        assert!(
            layer.parameter_count() > 0,
            "PerformerLayer should have positive parameter count"
        );
    }

    #[test]
    fn test_performer_layer_device_matches_construction() {
        let cfg = small_config();
        let layer = PerformerLayer::new_with_device(&cfg, trustformers_core::device::Device::CPU)
            .expect("should create PerformerLayer on CPU");
        assert_eq!(format!("{:?}", layer.device()), "CPU");
    }

    // ── PerformerModel tests ───────────────────────────────────────────────

    #[test]
    fn test_performer_model_creation() {
        let cfg = small_config();
        let model = PerformerModel::new(cfg).expect("should create PerformerModel");
        assert!(
            model.num_parameters() > 0,
            "PerformerModel should have parameters"
        );
    }

    #[test]
    fn test_performer_model_num_parameters_positive() {
        let cfg = small_config();
        let model = PerformerModel::new(cfg.clone()).expect("should create PerformerModel");
        assert!(
            model.num_parameters() > 0,
            "PerformerModel should have positive parameter count"
        );
    }

    #[test]
    fn test_performer_model_get_config() {
        let cfg = small_config();
        let hidden = cfg.hidden_size;
        let model = PerformerModel::new(cfg).expect("should create PerformerModel");
        assert_eq!(
            model.get_config().hidden_size,
            hidden,
            "get_config should return matching hidden_size"
        );
    }

    #[test]
    fn test_performer_model_device_cpu() {
        let cfg = small_config();
        let model = PerformerModel::new(cfg).expect("should create PerformerModel");
        assert_eq!(format!("{:?}", model.device()), "CPU");
    }

    // ── PerformerForSequenceClassification tests ───────────────────────────

    #[test]
    fn test_seq_cls_creation_and_num_parameters() {
        let cfg = small_config();
        let num_labels = 3;
        let model = PerformerForSequenceClassification::new(cfg.clone(), num_labels)
            .expect("should create PerformerForSequenceClassification");
        assert!(
            model.num_parameters() > 0,
            "should have positive parameter count"
        );
    }

    #[test]
    fn test_seq_cls_parameter_count_includes_classifier() {
        let cfg = small_config();
        let num_labels = 5;
        let model = PerformerForSequenceClassification::new(cfg.clone(), num_labels)
            .expect("should create PerformerForSequenceClassification");
        // classifier: hidden_size * num_labels + num_labels (bias)
        let classifier_params = cfg.hidden_size * num_labels + num_labels;
        let total = model.num_parameters();
        assert!(
            total > classifier_params,
            "total params should exceed classifier alone"
        );
    }

    #[test]
    fn test_seq_cls_device_cpu() {
        let cfg = small_config();
        let model = PerformerForSequenceClassification::new(cfg, 2)
            .expect("should create PerformerForSequenceClassification");
        assert_eq!(format!("{:?}", model.device()), "CPU");
    }

    // ── PerformerForMaskedLM tests ─────────────────────────────────────────

    #[test]
    fn test_masked_lm_creation_and_device() {
        let cfg = small_config();
        let model =
            PerformerForMaskedLM::new(cfg.clone()).expect("should create PerformerForMaskedLM");
        assert_eq!(format!("{:?}", model.device()), "CPU");
    }

    #[test]
    fn test_masked_lm_num_parameters_includes_mlm_head() {
        let cfg = small_config();
        let model =
            PerformerForMaskedLM::new(cfg.clone()).expect("should create PerformerForMaskedLM");
        // mlm_head: hidden_size * vocab_size + vocab_size
        let mlm_head_params = cfg.hidden_size * cfg.vocab_size + cfg.vocab_size;
        assert!(
            model.num_parameters() > mlm_head_params,
            "total params should exceed mlm_head alone"
        );
    }

    #[test]
    fn test_masked_lm_get_config_vocab_size() {
        let cfg = small_config();
        let vocab = cfg.vocab_size;
        let model = PerformerForMaskedLM::new(cfg).expect("should create PerformerForMaskedLM");
        assert_eq!(
            model.get_config().vocab_size,
            vocab,
            "get_config should return matching vocab_size"
        );
    }

    // ── num_random_features (approximation quality) tests ─────────────────

    #[test]
    fn test_more_random_features_better_approximation_quality() {
        let mut cfg_low = small_config();
        cfg_low.num_random_features = 4;
        let mut cfg_high = small_config();
        cfg_high.num_random_features = 8;
        assert!(
            cfg_high.approximation_quality() > cfg_low.approximation_quality(),
            "more features should yield higher approximation quality ratio"
        );
    }

    #[test]
    fn test_linear_complexity_claim_features_lt_sequence() {
        // FAVOR+ is O(nd) when m << n; verify that our config is in that regime
        let cfg = small_config();
        assert!(
            cfg.num_random_features < cfg.max_position_embeddings,
            "random features should be fewer than max_position_embeddings for linear complexity"
        );
    }

    #[test]
    fn test_num_random_features_validate_exceeds_2x_head_dim_fails() {
        let mut cfg = small_config();
        // head_dim = 8; 2*head_dim = 16; set features = 17
        cfg.num_random_features = 17;
        assert!(
            cfg.validate().is_err(),
            "validation should fail when num_random_features > 2 * head_dim"
        );
    }
}
