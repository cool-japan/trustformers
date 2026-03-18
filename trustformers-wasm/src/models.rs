use crate::core::tensor::WasmTensor;
use crate::layers::{gelu, softmax, Dropout, Embedding, LayerNorm, Linear};
use serde::{Deserialize, Serialize};
use std::string::ToString;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
}

#[wasm_bindgen]
impl BertConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> BertConfig {
        BertConfig {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
        }
    }

    pub fn tiny() -> BertConfig {
        BertConfig {
            vocab_size: 30522,
            hidden_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 512,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
        }
    }
}

impl Default for BertConfig {
    fn default() -> Self {
        Self::new()
    }
}

pub struct AttentionHead {
    query: Linear,
    key: Linear,
    value: Linear,
    dropout: Dropout,
    head_dim: usize,
}

impl AttentionHead {
    pub fn new(
        hidden_size: usize,
        head_dim: usize,
        dropout_prob: f32,
    ) -> Result<AttentionHead, JsValue> {
        Ok(AttentionHead {
            query: Linear::new(hidden_size, head_dim, true)?,
            key: Linear::new(hidden_size, head_dim, true)?,
            value: Linear::new(hidden_size, head_dim, true)?,
            dropout: Dropout::new(dropout_prob),
            head_dim,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &WasmTensor,
        attention_mask: Option<WasmTensor>,
    ) -> Result<WasmTensor, JsValue> {
        // Compute Q, K, V
        let q = self.query.forward(hidden_states)?;
        let k = self.key.forward(hidden_states)?;
        let v = self.value.forward(hidden_states)?;

        // Compute attention scores
        let kt = k.transpose()?;
        let scores = q.matmul(&kt)?;

        // Scale scores
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let scaled_scores = WasmTensor::new(
            scores.data().iter().map(|&x| x * scale).collect(),
            scores.shape(),
        )?;

        // Apply attention mask if provided
        let masked_scores = if let Some(ref _mask) = attention_mask {
            // In real implementation, would apply mask properly
            scaled_scores
        } else {
            scaled_scores
        };

        // Apply softmax
        let attention_probs = softmax(&masked_scores, -1)?;
        let attention_probs = self.dropout.forward(&attention_probs)?;

        // Apply attention to values
        attention_probs.matmul(&v)
    }
}

pub struct MultiHeadAttention {
    heads: Vec<AttentionHead>,
    output_projection: Linear,
    num_heads: usize,
}

impl MultiHeadAttention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        dropout_prob: f32,
    ) -> Result<MultiHeadAttention, JsValue> {
        if hidden_size % num_heads != 0 {
            return Err(JsValue::from_str(
                "Hidden size must be divisible by number of heads",
            ));
        }

        let head_dim = hidden_size / num_heads;
        let mut heads = Vec::with_capacity(num_heads);

        for _ in 0..num_heads {
            heads.push(AttentionHead::new(hidden_size, head_dim, dropout_prob)?);
        }

        Ok(MultiHeadAttention {
            heads,
            output_projection: Linear::new(hidden_size, hidden_size, true)?,
            num_heads,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &WasmTensor,
        attention_mask: Option<WasmTensor>,
    ) -> Result<WasmTensor, JsValue> {
        if self.heads.is_empty() {
            return Err(JsValue::from_str("No attention heads"));
        }

        // Run all attention heads in parallel and collect outputs
        let mut head_outputs = Vec::new();

        for head in &self.heads {
            let head_output = head.forward(hidden_states, attention_mask.clone())?;
            head_outputs.push(head_output);
        }

        // Concatenate all head outputs along the last dimension
        let concatenated_output = self.concatenate_head_outputs(head_outputs)?;

        // Apply output projection
        self.output_projection.forward(&concatenated_output)
    }

    /// Concatenate outputs from all attention heads along the last dimension
    fn concatenate_head_outputs(
        &self,
        head_outputs: Vec<WasmTensor>,
    ) -> Result<WasmTensor, JsValue> {
        if head_outputs.is_empty() {
            return Err(JsValue::from_str("No head outputs to concatenate"));
        }

        if head_outputs.len() == 1 {
            return Ok(head_outputs.into_iter().next().expect("len == 1"));
        }

        // Get shape of first tensor to validate compatibility
        let first_shape = head_outputs[0].shape();
        let batch_size = first_shape[0];
        let seq_len = first_shape[1];
        let head_dim = first_shape[2];

        // Validate all heads have the same shape
        for (i, head_output) in head_outputs.iter().enumerate() {
            let shape = head_output.shape();
            if shape[0] != batch_size || shape[1] != seq_len || shape[2] != head_dim {
                return Err(JsValue::from_str(&format!("Head {} shape mismatch", i)));
            }
        }

        // Create concatenated tensor: [batch_size, seq_len, num_heads * head_dim]
        let concat_dim = self.num_heads * head_dim;
        let mut concat_data = Vec::with_capacity(batch_size * seq_len * concat_dim);

        // Concatenate data from all heads
        for batch_idx in 0..batch_size {
            for seq_idx in 0..seq_len {
                for head_output in &head_outputs {
                    let data = head_output.data();
                    let head_start = (batch_idx * seq_len + seq_idx) * head_dim;
                    concat_data.extend_from_slice(&data[head_start..head_start + head_dim]);
                }
            }
        }

        // Create new tensor with concatenated data
        WasmTensor::new(concat_data, vec![batch_size, seq_len, concat_dim])
    }
}

pub struct BertLayer {
    attention: MultiHeadAttention,
    attention_output_norm: LayerNorm,
    intermediate: Linear,
    output: Linear,
    output_norm: LayerNorm,
    dropout: Dropout,
}

impl BertLayer {
    pub fn new(config: &BertConfig) -> Result<BertLayer, JsValue> {
        Ok(BertLayer {
            attention: MultiHeadAttention::new(
                config.hidden_size,
                config.num_attention_heads,
                config.attention_probs_dropout_prob,
            )?,
            attention_output_norm: LayerNorm::new(vec![config.hidden_size], 1e-12)?,
            intermediate: Linear::new(config.hidden_size, config.intermediate_size, true)?,
            output: Linear::new(config.intermediate_size, config.hidden_size, true)?,
            output_norm: LayerNorm::new(vec![config.hidden_size], 1e-12)?,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }

    pub fn forward(
        &self,
        hidden_states: &WasmTensor,
        attention_mask: Option<WasmTensor>,
    ) -> Result<WasmTensor, JsValue> {
        // Self-attention
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        let attention_output = self.dropout.forward(&attention_output)?;

        // Add & Norm
        let attention_output = hidden_states.add(&attention_output)?;
        let attention_output = self.attention_output_norm.forward(&attention_output)?;

        // Feed-forward
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        let intermediate_output = gelu(&intermediate_output);
        let layer_output = self.output.forward(&intermediate_output)?;
        let layer_output = self.dropout.forward(&layer_output)?;

        // Add & Norm
        let layer_output = attention_output.add(&layer_output)?;
        self.output_norm.forward(&layer_output)
    }
}

#[wasm_bindgen]
pub struct BertModelWasm {
    embeddings: BertEmbeddings,
    layers: Vec<BertLayer>,
    config: BertConfig,
}

#[wasm_bindgen]
pub struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

#[wasm_bindgen]
impl BertEmbeddings {
    pub fn new(config: &BertConfig) -> Result<BertEmbeddings, JsValue> {
        Ok(BertEmbeddings {
            word_embeddings: Embedding::new(config.vocab_size, config.hidden_size)?,
            position_embeddings: Embedding::new(
                config.max_position_embeddings,
                config.hidden_size,
            )?,
            token_type_embeddings: Embedding::new(config.type_vocab_size, config.hidden_size)?,
            layer_norm: LayerNorm::new(vec![config.hidden_size], 1e-12)?,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }

    pub fn forward(
        &self,
        input_ids: &[usize],
        #[allow(unused_variables)] use_token_type_ids: bool,
    ) -> Result<WasmTensor, JsValue> {
        let seq_length = input_ids.len();

        // Get word embeddings
        let word_embeddings = self.word_embeddings.forward(input_ids)?;

        // Get position embeddings
        let position_ids: Vec<usize> = (0..seq_length).collect();
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;

        // Get token type embeddings
        let token_type_ids = vec![0; seq_length];
        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids)?;

        // Add all embeddings
        let embeddings = word_embeddings.add(&position_embeddings)?;
        let embeddings = embeddings.add(&token_type_embeddings)?;

        // Apply layer norm and dropout
        let embeddings = self.layer_norm.forward(&embeddings)?;
        self.dropout.forward(&embeddings)
    }
}

#[wasm_bindgen]
impl BertModelWasm {
    #[wasm_bindgen(constructor)]
    pub fn new(config: BertConfig) -> Result<BertModelWasm, JsValue> {
        let embeddings = BertEmbeddings::new(&config)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(BertLayer::new(&config)?);
        }

        Ok(BertModelWasm {
            embeddings,
            layers,
            config,
        })
    }

    pub fn forward(
        &self,
        input_ids: &[usize],
        attention_mask: Option<WasmTensor>,
    ) -> Result<WasmTensor, JsValue> {
        // Get embeddings
        let mut hidden_states = self.embeddings.forward(input_ids, false)?;

        // Pass through all layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask.clone())?;
        }

        Ok(hidden_states)
    }

    pub fn get_config(&self) -> BertConfig {
        self.config.clone()
    }
}

// Simple text generation model
#[wasm_bindgen]
pub struct TextGenerator {
    model: BertModelWasm,
    lm_head: Linear,
}

#[wasm_bindgen]
impl TextGenerator {
    #[wasm_bindgen(constructor)]
    pub fn new(config: BertConfig) -> Result<TextGenerator, JsValue> {
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;

        Ok(TextGenerator {
            model: BertModelWasm::new(config)?,
            lm_head: Linear::new(hidden_size, vocab_size, true)?,
        })
    }

    pub fn generate(&self, input_ids: &[usize], max_length: usize) -> Result<Vec<usize>, JsValue> {
        let mut generated = input_ids.to_vec();

        while generated.len() < max_length {
            // Get model output
            let hidden_states = self.model.forward(&generated, None)?;

            // Get last token's hidden state
            let seq_len = generated.len();
            let last_hidden = hidden_states
                .slice(&[seq_len - 1, 0], &[seq_len, self.model.config.hidden_size])
                .map_err(|e| JsValue::from_str(&e.to_string()))?;

            // Apply language modeling head
            let logits = self.lm_head.forward(&last_hidden)?;

            // Get argmax (greedy decoding)
            let next_token = logits
                .data()
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            generated.push(next_token);

            // Stop if we generate an end token (assuming 0 is end token)
            if next_token == 0 {
                break;
            }
        }

        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bert_config() {
        let config = BertConfig::new();
        assert_eq!(config.vocab_size, 30522);
        assert_eq!(config.hidden_size, 768);

        let tiny_config = BertConfig::tiny();
        assert_eq!(tiny_config.hidden_size, 128);
        assert_eq!(tiny_config.num_hidden_layers, 2);
    }

    #[test]
    fn test_bert_model_creation() {
        let config = BertConfig::tiny();
        let model = BertModelWasm::new(config).unwrap();
        assert_eq!(model.layers.len(), 2);
    }
}
