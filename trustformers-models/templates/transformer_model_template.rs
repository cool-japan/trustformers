//! {{MODEL_NAME}} - {{MODEL_DESCRIPTION}}
//!
//! This is an auto-generated model implementation from the TrustformeRS template.
//!
//! ## Architecture Details
//! {{ARCHITECTURE_DETAILS}}
//!
//! ## Usage Example
//! ```rust
//! use trustformers_models::{{model_name_snake}}::{{ModelName}};
//!
//! let model = {{ModelName}}::from_pretrained("{{model_id}}")?;
//! let outputs = model.forward(&inputs)?;
//! ```

use crate::{
    ModelBase, ModelConfig, ModelOutput, PreTrainedModel,
    utils::{activations::Activation, shape_utils::ShapeUtils},
};
use trustformers_core::{
    Result, Tensor, Module, ModuleList, Embedding, LayerNorm, Linear, Dropout,
    nn::{MultiHeadAttention, PositionalEncoding},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for {{MODEL_NAME}}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct {{ModelName}}Config {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Intermediate (FFN) size
    pub intermediate_size: usize,
    /// Hidden activation function
    pub hidden_act: Activation,
    /// Dropout probability
    pub hidden_dropout_prob: f32,
    /// Attention dropout probability
    pub attention_probs_dropout_prob: f32,
    /// Maximum sequence length
    pub max_position_embeddings: usize,
    /// Type vocabulary size (for models with token types)
    pub type_vocab_size: usize,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
    /// Initializer range for weights
    pub initializer_range: f32,
    /// Whether to use cache for generation
    pub use_cache: bool,
    /// Padding token ID
    pub pad_token_id: usize,
    /// Beginning of sequence token ID
    pub bos_token_id: Option<usize>,
    /// End of sequence token ID
    pub eos_token_id: Option<usize>,

    // Add model-specific configuration parameters here
    {{#if has_custom_params}}
    {{custom_params}}
    {{/if}}
}

impl Default for {{ModelName}}Config {
    fn default() -> Self {
        Self {
            vocab_size: {{default_vocab_size}},
            hidden_size: {{default_hidden_size}},
            num_hidden_layers: {{default_num_layers}},
            num_attention_heads: {{default_num_heads}},
            intermediate_size: {{default_intermediate_size}},
            hidden_act: Activation::{{default_activation}},
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: {{default_max_positions}},
            type_vocab_size: {{default_type_vocab_size}},
            layer_norm_eps: 1e-12,
            initializer_range: 0.02,
            use_cache: true,
            pad_token_id: 0,
            bos_token_id: {{default_bos_token_id}},
            eos_token_id: {{default_eos_token_id}},
            {{#if has_custom_params}}
            {{custom_params_defaults}}
            {{/if}}
        }
    }
}

impl ModelConfig for {{ModelName}}Config {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn num_labels(&self) -> Option<usize> {
        None
    }

    fn id2label(&self) -> Option<&HashMap<usize, String>> {
        None
    }

    fn label2id(&self) -> Option<&HashMap<String, usize>> {
        None
    }

    fn is_decoder(&self) -> bool {
        {{is_decoder}}
    }

    fn is_encoder_decoder(&self) -> bool {
        {{is_encoder_decoder}}
    }
}

/// {{MODEL_NAME}} Embeddings
struct {{ModelName}}Embeddings {
    word_embeddings: Embedding,
    {{#if use_position_embeddings}}
    position_embeddings: Embedding,
    {{/if}}
    {{#if use_token_type_embeddings}}
    token_type_embeddings: Embedding,
    {{/if}}
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl {{ModelName}}Embeddings {
    fn new(config: &{{ModelName}}Config) -> Result<Self> {
        Ok(Self {
            word_embeddings: Embedding::new(config.vocab_size, config.hidden_size)?,
            {{#if use_position_embeddings}}
            position_embeddings: Embedding::new(
                config.max_position_embeddings,
                config.hidden_size
            )?,
            {{/if}}
            {{#if use_token_type_embeddings}}
            token_type_embeddings: Embedding::new(
                config.type_vocab_size,
                config.hidden_size
            )?,
            {{/if}}
            layer_norm: LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let seq_length = input_ids.shape()[1];

        // Word embeddings
        let mut embeddings = self.word_embeddings.forward(input_ids)?;

        {{#if use_position_embeddings}}
        // Position embeddings
        let position_ids = match position_ids {
            Some(ids) => ids.clone(),
            None => Tensor::arange(0, seq_length as i64, 1)?
                .unsqueeze(0)?
                .expand(&[input_ids.shape()[0], seq_length])?,
        };
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;
        embeddings = embeddings.add(&position_embeddings)?;
        {{/if}}

        {{#if use_token_type_embeddings}}
        // Token type embeddings
        let token_type_ids = match token_type_ids {
            Some(ids) => ids.clone(),
            None => Tensor::zeros(&input_ids.shape())?,
        };
        let token_type_embeddings = self.token_type_embeddings.forward(&token_type_ids)?;
        embeddings = embeddings.add(&token_type_embeddings)?;
        {{/if}}

        // Layer norm and dropout
        embeddings = self.layer_norm.forward(&embeddings)?;
        embeddings = self.dropout.forward(&embeddings)?;

        Ok(embeddings)
    }
}

/// {{MODEL_NAME}} Attention Layer
struct {{ModelName}}Attention {
    self_attention: MultiHeadAttention,
    output_dense: Linear,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl {{ModelName}}Attention {
    fn new(config: &{{ModelName}}Config) -> Result<Self> {
        Ok(Self {
            self_attention: MultiHeadAttention::new(
                config.hidden_size,
                config.num_attention_heads,
                config.attention_probs_dropout_prob,
            )?,
            output_dense: Linear::new(config.hidden_size, config.hidden_size, true)?,
            layer_norm: LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        past_key_value: Option<(&Tensor, &Tensor)>,
        use_cache: bool,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        // Self-attention
        let (attention_output, new_key_value) = self.self_attention.forward_with_cache(
            hidden_states,
            hidden_states,
            hidden_states,
            attention_mask,
            past_key_value,
            use_cache,
        )?;

        // Output projection
        let attention_output = self.output_dense.forward(&attention_output)?;
        let attention_output = self.dropout.forward(&attention_output)?;

        // Residual connection and layer norm
        let attention_output = self.layer_norm.forward(&attention_output.add(hidden_states)?)?;

        Ok((attention_output, new_key_value))
    }
}

/// {{MODEL_NAME}} Feed-Forward Network
struct {{ModelName}}FFN {
    dense_1: Linear,
    dense_2: Linear,
    activation: Activation,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl {{ModelName}}FFN {
    fn new(config: &{{ModelName}}Config) -> Result<Self> {
        Ok(Self {
            dense_1: Linear::new(config.hidden_size, config.intermediate_size, true)?,
            dense_2: Linear::new(config.intermediate_size, config.hidden_size, true)?,
            activation: config.hidden_act.clone(),
            layer_norm: LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?,
            dropout: Dropout::new(config.hidden_dropout_prob),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // First linear transformation
        let hidden_states_transformed = self.dense_1.forward(hidden_states)?;
        let hidden_states_transformed = self.activation.forward(&hidden_states_transformed)?;

        // Second linear transformation
        let hidden_states_transformed = self.dense_2.forward(&hidden_states_transformed)?;
        let hidden_states_transformed = self.dropout.forward(&hidden_states_transformed)?;

        // Residual connection and layer norm
        let output = self.layer_norm.forward(&hidden_states_transformed.add(hidden_states)?)?;

        Ok(output)
    }
}

/// {{MODEL_NAME}} Layer
struct {{ModelName}}Layer {
    attention: {{ModelName}}Attention,
    ffn: {{ModelName}}FFN,
    {{#if has_cross_attention}}
    cross_attention: Option<{{ModelName}}Attention>,
    {{/if}}
}

impl {{ModelName}}Layer {
    fn new(config: &{{ModelName}}Config) -> Result<Self> {
        Ok(Self {
            attention: {{ModelName}}Attention::new(config)?,
            ffn: {{ModelName}}FFN::new(config)?,
            {{#if has_cross_attention}}
            cross_attention: if config.is_decoder {
                Some({{ModelName}}Attention::new(config)?)
            } else {
                None
            },
            {{/if}}
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        {{#if has_cross_attention}}
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        {{/if}}
        past_key_value: Option<(&Tensor, &Tensor)>,
        use_cache: bool,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        // Self-attention
        let (attention_output, new_key_value) = self.attention.forward(
            hidden_states,
            attention_mask,
            past_key_value,
            use_cache,
        )?;

        {{#if has_cross_attention}}
        // Cross-attention (if decoder)
        let attention_output = if let (Some(cross_attn), Some(encoder_states)) =
            (&self.cross_attention, encoder_hidden_states) {
            let (cross_output, _) = cross_attn.forward(
                &attention_output,
                encoder_attention_mask,
                None,
                false,
            )?;
            cross_output
        } else {
            attention_output
        };
        {{/if}}

        // Feed-forward network
        let output = self.ffn.forward(&attention_output)?;

        Ok((output, new_key_value))
    }
}

/// {{MODEL_NAME}} Encoder/Decoder Stack
struct {{ModelName}}Stack {
    layers: ModuleList<{{ModelName}}Layer>,
    final_layer_norm: Option<LayerNorm>,
}

impl {{ModelName}}Stack {
    fn new(config: &{{ModelName}}Config) -> Result<Self> {
        let mut layers = ModuleList::new();
        for _ in 0..config.num_hidden_layers {
            layers.push({{ModelName}}Layer::new(config)?);
        }

        Ok(Self {
            layers,
            final_layer_norm: {{#if use_final_layer_norm}}
                Some(LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?)
            {{else}}
                None
            {{/if}},
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        {{#if has_cross_attention}}
        encoder_hidden_states: Option<&Tensor>,
        encoder_attention_mask: Option<&Tensor>,
        {{/if}}
        past_key_values: Option<Vec<(Tensor, Tensor)>>,
        use_cache: bool,
    ) -> Result<(Tensor, Option<Vec<(Tensor, Tensor)>>)> {
        let mut hidden_states = hidden_states.clone();
        let mut new_key_values = if use_cache { Some(Vec::new()) } else { None };

        for (i, layer) in self.layers.iter().enumerate() {
            let past_kv = past_key_values.as_ref()
                .and_then(|kvs| kvs.get(i))
                .map(|(k, v)| (k.as_ref(), v.as_ref()));

            let (layer_output, layer_kv) = layer.forward(
                &hidden_states,
                attention_mask,
                {{#if has_cross_attention}}
                encoder_hidden_states,
                encoder_attention_mask,
                {{/if}}
                past_kv,
                use_cache,
            )?;

            hidden_states = layer_output;

            if let (Some(ref mut kvs), Some(kv)) = (&mut new_key_values, layer_kv) {
                kvs.push(kv);
            }
        }

        // Apply final layer norm if present
        if let Some(ref ln) = self.final_layer_norm {
            hidden_states = ln.forward(&hidden_states)?;
        }

        Ok((hidden_states, new_key_values))
    }
}

/// {{MODEL_NAME}} Model
pub struct {{ModelName}} {
    config: {{ModelName}}Config,
    embeddings: {{ModelName}}Embeddings,
    {{#if is_encoder_decoder}}
    encoder: {{ModelName}}Stack,
    decoder: {{ModelName}}Stack,
    {{else}}
    transformer: {{ModelName}}Stack,
    {{/if}}
    {{#if has_pooler}}
    pooler: Option<{{ModelName}}Pooler>,
    {{/if}}
}

impl {{ModelName}} {
    /// Create a new {{MODEL_NAME}} model
    pub fn new(config: {{ModelName}}Config) -> Result<Self> {
        Ok(Self {
            embeddings: {{ModelName}}Embeddings::new(&config)?,
            {{#if is_encoder_decoder}}
            encoder: {{ModelName}}Stack::new(&config)?,
            decoder: {{ModelName}}Stack::new(&config)?,
            {{else}}
            transformer: {{ModelName}}Stack::new(&config)?,
            {{/if}}
            {{#if has_pooler}}
            pooler: Some({{ModelName}}Pooler::new(&config)?),
            {{/if}}
            config,
        })
    }

    /// Forward pass through the model
    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        token_type_ids: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        {{#if is_encoder_decoder}}
        decoder_input_ids: Option<&Tensor>,
        decoder_attention_mask: Option<&Tensor>,
        {{/if}}
        past_key_values: Option<Vec<(Tensor, Tensor)>>,
        use_cache: bool,
    ) -> Result<{{ModelName}}Output> {
        // Compute embeddings
        let hidden_states = self.embeddings.forward(
            input_ids,
            token_type_ids,
            position_ids,
        )?;

        // Prepare attention mask
        let attention_mask = self.prepare_attention_mask(attention_mask, input_ids)?;

        {{#if is_encoder_decoder}}
        // Encoder forward pass
        let (encoder_hidden_states, _) = self.encoder.forward(
            &hidden_states,
            attention_mask.as_ref(),
            None,
            None,
            None,
            false,
        )?;

        // Decoder forward pass
        let decoder_embeds = if let Some(dec_ids) = decoder_input_ids {
            self.embeddings.forward(dec_ids, None, None)?
        } else {
            return Err(TrustformersError::ValueError { message: "Decoder input IDs required for encoder-decoder model".into()));
        };

        let (hidden_states, new_key_values) = self.decoder.forward(
            &decoder_embeds,
            decoder_attention_mask,
            Some(&encoder_hidden_states),
            attention_mask.as_ref(),
            past_key_values,
            use_cache,
        )?;
        {{else}}
        // Transformer forward pass
        let (hidden_states, new_key_values) = self.transformer.forward(
            &hidden_states,
            attention_mask.as_ref(),
            past_key_values,
            use_cache,
        )?;
        {{/if}}

        {{#if has_pooler}}
        // Pooling
        let pooled_output = self.pooler.as_ref()
            .map(|pooler| pooler.forward(&hidden_states))
            .transpose()?;
        {{/if}}

        Ok({{ModelName}}Output {
            last_hidden_state: hidden_states,
            {{#if has_pooler}}
            pooler_output: pooled_output,
            {{/if}}
            past_key_values: new_key_values,
            {{#if is_encoder_decoder}}
            encoder_last_hidden_state: Some(encoder_hidden_states),
            {{/if}}
        })
    }

    /// Prepare attention mask
    fn prepare_attention_mask(
        &self,
        attention_mask: Option<&Tensor>,
        input_ids: &Tensor,
    ) -> Result<Option<Tensor>> {
        match attention_mask {
            Some(mask) => {
                // Convert to correct format (1.0 for attended, large negative for masked)
                let extended_mask = mask.unsqueeze(1)?.unsqueeze(2)?;
                let extended_mask = extended_mask.to_dtype(DataType::Float32)?;
                let extended_mask = (Tensor::ones_like(&extended_mask)? - extended_mask)? * -10000.0;
                Ok(Some(extended_mask))
            }
            None => {
                // Create default causal mask if needed
                {{#if is_decoder}}
                let seq_len = input_ids.shape()[1];
                let mask = self.create_causal_mask(seq_len)?;
                Ok(Some(mask))
                {{else}}
                Ok(None)
                {{/if}}
            }
        }
    }

    {{#if is_decoder}}
    /// Create causal attention mask
    fn create_causal_mask(&self, seq_len: usize) -> Result<Tensor> {
        let mask = Tensor::ones(&[seq_len, seq_len])?;
        let mask = mask.tril(0)?; // Lower triangular
        let mask = mask.unsqueeze(0)?.unsqueeze(0)?; // Add batch and head dims
        let mask = (Tensor::ones_like(&mask)? - mask)? * -10000.0;
        Ok(mask)
    }
    {{/if}}
}

{{#if has_pooler}}
/// {{MODEL_NAME}} Pooler
struct {{ModelName}}Pooler {
    dense: Linear,
    activation: Activation,
}

impl {{ModelName}}Pooler {
    fn new(config: &{{ModelName}}Config) -> Result<Self> {
        Ok(Self {
            dense: Linear::new(config.hidden_size, config.hidden_size, true)?,
            activation: Activation::Tanh,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Pool the first token (CLS token)
        let first_token = hidden_states.select(1, 0)?;
        let pooled = self.dense.forward(&first_token)?;
        let pooled = self.activation.forward(&pooled)?;
        Ok(pooled)
    }
}
{{/if}}

/// Output from {{MODEL_NAME}}
#[derive(Debug)]
pub struct {{ModelName}}Output {
    /// Last hidden states from the model
    pub last_hidden_state: Tensor,
    {{#if has_pooler}}
    /// Pooled output (if model has pooler)
    pub pooler_output: Option<Tensor>,
    {{/if}}
    /// Past key values for generation
    pub past_key_values: Option<Vec<(Tensor, Tensor)>>,
    {{#if is_encoder_decoder}}
    /// Encoder last hidden state
    pub encoder_last_hidden_state: Option<Tensor>,
    {{/if}}
}

impl ModelOutput for {{ModelName}}Output {
    fn logits(&self) -> Option<&Tensor> {
        None // Base model doesn't have logits
    }

    fn loss(&self) -> Option<&Tensor> {
        None
    }

    fn hidden_states(&self) -> Option<&Vec<Tensor>> {
        None
    }

    fn attentions(&self) -> Option<&Vec<Tensor>> {
        None
    }
}

impl ModelBase for {{ModelName}} {
    type Config = {{ModelName}}Config;
    type Output = {{ModelName}}Output;

    fn new(config: Self::Config) -> Result<Self> {
        Self::new(config)
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Self::Output> {
        self.forward(
            input_ids,
            attention_mask,
            None, // token_type_ids
            None, // position_ids
            {{#if is_encoder_decoder}}
            None, // decoder_input_ids
            None, // decoder_attention_mask
            {{/if}}
            None, // past_key_values
            false, // use_cache
        )
    }
}

impl Module for {{ModelName}} {
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();

        // Collect parameters from embeddings
        params.extend(self.embeddings.word_embeddings.parameters());
        {{#if use_position_embeddings}}
        params.extend(self.embeddings.position_embeddings.parameters());
        {{/if}}
        {{#if use_token_type_embeddings}}
        params.extend(self.embeddings.token_type_embeddings.parameters());
        {{/if}}
        params.extend(self.embeddings.layer_norm.parameters());

        // Collect parameters from transformer layers
        {{#if is_encoder_decoder}}
        for layer in &self.encoder.layers {
            params.extend(collect_layer_params(layer));
        }
        for layer in &self.decoder.layers {
            params.extend(collect_layer_params(layer));
        }
        {{else}}
        for layer in &self.transformer.layers {
            params.extend(collect_layer_params(layer));
        }
        {{/if}}

        {{#if has_pooler}}
        if let Some(ref pooler) = self.pooler {
            params.extend(pooler.dense.parameters());
        }
        {{/if}}

        params
    }
}

fn collect_layer_params(layer: &{{ModelName}}Layer) -> Vec<&Tensor> {
    let mut params = Vec::new();

    // Attention parameters
    params.extend(layer.attention.self_attention.parameters());
    params.extend(layer.attention.output_dense.parameters());
    params.extend(layer.attention.layer_norm.parameters());

    {{#if has_cross_attention}}
    if let Some(ref cross_attn) = layer.cross_attention {
        params.extend(cross_attn.self_attention.parameters());
        params.extend(cross_attn.output_dense.parameters());
        params.extend(cross_attn.layer_norm.parameters());
    }
    {{/if}}

    // FFN parameters
    params.extend(layer.ffn.dense_1.parameters());
    params.extend(layer.ffn.dense_2.parameters());
    params.extend(layer.ffn.layer_norm.parameters());

    params
}

// Model-specific heads (to be implemented in separate files)

/// {{MODEL_NAME}} for Masked Language Modeling
pub struct {{ModelName}}ForMaskedLM {
    {{model_name_snake}}: {{ModelName}},
    lm_head: Linear,
}

/// {{MODEL_NAME}} for Sequence Classification
pub struct {{ModelName}}ForSequenceClassification {
    {{model_name_snake}}: {{ModelName}},
    classifier: Linear,
    dropout: Dropout,
}

/// {{MODEL_NAME}} for Token Classification
pub struct {{ModelName}}ForTokenClassification {
    {{model_name_snake}}: {{ModelName}},
    classifier: Linear,
    dropout: Dropout,
}

/// {{MODEL_NAME}} for Question Answering
pub struct {{ModelName}}ForQuestionAnswering {
    {{model_name_snake}}: {{ModelName}},
    qa_outputs: Linear,
}

{{#if is_encoder_decoder}}
/// {{MODEL_NAME}} for Conditional Generation
pub struct {{ModelName}}ForConditionalGeneration {
    {{model_name_snake}}: {{ModelName}},
    lm_head: Linear,
}
{{/if}}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_{{model_name_snake}}_config() {
        let config = {{ModelName}}Config::default();
        assert_eq!(config.hidden_size, {{default_hidden_size}});
        assert_eq!(config.num_hidden_layers, {{default_num_layers}});
        assert_eq!(config.num_attention_heads, {{default_num_heads}});
    }

    #[test]
    fn test_{{model_name_snake}}_creation() {
        let config = {{ModelName}}Config::default();
        let model = {{ModelName}}::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_{{model_name_snake}}_forward() {
        let config = {{ModelName}}Config {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 64,
            max_position_embeddings: 128,
            ..Default::default()
        };

        let model = {{ModelName}}::new(config).unwrap();

        // Create dummy input
        let batch_size = 2;
        let seq_len = 10;
        let input_ids = Tensor::randint(0, 100, &[batch_size, seq_len]).unwrap();

        // Forward pass
        let output = model.forward(&input_ids, None);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.last_hidden_state.shape(), &[batch_size, seq_len, 32]);
    }
}

// Re-export commonly used items
pub use self::{{ModelName}}Config as Config;
pub use self::{{ModelName}} as Model;