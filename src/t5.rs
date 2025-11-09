use tch::{nn, Tensor, Kind, Device};
use crate::models::{TransformerConfig, TransformerEncoder, TransformerDecoder, LayerNormTrait, LayerNorm, RMSNorm};
use crate::attention::{SelfAttention, CrossAttention};
use crate::layers::{FeedForward, ActivationType};
use crate::utils::{PositionalEncoding, PositionalEncodingType};
use std::collections::HashMap;

#[derive(Clone)]
pub struct T5Config {
    pub vocab_size: i64,
    pub d_model: i64,
    pub d_kv: i64,
    pub d_ff: i64,
    pub num_layers: i64,
    pub num_decoder_layers: i64,
    pub num_heads: i64,
    pub relative_attention_num_buckets: i64,
    pub relative_attention_max_distance: i64,
    pub dropout_rate: f64,
    pub layer_norm_eps: f64,
    pub initializer_factor: f64,
    pub feed_forward_proj: String,
    pub is_encoder_decoder: bool,
    pub use_cache: bool,
    pub pad_token_id: i64,
    pub eos_token_id: i64,
    pub decoder_start_token_id: i64,
}

impl Default for T5Config {
    fn default() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 512,
            d_kv: 64,
            d_ff: 2048,
            num_layers: 6,
            num_decoder_layers: 6,
            num_heads: 8,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_eps: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: "relu".to_string(),
            is_encoder_decoder: true,
            use_cache: true,
            pad_token_id: 0,
            eos_token_id: 1,
            decoder_start_token_id: 0,
        }
    }
}

pub struct T5Model {
    config: T5Config,
    shared: nn::Embedding,
    encoder: T5Stack,
    decoder: T5Stack,
}

pub struct T5Stack {
    embed_tokens: Option<nn::Embedding>,
    layers: Vec<T5Block>,
    final_layer_norm: Box<dyn LayerNormTrait>,
    dropout: nn::Dropout,
    config: T5Config,
}

pub struct T5Block {
    layer: Vec<T5LayerWrapper>,
    config: T5Config,
}

pub struct T5LayerWrapper {
    layer: T5Layer,
    layer_norm: Box<dyn LayerNormTrait>,
    dropout: nn::Dropout,
}

pub enum T5Layer {
    SelfAttention(T5Attention),
    CrossAttention(T5Attention),
    DenseReluDense(T5DenseReluDense),
    DenseGatedGeluDense(T5DenseGatedGeluDense),
}

pub struct T5Attention {
    q: nn::Linear,
    k: nn::Linear,
    v: nn::Linear,
    o: nn::Linear,
    relative_attention_bias: Option<nn::Embedding>,
    n_heads: i64,
    d_model: i64,
    d_kv: i64,
    inner_dim: i64,
    dropout: f64,
    has_relative_attention_bias: bool,
    relative_attention_num_buckets: i64,
    relative_attention_max_distance: i64,
}

pub struct T5DenseReluDense {
    wi: nn::Linear,
    wo: nn::Linear,
    dropout: nn::Dropout,
}

pub struct T5DenseGatedGeluDense {
    wi_0: nn::Linear,
    wi_1: nn::Linear,
    wo: nn::Linear,
    dropout: nn::Dropout,
}

impl T5Model {
    pub fn new(vs: &nn::Path, config: T5Config) -> Self {
        let shared = nn::embedding(vs / "shared", config.vocab_size, config.d_model, Default::default());

        let encoder = T5Stack::new(&(vs / "encoder"), config.clone(), true, false);
        let decoder = T5Stack::new(&(vs / "decoder"), config.clone(), false, true);

        Self {
            config,
            shared,
            encoder,
            decoder,
        }
    }

    pub fn forward(&self,
                   input_ids: Option<&Tensor>,
                   attention_mask: Option<&Tensor>,
                   decoder_input_ids: Option<&Tensor>,
                   decoder_attention_mask: Option<&Tensor>,
                   encoder_outputs: Option<&Tensor>) -> T5ModelOutput {

        let encoder_outputs = if let Some(encoder_outputs) = encoder_outputs {
            encoder_outputs.shallow_clone()
        } else if let Some(input_ids) = input_ids {
            let inputs_embeds = input_ids.apply(&self.shared);
            self.encoder.forward(&inputs_embeds, attention_mask, None, None)
        } else {
            panic!("Either input_ids or encoder_outputs must be provided");
        };

        let decoder_outputs = if let Some(decoder_input_ids) = decoder_input_ids {
            let decoder_inputs_embeds = decoder_input_ids.apply(&self.shared);
            self.decoder.forward(&decoder_inputs_embeds, decoder_attention_mask, Some(&encoder_outputs), attention_mask)
        } else {
            encoder_outputs.shallow_clone()
        };

        T5ModelOutput {
            last_hidden_state: decoder_outputs,
            encoder_last_hidden_state: Some(encoder_outputs),
        }
    }

    pub fn get_input_embeddings(&self) -> &nn::Embedding {
        &self.shared
    }

    pub fn get_encoder(&self) -> &T5Stack {
        &self.encoder
    }

    pub fn get_decoder(&self) -> &T5Stack {
        &self.decoder
    }
}

impl T5Stack {
    fn new(vs: &nn::Path, config: T5Config, is_decoder: bool, has_relative_attention_bias: bool) -> Self {
        let embed_tokens = if is_decoder {
            None // Decoder uses shared embeddings
        } else {
            None // Encoder uses shared embeddings
        };

        let mut layers = Vec::new();
        for i in 0..if is_decoder { config.num_decoder_layers } else { config.num_layers } {
            let layer_vs = vs / format!("block.{}", i);
            layers.push(T5Block::new(&layer_vs, config.clone(), is_decoder, i == 0 && has_relative_attention_bias));
        }

        let final_layer_norm = Box::new(LayerNorm::new(
            vs / "final_layer_norm",
            vec![config.d_model],
            config.layer_norm_eps,
            true,
        ));

        let dropout = nn::dropout(vs / "dropout", config.dropout_rate);

        Self {
            embed_tokens,
            layers,
            final_layer_norm,
            dropout,
            config,
        }
    }

    fn forward(&self,
               input_embeds: &Tensor,
               attention_mask: Option<&Tensor>,
               encoder_hidden_states: Option<&Tensor>,
               encoder_attention_mask: Option<&Tensor>) -> Tensor {

        let mut hidden_states = input_embeds.apply(&self.dropout);

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask);
        }

        self.final_layer_norm.forward(&hidden_states)
    }
}

impl T5Block {
    fn new(vs: &nn::Path, config: T5Config, is_decoder: bool, has_relative_attention_bias: bool) -> Self {
        let mut layer = Vec::new();

        // Self-attention layer
        let self_attention = T5LayerWrapper::new(
            &(vs / "layer.0"),
            T5Layer::SelfAttention(T5Attention::new(
                &(vs / "layer.0" / "SelfAttention"),
                config.clone(),
                has_relative_attention_bias,
            )),
            config.clone(),
        );
        layer.push(self_attention);

        // Cross-attention layer (only for decoder)
        if is_decoder {
            let cross_attention = T5LayerWrapper::new(
                &(vs / "layer.1"),
                T5Layer::CrossAttention(T5Attention::new(
                    &(vs / "layer.1" / "EncDecAttention"),
                    config.clone(),
                    false,
                )),
                config.clone(),
            );
            layer.push(cross_attention);
        }

        // Feed-forward layer
        let ff_layer_idx = if is_decoder { 2 } else { 1 };
        let feed_forward = if config.feed_forward_proj == "gated-gelu" {
            T5LayerWrapper::new(
                &(vs / format!("layer.{}", ff_layer_idx)),
                T5Layer::DenseGatedGeluDense(T5DenseGatedGeluDense::new(
                    &(vs / format!("layer.{}", ff_layer_idx) / "DenseReluDense"),
                    config.clone(),
                )),
                config.clone(),
            )
        } else {
            T5LayerWrapper::new(
                &(vs / format!("layer.{}", ff_layer_idx)),
                T5Layer::DenseReluDense(T5DenseReluDense::new(
                    &(vs / format!("layer.{}", ff_layer_idx) / "DenseReluDense"),
                    config.clone(),
                )),
                config.clone(),
            )
        };
        layer.push(feed_forward);

        Self { layer, config }
    }

    fn forward(&self,
               hidden_states: &Tensor,
               attention_mask: Option<&Tensor>,
               encoder_hidden_states: Option<&Tensor>,
               encoder_attention_mask: Option<&Tensor>) -> Tensor {

        let mut layer_output = hidden_states.shallow_clone();

        for layer_module in &self.layer {
            layer_output = layer_module.forward(&layer_output, attention_mask, encoder_hidden_states, encoder_attention_mask);
        }

        layer_output
    }
}

impl T5LayerWrapper {
    fn new(vs: &nn::Path, layer: T5Layer, config: T5Config) -> Self {
        let layer_norm = Box::new(LayerNorm::new(
            vs / "layer_norm",
            vec![config.d_model],
            config.layer_norm_eps,
            true,
        ));
        let dropout = nn::dropout(vs / "dropout", config.dropout_rate);

        Self {
            layer,
            layer_norm,
            dropout,
        }
    }

    fn forward(&self,
               hidden_states: &Tensor,
               attention_mask: Option<&Tensor>,
               encoder_hidden_states: Option<&Tensor>,
               encoder_attention_mask: Option<&Tensor>) -> Tensor {

        let normed_hidden_states = self.layer_norm.forward(hidden_states);

        let layer_output = match &self.layer {
            T5Layer::SelfAttention(attention) => {
                attention.forward(&normed_hidden_states, &normed_hidden_states, &normed_hidden_states, attention_mask)
            },
            T5Layer::CrossAttention(attention) => {
                if let Some(encoder_hidden_states) = encoder_hidden_states {
                    attention.forward(&normed_hidden_states, encoder_hidden_states, encoder_hidden_states, encoder_attention_mask)
                } else {
                    panic!("encoder_hidden_states must be provided for cross-attention");
                }
            },
            T5Layer::DenseReluDense(ff) => ff.forward(&normed_hidden_states),
            T5Layer::DenseGatedGeluDense(ff) => ff.forward(&normed_hidden_states),
        };

        hidden_states + layer_output.apply(&self.dropout)
    }
}

impl T5Attention {
    fn new(vs: &nn::Path, config: T5Config, has_relative_attention_bias: bool) -> Self {
        let inner_dim = config.num_heads * config.d_kv;

        let q = nn::linear(vs / "q", config.d_model, inner_dim, nn::LinearConfig { bias: false, ..Default::default() });
        let k = nn::linear(vs / "k", config.d_model, inner_dim, nn::LinearConfig { bias: false, ..Default::default() });
        let v = nn::linear(vs / "v", config.d_model, inner_dim, nn::LinearConfig { bias: false, ..Default::default() });
        let o = nn::linear(vs / "o", inner_dim, config.d_model, nn::LinearConfig { bias: false, ..Default::default() });

        let relative_attention_bias = if has_relative_attention_bias {
            Some(nn::embedding(vs / "relative_attention_bias", config.relative_attention_num_buckets, config.num_heads, Default::default()))
        } else {
            None
        };

        Self {
            q,
            k,
            v,
            o,
            relative_attention_bias,
            n_heads: config.num_heads,
            d_model: config.d_model,
            d_kv: config.d_kv,
            inner_dim,
            dropout: config.dropout_rate,
            has_relative_attention_bias,
            relative_attention_num_buckets: config.relative_attention_num_buckets,
            relative_attention_max_distance: config.relative_attention_max_distance,
        }
    }

    fn forward(&self, query: &Tensor, key: &Tensor, value: &Tensor, mask: Option<&Tensor>) -> Tensor {
        let batch_size = query.size()[0];
        let seq_len_q = query.size()[1];
        let seq_len_kv = key.size()[1];

        // Project to query, key, value
        let q = self.shape(&query.apply(&self.q), batch_size, seq_len_q);
        let k = self.shape(&key.apply(&self.k), batch_size, seq_len_kv);
        let v = self.shape(&value.apply(&self.v), batch_size, seq_len_kv);

        // Compute attention scores
        let scores = q.matmul(&k.transpose(-1, -2));

        // Add relative position bias if available
        let scores = if let Some(relative_attention_bias) = &self.relative_attention_bias {
            let position_bias = self.compute_bias(seq_len_q, seq_len_kv);
            scores + position_bias
        } else {
            scores
        };

        // Apply mask if provided
        let scores = if let Some(mask) = mask {
            scores.masked_fill(&mask.eq(0), f64::NEG_INFINITY)
        } else {
            scores
        };

        // Apply softmax and dropout
        let attention_weights = scores.softmax(-1, Kind::Float);
        let attention_weights = if self.dropout > 0.0 {
            attention_weights.dropout(self.dropout, true)
        } else {
            attention_weights
        };

        // Apply attention to values
        let attention_output = attention_weights.matmul(&v);

        // Reshape and project output
        let attention_output = self.unshape(&attention_output, batch_size, seq_len_q);
        attention_output.apply(&self.o)
    }

    fn shape(&self, tensor: &Tensor, batch_size: i64, seq_len: i64) -> Tensor {
        tensor
            .view([batch_size, seq_len, self.n_heads, self.d_kv])
            .transpose(1, 2)
    }

    fn unshape(&self, tensor: &Tensor, batch_size: i64, seq_len: i64) -> Tensor {
        tensor
            .transpose(1, 2)
            .contiguous()
            .view([batch_size, seq_len, self.inner_dim])
    }

    fn compute_bias(&self, query_length: i64, key_length: i64) -> Tensor {
        // Simplified relative position bias computation
        // In a full implementation, this would use the relative_attention_bias embedding
        let device = Device::Cpu; // This should be extracted from the input tensor
        let context_position = Tensor::arange(query_length, (Kind::Int64, device)).unsqueeze(1);
        let memory_position = Tensor::arange(key_length, (Kind::Int64, device)).unsqueeze(0);
        let relative_position = memory_position - context_position;

        // Convert to buckets (simplified)
        let relative_buckets = self.relative_position_to_bucket(
            &relative_position,
            true, // bidirectional
            self.relative_attention_num_buckets,
            self.relative_attention_max_distance,
        );

        if let Some(relative_attention_bias) = &self.relative_attention_bias {
            let values = relative_buckets.apply(relative_attention_bias);
            values.permute(&[2, 0, 1]).unsqueeze(0)
        } else {
            Tensor::zeros(&[1, self.n_heads, query_length, key_length], (Kind::Float, device))
        }
    }

    fn relative_position_to_bucket(&self, relative_position: &Tensor, bidirectional: bool, num_buckets: i64, max_distance: i64) -> Tensor {
        let mut ret = relative_position.shallow_clone();
        let mut n = num_buckets;

        if bidirectional {
            n = n / 2;
            ret = ret + (ret.lt(0).to_kind(Kind::Int64) * n);
            ret = ret.abs();
        } else {
            ret = ret.clamp_max(0) * -1;
        }

        // Half of the buckets are for exact increments in positions
        let max_exact = n / 2;
        let is_small = ret.lt(max_exact);

        // The other half of the buckets are for logarithmically bigger bins
        let val_if_large = max_exact +
            ((ret.to_kind(Kind::Float) / max_exact as f64).log() /
             (max_distance as f64 / max_exact as f64).log() *
             (n - max_exact) as f64).to_kind(Kind::Int64);
        let val_if_large = val_if_large.clamp_max(n - 1);

        ret.where_self(&is_small, &val_if_large)
    }
}

impl T5DenseReluDense {
    fn new(vs: &nn::Path, config: T5Config) -> Self {
        let wi = nn::linear(vs / "wi", config.d_model, config.d_ff, nn::LinearConfig { bias: false, ..Default::default() });
        let wo = nn::linear(vs / "wo", config.d_ff, config.d_model, nn::LinearConfig { bias: false, ..Default::default() });
        let dropout = nn::dropout(vs / "dropout", config.dropout_rate);

        Self { wi, wo, dropout }
    }

    fn forward(&self, hidden_states: &Tensor) -> Tensor {
        let hidden_states = hidden_states.apply(&self.wi).relu();
        let hidden_states = hidden_states.apply(&self.dropout);
        hidden_states.apply(&self.wo)
    }
}

impl T5DenseGatedGeluDense {
    fn new(vs: &nn::Path, config: T5Config) -> Self {
        let wi_0 = nn::linear(vs / "wi_0", config.d_model, config.d_ff, nn::LinearConfig { bias: false, ..Default::default() });
        let wi_1 = nn::linear(vs / "wi_1", config.d_model, config.d_ff, nn::LinearConfig { bias: false, ..Default::default() });
        let wo = nn::linear(vs / "wo", config.d_ff, config.d_model, nn::LinearConfig { bias: false, ..Default::default() });
        let dropout = nn::dropout(vs / "dropout", config.dropout_rate);

        Self { wi_0, wi_1, wo, dropout }
    }

    fn forward(&self, hidden_states: &Tensor) -> Tensor {
        let hidden_gelu = hidden_states.apply(&self.wi_0).gelu("none");
        let hidden_linear = hidden_states.apply(&self.wi_1);
        let hidden_states = hidden_gelu * hidden_linear;
        let hidden_states = hidden_states.apply(&self.dropout);
        hidden_states.apply(&self.wo)
    }
}

pub struct T5ModelOutput {
    pub last_hidden_state: Tensor,
    pub encoder_last_hidden_state: Option<Tensor>,
}

pub struct T5ForConditionalGeneration {
    model_dim: i64,
    shared: nn::Embedding,
    encoder: T5Stack,
    decoder: T5Stack,
    lm_head: nn::Linear,
    config: T5Config,
}

impl T5ForConditionalGeneration {
    pub fn new(vs: &nn::Path, config: T5Config) -> Self {
        let shared = nn::embedding(vs / "shared", config.vocab_size, config.d_model, Default::default());
        let encoder = T5Stack::new(&(vs / "encoder"), config.clone(), false, true);
        let decoder = T5Stack::new(&(vs / "decoder"), config.clone(), true, false);
        let lm_head = nn::linear(vs / "lm_head", config.d_model, config.vocab_size, nn::LinearConfig { bias: false, ..Default::default() });

        Self {
            model_dim: config.d_model,
            shared,
            encoder,
            decoder,
            lm_head,
            config,
        }
    }

    pub fn forward(&self,
                   input_ids: Option<&Tensor>,
                   attention_mask: Option<&Tensor>,
                   decoder_input_ids: Option<&Tensor>,
                   decoder_attention_mask: Option<&Tensor>,
                   labels: Option<&Tensor>) -> T5ConditionalGenerationOutput {

        // Encoder
        let encoder_outputs = if let Some(input_ids) = input_ids {
            let inputs_embeds = input_ids.apply(&self.shared) * (self.model_dim as f64).sqrt();
            self.encoder.forward(&inputs_embeds, attention_mask, None, None)
        } else {
            panic!("input_ids must be provided");
        };

        // Decoder
        let decoder_outputs = if let Some(decoder_input_ids) = decoder_input_ids {
            let decoder_inputs_embeds = decoder_input_ids.apply(&self.shared) * (self.model_dim as f64).sqrt();
            self.decoder.forward(&decoder_inputs_embeds, decoder_attention_mask, Some(&encoder_outputs), attention_mask)
        } else {
            // Use decoder_start_token_id for generation
            let batch_size = encoder_outputs.size()[0];
            let device = encoder_outputs.device();
            let decoder_start_token = Tensor::full(&[batch_size, 1], self.config.decoder_start_token_id, (Kind::Int64, device));
            let decoder_inputs_embeds = decoder_start_token.apply(&self.shared) * (self.model_dim as f64).sqrt();
            self.decoder.forward(&decoder_inputs_embeds, decoder_attention_mask, Some(&encoder_outputs), attention_mask)
        };

        // Language model head
        let sequence_output = decoder_outputs;
        let lm_logits = if self.config.tie_word_embeddings {
            // Tie weights with shared embeddings
            sequence_output.matmul(&self.shared.ws.transpose(0, 1))
        } else {
            sequence_output.apply(&self.lm_head)
        };

        // Compute loss if labels are provided
        let loss = if let Some(labels) = labels {
            let shift_logits = lm_logits.narrow(1, 0, lm_logits.size()[1] - 1);
            let shift_labels = labels.narrow(1, 1, labels.size()[1] - 1);

            // Mask out padding tokens
            let active_loss = shift_labels.ne(self.config.pad_token_id);
            let active_logits = shift_logits.view([-1, shift_logits.size()[-1]]);
            let active_labels = shift_labels.view([-1]);

            Some(active_logits.cross_entropy_for_logits(&active_labels))
        } else {
            None
        };

        T5ConditionalGenerationOutput {
            loss,
            logits: lm_logits,
            encoder_last_hidden_state: encoder_outputs,
            decoder_last_hidden_state: sequence_output,
        }
    }

    pub fn generate(&self,
                   input_ids: &Tensor,
                   max_length: i64,
                   num_beams: Option<i64>,
                   temperature: f64,
                   top_k: Option<i64>,
                   top_p: Option<f64>) -> Tensor {

        let batch_size = input_ids.size()[0];
        let device = input_ids.device();

        // Encode input
        let inputs_embeds = input_ids.apply(&self.shared) * (self.model_dim as f64).sqrt();
        let encoder_outputs = self.encoder.forward(&inputs_embeds);

        // Initialize decoder input with start token
        let mut decoder_input_ids = Tensor::full(&[batch_size, 1], self.config.decoder_start_token_id, (Kind::Int64, device));

        for _ in 1..max_length {
            let decoder_inputs_embeds = decoder_input_ids.apply(&self.shared) * (self.model_dim as f64).sqrt();
            let decoder_outputs = self.decoder.forward(&decoder_inputs_embeds, None, Some(&encoder_outputs), None);

            let lm_logits = if self.config.tie_word_embeddings {
                decoder_outputs.matmul(&self.shared.ws.transpose(0, 1))
            } else {
                decoder_outputs.apply(&self.lm_head)
            };

            let next_token_logits = lm_logits.narrow(1, -1, 1).squeeze_dim(1);

            let next_token = if temperature == 0.0 {
                next_token_logits.argmax(-1, false)
            } else {
                let logits = next_token_logits / temperature;
                let probs = logits.softmax(-1, Kind::Float);
                probs.multinomial(1, false).squeeze_dim(-1)
            };

            decoder_input_ids = Tensor::cat(&[decoder_input_ids, next_token.unsqueeze(1)], 1);

            // Check for EOS token
            if next_token.int64_value(&[]) == self.config.eos_token_id {
                break;
            }
        }

        decoder_input_ids
    }
}

pub struct T5ConditionalGenerationOutput {
    pub loss: Option<Tensor>,
    pub logits: Tensor,
    pub encoder_last_hidden_state: Tensor,
    pub decoder_last_hidden_state: Tensor,
}

// T5 model configurations
pub fn t5_small_config() -> T5Config {
    T5Config {
        vocab_size: 32128,
        d_model: 512,
        d_kv: 64,
        d_ff: 2048,
        num_layers: 6,
        num_decoder_layers: 6,
        num_heads: 8,
        ..Default::default()
    }
}

pub fn t5_base_config() -> T5Config {
    T5Config {
        vocab_size: 32128,
        d_model: 768,
        d_kv: 64,
        d_ff: 3072,
        num_layers: 12,
        num_decoder_layers: 12,
        num_heads: 12,
        ..Default::default()
    }
}

pub fn t5_large_config() -> T5Config {
    T5Config {
        vocab_size: 32128,
        d_model: 1024,
        d_kv: 64,
        d_ff: 4096,
        num_layers: 24,
        num_decoder_layers: 24,
        num_heads: 16,
        ..Default::default()
    }
}