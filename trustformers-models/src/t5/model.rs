use scirs2_core::ndarray::{ArrayD, IxDyn}; // SciRS2 Integration Policy
use std::io::Read;
use trustformers_core::{
    errors::{Result, TrustformersError},
    layers::{Embedding, Linear},
    tensor::Tensor,
    traits::{Config, Layer, Model, TokenizedInput, WeightReader},
};

use super::config::T5Config;

/// T5 base model (encoder-decoder transformer)
#[derive(Clone)]
pub struct T5Model {
    config: T5Config,
    shared: Embedding,
    encoder: T5Stack,
    decoder: T5Stack,
}

impl T5Model {
    pub fn new(config: T5Config) -> Result<Self> {
        config.validate()?;

        // Shared embeddings between encoder and decoder
        let shared = Embedding::new(config.vocab_size, config.d_model, None)?;

        let encoder_config = config.clone();
        let decoder_config = config.clone();

        Ok(Self {
            config,
            shared,
            encoder: T5Stack::new(encoder_config, true)?,
            decoder: T5Stack::new(decoder_config, false)?,
        })
    }

    /// Load weights from a WeightReader (e.g., SafeTensors)
    pub fn load_weights_from_reader(&mut self, reader: &mut dyn WeightReader) -> Result<()> {
        // Load shared embeddings
        self.shared.set_weight(reader.read_tensor("shared.weight")?)?;

        // Load encoder weights
        self.encoder.load_weights(reader, "encoder")?;

        // Load decoder weights
        self.decoder.load_weights(reader, "decoder")?;

        Ok(())
    }
}

impl Model for T5Model {
    type Config = T5Config;
    type Input = T5Input;
    type Output = T5Output;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Encode input
        let encoder_hidden_states = if let Some(encoder_outputs) = input.encoder_outputs {
            encoder_outputs
        } else {
            let input_embeds = self.shared.forward(input.input_ids.input_ids)?;
            self.encoder.forward(input_embeds, None)?
        };

        // Decode if decoder input is provided
        let decoder_outputs = if let Some(decoder_input) = input.decoder_input_ids {
            let decoder_embeds = self.shared.forward(decoder_input.input_ids)?;
            let decoder_hidden =
                self.decoder.forward(decoder_embeds, Some(&encoder_hidden_states))?;
            Some(decoder_hidden)
        } else {
            None
        };

        Ok(T5Output {
            last_hidden_state: decoder_outputs.unwrap_or(encoder_hidden_states.clone()),
            encoder_last_hidden_state: Some(encoder_hidden_states),
            past_key_values: None,
        })
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(TrustformersError::model_error(
            "Use load_weights_from_reader instead".to_string(),
        ))
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        self.shared.parameter_count()
            + self.encoder.parameter_count()
            + self.decoder.parameter_count()
    }
}

/// T5 for conditional generation (seq2seq tasks)
#[derive(Clone)]
pub struct T5ForConditionalGeneration {
    transformer: T5Model,
    lm_head: Linear,
}

impl T5ForConditionalGeneration {
    pub fn new(config: T5Config) -> Result<Self> {
        let transformer = T5Model::new(config.clone())?;
        // T5 uses shared embeddings as lm_head
        let lm_head = Linear::new(config.d_model, config.vocab_size, false);

        Ok(Self {
            transformer,
            lm_head,
        })
    }

    /// Load weights from a WeightReader (e.g., SafeTensors)
    pub fn load_weights_from_reader(&mut self, reader: &mut dyn WeightReader) -> Result<()> {
        // Load transformer weights
        self.transformer.load_weights_from_reader(reader)?;

        // T5 shares embeddings with lm_head
        let shared_weight = reader.read_tensor("shared.weight")?;
        self.lm_head.set_weight(shared_weight)?;

        Ok(())
    }

    /// Generate text given input using autoregressive decoding
    pub fn generate(
        &self,
        input_ids: Vec<u32>,
        max_length: usize,
        num_beams: usize,
    ) -> Result<Vec<u32>> {
        if input_ids.is_empty() {
            return Err(TrustformersError::model_error(
                "Empty input_ids provided".to_string(),
            ));
        }

        // Encode input sequence
        let encoder_input = T5Input {
            input_ids: TokenizedInput {
                input_ids: input_ids.clone(),
                attention_mask: vec![1; input_ids.len()],
                token_type_ids: None,
                special_tokens_mask: None,
                offset_mapping: None,
                overflowing_tokens: None,
            },
            decoder_input_ids: None,
            encoder_outputs: None,
        };
        let encoder_output = self.transformer.forward(encoder_input)?;
        let encoder_hidden_states = encoder_output.encoder_last_hidden_state.ok_or_else(|| {
            TrustformersError::model_error("No encoder outputs available".to_string())
        })?;

        if num_beams > 1 {
            self.beam_search_generate(encoder_hidden_states, max_length, num_beams)
        } else {
            self.greedy_generate(encoder_hidden_states, max_length)
        }
    }

    /// Greedy generation (num_beams = 1)
    fn greedy_generate(
        &self,
        encoder_hidden_states: Tensor,
        max_length: usize,
    ) -> Result<Vec<u32>> {
        let mut generated_ids = vec![0]; // Start with BOS token (ID 0)

        for _ in 0..max_length {
            // Create decoder input
            let decoder_input = T5Input {
                input_ids: TokenizedInput {
                    input_ids: vec![],
                    attention_mask: vec![],
                    token_type_ids: None,
                    special_tokens_mask: None,
                    offset_mapping: None,
                    overflowing_tokens: None,
                }, // Empty since we use encoder outputs directly
                decoder_input_ids: Some(TokenizedInput {
                    input_ids: generated_ids.clone(),
                    attention_mask: vec![1; generated_ids.len()],
                    token_type_ids: None,
                    special_tokens_mask: None,
                    offset_mapping: None,
                    overflowing_tokens: None,
                }),
                encoder_outputs: Some(encoder_hidden_states.clone()),
            };

            // Forward pass through model
            let output = self.forward(decoder_input)?;

            // Get logits for the last token
            let logits = match &output.logits {
                Tensor::F32(arr) => {
                    let shape = arr.shape();
                    let seq_len = shape[shape.len() - 2];
                    let _vocab_size = shape[shape.len() - 1];

                    // Get last token logits
                    let last_token_slice = if shape.len() == 3 {
                        arr.slice(ndarray::s![0, seq_len - 1, ..])
                    } else {
                        arr.slice(ndarray::s![seq_len - 1, ..])
                    };
                    last_token_slice.to_owned()
                },
                _ => {
                    return Err(TrustformersError::tensor_op_error(
                        "Logits must be F32 tensor",
                        "tensor_operation",
                    ))
                },
            };

            // Find token with highest probability
            let next_token_id = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .ok_or_else(|| {
                    TrustformersError::model_error("Failed to find next token".to_string())
                })?;

            // Check for EOS token (ID 1 in T5)
            if next_token_id == 1 {
                break;
            }

            generated_ids.push(next_token_id);
        }

        // Remove BOS token from output
        if !generated_ids.is_empty() {
            generated_ids.remove(0);
        }

        Ok(generated_ids)
    }

    /// Beam search generation (num_beams > 1)
    fn beam_search_generate(
        &self,
        encoder_hidden_states: Tensor,
        max_length: usize,
        num_beams: usize,
    ) -> Result<Vec<u32>> {
        #[derive(Clone)]
        struct Beam {
            tokens: Vec<u32>,
            score: f32,
        }

        let mut beams = vec![Beam {
            tokens: vec![0],
            score: 0.0,
        }]; // Start with BOS token
        let mut finished_beams = Vec::new();

        for _step in 0..max_length {
            let mut all_candidates = Vec::new();

            for beam in &beams {
                // Create decoder input for this beam
                let decoder_input = T5Input {
                    input_ids: TokenizedInput {
                        input_ids: vec![],
                        attention_mask: vec![],
                        token_type_ids: None,
                        special_tokens_mask: None,
                        offset_mapping: None,
                        overflowing_tokens: None,
                    },
                    decoder_input_ids: Some(TokenizedInput {
                        input_ids: beam.tokens.clone(),
                        attention_mask: vec![1; beam.tokens.len()],
                        token_type_ids: None,
                        special_tokens_mask: None,
                        offset_mapping: None,
                        overflowing_tokens: None,
                    }),
                    encoder_outputs: Some(encoder_hidden_states.clone()),
                };

                // Forward pass
                let output = self.forward(decoder_input)?;

                // Get logits for the last token and convert to probabilities
                let logits = match &output.logits {
                    Tensor::F32(arr) => {
                        let shape = arr.shape();
                        let seq_len = shape[shape.len() - 2];

                        let last_token_slice = if shape.len() == 3 {
                            arr.slice(ndarray::s![0, seq_len - 1, ..])
                        } else {
                            arr.slice(ndarray::s![seq_len - 1, ..])
                        };
                        last_token_slice.to_owned()
                    },
                    _ => {
                        return Err(TrustformersError::tensor_op_error(
                            "Logits must be F32 tensor",
                            "tensor_operation",
                        ))
                    },
                };

                // Apply softmax to get probabilities
                let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_logits: Vec<f32> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
                let sum_exp: f32 = exp_logits.iter().sum();
                let probs: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp).collect();

                // Get top candidates
                let mut token_scores: Vec<(usize, f32)> =
                    probs.iter().enumerate().map(|(idx, &prob)| (idx, prob.ln())).collect();
                token_scores
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Add top candidates to all_candidates
                for &(token_id, log_prob) in token_scores.iter().take(num_beams) {
                    let mut new_tokens = beam.tokens.clone();
                    new_tokens.push(token_id as u32);
                    let new_score = beam.score + log_prob;

                    if token_id == 1 {
                        // EOS token
                        finished_beams.push(Beam {
                            tokens: new_tokens,
                            score: new_score,
                        });
                    } else {
                        all_candidates.push(Beam {
                            tokens: new_tokens,
                            score: new_score,
                        });
                    }
                }
            }

            // Select top beams
            all_candidates
                .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            beams = all_candidates.into_iter().take(num_beams).collect();

            // Early stopping if all beams are finished
            if beams.is_empty() {
                break;
            }
        }

        // Combine finished beams with current beams
        finished_beams.extend(beams);

        // Select best beam
        if finished_beams.is_empty() {
            return Err(TrustformersError::model_error(
                "No valid sequences generated".to_string(),
            ));
        }

        finished_beams
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        let best_beam = &finished_beams[0];

        // Remove BOS token from output
        let mut result = best_beam.tokens.clone();
        if !result.is_empty() {
            result.remove(0);
        }

        // Remove EOS token if present
        if let Some(&1) = result.last() {
            result.pop();
        }

        Ok(result)
    }
}

impl Model for T5ForConditionalGeneration {
    type Config = T5Config;
    type Input = T5Input;
    type Output = T5LMOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let base_output = self.transformer.forward(input)?;

        // Apply language modeling head
        let logits = self.lm_head.forward(base_output.last_hidden_state)?;

        Ok(T5LMOutput {
            logits,
            past_key_values: base_output.past_key_values,
            encoder_last_hidden_state: base_output.encoder_last_hidden_state,
        })
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.transformer.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.transformer.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.transformer.num_parameters() + self.lm_head.parameter_count()
    }
}

/// T5 encoder or decoder stack
#[derive(Clone)]
#[allow(dead_code)]
struct T5Stack {
    #[allow(dead_code)]
    config: T5Config,
    is_encoder: bool,
    embed_tokens: Option<Embedding>, // Only used if not sharing embeddings
    block: Vec<T5Block>,
    final_layer_norm: T5LayerNorm,
    dropout: f32,
}

impl T5Stack {
    fn new(config: T5Config, is_encoder: bool) -> Result<Self> {
        let num_layers = if is_encoder {
            config.num_layers
        } else {
            config.num_decoder_layers.unwrap_or(config.num_layers)
        };

        let mut block = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            block.push(T5Block::new(&config, is_encoder)?);
        }

        Ok(Self {
            config: config.clone(),
            is_encoder,
            embed_tokens: None,
            block,
            final_layer_norm: T5LayerNorm::new(config.d_model, config.layer_norm_epsilon),
            dropout: config.dropout_rate,
        })
    }

    fn load_weights(&mut self, reader: &mut dyn WeightReader, prefix: &str) -> Result<()> {
        // Load block weights
        for (i, block) in self.block.iter_mut().enumerate() {
            block.load_weights(reader, &format!("{}.block.{}", prefix, i))?;
        }

        // Load final layer norm
        self.final_layer_norm
            .load_weights(reader, &format!("{}.final_layer_norm", prefix))?;

        Ok(())
    }

    fn forward(
        &self,
        hidden_states: Tensor,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden_states = hidden_states;

        // Create attention mask if needed
        let attention_mask = create_attention_mask(&hidden_states)?;

        // Pass through blocks
        for block in &self.block {
            hidden_states = block.forward(
                hidden_states,
                Some(&attention_mask),
                encoder_hidden_states,
                None, // cross_attention_mask
            )?;
        }

        // Apply final layer norm
        self.final_layer_norm.forward(hidden_states)
    }

    fn parameter_count(&self) -> usize {
        let mut total = 0;

        // Add embed_tokens if present (not shared)
        if let Some(ref embed) = self.embed_tokens {
            total += embed.parameter_count();
        }

        // Add blocks
        for block in &self.block {
            total += block.parameter_count();
        }

        // Add final layer norm
        total += self.final_layer_norm.parameter_count();

        total
    }
}

/// T5 transformer block
#[derive(Clone)]
struct T5Block {
    is_encoder: bool,
    self_attention: T5Attention,
    cross_attention: Option<T5Attention>,
    feed_forward: T5DenseReluDense,
}

impl T5Block {
    fn new(config: &T5Config, is_encoder: bool) -> Result<Self> {
        let cross_attention =
            if !is_encoder { Some(T5Attention::new(config, true)?) } else { None };

        Ok(Self {
            is_encoder,
            self_attention: T5Attention::new(config, false)?,
            cross_attention,
            feed_forward: T5DenseReluDense::new(config)?,
        })
    }

    fn load_weights(&mut self, reader: &mut dyn WeightReader, prefix: &str) -> Result<()> {
        // Load self-attention weights
        self.self_attention.load_weights(reader, &format!("{}.layer.0", prefix))?;

        // Load cross-attention weights if decoder
        if let Some(ref mut cross_attn) = self.cross_attention {
            cross_attn.load_weights(reader, &format!("{}.layer.1", prefix))?;
        }

        // Load feed-forward weights
        let ff_idx = if self.is_encoder { 1 } else { 2 };
        self.feed_forward
            .load_weights(reader, &format!("{}.layer.{}", prefix, ff_idx))?;

        Ok(())
    }

    fn forward(
        &self,
        hidden_states: Tensor,
        attention_mask: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
        cross_attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Self-attention
        let mut hidden_states = self.self_attention.forward(
            hidden_states.clone(),
            None, // key_value_states
            attention_mask,
        )?;

        // Cross-attention (decoder only)
        if let Some(ref cross_attn) = self.cross_attention {
            if let Some(encoder_hidden) = encoder_hidden_states {
                hidden_states = cross_attn.forward(
                    hidden_states,
                    Some(encoder_hidden),
                    cross_attention_mask,
                )?;
            }
        }

        // Feed-forward
        self.feed_forward.forward(hidden_states)
    }

    fn parameter_count(&self) -> usize {
        let mut total = self.self_attention.parameter_count() + self.feed_forward.parameter_count();

        if let Some(ref cross_attn) = self.cross_attention {
            total += cross_attn.parameter_count();
        }

        total
    }
}

/// T5 attention module
#[derive(Clone)]
struct T5Attention {
    is_cross_attention: bool,
    layer_norm: T5LayerNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
    d_kv: usize,
    #[allow(dead_code)]
    dropout: f32,
    has_relative_attention_bias: bool,
    relative_attention_num_buckets: usize,
    relative_attention_max_distance: usize,
    relative_attention_bias: Option<Embedding>, // For storing learned relative position biases
}

impl T5Attention {
    fn new(config: &T5Config, is_cross_attention: bool) -> Result<Self> {
        let has_relative_bias = !is_cross_attention;
        let relative_attention_bias = if has_relative_bias {
            Some(Embedding::new(
                config.relative_attention_num_buckets,
                config.num_heads,
                None,
            )?)
        } else {
            None
        };

        Ok(Self {
            is_cross_attention,
            layer_norm: T5LayerNorm::new(config.d_model, config.layer_norm_epsilon),
            q: Linear::new(config.d_model, config.num_heads * config.d_kv, false),
            k: Linear::new(config.d_model, config.num_heads * config.d_kv, false),
            v: Linear::new(config.d_model, config.num_heads * config.d_kv, false),
            o: Linear::new(config.num_heads * config.d_kv, config.d_model, false),
            n_heads: config.num_heads,
            d_kv: config.d_kv,
            dropout: config.dropout_rate,
            has_relative_attention_bias: has_relative_bias,
            relative_attention_num_buckets: config.relative_attention_num_buckets,
            relative_attention_max_distance: config.relative_attention_max_distance,
            relative_attention_bias,
        })
    }

    fn load_weights(&mut self, reader: &mut dyn WeightReader, prefix: &str) -> Result<()> {
        self.layer_norm.load_weights(reader, &format!("{}.layer_norm", prefix))?;

        // Load attention weights
        let attn_name = if self.is_cross_attention { "EncDecAttention" } else { "SelfAttention" };
        let attn_prefix = format!("{}.{}", prefix, attn_name);

        self.q.set_weight(reader.read_tensor(&format!("{}.q.weight", attn_prefix))?)?;
        self.k.set_weight(reader.read_tensor(&format!("{}.k.weight", attn_prefix))?)?;
        self.v.set_weight(reader.read_tensor(&format!("{}.v.weight", attn_prefix))?)?;
        self.o.set_weight(reader.read_tensor(&format!("{}.o.weight", attn_prefix))?)?;

        // Load relative attention bias if present
        if let Some(ref mut bias) = self.relative_attention_bias {
            if let Ok(bias_weight) =
                reader.read_tensor(&format!("{}.relative_attention_bias.weight", attn_prefix))
            {
                bias.set_weight(bias_weight)?;
            }
        }

        Ok(())
    }

    fn forward(
        &self,
        hidden_states: Tensor,
        key_value_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Pre-norm
        let normed_hidden = self.layer_norm.forward(hidden_states.clone())?;

        // Get shape info
        let shape = normed_hidden.shape();
        let batch_size = if shape.len() == 3 { shape[0] } else { 1 };
        let _seq_len = if shape.len() == 3 { shape[1] } else { shape[0] };

        // Ensure 3D tensor
        let normed_hidden = match &normed_hidden {
            Tensor::F32(arr) => {
                if arr.ndim() == 2 {
                    Tensor::F32(arr.clone().insert_axis(ndarray::Axis(0)).to_owned())
                } else {
                    normed_hidden
                }
            },
            _ => normed_hidden,
        };

        // Project Q, K, V
        let query = self.q.forward(normed_hidden.clone())?;
        let key = if let Some(kv_states) = key_value_states {
            self.k.forward(kv_states.clone())?
        } else {
            self.k.forward(normed_hidden.clone())?
        };
        let value = if let Some(kv_states) = key_value_states {
            self.v.forward(kv_states.clone())?
        } else {
            self.v.forward(normed_hidden)?
        };

        // Reshape Q, K, V for multi-head attention
        let attention_output = match (&query, &key, &value) {
            (Tensor::F32(q_arr), Tensor::F32(k_arr), Tensor::F32(v_arr)) => {
                // The linear outputs have shape [batch, seq_len, n_heads * d_kv]
                // We need to determine the sequence lengths
                let q_shape = q_arr.shape();
                let k_shape = k_arr.shape();
                let v_shape = v_arr.shape();

                // For 2D inputs (which were expanded to 3D), shape is [1, seq_len, n_heads * d_kv]
                // For 3D inputs, shape is [batch, seq_len, n_heads * d_kv]
                let q_seq_len = if q_shape.len() == 3 { q_shape[1] } else { q_shape[0] };
                let k_seq_len = if k_shape.len() == 3 { k_shape[1] } else { k_shape[0] };
                let v_seq_len = if v_shape.len() == 3 { v_shape[1] } else { v_shape[0] };

                // Ensure tensors are 3D
                let q_arr = if q_arr.ndim() == 2 {
                    q_arr.clone().insert_axis(ndarray::Axis(0)).to_owned()
                } else {
                    q_arr.clone()
                };
                let k_arr = if k_arr.ndim() == 2 {
                    k_arr.clone().insert_axis(ndarray::Axis(0)).to_owned()
                } else {
                    k_arr.clone()
                };
                let v_arr = if v_arr.ndim() == 2 {
                    v_arr.clone().insert_axis(ndarray::Axis(0)).to_owned()
                } else {
                    v_arr.clone()
                };

                // Reshape to [batch, seq_len, n_heads, d_kv]
                let q = q_arr
                    .to_shape(ndarray::IxDyn(&[
                        batch_size,
                        q_seq_len,
                        self.n_heads,
                        self.d_kv,
                    ]))
                    .map_err(|e| {
                        TrustformersError::shape_error(format!(
                            "Failed to reshape Q from {:?} to [{}, {}, {}, {}]: {}",
                            q_arr.shape(),
                            batch_size,
                            q_seq_len,
                            self.n_heads,
                            self.d_kv,
                            e
                        ))
                    })?
                    .to_owned();
                let k = k_arr
                    .to_shape(ndarray::IxDyn(&[
                        batch_size,
                        k_seq_len,
                        self.n_heads,
                        self.d_kv,
                    ]))
                    .map_err(|e| {
                        TrustformersError::shape_error(format!(
                            "Failed to reshape K from {:?} to [{}, {}, {}, {}]: {}",
                            k_arr.shape(),
                            batch_size,
                            k_seq_len,
                            self.n_heads,
                            self.d_kv,
                            e
                        ))
                    })?
                    .to_owned();
                let v = v_arr
                    .to_shape(ndarray::IxDyn(&[
                        batch_size,
                        v_seq_len,
                        self.n_heads,
                        self.d_kv,
                    ]))
                    .map_err(|e| {
                        TrustformersError::shape_error(format!(
                            "Failed to reshape V from {:?} to [{}, {}, {}, {}]: {}",
                            v_arr.shape(),
                            batch_size,
                            v_seq_len,
                            self.n_heads,
                            self.d_kv,
                            e
                        ))
                    })?
                    .to_owned();

                // Transpose to [batch, n_heads, seq_len, d_kv]
                let q = q.permuted_axes(vec![0, 2, 1, 3]);
                let k = k.permuted_axes(vec![0, 2, 1, 3]);
                let v = v.permuted_axes(vec![0, 2, 1, 3]);

                // Compute attention scores
                let scale = 1.0 / (self.d_kv as f32).sqrt();
                let key_seq_len = k.shape()[2];
                let k_t = k.permuted_axes(vec![0, 1, 3, 2]);

                // Compute Q * K^T
                let mut scores = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[
                    batch_size,
                    self.n_heads,
                    q_seq_len,
                    key_seq_len,
                ]));
                for b in 0..batch_size {
                    for h in 0..self.n_heads {
                        let q_head = q.slice(ndarray::s![b, h, .., ..]);
                        let k_head_t = k_t.slice(ndarray::s![b, h, .., ..]);
                        let score = q_head.dot(&k_head_t);
                        scores.slice_mut(ndarray::s![b, h, .., ..]).assign(&score);
                    }
                }
                scores *= scale;

                // Add relative position bias if enabled
                if self.has_relative_attention_bias {
                    let position_bias =
                        self.compute_relative_position_bias(q_seq_len, key_seq_len)?;
                    match position_bias {
                        Tensor::F32(bias_arr) => {
                            // Add bias to scores
                            for b in 0..batch_size {
                                let mut slice = scores.slice_mut(ndarray::s![b, .., .., ..]);
                                slice += &bias_arr;
                            }
                        },
                        _ => {
                            return Err(TrustformersError::tensor_op_error(
                                "Position bias must be F32",
                                "tensor_operation",
                            ))
                        },
                    }
                }

                // Apply attention mask if provided
                if let Some(mask) = attention_mask {
                    match mask {
                        Tensor::F32(mask_arr) => {
                            scores += mask_arr;
                        },
                        _ => {
                            return Err(TrustformersError::tensor_op_error(
                                "Attention mask must be F32",
                                "tensor_operation",
                            ))
                        },
                    }
                }

                // Softmax
                let mut attention_probs = scores.clone();
                for b in 0..batch_size {
                    for h in 0..self.n_heads {
                        for i in 0..q_seq_len {
                            let mut row = attention_probs.slice_mut(ndarray::s![b, h, i, ..]);
                            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                            row.mapv_inplace(|x| (x - max_val).exp());
                            let sum: f32 = row.iter().sum();
                            if sum > 0.0 {
                                row.mapv_inplace(|x| x / sum);
                            }
                        }
                    }
                }

                // Apply attention to values
                let mut output = ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[
                    batch_size,
                    self.n_heads,
                    q_seq_len,
                    self.d_kv,
                ]));
                for b in 0..batch_size {
                    for h in 0..self.n_heads {
                        let attn_probs_head = attention_probs.slice(ndarray::s![b, h, .., ..]);
                        let v_head = v.slice(ndarray::s![b, h, .., ..]);
                        let out = attn_probs_head.dot(&v_head);
                        output.slice_mut(ndarray::s![b, h, .., ..]).assign(&out);
                    }
                }

                // Transpose back and reshape
                let output = output.permuted_axes(vec![0, 2, 1, 3]);
                let output = output
                    .to_shape(ndarray::IxDyn(&[
                        batch_size,
                        q_seq_len,
                        self.n_heads * self.d_kv,
                    ]))
                    .map_err(|_| {
                        TrustformersError::shape_error("Failed to reshape output".to_string())
                    })?
                    .to_owned();

                Tensor::F32(output)
            },
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Unsupported tensor type",
                    "tensor_operation",
                ))
            },
        };

        // Project output
        let output = self.o.forward(attention_output)?;

        // Remove batch dimension if input was 2D
        let output = if shape.len() == 2 {
            match output {
                Tensor::F32(arr) => Tensor::F32(arr.remove_axis(ndarray::Axis(0))),
                _ => output,
            }
        } else {
            output
        };

        // Add residual
        hidden_states.add(&output)
    }

    /// Compute relative position bias for T5
    fn compute_relative_position_bias(&self, query_len: usize, key_len: usize) -> Result<Tensor> {
        // Compute relative positions
        let mut relative_positions = ndarray::Array2::<i32>::zeros((query_len, key_len));
        for i in 0..query_len {
            for j in 0..key_len {
                relative_positions[[i, j]] = j as i32 - i as i32;
            }
        }

        // Convert to buckets
        let buckets = self.relative_position_bucket(relative_positions);

        // Get embeddings
        if let Some(ref bias_embedding) = self.relative_attention_bias {
            let bucket_indices: Vec<u32> = buckets.iter().cloned().map(|b| b as u32).collect();
            let bias = bias_embedding.forward(bucket_indices)?;

            // Reshape to [n_heads, query_len, key_len]
            match bias {
                Tensor::F32(bias_arr) => {
                    let reshaped = bias_arr
                        .to_shape(ndarray::IxDyn(&[query_len, key_len, self.n_heads]))
                        .map_err(|_| {
                            TrustformersError::shape_error("Failed to reshape bias".to_string())
                        })?
                        .to_owned();
                    // Transpose to [n_heads, query_len, key_len]
                    let transposed = reshaped.permuted_axes(vec![2, 0, 1]);
                    Ok(Tensor::F32(transposed))
                },
                _ => Err(TrustformersError::tensor_op_error(
                    "Unsupported tensor type",
                    "tensor_operation",
                )),
            }
        } else {
            // No relative bias, return zeros
            let zeros =
                ndarray::ArrayD::<f32>::zeros(ndarray::IxDyn(&[self.n_heads, query_len, key_len]));
            Ok(Tensor::F32(zeros))
        }
    }

    /// Convert relative positions to bucket indices
    fn relative_position_bucket(
        &self,
        relative_positions: ndarray::Array2<i32>,
    ) -> ndarray::Array2<i32> {
        let num_buckets = self.relative_attention_num_buckets;
        let max_distance = self.relative_attention_max_distance;
        let mut buckets = relative_positions.mapv(|_x| 0);

        // Half of the buckets are for positive positions
        let boundary = num_buckets as i32 / 2;

        for ((i, j), val) in relative_positions.indexed_iter() {
            let mut bucket = if *val > 0 { boundary } else { 0 };
            let abs_val = val.abs();

            if abs_val < boundary {
                bucket += abs_val;
            } else {
                // Logarithmic buckets for larger distances
                let max_exact = boundary;
                let log_val = ((abs_val as f32 / max_exact as f32).ln()
                    / (max_distance as f32 / max_exact as f32).ln()
                    * (boundary - max_exact) as f32) as i32;
                bucket += max_exact + log_val.min(boundary - max_exact - 1);
            }

            if *val > 0 {
                buckets[[i, j]] = bucket;
            } else {
                buckets[[i, j]] = bucket;
            }
        }

        buckets
    }

    fn parameter_count(&self) -> usize {
        let mut total = self.layer_norm.parameter_count()
            + self.q.parameter_count()
            + self.k.parameter_count()
            + self.v.parameter_count()
            + self.o.parameter_count();

        if let Some(ref bias) = self.relative_attention_bias {
            total += bias.parameter_count();
        }

        total
    }
}

/// T5 feed-forward module
#[derive(Clone)]
struct T5DenseReluDense {
    layer_norm: T5LayerNorm,
    wi: Linear,
    wo: Linear,
    #[allow(dead_code)]
    dropout: f32,
}

impl T5DenseReluDense {
    fn new(config: &T5Config) -> Result<Self> {
        Ok(Self {
            layer_norm: T5LayerNorm::new(config.d_model, config.layer_norm_epsilon),
            wi: Linear::new(config.d_model, config.d_ff, false),
            wo: Linear::new(config.d_ff, config.d_model, false),
            dropout: config.dropout_rate,
        })
    }

    fn load_weights(&mut self, reader: &mut dyn WeightReader, prefix: &str) -> Result<()> {
        self.layer_norm.load_weights(reader, &format!("{}.layer_norm", prefix))?;

        let dense_prefix = format!("{}.DenseReluDense", prefix);
        self.wi
            .set_weight(reader.read_tensor(&format!("{}.wi.weight", dense_prefix))?)?;
        self.wo
            .set_weight(reader.read_tensor(&format!("{}.wo.weight", dense_prefix))?)?;

        Ok(())
    }

    fn forward(&self, hidden_states: Tensor) -> Result<Tensor> {
        // Pre-norm
        let normed = self.layer_norm.forward(hidden_states.clone())?;

        // Feed-forward with ReLU
        let ff_output = self.wi.forward(normed)?;
        let ff_output = relu(ff_output)?;
        let ff_output = self.wo.forward(ff_output)?;

        // Add residual
        hidden_states.add(&ff_output)
    }

    fn parameter_count(&self) -> usize {
        self.layer_norm.parameter_count() + self.wi.parameter_count() + self.wo.parameter_count()
    }
}

/// T5-specific layer normalization (no bias, RMS norm)
#[derive(Clone)]
struct T5LayerNorm {
    weight: Tensor,
    epsilon: f32,
}

impl T5LayerNorm {
    fn new(hidden_size: usize, epsilon: f32) -> Self {
        Self {
            weight: Tensor::ones(&[hidden_size]).unwrap(),
            epsilon,
        }
    }

    fn load_weights(&mut self, reader: &mut dyn WeightReader, prefix: &str) -> Result<()> {
        self.weight = reader.read_tensor(&format!("{}.weight", prefix))?;
        Ok(())
    }

    fn forward(&self, hidden_states: Tensor) -> Result<Tensor> {
        // T5 uses RMS norm without bias
        match (&hidden_states, &self.weight) {
            (Tensor::F32(x), Tensor::F32(w)) => {
                // Calculate RMS
                let variance = x.mapv(|v| v * v).mean().unwrap() + self.epsilon;
                let x = x / variance.sqrt();

                // Apply weight
                let result = &x * w;
                Ok(Tensor::F32(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "T5LayerNorm only supports F32 tensors",
                "tensor_operation",
            )),
        }
    }

    fn parameter_count(&self) -> usize {
        self.weight.len()
    }
}

/// ReLU activation function
fn relu(x: Tensor) -> Result<Tensor> {
    match &x {
        Tensor::F32(arr) => {
            let result = arr.mapv(|val| val.max(0.0));
            Ok(Tensor::F32(result))
        },
        _ => Err(TrustformersError::tensor_op_error(
            "ReLU only supports F32 tensors",
            "tensor_operation",
        )),
    }
}

/// Create attention mask
fn create_attention_mask(hidden_states: &Tensor) -> Result<Tensor> {
    let shape = hidden_states.shape();
    let seq_len = shape[shape.len() - 2];

    // Create a simple causal mask for now
    let mask = ArrayD::<f32>::ones(IxDyn(&[1, 1, seq_len, seq_len]));
    Ok(Tensor::F32(mask))
}

/// Input for T5 models
pub struct T5Input {
    pub input_ids: TokenizedInput,
    pub decoder_input_ids: Option<TokenizedInput>,
    pub encoder_outputs: Option<Tensor>,
}

/// Output from T5 base model
pub struct T5Output {
    pub last_hidden_state: Tensor,
    pub encoder_last_hidden_state: Option<Tensor>,
    pub past_key_values: Option<Vec<Tensor>>,
}

/// Output from T5 language modeling head
pub struct T5LMOutput {
    pub logits: Tensor,
    pub past_key_values: Option<Vec<Tensor>>,
    pub encoder_last_hidden_state: Option<Tensor>,
}
