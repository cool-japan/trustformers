use scirs2_core::ndarray::{s, ArrayD, Axis, IxDyn}; // SciRS2 Integration Policy
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{invalid_config, tensor_op_error, Result, TrustformersError},
    layers::{Embedding, LayerNorm, Linear},
    tensor::Tensor,
    traits::{Config, Layer, Model, TokenizedInput, WeightReader},
};

use super::config::Gpt2Config;

/// Transpose a 2D tensor (swap dimensions 0 and 1)
/// PyTorch Linear weights are [out_features, in_features]
/// but we need [in_features, out_features] for our matmul
fn transpose_tensor(tensor: Tensor) -> Result<Tensor> {
    match tensor {
        Tensor::F32(arr) => {
            if arr.ndim() != 2 {
                return Err(TrustformersError::shape_error(format!(
                    "Expected 2D tensor, got {}D",
                    arr.ndim()
                )));
            }
            // Transpose using ndarray's .t() and convert back to owned
            let transposed = arr.t().to_owned();
            Ok(Tensor::F32(transposed))
        },
        _ => Err(TrustformersError::tensor_op_error(
            "Only F32 tensors supported",
            "transpose",
        )),
    }
}

/// GPT-2 base model (decoder-only transformer)
#[derive(Clone)]
pub struct Gpt2Model {
    config: Gpt2Config,
    wte: Embedding,    // Word token embeddings
    wpe: Embedding,    // Positional embeddings
    h: Vec<Gpt2Block>, // Transformer blocks
    ln_f: LayerNorm,   // Final layer norm
    device: Device,    // Compute device (CPU, Metal, CUDA, etc.)
}

impl Gpt2Model {
    pub fn new(config: Gpt2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    /// Create a GPT-2 model with specified device (CPU, Metal, CUDA, etc.)
    pub fn new_with_device(config: Gpt2Config, device: Device) -> Result<Self> {
        config.validate()?;

        // Initialize embeddings
        let wte = Embedding::new(config.vocab_size, config.n_embd, None)?;
        let wpe = Embedding::new(config.n_positions, config.n_embd, None)?;

        // Initialize transformer blocks with device
        let mut h = Vec::with_capacity(config.n_layer);
        for _ in 0..config.n_layer {
            h.push(Gpt2Block::new_with_device(&config, device)?);
        }

        // Initialize final layer norm
        let ln_f = LayerNorm::new_simple(config.n_embd, config.layer_norm_epsilon);

        Ok(Self {
            config,
            wte,
            wpe,
            h,
            ln_f,
            device,
        })
    }

    /// Get the device this model uses
    pub fn device(&self) -> Device {
        self.device
    }

    /// Move this model to a different device
    pub fn to_device(mut self, device: Device) -> Self {
        self.device = device;
        // Move all blocks to the new device
        for block in &mut self.h {
            *block = block.clone().to_device(device);
        }
        self
    }

    /// Load weights from a WeightReader (e.g., SafeTensors)
    pub fn load_weights_from_reader(&mut self, reader: &mut dyn WeightReader) -> Result<()> {
        // Load embeddings
        self.wte.set_weight(reader.read_tensor("wte.weight")?)?;
        self.wpe.set_weight(reader.read_tensor("wpe.weight")?)?;

        // Load transformer blocks
        for (i, block) in self.h.iter_mut().enumerate() {
            let prefix = format!("h.{}", i);
            block.load_weights(reader, &prefix)?;
        }

        // Load final layer norm
        self.ln_f.set_weight(reader.read_tensor("ln_f.weight")?)?;
        self.ln_f.set_bias(reader.read_tensor("ln_f.bias")?)?;

        Ok(())
    }

    fn forward_internal(
        &self,
        input_ids: &[Vec<u32>],
        position_ids: Option<&[Vec<u32>]>,
        mut past_key_values: Option<&mut KVCache>,
    ) -> Result<Tensor> {
        let batch_size = input_ids.len();
        if batch_size == 0 {
            return Err(TrustformersError::model_error(
                "Empty batch not supported".to_string(),
            ));
        }

        let seq_len = input_ids[0].len();

        // Validate batch consistency
        for (i, seq) in input_ids.iter().enumerate() {
            if seq.len() != seq_len {
                return Err(TrustformersError::model_error(format!(
                    "Inconsistent sequence length in batch. Expected {}, got {} at index {}",
                    seq_len,
                    seq.len(),
                    i
                )));
            }
        }

        // Process embeddings for entire batch
        let mut batch_word_embeds = Vec::new();
        let mut batch_position_embeds = Vec::new();

        for (batch_idx, seq_input_ids) in input_ids.iter().enumerate() {
            // Get word embeddings for this sequence
            let word_embeds = self.wte.forward(seq_input_ids.clone())?;

            // Generate position IDs if not provided
            let pos_ids: Vec<u32> = if let Some(pos_batch) = position_ids {
                pos_batch[batch_idx].clone()
            } else {
                (0..seq_len as u32).collect()
            };

            // Get position embeddings for this sequence
            let position_embeds = self.wpe.forward(pos_ids)?;

            batch_word_embeds.push(word_embeds);
            batch_position_embeds.push(position_embeds);
        }

        // Combine embeddings for each sequence in the batch
        let mut batch_hidden_states = Vec::new();
        for i in 0..batch_size {
            let combined = batch_word_embeds[i].add(&batch_position_embeds[i])?;
            batch_hidden_states.push(combined);
        }

        // Stack batch embeddings into a single tensor
        let mut hidden_states = stack_tensors(&batch_hidden_states)?;

        // Add batch dimension if needed: [batch, seq_len, hidden_size]
        match &hidden_states {
            Tensor::F32(arr) => {
                if arr.ndim() == 2 {
                    // Single sequence case, add batch dimension
                    hidden_states = Tensor::F32(arr.clone().insert_axis(Axis(0)).to_owned());
                }
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported tensor type".to_string(),
                ))
            },
        }

        // Create causal mask for attention
        let causal_mask = create_causal_mask(seq_len)?;

        // Pass through transformer blocks with optional caching
        for (layer_idx, block) in self.h.iter().enumerate() {
            let layer_cache = past_key_values.as_mut().map(|cache| &mut cache.layers[layer_idx]);
            hidden_states =
                block.forward_with_cache(hidden_states, Some(&causal_mask), layer_cache)?;
        }

        // Apply final layer norm
        self.ln_f.forward(hidden_states)
    }
}

impl Model for Gpt2Model {
    type Config = Gpt2Config;
    type Input = TokenizedInput;
    type Output = Gpt2Output;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let input_ids = vec![input.input_ids]; // Convert single sequence to batch
        let hidden_states = self.forward_internal(&input_ids, None, None)?;

        Ok(Gpt2Output {
            last_hidden_state: hidden_states,
            past_key_values: None,
        })
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        // GPT-2 uses a different Read interface for now
        // We'll implement weight loading through a separate method
        Err(TrustformersError::model_error(
            "Use load_weights_from_reader instead".to_string(),
        ))
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Embeddings
        total += self.wte.parameter_count();
        total += self.wpe.parameter_count();

        // Transformer blocks
        for block in &self.h {
            total += block.parameter_count();
        }

        // Final layer norm
        total += self.ln_f.parameter_count();

        total
    }
}

/// GPT-2 with language modeling head
#[derive(Clone)]
pub struct Gpt2LMHeadModel {
    transformer: Gpt2Model,
    lm_head: Linear,
}

impl Gpt2LMHeadModel {
    pub fn new(config: Gpt2Config) -> Result<Self> {
        let transformer = Gpt2Model::new(config.clone())?;
        let lm_head = Linear::new(config.n_embd, config.vocab_size, true);

        Ok(Self {
            transformer,
            lm_head,
        })
    }

    /// Load weights from a WeightReader (e.g., SafeTensors)
    pub fn load_weights_from_reader(&mut self, reader: &mut dyn WeightReader) -> Result<()> {
        // Load transformer weights
        self.transformer.load_weights_from_reader(reader)?;

        // Load language modeling head weights
        // Note: In GPT-2, the LM head shares weights with the input embeddings
        let wte_weight = reader.read_tensor("wte.weight")?;
        self.lm_head.set_weight(wte_weight)?;

        // No bias for LM head in GPT-2

        Ok(())
    }

    /// Generate text given a prompt
    pub fn generate(
        &self,
        input_ids: Vec<u32>,
        max_length: usize,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
    ) -> Result<Vec<u32>> {
        let mut generated = input_ids.clone();

        while generated.len() < max_length {
            // Prepare input
            let input = TokenizedInput {
                input_ids: generated.clone(),
                attention_mask: vec![1u8; generated.len()],
                token_type_ids: None,
                special_tokens_mask: None,
                offset_mapping: None,
                overflowing_tokens: None,
            };

            // Forward pass
            let output = self.forward(input)?;

            // Get logits for the last token
            let logits = output.logits;
            let last_logits = match &logits {
                Tensor::F32(arr) => {
                    // Get the last token's logits (shape: [batch, seq_len, vocab_size])
                    let shape = arr.shape();
                    if shape.len() != 3 {
                        return Err(tensor_op_error(
                            "tensor_operation",
                            "Unsupported tensor type".to_string(),
                        ));
                    }
                    let seq_len = shape[1];
                    {
                        let shape = arr.shape();
                        let vocab_size = shape[2];
                        let slice = arr.slice(s![0, seq_len - 1, ..]);
                        ArrayD::from_shape_vec(
                            IxDyn(&[vocab_size]),
                            slice.iter().cloned().collect(),
                        )
                        .unwrap()
                    }
                },
                _ => {
                    return Err(tensor_op_error(
                        "tensor_operation",
                        "Unsupported tensor type".to_string(),
                    ))
                },
            };

            // Apply temperature
            let scaled_logits = if temperature != 1.0 {
                last_logits.mapv(|x| x / temperature)
            } else {
                last_logits
            };

            // Apply top-k filtering
            let filtered_logits = if let Some(k) = top_k {
                apply_top_k_filtering(scaled_logits, k)?
            } else {
                scaled_logits
            };

            // Apply top-p (nucleus) filtering
            let final_logits = if let Some(p) = top_p {
                apply_top_p_filtering(filtered_logits, p)?
            } else {
                filtered_logits
            };

            // Sample from the distribution
            let next_token = sample_from_logits(final_logits)?;
            generated.push(next_token);

            // Check for EOS token (assuming 50256 is EOS for GPT-2)
            if next_token == 50256 {
                break;
            }
        }

        Ok(generated)
    }

    /// Generate text using greedy decoding
    pub fn generate_greedy(&self, input_ids: Vec<u32>, max_length: usize) -> Result<Vec<u32>> {
        let mut generated = input_ids.clone();

        while generated.len() < max_length {
            // Prepare input
            let input = TokenizedInput {
                input_ids: generated.clone(),
                attention_mask: vec![1u8; generated.len()],
                token_type_ids: None,
                special_tokens_mask: None,
                offset_mapping: None,
                overflowing_tokens: None,
            };

            // Forward pass
            let output = self.forward(input)?;

            // Get logits for the last token
            let logits = output.logits;
            let next_token = match &logits {
                Tensor::F32(arr) => {
                    // Get the last token's logits
                    let shape = arr.shape();
                    if shape.len() != 3 {
                        return Err(tensor_op_error(
                            "tensor_operation",
                            "Unsupported tensor type".to_string(),
                        ));
                    }
                    let seq_len = shape[1];
                    let last_logits = arr.slice(s![0, seq_len - 1, ..]);

                    // Find argmax
                    let mut max_idx = 0;
                    let mut max_val = f32::NEG_INFINITY;
                    for (idx, &val) in last_logits.iter().enumerate() {
                        if val > max_val {
                            max_val = val;
                            max_idx = idx;
                        }
                    }
                    max_idx as u32
                },
                _ => {
                    return Err(tensor_op_error(
                        "tensor_operation",
                        "Unsupported tensor type".to_string(),
                    ))
                },
            };

            generated.push(next_token);

            // Check for EOS token
            if next_token == 50256 {
                break;
            }
        }

        Ok(generated)
    }

    /// Generate text using greedy decoding with KV-cache for speed
    pub fn generate_greedy_with_cache(&self, input_ids: Vec<u32>, max_length: usize) -> Result<Vec<u32>> {
        let mut generated = input_ids.clone();
        let mut cache = KVCache::new(self.transformer.config.n_layer);

        while generated.len() < max_length {
            // Prepare input - only process new tokens after first iteration
            let input_batch = vec![generated.clone()];

            // Forward pass with cache
            let hidden_states = self.transformer.forward_internal(
                &input_batch,
                None,
                Some(&mut cache),
            )?;

            // Apply LM head
            let logits = self.lm_head.forward(hidden_states)?;

            // Get logits for the last token
            let next_token = match &logits {
                Tensor::F32(arr) => {
                    let shape = arr.shape();
                    if shape.len() != 3 {
                        return Err(tensor_op_error(
                            "tensor_operation",
                            "Unsupported tensor type".to_string(),
                        ));
                    }
                    let seq_len = shape[1];
                    let last_logits = arr.slice(s![0, seq_len - 1, ..]);

                    // Find argmax
                    let mut max_idx = 0;
                    let mut max_val = f32::NEG_INFINITY;
                    for (idx, &val) in last_logits.iter().enumerate() {
                        if val > max_val {
                            max_val = val;
                            max_idx = idx;
                        }
                    }
                    max_idx as u32
                },
                _ => {
                    return Err(tensor_op_error(
                        "tensor_operation",
                        "Unsupported tensor type".to_string(),
                    ))
                },
            };

            generated.push(next_token);

            // Check for EOS token
            if next_token == 50256 {
                break;
            }
        }

        Ok(generated)
    }

    /// Generate text using beam search
    pub fn generate_beam_search(
        &self,
        input_ids: Vec<u32>,
        max_length: usize,
        num_beams: usize,
    ) -> Result<Vec<u32>> {
        if num_beams == 1 {
            return self.generate_greedy(input_ids, max_length);
        }

        // Initialize beams
        let mut beams = vec![(0.0, input_ids.clone()); num_beams];

        for _ in input_ids.len()..max_length {
            let mut candidates = Vec::new();

            for (score, sequence) in &beams {
                // Prepare input
                let input = TokenizedInput {
                    input_ids: sequence.clone(),
                    attention_mask: vec![1u8; sequence.len()],
                    token_type_ids: None,
                    special_tokens_mask: None,
                    offset_mapping: None,
                    overflowing_tokens: None,
                };

                // Forward pass
                let output = self.forward(input)?;

                // Get logits for the last token
                let logits = output.logits;
                let last_logits = match &logits {
                    Tensor::F32(arr) => {
                        let shape = arr.shape();
                        if shape.len() != 3 {
                            return Err(tensor_op_error(
                                "tensor_operation",
                                "Expected 3D logits tensor",
                            ));
                        }
                        let seq_len = shape[1];
                        {
                            let shape = arr.shape();
                            let vocab_size = shape[2];
                            let slice = arr.slice(s![0, seq_len - 1, ..]);
                            ArrayD::from_shape_vec(
                                IxDyn(&[vocab_size]),
                                slice.iter().cloned().collect(),
                            )
                            .unwrap()
                        }
                    },
                    _ => {
                        return Err(tensor_op_error(
                            "tensor_operation",
                            "Unsupported tensor type".to_string(),
                        ))
                    },
                };

                // Convert to log probabilities
                let log_probs = log_softmax(last_logits)?;

                // Get top k tokens for this beam
                let mut token_scores: Vec<(f32, usize)> =
                    log_probs.iter().enumerate().map(|(idx, &log_prob)| (log_prob, idx)).collect();
                token_scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

                // Add top candidates
                for (log_prob, token_idx) in token_scores.iter().take(num_beams) {
                    let new_score = score + log_prob;
                    let mut new_sequence = sequence.clone();
                    new_sequence.push(*token_idx as u32);
                    candidates.push((new_score, new_sequence));
                }
            }

            // Select top beams for next iteration
            candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            beams = candidates.into_iter().take(num_beams).collect();

            // Check if all beams ended with EOS
            if beams.iter().all(|(_, seq)| seq.last() == Some(&50256)) {
                break;
            }
        }

        // Return the best sequence
        Ok(beams[0].1.clone())
    }
}

impl Model for Gpt2LMHeadModel {
    type Config = Gpt2Config;
    type Input = TokenizedInput;
    type Output = Gpt2LMOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Get transformer output
        let transformer_output = self.transformer.forward(input)?;

        // Apply language modeling head
        let logits = self.lm_head.forward(transformer_output.last_hidden_state)?;

        Ok(Gpt2LMOutput {
            logits,
            past_key_values: transformer_output.past_key_values,
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

/// Key-Value cache for a single layer
#[derive(Clone, Debug)]
pub struct LayerCache {
    pub key: Option<Tensor>,
    pub value: Option<Tensor>,
}

impl Default for LayerCache {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerCache {
    pub fn new() -> Self {
        Self {
            key: None,
            value: None,
        }
    }
}

/// Key-Value cache for all layers
#[derive(Clone, Debug)]
pub struct KVCache {
    pub layers: Vec<LayerCache>,
}

impl KVCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| LayerCache::new()).collect(),
        }
    }
}

/// Output from GPT-2 base model
pub struct Gpt2Output {
    pub last_hidden_state: Tensor,
    pub past_key_values: Option<KVCache>,
}

/// Output from GPT-2 language modeling head
pub struct Gpt2LMOutput {
    pub logits: Tensor,
    pub past_key_values: Option<KVCache>,
}

/// GPT-2 transformer block
#[derive(Clone)]
struct Gpt2Block {
    ln_1: LayerNorm,
    attn: Gpt2Attention,
    ln_2: LayerNorm,
    mlp: Gpt2MLP,
}

impl Gpt2Block {
    fn new(config: &Gpt2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    fn new_with_device(config: &Gpt2Config, device: Device) -> Result<Self> {
        Ok(Self {
            ln_1: LayerNorm::new_simple(config.n_embd, config.layer_norm_epsilon),
            attn: Gpt2Attention::new_with_device(config, device)?,
            ln_2: LayerNorm::new_simple(config.n_embd, config.layer_norm_epsilon),
            mlp: Gpt2MLP::new_with_device(config, device)?,
        })
    }

    fn to_device(mut self, device: Device) -> Self {
        self.attn = self.attn.to_device(device);
        self.mlp = self.mlp.to_device(device);
        self
    }

    fn load_weights(&mut self, reader: &mut dyn WeightReader, prefix: &str) -> Result<()> {
        // Load layer norm weights
        self.ln_1.set_weight(reader.read_tensor(&format!("{}.ln_1.weight", prefix))?)?;
        self.ln_1.set_bias(reader.read_tensor(&format!("{}.ln_1.bias", prefix))?)?;

        self.ln_2.set_weight(reader.read_tensor(&format!("{}.ln_2.weight", prefix))?)?;
        self.ln_2.set_bias(reader.read_tensor(&format!("{}.ln_2.bias", prefix))?)?;

        // Load attention weights
        self.attn.load_weights(reader, &format!("{}.attn", prefix))?;

        // Load MLP weights
        self.mlp.load_weights(reader, &format!("{}.mlp", prefix))?;

        Ok(())
    }

    fn parameter_count(&self) -> usize {
        self.ln_1.parameter_count()
            + self.attn.parameter_count()
            + self.ln_2.parameter_count()
            + self.mlp.parameter_count()
    }

    #[allow(dead_code)]
    fn forward(&self, hidden_states: Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.forward_with_cache(hidden_states, attention_mask, None)
    }

    fn forward_with_cache(
        &self,
        hidden_states: Tensor,
        attention_mask: Option<&Tensor>,
        layer_cache: Option<&mut LayerCache>,
    ) -> Result<Tensor> {
        // Pre-norm architecture (GPT-2 style)
        let residual = hidden_states.clone();

        // Self-attention with residual and optional caching
        let norm_hidden = self.ln_1.forward(hidden_states)?;
        let attn_output = self.attn.forward_with_cache(norm_hidden, attention_mask, layer_cache)?;
        let hidden_states = residual.add(&attn_output)?;

        // MLP with residual
        let residual = hidden_states.clone();
        let norm_hidden = self.ln_2.forward(hidden_states)?;
        let mlp_output = self.mlp.forward(norm_hidden)?;
        let hidden_states = residual.add(&mlp_output)?;

        Ok(hidden_states)
    }
}

/// GPT-2 attention module
#[derive(Clone)]
#[allow(dead_code)]
struct Gpt2Attention {
    n_head: usize,
    d_head: usize,
    c_attn: Linear, // Combined QKV projection
    c_proj: Linear, // Output projection
    #[allow(dead_code)]
    attn_dropout: f32,
    resid_dropout: f32,
}

impl Gpt2Attention {
    fn new(config: &Gpt2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    fn new_with_device(config: &Gpt2Config, device: Device) -> Result<Self> {
        if config.n_embd % config.n_head != 0 {
            return Err(invalid_config(
                "n_embd",
                "n_embd must be divisible by n_head",
            ));
        }

        let d_head = config.n_embd / config.n_head;

        Ok(Self {
            n_head: config.n_head,
            d_head,
            c_attn: Linear::new_with_device(config.n_embd, 3 * config.n_embd, true, device),
            c_proj: Linear::new_with_device(config.n_embd, config.n_embd, true, device),
            attn_dropout: config.attn_pdrop,
            resid_dropout: config.resid_pdrop,
        })
    }

    fn to_device(self, device: Device) -> Self {
        Self {
            n_head: self.n_head,
            d_head: self.d_head,
            c_attn: self.c_attn.to_device(device),
            c_proj: self.c_proj.to_device(device),
            attn_dropout: self.attn_dropout,
            resid_dropout: self.resid_dropout,
        }
    }

    fn load_weights(&mut self, reader: &mut dyn WeightReader, prefix: &str) -> Result<()> {
        // Load combined QKV weights
        // PyTorch stores as [out, in], we need [in, out], so transpose
        let c_attn_weight = reader.read_tensor(&format!("{}.c_attn.weight", prefix))?;
        self.c_attn.set_weight(transpose_tensor(c_attn_weight)?)?;
        self.c_attn.set_bias(reader.read_tensor(&format!("{}.c_attn.bias", prefix))?)?;

        // Load output projection weights (also needs transpose)
        let c_proj_weight = reader.read_tensor(&format!("{}.c_proj.weight", prefix))?;
        self.c_proj.set_weight(transpose_tensor(c_proj_weight)?)?;
        self.c_proj.set_bias(reader.read_tensor(&format!("{}.c_proj.bias", prefix))?)?;

        Ok(())
    }

    fn parameter_count(&self) -> usize {
        self.c_attn.parameter_count() + self.c_proj.parameter_count()
    }

    #[allow(dead_code)]
    fn forward(&self, hidden_states: Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        self.forward_with_cache(hidden_states, attention_mask, None)
    }

    fn forward_with_cache(
        &self,
        hidden_states: Tensor,
        attention_mask: Option<&Tensor>,
        layer_cache: Option<&mut LayerCache>,
    ) -> Result<Tensor> {
        // Get the shape of hidden states and ensure it's 3D
        let (hidden_states, was_2d) = match &hidden_states {
            Tensor::F32(arr) => {
                if arr.ndim() == 2 {
                    // Add batch dimension: [seq_len, hidden_size] -> [1, seq_len, hidden_size]
                    let _shape = arr.shape();
                    let expanded = arr.clone().insert_axis(Axis(0)).to_owned();
                    (Tensor::F32(expanded), true)
                } else {
                    (hidden_states, false)
                }
            },
            _ => (hidden_states, false),
        };

        let shape = hidden_states.shape();
        let batch_size = shape[0];
        let seq_len = shape[1];
        let hidden_size = shape[2];

        // Project to Q, K, V using the combined projection
        let qkv = self.c_attn.forward(hidden_states)?;

        // Split QKV into separate Q, K, V tensors
        match &qkv {
            Tensor::F32(arr) => {
                // qkv shape: [batch, seq_len, 3 * hidden_size]
                // Split into 3 equal parts
                let _qkv_shape = arr.shape();
                let chunk_size = hidden_size;

                // Extract Q, K, V
                let q = arr.slice(s![.., .., ..chunk_size]).to_owned();
                let k = arr.slice(s![.., .., chunk_size..2 * chunk_size]).to_owned();
                let v = arr.slice(s![.., .., 2 * chunk_size..]).to_owned();

                // Reshape for multi-head attention
                // From [batch, seq_len, hidden_size] to [batch, seq_len, n_heads, head_dim]
                let head_dim = self.d_head;
                let n_heads = self.n_head;

                let q = q
                    .to_shape(IxDyn(&[batch_size, seq_len, n_heads, head_dim]))
                    .map_err(|_| TrustformersError::shape_error("Failed to reshape Q".into()))?
                    .to_owned();
                let k = k
                    .to_shape(IxDyn(&[batch_size, seq_len, n_heads, head_dim]))
                    .map_err(|_| TrustformersError::shape_error("Failed to reshape K".into()))?
                    .to_owned();
                let v = v
                    .to_shape(IxDyn(&[batch_size, seq_len, n_heads, head_dim]))
                    .map_err(|_| TrustformersError::shape_error("Failed to reshape V".into()))?
                    .to_owned();

                // Transpose to [batch, n_heads, seq_len, head_dim]
                let q = q.permuted_axes(vec![0, 2, 1, 3]);
                let k = k.permuted_axes(vec![0, 2, 1, 3]);
                let v = v.permuted_axes(vec![0, 2, 1, 3]);

                // Compute attention scores
                // Q * K^T / sqrt(head_dim)
                let scale = 1.0 / (head_dim as f32).sqrt();
                let k_t = k.clone().permuted_axes(vec![0, 1, 3, 2]); // Transpose last two dims

                // Compute Q * K^T
                let mut scores =
                    ArrayD::<f32>::zeros(IxDyn(&[batch_size, n_heads, seq_len, seq_len]));
                for b in 0..batch_size {
                    for h in 0..n_heads {
                        let q_head = q.slice(s![b, h, .., ..]);
                        let k_head_t = k_t.slice(s![b, h, .., ..]);
                        let score = q_head.dot(&k_head_t);
                        scores.slice_mut(s![b, h, .., ..]).assign(&score);
                    }
                }
                scores *= scale;

                // Apply attention mask if provided
                if let Some(mask) = attention_mask {
                    match mask {
                        Tensor::F32(mask_arr) => {
                            scores += mask_arr;
                        },
                        _ => {
                            return Err(tensor_op_error(
                                "tensor_operation",
                                "Attention mask must be F32",
                            ));
                        },
                    }
                }

                // Softmax
                let mut attention_probs = scores.clone();
                for b in 0..batch_size {
                    for h in 0..n_heads {
                        for i in 0..seq_len {
                            let mut row = attention_probs.slice_mut(s![b, h, i, ..]);
                            let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                            row.mapv_inplace(|x| (x - max_val).exp());
                            let sum: f32 = row.iter().sum();
                            row.mapv_inplace(|x| x / sum);
                        }
                    }
                }

                // Apply dropout (skip for now during inference)

                // Compute attention output: attention_probs * V
                let mut output =
                    ArrayD::<f32>::zeros(IxDyn(&[batch_size, n_heads, seq_len, head_dim]));
                for b in 0..batch_size {
                    for h in 0..n_heads {
                        let attn_probs_head = attention_probs.slice(s![b, h, .., ..]);
                        let v_head = v.slice(s![b, h, .., ..]);
                        let out = attn_probs_head.dot(&v_head);
                        output.slice_mut(s![b, h, .., ..]).assign(&out);
                    }
                }

                // Transpose back to [batch, seq_len, n_heads, head_dim]
                let output = output.permuted_axes(vec![0, 2, 1, 3]);

                // Reshape to [batch, seq_len, hidden_size]
                let output = output
                    .to_shape(IxDyn(&[batch_size, seq_len, hidden_size]))
                    .map_err(|_| TrustformersError::shape_error("Failed to reshape output".into()))?
                    .to_owned();

                // Update cache if provided
                if let Some(cache) = layer_cache {
                    cache.key = Some(Tensor::F32(k.clone()));
                    cache.value = Some(Tensor::F32(v.clone()));
                }

                // Apply output projection
                let output = self.c_proj.forward(Tensor::F32(output))?;

                // Remove batch dimension if input was 2D
                if was_2d {
                    match output {
                        Tensor::F32(arr) => Ok(Tensor::F32(arr.remove_axis(Axis(0)))),
                        _ => Ok(output),
                    }
                } else {
                    Ok(output)
                }
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported tensor type".to_string(),
            )),
        }
    }
}

/// GPT-2 MLP (feedforward) module
#[derive(Clone)]
struct Gpt2MLP {
    c_fc: Linear,
    c_proj: Linear,
    act_fn: ActivationType,
    #[allow(dead_code)]
    dropout: f32,
}

impl Gpt2MLP {
    fn new(config: &Gpt2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    fn new_with_device(config: &Gpt2Config, device: Device) -> Result<Self> {
        let inner_dim = if let Some(dim) = config.n_inner { dim } else { 4 * config.n_embd };

        Ok(Self {
            c_fc: Linear::new_with_device(config.n_embd, inner_dim, true, device),
            c_proj: Linear::new_with_device(inner_dim, config.n_embd, true, device),
            act_fn: ActivationType::from_str(&config.activation_function)?,
            dropout: config.resid_pdrop,
        })
    }

    fn to_device(self, device: Device) -> Self {
        Self {
            c_fc: self.c_fc.to_device(device),
            c_proj: self.c_proj.to_device(device),
            act_fn: self.act_fn,
            dropout: self.dropout,
        }
    }

    fn load_weights(&mut self, reader: &mut dyn WeightReader, prefix: &str) -> Result<()> {
        // Transpose MLP weights too
        let c_fc_weight = reader.read_tensor(&format!("{}.c_fc.weight", prefix))?;
        self.c_fc.set_weight(transpose_tensor(c_fc_weight)?)?;
        self.c_fc.set_bias(reader.read_tensor(&format!("{}.c_fc.bias", prefix))?)?;

        let c_proj_weight = reader.read_tensor(&format!("{}.c_proj.weight", prefix))?;
        self.c_proj.set_weight(transpose_tensor(c_proj_weight)?)?;
        self.c_proj.set_bias(reader.read_tensor(&format!("{}.c_proj.bias", prefix))?)?;

        Ok(())
    }

    fn parameter_count(&self) -> usize {
        self.c_fc.parameter_count() + self.c_proj.parameter_count()
    }

    fn forward(&self, hidden_states: Tensor) -> Result<Tensor> {
        let hidden_states = self.c_fc.forward(hidden_states)?;
        let hidden_states = self.act_fn.apply(hidden_states)?;
        self.c_proj.forward(hidden_states)
    }
}

/// Activation function types
#[derive(Clone)]
enum ActivationType {
    Gelu,
    Relu,
    Swish,
}

impl ActivationType {
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "gelu" | "gelu_new" => Ok(Self::Gelu),
            "relu" => Ok(Self::Relu),
            "swish" | "silu" => Ok(Self::Swish),
            _ => Err(invalid_config(
                "activation",
                format!("Unknown activation: {}", s),
            )),
        }
    }

    fn apply(&self, x: Tensor) -> Result<Tensor> {
        match self {
            Self::Gelu => gelu(x),
            Self::Relu => relu(x),
            Self::Swish => swish(x),
        }
    }
}

/// GELU activation function
fn gelu(x: Tensor) -> Result<Tensor> {
    // GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    // For now, use approximate version
    match &x {
        Tensor::F32(arr) => {
            let result = arr.mapv(|val| {
                val * 0.5
                    * (1.0
                        + ((2.0_f32 / std::f32::consts::PI).sqrt()
                            * (val + 0.044715 * val.powi(3)))
                        .tanh())
            });
            Ok(Tensor::F32(result))
        },
        _ => Err(tensor_op_error(
            "tensor_operation",
            "GELU only supports F32 tensors",
        )),
    }
}

/// ReLU activation function
fn relu(x: Tensor) -> Result<Tensor> {
    match &x {
        Tensor::F32(arr) => {
            let result = arr.mapv(|val| val.max(0.0));
            Ok(Tensor::F32(result))
        },
        _ => Err(tensor_op_error(
            "tensor_operation",
            "ReLU only supports F32 tensors",
        )),
    }
}

/// Swish/SiLU activation function
fn swish(x: Tensor) -> Result<Tensor> {
    match &x {
        Tensor::F32(arr) => {
            let result = arr.mapv(|val| val * (1.0 / (1.0 + (-val).exp())));
            Ok(Tensor::F32(result))
        },
        _ => Err(tensor_op_error(
            "tensor_operation",
            "Swish only supports F32 tensors",
        )),
    }
}

/// Create a causal mask for attention
fn create_causal_mask(seq_len: usize) -> Result<Tensor> {
    let mut mask = ArrayD::<f32>::zeros(IxDyn(&[1, 1, seq_len, seq_len]));

    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[[0, 0, i, j]] = f32::NEG_INFINITY;
        }
    }

    Ok(Tensor::F32(mask))
}

/// Apply top-k filtering to logits
fn apply_top_k_filtering(logits: ArrayD<f32>, k: usize) -> Result<ArrayD<f32>> {
    let mut result = logits.clone();
    let mut indices_and_values: Vec<(usize, f32)> =
        logits.iter().enumerate().map(|(idx, &val)| (idx, val)).collect();

    // Sort by value in descending order
    indices_and_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Set all values outside top-k to -inf
    for (idx, _) in indices_and_values.iter().skip(k) {
        result[*idx] = f32::NEG_INFINITY;
    }

    Ok(result)
}

/// Apply top-p (nucleus) filtering to logits
fn apply_top_p_filtering(logits: ArrayD<f32>, p: f32) -> Result<ArrayD<f32>> {
    // Convert to probabilities
    let probs = softmax(logits.clone())?;

    let mut indices_and_probs: Vec<(usize, f32)> =
        probs.iter().enumerate().map(|(idx, &prob)| (idx, prob)).collect();

    // Sort by probability in descending order
    indices_and_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Find the smallest set of tokens with cumulative probability > p
    let mut cumsum = 0.0;
    let mut cutoff_idx = indices_and_probs.len();

    for (i, (_, prob)) in indices_and_probs.iter().enumerate() {
        cumsum += prob;
        if cumsum > p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Create result with -inf for tokens outside the nucleus
    let mut result = logits;
    let selected_indices: std::collections::HashSet<_> =
        indices_and_probs.iter().take(cutoff_idx).map(|(idx, _)| *idx).collect();

    for (idx, val) in result.iter_mut().enumerate() {
        if !selected_indices.contains(&idx) {
            *val = f32::NEG_INFINITY;
        }
    }

    Ok(result)
}

/// Sample from logits using multinomial sampling
fn sample_from_logits(logits: ArrayD<f32>) -> Result<u32> {
    use scirs2_core::random::*; // SciRS2 Integration Policy (includes WeightedIndex)

    // Convert to probabilities
    let probs = softmax(logits)?;

    // Create weighted distribution
    let weights: Vec<f32> = probs.iter().copied().collect();
    let dist = WeightedIndex::new(weights).map_err(|e| {
        TrustformersError::model_error(format!("Failed to create distribution: {}", e))
    })?;

    // Sample
    let mut rng = thread_rng(); // From scirs2_core::random
    Ok(rng.sample(&dist) as u32)
}

/// Compute softmax of logits
fn softmax(logits: ArrayD<f32>) -> Result<ArrayD<f32>> {
    // Find max for numerical stability
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Compute exp(x - max)
    let exp_vals = logits.mapv(|x| (x - max_val).exp());

    // Sum of exp values
    let sum: f32 = exp_vals.iter().sum();

    // Normalize
    Ok(exp_vals / sum)
}

/// Compute log softmax of logits
fn log_softmax(logits: ArrayD<f32>) -> Result<ArrayD<f32>> {
    // Find max for numerical stability
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Compute log(sum(exp(x - max))) + max
    let shifted = logits.mapv(|x| x - max_val);
    let exp_sum = shifted.mapv(|x| x.exp()).sum();
    let log_sum_exp = exp_sum.ln() + max_val;

    // Return log probabilities
    Ok(logits.mapv(|x| x - log_sum_exp))
}

/// Stack a vector of tensors into a batch tensor
fn stack_tensors(tensors: &[Tensor]) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(tensor_op_error(
            "tensor_operation",
            "Cannot stack empty tensor list".to_string(),
        ));
    }

    match &tensors[0] {
        Tensor::F32(first_arr) => {
            let first_shape = first_arr.shape();
            let batch_size = tensors.len();

            // Create new shape with batch dimension
            let mut new_shape = vec![batch_size];
            new_shape.extend_from_slice(first_shape);

            // Collect all tensor data
            let mut data = Vec::new();
            for tensor in tensors {
                match tensor {
                    Tensor::F32(arr) => {
                        if arr.shape() != first_shape {
                            return Err(TrustformersError::shape_error(
                                "All tensors must have the same shape for stacking".to_string(),
                            ));
                        }
                        data.extend(arr.iter().cloned());
                    },
                    _ => {
                        return Err(tensor_op_error(
                            "tensor_operation",
                            "All tensors must be F32 for stacking".to_string(),
                        ))
                    },
                }
            }

            // Create stacked array
            let stacked = ArrayD::from_shape_vec(IxDyn(&new_shape), data).map_err(|_| {
                TrustformersError::shape_error("Failed to create stacked tensor".into())
            })?;

            Ok(Tensor::F32(stacked))
        },
        _ => Err(tensor_op_error(
            "tensor_operation",
            "Only F32 tensors supported for stacking".to_string(),
        )),
    }
}
