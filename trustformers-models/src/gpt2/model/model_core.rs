//! Core GPT-2 model structs and implementations
//!
//! Contains Gpt2Model, Gpt2LMHeadModel, KVCache, LayerCache, and output types.

use scirs2_core::ndarray::{s, ArrayD, Axis, IxDyn}; // SciRS2 Integration Policy
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result, TrustformersError},
    layers::{Embedding, LayerNorm, Linear},
    tensor::Tensor,
    traits::{Config, Layer, Model, TokenizedInput, WeightReader},
};

use super::model_blocks::Gpt2Block;
use super::model_ops::{
    apply_top_k_filtering, apply_top_p_filtering, create_causal_mask, log_softmax,
    sample_from_logits, stack_tensors,
};
use crate::gpt2::config::Gpt2Config;
use crate::weight_loading::checkpoint::CheckpointReader;

/// Transpose a 2D tensor (swap dimensions 0 and 1)
/// PyTorch Linear weights are [out_features, in_features]
/// but we need [in_features, out_features] for our matmul
pub(crate) fn transpose_tensor(tensor: Tensor) -> Result<Tensor> {
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

/// Read a `[batch, seq, width]` or `[seq, width]` F32 tensor as `seq` rows,
/// taking the **first** batch element when a batch axis is present.
///
/// # Errors
///
/// Fails for non-F32 tensors and for ranks other than 2 or 3.
fn rows_of_first_batch(tensor: &Tensor, label: &str) -> Result<Vec<Vec<f32>>> {
    let Tensor::F32(arr) = tensor else {
        return Err(TrustformersError::tensor_op_error(
            "only CPU F32 tensors can be read row-wise",
            label,
        ));
    };
    let shape = arr.shape();
    let (seq_len, width) = match shape.len() {
        2 => (shape[0], shape[1]),
        3 => (shape[1], shape[2]),
        other => {
            return Err(TrustformersError::shape_error(format!(
                "expected 2-D or 3-D {label}, got {other} dimensions"
            )))
        },
    };

    let flat: Vec<f32> = arr.iter().copied().collect();
    // With a batch axis the first element's rows come first, which is what the
    // single-sequence generation paths need.
    Ok((0..seq_len).map(|i| flat[i * width..(i + 1) * width].to_vec()).collect())
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

    /// Upload model weights to GPU (Metal)
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn weights_to_gpu(&mut self, device: &Device) -> Result<()> {
        if !matches!(device, Device::Metal(_)) {
            return Ok(());
        }
        self.device = *device;
        self.wte.weights_to_gpu(device)?;
        self.wpe.weights_to_gpu(device)?;
        for block in &mut self.h {
            block.weights_to_gpu(device)?;
        }
        self.ln_f.weights_to_gpu(device)?;
        Ok(())
    }

    /// Upload model weights to GPU (CUDA)
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    pub fn weights_to_gpu_cuda(&mut self, device: &Device) -> Result<()> {
        if !matches!(device, Device::CUDA(_)) {
            return Ok(());
        }
        self.device = *device;
        for block in &mut self.h {
            block.weights_to_gpu_cuda(device)?;
        }
        self.ln_f.weights_to_gpu_cuda(device)?;
        Ok(())
    }

    /// Load weights from a WeightReader (e.g., SafeTensors)
    pub fn load_weights_from_reader(&mut self, reader: &mut dyn WeightReader) -> Result<()> {
        // Detect if weights have "transformer." prefix (HuggingFace format)
        let tensor_names = reader.list_tensors();
        let has_transformer_prefix =
            tensor_names.iter().any(|name| name.starts_with("transformer."));
        let prefix = if has_transformer_prefix { "transformer." } else { "" };

        // Load embeddings
        self.wte.set_weight(reader.read_tensor(&format!("{}wte.weight", prefix))?)?;
        self.wpe.set_weight(reader.read_tensor(&format!("{}wpe.weight", prefix))?)?;

        // Load transformer blocks
        for (i, block) in self.h.iter_mut().enumerate() {
            let block_prefix = format!("{}h.{}", prefix, i);
            block.load_weights(reader, &block_prefix)?;
        }

        // Load final layer norm
        self.ln_f.set_weight(reader.read_tensor(&format!("{}ln_f.weight", prefix))?)?;
        self.ln_f.set_bias(reader.read_tensor(&format!("{}ln_f.bias", prefix))?)?;

        Ok(())
    }

    /// Load weights from a WeightLoader (e.g., HuggingFace loader)
    pub fn load_weights_from_loader(
        &mut self,
        loader: &mut dyn crate::weight_loading::WeightLoader,
    ) -> Result<()> {
        // Detect if weights have "transformer." prefix (HuggingFace format)
        let tensor_names = loader.list_tensors()?;
        let has_transformer_prefix =
            tensor_names.iter().any(|name| name.starts_with("transformer."));
        let prefix = if has_transformer_prefix { "transformer." } else { "" };

        // Load embeddings
        self.wte.set_weight(loader.load_tensor(&format!("{}wte.weight", prefix))?)?;
        self.wpe.set_weight(loader.load_tensor(&format!("{}wpe.weight", prefix))?)?;

        // Load transformer blocks
        for (i, block) in self.h.iter_mut().enumerate() {
            let block_prefix = format!("{}h.{}", prefix, i);
            block.load_weights_from_loader(loader, &block_prefix)?;
        }

        // Load final layer norm
        self.ln_f.set_weight(loader.load_tensor(&format!("{}ln_f.weight", prefix))?)?;
        self.ln_f.set_bias(loader.load_tensor(&format!("{}ln_f.bias", prefix))?)?;

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

        // Determine starting position based on cache state
        let position_offset = if let Some(ref cache) = past_key_values {
            // If cache exists and has keys, start from past sequence length
            if let Some(first_layer_cache) = cache.layers.first() {
                match &first_layer_cache.key {
                    Some(Tensor::F32(ref past_k)) => {
                        past_k.shape()[1] as u32 // past_seq_len
                    },
                    #[cfg(all(target_os = "macos", feature = "metal"))]
                    Some(Tensor::Metal(ref metal_data)) => {
                        metal_data.shape[1] as u32 // past_seq_len from Metal tensor
                    },
                    None => 0,
                    _ => 0,
                }
            } else {
                0
            }
        } else {
            0
        };

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
                // Start from position_offset (for KV-cache continuation)
                (position_offset..(position_offset + seq_len as u32)).collect()
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
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Tensor::Metal(metal_data) => {
                // For Metal tensors, check shape and add batch dimension if needed
                if metal_data.shape.len() == 2 {
                    // Convert to CPU, add dimension, convert back
                    let cpu_tensor = hidden_states.to_device_enum(&Device::CPU)?;
                    if let Tensor::F32(arr) = cpu_tensor {
                        let batched = Tensor::F32(arr.insert_axis(Axis(0)).to_owned());
                        hidden_states = batched.to_device_enum(&Device::Metal(0))?;
                    }
                }
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported tensor type".to_string(),
                ))
            },
        }

        // Upload hidden states to GPU if weights are on GPU (enables GPU-to-GPU Linear pipeline)
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if matches!(self.device, Device::Metal(_)) {
                hidden_states = hidden_states.to_device_enum(&self.device)?;
            }
        }

        #[cfg(feature = "cuda")]
        {
            if matches!(self.device, Device::CUDA(_)) {
                hidden_states = hidden_states.to_device_enum(&self.device)?;
            }
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

    /// Load pretrained weights from a safetensors or PyTorch byte stream.
    ///
    /// The stream is parsed by [`Checkpoint`] and bound through
    /// [`Gpt2Model::load_weights_from_reader`], which maps HuggingFace's GPT-2
    /// names (including the `transformer.` prefix used by task checkpoints) and
    /// transposes the Conv1D weights that HF stores as `[in, out]`.
    ///
    /// This used to return `"Use load_weights_from_reader instead"` for every
    /// call, which left `Model::load_pretrained` unusable for GPT-2.
    ///
    /// # Errors
    ///
    /// Fails when the container cannot be parsed or when a required tensor is
    /// missing from the checkpoint.
    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        let mut checkpoint_reader = CheckpointReader::from_reader(reader)?;
        self.load_weights_from_reader(&mut checkpoint_reader)
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

    /// Enumerate the backbone's live parameters under HuggingFace GPT-2 names.
    ///
    /// The names are exactly the ones [`Gpt2Model::load_weights_from_reader`]
    /// looks up, minus the optional `transformer.` prefix that only task
    /// checkpoints carry: `wte.weight`, `wpe.weight`, `h.{i}.…`, `ln_f.{weight,
    /// bias}`. `Gpt2LMHeadModel` re-publishes them *with* the prefix, matching
    /// the `transformer.`-prefixed checkpoints HF ships for the LM head model.
    ///
    /// Ordering is embeddings, then blocks in index order, then the final norm —
    /// stable across calls, as the trait requires.
    ///
    /// # Weight layout
    ///
    /// See [`Gpt2Block::collect_named_parameters`](super::model_blocks::Gpt2Block::collect_named_parameters):
    /// the four `Conv1D`-derived projections per block are exposed in this
    /// crate's `[out, in]` layout, not HuggingFace's on-disk `[in, out]`.
    fn named_tensors(&self) -> Vec<(String, &Tensor)> {
        let mut tensors = Vec::with_capacity(2 + self.h.len() * 12 + 2);
        self.collect_named_parameters("", &mut tensors);
        tensors
    }

    fn named_tensors_mut(&mut self) -> Vec<(String, &mut Tensor)> {
        let mut tensors = Vec::with_capacity(2 + self.h.len() * 12 + 2);
        self.collect_named_parameters_mut("", &mut tensors);
        tensors
    }
}

impl Gpt2Model {
    /// Append every backbone parameter to `into`, prefixed with `prefix`.
    ///
    /// Factored out of [`Model::named_tensors`] so `Gpt2LMHeadModel` can publish
    /// the same tensors under the `transformer.` prefix without duplicating the
    /// name table.
    pub(crate) fn collect_named_parameters<'a>(
        &'a self,
        prefix: &str,
        into: &mut Vec<(String, &'a Tensor)>,
    ) {
        let scope = |name: &str| {
            if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}.{name}")
            }
        };

        self.wte.collect_named_parameters(&scope("wte"), into);
        self.wpe.collect_named_parameters(&scope("wpe"), into);
        for (index, block) in self.h.iter().enumerate() {
            block.collect_named_parameters(&scope(&format!("h.{index}")), into);
        }
        self.ln_f.collect_named_parameters(&scope("ln_f"), into);
    }

    /// Mutable counterpart of [`Gpt2Model::collect_named_parameters`].
    pub(crate) fn collect_named_parameters_mut<'a>(
        &'a mut self,
        prefix: &str,
        into: &mut Vec<(String, &'a mut Tensor)>,
    ) {
        // Every name is materialised before the mutable borrows start, because
        // `scope` would otherwise need to borrow `prefix` across them.
        let scope = |name: &str| {
            if prefix.is_empty() {
                name.to_string()
            } else {
                format!("{prefix}.{name}")
            }
        };
        let wte_name = scope("wte");
        let wpe_name = scope("wpe");
        let block_names: Vec<String> =
            (0..self.h.len()).map(|index| scope(&format!("h.{index}"))).collect();
        let ln_f_name = scope("ln_f");

        self.wte.collect_named_parameters_mut(&wte_name, into);
        self.wpe.collect_named_parameters_mut(&wpe_name, into);
        for (block, name) in self.h.iter_mut().zip(&block_names) {
            block.collect_named_parameters_mut(name, into);
        }
        self.ln_f.collect_named_parameters_mut(&ln_f_name, into);
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
        Self::new_with_device(config, Device::CPU)
    }

    /// Create a GPT-2 LM Head model with specified device (CPU, Metal, CUDA, etc.)
    pub fn new_with_device(config: Gpt2Config, device: Device) -> Result<Self> {
        let transformer = Gpt2Model::new_with_device(config.clone(), device)?;
        let lm_head = Linear::new_with_device(config.n_embd, config.vocab_size, true, device);

        Ok(Self {
            transformer,
            lm_head,
        })
    }

    /// Get the device this model uses
    pub fn device(&self) -> Device {
        self.transformer.device()
    }

    /// Move this model to a different device
    pub fn to_device(mut self, device: Device) -> Self {
        self.transformer = self.transformer.to_device(device);
        self.lm_head = self.lm_head.to_device(device);
        self
    }

    /// Upload model weights to GPU (Metal)
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn weights_to_gpu(&mut self, device: &Device) -> Result<()> {
        if !matches!(device, Device::Metal(_)) {
            return Ok(());
        }
        self.transformer.weights_to_gpu(device)?;
        self.lm_head.weights_to_gpu(device)?;
        Ok(())
    }

    /// Upload model weights to GPU (CUDA)
    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    pub fn weights_to_gpu_cuda(&mut self, device: &Device) -> Result<()> {
        if !matches!(device, Device::CUDA(_)) {
            return Ok(());
        }
        self.transformer.weights_to_gpu_cuda(device)?;
        self.lm_head.weights_to_gpu_cuda(device)?;
        Ok(())
    }

    /// Load model weights from HuggingFace format
    pub fn load_from_path(&mut self, model_path: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::weight_loading::auto_create_loader;

        let model_path = model_path.as_ref();

        let mut loader = auto_create_loader(model_path, None)?;

        // Load transformer weights using WeightLoader
        self.transformer.load_weights_from_loader(&mut *loader)?;

        // Load LM head weights
        // Try with "lm_head" first, then fallback to "transformer.wte" (weight tying)
        match loader.load_tensor("lm_head.weight") {
            Ok(lm_head_weight) => {
                self.lm_head.set_weight(lm_head_weight)?;
            },
            Err(_) => {
                // Weight tying: LM head shares weights with token embeddings
                if let Ok(wte_weight) = loader.load_tensor("transformer.wte.weight") {
                    self.lm_head.set_weight(wte_weight)?;
                } else if let Ok(wte_weight) = loader.load_tensor("wte.weight") {
                    self.lm_head.set_weight(wte_weight)?;
                }
            },
        }

        loader.close()?;
        Ok(())
    }

    /// The transformer backbone under the language-modelling head.
    pub fn transformer(&self) -> &Gpt2Model {
        &self.transformer
    }

    /// Last-layer hidden states and next-token logits for one sequence.
    ///
    /// Returns `(logits, hidden_states)` where `logits` are the vocabulary
    /// scores for the position after the final input token, and `hidden_states`
    /// holds the post-`ln_f` representation of every input position as
    /// `seq_len` rows of `n_embd` values.
    ///
    /// Contrastive search needs the representations themselves, not just the
    /// logits, to compute its degeneration penalty; this is the accessor that
    /// makes that possible without a second forward pass.
    ///
    /// # Errors
    ///
    /// Fails when `input_ids` is empty, when the forward pass fails, or when the
    /// backbone returns a tensor this method cannot read on the CPU.
    pub fn logits_and_hidden_states(&self, input_ids: &[u32]) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        if input_ids.is_empty() {
            return Err(TrustformersError::model_error(
                "logits_and_hidden_states requires at least one input token".to_string(),
            ));
        }

        let batch = vec![input_ids.to_vec()];
        let hidden = self.transformer.forward_internal(&batch, None, None)?;
        let logits = self.lm_head.forward(hidden.clone())?;

        let hidden_rows = rows_of_first_batch(&hidden, "hidden states")?;
        let logit_rows = rows_of_first_batch(&logits, "logits")?;
        let last_logits = logit_rows.last().cloned().ok_or_else(|| {
            TrustformersError::model_error("forward pass produced no logits".to_string())
        })?;

        Ok((last_logits, hidden_rows))
    }

    /// Forward pass with KV-cache support for efficient generation
    pub fn forward_with_cache(
        &self,
        input: TokenizedInput,
        past_key_values: &mut Option<KVCache>,
    ) -> Result<Gpt2LMOutput> {
        // Get transformer output with KV-cache
        let batch_input_ids = vec![input.input_ids.clone()];
        let transformer_output =
            self.transformer
                .forward_internal(&batch_input_ids, None, past_key_values.as_mut())?;

        // Apply language modeling head
        let logits = self.lm_head.forward(transformer_output)?;

        Ok(Gpt2LMOutput {
            logits,
            past_key_values: past_key_values.clone(),
        })
    }

    /// Load weights from a WeightReader (e.g., SafeTensors)
    pub fn load_weights_from_reader(&mut self, reader: &mut dyn WeightReader) -> Result<()> {
        // Load transformer weights (handles prefix detection internally)
        self.transformer.load_weights_from_reader(reader)?;

        // Load language modeling head weights
        // Note: In GPT-2, the LM head shares weights with the input embeddings
        // Try with and without "transformer." prefix
        let wte_weight = reader
            .read_tensor("transformer.wte.weight")
            .or_else(|_| reader.read_tensor("wte.weight"))?;
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
            // Prepare input (full sequence - stable version)
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
                        .map_err(|e| {
                            tensor_op_error(
                                "from_shape_vec",
                                format!("Failed to create array from shape: {}", e),
                            )
                        })?
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
                #[cfg(all(target_os = "macos", feature = "metal"))]
                Tensor::Metal(metal_data) => {
                    use trustformers_core::gpu_ops::metal::get_metal_backend;

                    // Download logits from GPU to CPU
                    let backend = get_metal_backend()?;
                    let data = backend.download_buffer_to_vec(&metal_data.buffer_id())?;

                    // Shape should be [batch, seq_len, vocab_size]
                    if metal_data.shape.len() != 3 {
                        return Err(tensor_op_error(
                            "tensor_operation",
                            format!("Expected 3D logits, got shape: {:?}", metal_data.shape),
                        ));
                    }

                    let seq_len = metal_data.shape[1];
                    let vocab_size = metal_data.shape[2];

                    // Get logits for last token: offset = (batch=0, seq_len-1, vocab=0)
                    let offset = (seq_len - 1) * vocab_size;
                    let last_logits = &data[offset..offset + vocab_size];

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

            // Check for EOS token (GPT-2 default, should use config.eos_token_id)
            if next_token == 50256 || next_token == self.transformer.config.eos_token_id {
                break;
            }
        }

        Ok(generated)
    }

    /// Generate text using greedy decoding with KV-cache for speed
    pub fn generate_greedy_with_cache(
        &self,
        input_ids: Vec<u32>,
        max_length: usize,
    ) -> Result<Vec<u32>> {
        let mut generated = input_ids.clone();
        let mut cache = KVCache::new(self.transformer.config.n_layer);
        let mut is_first_iteration = true;

        while generated.len() < max_length {
            // Prepare input - only process new token after first iteration
            let input_batch = if is_first_iteration {
                // First iteration: process full prompt
                vec![generated.clone()]
            } else {
                let last_token = *generated.last().ok_or_else(|| {
                    tensor_op_error("generation", "Generated sequence is empty".to_string())
                })?;
                // Subsequent iterations: process only last generated token
                vec![vec![last_token]]
            };

            // Forward pass with cache
            let hidden_states =
                self.transformer.forward_internal(&input_batch, None, Some(&mut cache))?;

            // Apply LM head
            let logits = self.lm_head.forward(hidden_states)?;

            // Debug: Check which match arm will be taken
            match &logits {
                Tensor::F32(_) => {},
                #[cfg(all(target_os = "macos", feature = "metal"))]
                Tensor::Metal(_) => {},
                _ => {},
            }

            is_first_iteration = false;

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
                #[cfg(all(target_os = "macos", feature = "metal"))]
                Tensor::Metal(metal_data) => {
                    use trustformers_core::gpu_ops::metal::get_metal_backend;

                    // Download logits from GPU to CPU
                    let backend = get_metal_backend()?;
                    let data = backend.download_buffer_to_vec(&metal_data.buffer_id())?;

                    // Shape should be [batch, seq_len, vocab_size]
                    if metal_data.shape.len() != 3 {
                        return Err(tensor_op_error(
                            "tensor_operation",
                            format!("Expected 3D logits, got shape: {:?}", metal_data.shape),
                        ));
                    }

                    let _batch_size = metal_data.shape[0];
                    let seq_len = metal_data.shape[1];
                    let vocab_size = metal_data.shape[2];

                    // Get logits for last token: offset = (batch=0, seq_len-1, vocab=0)
                    let offset = (seq_len - 1) * vocab_size;
                    let last_logits = &data[offset..offset + vocab_size];

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

            // Check for EOS token (GPT-2 default, should use config.eos_token_id)
            if next_token == 50256 || next_token == self.transformer.config.eos_token_id {
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
                            .map_err(|e| {
                                tensor_op_error(
                                    "from_shape_vec",
                                    format!("Failed to create array from shape: {}", e),
                                )
                            })?
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
                token_scores
                    .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

                // Add top candidates
                for (log_prob, token_idx) in token_scores.iter().take(num_beams) {
                    let new_score = score + log_prob;
                    let mut new_sequence = sequence.clone();
                    new_sequence.push(*token_idx as u32);
                    candidates.push((new_score, new_sequence));
                }
            }

            // Select top beams for next iteration
            candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
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

    /// Load pretrained weights for the backbone and the language-modelling head.
    ///
    /// When the checkpoint has no `lm_head.weight` the head is tied to the token
    /// embedding table, as GPT-2 itself does, instead of being left randomly
    /// initialised.
    ///
    /// # Errors
    ///
    /// Fails when the container cannot be parsed, when a backbone tensor is
    /// missing, or when neither an LM head nor a token embedding table is present.
    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        let mut checkpoint_reader = CheckpointReader::from_reader(reader)?;
        self.transformer.load_weights_from_reader(&mut checkpoint_reader)?;

        let checkpoint = checkpoint_reader.checkpoint();
        let head = ["lm_head.weight", "transformer.wte.weight", "wte.weight"]
            .iter()
            .find_map(|name| checkpoint.get(name))
            .ok_or_else(|| {
                TrustformersError::weight_load_error(
                    "checkpoint has no lm_head.weight and no token embedding table to tie it to"
                        .to_string(),
                )
            })?
            .clone();
        self.lm_head.set_weight(head)
    }

    fn get_config(&self) -> &Self::Config {
        self.transformer.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.transformer.num_parameters() + self.lm_head.parameter_count()
    }

    /// Enumerate the backbone under `transformer.…` plus the LM head.
    ///
    /// This matches the checkpoints HuggingFace ships for `GPT2LMHeadModel`,
    /// which is also the prefix [`Gpt2Model::load_weights_from_reader`] detects.
    ///
    /// # Tied weights
    ///
    /// GPT-2 ties `lm_head.weight` to `transformer.wte.weight`, and
    /// [`Gpt2LMHeadModel::load_pretrained`] reproduces that by *copying* the
    /// embedding table into the head when the checkpoint has no `lm_head.weight`.
    /// The two are therefore distinct tensors here and both are listed: the
    /// trait forbids duplicate names, and skipping the head would silently drop
    /// a parameter that this implementation really does own separately.
    fn named_tensors(&self) -> Vec<(String, &Tensor)> {
        let mut tensors = Vec::new();
        self.transformer.collect_named_parameters("transformer", &mut tensors);
        self.lm_head.collect_named_parameters("lm_head", &mut tensors);
        tensors
    }

    fn named_tensors_mut(&mut self) -> Vec<(String, &mut Tensor)> {
        let mut tensors = Vec::new();
        self.transformer.collect_named_parameters_mut("transformer", &mut tensors);
        self.lm_head.collect_named_parameters_mut("lm_head", &mut tensors);
        tensors
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

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;

    fn small_gpt2_config() -> Gpt2Config {
        Gpt2Config {
            vocab_size: 100,
            n_positions: 64,
            n_embd: 32,
            n_layer: 2,
            n_head: 4,
            n_inner: None,
            activation_function: "gelu".to_string(),
            resid_pdrop: 0.0,
            embd_pdrop: 0.0,
            attn_pdrop: 0.0,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            bos_token_id: 50256,
            eos_token_id: 50256,
            model_type: "gpt2".to_string(),
        }
    }

    #[test]
    fn test_gpt2_config_default() {
        let config = Gpt2Config::default();
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.n_positions, 1024);
        assert_eq!(config.n_embd, 768);
        assert_eq!(config.n_layer, 12);
        assert_eq!(config.n_head, 12);
    }

    #[test]
    fn test_gpt2_config_validate() {
        let config = small_gpt2_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_gpt2_model_creation() {
        let config = small_gpt2_config();
        let result = Gpt2Model::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gpt2_model_with_device() {
        let config = small_gpt2_config();
        let result = Gpt2Model::new_with_device(config, Device::CPU);
        assert!(result.is_ok());
        let model = result.expect("model creation should succeed");
        assert!(matches!(model.device(), Device::CPU));
    }

    #[test]
    fn test_gpt2_model_config() {
        let config = small_gpt2_config();
        let model = Gpt2Model::new(config.clone()).expect("model creation should succeed");
        let model_config = model.get_config();
        assert_eq!(model_config.vocab_size, config.vocab_size);
        assert_eq!(model_config.n_embd, config.n_embd);
    }

    #[test]
    fn test_gpt2_model_num_parameters() {
        let config = small_gpt2_config();
        let model = Gpt2Model::new(config).expect("model creation should succeed");
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_gpt2_lm_head_creation() {
        let config = small_gpt2_config();
        let result = Gpt2LMHeadModel::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gpt2_lm_head_num_parameters() {
        let config = small_gpt2_config();
        let model = Gpt2LMHeadModel::new(config).expect("model creation should succeed");
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_layer_cache_new() {
        let cache = LayerCache::new();
        assert!(cache.key.is_none());
        assert!(cache.value.is_none());
    }

    #[test]
    fn test_layer_cache_default() {
        let cache = LayerCache::default();
        assert!(cache.key.is_none());
        assert!(cache.value.is_none());
    }

    #[test]
    fn test_kv_cache_new() {
        let cache = KVCache::new(4);
        assert_eq!(cache.layers.len(), 4);
        for layer in &cache.layers {
            assert!(layer.key.is_none());
            assert!(layer.value.is_none());
        }
    }

    #[test]
    fn test_kv_cache_zero_layers() {
        let cache = KVCache::new(0);
        assert!(cache.layers.is_empty());
    }

    #[test]
    fn test_transpose_tensor_2d() {
        let arr = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("array creation should succeed");
        let tensor = Tensor::F32(arr);
        let result = transpose_tensor(tensor);
        assert!(result.is_ok());
        let transposed = result.expect("transpose should succeed");
        assert_eq!(transposed.shape(), vec![3, 2]);
    }

    #[test]
    fn test_transpose_tensor_non_2d_fails() {
        let arr = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), vec![0.0f32; 24])
            .expect("array creation should succeed");
        let tensor = Tensor::F32(arr);
        let result = transpose_tensor(tensor);
        assert!(result.is_err());
    }

    #[test]
    fn test_gpt2_model_to_device() {
        let config = small_gpt2_config();
        let model = Gpt2Model::new(config).expect("model creation should succeed");
        let model_cpu = model.to_device(Device::CPU);
        assert!(matches!(model_cpu.device(), Device::CPU));
    }

    #[test]
    fn test_gpt2_config_n_inner_override() {
        let mut config = small_gpt2_config();
        config.n_inner = Some(128);
        let result = Gpt2Model::new(config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gpt2_config_activation_functions() {
        for activation in &["gelu", "relu", "silu"] {
            let mut config = small_gpt2_config();
            config.activation_function = activation.to_string();
            let result = Gpt2Model::new(config);
            assert!(result.is_ok(), "Failed with activation: {}", activation);
        }
    }

    #[test]
    fn test_gpt2_lm_head_with_device() {
        let config = small_gpt2_config();
        let result = Gpt2LMHeadModel::new_with_device(config, Device::CPU);
        assert!(result.is_ok());
    }

    #[test]
    fn test_kv_cache_clone() {
        let cache = KVCache::new(3);
        let cloned = cache.clone();
        assert_eq!(cloned.layers.len(), 3);
    }

    #[test]
    fn test_layer_cache_clone() {
        let cache = LayerCache::new();
        let cloned = cache.clone();
        assert!(cloned.key.is_none());
    }

    #[test]
    fn test_gpt2_model_forward() {
        let config = small_gpt2_config();
        let model = Gpt2Model::new(config).expect("model creation should succeed");
        let input = TokenizedInput::new(vec![1, 2, 3], vec![1, 1, 1]);
        let result = model.forward(input);
        assert!(result.is_ok());
    }
}
