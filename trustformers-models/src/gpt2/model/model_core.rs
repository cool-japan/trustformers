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

use super::model_blocks::{
    apply_top_k_filtering, apply_top_p_filtering, create_causal_mask, log_softmax,
    sample_from_logits, stack_tensors, Gpt2Block,
};
use crate::gpt2::config::Gpt2Config;

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
        println!(
            "✓ Gpt2Model: All layer weights cached on CUDA GPU ({} blocks)",
            self.h.len()
        );
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
            eprintln!("🔍 Cache exists: {} layers", cache.layers.len());
            // If cache exists and has keys, start from past sequence length
            if let Some(first_layer_cache) = cache.layers.first() {
                eprintln!(
                    "🔍 First layer cache - key type: {:?}",
                    first_layer_cache.key.as_ref().map(std::mem::discriminant)
                );
                match &first_layer_cache.key {
                    Some(Tensor::F32(ref past_k)) => {
                        eprintln!("🔍 F32 key shape: {:?}", past_k.shape());
                        past_k.shape()[1] as u32 // past_seq_len
                    },
                    #[cfg(all(target_os = "macos", feature = "metal"))]
                    Some(Tensor::Metal(ref metal_data)) => {
                        eprintln!("🔍 Metal key shape: {:?}", metal_data.shape);
                        metal_data.shape[1] as u32 // past_seq_len from Metal tensor
                    },
                    None => {
                        eprintln!("🔍 Key is None!");
                        0
                    },
                    _ => {
                        eprintln!("🔍 Key is unknown type!");
                        0
                    },
                }
            } else {
                eprintln!("🔍 No first layer cache!");
                0
            }
        } else {
            eprintln!("🔍 No cache!");
            0
        };

        eprintln!("🔍 Position offset: {} (from cache)", position_offset);

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

            eprintln!("🔍 Position IDs for batch {}: {:?}", batch_idx, pos_ids);

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
                eprintln!(
                    "🔄 Converting hidden_states from {:?} to Metal device",
                    std::mem::discriminant(&hidden_states)
                );

                // Debug: Check values before GPU upload
                if let Tensor::F32(ref arr) = hidden_states {
                    let data: Vec<f32> = arr.iter().cloned().collect();
                    eprintln!(
                        "🔍 Embedding output (CPU) first 10: {:?}",
                        &data[..10.min(data.len())]
                    );
                    eprintln!(
                        "🔍 Embedding stats: min={:.4}, max={:.4}, mean={:.4}",
                        data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                        data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                        data.iter().sum::<f32>() / data.len() as f32
                    );
                }

                hidden_states = hidden_states.to_device_enum(&self.device)?;
                eprintln!(
                    "✅ hidden_states converted to: {:?}",
                    std::mem::discriminant(&hidden_states)
                );

                // Debug: Check values after GPU upload
                if let Tensor::Metal(ref metal_data) = hidden_states {
                    use trustformers_core::gpu_ops::metal::get_metal_backend;
                    let backend = get_metal_backend()?;
                    eprintln!(
                        "🔍 After GPU upload: buffer_id={:?}, shape={:?}",
                        metal_data.buffer_id, metal_data.shape
                    );
                    let gpu_data = backend.download_buffer_to_vec(&metal_data.buffer_id)?;
                    eprintln!(
                        "🔍 After GPU upload: Downloaded {} f32 values",
                        gpu_data.len()
                    );
                    eprintln!(
                        "🔍 After GPU upload first 10: {:?}",
                        &gpu_data[..10.min(gpu_data.len())]
                    );
                    eprintln!(
                        "🔍 After GPU upload stats: min={:.4}, max={:.4}, mean={:.4}",
                        gpu_data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                        gpu_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                        gpu_data.iter().sum::<f32>() / gpu_data.len() as f32
                    );
                }
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
        println!("✓ Gpt2LMHeadModel: All weights uploaded to CUDA GPU");
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
                    let vocab_size = last_logits.len();

                    // Debug: Show logits statistics for first few iterations
                    if generated.len() <= 8 {
                        eprintln!("\n🔍 CPU Logits Debug (iteration {}):", generated.len());
                        eprintln!("   Shape: {:?}", shape);

                        let logits_vec: Vec<f32> = last_logits.iter().copied().collect();
                        eprintln!(
                            "   Last token logits - first 10: {:?}",
                            &logits_vec[..10.min(vocab_size)]
                        );
                        eprintln!(
                            "   Last token logits - last 10: {:?}",
                            &logits_vec[vocab_size.saturating_sub(10)..]
                        );

                        // Find top 5 predictions
                        let mut top_indices: Vec<usize> = (0..vocab_size).collect();
                        top_indices.sort_by(|&a, &b| {
                            logits_vec[b]
                                .partial_cmp(&logits_vec[a])
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        eprintln!("   Top 5 predictions:");
                        for &idx in &top_indices[..5.min(vocab_size)] {
                            eprintln!("      token {} → logit {:.4}", idx, logits_vec[idx]);
                        }

                        // Statistics
                        let min = logits_vec.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                        let max = logits_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        let mean = logits_vec.iter().sum::<f32>() / vocab_size as f32;
                        eprintln!("   Stats: min={:.4}, max={:.4}, mean={:.4}", min, max, mean);
                    }

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
                    let data = backend.download_buffer_to_vec(&metal_data.buffer_id)?;

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

                    // Debug: Show logits statistics for first few iterations
                    if generated.len() <= 8 {
                        eprintln!("\n🔍 GPU Logits Debug (iteration {}):", generated.len());
                        eprintln!("   Shape: {:?}", metal_data.shape);
                        eprintln!(
                            "   Last token logits - first 10: {:?}",
                            &last_logits[..10.min(vocab_size)]
                        );
                        eprintln!(
                            "   Last token logits - last 10: {:?}",
                            &last_logits[vocab_size.saturating_sub(10)..]
                        );

                        // Find top 5 predictions
                        let mut top_indices: Vec<usize> = (0..vocab_size).collect();
                        top_indices.sort_by(|&a, &b| {
                            last_logits[b]
                                .partial_cmp(&last_logits[a])
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        eprintln!("   Top 5 predictions:");
                        for &idx in &top_indices[..5.min(vocab_size)] {
                            eprintln!("      token {} → logit {:.4}", idx, last_logits[idx]);
                        }

                        // Statistics
                        let min = last_logits.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                        let max = last_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                        let mean = last_logits.iter().sum::<f32>() / vocab_size as f32;
                        eprintln!("   Stats: min={:.4}, max={:.4}, mean={:.4}", min, max, mean);
                    }

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

            eprintln!(
                "🎲 Generated token: {} (total: {})",
                next_token,
                generated.len() + 1
            );
            generated.push(next_token);

            // Check for EOS token (GPT-2 default, should use config.eos_token_id)
            if next_token == 50256 || next_token == self.transformer.config.eos_token_id {
                eprintln!("🛑 EOS token detected, stopping generation");
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

        eprintln!(
            "🔄 Starting generation: input_len={}, max_length={}, will generate {} tokens",
            generated.len(),
            max_length,
            max_length - generated.len()
        );

        while generated.len() < max_length {
            eprintln!(
                "\n━━━ Loop iteration: current_len={}, target={} ━━━",
                generated.len(),
                max_length
            );

            // Prepare input - only process new token after first iteration
            let input_batch = if is_first_iteration {
                eprintln!(
                    "📥 First iteration: processing full prompt ({} tokens)",
                    generated.len()
                );
                // First iteration: process full prompt
                vec![generated.clone()]
            } else {
                let last_token = *generated.last().ok_or_else(|| {
                    tensor_op_error("generation", "Generated sequence is empty".to_string())
                })?;
                eprintln!(
                    "📤 Subsequent iteration: processing last token [{}]",
                    last_token
                );
                // Subsequent iterations: process only last generated token
                vec![vec![last_token]]
            };

            // Forward pass with cache
            let hidden_states =
                self.transformer.forward_internal(&input_batch, None, Some(&mut cache))?;

            // Apply LM head
            eprintln!(
                "🔍 Hidden states before lm_head: shape={:?}, type={:?}",
                match &hidden_states {
                    Tensor::F32(arr) => format!("{:?}", arr.shape()),
                    #[cfg(all(target_os = "macos", feature = "metal"))]
                    Tensor::Metal(m) => format!("{:?}", m.shape),
                    _ => "unknown".to_string(),
                },
                std::mem::discriminant(&hidden_states)
            );

            // Debug: Download and check hidden state values
            #[cfg(all(target_os = "macos", feature = "metal"))]
            if let Tensor::Metal(ref metal_data) = hidden_states {
                use trustformers_core::gpu_ops::metal::get_metal_backend;
                let backend = get_metal_backend()?;
                let hidden_data = backend.download_buffer_to_vec(&metal_data.buffer_id)?;
                eprintln!(
                    "🔍 Hidden states first 10 values: {:?}",
                    &hidden_data[..10.min(hidden_data.len())]
                );
                eprintln!(
                    "🔍 Hidden states stats: min={:.4}, max={:.4}, mean={:.4}",
                    hidden_data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                    hidden_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                    hidden_data.iter().sum::<f32>() / hidden_data.len() as f32
                );
            }

            eprintln!("🔍 About to call lm_head.forward...");
            let logits = self.lm_head.forward(hidden_states)?;
            eprintln!("🔍 lm_head.forward returned successfully!");
            eprintln!(
                "🔍 Logits after lm_head: shape={:?}, type={:?}",
                match &logits {
                    Tensor::F32(arr) => format!("{:?}", arr.shape()),
                    #[cfg(all(target_os = "macos", feature = "metal"))]
                    Tensor::Metal(m) => format!("{:?}", m.shape),
                    _ => "unknown".to_string(),
                },
                std::mem::discriminant(&logits)
            );

            // Debug: Check which match arm will be taken
            match &logits {
                Tensor::F32(_) => eprintln!("🔍 Logits match: Tensor::F32"),
                #[cfg(all(target_os = "macos", feature = "metal"))]
                Tensor::Metal(_) => eprintln!("🔍 Logits match: Tensor::Metal"),
                _ => eprintln!("❌ Logits match: WILDCARD (unsupported!)"),
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
                    eprintln!("🔍 Argmax (F32): idx={}, val={:.4}", max_idx, max_val);
                    max_idx as u32
                },
                #[cfg(all(target_os = "macos", feature = "metal"))]
                Tensor::Metal(metal_data) => {
                    use trustformers_core::gpu_ops::metal::get_metal_backend;

                    // Download logits from GPU to CPU
                    let backend = get_metal_backend()?;
                    let data = backend.download_buffer_to_vec(&metal_data.buffer_id)?;

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

                    // Debug: Print first 10 logits values
                    eprintln!(
                        "🔍 First 10 logits: {:?}",
                        &last_logits[..10.min(last_logits.len())]
                    );
                    eprintln!(
                        "🔍 Logits stats: min={:.4}, max={:.4}, mean={:.4}",
                        last_logits.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                        last_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)),
                        last_logits.iter().sum::<f32>() / last_logits.len() as f32
                    );

                    // Find argmax
                    let mut max_idx = 0;
                    let mut max_val = f32::NEG_INFINITY;
                    for (idx, &val) in last_logits.iter().enumerate() {
                        if val > max_val {
                            max_val = val;
                            max_idx = idx;
                        }
                    }
                    eprintln!("🔍 Argmax: idx={}, val={:.4}", max_idx, max_val);
                    max_idx as u32
                },
                _ => {
                    return Err(tensor_op_error(
                        "tensor_operation",
                        "Unsupported tensor type".to_string(),
                    ))
                },
            };

            eprintln!(
                "🎲 Generated token: {} (total: {})",
                next_token,
                generated.len() + 1
            );
            generated.push(next_token);

            // Check for EOS token (GPT-2 default, should use config.eos_token_id)
            if next_token == 50256 || next_token == self.transformer.config.eos_token_id {
                eprintln!("🛑 EOS token detected, stopping generation");
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
