//! GPT-2 building blocks and helper functions
//!
//! Contains Gpt2Block, Gpt2Attention, Gpt2MLP, ActivationType,
//! and utility functions (causal mask, sampling, softmax).

use scirs2_core::ndarray::{s, ArrayD, Axis, IxDyn};
use trustformers_core::{
    device::Device,
    errors::{invalid_config, tensor_op_error, Result, TrustformersError},
    layers::{LayerNorm, Linear},
    ops::activations::{gelu as gelu_core, relu, silu},
    tensor::Tensor,
    traits::{Layer, WeightReader},
};

use super::model_core::{transpose_tensor, LayerCache};
use crate::gpt2::config::Gpt2Config;

#[derive(Clone)]
pub(crate) struct Gpt2Block {
    ln_1: LayerNorm,
    attn: Gpt2Attention,
    ln_2: LayerNorm,
    mlp: Gpt2MLP,
}

impl Gpt2Block {
    #[allow(dead_code)]
    pub(crate) fn new(config: &Gpt2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub(crate) fn new_with_device(config: &Gpt2Config, device: Device) -> Result<Self> {
        Ok(Self {
            ln_1: LayerNorm::new_simple(config.n_embd, config.layer_norm_epsilon),
            attn: Gpt2Attention::new_with_device(config, device)?,
            ln_2: LayerNorm::new_simple(config.n_embd, config.layer_norm_epsilon),
            mlp: Gpt2MLP::new_with_device(config, device)?,
        })
    }

    pub(crate) fn to_device(mut self, device: Device) -> Self {
        self.attn = self.attn.to_device(device);
        self.mlp = self.mlp.to_device(device);
        self
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn weights_to_gpu(&mut self, device: &Device) -> Result<()> {
        if !matches!(device, Device::Metal(_)) {
            return Ok(());
        }
        self.ln_1.weights_to_gpu(device)?;
        self.attn.weights_to_gpu(device)?;
        self.ln_2.weights_to_gpu(device)?;
        self.mlp.weights_to_gpu(device)?;
        Ok(())
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    pub(crate) fn weights_to_gpu_cuda(&mut self, device: &Device) -> Result<()> {
        if !matches!(device, Device::CUDA(_)) {
            return Ok(());
        }
        self.ln_1.weights_to_gpu_cuda(device)?;
        self.attn.weights_to_gpu_cuda(device)?;
        self.ln_2.weights_to_gpu_cuda(device)?;
        self.mlp.weights_to_gpu_cuda(device)?;
        Ok(())
    }

    pub(crate) fn load_weights(
        &mut self,
        reader: &mut dyn WeightReader,
        prefix: &str,
    ) -> Result<()> {
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

    pub(crate) fn load_weights_from_loader(
        &mut self,
        loader: &mut dyn crate::weight_loading::WeightLoader,
        prefix: &str,
    ) -> Result<()> {
        // Load layer norm weights
        self.ln_1.set_weight(loader.load_tensor(&format!("{}.ln_1.weight", prefix))?)?;
        self.ln_1.set_bias(loader.load_tensor(&format!("{}.ln_1.bias", prefix))?)?;

        self.ln_2.set_weight(loader.load_tensor(&format!("{}.ln_2.weight", prefix))?)?;
        self.ln_2.set_bias(loader.load_tensor(&format!("{}.ln_2.bias", prefix))?)?;

        // Load attention weights
        self.attn.load_weights_from_loader(loader, &format!("{}.attn", prefix))?;

        // Load MLP weights
        self.mlp.load_weights_from_loader(loader, &format!("{}.mlp", prefix))?;

        Ok(())
    }

    pub(crate) fn parameter_count(&self) -> usize {
        self.ln_1.parameter_count()
            + self.attn.parameter_count()
            + self.ln_2.parameter_count()
            + self.mlp.parameter_count()
    }

    #[allow(dead_code)]
    pub(crate) fn forward(
        &self,
        hidden_states: Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_with_cache(hidden_states, attention_mask, None)
    }

    pub(crate) fn forward_with_cache(
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
pub(crate) struct Gpt2Attention {
    n_head: usize,
    d_head: usize,
    c_attn: Linear, // Combined QKV projection
    c_proj: Linear, // Output projection
    #[allow(dead_code)]
    attn_dropout: f32,
    resid_dropout: f32,
}

impl Gpt2Attention {
    #[allow(dead_code)]
    fn new(config: &Gpt2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    fn new_with_device(config: &Gpt2Config, device: Device) -> Result<Self> {
        if !config.n_embd.is_multiple_of(config.n_head) {
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

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn weights_to_gpu(&mut self, device: &Device) -> Result<()> {
        if !matches!(device, Device::Metal(_)) {
            return Ok(());
        }
        self.c_attn.weights_to_gpu(device)?;
        self.c_proj.weights_to_gpu(device)?;
        Ok(())
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    fn weights_to_gpu_cuda(&mut self, device: &Device) -> Result<()> {
        if !matches!(device, Device::CUDA(_)) {
            return Ok(());
        }
        self.c_attn.weights_to_gpu_cuda(device)?;
        self.c_proj.weights_to_gpu_cuda(device)?;
        Ok(())
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

    fn load_weights_from_loader(
        &mut self,
        loader: &mut dyn crate::weight_loading::WeightLoader,
        prefix: &str,
    ) -> Result<()> {
        // Load combined QKV weights
        // PyTorch stores as [out, in], we need [in, out], so transpose
        let c_attn_weight = loader.load_tensor(&format!("{}.c_attn.weight", prefix))?;
        self.c_attn.set_weight(transpose_tensor(c_attn_weight)?)?;
        self.c_attn.set_bias(loader.load_tensor(&format!("{}.c_attn.bias", prefix))?)?;

        // Load output projection weights (also needs transpose)
        let c_proj_weight = loader.load_tensor(&format!("{}.c_proj.weight", prefix))?;
        self.c_proj.set_weight(transpose_tensor(c_proj_weight)?)?;
        self.c_proj.set_bias(loader.load_tensor(&format!("{}.c_proj.bias", prefix))?)?;

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

        // GPU attention path with GPU-aware KV-cache (ZERO CPU transfers!)
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Tensor::Metal(qkv_data) = &qkv {
            use trustformers_core::gpu_ops::metal::get_metal_backend;
            use trustformers_core::tensor::MetalTensorData;

            let backend = get_metal_backend()?;

            // Split QKV on GPU: [batch, seq, 3*hidden] → 3x [batch, seq, hidden]
            let (q_id, k_new_id, v_new_id) =
                backend.split_qkv_gpu(&qkv_data.buffer_id, batch_size, seq_len, hidden_size)?;

            // Get cached K/V buffer IDs and sequence length (if cache exists)
            let (cached_k_id, cached_v_id, cached_seq_len) = if let Some(cache) = &layer_cache {
                match (&cache.key, &cache.value) {
                    (Some(Tensor::Metal(k_metal)), Some(Tensor::Metal(v_metal))) => {
                        let cached_shape = &k_metal.shape; // [batch, num_heads, cached_seq, head_dim]
                        let cached_seq = cached_shape[2];
                        #[cfg(debug_assertions)]
                        eprintln!("🔗 GPU cache found: cached_seq={}", cached_seq);
                        (
                            Some(&k_metal.buffer_id),
                            Some(&v_metal.buffer_id),
                            cached_seq,
                        )
                    },
                    _ => {
                        // eprintln!("🚀 GPU attention (first token, no cache)");
                        (None, None, 0)
                    },
                }
            } else {
                // eprintln!("🚀 GPU attention (no cache layer)");
                (None, None, 0)
            };

            // Reshape Q, K_new, V_new to multi-head format
            // [batch, seq, hidden] → [batch, num_heads, seq, head_dim]
            let q_heads_id =
                backend.reshape_to_heads_gpu(&q_id, seq_len, self.n_head, self.d_head)?;
            let k_new_heads_id =
                backend.reshape_to_heads_gpu(&k_new_id, seq_len, self.n_head, self.d_head)?;
            let v_new_heads_id =
                backend.reshape_to_heads_gpu(&v_new_id, seq_len, self.n_head, self.d_head)?;

            // Concatenate with cached K/V on GPU (stays on GPU!)
            let k_heads_id = backend.concat_kv_cache(
                cached_k_id,
                &k_new_heads_id,
                batch_size,
                self.n_head,
                cached_seq_len,
                seq_len, // new_seq_len
                self.d_head,
            )?;

            let v_heads_id = backend.concat_kv_cache(
                cached_v_id,
                &v_new_heads_id,
                batch_size,
                self.n_head,
                cached_seq_len,
                seq_len,
                self.d_head,
            )?;

            let total_seq_len = cached_seq_len + seq_len;

            // Execute GPU attention with cached K/V
            // Q: [batch, num_heads, seq_len, head_dim] (current tokens)
            // K: [batch, num_heads, total_seq_len, head_dim] (cached + new)
            // V: [batch, num_heads, total_seq_len, head_dim] (cached + new)
            let attn_heads_output_id = backend.attention_with_cache_gpu_to_gpu(
                &q_heads_id,
                &k_heads_id,
                &v_heads_id,
                batch_size,
                seq_len,       // q_seq_len
                total_seq_len, // kv_seq_len
                self.n_head,
                self.d_head,
            )?;

            // Reshape from [batch, num_heads, seq_len, head_dim] back to [batch, seq_len, hidden_size]
            let attn_output_id = backend.reshape_from_heads_gpu(
                &attn_heads_output_id,
                seq_len,
                self.n_head,
                self.d_head,
            )?;

            // Update cache with full K/V (keep on GPU!)
            if let Some(cache) = layer_cache {
                cache.key = Some(Tensor::Metal(MetalTensorData {
                    buffer_id: k_heads_id,
                    shape: vec![batch_size, self.n_head, total_seq_len, self.d_head],
                    dtype: qkv_data.dtype,
                }));
                cache.value = Some(Tensor::Metal(MetalTensorData {
                    buffer_id: v_heads_id,
                    shape: vec![batch_size, self.n_head, total_seq_len, self.d_head],
                    dtype: qkv_data.dtype,
                }));
                #[cfg(debug_assertions)]
                eprintln!("✅ GPU cache updated: total_seq={}", total_seq_len);
            }

            // Wrap in Metal tensor and apply output projection
            let attn_output = Tensor::Metal(MetalTensorData {
                buffer_id: attn_output_id,
                shape: vec![batch_size, seq_len, hidden_size],
                dtype: qkv_data.dtype,
            });

            // Apply output projection (stays on GPU)
            let output = self.c_proj.forward(attn_output)?;

            // Remove batch dimension if it was added
            return if was_2d {
                match output {
                    Tensor::Metal(metal_data) if metal_data.shape[0] == 1 => {
                        // Reshape [1, seq, hidden] → [seq, hidden]
                        let new_shape = vec![metal_data.shape[1], metal_data.shape[2]];
                        Ok(Tensor::Metal(MetalTensorData {
                            buffer_id: metal_data.buffer_id,
                            shape: new_shape,
                            dtype: metal_data.dtype,
                        }))
                    },
                    _ => Ok(output),
                }
            } else {
                Ok(output)
            };
        }

        // Fallback: CPU attention path (with cache support)
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let qkv = match &qkv {
            Tensor::Metal(qkv_data) => {
                use trustformers_core::gpu_ops::metal::get_metal_backend;

                eprintln!("⚠️  Attention: CPU path (has cache), downloading Q/K/V");

                let backend = get_metal_backend()?;

                // Split QKV on GPU then download
                let (q_id, k_id, v_id) =
                    backend.split_qkv_gpu(&qkv_data.buffer_id, batch_size, seq_len, hidden_size)?;

                let q_data = backend.download_buffer_to_vec(&q_id)?;
                let k_data = backend.download_buffer_to_vec(&k_id)?;
                let v_data = backend.download_buffer_to_vec(&v_id)?;

                // Reconstruct QKV array for CPU processing
                use scirs2_core::ndarray::ArrayD;
                let mut qkv_vec = Vec::with_capacity(batch_size * seq_len * 3 * hidden_size);
                for i in 0..(batch_size * seq_len) {
                    let offset = i * hidden_size;
                    qkv_vec.extend_from_slice(&q_data[offset..offset + hidden_size]);
                    qkv_vec.extend_from_slice(&k_data[offset..offset + hidden_size]);
                    qkv_vec.extend_from_slice(&v_data[offset..offset + hidden_size]);
                }

                let qkv_arr = ArrayD::from_shape_vec(
                    scirs2_core::ndarray::IxDyn(&[batch_size, seq_len, 3 * hidden_size]),
                    qkv_vec,
                )
                .map_err(|e| {
                    TrustformersError::tensor_op_error(
                        &format!("Failed to create QKV array: {}", e),
                        "forward_with_cache",
                    )
                })?;

                Tensor::F32(qkv_arr)
            },
            _ => qkv,
        };

        #[cfg(not(feature = "metal"))]
        let qkv = qkv;

        // Split QKV into separate Q, K, V tensors
        match &qkv {
            Tensor::F32(arr) => {
                // qkv shape: [batch, seq_len, 3 * hidden_size]
                // Split into 3 equal parts
                let _qkv_shape = arr.shape();
                let chunk_size = hidden_size;

                // Extract Q, K, V
                let q = arr.slice(s![.., .., ..chunk_size]).to_owned();
                let k_new_slice = arr.slice(s![.., .., chunk_size..2 * chunk_size]);
                let v_new_slice = arr.slice(s![.., .., 2 * chunk_size..]);

                // Convert to ArrayD for uniform handling
                let k_new = k_new_slice.to_owned().into_dyn();
                let v_new = v_new_slice.to_owned().into_dyn();

                // Concatenate with past K/V if cache exists
                let mut k = k_new.clone();
                let mut v = v_new.clone();

                if let Some(cache) = &layer_cache {
                    if let (Some(Tensor::F32(past_k)), Some(Tensor::F32(past_v))) =
                        (&cache.key, &cache.value)
                    {
                        // Concatenate: [past_seq, hidden] + [1, hidden] → [past_seq+1, hidden]
                        let past_seq = past_k.shape()[1];
                        let new_seq = k_new.shape()[1];
                        let total_seq = past_seq + new_seq;

                        let mut k_concat =
                            ArrayD::zeros(IxDyn(&[batch_size, total_seq, hidden_size]));
                        let mut v_concat =
                            ArrayD::zeros(IxDyn(&[batch_size, total_seq, hidden_size]));

                        // Copy past
                        k_concat.slice_mut(s![.., 0..past_seq, ..]).assign(past_k);
                        v_concat.slice_mut(s![.., 0..past_seq, ..]).assign(past_v);

                        // Append new
                        k_concat.slice_mut(s![.., past_seq..total_seq, ..]).assign(&k_new);
                        v_concat.slice_mut(s![.., past_seq..total_seq, ..]).assign(&v_new);

                        k = k_concat;
                        v = v_concat;
                    }
                }

                // Save for cache (before reshape) - already ArrayD
                let k_for_cache = k.clone();
                let v_for_cache = v.clone();

                // Reshape for multi-head attention
                // From [batch, seq_len, hidden_size] to [batch, seq_len, n_heads, head_dim]
                let head_dim = self.d_head;
                let n_heads = self.n_head;

                // Get actual sequence lengths (Q is current, K/V may be concatenated)
                let q_seq_len = seq_len;
                let kv_seq_len = k.shape()[1];

                let q = q
                    .to_shape(IxDyn(&[batch_size, q_seq_len, n_heads, head_dim]))
                    .map_err(|_| TrustformersError::shape_error("Failed to reshape Q".into()))?
                    .to_owned();
                let k = k
                    .to_shape(IxDyn(&[batch_size, kv_seq_len, n_heads, head_dim]))
                    .map_err(|_| TrustformersError::shape_error("Failed to reshape K".into()))?
                    .to_owned();
                let v = v
                    .to_shape(IxDyn(&[batch_size, kv_seq_len, n_heads, head_dim]))
                    .map_err(|_| TrustformersError::shape_error("Failed to reshape V".into()))?
                    .to_owned();

                // Transpose to [batch, n_heads, seq_len, head_dim]
                let q = q.permuted_axes(vec![0, 2, 1, 3]);
                let k = k.permuted_axes(vec![0, 2, 1, 3]);
                let v = v.permuted_axes(vec![0, 2, 1, 3]);

                // Compute attention scores
                // Q * K^T / sqrt(head_dim)
                let scale = 1.0 / (head_dim as f32).sqrt();
                #[allow(unused_variables)]
                let k_t = k.clone().permuted_axes(vec![0, 1, 3, 2]); // Transpose last two dims

                // Compute Q * K^T
                // Q: [batch, n_heads, q_seq_len, head_dim]
                // K^T: [batch, n_heads, head_dim, kv_seq_len]
                // Result: [batch, n_heads, q_seq_len, kv_seq_len]
                let mut scores =
                    ArrayD::<f32>::zeros(IxDyn(&[batch_size, n_heads, q_seq_len, kv_seq_len]));

                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    use trustformers_core::gpu_ops::metal::get_metal_backend;
                    // Only use GPU for small models (overhead from transfers dominates for large models)
                    // Threshold: <= 12 heads is acceptable (GPT-2 124M)
                    // rinna-1b has 16 heads with too much transfer overhead
                    let use_gpu = get_metal_backend().is_ok() && n_heads <= 12;
                    if use_gpu {
                        if let Ok(backend) = get_metal_backend() {
                            for b in 0..batch_size {
                                for h in 0..n_heads {
                                    let q_head = q.slice(s![b, h, .., ..]);
                                    let k_head_t = k_t.slice(s![b, h, .., ..]);

                                    // Convert to contiguous arrays for GPU
                                    let q_data: Vec<f32> = q_head.iter().cloned().collect();
                                    let k_data: Vec<f32> = k_head_t.iter().cloned().collect();

                                    // GPU matmul: Q(q_seq_len × head_dim) * K^T(head_dim × kv_seq_len)
                                    let score_vec = backend.matmul_f32(
                                        &q_data, &k_data, q_seq_len, head_dim, kv_seq_len,
                                    )?;

                                    let score_array = ArrayD::from_shape_vec(
                                        IxDyn(&[q_seq_len, kv_seq_len]),
                                        score_vec,
                                    )
                                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;

                                    scores.slice_mut(s![b, h, .., ..]).assign(&score_array);
                                }
                            }
                        }
                    } else {
                        // CPU fallback - parallelize only for large models (>12 heads)
                        if n_heads > 12 {
                            use scirs2_core::parallel_ops::*;

                            let indices: Vec<(usize, usize)> = (0..batch_size)
                                .flat_map(|b| (0..n_heads).map(move |h| (b, h)))
                                .collect();

                            // Compute scores in parallel
                            let score_results: Vec<((usize, usize), ArrayD<f32>)> = indices
                                .par_iter()
                                .map(|&(b, h)| {
                                    let q_head = q.slice(s![b, h, .., ..]);
                                    let k_head_t = k_t.slice(s![b, h, .., ..]);
                                    let score = q_head.dot(&k_head_t);
                                    ((b, h), score.into_dyn())
                                })
                                .collect();

                            // Assign results sequentially
                            for ((b, h), score_arr) in score_results {
                                scores.slice_mut(s![b, h, .., ..]).assign(&score_arr);
                            }
                        } else {
                            // Sequential for small models
                            for b in 0..batch_size {
                                for h in 0..n_heads {
                                    let q_head = q.slice(s![b, h, .., ..]);
                                    let k_head_t = k_t.slice(s![b, h, .., ..]);
                                    let score = q_head.dot(&k_head_t);
                                    scores.slice_mut(s![b, h, .., ..]).assign(&score);
                                }
                            }
                        }
                    }
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    // CPU fallback - parallelize only for large models (>12 heads)
                    if n_heads > 12 {
                        use scirs2_core::parallel_ops::*;

                        let indices: Vec<(usize, usize)> = (0..batch_size)
                            .flat_map(|b| (0..n_heads).map(move |h| (b, h)))
                            .collect();

                        // Compute scores in parallel
                        let score_results: Vec<((usize, usize), ArrayD<f32>)> = indices
                            .par_iter()
                            .map(|&(b, h)| {
                                let q_head = q.slice(s![b, h, .., ..]);
                                let k_head_t = k_t.slice(s![b, h, .., ..]);
                                let score = q_head.dot(&k_head_t);
                                ((b, h), score.into_dyn())
                            })
                            .collect();

                        // Assign results sequentially
                        for ((b, h), score_arr) in score_results {
                            scores.slice_mut(s![b, h, .., ..]).assign(&score_arr);
                        }
                    } else {
                        // Sequential for small models
                        for b in 0..batch_size {
                            for h in 0..n_heads {
                                let q_head = q.slice(s![b, h, .., ..]);
                                let k_head_t = k_t.slice(s![b, h, .., ..]);
                                let score = q_head.dot(&k_head_t);
                                scores.slice_mut(s![b, h, .., ..]).assign(&score);
                            }
                        }
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

                // Softmax over kv_seq_len dimension
                let mut attention_probs = scores.clone();
                for b in 0..batch_size {
                    for h in 0..n_heads {
                        for i in 0..q_seq_len {
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
                // attention_probs: [batch, n_heads, q_seq_len, kv_seq_len]
                // V: [batch, n_heads, kv_seq_len, head_dim]
                // Result: [batch, n_heads, q_seq_len, head_dim]
                let mut output =
                    ArrayD::<f32>::zeros(IxDyn(&[batch_size, n_heads, q_seq_len, head_dim]));

                #[cfg(all(target_os = "macos", feature = "metal"))]
                {
                    use trustformers_core::gpu_ops::metal::get_metal_backend;
                    // Only use GPU for small models (same threshold as Q*K^T)
                    let use_gpu = get_metal_backend().is_ok() && n_heads <= 12;
                    if use_gpu {
                        if let Ok(backend) = get_metal_backend() {
                            for b in 0..batch_size {
                                for h in 0..n_heads {
                                    let attn_probs_head = attention_probs.slice(s![b, h, .., ..]);
                                    let v_head = v.slice(s![b, h, .., ..]);

                                    // Convert to contiguous arrays for GPU
                                    let attn_data: Vec<f32> =
                                        attn_probs_head.iter().cloned().collect();
                                    let v_data: Vec<f32> = v_head.iter().cloned().collect();

                                    // GPU matmul: attn_probs(q_seq_len × kv_seq_len) * V(kv_seq_len × head_dim)
                                    let out_vec = backend.matmul_f32(
                                        &attn_data, &v_data, q_seq_len, kv_seq_len, head_dim,
                                    )?;

                                    let out_array = ArrayD::from_shape_vec(
                                        IxDyn(&[q_seq_len, head_dim]),
                                        out_vec,
                                    )
                                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?;

                                    output.slice_mut(s![b, h, .., ..]).assign(&out_array);
                                }
                            }
                        }
                    } else {
                        // CPU fallback - parallelize only for large models (>12 heads)
                        if n_heads > 12 {
                            use scirs2_core::parallel_ops::*;

                            let indices: Vec<(usize, usize)> = (0..batch_size)
                                .flat_map(|b| (0..n_heads).map(move |h| (b, h)))
                                .collect();

                            // Compute outputs in parallel
                            let output_results: Vec<((usize, usize), ArrayD<f32>)> = indices
                                .par_iter()
                                .map(|&(b, h)| {
                                    let attn_probs_head = attention_probs.slice(s![b, h, .., ..]);
                                    let v_head = v.slice(s![b, h, .., ..]);
                                    let out = attn_probs_head.dot(&v_head);
                                    ((b, h), out.into_dyn())
                                })
                                .collect();

                            // Assign results sequentially
                            for ((b, h), out_arr) in output_results {
                                output.slice_mut(s![b, h, .., ..]).assign(&out_arr);
                            }
                        } else {
                            // Sequential for small models
                            for b in 0..batch_size {
                                for h in 0..n_heads {
                                    let attn_probs_head = attention_probs.slice(s![b, h, .., ..]);
                                    let v_head = v.slice(s![b, h, .., ..]);
                                    let out = attn_probs_head.dot(&v_head);
                                    output.slice_mut(s![b, h, .., ..]).assign(&out);
                                }
                            }
                        }
                    }
                }
                #[cfg(not(all(target_os = "macos", feature = "metal")))]
                {
                    // CPU fallback - parallelize only for large models (>12 heads)
                    if n_heads > 12 {
                        use scirs2_core::parallel_ops::*;

                        let indices: Vec<(usize, usize)> = (0..batch_size)
                            .flat_map(|b| (0..n_heads).map(move |h| (b, h)))
                            .collect();

                        // Compute outputs in parallel
                        let output_results: Vec<((usize, usize), ArrayD<f32>)> = indices
                            .par_iter()
                            .map(|&(b, h)| {
                                let attn_probs_head = attention_probs.slice(s![b, h, .., ..]);
                                let v_head = v.slice(s![b, h, .., ..]);
                                let out = attn_probs_head.dot(&v_head);
                                ((b, h), out.into_dyn())
                            })
                            .collect();

                        // Assign results sequentially
                        for ((b, h), out_arr) in output_results {
                            output.slice_mut(s![b, h, .., ..]).assign(&out_arr);
                        }
                    } else {
                        // Sequential for small models
                        for b in 0..batch_size {
                            for h in 0..n_heads {
                                let attn_probs_head = attention_probs.slice(s![b, h, .., ..]);
                                let v_head = v.slice(s![b, h, .., ..]);
                                let out = attn_probs_head.dot(&v_head);
                                output.slice_mut(s![b, h, .., ..]).assign(&out);
                            }
                        }
                    }
                }

                // Transpose back to [batch, seq_len, n_heads, head_dim]
                let output = output.permuted_axes(vec![0, 2, 1, 3]);

                // Reshape to [batch, q_seq_len, hidden_size]
                let output = output
                    .to_shape(IxDyn(&[batch_size, q_seq_len, hidden_size]))
                    .map_err(|_| TrustformersError::shape_error("Failed to reshape output".into()))?
                    .to_owned();

                // Update cache: store full K/V in original shape [batch, kv_seq_len, hidden_size]
                if let Some(cache) = layer_cache {
                    cache.key = Some(Tensor::F32(k_for_cache));
                    cache.value = Some(Tensor::F32(v_for_cache));
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
pub(crate) struct Gpt2MLP {
    c_fc: Linear,
    c_proj: Linear,
    act_fn: ActivationType,
    #[allow(dead_code)]
    dropout: f32,
}

impl Gpt2MLP {
    #[allow(dead_code)]
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

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn weights_to_gpu(&mut self, device: &Device) -> Result<()> {
        if !matches!(device, Device::Metal(_)) {
            return Ok(());
        }
        self.c_fc.weights_to_gpu(device)?;
        self.c_proj.weights_to_gpu(device)?;
        Ok(())
    }

    #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
    fn weights_to_gpu_cuda(&mut self, device: &Device) -> Result<()> {
        if !matches!(device, Device::CUDA(_)) {
            return Ok(());
        }
        self.c_fc.weights_to_gpu_cuda(device)?;
        self.c_proj.weights_to_gpu_cuda(device)?;
        Ok(())
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

    fn load_weights_from_loader(
        &mut self,
        loader: &mut dyn crate::weight_loading::WeightLoader,
        prefix: &str,
    ) -> Result<()> {
        // Transpose MLP weights too
        let c_fc_weight = loader.load_tensor(&format!("{}.c_fc.weight", prefix))?;
        self.c_fc.set_weight(transpose_tensor(c_fc_weight)?)?;
        self.c_fc.set_bias(loader.load_tensor(&format!("{}.c_fc.bias", prefix))?)?;

        let c_proj_weight = loader.load_tensor(&format!("{}.c_proj.weight", prefix))?;
        self.c_proj.set_weight(transpose_tensor(c_proj_weight)?)?;
        self.c_proj.set_bias(loader.load_tensor(&format!("{}.c_proj.bias", prefix))?)?;

        Ok(())
    }

    fn parameter_count(&self) -> usize {
        self.c_fc.parameter_count() + self.c_proj.parameter_count()
    }

    fn forward(&self, hidden_states: Tensor) -> Result<Tensor> {
        // TODO: Fused matmul+bias+GELU kernel for Metal GPU
        // The kernel is implemented and tested (trustformers-core/src/gpu_ops/metal/metalbackend_matmul_gelu_f32_group.rs)
        // Full integration requires GPU-resident buffer operations in Linear layer
        // Current implementation uses MPS/Accelerate which is already highly optimized
        let hidden_states = self.c_fc.forward(hidden_states)?;
        let hidden_states = self.act_fn.apply(hidden_states)?;
        self.c_proj.forward(hidden_states)
    }
}

/// Activation function types
#[derive(Clone)]
pub(crate) enum ActivationType {
    Gelu,
    Relu,
    Swish,
}

impl ActivationType {
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "gelu" | "gelu_new" | "gelu_fast" => Ok(Self::Gelu),
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
            Self::Gelu => gelu_core(&x), // Use NaN-safe version from trustformers_core
            Self::Relu => relu(&x),
            Self::Swish => silu(&x), // SiLU = Swish
        }
    }
}

/// Create a causal mask for attention
pub(crate) fn create_causal_mask(seq_len: usize) -> Result<Tensor> {
    let mut mask = ArrayD::<f32>::zeros(IxDyn(&[1, 1, seq_len, seq_len]));

    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[[0, 0, i, j]] = f32::NEG_INFINITY;
        }
    }

    Ok(Tensor::F32(mask))
}

/// Apply top-k filtering to logits
pub(crate) fn apply_top_k_filtering(logits: ArrayD<f32>, k: usize) -> Result<ArrayD<f32>> {
    let mut result = logits.clone();
    let mut indices_and_values: Vec<(usize, f32)> =
        logits.iter().enumerate().map(|(idx, &val)| (idx, val)).collect();

    // Sort by value in descending order
    indices_and_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Set all values outside top-k to -inf
    for (idx, _) in indices_and_values.iter().skip(k) {
        result[*idx] = f32::NEG_INFINITY;
    }

    Ok(result)
}

/// Apply top-p (nucleus) filtering to logits
pub(crate) fn apply_top_p_filtering(logits: ArrayD<f32>, p: f32) -> Result<ArrayD<f32>> {
    // Convert to probabilities
    let probs = softmax(logits.clone())?;

    let mut indices_and_probs: Vec<(usize, f32)> =
        probs.iter().enumerate().map(|(idx, &prob)| (idx, prob)).collect();

    // Sort by probability in descending order
    indices_and_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

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
pub(crate) fn sample_from_logits(logits: ArrayD<f32>) -> Result<u32> {
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
pub(crate) fn softmax(logits: ArrayD<f32>) -> Result<ArrayD<f32>> {
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
pub(crate) fn log_softmax(logits: ArrayD<f32>) -> Result<ArrayD<f32>> {
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
pub(crate) fn stack_tensors(tensors: &[Tensor]) -> Result<Tensor> {
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
        #[cfg(all(target_os = "macos", feature = "metal"))]
        Tensor::Metal(first_data) => {
            use trustformers_core::gpu_ops::metal::get_metal_backend;
            use trustformers_core::tensor::MetalTensorData;

            // Try to use GPU stacking kernel
            if let Ok(backend) = get_metal_backend() {
                // All tensors must have the same shape
                let first_shape = &first_data.shape;
                if first_shape.len() == 2 {
                    let seq_len = first_shape[0];
                    let hidden_size = first_shape[1];

                    // Collect all buffer IDs
                    let buffer_ids: Vec<_> = tensors
                        .iter()
                        .map(|t| match t {
                            Tensor::Metal(data) => Ok(data.buffer_id),
                            _ => Err(TrustformersError::tensor_op_error(
                                "All tensors must be Metal for GPU stacking",
                                "stack_tensors",
                            )),
                        })
                        .collect::<Result<Vec<_>>>()?;

                    // Stack on GPU
                    let stacked_buffer_id =
                        backend.stack_gpu_buffers(&buffer_ids, seq_len, hidden_size)?;

                    // Create output shape: [batch_size, seq_len, hidden_size]
                    let output_shape = vec![tensors.len(), seq_len, hidden_size];

                    return Ok(Tensor::Metal(MetalTensorData {
                        buffer_id: stacked_buffer_id,
                        shape: output_shape,
                        dtype: first_data.dtype,
                    }));
                }
            }

            // Fallback: convert to CPU, stack, then convert back to Metal
            let cpu_tensors: Vec<Tensor> = tensors
                .iter()
                .map(|t| t.to_device_enum(&Device::CPU))
                .collect::<Result<Vec<_>>>()?;

            let cpu_stacked = stack_tensors(&cpu_tensors)?;

            let metal_device = Device::Metal(0);
            let metal_stacked = cpu_stacked.to_device_enum(&metal_device)?;

            Ok(metal_stacked)
        },
        _ => Err(tensor_op_error(
            "tensor_operation",
            "Only F32 tensors supported for stacking".to_string(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpt2::config::Gpt2Config;
    use scirs2_core::ndarray::{ArrayD, IxDyn};
    use trustformers_core::tensor::Tensor;

    // LCG PRNG: a=6364136223846793005, c=1442695040888963407
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *state
    }

    fn lcg_f32_range(state: &mut u64, lo: f32, hi: f32) -> f32 {
        let raw = (lcg_next(state) >> 11) as f32 / (1u64 << 53) as f32;
        lo + raw * (hi - lo)
    }

    fn make_array(shape: &[usize], seed: u64) -> ArrayD<f32> {
        let mut state = seed;
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| lcg_f32_range(&mut state, -1.0, 1.0)).collect();
        ArrayD::from_shape_vec(IxDyn(shape), data).expect("Failed to create array")
    }

    fn make_tensor(shape: &[usize], seed: u64) -> Tensor {
        Tensor::F32(make_array(shape, seed))
    }

    // ---- create_causal_mask tests ----

    #[test]
    fn test_causal_mask_shape() {
        let seq_len = 5;
        let mask = create_causal_mask(seq_len).expect("create_causal_mask failed");
        let shape = mask.shape();
        assert_eq!(shape, &[1, 1, seq_len, seq_len]);
    }

    #[test]
    fn test_causal_mask_diagonal_not_neg_inf() {
        let seq_len = 4;
        let mask = create_causal_mask(seq_len).expect("create_causal_mask failed");
        if let Tensor::F32(arr) = &mask {
            for i in 0..seq_len {
                let val = arr[[0, 0, i, i]];
                assert!(
                    val.is_finite(),
                    "Diagonal of causal mask must be finite at ({i},{i})"
                );
            }
        } else {
            panic!("Expected F32 tensor");
        }
    }

    #[test]
    fn test_causal_mask_future_tokens_are_neg_inf() {
        let seq_len = 5;
        let mask = create_causal_mask(seq_len).expect("create_causal_mask failed");
        if let Tensor::F32(arr) = &mask {
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    let val = arr[[0, 0, i, j]];
                    assert!(
                        val.is_infinite() && val < 0.0,
                        "Future token at ({i},{j}) must be -inf, got {val}"
                    );
                }
            }
        } else {
            panic!("Expected F32 tensor");
        }
    }

    #[test]
    fn test_causal_mask_past_tokens_are_zero() {
        let seq_len = 4;
        let mask = create_causal_mask(seq_len).expect("create_causal_mask failed");
        if let Tensor::F32(arr) = &mask {
            for i in 0..seq_len {
                for j in 0..=i {
                    let val = arr[[0, 0, i, j]];
                    assert!(
                        val == 0.0,
                        "Past/current token at ({i},{j}) must be 0, got {val}"
                    );
                }
            }
        } else {
            panic!("Expected F32 tensor");
        }
    }

    #[test]
    fn test_causal_mask_length_1() {
        let mask = create_causal_mask(1).expect("create_causal_mask(1) failed");
        if let Tensor::F32(arr) = &mask {
            assert_eq!(arr[[0, 0, 0, 0]], 0.0);
        }
    }

    // ---- softmax tests ----

    #[test]
    fn test_softmax_sums_to_one() {
        let mut state = 7u64;
        let n = 10;
        let data: Vec<f32> = (0..n).map(|_| lcg_f32_range(&mut state, -2.0, 2.0)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).expect("array creation failed");
        let result = softmax(arr).expect("softmax failed");
        let sum: f32 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax sum must be ~1.0, got {sum}"
        );
    }

    #[test]
    fn test_softmax_all_positive() {
        let mut state = 13u64;
        let n = 8;
        let data: Vec<f32> = (0..n).map(|_| lcg_f32_range(&mut state, -3.0, 3.0)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).expect("array creation failed");
        let result = softmax(arr).expect("softmax failed");
        for val in result.iter() {
            assert!(*val >= 0.0, "softmax output must be non-negative");
        }
    }

    // ---- log_softmax tests ----

    #[test]
    fn test_log_softmax_non_positive() {
        let mut state = 17u64;
        let n = 8;
        let data: Vec<f32> = (0..n).map(|_| lcg_f32_range(&mut state, -2.0, 2.0)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).expect("array creation failed");
        let result = log_softmax(arr).expect("log_softmax failed");
        for val in result.iter() {
            assert!(
                *val <= 0.0 + 1e-6,
                "log_softmax output must be <= 0, got {val}"
            );
        }
    }

    #[test]
    fn test_log_softmax_exp_sums_to_one() {
        let mut state = 31u64;
        let n = 6;
        let data: Vec<f32> = (0..n).map(|_| lcg_f32_range(&mut state, -1.0, 1.0)).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[n]), data).expect("array creation failed");
        let result = log_softmax(arr).expect("log_softmax failed");
        let sum_exp: f32 = result.iter().map(|x| x.exp()).sum();
        assert!(
            (sum_exp - 1.0).abs() < 1e-5,
            "exp(log_softmax) must sum to 1, got {sum_exp}"
        );
    }

    // ---- apply_top_k_filtering tests ----

    #[test]
    fn test_top_k_keeps_k_finite_values() {
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[10]), data).expect("array creation failed");
        let k = 3;
        let result = apply_top_k_filtering(arr, k).expect("top_k filter failed");
        let finite_count = result.iter().filter(|&&v| v.is_finite()).count();
        assert_eq!(finite_count, k, "top-k should keep exactly k finite values");
    }

    #[test]
    fn test_top_k_largest_values_retained() {
        // data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[10]), data).expect("array failed");
        let k = 3;
        let result = apply_top_k_filtering(arr, k).expect("top_k filter failed");
        // Top 3 values are 7, 8, 9 at indices 7, 8, 9
        assert!(result[7].is_finite());
        assert!(result[8].is_finite());
        assert!(result[9].is_finite());
        assert!(result[0].is_infinite());
    }

    // ---- apply_top_p_filtering tests ----

    #[test]
    fn test_top_p_at_least_one_finite() {
        let data: Vec<f32> = (0..10).map(|i| i as f32 + 1.0).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[10]), data).expect("array failed");
        let result = apply_top_p_filtering(arr, 0.5).expect("top_p filter failed");
        let finite_count = result.iter().filter(|&&v| v.is_finite()).count();
        assert!(finite_count >= 1, "top-p must keep at least one token");
    }

    #[test]
    fn test_top_p_full_probability_keeps_all() {
        let data: Vec<f32> = (0..5).map(|i| i as f32 + 1.0).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[5]), data).expect("array failed");
        let result = apply_top_p_filtering(arr, 1.0).expect("top_p filter failed");
        let finite_count = result.iter().filter(|&&v| v.is_finite()).count();
        assert_eq!(finite_count, 5, "p=1.0 should keep all tokens");
    }

    // ---- stack_tensors tests ----

    #[test]
    fn test_stack_tensors_basic() {
        let t1 = make_tensor(&[3, 4], 11);
        let t2 = make_tensor(&[3, 4], 22);
        let stacked = stack_tensors(&[t1, t2]).expect("stack_tensors failed");
        let shape = stacked.shape();
        assert_eq!(shape[0], 2, "Batch dim must be 2");
        assert_eq!(shape[1], 3);
        assert_eq!(shape[2], 4);
    }

    #[test]
    fn test_stack_tensors_empty_fails() {
        let result = stack_tensors(&[]);
        assert!(result.is_err(), "Stacking empty list must fail");
    }

    #[test]
    fn test_stack_tensors_shape_mismatch_fails() {
        let t1 = make_tensor(&[3, 4], 11);
        let t2 = make_tensor(&[4, 4], 22); // different shape
        let result = stack_tensors(&[t1, t2]);
        assert!(
            result.is_err(),
            "Stacking tensors with different shapes must fail"
        );
    }

    // ---- Gpt2Block creation test ----

    #[test]
    fn test_gpt2_block_creates_ok() {
        let cfg = Gpt2Config::default();
        let block = Gpt2Block::new(&cfg);
        assert!(
            block.is_ok(),
            "Gpt2Block::new should succeed with default config"
        );
    }

    #[test]
    fn test_gpt2_block_parameter_count_nonzero() {
        let cfg = Gpt2Config::default();
        let block = Gpt2Block::new(&cfg).expect("Block creation failed");
        assert!(block.parameter_count() > 0, "Block must have parameters");
    }

    // ---- MLP inner dim test ----

    #[test]
    fn test_gpt2_mlp_inner_dim_4x() {
        // When n_inner is None, inner dim = 4 * n_embd
        let cfg = Gpt2Config::default();
        assert!(cfg.n_inner.is_none(), "Default n_inner must be None");
        // The MLP created with this config should have inner_dim = 4 * 768 = 3072
        // We verify by checking the block can be created (it uses 4*n_embd internally)
        let block = Gpt2Block::new(&cfg).expect("Block creation failed");
        // The parameter count should reflect the 4x expansion
        let count = block.parameter_count();
        // rough lower bound: at least n_embd * 4 * n_embd for c_fc weight
        assert!(
            count > 768 * 3072,
            "MLP param count must reflect 4x expansion"
        );
    }

    // ---- ActivationType tests ----

    #[test]
    fn test_gelu_activation_on_zero() {
        let t = Tensor::from_vec(vec![0.0f32], &[1]).expect("tensor creation failed");
        let result = trustformers_core::ops::activations::gelu(&t).expect("gelu failed");
        if let Tensor::F32(arr) = result {
            assert!(arr[0].abs() < 1e-5, "gelu(0) must be ~0");
        }
    }

    #[test]
    fn test_silu_activation_on_positive() {
        let t = Tensor::from_vec(vec![2.0f32], &[1]).expect("tensor creation failed");
        let result = trustformers_core::ops::activations::silu(&t).expect("silu failed");
        if let Tensor::F32(arr) = result {
            // SiLU(2) = 2 * sigmoid(2) ≈ 1.762
            assert!(
                arr[0] > 1.5 && arr[0] < 2.0,
                "SiLU(2) should be ~1.76, got {}",
                arr[0]
            );
        }
    }
}
