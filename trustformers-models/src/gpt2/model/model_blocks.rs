//! GPT-2 building blocks and helper functions
//!
//! Contains Gpt2Block, Gpt2Attention, Gpt2MLP, ActivationType,
//! and utility functions (causal mask, sampling, softmax).

use scirs2_core::ndarray::{s, ArrayD, Axis, IxDyn};
use trustformers_core::{
    device::Device,
    errors::{invalid_config, tensor_op_error, Result, TrustformersError},
    layers::{LayerNorm, Linear},
    tensor::Tensor,
    traits::{Layer, WeightReader},
};

use super::model_core::{transpose_tensor, LayerCache};
use super::model_ops::ActivationType;
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

    /// Append this block's parameters under `<prefix>.…` in HuggingFace order.
    ///
    /// `prefix` is the block's checkpoint path (`transformer.h.3`, say). The
    /// names produced here mirror [`Gpt2Block::load_weights`] exactly, so a file
    /// written from `named_tensors` reloads through `load_pretrained`.
    ///
    /// # Weight layout
    ///
    /// `attn.c_attn`, `attn.c_proj`, `mlp.c_fc` and `mlp.c_proj` are HuggingFace
    /// `Conv1D` layers, stored `[in, out]` in the checkpoint and transposed to
    /// `[out, in]` on load. The trait contract requires *live* references, so
    /// they are exposed in the model's own `[out, in]` layout — transposing here
    /// would mean returning references to temporaries, which is impossible, and
    /// silently copying would break the "live parameters" contract. Consumers
    /// that need HF's on-disk layout must transpose these four names themselves.
    pub(crate) fn collect_named_parameters<'a>(
        &'a self,
        prefix: &str,
        into: &mut Vec<(String, &'a Tensor)>,
    ) {
        self.ln_1.collect_named_parameters(&format!("{prefix}.ln_1"), into);
        self.attn.collect_named_parameters(&format!("{prefix}.attn"), into);
        self.ln_2.collect_named_parameters(&format!("{prefix}.ln_2"), into);
        self.mlp.collect_named_parameters(&format!("{prefix}.mlp"), into);
    }

    /// Mutable counterpart of [`Gpt2Block::collect_named_parameters`].
    pub(crate) fn collect_named_parameters_mut<'a>(
        &'a mut self,
        prefix: &str,
        into: &mut Vec<(String, &'a mut Tensor)>,
    ) {
        self.ln_1.collect_named_parameters_mut(&format!("{prefix}.ln_1"), into);
        self.attn.collect_named_parameters_mut(&format!("{prefix}.attn"), into);
        self.ln_2.collect_named_parameters_mut(&format!("{prefix}.ln_2"), into);
        self.mlp.collect_named_parameters_mut(&format!("{prefix}.mlp"), into);
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

    /// GPU-resident attention with a GPU-resident KV cache over the oxicuda
    /// CUDA backend — the CUDA counterpart of the Metal fast path above.
    ///
    /// Returns `Ok(None)` when the fast path does not apply (batch != 1,
    /// non-F32 resident dtype, dimension mismatch, or a host-format cache
    /// left by earlier CPU steps); the caller then downloads QKV once and
    /// continues on the host path. Like the Metal path, the resident chain
    /// ignores `attention_mask` (batch-1 causal generation).
    ///
    /// Cache layout (device-resident, this path's private convention):
    /// `cache.key` and `cache.value` both hold heads-major
    /// `[1, n_head, kv_len, d_head]` buffers — `K^T` is applied as a GEMM
    /// transpose flag by the backend (transpose-aware since oxicuda 0.4.1),
    /// so decode steps append a contiguous row per head to either cache.
    ///
    /// Routing: an empty cache runs bulk causal prefill; a non-empty cache
    /// runs single-query decode per new token (exact causal semantics for any
    /// `seq_len`, one token at a time). Once this path has populated a CUDA
    /// cache, mixed configurations that would silently drop that cache on the
    /// host path are reported as hard errors instead.
    #[cfg(feature = "cuda")]
    fn cuda_resident_attention(
        &self,
        qkv: &Tensor,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
        was_2d: bool,
        layer_cache: Option<&mut LayerCache>,
    ) -> Result<Option<Tensor>> {
        use trustformers_core::gpu_ops::cuda::get_cuda_backend;
        use trustformers_core::tensor::{CudaTensorData, DType};

        let Tensor::CUDA(qkv_data) = qkv else {
            return Ok(None);
        };
        let n_head = self.n_head;
        let d_head = self.d_head;
        let device_id = qkv_data.device_id();

        // Inspect the cache first: a resident cache must never leak into the
        // host fallback (the CPU path only merges F32 caches and would
        // silently drop the history).
        let (cached_k, cached_v, cached_len) = match &layer_cache {
            Some(cache) => match (&cache.key, &cache.value) {
                (Some(Tensor::CUDA(k)), Some(Tensor::CUDA(v)))
                    if k.device_id() == device_id
                        && v.device_id() == device_id
                        && matches!(
                            (k.shape.as_slice(), v.shape.as_slice()),
                            ([1, kh, k_kv, kd], [1, vh, v_kv, vd])
                                if *kh == n_head && *kd == d_head && *vh == n_head
                                    && *vd == d_head && k_kv == v_kv
                        ) =>
                {
                    let kv_len = k.shape[2];
                    (
                        Some(Tensor::CUDA(k.clone())),
                        Some(Tensor::CUDA(v.clone())),
                        kv_len,
                    )
                },
                (Some(Tensor::CUDA(_)), _) | (_, Some(Tensor::CUDA(_))) => {
                    return Err(tensor_op_error(
                        "Gpt2Attention::cuda_resident_attention",
                        "GPU-resident KV cache has an unexpected layout or device; \
                         cannot continue on the host path without dropping it",
                    ));
                },
                (None, None) => (None, None, 0),
                // Host-format cache (populated by earlier CPU steps): the
                // host path owns that history — decline the resident path
                // instead of restarting from an "empty" cache.
                _ => return Ok(None),
            },
            None => (None, None, 0),
        };
        if batch_size != 1
            || qkv_data.dtype != DType::F32
            || hidden_size != n_head * d_head
            || seq_len == 0
        {
            if cached_len > 0 {
                return Err(tensor_op_error(
                    "Gpt2Attention::cuda_resident_attention",
                    "GPU-resident KV cache exists but the resident path no longer \
                     applies (batch/dtype/shape changed mid-generation)",
                ));
            }
            return Ok(None);
        }

        let backend = get_cuda_backend(device_id)?;
        let row_stride = 3 * hidden_size;
        let scale = 1.0 / (d_head as f32).sqrt();
        let dtype = qkv_data.dtype;
        // Every intermediate id is wrapped in a `Tensor::CUDA` immediately so
        // the refcounted handle lifecycle frees it on all paths.
        let wrap =
            |id, shape: Vec<usize>| Tensor::CUDA(CudaTensorData::new(id, device_id, shape, dtype));
        let id_of = |t: &Tensor| -> Result<trustformers_core::gpu_ops::cuda::BufferId> {
            match t {
                Tensor::CUDA(data) => Ok(data.buffer_id()),
                _ => Err(tensor_op_error(
                    "Gpt2Attention::cuda_resident_attention",
                    "expected resident tensor",
                )),
            }
        };
        let out_shape = |rows: usize| {
            if was_2d {
                vec![rows, hidden_size]
            } else {
                vec![1, rows, hidden_size]
            }
        };

        if cached_len == 0 {
            // Prefill from an empty cache: bulk per-head causal attention.
            // GPT-2 QKV packing: row = [Q(hidden) | K(hidden) | V(hidden)],
            // head h at column h * d_head inside each component.
            let q_heads = wrap(
                backend.gather_heads_gpu_to_gpu(
                    &qkv_data.buffer_id(),
                    seq_len,
                    row_stride,
                    0,
                    d_head,
                    n_head,
                    d_head,
                    true,
                )?,
                vec![1, n_head, seq_len, d_head],
            );
            // K is gathered heads-major [1, H, seq, d] like Q/V: the score
            // GEMM applies K^T as a transpose flag (transpose-aware since
            // oxicuda 0.4.1) and the same buffer doubles as the cache entry.
            let k_heads = wrap(
                backend.gather_heads_gpu_to_gpu(
                    &qkv_data.buffer_id(),
                    seq_len,
                    row_stride,
                    hidden_size,
                    d_head,
                    n_head,
                    d_head,
                    true,
                )?,
                vec![1, n_head, seq_len, d_head],
            );
            let v_heads = wrap(
                backend.gather_heads_gpu_to_gpu(
                    &qkv_data.buffer_id(),
                    seq_len,
                    row_stride,
                    2 * hidden_size,
                    d_head,
                    n_head,
                    d_head,
                    true,
                )?,
                vec![1, n_head, seq_len, d_head],
            );

            let attn_heads = wrap(
                backend.attention_prefill_gpu_to_gpu(
                    &id_of(&q_heads)?,
                    &id_of(&k_heads)?,
                    &id_of(&v_heads)?,
                    n_head,
                    seq_len,
                    d_head,
                    scale,
                )?,
                vec![1, n_head, seq_len, d_head],
            );
            let merged = wrap(
                backend.gather_heads_gpu_to_gpu(
                    &id_of(&attn_heads)?,
                    seq_len,
                    d_head,
                    0,
                    seq_len * d_head,
                    n_head,
                    d_head,
                    false,
                )?,
                out_shape(seq_len),
            );

            if let Some(cache) = layer_cache {
                cache.key = Some(k_heads.clone());
                cache.value = Some(v_heads.clone());
            }

            return self.c_proj.forward(merged).map(Some);
        }

        // Cached generation: single-query decode per new token (each token
        // appends its K/V and attends to everything before and including it).
        let mut k_cur = cached_k.ok_or_else(|| {
            tensor_op_error(
                "Gpt2Attention::cuda_resident_attention",
                "resident cache length > 0 but key tensor missing",
            )
        })?;
        let mut v_cur = cached_v.ok_or_else(|| {
            tensor_op_error(
                "Gpt2Attention::cuda_resident_attention",
                "resident cache length > 0 but value tensor missing",
            )
        })?;
        let mut kv_len = cached_len;
        let mut merged: Option<Tensor> = None;

        for t in 0..seq_len {
            let token_base = t * row_stride;
            // For a single row the Q/K/V component slices are contiguous
            // [n_head, d_head] blocks — the same bytes serve as [H, d] and
            // [H, 1, d] (one K or V row per head).
            let q_t = wrap(
                backend.gather_heads_gpu_to_gpu(
                    &qkv_data.buffer_id(),
                    1,
                    row_stride,
                    token_base,
                    d_head,
                    n_head,
                    d_head,
                    true,
                )?,
                vec![n_head, d_head],
            );
            let k_new = wrap(
                backend.gather_heads_gpu_to_gpu(
                    &qkv_data.buffer_id(),
                    1,
                    row_stride,
                    token_base + hidden_size,
                    d_head,
                    n_head,
                    d_head,
                    true,
                )?,
                vec![n_head, 1, d_head],
            );
            let v_new = wrap(
                backend.gather_heads_gpu_to_gpu(
                    &qkv_data.buffer_id(),
                    1,
                    row_stride,
                    token_base + 2 * hidden_size,
                    d_head,
                    n_head,
                    d_head,
                    true,
                )?,
                vec![n_head, 1, d_head],
            );

            k_cur = wrap(
                backend.concat_v_cache_gpu_to_gpu(
                    Some(&id_of(&k_cur)?),
                    &id_of(&k_new)?,
                    n_head,
                    kv_len,
                    1,
                    d_head,
                )?,
                vec![1, n_head, kv_len + 1, d_head],
            );
            v_cur = wrap(
                backend.concat_v_cache_gpu_to_gpu(
                    Some(&id_of(&v_cur)?),
                    &id_of(&v_new)?,
                    n_head,
                    kv_len,
                    1,
                    d_head,
                )?,
                vec![1, n_head, kv_len + 1, d_head],
            );
            kv_len += 1;

            let out_t = wrap(
                backend.attention_decode_gpu_to_gpu(
                    &id_of(&q_t)?,
                    &id_of(&k_cur)?,
                    &id_of(&v_cur)?,
                    n_head,
                    kv_len,
                    d_head,
                    scale,
                )?,
                out_shape(1),
            );
            merged = Some(match merged {
                None => out_t,
                Some(prev) => wrap(
                    backend.concat_v_cache_gpu_to_gpu(
                        Some(&id_of(&prev)?),
                        &id_of(&out_t)?,
                        1,
                        t,
                        1,
                        hidden_size,
                    )?,
                    out_shape(t + 1),
                ),
            });
        }

        let merged = merged.ok_or_else(|| {
            tensor_op_error(
                "Gpt2Attention::cuda_resident_attention",
                "decode loop produced no output rows",
            )
        })?;

        if let Some(cache) = layer_cache {
            cache.key = Some(k_cur.clone());
            cache.value = Some(v_cur.clone());
        }

        self.c_proj.forward(merged).map(Some)
    }

    /// Bind the fused QKV and output projections from a checkpoint.
    ///
    /// HuggingFace's GPT-2 uses `transformers.pytorch_utils.Conv1D`, not
    /// `nn.Linear`, and `Conv1D` stores its weight as `[in_features,
    /// out_features]` — the transpose of the `[out_features, in_features]`
    /// layout [`trustformers_core::layers::Linear`] expects. Hence the
    /// transposition on the weights and none on the biases, which are `[out]`
    /// in both conventions.
    fn load_weights(&mut self, reader: &mut dyn WeightReader, prefix: &str) -> Result<()> {
        // Fused QKV projection: Conv1D [in, 3*in] -> Linear [3*in, in].
        let c_attn_weight = reader.read_tensor(&format!("{}.c_attn.weight", prefix))?;
        self.c_attn.set_weight(transpose_tensor(c_attn_weight)?)?;
        self.c_attn.set_bias(reader.read_tensor(&format!("{}.c_attn.bias", prefix))?)?;

        // Output projection: Conv1D [in, in] -> Linear [in, in], still transposed.
        let c_proj_weight = reader.read_tensor(&format!("{}.c_proj.weight", prefix))?;
        self.c_proj.set_weight(transpose_tensor(c_proj_weight)?)?;
        self.c_proj.set_bias(reader.read_tensor(&format!("{}.c_proj.bias", prefix))?)?;

        Ok(())
    }

    /// Same binding as [`Gpt2Attention::load_weights`], driven by a
    /// [`crate::weight_loading::WeightLoader`] instead of a `WeightReader`.
    fn load_weights_from_loader(
        &mut self,
        loader: &mut dyn crate::weight_loading::WeightLoader,
        prefix: &str,
    ) -> Result<()> {
        // Fused QKV projection: Conv1D [in, 3*in] -> Linear [3*in, in].
        let c_attn_weight = loader.load_tensor(&format!("{}.c_attn.weight", prefix))?;
        self.c_attn.set_weight(transpose_tensor(c_attn_weight)?)?;
        self.c_attn.set_bias(loader.load_tensor(&format!("{}.c_attn.bias", prefix))?)?;

        // Output projection: Conv1D [in, in] -> Linear [in, in], still transposed.
        let c_proj_weight = loader.load_tensor(&format!("{}.c_proj.weight", prefix))?;
        self.c_proj.set_weight(transpose_tensor(c_proj_weight)?)?;
        self.c_proj.set_bias(loader.load_tensor(&format!("{}.c_proj.bias", prefix))?)?;

        Ok(())
    }

    fn parameter_count(&self) -> usize {
        self.c_attn.parameter_count() + self.c_proj.parameter_count()
    }

    /// Append the fused QKV and output projections under `<prefix>.…`.
    ///
    /// Names mirror [`Gpt2Attention::load_weights`]; see
    /// [`Gpt2Block::collect_named_parameters`] for the `Conv1D` layout caveat.
    fn collect_named_parameters<'a>(&'a self, prefix: &str, into: &mut Vec<(String, &'a Tensor)>) {
        self.c_attn.collect_named_parameters(&format!("{prefix}.c_attn"), into);
        self.c_proj.collect_named_parameters(&format!("{prefix}.c_proj"), into);
    }

    /// Mutable counterpart of [`Gpt2Attention::collect_named_parameters`].
    fn collect_named_parameters_mut<'a>(
        &'a mut self,
        prefix: &str,
        into: &mut Vec<(String, &'a mut Tensor)>,
    ) {
        self.c_attn.collect_named_parameters_mut(&format!("{prefix}.c_attn"), into);
        self.c_proj.collect_named_parameters_mut(&format!("{prefix}.c_proj"), into);
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
                backend.split_qkv_gpu(&qkv_data.buffer_id(), batch_size, seq_len, hidden_size)?;

            // Get cached K/V buffer IDs and sequence length (if cache exists).
            // Owned `BufferId`s (not references): they must outlive this
            // statement to reach `concat_kv_cache` below, and `BufferId` is
            // `Copy`, so there is no reason to borrow from the cache tensors.
            let (cached_k_id, cached_v_id, cached_seq_len) = if let Some(cache) = &layer_cache {
                match (&cache.key, &cache.value) {
                    (Some(Tensor::Metal(k_metal)), Some(Tensor::Metal(v_metal))) => {
                        let cached_shape = &k_metal.shape; // [batch, num_heads, cached_seq, head_dim]
                        let cached_seq = cached_shape[2];
                        (
                            Some(k_metal.buffer_id()),
                            Some(v_metal.buffer_id()),
                            cached_seq,
                        )
                    },
                    // First token of a sequence: the cache holds nothing yet.
                    _ => (None, None, 0),
                }
            } else {
                // This layer has no cache slot, so there is nothing to extend.
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
                cached_k_id.as_ref(),
                &k_new_heads_id,
                batch_size,
                self.n_head,
                cached_seq_len,
                seq_len, // new_seq_len
                self.d_head,
            )?;

            let v_heads_id = backend.concat_kv_cache(
                cached_v_id.as_ref(),
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

            // Update cache with full K/V (keep on GPU!). Each id is fresh out of
            // `concat_kv_cache` above and has never been wrapped in a handle yet,
            // so `::new` here is the required first (and only) wrap.
            if let Some(cache) = layer_cache {
                cache.key = Some(Tensor::Metal(MetalTensorData::new(
                    &backend,
                    k_heads_id,
                    vec![batch_size, self.n_head, total_seq_len, self.d_head],
                    qkv_data.dtype,
                )?));
                cache.value = Some(Tensor::Metal(MetalTensorData::new(
                    &backend,
                    v_heads_id,
                    vec![batch_size, self.n_head, total_seq_len, self.d_head],
                    qkv_data.dtype,
                )?));
            }

            // Wrap in Metal tensor and apply output projection. `attn_output_id` is
            // likewise fresh out of `reshape_from_heads_gpu` above.
            let attn_output = Tensor::Metal(MetalTensorData::new(
                &backend,
                attn_output_id,
                vec![batch_size, seq_len, hidden_size],
                qkv_data.dtype,
            )?);

            // Apply output projection (stays on GPU)
            let output = self.c_proj.forward(attn_output)?;

            // Remove batch dimension if it was added
            return if was_2d {
                match output {
                    Tensor::Metal(mut metal_data) if metal_data.shape[0] == 1 => {
                        // Reshape [1, seq, hidden] → [seq, hidden]: same buffer, new
                        // shape. Move the handle `output` already owns instead of
                        // minting a second one for an id it already wraps —
                        // `MetalTensorData::new`'s contract is that each raw id is
                        // wrapped at most once and all further sharing goes through
                        // `clone()`, which this is not (it's a single-owner reshape).
                        metal_data.shape = vec![metal_data.shape[1], metal_data.shape[2]];
                        Ok(Tensor::Metal(metal_data))
                    },
                    _ => Ok(output),
                }
            } else {
                Ok(output)
            };
        }

        // GPU attention path with GPU-resident KV-cache (CUDA / oxicuda).
        // Mirrors the Metal fast path above: prefill runs bulk causal
        // attention, generation runs single-query decode against the resident
        // cache. `Ok(None)` means the fast path declined; the download
        // fallback below then applies.
        // Rebind mutably only for the CUDA path: `as_deref_mut` needs a
        // mutable binding, and adding `mut` to the parameter itself would
        // trip `unused_mut` in non-CUDA builds.
        #[cfg(feature = "cuda")]
        let mut layer_cache = layer_cache;
        #[cfg(feature = "cuda")]
        if matches!(&qkv, Tensor::CUDA(_)) {
            if let Some(output) = self.cuda_resident_attention(
                &qkv,
                batch_size,
                seq_len,
                hidden_size,
                was_2d,
                layer_cache.as_deref_mut(),
            )? {
                return Ok(output);
            }
        }

        // CUDA fallback: the resident fast path declined (batch > 1,
        // host-format cache from earlier CPU steps, ...) — download QKV once
        // and continue on the host path below.
        #[cfg(feature = "cuda")]
        let qkv = match &qkv {
            Tensor::CUDA(_) => qkv.to_device_enum(&Device::CPU)?,
            _ => qkv,
        };

        // Fallback: CPU attention path (with cache support)
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let qkv = match &qkv {
            Tensor::Metal(qkv_data) => {
                use trustformers_core::gpu_ops::metal::get_metal_backend;

                let backend = get_metal_backend()?;

                // Split QKV on GPU then download
                let (q_id, k_id, v_id) = backend.split_qkv_gpu(
                    &qkv_data.buffer_id(),
                    batch_size,
                    seq_len,
                    hidden_size,
                )?;

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

    /// Bind the two MLP projections from a checkpoint.
    ///
    /// Like the attention block, GPT-2's MLP is built from `Conv1D` layers whose
    /// weights are stored `[in_features, out_features]`, so both need
    /// transposing into the `[out_features, in_features]` layout `Linear` uses.
    fn load_weights(&mut self, reader: &mut dyn WeightReader, prefix: &str) -> Result<()> {
        let c_fc_weight = reader.read_tensor(&format!("{}.c_fc.weight", prefix))?;
        self.c_fc.set_weight(transpose_tensor(c_fc_weight)?)?;
        self.c_fc.set_bias(reader.read_tensor(&format!("{}.c_fc.bias", prefix))?)?;

        let c_proj_weight = reader.read_tensor(&format!("{}.c_proj.weight", prefix))?;
        self.c_proj.set_weight(transpose_tensor(c_proj_weight)?)?;
        self.c_proj.set_bias(reader.read_tensor(&format!("{}.c_proj.bias", prefix))?)?;

        Ok(())
    }

    /// Same binding as [`Gpt2MLP::load_weights`], driven by a
    /// [`crate::weight_loading::WeightLoader`] instead of a `WeightReader`.
    fn load_weights_from_loader(
        &mut self,
        loader: &mut dyn crate::weight_loading::WeightLoader,
        prefix: &str,
    ) -> Result<()> {
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

    /// Append the two MLP projections under `<prefix>.…`.
    ///
    /// Names mirror [`Gpt2MLP::load_weights`]; see
    /// [`Gpt2Block::collect_named_parameters`] for the `Conv1D` layout caveat.
    fn collect_named_parameters<'a>(&'a self, prefix: &str, into: &mut Vec<(String, &'a Tensor)>) {
        self.c_fc.collect_named_parameters(&format!("{prefix}.c_fc"), into);
        self.c_proj.collect_named_parameters(&format!("{prefix}.c_proj"), into);
    }

    /// Mutable counterpart of [`Gpt2MLP::collect_named_parameters`].
    fn collect_named_parameters_mut<'a>(
        &'a mut self,
        prefix: &str,
        into: &mut Vec<(String, &'a mut Tensor)>,
    ) {
        self.c_fc.collect_named_parameters_mut(&format!("{prefix}.c_fc"), into);
        self.c_proj.collect_named_parameters_mut(&format!("{prefix}.c_proj"), into);
    }

    /// Fused `matmul + bias + GELU` for the `c_fc` projection on the Metal GPU.
    ///
    /// Collapses the three separate operations performed by `c_fc.forward`
    /// (matmul, bias-add) followed by the GELU activation into a single
    /// `MetalBackend::matmul_bias_gelu_f32` kernel dispatch (defined in
    /// `trustformers-core/src/gpu_ops/metal/metalbackend_matmul_gelu_f32_group.rs`).
    /// The Metal kernel uses the exact same tanh GELU approximation as the CPU
    /// `gelu` op, so the result is numerically equivalent (within f32 rounding)
    /// to the separate path.
    ///
    /// Returns `Ok(Some(out))` when the fused GPU path ran, or `Ok(None)` when it
    /// is not applicable (non-Metal device, non-GELU activation, missing bias, or
    /// non-F32 data) and the caller should fall back to the separate ops.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn try_fused_c_fc_gelu(&self, hidden_states: &Tensor) -> Result<Option<Tensor>> {
        use trustformers_core::gpu_ops::metal::get_metal_backend;

        // The fused kernel computes GELU(A @ B + bias); only valid for the GELU
        // activation with a bias present, running on a Metal device.
        if !matches!(self.act_fn, ActivationType::Gelu) {
            return Ok(None);
        }
        if !matches!(self.c_fc.device(), Device::Metal(_)) {
            return Ok(None);
        }
        let bias = match self.c_fc.bias() {
            Some(bias) => bias,
            None => return Ok(None),
        };

        // Materialize the input as a contiguous f32 CPU array (no copy for F32).
        let input_cpu;
        let input_arr = match hidden_states {
            Tensor::F32(arr) => arr,
            Tensor::Metal(_) => {
                input_cpu = hidden_states.to_device_enum(&Device::CPU)?;
                match &input_cpu {
                    Tensor::F32(arr) => arr,
                    _ => return Ok(None),
                }
            },
            _ => return Ok(None),
        };
        let input_shape = input_arr.shape().to_vec();
        if input_shape.len() < 2 {
            return Ok(None);
        }
        let k = input_shape[input_shape.len() - 1]; // in_features
        let m: usize = input_shape[..input_shape.len() - 1].iter().product();

        // Weight is stored as [out_features, in_features]; the fused kernel expects
        // B = [K, N] = [in_features, out_features], i.e. the transposed weight.
        let weight_t = self.c_fc.weight().transpose(0, 1)?;
        let weight_arr = match &weight_t {
            Tensor::F32(arr) => arr,
            _ => return Ok(None),
        };
        if weight_arr.ndim() != 2 || weight_arr.shape()[0] != k {
            return Ok(None);
        }
        let n = weight_arr.shape()[1]; // out_features

        // Bias as a contiguous f32 [N] slice.
        let bias_cpu = bias.to_device_enum(&Device::CPU)?;
        let bias_arr = match &bias_cpu {
            Tensor::F32(arr) => arr,
            _ => return Ok(None),
        };
        if bias_arr.len() != n {
            return Ok(None);
        }

        // Contiguous, row-major data for the GPU upload.
        let input_std = input_arr.as_standard_layout();
        let input_data: Vec<f32> = input_std.iter().copied().collect();
        let weight_std = weight_arr.as_standard_layout();
        let weight_data: Vec<f32> = weight_std.iter().copied().collect();
        let bias_data: Vec<f32> = bias_arr.iter().copied().collect();

        // Single fused GPU dispatch: GELU(input @ weightᵀ + bias).
        let backend = get_metal_backend()?;
        let result =
            backend.matmul_bias_gelu_f32(&input_data, &weight_data, &bias_data, m, k, n)?;

        // Restore the leading (batch) dimensions, replacing the feature dim with N.
        let mut output_shape = input_shape[..input_shape.len() - 1].to_vec();
        output_shape.push(n);
        let output_arr = ArrayD::from_shape_vec(IxDyn(&output_shape), result).map_err(|e| {
            tensor_op_error(
                "Gpt2MLP::try_fused_c_fc_gelu",
                format!("failed to reshape fused matmul+bias+GELU result: {e}"),
            )
        })?;

        Ok(Some(Tensor::F32(output_arr)))
    }

    fn forward(&self, hidden_states: Tensor) -> Result<Tensor> {
        // Metal GPU fast path: fuse the `c_fc` matmul + bias + GELU into a single
        // kernel dispatch (matmul_bias_gelu_f32) when the activation is GELU and a
        // bias is present. Falls back to the separate ops below when not applicable.
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            if let Some(fused) = self.try_fused_c_fc_gelu(&hidden_states)? {
                return self.c_proj.forward(fused);
            }
        }

        let hidden_states = self.c_fc.forward(hidden_states)?;
        let hidden_states = self.act_fn.apply(hidden_states)?;
        self.c_proj.forward(hidden_states)
    }
}

#[cfg(test)]
#[path = "model_blocks_tests.rs"]
mod tests;

/// Parity tests for the fused Metal `matmul + bias + GELU` MLP path.
///
/// These assert that `Gpt2MLP::forward` running the fused single-kernel Metal
/// path produces the same result (within f32 tolerance) as the separate
/// matmul -> bias -> GELU ops on CPU. They run on the actual GPU.
#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod metal_fused_mlp_tests {
    use super::*;
    use crate::gpt2::config::Gpt2Config;
    use scirs2_core::ndarray::{ArrayD, IxDyn};
    use trustformers_core::device::Device;
    use trustformers_core::tensor::Tensor;

    // Deterministic LCG so the comparison is fully reproducible.
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *state
    }

    /// Build a deterministic array with values in `[-scale, scale)`.
    fn small_array(shape: &[usize], seed: u64, scale: f32) -> ArrayD<f32> {
        let mut state = seed;
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|_| {
                let raw = (lcg_next(&mut state) >> 11) as f32 / (1u64 << 53) as f32; // [0, 1)
                (raw - 0.5) * 2.0 * scale
            })
            .collect();
        ArrayD::from_shape_vec(IxDyn(shape), data).expect("array shape must match data length")
    }

    /// Build a GPT-2 MLP with identical deterministic weights on the given device.
    fn build_mlp(device: Device) -> Result<Gpt2MLP> {
        let config = Gpt2Config {
            n_embd: 16,
            n_inner: Some(64),
            n_head: 4,
            ..Default::default()
        };
        let mut mlp = Gpt2MLP::new_with_device(&config, device)?;
        // Small weights (~GPT-2 init scale) keep f32 accumulation error well below tol.
        mlp.c_fc.set_weight(Tensor::F32(small_array(&[64, 16], 0x1111_1111, 0.1)))?;
        mlp.c_fc.set_bias(Tensor::F32(small_array(&[64], 0x2222_2222, 0.1)))?;
        mlp.c_proj.set_weight(Tensor::F32(small_array(&[16, 64], 0x3333_3333, 0.1)))?;
        mlp.c_proj.set_bias(Tensor::F32(small_array(&[16], 0x4444_4444, 0.1)))?;
        Ok(mlp)
    }

    fn assert_close(fused: &Tensor, separate: &Tensor, tol: f32) -> Result<()> {
        let a = fused.data()?;
        let b = separate.data()?;
        assert_eq!(a.len(), b.len(), "output length mismatch");
        let mut max_diff = 0.0f32;
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff <= tol,
                "element {i} differs: fused={x} separate={y} (diff={diff} > tol={tol})"
            );
        }
        Ok(())
    }

    #[test]
    fn test_fused_mlp_matches_separate_ops_2d() -> Result<()> {
        let mlp_cpu = build_mlp(Device::CPU)?;
        let mlp_metal = build_mlp(Device::Metal(0))?;

        // 2D input: [seq, n_embd].
        let input = Tensor::F32(small_array(&[8, 16], 0x00AB_CDEF, 1.0));

        // The fused Metal kernel must actually engage on the Metal device ...
        assert!(
            mlp_metal.try_fused_c_fc_gelu(&input)?.is_some(),
            "fused matmul+bias+GELU Metal path should engage (GELU + bias on Metal device)"
        );
        // ... and must NOT engage for the CPU reference (separate ops path).
        assert!(
            mlp_cpu.try_fused_c_fc_gelu(&input)?.is_none(),
            "CPU MLP must use the separate (non-fused) ops path"
        );

        let out_metal = mlp_metal.forward(input.clone())?; // fused single-kernel path
        let out_cpu = mlp_cpu.forward(input)?; // separate matmul -> bias -> GELU
        assert_eq!(out_metal.shape(), vec![8, 16]);
        assert_close(&out_metal, &out_cpu, 1e-3)?;
        Ok(())
    }

    #[test]
    fn test_fused_mlp_matches_separate_ops_3d() -> Result<()> {
        let mlp_cpu = build_mlp(Device::CPU)?;
        let mlp_metal = build_mlp(Device::Metal(0))?;

        // 3D input: [batch, seq, n_embd].
        let input = Tensor::F32(small_array(&[2, 5, 16], 0x0012_3456, 1.0));

        assert!(
            mlp_metal.try_fused_c_fc_gelu(&input)?.is_some(),
            "fused matmul+bias+GELU Metal path should engage for 3D input"
        );

        let out_metal = mlp_metal.forward(input.clone())?;
        let out_cpu = mlp_cpu.forward(input)?;
        assert_eq!(out_metal.shape(), vec![2, 5, 16]);
        assert_close(&out_metal, &out_cpu, 1e-3)?;
        Ok(())
    }
}
