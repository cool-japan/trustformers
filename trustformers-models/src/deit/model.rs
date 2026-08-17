//! DeiT base model implementation.
//!
//! DeiT (Data-efficient Image Transformers) extends ViT with a distillation token.
//! During training, both the CLS and distillation tokens receive separate supervision
//! signals (ground-truth labels and teacher soft predictions, respectively).
//! During inference, the logits from both heads are averaged.
//!
//! Architecture:
//! ```text
//! Image → PatchEmbedding → [CLS; DIST; patches] + pos_embed → TransformerEncoder → LayerNorm
//!                                                                       ↓
//!                                                              (cls_out, dist_out, hidden_states)
//! ```

use crate::deit::config::DeiTConfig;
use crate::weight_loading::binding::{bind_embedding, take_norm_bias, take_norm_weight};
use crate::weight_loading::checkpoint::{Checkpoint, LoadReport, UnusedTensors};
use scirs2_core::ndarray::{concatenate, s, Array1, Array2, Array3, Array4, Axis, Ix3};
use trustformers_core::device::Device;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::layers::{
    attention::MultiHeadAttention, embedding::Embedding, feedforward::FeedForward,
    layernorm::LayerNorm, linear::Linear,
};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Config, Layer, Model};

// ─────────────────────────────────────────────────────────────────────────────
// Patch Embedding
// ─────────────────────────────────────────────────────────────────────────────

/// Projects image patches into the hidden embedding space.
///
/// Splits an input image of shape `(B, H, W, C)` into non-overlapping patches
/// of size `patch_size × patch_size`, flattens each patch, and projects it with
/// a learned linear projection to `hidden_size`.
///
/// Output shape: `(B, num_patches, hidden_size)`.
#[derive(Debug, Clone)]
pub struct DeiTPatchEmbedding {
    pub projection: Linear,
    pub patch_size: usize,
    pub num_channels: usize,
    pub hidden_size: usize,
    device: Device,
}

impl DeiTPatchEmbedding {
    /// Construct with default CPU device.
    pub fn new(config: &DeiTConfig) -> Self {
        Self::new_with_device(config, Device::CPU)
    }

    /// Construct with the specified device.
    pub fn new_with_device(config: &DeiTConfig, device: Device) -> Self {
        let input_size = config.patch_size * config.patch_size * config.num_channels;
        Self {
            projection: Linear::new_with_device(input_size, config.hidden_size, true, device),
            patch_size: config.patch_size,
            num_channels: config.num_channels,
            hidden_size: config.hidden_size,
            device,
        }
    }

    /// Active device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Extract and embed patches from a batch of images.
    ///
    /// # Arguments
    ///
    /// * `images` — shape `(batch, height, width, channels)`.
    ///
    /// # Returns
    ///
    /// Tensor of shape `(batch, num_patches, hidden_size)`.
    pub fn forward(&self, images: &Array4<f32>) -> Result<Array3<f32>> {
        let (batch_size, height, width, channels) = images.dim();

        if height % self.patch_size != 0 || width % self.patch_size != 0 {
            return Err(TrustformersError::invalid_input_simple(format!(
                "Image size {}x{} is not divisible by patch size {}",
                height, width, self.patch_size
            )));
        }

        if channels != self.num_channels {
            return Err(TrustformersError::invalid_input_simple(format!(
                "Expected {} channels, got {}",
                self.num_channels, channels
            )));
        }

        let num_patches_h = height / self.patch_size;
        let num_patches_w = width / self.patch_size;
        let num_patches = num_patches_h * num_patches_w;
        let patch_flat = self.patch_size * self.patch_size * channels;

        let mut patches = Array3::<f32>::zeros((batch_size, num_patches, patch_flat));

        for b in 0..batch_size {
            let mut patch_idx = 0usize;
            for i in 0..num_patches_h {
                for j in 0..num_patches_w {
                    let sh = i * self.patch_size;
                    let sw = j * self.patch_size;
                    let patch = images.slice(s![
                        b,
                        sh..sh + self.patch_size,
                        sw..sw + self.patch_size,
                        ..
                    ]);
                    let flat: Array1<f32> = patch.iter().cloned().collect();
                    patches.slice_mut(s![b, patch_idx, ..]).assign(&flat);
                    patch_idx += 1;
                }
            }
        }

        let patches_tensor = Tensor::F32(patches.into_dyn());
        match self.projection.forward(patches_tensor)? {
            Tensor::F32(result) => Ok(result
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?),
            _ => Err(TrustformersError::invalid_input_simple(
                "Expected F32 tensor from patch projection".to_string(),
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DeiT Embeddings  (CLS + distillation + patch + positional)
// ─────────────────────────────────────────────────────────────────────────────

/// Full embedding layer for DeiT.
///
/// Prepends a `[CLS]` token and, when `use_distillation_token` is set, a
/// distillation token to the sequence of patch embeddings, then adds learned
/// 1-D positional embeddings.
///
/// Output shape: `(B, seq_len, hidden_size)` where
/// `seq_len = num_patches + 1` (or `+ 2` with distillation token).
#[derive(Debug, Clone)]
pub struct DeiTEmbeddings {
    pub patch_embeddings: DeiTPatchEmbedding,
    pub position_embeddings: Embedding,
    /// Learnable `[CLS]` token (shape: `hidden_size`).
    pub cls_token: Array1<f32>,
    /// Learnable distillation token (shape: `hidden_size`), present when
    /// `config.use_distillation_token` is true.
    pub distillation_token: Option<Array1<f32>>,
    pub dropout: f32,
    pub config: DeiTConfig,
    device: Device,
}

impl DeiTEmbeddings {
    /// Construct with default CPU device.
    pub fn new(config: &DeiTConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    /// Construct with the specified device.
    pub fn new_with_device(config: &DeiTConfig, device: Device) -> Result<Self> {
        let patch_embeddings = DeiTPatchEmbedding::new_with_device(config, device);
        let position_embeddings =
            Embedding::new_with_device(config.seq_length(), config.hidden_size, None, device)?;

        let cls_token = Array1::zeros(config.hidden_size);
        let distillation_token = if config.use_distillation_token {
            Some(Array1::zeros(config.hidden_size))
        } else {
            None
        };

        Ok(Self {
            patch_embeddings,
            position_embeddings,
            cls_token,
            distillation_token,
            dropout: config.hidden_dropout_prob,
            config: config.clone(),
            device,
        })
    }

    /// Active device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Run the embedding forward pass.
    ///
    /// # Arguments
    ///
    /// * `images` — shape `(batch, height, width, channels)`.
    ///
    /// # Returns
    ///
    /// Embedded sequence of shape `(batch, seq_len, hidden_size)`.
    pub fn forward(&self, images: &Array4<f32>) -> Result<Array3<f32>> {
        let batch_size = images.dim().0;
        let hidden_size = self.config.hidden_size;

        // (B, num_patches, H)
        let patch_emb = self.patch_embeddings.forward(images)?;

        // Expand [CLS] token: (B, 1, H)
        let cls_broadcast =
            Array3::from_shape_fn((batch_size, 1, hidden_size), |(_, _, k)| self.cls_token[k]);

        // Concatenate [CLS | patches] → (B, 1+num_patches, H)
        let embeddings = concatenate![Axis(1), cls_broadcast, patch_emb];

        // Optionally prepend distillation token after CLS → [CLS | DIST | patches]
        let embeddings = if let Some(ref dist_tok) = self.distillation_token {
            let dist_broadcast =
                Array3::from_shape_fn((batch_size, 1, hidden_size), |(_, _, k)| dist_tok[k]);
            // Insert distillation token at position 1
            let cls_part = embeddings.slice(s![.., 0..1, ..]).to_owned();
            let patches_part = embeddings.slice(s![.., 1.., ..]).to_owned();
            concatenate![Axis(1), cls_part, dist_broadcast, patches_part]
        } else {
            embeddings
        };

        // Add positional embeddings
        let seq_len = embeddings.dim().1;
        let pos_ids: Vec<u32> = (0..seq_len as u32).collect();
        let pos_tensor = self.position_embeddings.forward(pos_ids)?;
        let pos_array = match pos_tensor {
            Tensor::F32(arr) => arr,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 tensor from position embeddings".to_string(),
                ))
            },
        };

        let mut embeddings = embeddings;
        for b in 0..batch_size {
            embeddings.slice_mut(s![b, .., ..]).zip_mut_with(&pos_array, |a, &p| *a += p);
        }

        // Dropout (training-mode approximation)
        if self.dropout > 0.0 {
            embeddings *= 1.0 - self.dropout;
        }

        Ok(embeddings)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Attention block
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-head self-attention sub-layer with pre-norm residual connection.
#[derive(Debug, Clone)]
pub struct DeiTAttention {
    pub attention: MultiHeadAttention,
    pub layer_norm: LayerNorm,
    pub dropout: f32,
    device: Device,
}

impl DeiTAttention {
    pub fn new(config: &DeiTConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &DeiTConfig, device: Device) -> Result<Self> {
        Ok(Self {
            attention: MultiHeadAttention::new_with_device(
                config.hidden_size,
                config.num_attention_heads,
                config.attention_probs_dropout_prob,
                true,
                device,
            )?,
            layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    /// Active device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Pre-norm: LayerNorm → Attention → Dropout → Residual.
    pub fn forward(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        let hs_tensor = Tensor::F32(hidden_states.clone().into_dyn());
        let normed = match self.layer_norm.forward(hs_tensor)? {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 from LayerNorm".to_string(),
                ))
            },
        };

        let attn_tensor = Tensor::F32(normed.into_dyn());
        let attn_out = match self.attention.forward(attn_tensor)? {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 from MultiHeadAttention".to_string(),
                ))
            },
        };

        let attn_out = if self.dropout > 0.0 { attn_out * (1.0 - self.dropout) } else { attn_out };

        Ok(hidden_states + &attn_out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MLP block
// ─────────────────────────────────────────────────────────────────────────────

/// Feed-forward MLP sub-layer with pre-norm residual connection.
#[derive(Debug, Clone)]
pub struct DeiTMLP {
    pub feed_forward: FeedForward,
    pub layer_norm: LayerNorm,
    pub dropout: f32,
    device: Device,
}

impl DeiTMLP {
    pub fn new(config: &DeiTConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &DeiTConfig, device: Device) -> Result<Self> {
        Ok(Self {
            feed_forward: FeedForward::new_with_device(
                config.hidden_size,
                config.intermediate_size,
                0.0,
                device,
            ),
            layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    /// Active device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Pre-norm: LayerNorm → FFN → Dropout → Residual.
    pub fn forward(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        let hs_tensor = Tensor::F32(hidden_states.clone().into_dyn());
        let normed = match self.layer_norm.forward(hs_tensor)? {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 from LayerNorm".to_string(),
                ))
            },
        };

        let ff_tensor = Tensor::F32(normed.into_dyn());
        let ff_out = match self.feed_forward.forward(ff_tensor)? {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 from FeedForward".to_string(),
                ))
            },
        };

        let ff_out = if self.dropout > 0.0 { ff_out * (1.0 - self.dropout) } else { ff_out };

        Ok(hidden_states + &ff_out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Transformer Layer
// ─────────────────────────────────────────────────────────────────────────────

/// A single DeiT transformer block: Attention + MLP with pre-norm residuals.
#[derive(Debug, Clone)]
pub struct DeiTLayer {
    pub attention: DeiTAttention,
    pub mlp: DeiTMLP,
    device: Device,
}

impl DeiTLayer {
    pub fn new(config: &DeiTConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &DeiTConfig, device: Device) -> Result<Self> {
        Ok(Self {
            attention: DeiTAttention::new_with_device(config, device)?,
            mlp: DeiTMLP::new_with_device(config, device)?,
            device,
        })
    }

    /// Active device.
    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        let after_attn = self.attention.forward(hidden_states)?;
        self.mlp.forward(&after_attn)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Encoder (stack of layers)
// ─────────────────────────────────────────────────────────────────────────────

/// Stack of `num_hidden_layers` DeiT transformer blocks.
#[derive(Debug, Clone)]
pub struct DeiTEncoder {
    pub layers: Vec<DeiTLayer>,
    device: Device,
}

impl DeiTEncoder {
    pub fn new(config: &DeiTConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &DeiTConfig, device: Device) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|_| DeiTLayer::new_with_device(config, device))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { layers, device })
    }

    /// Active device.
    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        let mut hs = hidden_states.clone();
        for layer in &self.layers {
            hs = layer.forward(&hs)?;
        }
        Ok(hs)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// DeiT Base Model
// ─────────────────────────────────────────────────────────────────────────────

/// DeiT base model (no task head).
///
/// Returns the full hidden-state sequence of shape `(B, seq_len, hidden_size)`.
/// The CLS representation is at index 0; the distillation representation is at
/// index 1 (when `use_distillation_token` is true).
///
/// # Example
///
/// ```rust,no_run
/// use trustformers_models::deit::{DeiTConfig, DeiTModel};
/// use scirs2_core::ndarray::Array4;
///
/// let config = DeiTConfig::deit_tiny_patch16_224();
/// let model = DeiTModel::new(config).expect("valid config constructs DeiT model");
/// let images = Array4::<f32>::zeros((1, 224, 224, 3));
/// let output = model.forward(&images).expect("well-formed input tensor succeeds");
/// // output shape: (1, 198, 192)  — 196 patches + CLS + distill
/// ```
#[derive(Debug, Clone)]
pub struct DeiTModel {
    pub embeddings: DeiTEmbeddings,
    pub encoder: DeiTEncoder,
    pub layer_norm: LayerNorm,
    pub config: DeiTConfig,
    device: Device,
}

impl DeiTModel {
    /// Create a new DeiT model on the CPU.
    pub fn new(config: DeiTConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    /// Create a new DeiT model on the given device.
    pub fn new_with_device(config: DeiTConfig, device: Device) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            embeddings: DeiTEmbeddings::new_with_device(&config, device)?,
            encoder: DeiTEncoder::new_with_device(&config, device)?,
            layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            config,
            device,
        })
    }

    /// Active device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Run the forward pass.
    ///
    /// Returns the post-norm hidden states: shape `(B, seq_len, hidden_size)`.
    pub fn forward(&self, images: &Array4<f32>) -> Result<Array3<f32>> {
        let emb = self.embeddings.forward(images)?;
        let enc = self.encoder.forward(&emb)?;

        let enc_tensor = Tensor::F32(enc.into_dyn());
        match self.layer_norm.forward(enc_tensor)? {
            Tensor::F32(result) => Ok(result
                .into_dimensionality::<Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?),
            _ => Err(TrustformersError::invalid_input_simple(
                "Expected F32 from final LayerNorm".to_string(),
            )),
        }
    }
}

impl Model for DeiTModel {
    type Config = DeiTConfig;
    /// `(batch, height, width, channels)` pixel values.
    type Input = Array4<f32>;
    /// `(batch, seq_len, hidden_size)` post-norm hidden states.
    type Output = Array3<f32>;

    fn forward(&self, images: Self::Input) -> Result<Self::Output> {
        DeiTModel::forward(self, &images)
    }

    /// Load a HuggingFace DeiT checkpoint (safetensors or `torch.save`).
    ///
    /// See [`DeiTModel::load_from_checkpoint`] for the name map.
    fn load_pretrained(&mut self, reader: &mut dyn std::io::Read) -> Result<()> {
        self.load_pretrained_report(reader).map(|_| ())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        self.parameter_count()
    }
}

impl DeiTModel {
    /// Checkpoint namespaces a DeiT backbone legitimately does not consume.
    pub(crate) const ALLOWED_UNUSED_PREFIXES: &'static [&'static str] = &[
        "classifier.",
        "cls_classifier.",
        "distillation_classifier.",
        "pooler.",
    ];

    /// Total learnable parameter count.
    pub fn parameter_count(&self) -> usize {
        let hidden = self.config.hidden_size;
        let special = if self.config.use_distillation_token { 2 } else { 1 };
        let embeddings = self.embeddings.patch_embeddings.projection.parameter_count()
            + self.config.seq_length() * hidden
            + special * hidden;
        let per_layer: usize = self
            .encoder
            .layers
            .iter()
            .map(|layer| {
                layer.attention.attention.parameter_count()
                    + layer.attention.layer_norm.parameter_count()
                    + layer.mlp.feed_forward.parameter_count()
                    + layer.mlp.layer_norm.parameter_count()
            })
            .sum();
        embeddings + per_layer + self.layer_norm.parameter_count()
    }

    /// Load pretrained weights and report exactly what was bound.
    ///
    /// # Errors
    ///
    /// See [`DeiTModel::load_from_checkpoint`].
    pub fn load_pretrained_report(&mut self, reader: &mut dyn std::io::Read) -> Result<LoadReport> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        self.load_from_checkpoint(&checkpoint)
    }

    /// Bind an already-parsed checkpoint into this model.
    ///
    /// DeiT is a ViT plus a second special token. Its checkpoints therefore
    /// carry ViT's tensor names with **two** prepended tokens rather than one:
    /// `embeddings.cls_token` *and* `embeddings.distillation_token`, and a
    /// position table one row longer to match. Binding a DeiT checkpoint with a
    /// ViT name map would silently leave the distillation token randomly
    /// initialised and mis-size the position embeddings by one row.
    ///
    /// The patch projection is stored as a `Conv2d` kernel
    /// `[hidden, channels, patch, patch]`; it is flattened to the
    /// `[hidden, channels * patch * patch]` matrix this implementation applies,
    /// which is the same linear map, not a reinterpretation.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint does not look like a DeiT checkpoint, when any
    /// tensor has the wrong shape, when a parameter is missing, or when the
    /// checkpoint carries weights this architecture does not recognise.
    pub fn load_from_checkpoint(&mut self, checkpoint: &Checkpoint) -> Result<LoadReport> {
        let prefix = checkpoint.detect_prefix(
            &["", "deit."],
            "embeddings.patch_embeddings.projection.weight",
        )?;
        let mut binder = checkpoint.binder(&prefix);

        let config = self.config.clone();
        let hidden = config.hidden_size;
        let patch = config.patch_size;
        let channels = config.num_channels;
        let patch_dim = channels * patch * patch;

        // Patch projection: accept either the flattened matrix or the Conv2d
        // kernel HuggingFace actually ships.
        let projection_name = binder.qualified("embeddings.patch_embeddings.projection.weight");
        match checkpoint.get(&projection_name) {
            Some(tensor) => {
                binder.mark_consumed(&projection_name);
                let shape = tensor.shape();
                let flattened = if shape == vec![hidden, patch_dim] {
                    tensor.clone()
                } else if shape == vec![hidden, channels, patch, patch] {
                    tensor.reshape(&[hidden, patch_dim])?
                } else {
                    return Err(TrustformersError::shape_error(format!(
                        "patch projection has shape {shape:?} but this model expects \
                         [{hidden}, {patch_dim}] or [{hidden}, {channels}, {patch}, {patch}]"
                    )));
                };
                self.embeddings.patch_embeddings.projection.set_weight(flattened)?;
            },
            None => {
                return Err(TrustformersError::weight_load_error(
                    "checkpoint carries no embeddings.patch_embeddings.projection.weight"
                        .to_string(),
                ))
            },
        }
        if let Some(bias) =
            binder.take_shaped("embeddings.patch_embeddings.projection.bias", &[hidden])?
        {
            self.embeddings.patch_embeddings.projection.set_bias(bias)?;
        }

        // Position table: one row per patch plus the special tokens.
        bind_embedding(
            &mut binder,
            "embeddings.position_embeddings",
            config.seq_length(),
            hidden,
            &mut self.embeddings.position_embeddings,
        )?;

        // The CLS token, and DeiT's extra distillation token.
        let take_token = |binder: &mut crate::weight_loading::checkpoint::WeightBinder<'_>,
                          name: &str|
         -> Result<Option<Array1<f32>>> {
            // HuggingFace stores these as [1, 1, hidden]; accept the bare
            // [hidden] spelling too.
            let qualified = binder.qualified(name);
            let Some(tensor) = binder.take(name) else {
                return Ok(None);
            };
            let count: usize = tensor.shape().iter().product();
            if count != hidden {
                return Err(TrustformersError::shape_error(format!(
                    "{qualified} holds {count} value(s) but the hidden size is {hidden}"
                )));
            }
            let values = tensor.data().map_err(|e| {
                TrustformersError::tensor_op_error(
                    "failed to read a token embedding",
                    &e.to_string(),
                )
            })?;
            Ok(Some(Array1::from_vec(values)))
        };

        if let Some(token) = take_token(&mut binder, "embeddings.cls_token")? {
            self.embeddings.cls_token = token;
        }
        if self.config.use_distillation_token {
            if let Some(token) = take_token(&mut binder, "embeddings.distillation_token")? {
                self.embeddings.distillation_token = Some(token);
            }
        }

        for (index, layer) in self.encoder.layers.iter_mut().enumerate() {
            let base = format!("encoder.layer.{index}");

            // Pre-attention norm, then the QKV/out projections.
            let attn_norm = format!("{base}.layernorm_before");
            if let Some(w) = take_norm_weight(&mut binder, &attn_norm, hidden)? {
                layer.attention.layer_norm.set_weight(w)?;
            }
            if let Some(b) = take_norm_bias(&mut binder, &attn_norm, hidden)? {
                layer.attention.layer_norm.set_bias(b)?;
            }

            let bias = config.qkv_bias;
            for (name, setter) in [("query", 0usize), ("key", 1), ("value", 2)] {
                let full = format!("{base}.attention.attention.{name}");
                if let Some(w) = binder.take_shaped(&format!("{full}.weight"), &[hidden, hidden])? {
                    match setter {
                        0 => layer.attention.attention.set_query_weight(w)?,
                        1 => layer.attention.attention.set_key_weight(w)?,
                        _ => layer.attention.attention.set_value_weight(w)?,
                    }
                }
                if bias {
                    if let Some(b) = binder.take_shaped(&format!("{full}.bias"), &[hidden])? {
                        match setter {
                            0 => layer.attention.attention.set_query_bias(b)?,
                            1 => layer.attention.attention.set_key_bias(b)?,
                            _ => layer.attention.attention.set_value_bias(b)?,
                        }
                    }
                }
            }
            let out = format!("{base}.attention.output.dense");
            if let Some(w) = binder.take_shaped(&format!("{out}.weight"), &[hidden, hidden])? {
                layer.attention.attention.set_out_proj_weight(w)?;
            }
            if let Some(b) = binder.take_shaped(&format!("{out}.bias"), &[hidden])? {
                layer.attention.attention.set_out_proj_bias(b)?;
            }

            // Pre-MLP norm, then the two feed-forward projections.
            let mlp_norm = format!("{base}.layernorm_after");
            if let Some(w) = take_norm_weight(&mut binder, &mlp_norm, hidden)? {
                layer.mlp.layer_norm.set_weight(w)?;
            }
            if let Some(b) = take_norm_bias(&mut binder, &mlp_norm, hidden)? {
                layer.mlp.layer_norm.set_bias(b)?;
            }
            let intermediate = config.intermediate_size;
            if let Some(w) = binder.take_shaped(
                &format!("{base}.intermediate.dense.weight"),
                &[intermediate, hidden],
            )? {
                layer.mlp.feed_forward.set_dense_weight(w)?;
            }
            if let Some(b) =
                binder.take_shaped(&format!("{base}.intermediate.dense.bias"), &[intermediate])?
            {
                layer.mlp.feed_forward.set_dense_bias(b)?;
            }
            if let Some(w) = binder.take_shaped(
                &format!("{base}.output.dense.weight"),
                &[hidden, intermediate],
            )? {
                layer.mlp.feed_forward.set_output_weight(w)?;
            }
            if let Some(b) = binder.take_shaped(&format!("{base}.output.dense.bias"), &[hidden])? {
                layer.mlp.feed_forward.set_output_bias(b)?;
            }
        }

        if let Some(w) = take_norm_weight(&mut binder, "layernorm", hidden)? {
            self.layer_norm.set_weight(w)?;
        }
        if let Some(b) = take_norm_bias(&mut binder, "layernorm", hidden)? {
            self.layer_norm.set_bias(b)?;
        }

        binder.finish(UnusedTensors::new(Self::ALLOWED_UNUSED_PREFIXES, &[]))
    }
}

impl DeiTModel {
    /// Extract the CLS token representation: shape `(B, hidden_size)`.
    pub fn get_cls_output(&self, images: &Array4<f32>) -> Result<Array2<f32>> {
        let output = self.forward(images)?;
        Ok(output.slice(s![.., 0, ..]).to_owned())
    }

    /// Extract the distillation token representation: shape `(B, hidden_size)`.
    ///
    /// Returns an error if the model was built without a distillation token.
    pub fn get_distillation_output(&self, images: &Array4<f32>) -> Result<Array2<f32>> {
        if !self.config.use_distillation_token {
            return Err(TrustformersError::invalid_input_simple(
                "This DeiT model was built without a distillation token".to_string(),
            ));
        }
        let output = self.forward(images)?;
        Ok(output.slice(s![.., 1, ..]).to_owned())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deit::config::DeiTConfig;
    use scirs2_core::ndarray::Array4;
    use trustformers_core::traits::Config;

    fn tiny_config() -> DeiTConfig {
        DeiTConfig {
            image_size: 8,
            patch_size: 4,
            num_channels: 3,
            hidden_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 32,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            num_labels: 10,
            qkv_bias: true,
            layer_norm_eps: 1e-6,
            use_distillation_token: true,
            model_type: "deit".to_string(),
        }
    }

    fn make_images(batch: usize, config: &DeiTConfig) -> Array4<f32> {
        Array4::zeros((
            batch,
            config.image_size,
            config.image_size,
            config.num_channels,
        ))
    }

    // ── Config tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_config_validate_ok() {
        tiny_config().validate().expect("tiny_config should be valid");
    }

    #[test]
    fn test_config_num_patches() {
        let cfg = tiny_config(); // 8x8 image, 4x4 patches → 2x2 = 4 patches
        assert_eq!(cfg.num_patches(), 4, "8x8 image / 4 patch_size = 4 patches");
    }

    #[test]
    fn test_config_seq_length_with_distillation_token() {
        let cfg = tiny_config(); // use_distillation_token = true
                                 // seq_length = num_patches + CLS + distillation = 4 + 2 = 6
        assert_eq!(
            cfg.seq_length(),
            cfg.num_patches() + 2,
            "with distillation: num_patches + 2"
        );
    }

    #[test]
    fn test_config_seq_length_without_distillation_token() {
        let mut cfg = tiny_config();
        cfg.use_distillation_token = false;
        assert_eq!(
            cfg.seq_length(),
            cfg.num_patches() + 1,
            "without distillation: num_patches + 1"
        );
    }

    #[test]
    fn test_tiny_config_hidden_size() {
        let cfg = DeiTConfig::deit_tiny_patch16_224();
        assert_eq!(cfg.hidden_size, 192, "DeiT-Tiny hidden_size should be 192");
    }

    #[test]
    fn test_tiny_config_uses_distillation_token() {
        let cfg = DeiTConfig::deit_tiny_patch16_224();
        assert!(
            cfg.use_distillation_token,
            "DeiT-Tiny should use distillation token"
        );
    }

    // ── PatchEmbedding tests ───────────────────────────────────────────────────

    #[test]
    fn test_patch_embedding_output_shape() {
        let cfg = tiny_config();
        let pe = DeiTPatchEmbedding::new(&cfg);
        let images = make_images(1, &cfg);
        let output = pe.forward(&images).expect("patch embedding should succeed");
        let (batch, num_patches, hidden) = output.dim();
        assert_eq!(batch, 1, "batch preserved");
        assert_eq!(
            num_patches,
            cfg.num_patches(),
            "num_patches must equal config num_patches"
        );
        assert_eq!(hidden, cfg.hidden_size, "hidden_size must match config");
    }

    #[test]
    fn test_patch_embedding_non_divisible_image_fails() {
        let cfg = tiny_config(); // patch_size = 4
        let pe = DeiTPatchEmbedding::new(&cfg);
        // 6x6 image is not divisible by patch_size=4
        let bad_images = Array4::<f32>::zeros((1, 6, 6, cfg.num_channels));
        assert!(
            pe.forward(&bad_images).is_err(),
            "non-divisible image size should fail"
        );
    }

    // ── DeiTEmbeddings tests ───────────────────────────────────────────────────

    #[test]
    fn test_embeddings_cls_and_distillation_tokens() {
        let cfg = tiny_config();
        let emb = DeiTEmbeddings::new(&cfg).expect("embeddings creation should succeed");
        let images = make_images(1, &cfg);
        let output = emb.forward(&images).expect("embeddings forward should succeed");
        let (_batch, seq, _hidden) = output.dim();
        // With distillation token: seq = num_patches + 2
        assert_eq!(
            seq,
            cfg.num_patches() + 2,
            "seq includes CLS + DIST + patches"
        );
    }

    #[test]
    fn test_embeddings_without_distillation_token() {
        let mut cfg = tiny_config();
        cfg.use_distillation_token = false;
        let emb = DeiTEmbeddings::new(&cfg).expect("embeddings creation should succeed");
        let images = make_images(1, &cfg);
        let output = emb.forward(&images).expect("embeddings forward should succeed");
        let (_batch, seq, _hidden) = output.dim();
        // Without distillation: seq = num_patches + 1 (CLS only)
        assert_eq!(
            seq,
            cfg.num_patches() + 1,
            "seq includes CLS + patches only"
        );
    }

    // ── DeiTModel tests ────────────────────────────────────────────────────────

    #[test]
    fn test_model_creation() {
        let cfg = tiny_config();
        DeiTModel::new(cfg).expect("DeiTModel creation should succeed");
    }

    #[test]
    fn test_model_forward_output_shape() {
        let cfg = tiny_config();
        let model = DeiTModel::new(cfg.clone()).expect("model creation should succeed");
        let images = make_images(1, &cfg);
        let output = model.forward(&images).expect("model forward should succeed");
        let (batch, seq, hidden) = output.dim();
        assert_eq!(batch, 1, "batch preserved");
        assert_eq!(
            seq,
            cfg.seq_length(),
            "output seq_len must match config seq_length"
        );
        assert_eq!(
            hidden, cfg.hidden_size,
            "output hidden_size must match config"
        );
    }

    #[test]
    fn test_get_cls_output_shape() {
        let cfg = tiny_config();
        let model = DeiTModel::new(cfg.clone()).expect("model creation should succeed");
        let images = make_images(2, &cfg);
        let cls = model.get_cls_output(&images).expect("get_cls_output should succeed");
        let (batch, hidden) = cls.dim();
        assert_eq!(batch, 2, "batch preserved in CLS output");
        assert_eq!(
            hidden, cfg.hidden_size,
            "CLS output dim must equal hidden_size"
        );
    }

    #[test]
    fn test_get_distillation_output_shape() {
        let cfg = tiny_config(); // use_distillation_token = true
        let model = DeiTModel::new(cfg.clone()).expect("model creation should succeed");
        let images = make_images(1, &cfg);
        let dist = model
            .get_distillation_output(&images)
            .expect("get_distillation_output should succeed");
        let (batch, hidden) = dist.dim();
        assert_eq!(batch, 1, "batch preserved in distillation output");
        assert_eq!(
            hidden, cfg.hidden_size,
            "distillation output dim must equal hidden_size"
        );
    }

    #[test]
    fn test_get_distillation_output_fails_without_token() {
        let mut cfg = tiny_config();
        cfg.use_distillation_token = false;
        let model = DeiTModel::new(cfg).expect("model creation should succeed");
        let _images = make_images(1, &DeiTConfig::deit_tiny_patch16_224());
        // Use tiny_config image size
        let small_images = Array4::<f32>::zeros((1, 8, 8, 3));
        assert!(
            model.get_distillation_output(&small_images).is_err(),
            "get_distillation_output without distillation token should fail"
        );
    }

    #[test]
    fn test_dual_token_total_is_patches_plus_two() {
        // DeiT key property: total_tokens = patches + CLS + DIST = n_patches + 2
        let cfg = DeiTConfig::deit_tiny_patch16_224();
        let n_patches = cfg.num_patches();
        assert_eq!(
            cfg.seq_length(),
            n_patches + 2,
            "total tokens = patches + CLS + distillation"
        );
    }
}

#[cfg(test)]
mod loading_tests {
    use super::*;
    use crate::weight_loading::test_support::{build_safetensors, F32Tensor};

    fn loading_config() -> DeiTConfig {
        DeiTConfig {
            image_size: 8,
            patch_size: 4,
            num_channels: 3,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            intermediate_size: 16,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            num_labels: 2,
            qkv_bias: true,
            layer_norm_eps: 1e-6,
            use_distillation_token: true,
            model_type: "deit".to_string(),
        }
    }

    /// Every tensor a DeiT checkpoint of this shape carries, with the patch
    /// projection in HuggingFace's `Conv2d` kernel layout.
    fn deit_tensors(config: &DeiTConfig, prefix: &str) -> Vec<F32Tensor> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let patch = config.patch_size;
        let channels = config.num_channels;
        let mut seed = 0.0f32;
        let mut next = || {
            seed += 1.0;
            seed
        };

        let mut tensors = vec![
            F32Tensor::ramp(
                &format!("{prefix}embeddings.patch_embeddings.projection.weight"),
                &[hidden, channels, patch, patch],
                next(),
            ),
            F32Tensor::ramp(
                &format!("{prefix}embeddings.patch_embeddings.projection.bias"),
                &[hidden],
                next(),
            ),
            F32Tensor::ramp(
                &format!("{prefix}embeddings.position_embeddings.weight"),
                &[config.seq_length(), hidden],
                next(),
            ),
            F32Tensor::ramp(
                &format!("{prefix}embeddings.cls_token"),
                &[1, 1, hidden],
                next(),
            ),
        ];
        if config.use_distillation_token {
            tensors.push(F32Tensor::ramp(
                &format!("{prefix}embeddings.distillation_token"),
                &[1, 1, hidden],
                next(),
            ));
        }

        for layer in 0..config.num_hidden_layers {
            let base = format!("{prefix}encoder.layer.{layer}");
            for norm in ["layernorm_before", "layernorm_after"] {
                tensors.push(F32Tensor::ramp(
                    &format!("{base}.{norm}.weight"),
                    &[hidden],
                    next(),
                ));
                tensors.push(F32Tensor::ramp(
                    &format!("{base}.{norm}.bias"),
                    &[hidden],
                    next(),
                ));
            }
            for projection in ["query", "key", "value"] {
                tensors.push(F32Tensor::ramp(
                    &format!("{base}.attention.attention.{projection}.weight"),
                    &[hidden, hidden],
                    next(),
                ));
                if config.qkv_bias {
                    tensors.push(F32Tensor::ramp(
                        &format!("{base}.attention.attention.{projection}.bias"),
                        &[hidden],
                        next(),
                    ));
                }
            }
            tensors.push(F32Tensor::ramp(
                &format!("{base}.attention.output.dense.weight"),
                &[hidden, hidden],
                next(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{base}.attention.output.dense.bias"),
                &[hidden],
                next(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{base}.intermediate.dense.weight"),
                &[intermediate, hidden],
                next(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{base}.intermediate.dense.bias"),
                &[intermediate],
                next(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{base}.output.dense.weight"),
                &[hidden, intermediate],
                next(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{base}.output.dense.bias"),
                &[hidden],
                next(),
            ));
        }

        tensors.push(F32Tensor::ramp(
            &format!("{prefix}layernorm.weight"),
            &[hidden],
            next(),
        ));
        tensors.push(F32Tensor::ramp(
            &format!("{prefix}layernorm.bias"),
            &[hidden],
            next(),
        ));
        tensors
    }

    /// Regression: DeiT implemented no `Model` trait at all, so there was no
    /// `load_pretrained` to call — not even a handled error path. It now binds
    /// every parameter of a real checkpoint.
    #[test]
    fn load_pretrained_binds_every_parameter() {
        let config = loading_config();
        let tensors = deit_tensors(&config, "deit.");
        let bytes = build_safetensors(&tensors);

        let mut model = DeiTModel::new(config).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a matching checkpoint must load");
        assert!(
            report.is_complete(),
            "every parameter must be bound, missing: {:?}",
            report.missing
        );
        assert!(
            report.unexpected.is_empty(),
            "nothing should be left over: {:?}",
            report.unexpected
        );
    }

    /// DeiT's second special token is what distinguishes it from ViT: a loader
    /// using a ViT name map would leave it randomly initialised.
    #[test]
    fn load_pretrained_binds_the_distillation_token() {
        let config = loading_config();
        let tensors = deit_tensors(&config, "deit.");
        let expected = tensors
            .iter()
            .find(|t| t.name == "deit.embeddings.distillation_token")
            .expect("fixture must carry the distillation token")
            .values
            .clone();
        let bytes = build_safetensors(&tensors);

        let mut model = DeiTModel::new(config).expect("model must build");
        model.load_pretrained(&mut bytes.as_slice()).expect("checkpoint must load");

        let token = model
            .embeddings
            .distillation_token
            .as_ref()
            .expect("the distillation token must survive the load");
        assert_eq!(token.to_vec(), expected);
    }

    /// The patch projection arrives as a 4-D `Conv2d` kernel and must be
    /// flattened into the equivalent matrix, not rejected or reinterpreted.
    #[test]
    fn load_pretrained_flattens_the_conv2d_patch_kernel() {
        let config = loading_config();
        let tensors = deit_tensors(&config, "deit.");
        let expected = tensors
            .iter()
            .find(|t| t.name == "deit.embeddings.patch_embeddings.projection.weight")
            .expect("fixture must carry the projection")
            .values
            .clone();
        let bytes = build_safetensors(&tensors);

        let mut model = DeiTModel::new(config.clone()).expect("model must build");
        model.load_pretrained(&mut bytes.as_slice()).expect("checkpoint must load");

        let weight = model.embeddings.patch_embeddings.projection.weight();
        assert_eq!(
            weight.shape(),
            vec![
                config.hidden_size,
                config.num_channels * config.patch_size * config.patch_size
            ]
        );
        assert_eq!(weight.data().expect("readable"), expected);
    }

    #[test]
    fn load_pretrained_reports_a_missing_parameter() {
        let config = loading_config();
        let mut tensors = deit_tensors(&config, "deit.");
        tensors.retain(|t| t.name != "deit.encoder.layer.1.output.dense.weight");
        let bytes = build_safetensors(&tensors);

        let mut model = DeiTModel::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("an incomplete checkpoint must not load silently");
        assert!(
            err.to_string().contains("encoder.layer.1.output.dense.weight"),
            "the error must name the gap: {err}"
        );
    }

    #[test]
    fn load_pretrained_rejects_a_foreign_tensor() {
        let config = loading_config();
        let mut tensors = deit_tensors(&config, "deit.");
        tensors.push(F32Tensor::ramp(
            "deit.encoder.mystery.weight",
            &[8, 8],
            55.0,
        ));
        let bytes = build_safetensors(&tensors);

        let mut model = DeiTModel::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("an unrecognised weight must fail the load");
        assert!(
            err.to_string().contains("mystery.weight"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn load_pretrained_rejects_bytes_that_are_not_a_checkpoint() {
        let mut model = DeiTModel::new(loading_config()).expect("model must build");
        let garbage = vec![0xABu8; 4096];
        let err = model
            .load_pretrained(&mut garbage.as_slice())
            .expect_err("garbage must not be accepted as weights");
        assert!(
            err.to_string().contains("unrecognised checkpoint container"),
            "unexpected: {err}"
        );
    }
}
