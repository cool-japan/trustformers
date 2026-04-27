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
use scirs2_core::ndarray::{concatenate, s, Array1, Array2, Array3, Array4, Axis, Ix3};
use trustformers_core::device::Device;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::layers::{
    attention::MultiHeadAttention, embedding::Embedding, feedforward::FeedForward,
    layernorm::LayerNorm, linear::Linear,
};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Config, Layer};

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
/// Prepends a [CLS] token and, when `use_distillation_token` is set, a
/// distillation token to the sequence of patch embeddings, then adds learned
/// 1-D positional embeddings.
///
/// Output shape: `(B, seq_len, hidden_size)` where
/// `seq_len = num_patches + 1` (or `+ 2` with distillation token).
#[derive(Debug, Clone)]
pub struct DeiTEmbeddings {
    pub patch_embeddings: DeiTPatchEmbedding,
    pub position_embeddings: Embedding,
    /// Learnable [CLS] token (shape: `hidden_size`).
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
        let cls_broadcast = Array3::from_shape_fn((batch_size, 1, hidden_size), |(_, _, k)| {
            self.cls_token[k]
        });

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
            embeddings
                .slice_mut(s![b, .., ..])
                .zip_mut_with(&pos_array, |a, &p| *a += p);
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

        let attn_out = if self.dropout > 0.0 {
            attn_out * (1.0 - self.dropout)
        } else {
            attn_out
        };

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
        Array4::zeros((batch, config.image_size, config.image_size, config.num_channels))
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
        assert_eq!(cfg.seq_length(), cfg.num_patches() + 2, "with distillation: num_patches + 2");
    }

    #[test]
    fn test_config_seq_length_without_distillation_token() {
        let mut cfg = tiny_config();
        cfg.use_distillation_token = false;
        assert_eq!(cfg.seq_length(), cfg.num_patches() + 1, "without distillation: num_patches + 1");
    }

    #[test]
    fn test_tiny_config_hidden_size() {
        let cfg = DeiTConfig::deit_tiny_patch16_224();
        assert_eq!(cfg.hidden_size, 192, "DeiT-Tiny hidden_size should be 192");
    }

    #[test]
    fn test_tiny_config_uses_distillation_token() {
        let cfg = DeiTConfig::deit_tiny_patch16_224();
        assert!(cfg.use_distillation_token, "DeiT-Tiny should use distillation token");
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
        assert_eq!(num_patches, cfg.num_patches(), "num_patches must equal config num_patches");
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
        assert_eq!(seq, cfg.num_patches() + 2, "seq includes CLS + DIST + patches");
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
        assert_eq!(seq, cfg.num_patches() + 1, "seq includes CLS + patches only");
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
        assert_eq!(seq, cfg.seq_length(), "output seq_len must match config seq_length");
        assert_eq!(hidden, cfg.hidden_size, "output hidden_size must match config");
    }

    #[test]
    fn test_get_cls_output_shape() {
        let cfg = tiny_config();
        let model = DeiTModel::new(cfg.clone()).expect("model creation should succeed");
        let images = make_images(2, &cfg);
        let cls = model.get_cls_output(&images).expect("get_cls_output should succeed");
        let (batch, hidden) = cls.dim();
        assert_eq!(batch, 2, "batch preserved in CLS output");
        assert_eq!(hidden, cfg.hidden_size, "CLS output dim must equal hidden_size");
    }

    #[test]
    fn test_get_distillation_output_shape() {
        let cfg = tiny_config(); // use_distillation_token = true
        let model = DeiTModel::new(cfg.clone()).expect("model creation should succeed");
        let images = make_images(1, &cfg);
        let dist = model.get_distillation_output(&images)
            .expect("get_distillation_output should succeed");
        let (batch, hidden) = dist.dim();
        assert_eq!(batch, 1, "batch preserved in distillation output");
        assert_eq!(hidden, cfg.hidden_size, "distillation output dim must equal hidden_size");
    }

    #[test]
    fn test_get_distillation_output_fails_without_token() {
        let mut cfg = tiny_config();
        cfg.use_distillation_token = false;
        let model = DeiTModel::new(cfg).expect("model creation should succeed");
        let images = make_images(1, &DeiTConfig::deit_tiny_patch16_224());
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
