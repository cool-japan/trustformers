//! Task-specific heads for Swin Transformer.

use crate::swin::config::SwinConfig;
use crate::swin::model::SwinModel;
use scirs2_core::ndarray::{Array2, Array4, Ix2};
use std::collections::HashMap;
use trustformers_core::device::Device;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::layers::{layernorm::LayerNorm, linear::Linear};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Layer;

/// Swin Transformer for image classification.
///
/// The final feature vector from [`SwinModel`] (global average pooled,
/// shape `(B, final_dim)`) is passed through a [`LayerNorm`] and then a
/// linear classification head.
///
/// # Example
///
/// ```rust,no_run
/// use trustformers_models::swin::{SwinConfig, SwinForImageClassification};
/// use scirs2_core::ndarray::Array4;
///
/// let config = SwinConfig::swin_tiny_patch4_window7_224();
/// let model = SwinForImageClassification::new(config).expect("valid config constructs model");
/// let images = Array4::<f32>::zeros((1, 224, 224, 3));
/// let logits = model.forward(&images).expect("well-formed input tensor succeeds");
/// assert_eq!(logits.shape(), &[1, 1000]);
/// ```
#[derive(Debug, Clone)]
pub struct SwinForImageClassification {
    pub swin: SwinModel,
    pub norm: LayerNorm,
    pub head: Linear,
    device: Device,
}

impl SwinForImageClassification {
    /// Create on the CPU.
    pub fn new(config: SwinConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    /// Create on the specified device.
    pub fn new_with_device(config: SwinConfig, device: Device) -> Result<Self> {
        let final_dim = config.final_dim();
        let num_labels = config.num_labels;
        let layer_norm_eps = config.layer_norm_eps;

        Ok(Self {
            norm: LayerNorm::new_with_device(vec![final_dim], layer_norm_eps, device)?,
            head: Linear::new_with_device(final_dim, num_labels, true, device),
            swin: SwinModel::new_with_device(config, device)?,
            device,
        })
    }

    /// Active device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Classification forward pass.
    ///
    /// Returns logits of shape `(B, num_labels)`.
    pub fn forward(&self, images: &Array4<f32>) -> Result<Array2<f32>> {
        // (B, final_dim)
        let features = self.swin.forward(images)?;

        // Norm + linear head
        let feat_tensor = Tensor::F32(features.into_dyn());
        let normed = match self.norm.forward(feat_tensor)? {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix2>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 from LayerNorm".to_string(),
                ))
            },
        };

        let head_tensor = Tensor::F32(normed.into_dyn());
        match self.head.forward(head_tensor)? {
            Tensor::F32(arr) => Ok(arr
                .into_dimensionality::<Ix2>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?),
            _ => Err(TrustformersError::invalid_input_simple(
                "Expected F32 from classification head".to_string(),
            )),
        }
    }

    /// Generate a weight-name mapping compatible with HuggingFace Swin checkpoints.
    pub fn weight_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();

        // Patch embedding
        map.insert(
            "swin.embeddings.patch_embeddings.projection.weight".to_string(),
            "swin.patch_embed.projection.weight".to_string(),
        );
        map.insert(
            "swin.embeddings.norm.weight".to_string(),
            "swin.patch_embed.layer_norm.weight".to_string(),
        );

        // Layers
        for (s_idx, depth) in self.swin.config.depths.iter().enumerate() {
            for b_idx in 0..*depth {
                let hf_prefix = format!("swin.encoder.layers.{}.blocks.{}", s_idx, b_idx);
                let int_prefix = format!("swin.stages.{}.blocks.{}", s_idx, b_idx);

                map.insert(
                    format!("{}.layernorm_before.weight", hf_prefix),
                    format!("{}.norm1.weight", int_prefix),
                );
                map.insert(
                    format!("{}.layernorm_after.weight", hf_prefix),
                    format!("{}.norm2.weight", int_prefix),
                );
                map.insert(
                    format!("{}.attention.self.query.weight", hf_prefix),
                    format!("{}.attn.qkv.weight", int_prefix),
                );
                map.insert(
                    format!("{}.attention.output.dense.weight", hf_prefix),
                    format!("{}.attn.proj.weight", int_prefix),
                );
                map.insert(
                    format!("{}.intermediate.dense.weight", hf_prefix),
                    format!("{}.ffn.fc1.weight", int_prefix),
                );
                map.insert(
                    format!("{}.output.dense.weight", hf_prefix),
                    format!("{}.ffn.fc2.weight", int_prefix),
                );
            }

            // Downsample
            if s_idx < self.swin.config.num_stages() - 1 {
                let hf_ds = format!("swin.encoder.layers.{}.downsample", s_idx);
                let int_ds = format!("swin.stages.{}.downsample", s_idx);
                map.insert(
                    format!("{}.reduction.weight", hf_ds),
                    format!("{}.reduction.weight", int_ds),
                );
                map.insert(
                    format!("{}.norm.weight", hf_ds),
                    format!("{}.layer_norm.weight", int_ds),
                );
            }
        }

        // Final norm & head
        map.insert("swin.layernorm.weight".to_string(), "swin.norm.weight".to_string());
        map.insert("classifier.weight".to_string(), "head.weight".to_string());
        map.insert("classifier.bias".to_string(), "head.bias".to_string());

        map
    }
}
