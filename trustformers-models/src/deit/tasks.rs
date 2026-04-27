//! Task-specific heads built on top of the DeiT base model.
//!
//! Currently provides image classification via `DeiTForImageClassification`.

use crate::deit::config::DeiTConfig;
use crate::deit::model::DeiTModel;
use scirs2_core::ndarray::{Array2, Array4, Ix2};
use std::collections::HashMap;
use trustformers_core::device::Device;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::layers::linear::Linear;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Layer;

/// DeiT image-classification head.
///
/// Two linear heads are applied — one on the [CLS] token and one on the
/// distillation token.  During inference the logits are averaged, which was
/// shown to improve accuracy versus using either token alone.
///
/// # Example
///
/// ```rust,no_run
/// use trustformers_models::deit::{DeiTConfig, DeiTForImageClassification};
/// use scirs2_core::ndarray::Array4;
///
/// let config = DeiTConfig::deit_tiny_patch16_224();
/// let model = DeiTForImageClassification::new(config).expect("valid config constructs model");
/// let images = Array4::<f32>::zeros((2, 224, 224, 3));
/// let logits = model.forward(&images).expect("well-formed input tensor succeeds");
/// assert_eq!(logits.shape(), &[2, 1000]);
/// ```
#[derive(Debug, Clone)]
pub struct DeiTForImageClassification {
    pub deit: DeiTModel,
    /// Head applied to the [CLS] token.
    pub cls_head: Linear,
    /// Head applied to the distillation token (only present when
    /// `config.use_distillation_token` is true).
    pub distill_head: Option<Linear>,
    device: Device,
}

impl DeiTForImageClassification {
    /// Create on the CPU.
    pub fn new(config: DeiTConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    /// Create on the specified device.
    pub fn new_with_device(config: DeiTConfig, device: Device) -> Result<Self> {
        let distill_head = if config.use_distillation_token {
            Some(Linear::new_with_device(
                config.hidden_size,
                config.num_labels,
                true,
                device,
            ))
        } else {
            None
        };

        Ok(Self {
            cls_head: Linear::new_with_device(
                config.hidden_size,
                config.num_labels,
                true,
                device,
            ),
            distill_head,
            deit: DeiTModel::new_with_device(config, device)?,
            device,
        })
    }

    /// Return the active device.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Run a forward pass for image classification.
    ///
    /// Returns logits of shape `(batch, num_labels)`.
    ///
    /// When both CLS and distillation heads are present, the output is the
    /// element-wise average of the two sets of logits.  During distillation
    /// training each head is supervised separately; at inference time their
    /// average is returned here.
    pub fn forward(&self, images: &Array4<f32>) -> Result<Array2<f32>> {
        let hidden = self.deit.forward(images)?;

        // CLS token is always at position 0
        let cls_repr = hidden.slice(scirs2_core::ndarray::s![.., 0, ..]).to_owned();
        let cls_tensor = Tensor::F32(cls_repr.into_dyn());
        let cls_logits = match self.cls_head.forward(cls_tensor)? {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix2>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 from cls_head".to_string(),
                ))
            },
        };

        // Average with distillation head when available
        if let Some(ref dist_head) = self.distill_head {
            // Distillation token is at position 1
            let dist_repr = hidden.slice(scirs2_core::ndarray::s![.., 1, ..]).to_owned();
            let dist_tensor = Tensor::F32(dist_repr.into_dyn());
            let dist_logits = match dist_head.forward(dist_tensor)? {
                Tensor::F32(arr) => arr
                    .into_dimensionality::<Ix2>()
                    .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
                _ => {
                    return Err(TrustformersError::invalid_input_simple(
                        "Expected F32 from distill_head".to_string(),
                    ))
                },
            };
            Ok((cls_logits + dist_logits) * 0.5)
        } else {
            Ok(cls_logits)
        }
    }

    /// Return the raw CLS-head logits (no averaging).
    pub fn forward_cls_only(&self, images: &Array4<f32>) -> Result<Array2<f32>> {
        let cls_repr = self.deit.get_cls_output(images)?;
        let cls_tensor = Tensor::F32(cls_repr.into_dyn());
        match self.cls_head.forward(cls_tensor)? {
            Tensor::F32(arr) => Ok(arr
                .into_dimensionality::<Ix2>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?),
            _ => Err(TrustformersError::invalid_input_simple(
                "Expected F32 from cls_head".to_string(),
            )),
        }
    }

    /// Generate a named weight map for use with HuggingFace-style weight loading.
    ///
    /// Returns a map from HuggingFace parameter names to internal identifiers.
    /// This is the canonical way to load pre-trained weights into DeiT.
    pub fn weight_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();

        // Patch embedding
        map.insert(
            "deit.embeddings.patch_embeddings.projection.weight".to_string(),
            "deit.embeddings.patch_embeddings.projection.weight".to_string(),
        );
        map.insert(
            "deit.embeddings.patch_embeddings.projection.bias".to_string(),
            "deit.embeddings.patch_embeddings.projection.bias".to_string(),
        );

        // Special tokens
        map.insert(
            "deit.embeddings.cls_token".to_string(),
            "deit.embeddings.cls_token".to_string(),
        );
        if self.deit.config.use_distillation_token {
            map.insert(
                "deit.embeddings.distillation_token".to_string(),
                "deit.embeddings.distillation_token".to_string(),
            );
        }

        // Position embeddings
        map.insert(
            "deit.embeddings.position_embeddings".to_string(),
            "deit.embeddings.position_embeddings".to_string(),
        );

        // Encoder layers
        for i in 0..self.deit.config.num_hidden_layers {
            let prefix = format!("deit.encoder.layer.{}", i);
            map.insert(
                format!("{}.attention.attention.query.weight", prefix),
                format!("encoder.layers.{}.attention.attention.query.weight", i),
            );
            map.insert(
                format!("{}.attention.attention.key.weight", prefix),
                format!("encoder.layers.{}.attention.attention.key.weight", i),
            );
            map.insert(
                format!("{}.attention.attention.value.weight", prefix),
                format!("encoder.layers.{}.attention.attention.value.weight", i),
            );
            map.insert(
                format!("{}.layernorm_before.weight", prefix),
                format!("encoder.layers.{}.attention.layer_norm.weight", i),
            );
            map.insert(
                format!("{}.layernorm_after.weight", prefix),
                format!("encoder.layers.{}.mlp.layer_norm.weight", i),
            );
            map.insert(
                format!("{}.intermediate.dense.weight", prefix),
                format!("encoder.layers.{}.mlp.feed_forward.fc1.weight", i),
            );
            map.insert(
                format!("{}.output.dense.weight", prefix),
                format!("encoder.layers.{}.mlp.feed_forward.fc2.weight", i),
            );
        }

        // Final layer norm
        map.insert(
            "deit.layernorm.weight".to_string(),
            "deit.layer_norm.weight".to_string(),
        );
        map.insert(
            "deit.layernorm.bias".to_string(),
            "deit.layer_norm.bias".to_string(),
        );

        // Classification heads
        map.insert(
            "classifier.weight".to_string(),
            "cls_head.weight".to_string(),
        );
        map.insert("classifier.bias".to_string(), "cls_head.bias".to_string());
        if self.deit.config.use_distillation_token {
            map.insert(
                "dist_head.weight".to_string(),
                "distill_head.weight".to_string(),
            );
            map.insert(
                "dist_head.bias".to_string(),
                "distill_head.bias".to_string(),
            );
        }

        map
    }
}
