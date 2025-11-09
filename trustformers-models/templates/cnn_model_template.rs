//! {{MODEL_NAME}} - {{MODEL_DESCRIPTION}}
//!
//! This is an auto-generated CNN model implementation from the TrustformeRS template.
//!
//! ## Architecture Details
//! {{ARCHITECTURE_DETAILS}}
//!
//! ## Usage Example
//! ```rust
//! use trustformers_models::{{model_name_snake}}::{{ModelName}};
//!
//! let model = {{ModelName}}::from_pretrained("{{model_id}}")?;
//! let outputs = model.forward(&images)?;
//! ```

use crate::{
    ModelBase, ModelConfig, ModelOutput, PreTrainedModel,
    utils::activations::Activation,
};
use trustformers_core::{
    Result, Tensor, Module, ModuleList,
    nn::{Conv2d, BatchNorm2d, Linear, Dropout, MaxPool2d, AdaptiveAvgPool2d},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for {{MODEL_NAME}}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct {{ModelName}}Config {
    /// Number of input channels (3 for RGB)
    pub num_channels: usize,
    /// Number of output classes
    pub num_classes: usize,
    /// Image input size (height, width)
    pub image_size: (usize, usize),
    /// Layer configuration
    pub layer_config: Vec<LayerConfig>,
    /// Hidden activation function
    pub hidden_act: Activation,
    /// Dropout probability
    pub dropout_prob: f32,
    /// Whether to use batch normalization
    pub use_batch_norm: bool,
    /// Initializer range for weights
    pub initializer_range: f32,

    // Add model-specific configuration parameters here
    {{#if has_custom_params}}
    {{custom_params}}
    {{/if}}
}

/// Configuration for a single layer/block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub num_blocks: usize,
    {{#if has_custom_layer_params}}
    {{custom_layer_params}}
    {{/if}}
}

impl Default for {{ModelName}}Config {
    fn default() -> Self {
        Self {
            num_channels: 3,
            num_classes: {{default_num_classes}},
            image_size: ({{default_image_height}}, {{default_image_width}}),
            layer_config: vec![
                {{#each default_layers}}
                LayerConfig {
                    out_channels: {{out_channels}},
                    kernel_size: {{kernel_size}},
                    stride: {{stride}},
                    padding: {{padding}},
                    num_blocks: {{num_blocks}},
                },
                {{/each}}
            ],
            hidden_act: Activation::{{default_activation}},
            dropout_prob: {{default_dropout}},
            use_batch_norm: {{default_use_bn}},
            initializer_range: 0.02,
            {{#if has_custom_params}}
            {{custom_params_defaults}}
            {{/if}}
        }
    }
}

impl ModelConfig for {{ModelName}}Config {
    fn hidden_size(&self) -> usize {
        // For CNNs, return the feature dimension before classification
        self.layer_config.last()
            .map(|cfg| cfg.out_channels)
            .unwrap_or(512)
    }

    fn num_labels(&self) -> Option<usize> {
        Some(self.num_classes)
    }

    fn id2label(&self) -> Option<&HashMap<usize, String>> {
        None
    }

    fn label2id(&self) -> Option<&HashMap<String, usize>> {
        None
    }

    fn is_decoder(&self) -> bool {
        false
    }

    fn is_encoder_decoder(&self) -> bool {
        false
    }
}

/// {{MODEL_NAME}} Convolutional Block
struct {{ModelName}}Block {
    conv1: Conv2d,
    {{#if use_batch_norm}}
    bn1: BatchNorm2d,
    {{/if}}
    activation: Activation,
    {{#if has_residual}}
    conv2: Option<Conv2d>,
    bn2: Option<BatchNorm2d>,
    shortcut: Option<Conv2d>,
    bn_shortcut: Option<BatchNorm2d>,
    {{/if}}
    {{#if has_depthwise}}
    depthwise: Conv2d,
    pointwise: Conv2d,
    {{/if}}
}

impl {{ModelName}}Block {
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        config: &{{ModelName}}Config,
    ) -> Result<Self> {
        Ok(Self {
            conv1: Conv2d::new(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                1, // dilation
                1, // groups
                true, // bias
            )?,
            {{#if use_batch_norm}}
            bn1: BatchNorm2d::new(out_channels)?,
            {{/if}}
            activation: config.hidden_act.clone(),
            {{#if has_residual}}
            conv2: Some(Conv2d::new(
                out_channels,
                out_channels,
                3,
                1,
                1,
                1,
                1,
                !config.use_batch_norm,
            )?),
            bn2: if config.use_batch_norm {
                Some(BatchNorm2d::new(out_channels)?)
            } else {
                None
            },
            shortcut: if in_channels != out_channels || stride != 1 {
                Some(Conv2d::new(in_channels, out_channels, 1, stride, 0, 1, 1, false)?)
            } else {
                None
            },
            bn_shortcut: if (in_channels != out_channels || stride != 1) && config.use_batch_norm {
                Some(BatchNorm2d::new(out_channels)?)
            } else {
                None
            },
            {{/if}}
            {{#if has_depthwise}}
            depthwise: Conv2d::new(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                1,
                in_channels, // Depthwise convolution
                false,
            )?,
            pointwise: Conv2d::new(
                in_channels,
                out_channels,
                1,
                1,
                0,
                1,
                1,
                !config.use_batch_norm,
            )?,
            {{/if}}
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        {{#if has_residual}}
        let identity = x.clone();
        {{/if}}

        {{#if has_depthwise}}
        // Depthwise separable convolution
        let out = self.depthwise.forward(x)?;
        {{#if use_batch_norm}}
        let out = self.bn1.forward(&out)?;
        {{/if}}
        let out = self.activation.forward(&out)?;
        let out = self.pointwise.forward(&out)?;
        {{else}}
        // Standard convolution
        let out = self.conv1.forward(x)?;
        {{#if use_batch_norm}}
        let out = self.bn1.forward(&out)?;
        {{/if}}
        let out = self.activation.forward(&out)?;
        {{/if}}

        {{#if has_residual}}
        // Second convolution for residual blocks
        if let Some(ref conv2) = self.conv2 {
            let out = conv2.forward(&out)?;
            if let Some(ref bn2) = self.bn2 {
                let out = bn2.forward(&out)?;
            }
        }

        // Shortcut connection
        let identity = if let Some(ref shortcut) = self.shortcut {
            let id = shortcut.forward(&identity)?;
            if let Some(ref bn) = self.bn_shortcut {
                bn.forward(&id)?
            } else {
                id
            }
        } else {
            identity
        };

        // Add residual
        let out = out.add(&identity)?;
        let out = self.activation.forward(&out)?;
        {{/if}}

        Ok(out)
    }
}

/// {{MODEL_NAME}} Stage (multiple blocks)
struct {{ModelName}}Stage {
    blocks: ModuleList<{{ModelName}}Block>,
    {{#if has_se_module}}
    se_module: Option<SEModule>,
    {{/if}}
}

impl {{ModelName}}Stage {
    fn new(
        in_channels: usize,
        layer_cfg: &LayerConfig,
        config: &{{ModelName}}Config,
    ) -> Result<Self> {
        let mut blocks = ModuleList::new();

        // First block might have stride > 1 for downsampling
        blocks.push({{ModelName}}Block::new(
            in_channels,
            layer_cfg.out_channels,
            layer_cfg.kernel_size,
            layer_cfg.stride,
            layer_cfg.padding,
            config,
        )?);

        // Remaining blocks have stride 1
        for _ in 1..layer_cfg.num_blocks {
            blocks.push({{ModelName}}Block::new(
                layer_cfg.out_channels,
                layer_cfg.out_channels,
                layer_cfg.kernel_size,
                1, // stride
                layer_cfg.padding,
                config,
            )?);
        }

        Ok(Self {
            blocks,
            {{#if has_se_module}}
            se_module: Some(SEModule::new(layer_cfg.out_channels)?),
            {{/if}}
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();

        for block in &self.blocks {
            out = block.forward(&out)?;
        }

        {{#if has_se_module}}
        if let Some(ref se) = self.se_module {
            out = se.forward(&out)?;
        }
        {{/if}}

        Ok(out)
    }
}

{{#if has_se_module}}
/// Squeeze-and-Excitation Module
struct SEModule {
    avg_pool: AdaptiveAvgPool2d,
    fc1: Linear,
    fc2: Linear,
    activation: Activation,
    sigmoid: Activation,
}

impl SEModule {
    fn new(channels: usize) -> Result<Self> {
        let reduction = 16;
        let reduced_channels = (channels / reduction).max(1);

        Ok(Self {
            avg_pool: AdaptiveAvgPool2d::new((1, 1))?,
            fc1: Linear::new(channels, reduced_channels, true)?,
            fc2: Linear::new(reduced_channels, channels, true)?,
            activation: Activation::ReLU,
            sigmoid: Activation::Sigmoid,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let b = x.shape()[0];
        let c = x.shape()[1];

        // Squeeze
        let y = self.avg_pool.forward(x)?;
        let y = y.view(&[b, c])?;

        // Excitation
        let y = self.fc1.forward(&y)?;
        let y = self.activation.forward(&y)?;
        let y = self.fc2.forward(&y)?;
        let y = self.sigmoid.forward(&y)?;

        // Scale
        let y = y.view(&[b, c, 1, 1])?;
        x.mul(&y)
    }
}
{{/if}}

/// {{MODEL_NAME}} Backbone
struct {{ModelName}}Backbone {
    {{#if has_stem}}
    stem: {{ModelName}}Stem,
    {{else}}
    initial_conv: Conv2d,
    initial_bn: Option<BatchNorm2d>,
    initial_pool: Option<MaxPool2d>,
    {{/if}}
    stages: ModuleList<{{ModelName}}Stage>,
    activation: Activation,
}

impl {{ModelName}}Backbone {
    fn new(config: &{{ModelName}}Config) -> Result<Self> {
        let mut stages = ModuleList::new();
        let mut in_channels = config.num_channels;

        {{#if has_stem}}
        let stem = {{ModelName}}Stem::new(config)?;
        in_channels = stem.out_channels();
        {{else}}
        // Initial convolution
        let initial_out = config.layer_config[0].out_channels / 2;
        let initial_conv = Conv2d::new(
            config.num_channels,
            initial_out,
            7,
            2,
            3,
            1,
            1,
            !config.use_batch_norm,
        )?;
        let initial_bn = if config.use_batch_norm {
            Some(BatchNorm2d::new(initial_out)?)
        } else {
            None
        };
        let initial_pool = Some(MaxPool2d::new(3, 2, 1)?);
        in_channels = initial_out;
        {{/if}}

        // Build stages
        for layer_cfg in &config.layer_config {
            stages.push({{ModelName}}Stage::new(in_channels, layer_cfg, config)?);
            in_channels = layer_cfg.out_channels;
        }

        Ok(Self {
            {{#if has_stem}}
            stem,
            {{else}}
            initial_conv,
            initial_bn,
            initial_pool,
            {{/if}}
            stages,
            activation: config.hidden_act.clone(),
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Vec<Tensor>> {
        let mut features = Vec::new();

        {{#if has_stem}}
        let x = self.stem.forward(x)?;
        {{else}}
        // Initial layers
        let x = self.initial_conv.forward(x)?;
        let x = if let Some(ref bn) = self.initial_bn {
            bn.forward(&x)?
        } else {
            x
        };
        let x = self.activation.forward(&x)?;
        let mut x = if let Some(ref pool) = self.initial_pool {
            pool.forward(&x)?
        } else {
            x
        };
        {{/if}}

        // Forward through stages
        for stage in &self.stages {
            x = stage.forward(&x)?;
            features.push(x.clone());
        }

        Ok(features)
    }
}

{{#if has_stem}}
/// {{MODEL_NAME}} Stem
struct {{ModelName}}Stem {
    // Define stem layers
    conv1: Conv2d,
    bn1: Option<BatchNorm2d>,
    conv2: Conv2d,
    bn2: Option<BatchNorm2d>,
    pool: MaxPool2d,
    activation: Activation,
}

impl {{ModelName}}Stem {
    fn new(config: &{{ModelName}}Config) -> Result<Self> {
        let stem_channels = 64;

        Ok(Self {
            conv1: Conv2d::new(config.num_channels, stem_channels / 2, 3, 2, 1, 1, 1, false)?,
            bn1: if config.use_batch_norm {
                Some(BatchNorm2d::new(stem_channels / 2)?)
            } else {
                None
            },
            conv2: Conv2d::new(stem_channels / 2, stem_channels, 3, 1, 1, 1, 1, false)?,
            bn2: if config.use_batch_norm {
                Some(BatchNorm2d::new(stem_channels)?)
            } else {
                None
            },
            pool: MaxPool2d::new(3, 2, 1)?,
            activation: config.hidden_act.clone(),
        })
    }

    fn out_channels(&self) -> usize {
        64
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?;
        let x = if let Some(ref bn) = self.bn1 {
            bn.forward(&x)?
        } else {
            x
        };
        let x = self.activation.forward(&x)?;

        let x = self.conv2.forward(&x)?;
        let x = if let Some(ref bn) = self.bn2 {
            bn.forward(&x)?
        } else {
            x
        };
        let x = self.activation.forward(&x)?;

        self.pool.forward(&x)
    }
}
{{/if}}

/// {{MODEL_NAME}} Model
pub struct {{ModelName}} {
    config: {{ModelName}}Config,
    backbone: {{ModelName}}Backbone,
    pooler: AdaptiveAvgPool2d,
    {{#if has_neck}}
    neck: {{ModelName}}Neck,
    {{/if}}
}

impl {{ModelName}} {
    /// Create a new {{MODEL_NAME}} model
    pub fn new(config: {{ModelName}}Config) -> Result<Self> {
        Ok(Self {
            backbone: {{ModelName}}Backbone::new(&config)?,
            pooler: AdaptiveAvgPool2d::new((1, 1))?,
            {{#if has_neck}}
            neck: {{ModelName}}Neck::new(&config)?,
            {{/if}}
            config,
        })
    }

    /// Forward pass through the model
    pub fn forward(&self, pixel_values: &Tensor) -> Result<{{ModelName}}Output> {
        // Check input shape
        if pixel_values.ndim() != 4 {
            return Err(TrustformersError::ValueError { message:
                format!("Expected 4D input tensor [B, C, H, W], got {}D", pixel_values.ndim())
            ));
        }

        // Forward through backbone
        let features = self.backbone.forward(pixel_values)?;
        let last_feature = features.last()
            .ok_or_else(|| TrustformersError::RuntimeError { message: "No features from backbone".into()))?;

        {{#if has_neck}}
        // Apply neck if present
        let last_feature = self.neck.forward(last_feature)?;
        {{/if}}

        // Global average pooling
        let pooled = self.pooler.forward(last_feature)?;
        let pooled = pooled.flatten(1)?;

        Ok({{ModelName}}Output {
            last_feature_map: last_feature.clone(),
            pooler_output: pooled,
            hidden_states: Some(features),
        })
    }

    /// Extract features at multiple scales
    pub fn extract_features(&self, pixel_values: &Tensor) -> Result<Vec<Tensor>> {
        self.backbone.forward(pixel_values)
    }
}

{{#if has_neck}}
/// {{MODEL_NAME}} Neck (FPN, etc.)
struct {{ModelName}}Neck {
    // Define neck layers for feature pyramid
    lateral_convs: ModuleList<Conv2d>,
    fpn_convs: ModuleList<Conv2d>,
}

impl {{ModelName}}Neck {
    fn new(config: &{{ModelName}}Config) -> Result<Self> {
        let mut lateral_convs = ModuleList::new();
        let mut fpn_convs = ModuleList::new();

        let fpn_channels = 256;

        for layer_cfg in config.layer_config.iter().rev() {
            lateral_convs.push(Conv2d::new(
                layer_cfg.out_channels,
                fpn_channels,
                1,
                1,
                0,
                1,
                1,
                true,
            )?);
            fpn_convs.push(Conv2d::new(
                fpn_channels,
                fpn_channels,
                3,
                1,
                1,
                1,
                1,
                true,
            )?);
        }

        Ok(Self {
            lateral_convs,
            fpn_convs,
        })
    }

    fn forward(&self, features: &[Tensor]) -> Result<Vec<Tensor>> {
        // Simple FPN implementation
        let mut fpn_features = Vec::new();

        // Top-down pathway
        for (i, feat) in features.iter().rev().enumerate() {
            let lateral = self.lateral_convs[i].forward(feat)?;

            if i == 0 {
                fpn_features.push(lateral);
            } else {
                // Upsample and add
                let prev = &fpn_features[i - 1];
                let upsampled = prev.upsample_nearest2d(&[feat.shape()[2], feat.shape()[3]])?;
                let combined = lateral.add(&upsampled)?;
                fpn_features.push(combined);
            }
        }

        // Apply FPN convs
        let mut outputs = Vec::new();
        for (i, feat) in fpn_features.iter().enumerate() {
            outputs.push(self.fpn_convs[i].forward(feat)?);
        }

        outputs.reverse();
        Ok(outputs)
    }
}
{{/if}}

/// Output from {{MODEL_NAME}}
#[derive(Debug)]
pub struct {{ModelName}}Output {
    /// Last feature map from backbone
    pub last_feature_map: Tensor,
    /// Pooled features
    pub pooler_output: Tensor,
    /// Hidden states from each stage
    pub hidden_states: Option<Vec<Tensor>>,
}

impl ModelOutput for {{ModelName}}Output {
    fn logits(&self) -> Option<&Tensor> {
        None // Base model doesn't have logits
    }

    fn loss(&self) -> Option<&Tensor> {
        None
    }

    fn hidden_states(&self) -> Option<&Vec<Tensor>> {
        self.hidden_states.as_ref()
    }

    fn attentions(&self) -> Option<&Vec<Tensor>> {
        None // CNN doesn't have attention
    }
}

impl ModelBase for {{ModelName}} {
    type Config = {{ModelName}}Config;
    type Output = {{ModelName}}Output;

    fn new(config: Self::Config) -> Result<Self> {
        Self::new(config)
    }

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn forward(&self, input_ids: &Tensor, _attention_mask: Option<&Tensor>) -> Result<Self::Output> {
        // For CNN, input_ids are pixel values
        self.forward(input_ids)
    }
}

impl Module for {{ModelName}} {
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();

        // Backbone parameters
        {{#if has_stem}}
        params.extend(self.backbone.stem.conv1.parameters());
        if let Some(ref bn) = self.backbone.stem.bn1 {
            params.extend(bn.parameters());
        }
        params.extend(self.backbone.stem.conv2.parameters());
        if let Some(ref bn) = self.backbone.stem.bn2 {
            params.extend(bn.parameters());
        }
        {{else}}
        params.extend(self.backbone.initial_conv.parameters());
        if let Some(ref bn) = self.backbone.initial_bn {
            params.extend(bn.parameters());
        }
        {{/if}}

        // Stage parameters
        for stage in &self.backbone.stages {
            for block in &stage.blocks {
                params.extend(collect_block_params(block));
            }
            {{#if has_se_module}}
            if let Some(ref se) = stage.se_module {
                params.extend(se.fc1.parameters());
                params.extend(se.fc2.parameters());
            }
            {{/if}}
        }

        {{#if has_neck}}
        // Neck parameters
        for conv in &self.neck.lateral_convs {
            params.extend(conv.parameters());
        }
        for conv in &self.neck.fpn_convs {
            params.extend(conv.parameters());
        }
        {{/if}}

        params
    }
}

fn collect_block_params(block: &{{ModelName}}Block) -> Vec<&Tensor> {
    let mut params = Vec::new();

    params.extend(block.conv1.parameters());
    {{#if use_batch_norm}}
    params.extend(block.bn1.parameters());
    {{/if}}

    {{#if has_residual}}
    if let Some(ref conv2) = block.conv2 {
        params.extend(conv2.parameters());
    }
    if let Some(ref bn2) = block.bn2 {
        params.extend(bn2.parameters());
    }
    if let Some(ref shortcut) = block.shortcut {
        params.extend(shortcut.parameters());
    }
    if let Some(ref bn) = block.bn_shortcut {
        params.extend(bn.parameters());
    }
    {{/if}}

    {{#if has_depthwise}}
    params.extend(block.depthwise.parameters());
    params.extend(block.pointwise.parameters());
    {{/if}}

    params
}

// Model-specific heads (to be implemented in separate files)

/// {{MODEL_NAME}} for Image Classification
pub struct {{ModelName}}ForImageClassification {
    {{model_name_snake}}: {{ModelName}},
    classifier: Linear,
    dropout: Option<Dropout>,
}

impl {{ModelName}}ForImageClassification {
    pub fn new(config: {{ModelName}}Config) -> Result<Self> {
        let {{model_name_snake}} = {{ModelName}}::new(config.clone())?;
        let hidden_size = config.layer_config.last()
            .map(|cfg| cfg.out_channels)
            .unwrap_or(512);

        Ok(Self {
            {{model_name_snake}},
            classifier: Linear::new(hidden_size, config.num_classes, true)?,
            dropout: if config.dropout_prob > 0.0 {
                Some(Dropout::new(config.dropout_prob))
            } else {
                None
            },
        })
    }

    pub fn forward(&self, pixel_values: &Tensor) -> Result<ClassificationOutput> {
        let outputs = self.{{model_name_snake}}.forward(pixel_values)?;

        let pooled = if let Some(ref dropout) = self.dropout {
            dropout.forward(&outputs.pooler_output)?
        } else {
            outputs.pooler_output.clone()
        };

        let logits = self.classifier.forward(&pooled)?;

        Ok(ClassificationOutput {
            logits,
            hidden_states: outputs.hidden_states,
        })
    }
}

/// Classification output
#[derive(Debug)]
pub struct ClassificationOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Vec<Tensor>>,
}

{{#if supports_detection}}
/// {{MODEL_NAME}} for Object Detection
pub struct {{ModelName}}ForObjectDetection {
    {{model_name_snake}}: {{ModelName}},
    detection_head: DetectionHead,
}

/// Simple detection head
struct DetectionHead {
    class_conv: Conv2d,
    bbox_conv: Conv2d,
    num_classes: usize,
}
{{/if}}

{{#if supports_segmentation}}
/// {{MODEL_NAME}} for Semantic Segmentation
pub struct {{ModelName}}ForSemanticSegmentation {
    {{model_name_snake}}: {{ModelName}},
    segmentation_head: SegmentationHead,
}

/// Simple segmentation head
struct SegmentationHead {
    conv1: Conv2d,
    conv2: Conv2d,
    num_classes: usize,
}
{{/if}}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_{{model_name_snake}}_config() {
        let config = {{ModelName}}Config::default();
        assert_eq!(config.num_channels, 3);
        assert_eq!(config.layer_config.len(), {{num_default_layers}});
    }

    #[test]
    fn test_{{model_name_snake}}_creation() {
        let config = {{ModelName}}Config::default();
        let model = {{ModelName}}::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_{{model_name_snake}}_forward() {
        let config = {{ModelName}}Config {
            num_channels: 3,
            num_classes: 10,
            image_size: (32, 32),
            layer_config: vec![
                LayerConfig { out_channels: 16, kernel_size: 3, stride: 1, padding: 1, num_blocks: 2 },
                LayerConfig { out_channels: 32, kernel_size: 3, stride: 2, padding: 1, num_blocks: 2 },
            ],
            ..Default::default()
        };

        let model = {{ModelName}}::new(config).unwrap();

        // Create dummy input
        let batch_size = 2;
        let pixel_values = Tensor::randn(&[batch_size, 3, 32, 32]).unwrap();

        // Forward pass
        let output = model.forward(&pixel_values);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.pooler_output.shape(), &[batch_size, 32]);
    }

    #[test]
    fn test_{{model_name_snake}}_classification() {
        let config = {{ModelName}}Config {
            num_classes: 10,
            ..Default::default()
        };

        let model = {{ModelName}}ForImageClassification::new(config).unwrap();
        let pixel_values = Tensor::randn(&[2, 3, 32, 32]).unwrap();

        let output = model.forward(&pixel_values);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.logits.shape(), &[2, 10]);
    }
}

// Re-export commonly used items
pub use self::{{ModelName}}Config as Config;
pub use self::{{ModelName}} as Model;