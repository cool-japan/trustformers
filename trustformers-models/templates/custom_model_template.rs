//! {{MODEL_NAME}} - {{MODEL_DESCRIPTION}}
//!
//! This is an auto-generated custom model implementation from the TrustformeRS template.
//!
//! ## Architecture Details
//! {{ARCHITECTURE_DETAILS}}
//!
//! ## Usage Example
//! ```rust
//! use trustformers_models::{{model_name_snake}}::{{ModelName}};
//!
//! let model = {{ModelName}}::from_pretrained("{{model_id}}")?;
//! let outputs = model.forward(&inputs)?;
//! ```

use crate::{
    ModelBase, ModelConfig, ModelOutput, PreTrainedModel,
    utils::activations::Activation,
};
use trustformers_core::{
    Result, Tensor, Module, ModuleList, ModuleDict,
    nn::{Linear, Dropout, LayerNorm, Embedding},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for {{MODEL_NAME}}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct {{ModelName}}Config {
    // Core configuration
    {{#each config_params}}
    /// {{description}}
    pub {{name}}: {{type}},
    {{/each}}

    // Common parameters
    /// Dropout probability
    pub dropout_prob: f32,
    /// Activation function
    pub activation: Activation,
    /// Layer normalization epsilon
    pub layer_norm_eps: f32,
    /// Initializer range
    pub initializer_range: f32,
}

impl Default for {{ModelName}}Config {
    fn default() -> Self {
        Self {
            {{#each config_defaults}}
            {{name}}: {{value}},
            {{/each}}
            dropout_prob: 0.1,
            activation: Activation::{{default_activation}},
            layer_norm_eps: 1e-12,
            initializer_range: 0.02,
        }
    }
}

impl ModelConfig for {{ModelName}}Config {
    fn hidden_size(&self) -> usize {
        {{hidden_size_expr}}
    }

    fn num_labels(&self) -> Option<usize> {
        {{#if has_labels}}
        Some(self.num_labels)
        {{else}}
        None
        {{/if}}
    }

    fn id2label(&self) -> Option<&HashMap<usize, String>> {
        None
    }

    fn label2id(&self) -> Option<&HashMap<String, usize>> {
        None
    }

    fn is_decoder(&self) -> bool {
        {{is_decoder}}
    }

    fn is_encoder_decoder(&self) -> bool {
        {{is_encoder_decoder}}
    }
}

{{#each components}}
/// {{component_description}}
struct {{component_name}} {
    {{#each fields}}
    {{name}}: {{type}},
    {{/each}}
}

impl {{component_name}} {
    fn new(config: &{{ModelName}}Config) -> Result<Self> {
        Ok(Self {
            {{#each field_initializers}}
            {{name}}: {{initializer}},
            {{/each}}
        })
    }

    fn forward(&self, {{#each forward_params}}{{name}}: {{type}}{{#unless @last}}, {{/unless}}{{/each}}) -> Result<{{return_type}}> {
        {{forward_implementation}}
    }
}
{{/each}}

/// {{MODEL_NAME}} Model
pub struct {{ModelName}} {
    config: {{ModelName}}Config,
    {{#each model_components}}
    {{name}}: {{type}},
    {{/each}}
}

impl {{ModelName}} {
    /// Create a new {{MODEL_NAME}} model
    pub fn new(config: {{ModelName}}Config) -> Result<Self> {
        Ok(Self {
            {{#each model_initializers}}
            {{name}}: {{initializer}},
            {{/each}}
            config,
        })
    }

    /// Main forward pass
    pub fn forward(
        &self,
        {{#each forward_args}}
        {{name}}: {{type}},
        {{/each}}
    ) -> Result<{{ModelName}}Output> {
        {{main_forward_implementation}}
    }

    {{#each additional_methods}}
    /// {{method_description}}
    pub fn {{method_name}}(&self, {{#each params}}{{name}}: {{type}}{{#unless @last}}, {{/unless}}{{/each}}) -> Result<{{return_type}}> {
        {{method_implementation}}
    }
    {{/each}}
}

/// Output from {{MODEL_NAME}}
#[derive(Debug)]
pub struct {{ModelName}}Output {
    {{#each output_fields}}
    /// {{description}}
    pub {{name}}: {{type}},
    {{/each}}
}

impl ModelOutput for {{ModelName}}Output {
    fn logits(&self) -> Option<&Tensor> {
        {{#if has_logits}}
        Some(&self.logits)
        {{else}}
        None
        {{/if}}
    }

    fn loss(&self) -> Option<&Tensor> {
        {{#if has_loss}}
        self.loss.as_ref()
        {{else}}
        None
        {{/if}}
    }

    fn hidden_states(&self) -> Option<&Vec<Tensor>> {
        {{#if returns_hidden_states}}
        self.hidden_states.as_ref()
        {{else}}
        None
        {{/if}}
    }

    fn attentions(&self) -> Option<&Vec<Tensor>> {
        {{#if returns_attentions}}
        self.attentions.as_ref()
        {{else}}
        None
        {{/if}}
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

    fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<Self::Output> {
        {{default_forward_call}}
    }
}

impl Module for {{ModelName}} {
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();

        {{#each parameter_collection}}
        {{collection_code}}
        {{/each}}

        params
    }

    fn named_parameters(&self) -> HashMap<String, &Tensor> {
        let mut params = HashMap::new();

        {{#each named_parameter_collection}}
        {{collection_code}}
        {{/each}}

        params
    }
}

{{#if has_specialized_heads}}
// Specialized model heads

{{#each specialized_heads}}
/// {{MODEL_NAME}} for {{task_name}}
pub struct {{ModelName}}For{{task_name}} {
    {{model_name_snake}}: {{ModelName}},
    {{#each head_components}}
    {{name}}: {{type}},
    {{/each}}
}

impl {{ModelName}}For{{task_name}} {
    pub fn new(config: {{ModelName}}Config) -> Result<Self> {
        Ok(Self {
            {{model_name_snake}}: {{ModelName}}::new(config.clone())?,
            {{#each head_initializers}}
            {{name}}: {{initializer}},
            {{/each}}
        })
    }

    pub fn forward(&self, {{#each task_forward_args}}{{name}}: {{type}}{{#unless @last}}, {{/unless}}{{/each}}) -> Result<{{task_output_type}}> {
        {{task_forward_implementation}}
    }
}
{{/each}}
{{/if}}

// Utility functions

{{#each utility_functions}}
/// {{function_description}}
fn {{function_name}}({{#each params}}{{name}}: {{type}}{{#unless @last}}, {{/unless}}{{/each}}) -> Result<{{return_type}}> {
    {{function_implementation}}
}
{{/each}}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_{{model_name_snake}}_config() {
        let config = {{ModelName}}Config::default();
        {{#each config_tests}}
        {{test_assertion}}
        {{/each}}
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
            {{#each test_config_overrides}}
            {{name}}: {{value}},
            {{/each}}
            ..Default::default()
        };

        let model = {{ModelName}}::new(config).unwrap();

        {{test_forward_setup}}

        let output = model.forward({{test_forward_args}});
        assert!(output.is_ok());

        let output = output.unwrap();
        {{#each output_tests}}
        {{test_assertion}}
        {{/each}}
    }

    {{#each additional_tests}}
    #[test]
    fn {{test_name}}() {
        {{test_implementation}}
    }
    {{/each}}
}

// Re-export commonly used items
pub use self::{{ModelName}}Config as Config;
pub use self::{{ModelName}} as Model;

// Example template variables for a Graph Neural Network:
// {{MODEL_NAME}} = "GraphTransformer"
// {{model_name_snake}} = "graph_transformer"
// {{config_params}} = [
//     {name: "num_nodes", type: "usize", description: "Maximum number of nodes"},
//     {name: "node_dim", type: "usize", description: "Node feature dimension"},
//     {name: "edge_dim", type: "usize", description: "Edge feature dimension"},
//     {name: "hidden_dim", type: "usize", description: "Hidden dimension"},
//     {name: "num_layers", type: "usize", description: "Number of graph layers"},
//     {name: "num_heads", type: "usize", description: "Number of attention heads"},
// ]
// {{components}} = [
//     {
//         component_name: "GraphAttentionLayer",
//         component_description: "Graph attention layer with edge features",
//         fields: [...],
//         forward_params: [...],
//         forward_implementation: "...",
//     },
//     {
//         component_name: "GraphAggregator",
//         component_description: "Aggregates node features",
//         fields: [...],
//         forward_params: [...],
//         forward_implementation: "...",
//     }
// ]

// Example template variables for a Multimodal Model:
// {{MODEL_NAME}} = "MultimodalFusion"
// {{model_name_snake}} = "multimodal_fusion"
// {{config_params}} = [
//     {name: "text_hidden_size", type: "usize", description: "Text encoder hidden size"},
//     {name: "vision_hidden_size", type: "usize", description: "Vision encoder hidden size"},
//     {name: "fusion_hidden_size", type: "usize", description: "Fusion layer hidden size"},
//     {name: "fusion_method", type: "FusionMethod", description: "How to fuse modalities"},
// ]

// Example template variables for a Time Series Model:
// {{MODEL_NAME}} = "TemporalConvNet"
// {{model_name_snake}} = "temporal_conv_net"
// {{config_params}} = [
//     {name: "input_channels", type: "usize", description: "Number of input channels"},
//     {name: "num_channels", type: "Vec<usize>", description: "Channels for each TCN layer"},
//     {name: "kernel_size", type: "usize", description: "Kernel size for convolutions"},
//     {name: "dropout", type: "f32", description: "Dropout probability"},
// ]