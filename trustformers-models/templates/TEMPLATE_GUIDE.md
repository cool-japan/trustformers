# TrustformeRS Model Template Guide

This guide explains how to use the TrustformeRS model templates to quickly implement new models.

## Overview

TrustformeRS provides three model templates:
1. **Transformer Template** - For attention-based models (BERT, GPT, T5, etc.)
2. **CNN Template** - For convolutional models (ResNet, EfficientNet, etc.)
3. **Custom Template** - For specialized architectures (GNNs, RNNs, etc.)

## Quick Start

### Using the Model Generator

```bash
# Generate a transformer model
python templates/generate_model.py --type transformer --name BERT --output bert/

# Generate a CNN model
python templates/generate_model.py --type cnn --name ResNet --output resnet/

# Generate with custom configuration
python templates/generate_model.py --type custom --name GraphNet --config graph_config.json --output graphnet/

# Interactive mode
python templates/generate_model.py --type transformer --name MyModel --output mymodel/ --interactive
```

### Manual Template Usage

1. Copy the appropriate template file
2. Replace template variables ({{VARIABLE_NAME}})
3. Implement model-specific logic
4. Add tests and documentation

## Template Variables

### Common Variables

All templates use these variables:
- `{{MODEL_NAME}}` - Model name (e.g., "BERT")
- `{{model_name_snake}}` - Snake case name (e.g., "bert")
- `{{MODEL_DESCRIPTION}}` - Model description
- `{{ARCHITECTURE_DETAILS}}` - Detailed architecture description
- `{{model_id}}` - Pretrained model ID

### Transformer Template Variables

```rust
// Architecture flags
{{is_decoder}}              // true/false - Decoder-only model
{{is_encoder_decoder}}      // true/false - Encoder-decoder model
{{has_cross_attention}}     // true/false - Cross-attention layers
{{has_pooler}}             // true/false - Pooling layer
{{use_position_embeddings}} // true/false - Position embeddings
{{use_token_type_embeddings}} // true/false - Token type embeddings

// Configuration defaults
{{default_vocab_size}}      // e.g., 30522
{{default_hidden_size}}     // e.g., 768
{{default_num_layers}}      // e.g., 12
{{default_num_heads}}       // e.g., 12
{{default_activation}}      // e.g., "GELU"
```

### CNN Template Variables

```rust
// Architecture flags
{{has_stem}}               // true/false - Stem layers
{{has_residual}}           // true/false - Residual connections
{{has_depthwise}}          // true/false - Depthwise convolutions
{{has_se_module}}          // true/false - Squeeze-excitation
{{has_neck}}               // true/false - Feature pyramid neck
{{supports_detection}}     // true/false - Object detection head
{{supports_segmentation}}  // true/false - Segmentation head

// Layer configuration
{{default_layers}}         // Array of layer configs
{{num_default_layers}}     // Number of layers
```

### Custom Template Variables

```rust
// Configuration
{{config_params}}          // Array of config parameters
{{config_defaults}}        // Default values
{{hidden_size_expr}}       // Expression for hidden size

// Components
{{components}}             // Array of model components
{{model_components}}       // Model struct fields
{{model_initializers}}     // Component initializers

// Methods
{{forward_args}}           // Forward method arguments
{{additional_methods}}     // Extra model methods
{{utility_functions}}      // Helper functions
```

## Configuration Files

### Transformer Configuration Example

```json
{
  "description": "BERT-like transformer model",
  "architecture_details": "12-layer transformer with masked language modeling",
  "model_id": "trustformers/bert-base",
  
  "vocab_size": 30522,
  "hidden_size": 768,
  "num_layers": 12,
  "num_heads": 12,
  "intermediate_size": 3072,
  "activation": "GELU",
  "max_positions": 512,
  
  "is_decoder": false,
  "is_encoder_decoder": false,
  "has_pooler": true,
  "use_position_embeddings": true,
  "use_token_type_embeddings": true,
  
  "custom_params": [
    {
      "name": "type_vocab_size",
      "type": "usize",
      "description": "Token type vocabulary size",
      "default": "2"
    }
  ]
}
```

### CNN Configuration Example

```json
{
  "description": "ResNet-50 implementation",
  "architecture_details": "50-layer residual network for image classification",
  "model_id": "trustformers/resnet50",
  
  "num_classes": 1000,
  "image_height": 224,
  "image_width": 224,
  "activation": "ReLU",
  "dropout": 0.0,
  
  "has_stem": true,
  "has_residual": true,
  "use_batch_norm": true,
  
  "layers": [
    {"out_channels": 64, "kernel_size": 7, "stride": 2, "padding": 3, "num_blocks": 3},
    {"out_channels": 128, "kernel_size": 3, "stride": 2, "padding": 1, "num_blocks": 4},
    {"out_channels": 256, "kernel_size": 3, "stride": 2, "padding": 1, "num_blocks": 6},
    {"out_channels": 512, "kernel_size": 3, "stride": 2, "padding": 1, "num_blocks": 3}
  ]
}
```

### Custom Model Configuration Example

```json
{
  "description": "Graph Neural Network for molecular property prediction",
  "architecture_details": "Message-passing GNN with attention mechanism",
  
  "config_params": [
    {"name": "node_dim", "type": "usize", "description": "Node feature dimension"},
    {"name": "edge_dim", "type": "usize", "description": "Edge feature dimension"},
    {"name": "hidden_dim", "type": "usize", "description": "Hidden dimension"},
    {"name": "num_layers", "type": "usize", "description": "Number of GNN layers"},
    {"name": "num_heads", "type": "usize", "description": "Number of attention heads"},
    {"name": "dropout", "type": "f32", "description": "Dropout probability"}
  ],
  
  "config_defaults": [
    {"name": "node_dim", "value": "128"},
    {"name": "edge_dim", "value": "64"},
    {"name": "hidden_dim", "value": "256"},
    {"name": "num_layers", "value": "4"},
    {"name": "num_heads", "value": "8"},
    {"name": "dropout", "value": "0.1"}
  ],
  
  "components": [
    {
      "component_name": "GraphAttentionLayer",
      "component_description": "Multi-head graph attention layer",
      "fields": [
        {"name": "query_proj", "type": "Linear"},
        {"name": "key_proj", "type": "Linear"},
        {"name": "value_proj", "type": "Linear"},
        {"name": "edge_proj", "type": "Linear"},
        {"name": "output_proj", "type": "Linear"},
        {"name": "dropout", "type": "Dropout"}
      ]
    }
  ]
}
```

## Template Conditionals

Templates support conditional blocks:

```rust
{{#if has_pooler}}
    pooler: Option<{{ModelName}}Pooler>,
{{/if}}

{{#if is_decoder}}
    // Decoder-specific code
{{else}}
    // Encoder-specific code
{{/if}}
```

## Template Loops

Templates support iteration:

```rust
{{#each layers}}
LayerConfig {
    out_channels: {{out_channels}},
    kernel_size: {{kernel_size}},
    stride: {{stride}},
    padding: {{padding}},
    num_blocks: {{num_blocks}},
},
{{/each}}
```

## Best Practices

### 1. Start with the Right Template

- **Transformer**: For models with self-attention (BERT, GPT, T5)
- **CNN**: For convolutional models (ResNet, MobileNet, EfficientNet)
- **Custom**: For everything else (GNN, RNN, VAE, GAN)

### 2. Configuration First

Create a detailed configuration file before generating:
- Define all architecture parameters
- Specify custom components
- Document expected behavior

### 3. Incremental Implementation

1. Generate basic structure
2. Implement core forward pass
3. Add specialized heads
4. Implement training logic
5. Add optimizations

### 4. Testing Strategy

Always include:
- Configuration tests
- Shape tests
- Forward pass tests
- Gradient flow tests
- Integration tests

## Examples

### Example 1: Creating a BERT Model

```bash
# 1. Create configuration
cat > bert_config.json << EOF
{
  "description": "BERT base model",
  "vocab_size": 30522,
  "hidden_size": 768,
  "num_layers": 12,
  "num_heads": 12,
  "intermediate_size": 3072,
  "activation": "GELU",
  "has_pooler": true,
  "use_position_embeddings": true,
  "use_token_type_embeddings": true
}
EOF

# 2. Generate model
python templates/generate_model.py --type transformer --name BERT --config bert_config.json --output bert/

# 3. Implement specialized heads
# Edit bert/model.rs to add BertForMaskedLM, BertForSequenceClassification, etc.
```

### Example 2: Creating a ResNet Model

```bash
# Interactive generation
python templates/generate_model.py --type cnn --name ResNet --output resnet/ --interactive

# Answer prompts:
# Model description: ResNet-50 for image classification
# Number of classes [1000]: 1000
# Image height [224]: 224
# Image width [224]: 224
# Use residual connections? [Y/n]: y
# Use stem? [y/N]: y
# Number of stages [4]: 4
# ...
```

### Example 3: Creating a Graph Neural Network

```json
// graph_config.json
{
  "description": "Graph Attention Network",
  "config_params": [
    {"name": "num_nodes", "type": "usize", "description": "Maximum nodes"},
    {"name": "node_features", "type": "usize", "description": "Node feature size"},
    {"name": "hidden_dim", "type": "usize", "description": "Hidden dimension"},
    {"name": "num_heads", "type": "usize", "description": "Attention heads"},
    {"name": "num_layers", "type": "usize", "description": "Number of GAT layers"}
  ],
  "components": [
    {
      "component_name": "GATLayer",
      "fields": [
        {"name": "attention", "type": "MultiHeadAttention"},
        {"name": "ffn", "type": "Linear"},
        {"name": "dropout", "type": "Dropout"}
      ]
    }
  ]
}
```

## Customization

### Adding New Template Variables

1. Edit the template file to add placeholders
2. Update `generate_model.py` to handle new variables
3. Document the variables in this guide

### Creating New Templates

1. Create a new template file in `templates/`
2. Add to `ModelGenerator.templates` dict
3. Implement `_prepare_<type>_vars()` method
4. Update documentation

## Troubleshooting

### Common Issues

1. **Missing variables**: Check all {{VARIABLE}} are defined
2. **Syntax errors**: Validate Rust syntax after generation
3. **Import errors**: Ensure all used types are imported
4. **Test failures**: Verify shapes and configurations

### Debug Mode

```python
# Add --debug flag for verbose output
python templates/generate_model.py --type transformer --name BERT --output bert/ --debug
```

## Contributing

To add new templates or improve existing ones:

1. Fork the repository
2. Create your template/improvement
3. Add comprehensive tests
4. Update this documentation
5. Submit a pull request

## Resources

- [TrustformeRS Architecture Guide](../../docs/ARCHITECTURE.md)
- [Model Implementation Tutorial](../../docs/tutorials/model_implementation.md)
- [Contributing Guidelines](../../CONTRIBUTING.md)
- [API Documentation](../../docs/api/)