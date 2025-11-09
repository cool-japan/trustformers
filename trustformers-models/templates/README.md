# TrustformeRS Model Templates

Quick and easy model implementation scaffolding for TrustformeRS.

## 🚀 Quick Start

```bash
# Run the quick start script
./quick_start.sh

# Or generate a model directly
python generate_model.py --type transformer --name BERT --output ../bert/
```

## 📁 Directory Structure

```
templates/
├── transformer_model_template.rs   # Template for attention-based models
├── cnn_model_template.rs          # Template for convolutional models
├── custom_model_template.rs       # Template for specialized architectures
├── generate_model.py              # Model generation script
├── TEMPLATE_GUIDE.md             # Comprehensive template documentation
├── quick_start.sh                # Quick start demo script
└── examples/                     # Example configurations
    ├── bert_config.json          # BERT configuration
    ├── resnet_config.json        # ResNet configuration
    └── gnn_config.json           # Graph Neural Network configuration
```

## 🎯 Usage

### 1. Choose Your Template

- **Transformer**: For models with self-attention (BERT, GPT, T5, etc.)
- **CNN**: For convolutional models (ResNet, MobileNet, etc.)
- **Custom**: For everything else (GNN, RNN, VAE, etc.)

### 2. Generate Your Model

```bash
# Using a configuration file
python generate_model.py --type transformer --name MyModel --config my_config.json --output ../mymodel/

# Interactive mode
python generate_model.py --type cnn --name MyModel --output ../mymodel/ --interactive

# With default configuration
python generate_model.py --type custom --name MyModel --output ../mymodel/
```

### 3. Customize and Implement

The generator creates:
- `mod.rs` - Module definition
- `model.rs` - Model implementation with TODOs
- `tests.rs` - Test scaffolding
- `README.md` - Model documentation
- `examples/` - Usage examples

## 🛠️ Template Features

### Conditional Blocks
```rust
{{#if has_pooler}}
    pooler: Option<{{ModelName}}Pooler>,
{{/if}}
```

### Loops
```rust
{{#each layers}}
LayerConfig {
    out_channels: {{out_channels}},
    kernel_size: {{kernel_size}},
},
{{/each}}
```

### Variable Substitution
```rust
pub struct {{ModelName}} {
    config: {{ModelName}}Config,
    // ...
}
```

## 📋 Example: Creating a BERT Model

1. **Use existing config or create your own:**
```bash
cp examples/bert_config.json my_bert_config.json
# Edit my_bert_config.json as needed
```

2. **Generate the model:**
```bash
python generate_model.py --type transformer --name BERT --config my_bert_config.json --output ../bert/
```

3. **Review generated code:**
```bash
cd ../bert/
cargo check
```

4. **Implement TODOs and add features:**
- Add specialized heads (ForMaskedLM, ForSequenceClassification)
- Implement custom forward logic
- Add comprehensive tests

## 🔧 Configuration Examples

### Minimal Transformer Config
```json
{
  "description": "Minimal transformer model",
  "vocab_size": 10000,
  "hidden_size": 256,
  "num_layers": 6,
  "num_heads": 8,
  "is_decoder": false
}
```

### CNN with Custom Layers
```json
{
  "description": "Custom CNN architecture",
  "num_classes": 100,
  "layers": [
    {"out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1, "num_blocks": 2},
    {"out_channels": 64, "kernel_size": 3, "stride": 2, "padding": 1, "num_blocks": 2}
  ],
  "has_residual": true,
  "activation": "ReLU"
}
```

## 📚 Documentation

- [TEMPLATE_GUIDE.md](TEMPLATE_GUIDE.md) - Comprehensive template documentation
- [Model Implementation Tutorial](../../docs/tutorials/model_implementation.md)
- [Architecture Guide](../../docs/ARCHITECTURE.md)

## 🤝 Contributing

To add new templates or improve existing ones:

1. Add your template to `templates/`
2. Update `generate_model.py` to handle new template
3. Add example configuration
4. Update documentation
5. Submit a pull request

## 📄 License

Same as TrustformeRS project.