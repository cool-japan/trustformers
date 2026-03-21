# TrustformeRS Code Generation Tools

A comprehensive code generation framework for TrustformeRS projects. Generate training scripts, data pipelines, evaluation tools, API bindings, benchmarks, and documentation with a single command.

## 🚀 Quick Start

```bash
# Make scripts executable
chmod +x generate.py

# Generate a training script interactively
./generate.py training MyModel --interactive

# Generate from a configuration file
./generate.py training MyModel -c config.json -o ./output/

# Create example configurations
./generate.py --create-examples
```

## 📋 Available Generators

1. **`training`** - Generate complete training scripts with:
   - Model initialization and configuration
   - Data loading and preprocessing
   - Training loop with validation
   - Checkpointing and logging
   - Distributed training support
   - Mixed precision training

2. **`data`** - Generate data pipeline components:
   - Dataset implementations
   - Data loaders with batching
   - Preprocessing utilities
   - Data augmentation
   - Custom transforms

3. **`evaluation`** - Generate evaluation scripts with:
   - Model evaluation loops
   - Metric calculations
   - Result visualization
   - Prediction export
   - Confusion matrices

4. **`api`** - Generate language bindings:
   - Python bindings (PyO3)
   - JavaScript/TypeScript bindings (wasm-bindgen)
   - C/C++ bindings (cbindgen)
   - Type definitions
   - Setup scripts

5. **`benchmark`** - Generate benchmark suites:
   - Forward pass benchmarks
   - Backward pass benchmarks
   - Memory usage profiling
   - Component-level benchmarks
   - Custom benchmark scenarios

6. **`docs`** - Generate documentation:
   - API reference documentation
   - Usage guides
   - Example galleries
   - Architecture diagrams
   - Migration guides

## 🛠️ Usage Examples

### Training Script Generation

```bash
# Interactive mode - step-by-step configuration
./generate.py training BertClassifier --interactive

# From configuration file
./generate.py training BertClassifier -c configs/bert_training.json

# With command-line options
./generate.py training ImageClassifier \
  --model-type cnn \
  --dataset-type image \
  --output ./training/
```

### Data Pipeline Generation

```bash
# Generate image dataset pipeline
./generate.py data ImageNet \
  --dataset-type image \
  -c configs/imagenet.json \
  -o ./data/

# Generate text dataset with augmentation
./generate.py data TextCorpus \
  --dataset-type text \
  --interactive
```

### API Bindings Generation

```bash
# Generate Python bindings
./generate.py api MyModel \
  --languages python \
  --model-type transformer

# Generate bindings for multiple languages
./generate.py api MyModel \
  --languages python javascript c \
  -c api_config.json
```

### Benchmark Generation

```bash
# Generate comprehensive benchmarks
./generate.py benchmark ModelBench \
  -c benchmark_config.json \
  --verbose

# Quick benchmark setup
./generate.py benchmark QuickBench --interactive
```

## 📁 Configuration Files

Configuration files can be in JSON or YAML format. Here are some examples:

### Training Configuration

```json
{
  "model_type": "transformer",
  "dataset_type": "text",
  "batch_size": 32,
  "learning_rate": 5e-5,
  "num_epochs": 3,
  "optimizer": "adamw",
  "mixed_precision": true,
  "max_length": 512,
  "warmup_steps": 500,
  "scheduler_type": "cosine",
  "use_scheduler": true,
  "distributed": false
}
```

### Data Pipeline Configuration

```json
{
  "dataset_type": "image",
  "preprocessing": true,
  "augmentation": true,
  "image_height": 224,
  "image_width": 224,
  "normalize": true,
  "batch_fields": [
    {"name": "labels", "type": "Tensor"},
    {"name": "metadata", "type": "HashMap<String, String>"}
  ]
}
```

### API Binding Configuration

```json
{
  "languages": ["python", "javascript"],
  "model_type": "transformer",
  "config_fields": [
    {
      "name": "vocab_size",
      "type": "usize",
      "python_type": "int",
      "default": 30522,
      "description": "Vocabulary size"
    },
    {
      "name": "hidden_size",
      "type": "usize",
      "python_type": "int",
      "default": 768,
      "description": "Hidden layer size"
    }
  ],
  "forward_params": [
    {
      "name": "input_ids",
      "type": "&Tensor",
      "description": "Input token IDs"
    }
  ]
}
```

## 🎨 Template System

The code generation framework uses a powerful template system with support for:

### Variables

```rust
// Simple variable substitution
pub struct {{name|pascal_case}}Model {
    config: {{name|pascal_case}}Config,
}
```

### Conditionals

```rust
{{#if use_batch_norm}}
    batch_norm: BatchNorm2d,
{{/if}}
```

### Loops

```rust
{{#each layers}}
Layer {
    in_features: {{in_features}},
    out_features: {{out_features}},
},
{{/each}}
```

### Filters

- `lower` - Convert to lowercase
- `upper` - Convert to uppercase
- `title` - Convert to title case
- `snake_case` - Convert to snake_case
- `camel_case` - Convert to camelCase
- `pascal_case` - Convert to PascalCase

### Includes

```rust
{{#include common/header.rs}}
```

## 🔧 Advanced Features

### Custom Templates

Create your own templates by placing them in a custom template directory:

```bash
mkdir my_templates
# Add your templates...
./generate.py training MyModel -t my_templates/
```

### Dry Run Mode

Preview what will be generated without creating files:

```bash
./generate.py training MyModel --dry-run --verbose
```

### Batch Generation

Generate multiple components at once using a shell script:

```bash
#!/bin/bash
# generate_all.sh

NAME="MyModel"
OUTPUT="./generated"

# Generate all components
./generate.py training $NAME -c config.json -o $OUTPUT/training/
./generate.py data ${NAME}Dataset -c config.json -o $OUTPUT/data/
./generate.py evaluation ${NAME}Eval -c config.json -o $OUTPUT/eval/
./generate.py api $NAME -c config.json -o $OUTPUT/api/
./generate.py benchmark ${NAME}Bench -c config.json -o $OUTPUT/bench/
./generate.py docs $NAME -c config.json -o $OUTPUT/docs/
```

## 📚 Template Variables Reference

### Common Variables

- `{{name}}` - The name provided to the generator
- `{{timestamp}}` - Current timestamp
- `{{generator}}` - Generator type used

### Model-Specific Variables

- `{{model_type}}` - Type of model (transformer, cnn, etc.)
- `{{hidden_size}}` - Hidden dimension size
- `{{num_layers}}` - Number of layers
- `{{num_heads}}` - Number of attention heads (transformer)

### Training Variables

- `{{batch_size}}` - Training batch size
- `{{learning_rate}}` - Initial learning rate
- `{{num_epochs}}` - Number of training epochs
- `{{optimizer}}` - Optimizer type
- `{{scheduler_type}}` - LR scheduler type

### Data Variables

- `{{dataset_type}}` - Type of dataset
- `{{max_length}}` - Maximum sequence length
- `{{image_height}}` / `{{image_width}}` - Image dimensions
- `{{preprocessing}}` - Whether to include preprocessing

## 🤝 Contributing

To add new generators or improve existing ones:

1. Create a new generator class inheriting from `CodeGenerator`
2. Implement the `generate()` method
3. Add templates to `builtin_templates/`
4. Register in `CodeGeneratorFactory`
5. Update documentation

Example:

```python
class MyCustomGenerator(CodeGenerator):
    def generate(self) -> List[str]:
        context = self.get_context()
        template = self.load_template("my_template.rs.template")
        content = self.engine.render(template, context)
        
        output_file = self.output_dir / f"{context['name']}.rs"
        self.write_file(output_file, content)
        
        return [str(output_file)]
```

## 📄 License

Same as TrustformeRS project.