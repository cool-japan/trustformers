#!/usr/bin/env python3
"""
TrustformeRS Model Generator

This script generates new model implementations from templates.

Usage:
    python generate_model.py --type transformer --name BERT --output ../bert/
    python generate_model.py --type cnn --name ResNet --output ../resnet/
    python generate_model.py --type custom --name GraphNet --config graph_config.json
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List

class ModelGenerator:
    """Generate model implementations from templates."""
    
    def __init__(self):
        self.template_dir = Path(__file__).parent
        self.templates = {
            'transformer': 'transformer_model_template.rs',
            'cnn': 'cnn_model_template.rs',
            'custom': 'custom_model_template.rs',
        }
    
    def generate(self, model_type: str, model_name: str, config: Dict[str, Any], output_dir: str):
        """Generate a model implementation."""
        # Load template
        template_path = self.template_dir / self.templates[model_type]
        with open(template_path, 'r') as f:
            template = f.read()
        
        # Prepare variables
        variables = self._prepare_variables(model_type, model_name, config)
        
        # Replace template variables
        generated = self._replace_variables(template, variables)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Write generated files
        self._write_files(output_path, model_name, generated, config)
        
        print(f"✅ Successfully generated {model_name} in {output_path}")
    
    def _prepare_variables(self, model_type: str, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare template variables."""
        snake_case = self._to_snake_case(model_name)
        
        base_vars = {
            'MODEL_NAME': model_name,
            'ModelName': model_name,
            'model_name_snake': snake_case,
            'MODEL_DESCRIPTION': config.get('description', f'{model_name} model implementation'),
            'ARCHITECTURE_DETAILS': config.get('architecture_details', 'Custom architecture'),
            'model_id': config.get('model_id', f'trustformers/{snake_case}-base'),
        }
        
        if model_type == 'transformer':
            return {**base_vars, **self._prepare_transformer_vars(config)}
        elif model_type == 'cnn':
            return {**base_vars, **self._prepare_cnn_vars(config)}
        else:
            return {**base_vars, **self._prepare_custom_vars(config)}
    
    def _prepare_transformer_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare transformer-specific variables."""
        return {
            # Model architecture
            'is_decoder': str(config.get('is_decoder', False)).lower(),
            'is_encoder_decoder': str(config.get('is_encoder_decoder', False)).lower(),
            'has_cross_attention': config.get('is_decoder', False),
            'has_pooler': config.get('has_pooler', True),
            'use_position_embeddings': config.get('use_position_embeddings', True),
            'use_token_type_embeddings': config.get('use_token_type_embeddings', True),
            'use_final_layer_norm': config.get('use_final_layer_norm', False),
            
            # Default configuration values
            'default_vocab_size': config.get('vocab_size', 30522),
            'default_hidden_size': config.get('hidden_size', 768),
            'default_num_layers': config.get('num_layers', 12),
            'default_num_heads': config.get('num_heads', 12),
            'default_intermediate_size': config.get('intermediate_size', 3072),
            'default_activation': config.get('activation', 'GELU'),
            'default_max_positions': config.get('max_positions', 512),
            'default_type_vocab_size': config.get('type_vocab_size', 2),
            'default_bos_token_id': config.get('bos_token_id', 'None'),
            'default_eos_token_id': config.get('eos_token_id', 'None'),
            
            # Custom parameters
            'has_custom_params': bool(config.get('custom_params')),
            'custom_params': self._format_custom_params(config.get('custom_params', [])),
            'custom_params_defaults': self._format_custom_defaults(config.get('custom_params', [])),
        }
    
    def _prepare_cnn_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare CNN-specific variables."""
        layers = config.get('layers', [
            {'out_channels': 64, 'kernel_size': 7, 'stride': 2, 'padding': 3, 'num_blocks': 3},
            {'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'num_blocks': 4},
            {'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'num_blocks': 6},
            {'out_channels': 512, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'num_blocks': 3},
        ])
        
        return {
            # Architecture flags
            'has_stem': config.get('has_stem', False),
            'has_residual': config.get('has_residual', True),
            'has_depthwise': config.get('has_depthwise', False),
            'has_se_module': config.get('has_se_module', False),
            'has_neck': config.get('has_neck', False),
            'use_batch_norm': config.get('use_batch_norm', True),
            'supports_detection': config.get('supports_detection', False),
            'supports_segmentation': config.get('supports_segmentation', False),
            
            # Default values
            'default_num_classes': config.get('num_classes', 1000),
            'default_image_height': config.get('image_height', 224),
            'default_image_width': config.get('image_width', 224),
            'default_activation': config.get('activation', 'ReLU'),
            'default_dropout': config.get('dropout', 0.2),
            'default_use_bn': str(config.get('use_batch_norm', True)).lower(),
            'default_layers': layers,
            'num_default_layers': len(layers),
            
            # Custom parameters
            'has_custom_params': bool(config.get('custom_params')),
            'custom_params': self._format_custom_params(config.get('custom_params', [])),
            'has_custom_layer_params': bool(config.get('custom_layer_params')),
            'custom_layer_params': self._format_custom_params(config.get('custom_layer_params', [])),
        }
    
    def _prepare_custom_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare custom model variables."""
        return {
            # Configuration
            'config_params': config.get('config_params', []),
            'config_defaults': config.get('config_defaults', []),
            'hidden_size_expr': config.get('hidden_size_expr', 'self.hidden_dim'),
            'has_labels': config.get('has_labels', False),
            
            # Architecture
            'components': config.get('components', []),
            'model_components': config.get('model_components', []),
            'model_initializers': config.get('model_initializers', []),
            
            # Forward pass
            'forward_args': config.get('forward_args', [
                {'name': 'inputs', 'type': '&Tensor'},
            ]),
            'main_forward_implementation': config.get('forward_impl', '// TODO: Implement forward pass'),
            
            # Output
            'output_fields': config.get('output_fields', [
                {'name': 'output', 'type': 'Tensor', 'description': 'Model output'},
            ]),
            'has_logits': config.get('has_logits', False),
            'has_loss': config.get('has_loss', False),
            'returns_hidden_states': config.get('returns_hidden_states', False),
            'returns_attentions': config.get('returns_attentions', False),
            
            # Methods
            'additional_methods': config.get('additional_methods', []),
            'default_forward_call': config.get('default_forward_call', 
                'self.forward(input_ids, attention_mask, None, false)'),
            
            # Parameters
            'parameter_collection': config.get('parameter_collection', []),
            'named_parameter_collection': config.get('named_parameter_collection', []),
            
            # Specialized heads
            'has_specialized_heads': bool(config.get('specialized_heads')),
            'specialized_heads': config.get('specialized_heads', []),
            
            # Utilities
            'utility_functions': config.get('utility_functions', []),
            
            # Tests
            'config_tests': config.get('config_tests', []),
            'test_config_overrides': config.get('test_config_overrides', []),
            'test_forward_setup': config.get('test_forward_setup', ''),
            'test_forward_args': config.get('test_forward_args', '&inputs, None'),
            'output_tests': config.get('output_tests', []),
            'additional_tests': config.get('additional_tests', []),
            
            # Misc
            'default_activation': config.get('activation', 'GELU'),
        }
    
    def _replace_variables(self, template: str, variables: Dict[str, Any]) -> str:
        """Replace template variables with actual values."""
        result = template
        
        # Handle conditional blocks
        result = self._process_conditionals(result, variables)
        
        # Handle loops
        result = self._process_loops(result, variables)
        
        # Replace simple variables
        for key, value in variables.items():
            result = result.replace(f'{{{{{key}}}}}', str(value))
        
        return result
    
    def _process_conditionals(self, template: str, variables: Dict[str, Any]) -> str:
        """Process conditional blocks in template."""
        # Pattern: {{#if condition}} content {{/if}}
        pattern = r'\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\}'
        
        def replacer(match):
            condition = match.group(1)
            content = match.group(2)
            
            # Check if condition is true
            if variables.get(condition, False):
                # Process nested conditionals
                return self._process_conditionals(content, variables)
            else:
                # Check for else block
                else_pattern = r'\{\{else\}\}'
                if else_pattern in content:
                    parts = re.split(else_pattern, content)
                    return self._process_conditionals(parts[1], variables)
                return ''
        
        return re.sub(pattern, replacer, template, flags=re.DOTALL)
    
    def _process_loops(self, template: str, variables: Dict[str, Any]) -> str:
        """Process loop blocks in template."""
        # Pattern: {{#each collection}} content {{/each}}
        pattern = r'\{\{#each\s+(\w+)\}\}(.*?)\{\{/each\}\}'
        
        def replacer(match):
            collection_name = match.group(1)
            content = match.group(2)
            collection = variables.get(collection_name, [])
            
            result = []
            for i, item in enumerate(collection):
                item_content = content
                
                # Replace item variables
                if isinstance(item, dict):
                    for key, value in item.items():
                        item_content = item_content.replace(f'{{{{{key}}}}}', str(value))
                
                # Replace special variables
                item_content = item_content.replace('{{@index}}', str(i))
                item_content = item_content.replace('{{@first}}', str(i == 0).lower())
                item_content = item_content.replace('{{@last}}', str(i == len(collection) - 1).lower())
                
                # Handle unless
                unless_pattern = r'\{\{#unless\s+@last\}\}(.*?)\{\{/unless\}\}'
                if i == len(collection) - 1:
                    item_content = re.sub(unless_pattern, '', item_content)
                else:
                    item_content = re.sub(unless_pattern, r'\1', item_content)
                
                result.append(item_content)
            
            return '\n'.join(result)
        
        return re.sub(pattern, replacer, template, flags=re.DOTALL)
    
    def _format_custom_params(self, params: List[Dict[str, Any]]) -> str:
        """Format custom parameters for config struct."""
        if not params:
            return ''
        
        lines = []
        for param in params:
            doc = f"/// {param.get('description', '')}"
            field = f"pub {param['name']}: {param['type']},"
            lines.extend([doc, field])
        
        return '\n    '.join(lines)
    
    def _format_custom_defaults(self, params: List[Dict[str, Any]]) -> str:
        """Format custom parameter defaults."""
        if not params:
            return ''
        
        lines = []
        for param in params:
            default = f"{param['name']}: {param.get('default', 'Default::default()')},"
            lines.append(default)
        
        return '\n            '.join(lines)
    
    def _to_snake_case(self, name: str) -> str:
        """Convert CamelCase to snake_case."""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def _write_files(self, output_dir: Path, model_name: str, generated: str, config: Dict[str, Any]):
        """Write generated files to output directory."""
        snake_name = self._to_snake_case(model_name)
        
        # Create mod.rs
        mod_content = f'''//! {model_name} model implementation

mod model;
pub use model::*;

#[cfg(test)]
mod tests;
'''
        (output_dir / 'mod.rs').write_text(mod_content)
        
        # Create model.rs
        (output_dir / 'model.rs').write_text(generated)
        
        # Create tests.rs
        test_content = f'''//! Tests for {model_name}

use super::*;
use trustformers_core::{{Result, Tensor}};

#[test]
fn test_{snake_name}_shapes() {{
    let config = {model_name}Config::default();
    let model = {model_name}::new(config).unwrap();
    
    // Add shape tests
}}

#[test]
fn test_{snake_name}_gradient_flow() {{
    // Test gradient flow through model
}}
'''
        (output_dir / 'tests.rs').write_text(test_content)
        
        # Create example usage
        example_content = f'''//! Example usage of {model_name}

use trustformers_models::{snake_name}::{{{model_name}, {model_name}Config}};
use trustformers_core::{{Result, Tensor}};

fn main() -> Result<()> {{
    // Create model configuration
    let config = {model_name}Config::default();
    
    // Initialize model
    let model = {model_name}::new(config)?;
    
    // Create dummy input
    let input = Tensor::randn(&[1, 512])?; // Adjust shape as needed
    
    // Forward pass
    let output = model.forward(&input, None)?;
    
    println!("Output shape: {{:?}}", output.shape());
    
    Ok(())
}}
'''
        examples_dir = output_dir / 'examples'
        examples_dir.mkdir(exist_ok=True)
        (examples_dir / f'{snake_name}_example.rs').write_text(example_content)
        
        # Create README
        readme_content = f'''# {model_name}

{config.get('description', f'{model_name} model implementation for TrustformeRS')}

## Architecture

{config.get('architecture_details', 'Custom architecture implementation')}

## Usage

```rust
use trustformers_models::{snake_name}::{{{model_name}, {model_name}Config}};

// Load pretrained model
let model = {model_name}::from_pretrained("{config.get('model_id', f'trustformers/{snake_name}-base')}")?;

// Or create with custom config
let config = {model_name}Config {{
    hidden_size: 768,
    num_layers: 12,
    ..Default::default()
}};
let model = {model_name}::new(config)?;
```

## Configuration

See `{model_name}Config` for all available configuration options.

## References

{config.get('references', '- Original paper: [TODO: Add reference]')}
'''
        (output_dir / 'README.md').write_text(readme_content)


def main():
    parser = argparse.ArgumentParser(description='Generate TrustformeRS model implementations')
    parser.add_argument('--type', choices=['transformer', 'cnn', 'custom'], required=True,
                       help='Type of model to generate')
    parser.add_argument('--name', required=True, help='Model name (e.g., BERT, ResNet)')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--config', help='JSON configuration file')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    elif args.interactive:
        config = interactive_config(args.type)
    else:
        config = get_default_config(args.type)
    
    # Generate model
    generator = ModelGenerator()
    generator.generate(args.type, args.name, config, args.output)


def get_default_config(model_type: str) -> Dict[str, Any]:
    """Get default configuration for model type."""
    if model_type == 'transformer':
        return {
            'description': 'Transformer-based model',
            'vocab_size': 30522,
            'hidden_size': 768,
            'num_layers': 12,
            'num_heads': 12,
            'intermediate_size': 3072,
            'activation': 'GELU',
            'max_positions': 512,
            'is_decoder': False,
            'has_pooler': True,
        }
    elif model_type == 'cnn':
        return {
            'description': 'Convolutional Neural Network',
            'num_classes': 1000,
            'image_height': 224,
            'image_width': 224,
            'activation': 'ReLU',
            'has_residual': True,
            'use_batch_norm': True,
            'layers': [
                {'out_channels': 64, 'kernel_size': 7, 'stride': 2, 'padding': 3, 'num_blocks': 3},
                {'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'num_blocks': 4},
                {'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'num_blocks': 6},
                {'out_channels': 512, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'num_blocks': 3},
            ],
        }
    else:
        return {
            'description': 'Custom model architecture',
            'config_params': [
                {'name': 'hidden_dim', 'type': 'usize', 'description': 'Hidden dimension'},
                {'name': 'num_layers', 'type': 'usize', 'description': 'Number of layers'},
            ],
            'config_defaults': [
                {'name': 'hidden_dim', 'value': '256'},
                {'name': 'num_layers', 'value': '4'},
            ],
        }


def interactive_config(model_type: str) -> Dict[str, Any]:
    """Interactive configuration builder."""
    print(f"\n🔧 Configuring {model_type} model...\n")
    
    config = {}
    
    # Common questions
    config['description'] = input("Model description: ")
    
    if model_type == 'transformer':
        config['vocab_size'] = int(input("Vocabulary size [30522]: ") or 30522)
        config['hidden_size'] = int(input("Hidden size [768]: ") or 768)
        config['num_layers'] = int(input("Number of layers [12]: ") or 12)
        config['num_heads'] = int(input("Number of attention heads [12]: ") or 12)
        config['is_decoder'] = input("Is decoder model? [y/N]: ").lower() == 'y'
        config['has_pooler'] = input("Has pooler? [Y/n]: ").lower() != 'n'
    
    elif model_type == 'cnn':
        config['num_classes'] = int(input("Number of classes [1000]: ") or 1000)
        config['image_height'] = int(input("Image height [224]: ") or 224)
        config['image_width'] = int(input("Image width [224]: ") or 224)
        config['has_residual'] = input("Use residual connections? [Y/n]: ").lower() != 'n'
        config['has_stem'] = input("Use stem? [y/N]: ").lower() == 'y'
        
        # Layer configuration
        num_stages = int(input("Number of stages [4]: ") or 4)
        config['layers'] = []
        for i in range(num_stages):
            print(f"\nStage {i+1}:")
            out_channels = int(input(f"  Output channels [{64 * (2**i)}]: ") or 64 * (2**i))
            num_blocks = int(input(f"  Number of blocks [{3 if i < 3 else 2}]: ") or (3 if i < 3 else 2))
            config['layers'].append({
                'out_channels': out_channels,
                'kernel_size': 3,
                'stride': 2 if i > 0 else 1,
                'padding': 1,
                'num_blocks': num_blocks,
            })
    
    else:  # custom
        # Custom configuration
        print("\nDefine configuration parameters:")
        config['config_params'] = []
        config['config_defaults'] = []
        
        while True:
            param_name = input("\nParameter name (empty to finish): ")
            if not param_name:
                break
            
            param_type = input("Parameter type [usize]: ") or "usize"
            param_desc = input("Parameter description: ")
            param_default = input("Default value: ")
            
            config['config_params'].append({
                'name': param_name,
                'type': param_type,
                'description': param_desc,
            })
            config['config_defaults'].append({
                'name': param_name,
                'value': param_default,
            })
    
    return config


if __name__ == '__main__':
    main()