#!/usr/bin/env python3
"""
TrustformeRS Code Generator CLI

A user-friendly command-line interface for generating TrustformeRS code.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add codegen directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from codegen_framework import (
    CodeGeneratorFactory,
    GeneratorConfig,
    load_config_file
)


class InteractiveConfig:
    """Interactive configuration builder"""
    
    def __init__(self):
        self.config = {}
    
    def build(self, generator_type: str) -> Dict[str, Any]:
        """Build configuration interactively"""
        print(f"\n🔧 Interactive configuration for {generator_type} generator\n")
        
        if generator_type == 'training':
            return self._build_training_config()
        elif generator_type == 'data':
            return self._build_data_config()
        elif generator_type == 'evaluation':
            return self._build_eval_config()
        elif generator_type == 'api':
            return self._build_api_config()
        elif generator_type == 'benchmark':
            return self._build_benchmark_config()
        elif generator_type == 'docs':
            return self._build_docs_config()
        else:
            return {}
    
    def _build_training_config(self) -> Dict[str, Any]:
        """Build training script configuration"""
        config = {}
        
        # Model type
        print("Select model type:")
        print("1. Transformer (BERT, GPT, etc.)")
        print("2. CNN (ResNet, EfficientNet, etc.)")
        print("3. RNN (LSTM, GRU)")
        print("4. Custom")
        
        choice = input("\nChoice (1-4): ").strip()
        model_types = ['transformer', 'cnn', 'rnn', 'custom']
        config['model_type'] = model_types[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= 4 else 'custom'
        
        # Dataset type
        print("\nSelect dataset type:")
        print("1. Text")
        print("2. Image")
        print("3. Audio")
        print("4. Custom")
        
        choice = input("\nChoice (1-4): ").strip()
        dataset_types = ['text', 'image', 'audio', 'custom']
        config['dataset_type'] = dataset_types[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= 4 else 'custom'
        
        # Training parameters
        config['batch_size'] = int(input("\nBatch size (default: 32): ").strip() or "32")
        config['learning_rate'] = float(input("Learning rate (default: 0.001): ").strip() or "0.001")
        config['num_epochs'] = int(input("Number of epochs (default: 10): ").strip() or "10")
        
        # Optimizer
        print("\nSelect optimizer:")
        print("1. Adam")
        print("2. SGD")
        print("3. AdamW")
        
        choice = input("\nChoice (1-3): ").strip()
        optimizers = ['adam', 'sgd', 'adamw']
        config['optimizer'] = optimizers[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= 3 else 'adam'
        
        # Additional options
        config['mixed_precision'] = input("\nUse mixed precision training? (y/N): ").strip().lower() == 'y'
        config['distributed'] = input("Use distributed training? (y/N): ").strip().lower() == 'y'
        
        if config['model_type'] == 'transformer':
            config['is_sequence_model'] = True
            config['max_length'] = int(input("\nMaximum sequence length (default: 512): ").strip() or "512")
        
        return config
    
    def _build_data_config(self) -> Dict[str, Any]:
        """Build data pipeline configuration"""
        config = {}
        
        print("Select dataset type:")
        print("1. Text")
        print("2. Image")
        print("3. Audio")
        print("4. Tabular")
        print("5. Custom")
        
        choice = input("\nChoice (1-5): ").strip()
        dataset_types = ['text', 'image', 'audio', 'tabular', 'custom']
        config['dataset_type'] = dataset_types[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= 5 else 'custom'
        
        config['preprocessing'] = input("\nInclude preprocessing utilities? (Y/n): ").strip().lower() != 'n'
        config['augmentation'] = input("Include data augmentation? (Y/n): ").strip().lower() != 'n'
        
        if config['dataset_type'] == 'image':
            config['image_height'] = int(input("\nImage height (default: 224): ").strip() or "224")
            config['image_width'] = int(input("Image width (default: 224): ").strip() or "224")
        
        return config
    
    def _build_eval_config(self) -> Dict[str, Any]:
        """Build evaluation script configuration"""
        config = {}
        
        config['model_type'] = input("Model type (classification/regression/generation): ").strip() or "classification"
        config['generate_metrics'] = input("\nGenerate metrics? (Y/n): ").strip().lower() != 'n'
        
        if config['generate_metrics']:
            print("\nSelect metrics to include:")
            print("1. Accuracy")
            print("2. Precision/Recall/F1")
            print("3. ROC-AUC")
            print("4. MSE/MAE")
            print("5. BLEU/ROUGE")
            
            choices = input("\nChoices (comma-separated, e.g., 1,2,3): ").strip()
            metric_map = {
                '1': {'name': 'accuracy'},
                '2': {'name': 'precision_recall_f1'},
                '3': {'name': 'roc_auc'},
                '4': {'name': 'mse_mae'},
                '5': {'name': 'bleu_rouge'}
            }
            
            config['metrics'] = []
            for choice in choices.split(','):
                choice = choice.strip()
                if choice in metric_map:
                    config['metrics'].append(metric_map[choice])
        
        config['visualization'] = input("\nGenerate visualization scripts? (y/N): ").strip().lower() == 'y'
        
        return config
    
    def _build_api_config(self) -> Dict[str, Any]:
        """Build API binding configuration"""
        config = {}
        
        print("Select languages for API bindings:")
        print("1. Python only")
        print("2. JavaScript/TypeScript only")
        print("3. C/C++ only")
        print("4. Python + JavaScript")
        print("5. All languages")
        
        choice = input("\nChoice (1-5): ").strip()
        language_map = {
            '1': ['python'],
            '2': ['javascript'],
            '3': ['c'],
            '4': ['python', 'javascript'],
            '5': ['python', 'javascript', 'c']
        }
        
        config['languages'] = language_map.get(choice, ['python'])
        config['model_type'] = input("\nModel type (transformer/cnn/custom): ").strip() or "custom"
        
        return config
    
    def _build_benchmark_config(self) -> Dict[str, Any]:
        """Build benchmark configuration"""
        config = {}
        
        config['benchmark_forward'] = True
        config['benchmark_backward'] = input("Benchmark backward pass? (Y/n): ").strip().lower() != 'n'
        config['benchmark_memory'] = input("Benchmark memory usage? (y/N): ").strip().lower() == 'y'
        config['benchmark_components'] = input("Benchmark individual components? (y/N): ").strip().lower() == 'y'
        
        return config
    
    def _build_docs_config(self) -> Dict[str, Any]:
        """Build documentation configuration"""
        config = {}
        
        config['generate_examples'] = input("Generate usage examples? (Y/n): ").strip().lower() != 'n'
        config['model_type'] = input("Model type (for examples): ").strip() or "custom"
        
        return config


def create_example_configs():
    """Create example configuration files"""
    examples_dir = Path("codegen/examples")
    examples_dir.mkdir(parents=True, exist_ok=True)
    
    # Training config example
    training_config = {
        "model_type": "transformer",
        "dataset_type": "text",
        "batch_size": 32,
        "learning_rate": 0.00005,
        "num_epochs": 3,
        "optimizer": "adamw",
        "mixed_precision": True,
        "is_sequence_model": True,
        "max_length": 512,
        "warmup_steps": 500,
        "scheduler_type": "cosine",
        "use_scheduler": True
    }
    
    with open(examples_dir / "training_config.json", 'w') as f:
        json.dump(training_config, f, indent=2)
    
    # Data pipeline config example
    data_config = {
        "dataset_type": "image",
        "preprocessing": True,
        "augmentation": True,
        "supports_augmentation": True,
        "image_height": 224,
        "image_width": 224,
        "batch_fields": [
            {"name": "metadata", "type": "Dict<String, Any>"}
        ]
    }
    
    with open(examples_dir / "data_config.json", 'w') as f:
        json.dump(data_config, f, indent=2)
    
    # API binding config example
    api_config = {
        "languages": ["python", "javascript"],
        "model_type": "transformer",
        "config_fields": [
            {"name": "vocab_size", "python_type": "int", "default": 30522},
            {"name": "hidden_size", "python_type": "int", "default": 768},
            {"name": "num_layers", "python_type": "int", "default": 12}
        ]
    }
    
    with open(examples_dir / "api_config.json", 'w') as f:
        json.dump(api_config, f, indent=2)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="TrustformeRS Code Generator - Generate boilerplate code for TrustformeRS projects",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate training script interactively
  python generate.py training MyModel --interactive
  
  # Generate from config file
  python generate.py training MyModel -c config.json -o ./output/
  
  # Generate data pipeline
  python generate.py data ImageDataset -c image_config.json
  
  # Generate API bindings for multiple languages
  python generate.py api MyModel --languages python javascript
  
  # Create example configurations
  python generate.py --create-examples
        """
    )
    
    parser.add_argument(
        'generator',
        nargs='?',
        choices=CodeGeneratorFactory.list_generators(),
        help='Type of code to generate'
    )
    
    parser.add_argument(
        'name',
        nargs='?',
        help='Name for the generated code'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='.',
        help='Output directory (default: current directory)'
    )
    
    parser.add_argument(
        '-c', '--config',
        help='Configuration file (JSON or YAML)'
    )
    
    parser.add_argument(
        '-t', '--template-dir',
        default='builtin_templates',
        help='Template directory (default: builtin_templates)'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Interactive mode - build configuration step by step'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing files'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be generated without writing files'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    # Quick actions
    parser.add_argument(
        '--list-generators',
        action='store_true',
        help='List available generators'
    )
    
    parser.add_argument(
        '--create-examples',
        action='store_true',
        help='Create example configuration files'
    )
    
    # Generator-specific options
    parser.add_argument(
        '--model-type',
        choices=['transformer', 'cnn', 'rnn', 'custom'],
        help='Model architecture type'
    )
    
    parser.add_argument(
        '--dataset-type',
        choices=['text', 'image', 'audio', 'tabular', 'custom'],
        help='Dataset type'
    )
    
    parser.add_argument(
        '--languages',
        nargs='+',
        choices=['python', 'javascript', 'c'],
        help='Languages for API bindings'
    )
    
    args = parser.parse_args()
    
    # Handle quick actions
    if args.list_generators:
        print("Available generators:")
        for gen in CodeGeneratorFactory.list_generators():
            print(f"  - {gen}")
        return 0
    
    if args.create_examples:
        create_example_configs()
        print("✅ Created example configurations in codegen/examples/")
        return 0
    
    # Validate required arguments
    if not args.generator or not args.name:
        parser.print_help()
        return 1
    
    # Build configuration
    config_dict = {}
    
    # Load from file if provided
    if args.config:
        config_dict = load_config_file(args.config)
    
    # Interactive mode
    elif args.interactive:
        interactive = InteractiveConfig()
        config_dict = interactive.build(args.generator)
    
    # Add command-line options
    if args.model_type:
        config_dict['model_type'] = args.model_type
    if args.dataset_type:
        config_dict['dataset_type'] = args.dataset_type
    if args.languages:
        config_dict['languages'] = args.languages
    
    # Set template directory relative to script location
    template_dir = args.template_dir
    if not os.path.isabs(template_dir):
        template_dir = os.path.join(os.path.dirname(__file__), template_dir)
    
    # Create generator configuration
    gen_config = GeneratorConfig(
        name=args.name,
        output_dir=args.output,
        template_dir=template_dir,
        config=config_dict,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    # Run generator
    try:
        print(f"\n🚀 Generating {args.generator} code for '{args.name}'...\n")
        
        generator = CodeGeneratorFactory.create(args.generator, gen_config)
        generated_files = generator.generate()
        
        print(f"\n✅ Successfully generated {len(generated_files)} files:")
        for file in generated_files:
            print(f"  📄 {file}")
        
        if args.dry_run:
            print("\n[DRY RUN] No files were actually written.")
        else:
            print(f"\n📁 Output directory: {args.output}")
            print("\n🎉 Generation complete! Next steps:")
            
            if args.generator == 'training':
                print("  1. Review and customize the generated training script")
                print("  2. Update the configuration file with your parameters")
                print("  3. Run: cargo run --bin train -- --config config.toml")
            elif args.generator == 'data':
                print("  1. Implement the TODO sections in the dataset")
                print("  2. Add your data loading logic")
                print("  3. Test with: cargo test")
            elif args.generator == 'api':
                print("  1. Build the bindings: cargo build --release")
                if 'python' in config_dict.get('languages', []):
                    print("  2. Install Python package: pip install -e .")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())