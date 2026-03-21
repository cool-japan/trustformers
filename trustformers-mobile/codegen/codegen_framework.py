#!/usr/bin/env python3
"""
TrustformeRS Code Generation Framework

A comprehensive code generation system for TrustformeRS projects.
Generates training scripts, data pipelines, evaluation scripts, API bindings,
benchmarks, and documentation from templates and configurations.
"""

import os
import json
import yaml
import re
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import textwrap
import argparse
from datetime import datetime


@dataclass
class GeneratorConfig:
    """Configuration for code generation"""
    name: str
    output_dir: str
    template_dir: str = "templates"
    config: Dict[str, Any] = field(default_factory=dict)
    overwrite: bool = False
    dry_run: bool = False
    verbose: bool = False


class TemplateEngine:
    """Simple template engine supporting variables, conditionals, and loops"""
    
    def __init__(self):
        self.filters = {
            'lower': str.lower,
            'upper': str.upper,
            'title': str.title,
            'snake_case': self._to_snake_case,
            'camel_case': self._to_camel_case,
            'pascal_case': self._to_pascal_case,
        }
    
    def render(self, template: str, context: Dict[str, Any]) -> str:
        """Render template with context"""
        result = template
        
        # Process includes
        result = self._process_includes(result, context)
        
        # Process loops
        result = self._process_loops(result, context)
        
        # Process conditionals
        result = self._process_conditionals(result, context)
        
        # Process variables with filters
        result = self._process_variables(result, context)
        
        return result
    
    def _process_includes(self, template: str, context: Dict[str, Any]) -> str:
        """Process {{#include file}} directives"""
        include_pattern = r'\{\{#include\s+([^\}]+)\}\}'
        
        def replace_include(match):
            file_path = match.group(1).strip()
            file_path = self._process_variables(file_path, context)
            
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return f.read()
            else:
                return f"// Include file not found: {file_path}"
        
        return re.sub(include_pattern, replace_include, template)
    
    def _process_loops(self, template: str, context: Dict[str, Any]) -> str:
        """Process {{#each collection}} loops"""
        loop_pattern = r'\{\{#each\s+(\w+)\}\}(.*?)\{\{/each\}\}'
        
        def replace_loop(match):
            collection_name = match.group(1)
            loop_body = match.group(2)
            
            if collection_name not in context:
                return ""
            
            collection = context[collection_name]
            if not isinstance(collection, (list, tuple)):
                return ""
            
            results = []
            for idx, item in enumerate(collection):
                loop_context = context.copy()
                loop_context['item'] = item
                loop_context['index'] = idx
                loop_context['first'] = idx == 0
                loop_context['last'] = idx == len(collection) - 1
                
                if isinstance(item, dict):
                    loop_context.update(item)
                
                rendered = self.render(loop_body, loop_context)
                results.append(rendered)
            
            return ''.join(results)
        
        return re.sub(loop_pattern, replace_loop, template, flags=re.DOTALL)
    
    def _process_conditionals(self, template: str, context: Dict[str, Any]) -> str:
        """Process {{#if condition}} blocks"""
        if_pattern = r'\{\{#if\s+([^\}]+)\}\}(.*?)(?:\{\{#else\}\}(.*?))?\{\{/if\}\}'
        
        def replace_conditional(match):
            condition = match.group(1).strip()
            if_body = match.group(2)
            else_body = match.group(3) or ""
            
            # Evaluate condition
            try:
                # Simple evaluation - check if variable exists and is truthy
                if '==' in condition:
                    left, right = condition.split('==')
                    left_val = self._get_variable_value(left.strip(), context)
                    right_val = self._get_variable_value(right.strip(), context)
                    condition_result = left_val == right_val
                elif '!=' in condition:
                    left, right = condition.split('!=')
                    left_val = self._get_variable_value(left.strip(), context)
                    right_val = self._get_variable_value(right.strip(), context)
                    condition_result = left_val != right_val
                else:
                    condition_result = self._get_variable_value(condition, context)
                
                if condition_result:
                    return self.render(if_body, context)
                else:
                    return self.render(else_body, context)
            except:
                return self.render(else_body, context)
        
        return re.sub(if_pattern, replace_conditional, template, flags=re.DOTALL)
    
    def _process_variables(self, template: str, context: Dict[str, Any]) -> str:
        """Process {{variable}} and {{variable|filter}} replacements"""
        var_pattern = r'\{\{([^\}]+)\}\}'
        
        def replace_variable(match):
            var_expr = match.group(1).strip()
            
            # Check for filters
            if '|' in var_expr:
                var_name, filter_name = var_expr.split('|', 1)
                var_name = var_name.strip()
                filter_name = filter_name.strip()
            else:
                var_name = var_expr
                filter_name = None
            
            value = self._get_variable_value(var_name, context)
            
            if filter_name and filter_name in self.filters:
                value = self.filters[filter_name](str(value))
            
            return str(value)
        
        return re.sub(var_pattern, replace_variable, template)
    
    def _get_variable_value(self, var_name: str, context: Dict[str, Any]) -> Any:
        """Get variable value from context, supporting dot notation"""
        if '.' in var_name:
            parts = var_name.split('.')
            value = context
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return ""
            return value
        else:
            return context.get(var_name, "")
    
    def _to_snake_case(self, text: str) -> str:
        """Convert to snake_case"""
        text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', text)
        text = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', text)
        return text.lower()
    
    def _to_camel_case(self, text: str) -> str:
        """Convert to camelCase"""
        parts = text.split('_')
        return parts[0] + ''.join(p.title() for p in parts[1:])
    
    def _to_pascal_case(self, text: str) -> str:
        """Convert to PascalCase"""
        return ''.join(p.title() for p in text.split('_'))


class CodeGenerator(ABC):
    """Abstract base class for code generators"""
    
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.engine = TemplateEngine()
        self.template_dir = Path(config.template_dir)
        self.output_dir = Path(config.output_dir)
        
        if not self.config.dry_run:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def generate(self) -> List[str]:
        """Generate code and return list of generated files"""
        pass
    
    def load_template(self, template_name: str) -> str:
        """Load template from file"""
        template_path = self.template_dir / template_name
        if template_path.exists():
            return template_path.read_text()
        else:
            # Try built-in templates
            builtin_path = Path(__file__).parent / "builtin_templates" / template_name
            if builtin_path.exists():
                return builtin_path.read_text()
            else:
                raise FileNotFoundError(f"Template not found: {template_name}")
    
    def write_file(self, file_path: Path, content: str) -> bool:
        """Write content to file"""
        if self.config.dry_run:
            print(f"[DRY RUN] Would write to: {file_path}")
            if self.config.verbose:
                print(f"Content preview:\n{content[:500]}...\n")
            return True
        
        if file_path.exists() and not self.config.overwrite:
            print(f"[SKIP] File already exists: {file_path}")
            return False
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        
        if self.config.verbose:
            print(f"[WRITE] Generated: {file_path}")
        
        return True
    
    def get_context(self) -> Dict[str, Any]:
        """Get rendering context"""
        context = {
            'name': self.config.name,
            'timestamp': datetime.now().isoformat(),
            'generator': self.__class__.__name__,
        }
        context.update(self.config.config)
        return context


class TrainingScriptGenerator(CodeGenerator):
    """Generate training scripts"""
    
    def generate(self) -> List[str]:
        """Generate training script"""
        context = self.get_context()
        generated_files = []
        
        # Generate main training script
        template = self.load_template("training_script.rs.template")
        content = self.engine.render(template, context)
        
        output_file = self.output_dir / f"{context['name']}_train.rs"
        if self.write_file(output_file, content):
            generated_files.append(str(output_file))
        
        # Generate training configuration
        config_template = self.load_template("training_config.toml.template")
        config_content = self.engine.render(config_template, context)
        
        config_file = self.output_dir / f"{context['name']}_config.toml"
        if self.write_file(config_file, config_content):
            generated_files.append(str(config_file))
        
        # Generate training utilities
        utils_template = self.load_template("training_utils.rs.template")
        utils_content = self.engine.render(utils_template, context)
        
        utils_file = self.output_dir / "training_utils.rs"
        if self.write_file(utils_file, utils_content):
            generated_files.append(str(utils_file))
        
        return generated_files


class DataPipelineGenerator(CodeGenerator):
    """Generate data loading and preprocessing pipelines"""
    
    def generate(self) -> List[str]:
        """Generate data pipeline"""
        context = self.get_context()
        generated_files = []
        
        # Generate dataset implementation
        template = self.load_template("dataset.rs.template")
        content = self.engine.render(template, context)
        
        output_file = self.output_dir / f"{context['name']}_dataset.rs"
        if self.write_file(output_file, content):
            generated_files.append(str(output_file))
        
        # Generate data loader
        loader_template = self.load_template("dataloader.rs.template")
        loader_content = self.engine.render(loader_template, context)
        
        loader_file = self.output_dir / f"{context['name']}_loader.rs"
        if self.write_file(loader_file, loader_content):
            generated_files.append(str(loader_file))
        
        # Generate preprocessing utilities
        if context.get('preprocessing', True):
            preproc_template = self.load_template("preprocessing.rs.template")
            preproc_content = self.engine.render(preproc_template, context)
            
            preproc_file = self.output_dir / "preprocessing.rs"
            if self.write_file(preproc_file, preproc_content):
                generated_files.append(str(preproc_file))
        
        return generated_files


class EvaluationScriptGenerator(CodeGenerator):
    """Generate evaluation and testing scripts"""
    
    def generate(self) -> List[str]:
        """Generate evaluation script"""
        context = self.get_context()
        generated_files = []
        
        # Generate evaluation script
        template = self.load_template("evaluation.rs.template")
        content = self.engine.render(template, context)
        
        output_file = self.output_dir / f"{context['name']}_eval.rs"
        if self.write_file(output_file, content):
            generated_files.append(str(output_file))
        
        # Generate metrics implementation
        metrics_template = self.load_template("metrics.rs.template")
        metrics_content = self.engine.render(metrics_template, context)
        
        metrics_file = self.output_dir / "metrics.rs"
        if self.write_file(metrics_file, metrics_content):
            generated_files.append(str(metrics_file))
        
        # Generate visualization utilities if requested
        if context.get('visualization', False):
            viz_template = self.load_template("visualization.py.template")
            viz_content = self.engine.render(viz_template, context)
            
            viz_file = self.output_dir / "visualize_results.py"
            if self.write_file(viz_file, viz_content):
                generated_files.append(str(viz_file))
        
        return generated_files


class APIBindingGenerator(CodeGenerator):
    """Generate API bindings for different languages"""
    
    def generate(self) -> List[str]:
        """Generate API bindings"""
        context = self.get_context()
        generated_files = []
        
        # Determine target languages
        languages = context.get('languages', ['python'])
        
        for lang in languages:
            if lang == 'python':
                generated_files.extend(self._generate_python_bindings(context))
            elif lang == 'javascript':
                generated_files.extend(self._generate_js_bindings(context))
            elif lang == 'c':
                generated_files.extend(self._generate_c_bindings(context))
        
        return generated_files
    
    def _generate_python_bindings(self, context: Dict[str, Any]) -> List[str]:
        """Generate Python bindings"""
        files = []
        
        # Generate pyo3 bindings
        template = self.load_template("python_bindings.rs.template")
        content = self.engine.render(template, context)
        
        output_file = self.output_dir / "src" / "python.rs"
        if self.write_file(output_file, content):
            files.append(str(output_file))
        
        # Generate Python wrapper
        py_template = self.load_template("python_wrapper.py.template")
        py_content = self.engine.render(py_template, context)
        
        py_file = self.output_dir / f"{context['name']}.py"
        if self.write_file(py_file, py_content):
            files.append(str(py_file))
        
        # Generate setup.py
        setup_template = self.load_template("setup.py.template")
        setup_content = self.engine.render(setup_template, context)
        
        setup_file = self.output_dir / "setup.py"
        if self.write_file(setup_file, setup_content):
            files.append(str(setup_file))
        
        return files
    
    def _generate_js_bindings(self, context: Dict[str, Any]) -> List[str]:
        """Generate JavaScript/WASM bindings"""
        files = []
        
        # Generate wasm-bindgen bindings
        template = self.load_template("wasm_bindings.rs.template")
        content = self.engine.render(template, context)
        
        output_file = self.output_dir / "src" / "wasm.rs"
        if self.write_file(output_file, content):
            files.append(str(output_file))
        
        # Generate TypeScript definitions
        ts_template = self.load_template("typescript_defs.d.ts.template")
        ts_content = self.engine.render(ts_template, context)
        
        ts_file = self.output_dir / f"{context['name']}.d.ts"
        if self.write_file(ts_file, ts_content):
            files.append(str(ts_file))
        
        return files
    
    def _generate_c_bindings(self, context: Dict[str, Any]) -> List[str]:
        """Generate C bindings"""
        files = []
        
        # Generate cbindgen configuration
        template = self.load_template("c_bindings.rs.template")
        content = self.engine.render(template, context)
        
        output_file = self.output_dir / "src" / "c_api.rs"
        if self.write_file(output_file, content):
            files.append(str(output_file))
        
        # Generate header file template
        header_template = self.load_template("c_header.h.template")
        header_content = self.engine.render(header_template, context)
        
        header_file = self.output_dir / f"{context['name']}.h"
        if self.write_file(header_file, header_content):
            files.append(str(header_file))
        
        return files


class BenchmarkGenerator(CodeGenerator):
    """Generate benchmark suites"""
    
    def generate(self) -> List[str]:
        """Generate benchmarks"""
        context = self.get_context()
        generated_files = []
        
        # Generate criterion benchmarks
        template = self.load_template("benchmarks.rs.template")
        content = self.engine.render(template, context)
        
        output_file = self.output_dir / "benches" / f"{context['name']}_bench.rs"
        if self.write_file(output_file, content):
            generated_files.append(str(output_file))
        
        # Generate benchmark utilities
        utils_template = self.load_template("bench_utils.rs.template")
        utils_content = self.engine.render(utils_template, context)
        
        utils_file = self.output_dir / "benches" / "utils.rs"
        if self.write_file(utils_file, utils_content):
            generated_files.append(str(utils_file))
        
        # Generate benchmark configuration
        config_template = self.load_template("bench_config.toml.template")
        config_content = self.engine.render(config_template, context)
        
        config_file = self.output_dir / "bench.toml"
        if self.write_file(config_file, config_content):
            generated_files.append(str(config_file))
        
        return generated_files


class DocumentationGenerator(CodeGenerator):
    """Generate documentation from code"""
    
    def generate(self) -> List[str]:
        """Generate documentation"""
        context = self.get_context()
        generated_files = []
        
        # Generate API documentation
        api_template = self.load_template("api_docs.md.template")
        api_content = self.engine.render(api_template, context)
        
        api_file = self.output_dir / "API.md"
        if self.write_file(api_file, api_content):
            generated_files.append(str(api_file))
        
        # Generate usage guide
        usage_template = self.load_template("usage_guide.md.template")
        usage_content = self.engine.render(usage_template, context)
        
        usage_file = self.output_dir / "USAGE.md"
        if self.write_file(usage_file, usage_content):
            generated_files.append(str(usage_file))
        
        # Generate examples documentation
        if context.get('generate_examples', True):
            examples_template = self.load_template("examples_docs.md.template")
            examples_content = self.engine.render(examples_template, context)
            
            examples_file = self.output_dir / "EXAMPLES.md"
            if self.write_file(examples_file, examples_content):
                generated_files.append(str(examples_file))
        
        return generated_files


class CodeGeneratorFactory:
    """Factory for creating code generators"""
    
    generators = {
        'training': TrainingScriptGenerator,
        'data': DataPipelineGenerator,
        'evaluation': EvaluationScriptGenerator,
        'api': APIBindingGenerator,
        'benchmark': BenchmarkGenerator,
        'docs': DocumentationGenerator,
    }
    
    @classmethod
    def create(cls, generator_type: str, config: GeneratorConfig) -> CodeGenerator:
        """Create a code generator instance"""
        if generator_type not in cls.generators:
            raise ValueError(f"Unknown generator type: {generator_type}")
        
        generator_class = cls.generators[generator_type]
        return generator_class(config)
    
    @classmethod
    def list_generators(cls) -> List[str]:
        """List available generator types"""
        return list(cls.generators.keys())


def load_config_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file"""
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        if path.suffix == '.json':
            return json.load(f)
        elif path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="TrustformeRS Code Generation Framework"
    )
    
    parser.add_argument(
        'generator',
        choices=CodeGeneratorFactory.list_generators(),
        help='Type of code to generate'
    )
    
    parser.add_argument(
        'name',
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
        default='templates',
        help='Template directory'
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
    
    # Additional options
    parser.add_argument(
        '--model-type',
        help='Model type (transformer, cnn, custom)'
    )
    
    parser.add_argument(
        '--dataset-type',
        help='Dataset type (text, image, audio, custom)'
    )
    
    parser.add_argument(
        '--languages',
        nargs='+',
        help='Languages for API bindings (python, javascript, c)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_dict = {}
    if args.config:
        config_dict = load_config_file(args.config)
    
    # Add command-line options to config
    if args.model_type:
        config_dict['model_type'] = args.model_type
    if args.dataset_type:
        config_dict['dataset_type'] = args.dataset_type
    if args.languages:
        config_dict['languages'] = args.languages
    
    # Create generator configuration
    gen_config = GeneratorConfig(
        name=args.name,
        output_dir=args.output,
        template_dir=args.template_dir,
        config=config_dict,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    
    # Create and run generator
    try:
        generator = CodeGeneratorFactory.create(args.generator, gen_config)
        generated_files = generator.generate()
        
        print(f"\n✅ Generated {len(generated_files)} files:")
        for file in generated_files:
            print(f"  - {file}")
        
        if args.dry_run:
            print("\n[DRY RUN] No files were actually written.")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())