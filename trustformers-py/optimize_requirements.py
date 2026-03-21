#!/usr/bin/env python3
"""
Requirements optimization script for trustformers-py

This script optimizes Python dependencies for different use cases:
- Core requirements (minimal installation)
- Full requirements (all features)
- Development requirements
- Platform-specific optimizations
"""

import os
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple

class RequirementsOptimizer:
    """Requirements optimizer for different deployment scenarios"""
    
    def __init__(self, package_dir: str = None):
        """Initialize requirements optimizer
        
        Args:
            package_dir: Path to package directory
        """
        self.package_dir = Path(package_dir or os.getcwd())
        self.pyproject_toml = self.package_dir / "pyproject.toml"
        
        # Core dependencies (always required)
        self.core_deps = {
            "numpy": ">=1.21.0",
            "typing-extensions": ">=4.0.0"
        }
        
        # Optional dependencies by category
        self.optional_deps = {
            "torch": {
                "torch": ">=1.9.0",
                "torchvision": ">=0.10.0",
            },
            "jax": {
                "jax[cpu]": ">=0.3.0",
                "jaxlib": ">=0.3.0",
            },
            "serving": {
                "fastapi": ">=0.68.0",
                "uvicorn[standard]": ">=0.15.0",
                "prometheus-client": ">=0.11.0",
            },
            "mlops": {
                "mlflow": ">=1.20.0",
                "wandb": ">=0.12.0",
                "tensorboard": ">=2.7.0",
            },
            "data": {
                "pandas": ">=1.3.0",
                "scikit-learn": ">=1.0.0",
                "matplotlib": ">=3.4.0",
                "seaborn": ">=0.11.0",
                "plotly": ">=5.0.0",
            },
            "jupyter": {
                "jupyter": ">=1.0.0",
                "ipywidgets": ">=7.6.0",
                "tqdm": ">=4.60.0",
            },
            "dev": {
                "pytest": ">=6.2.0",
                "pytest-cov": ">=2.12.0",
                "pytest-asyncio": ">=0.15.0",
                "pytest-xdist": ">=2.3.0",
                "black": ">=21.6.0",
                "isort": ">=5.9.0",
                "flake8": ">=3.9.0",
                "mypy": ">=0.910",
                "bandit": ">=1.7.0",
                "ruff": ">=0.0.90",
            },
            "build": {
                "maturin": ">=0.13.0",
                "build": ">=0.7.0",
                "wheel": ">=0.37.0",
                "twine": ">=3.4.0",
            }
        }
        
        # Platform-specific optimizations
        self.platform_optimizations = {
            "linux": {
                "torch": "torch>=1.9.0+cpu --index-url https://download.pytorch.org/whl/cpu",
                "jax": "jax[cpu]>=0.3.0",
            },
            "macos": {
                "torch": "torch>=1.9.0 --index-url https://download.pytorch.org/whl/cpu",
                "jax": "jax[cpu]>=0.3.0",
            },
            "windows": {
                "torch": "torch>=1.9.0+cpu --index-url https://download.pytorch.org/whl/cpu",
                "jax": "jax[cpu]>=0.3.0",
            }
        }
        
        # Conflict resolution
        self.conflicts = {
            ("tensorflow", "jax"): "Cannot install both TensorFlow and JAX simultaneously",
            ("torch", "tensorflow"): "Potential memory conflicts between PyTorch and TensorFlow"
        }
    
    def detect_platform(self) -> str:
        """Detect current platform"""
        import platform
        system = platform.system().lower()
        if system == "darwin":
            return "macos"
        return system
    
    def resolve_conflicts(self, deps: Dict[str, str]) -> Dict[str, str]:
        """Resolve dependency conflicts
        
        Args:
            deps: Dictionary of dependencies
            
        Returns:
            Resolved dependencies
        """
        resolved = deps.copy()
        
        for (pkg1, pkg2), message in self.conflicts.items():
            if pkg1 in resolved and pkg2 in resolved:
                print(f"Warning: {message}")
                print(f"Removing {pkg2} to resolve conflict")
                del resolved[pkg2]
        
        return resolved
    
    def create_core_requirements(self) -> Dict[str, str]:
        """Create core requirements file
        
        Returns:
            Core dependencies
        """
        return self.core_deps.copy()
    
    def create_full_requirements(self) -> Dict[str, str]:
        """Create full requirements with all optional dependencies
        
        Returns:
            Full dependencies
        """
        full_deps = self.core_deps.copy()
        
        # Add all optional dependencies except dev and build
        for category, deps in self.optional_deps.items():
            if category not in ["dev", "build"]:
                full_deps.update(deps)
        
        return self.resolve_conflicts(full_deps)
    
    def create_minimal_requirements(self) -> Dict[str, str]:
        """Create minimal requirements for basic functionality
        
        Returns:
            Minimal dependencies
        """
        minimal = self.core_deps.copy()
        
        # Add only essential packages
        minimal.update({
            "requests": ">=2.25.0",  # For model downloading
        })
        
        return minimal
    
    def create_category_requirements(self, categories: List[str]) -> Dict[str, str]:
        """Create requirements for specific categories
        
        Args:
            categories: List of category names
            
        Returns:
            Category-specific dependencies
        """
        deps = self.core_deps.copy()
        
        for category in categories:
            if category in self.optional_deps:
                deps.update(self.optional_deps[category])
        
        return self.resolve_conflicts(deps)
    
    def optimize_for_platform(self, deps: Dict[str, str], platform: str = None) -> Dict[str, str]:
        """Optimize dependencies for specific platform
        
        Args:
            deps: Base dependencies
            platform: Target platform
            
        Returns:
            Platform-optimized dependencies
        """
        if platform is None:
            platform = self.detect_platform()
        
        optimized = deps.copy()
        
        if platform in self.platform_optimizations:
            platform_opts = self.platform_optimizations[platform]
            
            # Apply platform-specific optimizations
            for pkg, optimized_spec in platform_opts.items():
                if any(pkg in dep for dep in optimized.keys()):
                    # Remove generic version and add optimized
                    keys_to_remove = [k for k in optimized.keys() if pkg in k]
                    for key in keys_to_remove:
                        del optimized[key]
                    optimized[pkg] = optimized_spec
        
        return optimized
    
    def analyze_dependency_tree(self, deps: Dict[str, str]) -> Dict[str, List[str]]:
        """Analyze dependency tree to find conflicts and redundancies
        
        Args:
            deps: Dependencies to analyze
            
        Returns:
            Dependency tree analysis
        """
        analysis = {
            "direct": list(deps.keys()),
            "conflicts": [],
            "redundant": [],
            "missing": []
        }
        
        # Check for known conflicts
        for (pkg1, pkg2), message in self.conflicts.items():
            if pkg1 in deps and pkg2 in deps:
                analysis["conflicts"].append(f"{pkg1} vs {pkg2}: {message}")
        
        # Check for redundant packages (simplified)
        redundant_patterns = [
            (["matplotlib", "seaborn"], "seaborn includes matplotlib"),
            (["numpy", "pandas"], "pandas includes numpy"),
        ]
        
        for packages, reason in redundant_patterns:
            if all(pkg in deps for pkg in packages):
                analysis["redundant"].append(f"{packages}: {reason}")
        
        return analysis
    
    def generate_requirements_txt(self, deps: Dict[str, str], 
                                 filename: str = "requirements.txt",
                                 include_hashes: bool = False) -> Path:
        """Generate requirements.txt file
        
        Args:
            deps: Dependencies dictionary
            filename: Output filename
            include_hashes: Whether to include package hashes
            
        Returns:
            Path to generated file
        """
        output_path = self.package_dir / filename
        
        with open(output_path, "w") as f:
            f.write("# Generated requirements file\n")
            f.write("# Install with: pip install -r requirements.txt\n\n")
            
            for package, version in sorted(deps.items()):
                if "--index-url" in version:
                    # Handle special index URLs
                    parts = version.split(" --index-url ")
                    f.write(f"{package}=={parts[0].split('>=')[1]} --index-url {parts[1]}\n")
                else:
                    f.write(f"{package}{version}\n")
        
        print(f"Generated requirements file: {output_path}")
        return output_path
    
    def generate_pyproject_toml_deps(self, deps_by_category: Dict[str, Dict[str, str]]) -> str:
        """Generate pyproject.toml dependencies section
        
        Args:
            deps_by_category: Dependencies organized by category
            
        Returns:
            TOML format string
        """
        toml_content = []
        
        # Core dependencies
        if "core" in deps_by_category:
            core_deps = deps_by_category["core"]
            toml_content.append("dependencies = [")
            for package, version in sorted(core_deps.items()):
                toml_content.append(f'    "{package}{version}",')
            toml_content.append("]")
            toml_content.append("")
        
        # Optional dependencies
        toml_content.append("[project.optional-dependencies]")
        
        for category, deps in deps_by_category.items():
            if category == "core":
                continue
                
            toml_content.append(f'{category} = [')
            for package, version in sorted(deps.items()):
                if "--index-url" not in version:  # Skip complex URLs in pyproject.toml
                    toml_content.append(f'    "{package}{version}",')
            toml_content.append("]")
            toml_content.append("")
        
        # Convenience groups
        toml_content.append('all = [')
        for category in deps_by_category.keys():
            if category != "core":
                toml_content.append(f'    "trustformers[{category}]",')
        toml_content.append("]")
        
        return "\n".join(toml_content)
    
    def create_conda_environment(self, deps: Dict[str, str], 
                                name: str = "trustformers") -> Path:
        """Create conda environment.yml file
        
        Args:
            deps: Dependencies
            name: Environment name
            
        Returns:
            Path to environment file
        """
        env_data = {
            "name": name,
            "channels": ["conda-forge", "pytorch", "nvidia"],
            "dependencies": []
        }
        
        # Convert pip dependencies to conda when possible
        conda_mappings = {
            "torch": "pytorch",
            "numpy": "numpy",
            "pandas": "pandas",
            "scikit-learn": "scikit-learn",
            "matplotlib": "matplotlib",
            "jupyter": "jupyter",
        }
        
        conda_deps = []
        pip_deps = []
        
        for package, version in deps.items():
            if package in conda_mappings:
                conda_name = conda_mappings[package]
                # Convert version spec
                version_spec = version.replace(">=", ">=")
                conda_deps.append(f"{conda_name}{version_spec}")
            else:
                pip_deps.append(f"{package}{version}")
        
        env_data["dependencies"].extend(conda_deps)
        
        if pip_deps:
            env_data["dependencies"].append({
                "pip": pip_deps
            })
        
        # Write environment file
        import yaml
        output_path = self.package_dir / f"environment-{name}.yml"
        
        with open(output_path, "w") as f:
            yaml.dump(env_data, f, default_flow_style=False)
        
        print(f"Generated conda environment: {output_path}")
        return output_path
    
    def benchmark_installation_size(self, deps: Dict[str, str]) -> Dict[str, float]:
        """Benchmark installation size for dependencies
        
        Args:
            deps: Dependencies to benchmark
            
        Returns:
            Size estimates in MB
        """
        # Known package sizes (approximate)
        package_sizes = {
            "numpy": 15.0,
            "torch": 750.0,
            "tensorflow": 400.0,
            "jax": 200.0,
            "jaxlib": 150.0,
            "pandas": 25.0,
            "scikit-learn": 20.0,
            "matplotlib": 30.0,
            "jupyter": 50.0,
            "fastapi": 5.0,
            "uvicorn": 8.0,
            "mlflow": 15.0,
            "wandb": 12.0,
            "tensorboard": 20.0,
        }
        
        total_size = 0.0
        breakdown = {}
        
        for package in deps.keys():
            # Extract base package name
            base_name = package.split("[")[0].split(">=")[0].split("==")[0]
            
            size = package_sizes.get(base_name, 5.0)  # Default 5MB
            breakdown[package] = size
            total_size += size
        
        breakdown["total"] = total_size
        return breakdown
    
    def generate_optimization_report(self) -> str:
        """Generate a comprehensive optimization report
        
        Returns:
            Optimization report
        """
        report = ["# Requirements Optimization Report", ""]
        
        # Analyze different configurations
        configs = {
            "minimal": self.create_minimal_requirements(),
            "core": self.create_core_requirements(),
            "full": self.create_full_requirements(),
            "ml": self.create_category_requirements(["torch", "data"]),
            "serving": self.create_category_requirements(["serving", "mlops"]),
            "dev": self.create_category_requirements(["dev", "build"]),
        }
        
        for config_name, deps in configs.items():
            report.append(f"## {config_name.title()} Configuration")
            report.append("")
            
            # Dependency count
            report.append(f"**Dependencies:** {len(deps)}")
            
            # Size estimate
            sizes = self.benchmark_installation_size(deps)
            report.append(f"**Estimated Size:** {sizes['total']:.1f} MB")
            
            # Top 5 largest dependencies
            sorted_sizes = sorted(
                [(k, v) for k, v in sizes.items() if k != "total"],
                key=lambda x: x[1], reverse=True
            )[:5]
            
            if sorted_sizes:
                report.append("**Largest Dependencies:**")
                for pkg, size in sorted_sizes:
                    report.append(f"- {pkg}: {size:.1f} MB")
            
            # Dependency analysis
            analysis = self.analyze_dependency_tree(deps)
            if analysis["conflicts"]:
                report.append("**Conflicts:**")
                for conflict in analysis["conflicts"]:
                    report.append(f"- {conflict}")
            
            if analysis["redundant"]:
                report.append("**Potential Redundancies:**")
                for redundant in analysis["redundant"]:
                    report.append(f"- {redundant}")
            
            report.append("")
        
        return "\n".join(report)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize requirements for trustformers-py")
    parser.add_argument("--config", choices=["minimal", "core", "full", "ml", "serving", "dev"],
                       default="core", help="Configuration to generate")
    parser.add_argument("--categories", nargs="+", 
                       choices=["torch", "jax", "serving", "mlops", "data", "jupyter", "dev", "build"],
                       help="Specific categories to include")
    parser.add_argument("--platform", choices=["linux", "macos", "windows"],
                       help="Target platform for optimization")
    parser.add_argument("--format", choices=["txt", "toml", "conda"],
                       default="txt", help="Output format")
    parser.add_argument("--output", metavar="FILE",
                       help="Output file path")
    parser.add_argument("--analyze", action="store_true",
                       help="Generate dependency analysis report")
    parser.add_argument("--all-configs", action="store_true",
                       help="Generate all configuration files")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = RequirementsOptimizer()
    
    try:
        if args.analyze:
            report = optimizer.generate_optimization_report()
            print(report)
            
            # Save report
            report_path = Path("requirements_analysis.md")
            with open(report_path, "w") as f:
                f.write(report)
            print(f"\nAnalysis report saved to: {report_path}")
            return
        
        if args.all_configs:
            # Generate all standard configurations
            configs = {
                "minimal": optimizer.create_minimal_requirements(),
                "core": optimizer.create_core_requirements(),
                "full": optimizer.create_full_requirements(),
                "ml": optimizer.create_category_requirements(["torch", "data"]),
                "serving": optimizer.create_category_requirements(["serving", "mlops"]),
                "dev": optimizer.create_category_requirements(["dev", "build"]),
            }
            
            for config_name, deps in configs.items():
                if args.platform:
                    deps = optimizer.optimize_for_platform(deps, args.platform)
                
                filename = f"requirements-{config_name}.txt"
                optimizer.generate_requirements_txt(deps, filename)
            
            print(f"Generated {len(configs)} requirement files")
            return
        
        # Generate specific configuration
        if args.categories:
            deps = optimizer.create_category_requirements(args.categories)
        elif args.config == "minimal":
            deps = optimizer.create_minimal_requirements()
        elif args.config == "core":
            deps = optimizer.create_core_requirements()
        elif args.config == "full":
            deps = optimizer.create_full_requirements()
        elif args.config == "ml":
            deps = optimizer.create_category_requirements(["torch", "data"])
        elif args.config == "serving":
            deps = optimizer.create_category_requirements(["serving", "mlops"])
        elif args.config == "dev":
            deps = optimizer.create_category_requirements(["dev", "build"])
        else:
            deps = optimizer.create_core_requirements()
        
        # Apply platform optimization
        if args.platform:
            deps = optimizer.optimize_for_platform(deps, args.platform)
        
        # Generate output
        if args.format == "txt":
            output_file = args.output or f"requirements-{args.config}.txt"
            optimizer.generate_requirements_txt(deps, output_file)
        elif args.format == "toml":
            # Generate pyproject.toml format
            deps_by_category = {"core": deps}
            toml_content = optimizer.generate_pyproject_toml_deps(deps_by_category)
            
            output_file = args.output or "dependencies.toml"
            with open(output_file, "w") as f:
                f.write(toml_content)
            print(f"Generated TOML dependencies: {output_file}")
        elif args.format == "conda":
            env_name = f"trustformers-{args.config}"
            optimizer.create_conda_environment(deps, env_name)
        
        # Show size estimate
        sizes = optimizer.benchmark_installation_size(deps)
        print(f"Estimated installation size: {sizes['total']:.1f} MB")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()