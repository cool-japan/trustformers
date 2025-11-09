#!/usr/bin/env python3
"""
Python Version Optimization Module for trustformers-py

This module provides comprehensive Python version-specific optimizations,
platform support enhancements, and compatibility utilities.
"""

import sys
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

class PythonVersion(Enum):
    """Supported Python versions with their characteristics"""
    PY38 = ("3.8", "abi3-py38", ["dataclasses", "typing-extensions"])
    PY39 = ("3.9", "abi3-py39", ["graphlib", "zoneinfo"])
    PY310 = ("3.10", "abi3-py310", ["match-case", "union-types"])
    PY311 = ("3.11", "abi3-py311", ["task-groups", "exception-groups"])
    PY312 = ("3.12", "abi3-py312", ["type-params", "buffer-protocol"])

    def __init__(self, version: str, abi_tag: str, features: List[str]):
        self.version = version
        self.abi_tag = abi_tag
        self.features = features

class PlatformArch(Enum):
    """Supported platform architectures"""
    X86_64_LINUX = ("x86_64-unknown-linux-gnu", "linux", "x86_64")
    AARCH64_LINUX = ("aarch64-unknown-linux-gnu", "linux", "aarch64")
    X86_64_MACOS = ("x86_64-apple-darwin", "macos", "x86_64")
    AARCH64_MACOS = ("aarch64-apple-darwin", "macos", "aarch64")
    X86_64_WINDOWS = ("x86_64-pc-windows-msvc", "windows", "x86_64")
    AARCH64_WINDOWS = ("aarch64-pc-windows-msvc", "windows", "aarch64")
    WASM32_WASI = ("wasm32-wasi", "wasm", "wasm32")

    def __init__(self, target: str, platform: str, arch: str):
        self.target = target
        self.platform = platform
        self.arch = arch

@dataclass
class OptimizationConfig:
    """Configuration for version-specific optimizations"""
    python_version: PythonVersion
    platform_arch: PlatformArch
    profile: str = "release"
    features: List[str] = None
    compile_flags: List[str] = None
    link_flags: List[str] = None
    extra_requirements: List[str] = None

class PythonVersionOptimizer:
    """Main Python version optimization class"""
    
    def __init__(self, package_dir: str = None):
        """Initialize the Python version optimizer
        
        Args:
            package_dir: Path to the package directory
        """
        self.package_dir = Path(package_dir or os.getcwd())
        self.current_python = self._detect_current_python()
        self.current_platform = self._detect_current_platform()
        
        # Version-specific optimizations
        self.version_optimizations = {
            PythonVersion.PY38: {
                "compile_flags": ["-O2", "-DNDEBUG"],
                "features": ["py38-compat"],
                "requirements": ["dataclasses>=0.8", "typing-extensions>=3.10"]
            },
            PythonVersion.PY39: {
                "compile_flags": ["-O2", "-DNDEBUG", "-DPYTHON_39_FEATURES"],
                "features": ["py39-features", "graphlib"],
                "requirements": []
            },
            PythonVersion.PY310: {
                "compile_flags": ["-O2", "-DNDEBUG", "-DPYTHON_310_FEATURES"],
                "features": ["py310-features", "match-case", "union-types"],
                "requirements": []
            },
            PythonVersion.PY311: {
                "compile_flags": ["-O3", "-DNDEBUG", "-DPYTHON_311_SPEEDUPS"],
                "features": ["py311-speedups", "task-groups", "exception-groups"],
                "requirements": []
            },
            PythonVersion.PY312: {
                "compile_flags": ["-O3", "-DNDEBUG", "-DPYTHON_312_SPEEDUPS"],
                "features": ["py312-speedups", "buffer-protocol", "type-params"],
                "requirements": []
            }
        }
        
        # Platform-specific optimizations
        self.platform_optimizations = {
            PlatformArch.X86_64_LINUX: {
                "compile_flags": ["-march=x86-64", "-mtune=generic"],
                "link_flags": ["-Wl,--gc-sections", "-Wl,--strip-all"],
                "features": ["simd", "avx2"],
                "requirements": []
            },
            PlatformArch.AARCH64_LINUX: {
                "compile_flags": ["-march=armv8-a", "-mtune=cortex-a72"],
                "link_flags": ["-Wl,--gc-sections", "-Wl,--strip-all"],
                "features": ["neon", "aarch64"],
                "requirements": []
            },
            PlatformArch.X86_64_MACOS: {
                "compile_flags": ["-march=x86-64", "-mtune=generic"],
                "link_flags": ["-Wl,-dead_strip"],
                "features": ["accelerate", "avx2"],
                "requirements": []
            },
            PlatformArch.AARCH64_MACOS: {
                "compile_flags": ["-mcpu=apple-a14"],
                "link_flags": ["-Wl,-dead_strip"],
                "features": ["accelerate", "neon", "apple-silicon"],
                "requirements": []
            },
            PlatformArch.X86_64_WINDOWS: {
                "compile_flags": ["/arch:AVX2", "/O2"],
                "link_flags": ["/OPT:REF", "/OPT:ICF"],
                "features": ["avx2"],
                "requirements": []
            },
            PlatformArch.AARCH64_WINDOWS: {
                "compile_flags": ["/O2"],
                "link_flags": ["/OPT:REF", "/OPT:ICF"],
                "features": ["neon", "aarch64"],
                "requirements": []
            },
            PlatformArch.WASM32_WASI: {
                "compile_flags": ["-Os", "-DWASM_BUILD"],
                "link_flags": ["-Wl,--gc-sections"],
                "features": ["wasm", "no-std"],
                "requirements": []
            }
        }
    
    def _detect_current_python(self) -> PythonVersion:
        """Detect the current Python version"""
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        for py_ver in PythonVersion:
            if py_ver.version == version:
                return py_ver
        raise RuntimeError(f"Unsupported Python version: {version}")
    
    def _detect_current_platform(self) -> PlatformArch:
        """Detect the current platform and architecture"""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Normalize architecture names
        if machine in ["x86_64", "amd64"]:
            arch = "x86_64"
        elif machine in ["aarch64", "arm64"]:
            arch = "aarch64"
        elif machine in ["wasm32"]:
            arch = "wasm32"
        else:
            arch = "x86_64"  # fallback
        
        # Match to PlatformArch
        for platform_arch in PlatformArch:
            if platform_arch.platform == system and platform_arch.arch == arch:
                return platform_arch
        
        # Fallback to x86_64 variant
        if system == "linux":
            return PlatformArch.X86_64_LINUX
        elif system == "darwin":
            return PlatformArch.X86_64_MACOS
        elif system == "windows":
            return PlatformArch.X86_64_WINDOWS
        else:
            return PlatformArch.X86_64_LINUX
    
    def create_optimization_config(self,
                                 python_version: PythonVersion = None,
                                 platform_arch: PlatformArch = None,
                                 profile: str = "release") -> OptimizationConfig:
        """Create an optimization configuration
        
        Args:
            python_version: Target Python version (defaults to current)
            platform_arch: Target platform (defaults to current)
            profile: Build profile
            
        Returns:
            OptimizationConfig object
        """
        python_version = python_version or self.current_python
        platform_arch = platform_arch or self.current_platform
        
        # Combine optimizations
        py_opts = self.version_optimizations.get(python_version, {})
        platform_opts = self.platform_optimizations.get(platform_arch, {})
        
        # Merge features
        features = []
        features.extend(py_opts.get("features", []))
        features.extend(platform_opts.get("features", []))
        
        # Merge compile flags
        compile_flags = []
        compile_flags.extend(py_opts.get("compile_flags", []))
        compile_flags.extend(platform_opts.get("compile_flags", []))
        
        # Merge link flags
        link_flags = []
        link_flags.extend(py_opts.get("link_flags", []))
        link_flags.extend(platform_opts.get("link_flags", []))
        
        # Merge requirements
        requirements = []
        requirements.extend(py_opts.get("requirements", []))
        requirements.extend(platform_opts.get("requirements", []))
        
        return OptimizationConfig(
            python_version=python_version,
            platform_arch=platform_arch,
            profile=profile,
            features=features,
            compile_flags=compile_flags,
            link_flags=link_flags,
            extra_requirements=requirements
        )
    
    def setup_build_environment(self, config: OptimizationConfig) -> Dict[str, str]:
        """Set up build environment for the given configuration
        
        Args:
            config: Optimization configuration
            
        Returns:
            Environment variables dictionary
        """
        env = os.environ.copy()
        
        # Python version specific settings
        env["PYTHON_VERSION"] = config.python_version.version
        env["PYTHON_ABI_TAG"] = config.python_version.abi_tag
        
        # Rust compilation flags
        rustflags = []
        
        # Profile-specific optimizations
        if config.profile == "release":
            rustflags.extend(["-C", "opt-level=3", "-C", "lto=fat"])
        elif config.profile == "size":
            rustflags.extend(["-C", "opt-level=s", "-C", "lto=fat", "-C", "codegen-units=1"])
        elif config.profile == "speed":
            rustflags.extend(["-C", "opt-level=3", "-C", "target-cpu=native"])
        elif config.profile == "debug":
            rustflags.extend(["-C", "opt-level=1"])
        
        # Platform-specific optimizations
        if config.platform_arch.arch == "aarch64":
            if "apple" in config.platform_arch.target:
                rustflags.extend(["-C", "target-cpu=apple-a14"])
            else:
                rustflags.extend(["-C", "target-cpu=cortex-a72"])
        elif config.platform_arch.arch == "x86_64":
            rustflags.extend(["-C", "target-feature=+avx2"])
        
        # Link-time optimizations
        if config.link_flags:
            for flag in config.link_flags:
                if flag.startswith("-Wl,"):
                    rustflags.extend(["-C", f"link-arg={flag}"])
        
        env["RUSTFLAGS"] = " ".join(rustflags)
        
        # Cargo profile settings
        env["CARGO_PROFILE_RELEASE_OPT_LEVEL"] = "3" if config.profile == "speed" else "s" if config.profile == "size" else "2"
        env["CARGO_PROFILE_RELEASE_LTO"] = "fat" if config.profile in ["release", "size", "speed"] else "thin"
        env["CARGO_PROFILE_RELEASE_CODEGEN_UNITS"] = "1" if config.profile in ["size", "speed"] else "16"
        env["CARGO_PROFILE_RELEASE_PANIC"] = "abort" if config.profile == "size" else "unwind"
        
        # PyO3 specific settings
        env["PYO3_PYTHON"] = sys.executable
        env["PYO3_CONFIG_FILE"] = str(self.package_dir / "pyo3-config.txt")
        
        return env
    
    def generate_pyo3_config(self, config: OptimizationConfig) -> Path:
        """Generate PyO3 configuration file
        
        Args:
            config: Optimization configuration
            
        Returns:
            Path to the generated config file
        """
        config_path = self.package_dir / "pyo3-config.txt"
        
        config_content = f"""
implementation=CPython
version={config.python_version.version}
shared=true
abi3=true
lib_name=python{config.python_version.version.replace('.', '')}
lib_dir={sys.prefix}/lib
executable={sys.executable}
pointer_width=64
build_flags=
suppress_build_script_link_lines=false
"""
        
        config_path.write_text(config_content.strip())
        return config_path
    
    def build_for_version(self,
                         python_version: PythonVersion,
                         platform_arch: PlatformArch = None,
                         profile: str = "release") -> Path:
        """Build wheel for specific Python version and platform
        
        Args:
            python_version: Target Python version
            platform_arch: Target platform (defaults to current)
            profile: Build profile
            
        Returns:
            Path to the built wheel
        """
        platform_arch = platform_arch or self.current_platform
        config = self.create_optimization_config(python_version, platform_arch, profile)
        
        print(f"Building for Python {python_version.version} on {platform_arch.target}")
        
        # Generate PyO3 config
        self.generate_pyo3_config(config)
        
        # Set up environment
        env = self.setup_build_environment(config)
        
        # Build command
        cmd = ["maturin", "build", "--release"]
        
        if config.features:
            cmd.extend(["--features", ",".join(config.features)])
        
        cmd.extend(["--target", platform_arch.target])
        
        # ABI3 support
        if python_version != PythonVersion.PY38:  # ABI3 not available for 3.8
            cmd.append("--compatibility")
            cmd.append("abi3")
        
        # Size optimization
        if profile == "size":
            cmd.append("--strip")
        
        print(f"Running: {' '.join(cmd)}")
        
        # Run build
        result = subprocess.run(
            cmd,
            cwd=self.package_dir,
            env=env,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Build failed: {result.stderr}")
            raise RuntimeError(f"Build failed with exit code {result.returncode}")
        
        # Find the built wheel
        target_dir = self.package_dir / "target" / "wheels"
        wheels = list(target_dir.glob("*.whl"))
        
        if not wheels:
            raise RuntimeError("No wheel found after build")
        
        # Return the most recent wheel
        wheel_path = max(wheels, key=lambda p: p.stat().st_mtime)
        print(f"Built wheel: {wheel_path}")
        
        return wheel_path
    
    def build_all_versions(self,
                          platform_arch: PlatformArch = None,
                          profile: str = "release") -> List[Path]:
        """Build wheels for all supported Python versions
        
        Args:
            platform_arch: Target platform (defaults to current)
            profile: Build profile
            
        Returns:
            List of built wheel paths
        """
        platform_arch = platform_arch or self.current_platform
        wheels = []
        
        for python_version in PythonVersion:
            try:
                wheel = self.build_for_version(python_version, platform_arch, profile)
                wheels.append(wheel)
            except Exception as e:
                print(f"Failed to build for Python {python_version.version}: {e}")
                continue
        
        return wheels
    
    def build_all_platforms(self,
                           python_version: PythonVersion = None,
                           profile: str = "release") -> List[Path]:
        """Build wheels for all supported platforms
        
        Args:
            python_version: Target Python version (defaults to current)
            profile: Build profile
            
        Returns:
            List of built wheel paths
        """
        python_version = python_version or self.current_python
        wheels = []
        
        # Skip WASM for now as it's experimental
        platforms = [p for p in PlatformArch if p != PlatformArch.WASM32_WASI]
        
        for platform_arch in platforms:
            try:
                wheel = self.build_for_version(python_version, platform_arch, profile)
                wheels.append(wheel)
            except Exception as e:
                print(f"Failed to build for platform {platform_arch.target}: {e}")
                continue
        
        return wheels
    
    def build_universal_matrix(self, profile: str = "release") -> Dict[str, List[Path]]:
        """Build wheels for all Python versions and platforms
        
        Args:
            profile: Build profile
            
        Returns:
            Dictionary mapping platform names to list of wheel paths
        """
        matrix = {}
        
        # Skip WASM for now
        platforms = [p for p in PlatformArch if p != PlatformArch.WASM32_WASI]
        
        for platform_arch in platforms:
            platform_wheels = []
            for python_version in PythonVersion:
                try:
                    wheel = self.build_for_version(python_version, platform_arch, profile)
                    platform_wheels.append(wheel)
                except Exception as e:
                    print(f"Failed to build {python_version.version} for {platform_arch.target}: {e}")
                    continue
            
            if platform_wheels:
                matrix[platform_arch.target] = platform_wheels
        
        return matrix
    
    def create_requirements_file(self,
                               config: OptimizationConfig,
                               output_path: Path = None) -> Path:
        """Create version-specific requirements file
        
        Args:
            config: Optimization configuration
            output_path: Output file path
            
        Returns:
            Path to the requirements file
        """
        output_path = output_path or self.package_dir / f"requirements-py{config.python_version.version.replace('.', '')}.txt"
        
        requirements = [
            "numpy>=1.21.0",
            "tqdm>=4.62.0", 
            "requests>=2.26.0",
            "regex>=2021.8.3",
            "pyyaml>=5.4.1",
            "packaging>=20.0",
        ]
        
        # Add version-specific requirements
        if config.extra_requirements:
            requirements.extend(config.extra_requirements)
        
        # Platform-specific adjustments
        if config.platform_arch == PlatformArch.WASM32_WASI:
            # WASM-specific lightweight requirements
            requirements = [
                "micropython-lib",
                "typing-extensions>=3.10.0",
            ]
        
        output_path.write_text("\n".join(requirements))
        return output_path

def main():
    """Main entry point for the optimization script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Python Version Optimization Tool")
    parser.add_argument("--python-version", choices=["3.8", "3.9", "3.10", "3.11", "3.12"], 
                       help="Target Python version")
    parser.add_argument("--platform", choices=["linux-x64", "linux-arm64", "macos-x64", "macos-arm64", 
                                              "windows-x64", "windows-arm64", "wasm32"],
                       help="Target platform")
    parser.add_argument("--profile", choices=["debug", "release", "size", "speed"], 
                       default="release", help="Build profile")
    parser.add_argument("--all-versions", action="store_true",
                       help="Build for all Python versions")
    parser.add_argument("--all-platforms", action="store_true",
                       help="Build for all platforms")
    parser.add_argument("--matrix", action="store_true",
                       help="Build complete matrix")
    parser.add_argument("--generate-requirements", action="store_true",
                       help="Generate requirements files")
    
    args = parser.parse_args()
    
    optimizer = PythonVersionOptimizer()
    
    if args.matrix:
        print("Building complete matrix...")
        matrix = optimizer.build_universal_matrix(args.profile)
        print(f"Built {sum(len(wheels) for wheels in matrix.values())} wheels")
        for platform, wheels in matrix.items():
            print(f"  {platform}: {len(wheels)} wheels")
    elif args.all_versions:
        print("Building for all Python versions...")
        wheels = optimizer.build_all_versions(profile=args.profile)
        print(f"Built {len(wheels)} wheels")
    elif args.all_platforms:
        print("Building for all platforms...")
        wheels = optimizer.build_all_platforms(profile=args.profile)
        print(f"Built {len(wheels)} wheels")
    else:
        # Single build
        python_version = None
        if args.python_version:
            for pv in PythonVersion:
                if pv.version == args.python_version:
                    python_version = pv
                    break
        
        platform_arch = None
        if args.platform:
            platform_map = {
                "linux-x64": PlatformArch.X86_64_LINUX,
                "linux-arm64": PlatformArch.AARCH64_LINUX,
                "macos-x64": PlatformArch.X86_64_MACOS,
                "macos-arm64": PlatformArch.AARCH64_MACOS,
                "windows-x64": PlatformArch.X86_64_WINDOWS,
                "windows-arm64": PlatformArch.AARCH64_WINDOWS,
                "wasm32": PlatformArch.WASM32_WASI,
            }
            platform_arch = platform_map.get(args.platform)
        
        wheel = optimizer.build_for_version(
            python_version or optimizer.current_python,
            platform_arch or optimizer.current_platform,
            args.profile
        )
        print(f"Built wheel: {wheel}")
    
    if args.generate_requirements:
        print("Generating requirements files...")
        for python_version in PythonVersion:
            for platform_arch in PlatformArch:
                config = optimizer.create_optimization_config(python_version, platform_arch)
                req_file = optimizer.create_requirements_file(config)
                print(f"Generated: {req_file}")

if __name__ == "__main__":
    main()