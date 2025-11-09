#!/usr/bin/env python3
"""
Package optimization script for trustformers-py

This script optimizes the Python package for size, performance, and distribution.
It includes utilities for:
- Wheel size reduction
- Platform-specific builds 
- CUDA variants
- CPU-only builds
- Source distributions
- Binary optimization
"""

import os
import sys
import shutil
import subprocess
import tempfile
import platform
from pathlib import Path
from typing import List, Dict, Optional, Tuple

class PackageOptimizer:
    """Main package optimizer class"""
    
    def __init__(self, package_dir: str = None):
        """Initialize the package optimizer
        
        Args:
            package_dir: Path to the package directory (defaults to current directory)
        """
        self.package_dir = Path(package_dir or os.getcwd())
        self.python_dir = self.package_dir / "python"
        self.cargo_toml = self.package_dir / "Cargo.toml"
        
        # Optimization settings
        self.optimization_flags = {
            "size": ["-Os", "-flto", "-ffunction-sections", "-fdata-sections"],
            "speed": ["-O3", "-flto", "-march=native"],
            "debug": ["-g", "-O1"],
            "release": ["-O2", "-DNDEBUG"]
        }
        
        # Platform-specific settings with architecture support
        self.platform_configs = {
            "linux-x86_64": {
                "target": "x86_64-unknown-linux-gnu",
                "features": ["cpu", "simd", "avx2"],
                "linker_flags": ["-Wl,--gc-sections", "-Wl,--strip-all"],
                "compile_flags": ["-march=x86-64", "-mtune=generic"],
                "optimization": {"sse": True, "avx": True, "avx2": True}
            },
            "linux-aarch64": {
                "target": "aarch64-unknown-linux-gnu",
                "features": ["cpu", "neon", "aarch64"],
                "linker_flags": ["-Wl,--gc-sections", "-Wl,--strip-all"],
                "compile_flags": ["-march=armv8-a", "-mtune=cortex-a72"],
                "optimization": {"neon": True, "crypto": True}
            },
            "macos-x86_64": {
                "target": "x86_64-apple-darwin", 
                "features": ["cpu", "accelerate", "avx2"],
                "linker_flags": ["-Wl,-dead_strip"],
                "compile_flags": ["-march=x86-64", "-mtune=generic"],
                "optimization": {"accelerate": True, "avx2": True}
            },
            "macos-aarch64": {
                "target": "aarch64-apple-darwin",
                "features": ["cpu", "accelerate", "neon", "apple-silicon"],
                "linker_flags": ["-Wl,-dead_strip"],
                "compile_flags": ["-mcpu=apple-a14"],
                "optimization": {"accelerate": True, "neon": True, "apple_silicon": True}
            },
            "windows-x86_64": {
                "target": "x86_64-pc-windows-msvc",
                "features": ["cpu", "avx2"],
                "linker_flags": ["/OPT:REF", "/OPT:ICF"],
                "compile_flags": ["/arch:AVX2", "/O2"],
                "optimization": {"avx2": True}
            },
            "windows-aarch64": {
                "target": "aarch64-pc-windows-msvc",
                "features": ["cpu", "neon", "aarch64"],
                "linker_flags": ["/OPT:REF", "/OPT:ICF"],
                "compile_flags": ["/O2"],
                "optimization": {"neon": True}
            },
            "wasm32-wasi": {
                "target": "wasm32-wasi",
                "features": ["wasm", "no-std"],
                "linker_flags": ["-Wl,--gc-sections"],
                "compile_flags": ["-Os", "-DWASM_BUILD"],
                "optimization": {"size": True, "wasm_simd": False}
            }
        }
    
    def detect_platform(self) -> str:
        """Detect the current platform and architecture"""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Normalize platform name
        if system == "darwin":
            system = "macos"
        
        # Normalize architecture name
        if machine in ["x86_64", "amd64"]:
            arch = "x86_64"
        elif machine in ["aarch64", "arm64"]:
            arch = "aarch64"
        elif machine in ["wasm32"]:
            arch = "wasm32"
            system = "wasm32"
        else:
            arch = "x86_64"  # fallback
        
        # Special handling for WASM
        if system == "wasm32":
            return "wasm32-wasi"
        
        return f"{system}-{arch}"
    
    def setup_build_environment(self, profile: str = "release") -> Dict[str, str]:
        """Set up build environment variables
        
        Args:
            profile: Build profile (release, debug, size, speed)
            
        Returns:
            Dictionary of environment variables
        """
        env = os.environ.copy()
        
        # Rust flags
        rustflags = []
        if profile in self.optimization_flags:
            # Map C flags to Rust equivalents
            if "-Os" in self.optimization_flags[profile]:
                rustflags.append("-C opt-level=s")
            elif "-O3" in self.optimization_flags[profile]:
                rustflags.append("-C opt-level=3")
            elif "-O2" in self.optimization_flags[profile]:
                rustflags.append("-C opt-level=2")
            
            if "-flto" in self.optimization_flags[profile]:
                rustflags.append("-C lto=fat")
            
            if "-march=native" in self.optimization_flags[profile]:
                rustflags.append("-C target-cpu=native")
        
        # Platform-specific flags
        platform_name = self.detect_platform()
        if platform_name in self.platform_configs:
            config = self.platform_configs[platform_name]
            
            # Add platform-specific Rust compilation flags
            linker_flags = config.get("linker_flags", [])
            for flag in linker_flags:
                if flag.startswith("-Wl,"):
                    rustflags.extend(["-C", f"link-arg={flag}"])
                elif flag.startswith("/"):
                    # Windows-style linker flags
                    rustflags.extend(["-C", f"link-arg={flag}"])
            
            # Add architecture-specific optimizations
            optimizations = config.get("optimization", {})
            if optimizations.get("avx2"):
                rustflags.extend(["-C", "target-feature=+avx2"])
            if optimizations.get("neon"):
                rustflags.extend(["-C", "target-feature=+neon"])
            if optimizations.get("apple_silicon"):
                rustflags.extend(["-C", "target-cpu=apple-a14"])
            
            # Set C/C++ compilation flags as environment variables
            compile_flags = config.get("compile_flags", [])
            if compile_flags:
                env["CFLAGS"] = " ".join(compile_flags)
                env["CXXFLAGS"] = " ".join(compile_flags)
        
        env["RUSTFLAGS"] = " ".join(rustflags)
        
        # Cargo settings
        env["CARGO_PROFILE_RELEASE_OPT_LEVEL"] = "3" if profile == "speed" else "s" if profile == "size" else "2"
        env["CARGO_PROFILE_RELEASE_LTO"] = "fat" if profile in ["release", "size", "speed"] else "off"
        env["CARGO_PROFILE_RELEASE_CODEGEN_UNITS"] = "1" if profile in ["size", "speed"] else "16"
        env["CARGO_PROFILE_RELEASE_PANIC"] = "abort" if profile == "size" else "unwind"
        
        return env
    
    def build_optimized_wheel(self, 
                            profile: str = "release",
                            features: List[str] = None,
                            target: str = None) -> Path:
        """Build an optimized wheel
        
        Args:
            profile: Build profile
            features: List of features to enable
            target: Target platform
            
        Returns:
            Path to the built wheel
        """
        print(f"Building optimized wheel with profile: {profile}")
        
        # Set up environment
        env = self.setup_build_environment(profile)
        
        # Build command
        cmd = ["maturin", "build", "--release"]
        
        if features:
            cmd.extend(["--features", ",".join(features)])
        
        if target:
            cmd.extend(["--target", target])
        
        # Add optimization flags
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
    
    def create_cpu_only_build(self) -> Path:
        """Create a CPU-only build without CUDA dependencies
        
        Returns:
            Path to the CPU-only wheel
        """
        print("Creating CPU-only build...")
        
        features = ["cpu"]
        platform_name = self.detect_platform()
        
        if platform_name in self.platform_configs:
            platform_features = self.platform_configs[platform_name].get("features", [])
            features.extend(f for f in platform_features if f != "cuda")
        
        return self.build_optimized_wheel(
            profile="size",
            features=features
        )
    
    def create_cuda_build(self, cuda_version: str = "11.8") -> Path:
        """Create a CUDA-enabled build
        
        Args:
            cuda_version: CUDA version to target
            
        Returns:
            Path to the CUDA wheel
        """
        print(f"Creating CUDA build for version {cuda_version}...")
        
        features = ["cuda", f"cuda{cuda_version.replace('.', '')}"]
        
        return self.build_optimized_wheel(
            profile="speed",
            features=features
        )
    
    def build_for_platform(self, platform_target: str, profile: str = "release") -> Path:
        """Build wheel for specific platform target
        
        Args:
            platform_target: Platform target (e.g., 'linux-aarch64', 'macos-aarch64')
            profile: Build profile
            
        Returns:
            Path to the built wheel
        """
        print(f"Building for platform: {platform_target}")
        
        if platform_target not in self.platform_configs:
            raise ValueError(f"Unsupported platform: {platform_target}")
        
        config = self.platform_configs[platform_target]
        
        return self.build_optimized_wheel(
            profile=profile,
            features=config.get("features", []),
            target=config["target"]
        )
    
    def build_arm64_variants(self, profile: str = "release") -> List[Path]:
        """Build ARM64 variants for all supported platforms
        
        Args:
            profile: Build profile
            
        Returns:
            List of built wheel paths
        """
        print("Building ARM64 variants...")
        
        arm64_platforms = [
            "linux-aarch64",
            "macos-aarch64", 
            "windows-aarch64"
        ]
        
        wheels = []
        for platform in arm64_platforms:
            try:
                wheel = self.build_for_platform(platform, profile)
                wheels.append(wheel)
                print(f"Successfully built {platform} wheel: {wheel}")
            except Exception as e:
                print(f"Failed to build {platform}: {e}")
                continue
        
        return wheels
    
    def build_universal_macos(self, profile: str = "release") -> Path:
        """Build universal macOS wheel (x86_64 + aarch64)
        
        Args:
            profile: Build profile
            
        Returns:
            Path to the universal wheel
        """
        print("Building universal macOS wheel...")
        
        # Build for both architectures
        x86_wheel = self.build_for_platform("macos-x86_64", profile)
        arm_wheel = self.build_for_platform("macos-aarch64", profile)
        
        # Combine into universal wheel using lipo
        universal_wheel = self._combine_wheels_with_lipo(x86_wheel, arm_wheel, profile)
        print(f"Built universal wheel: {universal_wheel}")
        
        return universal_wheel
    
    def _combine_wheels_with_lipo(self, x86_wheel: Path, arm_wheel: Path, profile: str) -> Path:
        """Combine x86_64 and ARM64 wheels into universal wheel using lipo
        
        Args:
            x86_wheel: Path to x86_64 wheel
            arm_wheel: Path to ARM64 wheel
            profile: Build profile
            
        Returns:
            Path to universal wheel
        """
        import zipfile
        import glob
        
        print("Combining wheels with lipo...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            x86_extract = temp_path / "x86"
            arm_extract = temp_path / "arm"
            universal_build = temp_path / "universal"
            
            # Extract both wheels
            print(f"Extracting {x86_wheel}")
            with zipfile.ZipFile(x86_wheel, 'r') as zip_ref:
                zip_ref.extractall(x86_extract)
                
            print(f"Extracting {arm_wheel}")
            with zipfile.ZipFile(arm_wheel, 'r') as zip_ref:
                zip_ref.extractall(arm_extract)
            
            # Copy ARM wheel as base (it should be the same structure)
            shutil.copytree(arm_extract, universal_build)
            
            # Find shared libraries that need combining
            so_files = list(universal_build.glob("**/*.so"))
            dylib_files = list(universal_build.glob("**/*.dylib"))
            all_libs = so_files + dylib_files
            
            print(f"Found {len(all_libs)} shared libraries to combine")
            
            for lib_file in all_libs:
                # Get corresponding file from x86 build
                rel_path = lib_file.relative_to(universal_build)
                x86_lib = x86_extract / rel_path
                arm_lib = arm_extract / rel_path
                
                if x86_lib.exists() and arm_lib.exists():
                    print(f"Combining {rel_path}")
                    
                    # Use lipo to create universal binary
                    try:
                        result = subprocess.run([
                            "lipo", "-create",
                            str(x86_lib),
                            str(arm_lib),
                            "-output", str(lib_file)
                        ], check=True, capture_output=True, text=True)
                        
                        # Verify the universal binary was created
                        verify_result = subprocess.run([
                            "lipo", "-info", str(lib_file)
                        ], capture_output=True, text=True)
                        
                        if "x86_64" in verify_result.stdout and "arm64" in verify_result.stdout:
                            print(f"  ✓ Successfully created universal binary for {rel_path}")
                        else:
                            print(f"  ⚠ Warning: {rel_path} may not be truly universal")
                            
                    except subprocess.CalledProcessError as e:
                        print(f"  ✗ Failed to combine {rel_path}: {e}")
                        # Keep the ARM64 version as fallback
                        shutil.copy2(arm_lib, lib_file)
                else:
                    print(f"  ⚠ Skipping {rel_path} - not found in both architectures")
            
            # Create new universal wheel name
            wheel_name = arm_wheel.name
            # Replace platform tag with universal
            if "macosx" in wheel_name:
                # Change from macosx_11_0_arm64 to macosx_11_0_universal2
                parts = wheel_name.split('-')
                platform_part = parts[-1].replace('.whl', '')
                if '_arm64' in platform_part:
                    universal_platform = platform_part.replace('_arm64', '_universal2')
                    parts[-1] = universal_platform + '.whl'
                    universal_wheel_name = '-'.join(parts)
                else:
                    universal_wheel_name = wheel_name.replace('.whl', '_universal2.whl')
            else:
                universal_wheel_name = wheel_name.replace('.whl', '_universal.whl')
            
            universal_wheel_path = self.package_dir / "dist" / universal_wheel_name
            
            # Create the universal wheel
            print(f"Creating universal wheel: {universal_wheel_name}")
            with zipfile.ZipFile(universal_wheel_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
                for file_path in universal_build.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(universal_build)
                        zip_ref.write(file_path, arcname)
            
            print(f"Universal wheel created: {universal_wheel_path}")
            return universal_wheel_path
    
    def build_wasm_experimental(self, profile: str = "size") -> Path:
        """Build experimental WASM version
        
        Args:
            profile: Build profile (defaults to 'size' for WASM)
            
        Returns:
            Path to the WASM wheel
        """
        print("Building experimental WASM version...")
        
        return self.build_for_platform("wasm32-wasi", profile)
    
    def build_all_platforms_matrix(self, profile: str = "release") -> Dict[str, Path]:
        """Build wheels for all supported platforms
        
        Args:
            profile: Build profile
            
        Returns:
            Dictionary mapping platform names to wheel paths
        """
        print("Building complete platform matrix...")
        
        platform_wheels = {}
        
        for platform_name in self.platform_configs.keys():
            try:
                # Skip WASM for regular builds
                if platform_name == "wasm32-wasi":
                    continue
                    
                wheel = self.build_for_platform(platform_name, profile)
                platform_wheels[platform_name] = wheel
                print(f"✓ {platform_name}: {wheel}")
            except Exception as e:
                print(f"✗ {platform_name}: {e}")
                continue
        
        return platform_wheels
    
    def optimize_binary_size(self, wheel_path: Path) -> Path:
        """Optimize binary size of an existing wheel
        
        Args:
            wheel_path: Path to the wheel to optimize
            
        Returns:
            Path to the optimized wheel
        """
        print(f"Optimizing binary size for: {wheel_path}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Extract wheel
            extract_dir = tmpdir_path / "extracted"
            extract_dir.mkdir()
            
            subprocess.run([
                "python", "-m", "wheel", "unpack", 
                str(wheel_path), str(extract_dir)
            ], check=True)
            
            # Find extracted directory
            extracted_dirs = list(extract_dir.iterdir())
            if not extracted_dirs:
                raise RuntimeError("Failed to extract wheel")
            
            wheel_dir = extracted_dirs[0]
            
            # Find shared libraries
            so_files = list(wheel_dir.rglob("*.so"))
            dylib_files = list(wheel_dir.rglob("*.dylib"))
            pyd_files = list(wheel_dir.rglob("*.pyd"))
            
            binary_files = so_files + dylib_files + pyd_files
            
            # Strip binaries
            for binary_file in binary_files:
                print(f"Stripping: {binary_file}")
                try:
                    if platform.system() == "Linux":
                        subprocess.run(["strip", str(binary_file)], check=True)
                    elif platform.system() == "Darwin":
                        subprocess.run(["strip", "-x", str(binary_file)], check=True)
                    elif platform.system() == "Windows":
                        # Windows doesn't have strip, but binaries are usually smaller
                        pass
                except subprocess.CalledProcessError:
                    print(f"Failed to strip {binary_file}")
            
            # Remove debug info and unnecessary files
            debug_files = list(wheel_dir.rglob("*.debug"))
            for debug_file in debug_files:
                debug_file.unlink()
            
            # Remove __pycache__ directories
            pycache_dirs = list(wheel_dir.rglob("__pycache__"))
            for pycache_dir in pycache_dirs:
                shutil.rmtree(pycache_dir)
            
            # Remove .pyc files
            pyc_files = list(wheel_dir.rglob("*.pyc"))
            for pyc_file in pyc_files:
                pyc_file.unlink()
            
            # Repack wheel
            optimized_wheel = tmpdir_path / wheel_path.name.replace(".whl", "_optimized.whl")
            
            subprocess.run([
                "python", "-m", "wheel", "pack",
                str(wheel_dir), "--dest-dir", str(tmpdir_path)
            ], check=True)
            
            # Find the repacked wheel
            repacked_wheels = list(tmpdir_path.glob("*.whl"))
            repacked_wheels = [w for w in repacked_wheels if w != optimized_wheel]
            
            if repacked_wheels:
                repacked_wheel = repacked_wheels[0]
                repacked_wheel.rename(optimized_wheel)
            
            # Copy to output location
            output_dir = wheel_path.parent
            final_path = output_dir / optimized_wheel.name
            shutil.copy2(optimized_wheel, final_path)
            
            return final_path
    
    def create_platform_wheels(self) -> List[Path]:
        """Create wheels for multiple platforms
        
        Returns:
            List of paths to created wheels
        """
        wheels = []
        
        for platform_name, config in self.platform_configs.items():
            try:
                print(f"Building wheel for {platform_name}...")
                
                wheel = self.build_optimized_wheel(
                    profile="release",
                    features=config.get("features", []),
                    target=config.get("target")
                )
                
                # Optimize the wheel
                optimized_wheel = self.optimize_binary_size(wheel)
                wheels.append(optimized_wheel)
                
            except Exception as e:
                print(f"Failed to build wheel for {platform_name}: {e}")
        
        return wheels
    
    def create_source_distribution(self) -> Path:
        """Create an optimized source distribution
        
        Returns:
            Path to the source distribution
        """
        print("Creating source distribution...")
        
        # Use maturin to create sdist
        result = subprocess.run([
            "maturin", "sdist"
        ], cwd=self.package_dir, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create sdist: {result.stderr}")
        
        # Find the created sdist
        target_dir = self.package_dir / "target" / "wheels"
        sdists = list(target_dir.glob("*.tar.gz"))
        
        if not sdists:
            raise RuntimeError("No sdist found after build")
        
        sdist_path = max(sdists, key=lambda p: p.stat().st_mtime)
        print(f"Created sdist: {sdist_path}")
        
        return sdist_path
    
    def analyze_wheel_size(self, wheel_path: Path) -> Dict[str, int]:
        """Analyze the size breakdown of a wheel
        
        Args:
            wheel_path: Path to the wheel to analyze
            
        Returns:
            Dictionary with size breakdown
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Extract wheel
            extract_dir = tmpdir_path / "extracted"
            extract_dir.mkdir()
            
            subprocess.run([
                "python", "-m", "wheel", "unpack",
                str(wheel_path), str(extract_dir)
            ], check=True)
            
            wheel_dir = list(extract_dir.iterdir())[0]
            
            # Analyze file sizes
            sizes = {
                "total": 0,
                "binaries": 0,
                "python": 0,
                "metadata": 0,
                "other": 0
            }
            
            binary_extensions = {".so", ".dylib", ".pyd", ".dll"}
            python_extensions = {".py", ".pyc"}
            metadata_extensions = {".dist-info", ".egg-info"}
            
            for file_path in wheel_dir.rglob("*"):
                if file_path.is_file():
                    size = file_path.stat().st_size
                    sizes["total"] += size
                    
                    if file_path.suffix in binary_extensions:
                        sizes["binaries"] += size
                    elif file_path.suffix in python_extensions:
                        sizes["python"] += size
                    elif any(meta in str(file_path) for meta in metadata_extensions):
                        sizes["metadata"] += size
                    else:
                        sizes["other"] += size
            
            return sizes
    
    def generate_optimization_report(self, wheels: List[Path]) -> str:
        """Generate an optimization report
        
        Args:
            wheels: List of wheel paths to analyze
            
        Returns:
            Optimization report as string
        """
        report = ["# Package Optimization Report", ""]
        
        for wheel in wheels:
            report.append(f"## {wheel.name}")
            report.append("")
            
            # File size
            file_size = wheel.stat().st_size
            report.append(f"**Total Size:** {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
            
            # Size breakdown
            try:
                sizes = self.analyze_wheel_size(wheel)
                report.append(f"**Size Breakdown:**")
                report.append(f"- Binaries: {sizes['binaries']:,} bytes ({sizes['binaries']/sizes['total']*100:.1f}%)")
                report.append(f"- Python code: {sizes['python']:,} bytes ({sizes['python']/sizes['total']*100:.1f}%)")
                report.append(f"- Metadata: {sizes['metadata']:,} bytes ({sizes['metadata']/sizes['total']*100:.1f}%)")
                report.append(f"- Other: {sizes['other']:,} bytes ({sizes['other']/sizes['total']*100:.1f}%)")
            except Exception as e:
                report.append(f"**Size analysis failed:** {e}")
            
            report.append("")
        
        return "\n".join(report)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimize trustformers-py package")
    parser.add_argument("--profile", choices=["release", "size", "speed", "debug"], 
                       default="release", help="Build profile")
    parser.add_argument("--cpu-only", action="store_true", 
                       help="Create CPU-only build")
    parser.add_argument("--cuda", metavar="VERSION", 
                       help="Create CUDA build for specified version")
    parser.add_argument("--all-platforms", action="store_true",
                       help="Build for all supported platforms")
    parser.add_argument("--source-dist", action="store_true",
                       help="Create source distribution")
    parser.add_argument("--optimize", metavar="WHEEL_PATH",
                       help="Optimize existing wheel")
    parser.add_argument("--analyze", metavar="WHEEL_PATH",
                       help="Analyze wheel size")
    parser.add_argument("--output-dir", metavar="DIR",
                       help="Output directory for built wheels")
    
    args = parser.parse_args()
    
    # Initialize optimizer
    optimizer = PackageOptimizer()
    
    wheels = []
    
    try:
        if args.cpu_only:
            wheel = optimizer.create_cpu_only_build()
            wheels.append(wheel)
        
        if args.cuda:
            wheel = optimizer.create_cuda_build(args.cuda)
            wheels.append(wheel)
        
        if args.all_platforms:
            platform_wheels = optimizer.create_platform_wheels()
            wheels.extend(platform_wheels)
        
        if args.source_dist:
            sdist = optimizer.create_source_distribution()
            print(f"Created source distribution: {sdist}")
        
        if args.optimize:
            wheel_path = Path(args.optimize)
            optimized = optimizer.optimize_binary_size(wheel_path)
            wheels.append(optimized)
            print(f"Optimized wheel: {optimized}")
        
        if args.analyze:
            wheel_path = Path(args.analyze)
            sizes = optimizer.analyze_wheel_size(wheel_path)
            print(f"Size analysis for {wheel_path.name}:")
            for category, size in sizes.items():
                if category == "total":
                    print(f"  {category}: {size:,} bytes ({size / 1024 / 1024:.2f} MB)")
                else:
                    percentage = size / sizes["total"] * 100 if sizes["total"] > 0 else 0
                    print(f"  {category}: {size:,} bytes ({percentage:.1f}%)")
        
        # If no specific action, build standard wheel
        if not any([args.cpu_only, args.cuda, args.all_platforms, 
                   args.source_dist, args.optimize, args.analyze]):
            wheel = optimizer.build_optimized_wheel(profile=args.profile)
            wheels.append(wheel)
        
        # Generate report
        if wheels:
            report = optimizer.generate_optimization_report(wheels)
            print(f"\n{report}")
            
            # Save report
            report_path = Path("optimization_report.md")
            with open(report_path, "w") as f:
                f.write(report)
            print(f"\nOptimization report saved to: {report_path}")
        
        # Move wheels to output directory if specified
        if args.output_dir and wheels:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for wheel in wheels:
                dest = output_dir / wheel.name
                shutil.copy2(wheel, dest)
                print(f"Copied wheel to: {dest}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()