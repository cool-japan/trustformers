#!/usr/bin/env python3
"""
WASM Deployment Utilities for trustformers-py

This module provides utilities for building and deploying TrustformeRS 
to WebAssembly (WASM) for web-based machine learning applications.
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class WASMConfig:
    """Configuration for WASM builds"""
    target: str = "wasm32-wasi"
    profile: str = "size"
    features: List[str] = None
    optimization_level: str = "s"
    enable_simd: bool = False
    enable_bulk_memory: bool = True
    enable_reference_types: bool = True
    memory_size: int = 16  # MB
    stack_size: int = 1  # MB

class WASMDeployer:
    """WASM deployment and optimization utilities"""
    
    def __init__(self, package_dir: str = None):
        """Initialize WASM deployer
        
        Args:
            package_dir: Path to the package directory
        """
        self.package_dir = Path(package_dir or os.getcwd())
        self.wasm_dir = self.package_dir / "wasm"
        self.dist_dir = self.package_dir / "dist" / "wasm"
        
    def setup_wasm_environment(self, config: WASMConfig) -> Dict[str, str]:
        """Set up WASM build environment
        
        Args:
            config: WASM configuration
            
        Returns:
            Environment variables dictionary
        """
        env = os.environ.copy()
        
        # WASM-specific Rust flags
        rustflags = [
            "-C", f"opt-level={config.optimization_level}",
            "-C", "lto=fat",
            "-C", "panic=abort",
            "-C", "codegen-units=1",
            "-C", "link-arg=--export-dynamic",
            "-C", "link-arg=--allow-undefined",
        ]
        
        # Memory configuration
        rustflags.extend([
            "-C", f"link-arg=--initial-memory={config.memory_size * 1024 * 1024}",
            "-C", f"link-arg=--max-memory={config.memory_size * 1024 * 1024}",
            "-C", f"link-arg=-z stack-size={config.stack_size * 1024 * 1024}",
        ])
        
        # WASM features
        if config.enable_simd:
            rustflags.extend(["-C", "target-feature=+simd128"])
        
        if config.enable_bulk_memory:
            rustflags.extend(["-C", "target-feature=+bulk-memory"])
        
        if config.enable_reference_types:
            rustflags.extend(["-C", "target-feature=+reference-types"])
        
        # Size optimizations
        rustflags.extend([
            "-C", "link-arg=--gc-sections",
            "-C", "link-arg=--strip-all",
        ])
        
        env["RUSTFLAGS"] = " ".join(rustflags)
        
        # Cargo configuration
        env["CARGO_PROFILE_RELEASE_OPT_LEVEL"] = config.optimization_level
        env["CARGO_PROFILE_RELEASE_LTO"] = "fat"
        env["CARGO_PROFILE_RELEASE_CODEGEN_UNITS"] = "1"
        env["CARGO_PROFILE_RELEASE_PANIC"] = "abort"
        
        return env
    
    def create_wasm_config_file(self, config: WASMConfig) -> Path:
        """Create WASM-specific configuration file
        
        Args:
            config: WASM configuration
            
        Returns:
            Path to the configuration file
        """
        self.wasm_dir.mkdir(exist_ok=True)
        config_path = self.wasm_dir / "wasm-config.json"
        
        config_data = {
            "target": config.target,
            "profile": config.profile,
            "features": config.features or ["wasm", "no-std"],
            "optimization": {
                "level": config.optimization_level,
                "simd": config.enable_simd,
                "bulk_memory": config.enable_bulk_memory,
                "reference_types": config.enable_reference_types
            },
            "memory": {
                "size_mb": config.memory_size,
                "stack_mb": config.stack_size
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        return config_path
    
    def build_wasm_module(self, config: WASMConfig = None) -> Path:
        """Build WASM module
        
        Args:
            config: WASM configuration
            
        Returns:
            Path to the built WASM file
        """
        config = config or WASMConfig()
        
        print(f"Building WASM module with target: {config.target}")
        
        # Set up environment
        env = self.setup_wasm_environment(config)
        
        # Create config file
        self.create_wasm_config_file(config)
        
        # Build command for WASM
        cmd = [
            "cargo", "build",
            "--target", config.target,
            "--release",
            "--lib"
        ]
        
        if config.features:
            cmd.extend(["--features", ",".join(config.features)])
        
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
            print(f"WASM build failed: {result.stderr}")
            raise RuntimeError(f"WASM build failed with exit code {result.returncode}")
        
        # Find the built WASM file
        target_dir = self.package_dir / "target" / config.target / "release"
        wasm_files = list(target_dir.glob("*.wasm"))
        
        if not wasm_files:
            # Try finding the library file
            lib_files = list(target_dir.glob("lib*.wasm"))
            if lib_files:
                wasm_files = lib_files
            else:
                raise RuntimeError("No WASM file found after build")
        
        wasm_path = wasm_files[0]
        print(f"Built WASM module: {wasm_path}")
        
        return wasm_path
    
    def optimize_wasm_module(self, wasm_path: Path) -> Path:
        """Optimize WASM module using wasm-opt
        
        Args:
            wasm_path: Path to the WASM file
            
        Returns:
            Path to the optimized WASM file
        """
        print(f"Optimizing WASM module: {wasm_path}")
        
        optimized_path = wasm_path.with_suffix(".opt.wasm")
        
        # Check if wasm-opt is available
        try:
            subprocess.run(["wasm-opt", "--version"], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: wasm-opt not found. Skipping optimization.")
            print("Install Binaryen toolkit for WASM optimization.")
            return wasm_path
        
        # Optimize with wasm-opt
        cmd = [
            "wasm-opt",
            "-Os",  # Optimize for size
            "--enable-bulk-memory",
            "--enable-reference-types",
            "--enable-simd",
            "--vacuum",
            "--dce",
            "--remove-unused-functions",
            "--remove-unused-module-elements", 
            "--strip-debug",
            "-o", str(optimized_path),
            str(wasm_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"WASM optimization failed: {result.stderr}")
            return wasm_path
        
        print(f"Optimized WASM module: {optimized_path}")
        return optimized_path
    
    def create_web_package(self, wasm_path: Path) -> Path:
        """Create web deployment package
        
        Args:
            wasm_path: Path to the WASM file
            
        Returns:
            Path to the web package directory
        """
        print("Creating web deployment package...")
        
        # Create distribution directory
        self.dist_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy WASM file
        wasm_dist_path = self.dist_dir / "trustformers.wasm"
        shutil.copy2(wasm_path, wasm_dist_path)
        
        # Create JavaScript bindings
        js_bindings = self._create_js_bindings()
        js_path = self.dist_dir / "trustformers.js"
        with open(js_path, 'w') as f:
            f.write(js_bindings)
        
        # Create TypeScript definitions
        ts_definitions = self._create_ts_definitions()
        ts_path = self.dist_dir / "trustformers.d.ts"
        with open(ts_path, 'w') as f:
            f.write(ts_definitions)
        
        # Create package.json
        package_json = self._create_package_json()
        package_path = self.dist_dir / "package.json"
        with open(package_path, 'w') as f:
            json.dump(package_json, f, indent=2)
        
        # Create example HTML
        html_example = self._create_html_example()
        html_path = self.dist_dir / "example.html"
        with open(html_path, 'w') as f:
            f.write(html_example)
        
        # Create README
        readme_content = self._create_readme()
        readme_path = self.dist_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        
        print(f"Web package created: {self.dist_dir}")
        return self.dist_dir
    
    def _create_js_bindings(self) -> str:
        """Create JavaScript bindings for WASM module"""
        return '''
/**
 * TrustformeRS WASM JavaScript Bindings
 * High-performance transformer library for the web
 */

class TrustformersWASM {
    constructor() {
        this.module = null;
        this.initialized = false;
    }

    async initialize(wasmPath = './trustformers.wasm') {
        if (this.initialized) {
            return;
        }

        try {
            // Fetch and instantiate WASM module
            const wasmModule = await WebAssembly.instantiateStreaming(
                fetch(wasmPath),
                {
                    env: {
                        memory: new WebAssembly.Memory({ initial: 16, maximum: 16 }),
                        abort: () => { throw new Error('WASM abort'); },
                        console_log: (ptr, len) => {
                            // Simple console.log implementation
                            const memory = new Uint8Array(this.module.instance.exports.memory.buffer);
                            const message = new TextDecoder().decode(memory.slice(ptr, ptr + len));
                            console.log(message);
                        }
                    }
                }
            );

            this.module = wasmModule;
            this.initialized = true;
            console.log('TrustformeRS WASM initialized successfully');
        } catch (error) {
            console.error('Failed to initialize TrustformeRS WASM:', error);
            throw error;
        }
    }

    createTensor(data, shape) {
        if (!this.initialized) {
            throw new Error('WASM module not initialized. Call initialize() first.');
        }

        // Call WASM function to create tensor
        // This is a placeholder - actual implementation depends on exported functions
        return this.module.instance.exports.create_tensor(data, shape);
    }

    loadModel(modelData) {
        if (!this.initialized) {
            throw new Error('WASM module not initialized. Call initialize() first.');
        }

        // Load model in WASM
        return this.module.instance.exports.load_model(modelData);
    }

    predict(modelId, inputTensor) {
        if (!this.initialized) {
            throw new Error('WASM module not initialized. Call initialize() first.');
        }

        // Run inference
        return this.module.instance.exports.predict(modelId, inputTensor);
    }

    tokenize(text, tokenizerConfig) {
        if (!this.initialized) {
            throw new Error('WASM module not initialized. Call initialize() first.');
        }

        // Tokenize text
        return this.module.instance.exports.tokenize(text, tokenizerConfig);
    }

    getMemoryUsage() {
        if (!this.initialized) {
            return { used: 0, available: 0 };
        }

        // Get memory statistics from WASM
        const used = this.module.instance.exports.get_memory_used();
        const available = this.module.instance.exports.get_memory_available();
        
        return { used, available };
    }

    cleanup() {
        if (this.module && this.module.instance.exports.cleanup) {
            this.module.instance.exports.cleanup();
        }
        this.initialized = false;
        this.module = null;
    }
}

// Export for both Node.js and browser
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TrustformersWASM;
} else {
    window.TrustformersWASM = TrustformersWASM;
}
'''
    
    def _create_ts_definitions(self) -> str:
        """Create TypeScript definitions"""
        return '''
/**
 * TypeScript definitions for TrustformeRS WASM
 */

export interface TensorData {
    data: number[] | Float32Array | Float64Array;
    shape: number[];
    dtype?: 'float32' | 'float64' | 'int32' | 'int64';
}

export interface ModelConfig {
    modelType: 'bert' | 'gpt2' | 't5' | 'llama';
    vocabSize: number;
    hiddenSize: number;
    numLayers: number;
    numAttentionHeads: number;
}

export interface TokenizerConfig {
    vocabSize: number;
    unknownToken: string;
    padToken?: string;
    bosToken?: string;
    eosToken?: string;
}

export interface MemoryUsage {
    used: number;
    available: number;
}

export declare class TrustformersWASM {
    constructor();
    
    /**
     * Initialize the WASM module
     * @param wasmPath Path to the WASM file
     */
    initialize(wasmPath?: string): Promise<void>;
    
    /**
     * Create a tensor from data
     * @param data Tensor data
     * @param shape Tensor shape
     */
    createTensor(data: number[] | TypedArray, shape: number[]): number;
    
    /**
     * Load a model
     * @param modelData Model configuration or binary data
     */
    loadModel(modelData: ModelConfig | ArrayBuffer): number;
    
    /**
     * Run inference
     * @param modelId Model identifier
     * @param inputTensor Input tensor
     */
    predict(modelId: number, inputTensor: number): number;
    
    /**
     * Tokenize text
     * @param text Input text
     * @param tokenizerConfig Tokenizer configuration
     */
    tokenize(text: string, tokenizerConfig: TokenizerConfig): number[];
    
    /**
     * Get memory usage statistics
     */
    getMemoryUsage(): MemoryUsage;
    
    /**
     * Clean up resources
     */
    cleanup(): void;
}

export default TrustformersWASM;
'''
    
    def _create_package_json(self) -> Dict[str, Any]:
        """Create package.json for npm distribution"""
        return {
            "name": "@trustformers/wasm",
            "version": "0.1.0",
            "description": "TrustformeRS WebAssembly bindings for high-performance transformer models in the browser",
            "main": "trustformers.js",
            "types": "trustformers.d.ts",
            "files": [
                "trustformers.wasm",
                "trustformers.js",
                "trustformers.d.ts",
                "README.md"
            ],
            "keywords": [
                "webassembly",
                "wasm",
                "transformers",
                "machine-learning",
                "nlp",
                "bert",
                "gpt2",
                "browser",
                "rust"
            ],
            "author": "TrustformeRS Team",
            "license": "Apache-2.0",
            "repository": {
                "type": "git",
                "url": "https://github.com/cool-japan/trustformers"
            },
            "engines": {
                "node": ">=14.0.0"
            },
            "scripts": {
                "test": "echo \\"No tests yet\\"",
                "example": "python -m http.server 8000"
            },
            "devDependencies": {
                "@types/node": "^18.0.0",
                "typescript": "^4.9.0"
            }
        }
    
    def _create_html_example(self) -> str:
        """Create HTML example"""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustformeRS WASM Example</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { margin: 20px 0; }
        textarea { width: 100%; height: 100px; margin: 10px 0; }
        button { padding: 10px 20px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        .output { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; white-space: pre-wrap; }
        .error { background: #f8d7da; color: #721c24; }
        .success { background: #d4edda; color: #155724; }
    </style>
</head>
<body>
    <h1>TrustformeRS WASM Example</h1>
    
    <div class="container">
        <h2>Initialize WASM Module</h2>
        <button id="initBtn">Initialize TrustformeRS</button>
        <div id="initStatus" class="output"></div>
    </div>
    
    <div class="container">
        <h2>Text Processing</h2>
        <textarea id="textInput" placeholder="Enter text to process...">Hello, this is a test sentence for transformer processing.</textarea>
        <br>
        <button id="tokenizeBtn" disabled>Tokenize Text</button>
        <button id="processBtn" disabled>Process Text</button>
        <div id="textOutput" class="output"></div>
    </div>
    
    <div class="container">
        <h2>Tensor Operations</h2>
        <button id="tensorBtn" disabled>Create Sample Tensor</button>
        <div id="tensorOutput" class="output"></div>
    </div>
    
    <div class="container">
        <h2>Memory Usage</h2>
        <button id="memoryBtn" disabled>Check Memory Usage</button>
        <div id="memoryOutput" class="output"></div>
    </div>

    <script src="trustformers.js"></script>
    <script>
        let trustformers = null;

        document.getElementById('initBtn').addEventListener('click', async () => {
            const statusEl = document.getElementById('initStatus');
            const initBtn = document.getElementById('initBtn');
            
            try {
                statusEl.textContent = 'Initializing WASM module...';
                statusEl.className = 'output';
                initBtn.disabled = true;
                
                trustformers = new TrustformersWASM();
                await trustformers.initialize('./trustformers.wasm');
                
                statusEl.textContent = 'TrustformeRS WASM initialized successfully!';
                statusEl.className = 'output success';
                
                // Enable other buttons
                document.getElementById('tokenizeBtn').disabled = false;
                document.getElementById('processBtn').disabled = false;
                document.getElementById('tensorBtn').disabled = false;
                document.getElementById('memoryBtn').disabled = false;
                
            } catch (error) {
                statusEl.textContent = `Initialization failed: ${error.message}`;
                statusEl.className = 'output error';
                initBtn.disabled = false;
            }
        });

        document.getElementById('tokenizeBtn').addEventListener('click', () => {
            const text = document.getElementById('textInput').value;
            const outputEl = document.getElementById('textOutput');
            
            try {
                // This is a placeholder - actual implementation depends on exported functions
                outputEl.textContent = `Tokenizing: "${text}"\\n\\nNote: This is a demo. Actual tokenization would call WASM functions.`;
                outputEl.className = 'output success';
            } catch (error) {
                outputEl.textContent = `Tokenization failed: ${error.message}`;
                outputEl.className = 'output error';
            }
        });

        document.getElementById('processBtn').addEventListener('click', () => {
            const text = document.getElementById('textInput').value;
            const outputEl = document.getElementById('textOutput');
            
            try {
                // This is a placeholder - actual implementation depends on exported functions
                outputEl.textContent = `Processing: "${text}"\\n\\nNote: This is a demo. Actual processing would run transformer inference.`;
                outputEl.className = 'output success';
            } catch (error) {
                outputEl.textContent = `Processing failed: ${error.message}`;
                outputEl.className = 'output error';
            }
        });

        document.getElementById('tensorBtn').addEventListener('click', () => {
            const outputEl = document.getElementById('tensorOutput');
            
            try {
                // Create sample tensor data
                const data = [1, 2, 3, 4, 5, 6];
                const shape = [2, 3];
                
                // This is a placeholder - actual implementation would call WASM
                outputEl.textContent = `Created tensor:\\nData: [${data.join(', ')}]\\nShape: [${shape.join(', ')}]\\n\\nNote: This is a demo. Actual tensor creation would call WASM functions.`;
                outputEl.className = 'output success';
            } catch (error) {
                outputEl.textContent = `Tensor creation failed: ${error.message}`;
                outputEl.className = 'output error';
            }
        });

        document.getElementById('memoryBtn').addEventListener('click', () => {
            const outputEl = document.getElementById('memoryOutput');
            
            try {
                if (trustformers) {
                    const memory = trustformers.getMemoryUsage();
                    outputEl.textContent = `Memory Usage:\\nUsed: ${memory.used} bytes\\nAvailable: ${memory.available} bytes`;
                    outputEl.className = 'output success';
                } else {
                    outputEl.textContent = 'WASM module not initialized';
                    outputEl.className = 'output error';
                }
            } catch (error) {
                outputEl.textContent = `Memory check failed: ${error.message}`;
                outputEl.className = 'output error';
            }
        });

        // Cleanup on page unload
        window.addEventListener('beforeunload', () => {
            if (trustformers) {
                trustformers.cleanup();
            }
        });
    </script>
</body>
</html>
'''
    
    def _create_readme(self) -> str:
        """Create README for WASM package"""
        return '''
# TrustformeRS WebAssembly

High-performance transformer models running in the browser via WebAssembly.

## Features

- **Fast**: Near-native performance with Rust + WASM
- **Lightweight**: Optimized for web deployment
- **Complete**: BERT, GPT-2, T5, and LLaMA support
- **Compatible**: Works in all modern browsers

## Quick Start

### Installation

```bash
npm install @trustformers/wasm
```

### Usage

```javascript
import TrustformersWASM from '@trustformers/wasm';

// Initialize the WASM module
const trustformers = new TrustformersWASM();
await trustformers.initialize();

// Create a tensor
const tensor = trustformers.createTensor([1, 2, 3, 4], [2, 2]);

// Load a model (placeholder)
const modelConfig = {
    modelType: 'bert',
    vocabSize: 30522,
    hiddenSize: 768,
    numLayers: 12,
    numAttentionHeads: 12
};
const modelId = trustformers.loadModel(modelConfig);

// Run inference (placeholder)
const output = trustformers.predict(modelId, tensor);

// Cleanup
trustformers.cleanup();
```

### Browser Example

```html
<!DOCTYPE html>
<html>
<head>
    <script src="trustformers.js"></script>
</head>
<body>
    <script>
        async function runExample() {
            const trustformers = new TrustformersWASM();
            await trustformers.initialize('./trustformers.wasm');
            
            // Your code here
            
            trustformers.cleanup();
        }
        
        runExample();
    </script>
</body>
</html>
```

## Development

To build from source:

```bash
# Install Rust and wasm-pack
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
cargo install wasm-pack

# Install the WASM target
rustup target add wasm32-wasi

# Build WASM module
python wasm_deployment.py --build

# Serve example
python -m http.server 8000
# Open http://localhost:8000/example.html
```

## API Reference

### TrustformersWASM

Main class for interacting with the WASM module.

#### Methods

- `initialize(wasmPath?: string): Promise<void>` - Initialize the WASM module
- `createTensor(data: number[], shape: number[]): number` - Create a tensor
- `loadModel(config: ModelConfig): number` - Load a model
- `predict(modelId: number, input: number): number` - Run inference
- `tokenize(text: string, config: TokenizerConfig): number[]` - Tokenize text
- `getMemoryUsage(): MemoryUsage` - Get memory statistics
- `cleanup(): void` - Clean up resources

## Performance Tips

1. **Memory Management**: Call `cleanup()` when done to free memory
2. **Batch Processing**: Process multiple inputs together when possible
3. **Model Caching**: Reuse loaded models instead of reloading
4. **SIMD**: Enable SIMD features for better performance

## Browser Compatibility

- Chrome 57+
- Firefox 52+
- Safari 11+
- Edge 16+

## License

Apache-2.0

## Contributing

See the main TrustformeRS repository for contribution guidelines.
'''

def main():
    """Main entry point for WASM deployment"""
    import argparse
    
    parser = argparse.ArgumentParser(description="WASM Deployment Tool")
    parser.add_argument("--build", action="store_true", help="Build WASM module")
    parser.add_argument("--optimize", action="store_true", help="Optimize WASM module")
    parser.add_argument("--package", action="store_true", help="Create web package")
    parser.add_argument("--all", action="store_true", help="Build, optimize, and package")
    parser.add_argument("--config", help="WASM config file")
    parser.add_argument("--memory-size", type=int, default=16, help="Memory size in MB")
    parser.add_argument("--enable-simd", action="store_true", help="Enable SIMD")
    
    args = parser.parse_args()
    
    deployer = WASMDeployer()
    
    # Create configuration
    config = WASMConfig(
        memory_size=args.memory_size,
        enable_simd=args.enable_simd
    )
    
    wasm_path = None
    
    if args.build or args.all:
        print("Building WASM module...")
        wasm_path = deployer.build_wasm_module(config)
    
    if args.optimize or args.all:
        if not wasm_path:
            # Find existing WASM file
            target_dir = deployer.package_dir / "target" / config.target / "release"
            wasm_files = list(target_dir.glob("*.wasm"))
            if wasm_files:
                wasm_path = wasm_files[0]
            else:
                print("No WASM file found. Run with --build first.")
                return
        
        print("Optimizing WASM module...")
        wasm_path = deployer.optimize_wasm_module(wasm_path)
    
    if args.package or args.all:
        if not wasm_path:
            print("No WASM file found. Run with --build first.")
            return
        
        print("Creating web package...")
        package_dir = deployer.create_web_package(wasm_path)
        print(f"Web package created at: {package_dir}")
        print("Run 'python -m http.server 8000' in the package directory to test")

if __name__ == "__main__":
    main()