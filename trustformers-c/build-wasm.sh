#!/bin/bash
# WebAssembly build script for TrustformeRS-C

set -e

echo "Building TrustformeRS-C for WebAssembly..."

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo "Error: wasm-pack is not installed."
    echo "Please install it with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    exit 1
fi

# Check if target is installed
if ! rustup target list --installed | grep -q "wasm32-unknown-unknown"; then
    echo "Installing wasm32-unknown-unknown target..."
    rustup target add wasm32-unknown-unknown
fi

# Build configurations
BUILD_DIR="pkg"
FEATURES="wasm"

# Clean previous builds
rm -rf "$BUILD_DIR"

echo "Building for browser environment..."
wasm-pack build \
    --target web \
    --out-dir "${BUILD_DIR}/web" \
    --features "$FEATURES" \
    --release

echo "Building for Node.js environment..."
wasm-pack build \
    --target nodejs \
    --out-dir "${BUILD_DIR}/nodejs" \
    --features "$FEATURES" \
    --release

echo "Building for bundler environment..."
wasm-pack build \
    --target bundler \
    --out-dir "${BUILD_DIR}/bundler" \
    --features "$FEATURES" \
    --release

# Build with SIMD support if available
if rustc --print target-features --target wasm32-unknown-unknown | grep -q "simd128"; then
    echo "Building with SIMD support..."
    
    RUST_TARGET_PATH=$(pwd) RUSTFLAGS="-C target-feature=+simd128" \
    wasm-pack build \
        --target web \
        --out-dir "${BUILD_DIR}/web-simd" \
        --features "$FEATURES,wasm-simd" \
        --release
fi

# Build with threads support if available
echo "Building with threads support..."
RUSTFLAGS="-C target-feature=+atomics,+bulk-memory,+mutable-globals" \
wasm-pack build \
    --target web \
    --out-dir "${BUILD_DIR}/web-threads" \
    --features "$FEATURES,wasm-threads" \
    --release -- \
    -Z build-std=panic_abort,std

echo "Creating TypeScript declarations..."

# Create unified TypeScript declarations
cat > "${BUILD_DIR}/trustformers.d.ts" << 'EOF'
/* TrustformeRS WebAssembly TypeScript Declarations */

export interface WasmMemoryStats {
    memory_limit_mb: number;
    estimated_usage_mb: number;
    chunk_size: number;
    simd_enabled: boolean;
    threads_enabled: boolean;
}

export interface PlatformInfo {
    architecture: string;
    operating_system: string;
    cpu_features: string[];
    has_gpu: boolean;
    memory_mb: number;
    cpu_cores: number;
}

export class WasmTrustformers {
    constructor();
    
    static get_version(): string;
    
    has_simd(): boolean;
    has_threads(): boolean;
    
    tokenize(text: string, max_length: number): Uint32Array;
    
    set_memory_limit(limit_mb: number): void;
    get_memory_limit(): number;
    
    get_chunk_size(): number;
    set_chunk_size(size: number): void;
    
    free(): void;
}

// C API exports
export function trustformers_init(): number;
export function trustformers_cleanup(): number;
export function trustformers_version(): string;
export function trustformers_has_feature(feature: string): number;

export function trustformers_wasm_init(): number;
export function trustformers_wasm_has_simd(): number;
export function trustformers_wasm_has_threads(): number;

export function trustformers_wasm_matrix_multiply(
    optimizer: number,
    a: Float32Array,
    b: Float32Array,
    c: Float32Array,
    m: number,
    n: number,
    k: number
): number;

export function trustformers_wasm_tensor_add(
    optimizer: number,
    a: Float32Array,
    b: Float32Array,
    result: Float32Array,
    len: number
): number;

export function trustformers_get_platform_info(): PlatformInfo;
export function trustformers_get_memory_usage(): any;

// Memory management
export function trustformers_free_string(ptr: number): void;
export function trustformers_wasm_free(optimizer: number): void;
EOF

echo "Creating package.json for npm distribution..."

# Create package.json for each build target
for target in web nodejs bundler; do
    if [ -d "${BUILD_DIR}/${target}" ]; then
        cat > "${BUILD_DIR}/${target}/package.json" << EOF
{
  "name": "@trustformers/c-${target}",
  "version": "$(grep version Cargo.toml | head -1 | cut -d'"' -f2)",
  "description": "TrustformeRS C API WebAssembly bindings for ${target}",
  "main": "trustformers_c.js",
  "types": "trustformers_c.d.ts",
  "files": [
    "trustformers_c.js",
    "trustformers_c.d.ts",
    "trustformers_c_bg.wasm",
    "trustformers_c_bg.wasm.d.ts"
  ],
  "keywords": [
    "webassembly",
    "transformers",
    "nlp",
    "machine-learning",
    "wasm",
    "${target}"
  ],
  "author": "TrustformeRS Team",
  "license": "MIT",
  "repository": {
    "type": "git",
    "url": "https://github.com/cool-japan/trustformers.git"
  },
  "homepage": "https://github.com/cool-japan/trustformers",
  "engines": {
    "node": ">=14.0.0"
  }
}
EOF
    fi
done

echo "Creating example HTML file..."

cat > "${BUILD_DIR}/example.html" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>TrustformeRS WASM Example</title>
</head>
<body>
    <h1>TrustformeRS WebAssembly Example</h1>
    <div id="output"></div>
    
    <script type="module">
        import init, { WasmTrustformers } from './web/trustformers_c.js';
        
        async function run() {
            // Initialize the WASM module
            await init();
            
            const output = document.getElementById('output');
            
            try {
                // Create TrustformeRS instance
                const trustformers = new WasmTrustformers();
                
                // Display version and capabilities
                output.innerHTML += `<p>Version: ${WasmTrustformers.get_version()}</p>`;
                output.innerHTML += `<p>SIMD Support: ${trustformers.has_simd()}</p>`;
                output.innerHTML += `<p>Threads Support: ${trustformers.has_threads()}</p>`;
                output.innerHTML += `<p>Memory Limit: ${trustformers.get_memory_limit()}MB</p>`;
                output.innerHTML += `<p>Chunk Size: ${trustformers.get_chunk_size()}</p>`;
                
                // Test tokenization
                const text = "Hello, World!";
                const tokens = trustformers.tokenize(text, 100);
                output.innerHTML += `<p>Tokenized "${text}": [${Array.from(tokens).join(', ')}]</p>`;
                
                // Clean up
                trustformers.free();
                
                output.innerHTML += '<p><strong>✓ All tests passed!</strong></p>';
            } catch (error) {
                output.innerHTML += `<p><strong>✗ Error:</strong> ${error}</p>`;
            }
        }
        
        run();
    </script>
</body>
</html>
EOF

echo "Creating Node.js example..."

cat > "${BUILD_DIR}/example.js" << 'EOF'
const { WasmTrustformers } = require('./nodejs/trustformers_c.js');

async function main() {
    try {
        console.log('TrustformeRS WebAssembly Node.js Example');
        console.log('=====================================');
        
        // Create TrustformeRS instance
        const trustformers = new WasmTrustformers();
        
        // Display version and capabilities
        console.log(`Version: ${WasmTrustformers.get_version()}`);
        console.log(`SIMD Support: ${trustformers.has_simd()}`);
        console.log(`Threads Support: ${trustformers.has_threads()}`);
        console.log(`Memory Limit: ${trustformers.get_memory_limit()}MB`);
        console.log(`Chunk Size: ${trustformers.get_chunk_size()}`);
        
        // Test tokenization
        const text = "Hello, World!";
        const tokens = trustformers.tokenize(text, 100);
        console.log(`Tokenized "${text}": [${Array.from(tokens).join(', ')}]`);
        
        // Test memory configuration
        trustformers.set_memory_limit(256);
        console.log(`Updated memory limit: ${trustformers.get_memory_limit()}MB`);
        
        // Clean up
        trustformers.free();
        
        console.log('\n✓ All tests passed!');
    } catch (error) {
        console.error('✗ Error:', error);
        process.exit(1);
    }
}

main();
EOF

echo "Creating README for WASM builds..."

cat > "${BUILD_DIR}/README.md" << 'EOF'
# TrustformeRS WebAssembly Bindings

This directory contains WebAssembly builds of TrustformeRS-C for different environments.

## Available Builds

- `web/` - For use in web browsers with ES modules
- `nodejs/` - For use in Node.js applications
- `bundler/` - For use with webpack, rollup, etc.
- `web-simd/` - Browser build with SIMD optimizations (if supported)
- `web-threads/` - Browser build with SharedArrayBuffer/threads support

## Quick Start

### Web Browser

```html
<script type="module">
import init, { WasmTrustformers } from './web/trustformers_c.js';

async function run() {
    await init();
    const trustformers = new WasmTrustformers();
    console.log('Version:', WasmTrustformers.get_version());
    trustformers.free();
}

run();
</script>
```

### Node.js

```javascript
const { WasmTrustformers } = require('./nodejs/trustformers_c.js');

const trustformers = new WasmTrustformers();
console.log('Version:', WasmTrustformers.get_version());
trustformers.free();
```

### Webpack/Bundler

```javascript
import init, { WasmTrustformers } from '@trustformers/c-bundler';

await init();
const trustformers = new WasmTrustformers();
// Use trustformers...
trustformers.free();
```

## Features

- Matrix multiplication with SIMD optimizations
- Tensor operations
- Memory-efficient chunked processing
- Platform capability detection
- Performance monitoring
- Memory management utilities

## Browser Compatibility

- Chrome 74+ (WASM SIMD support)
- Firefox 72+ (WASM SIMD support)
- Safari 14+ (WASM SIMD support)
- Edge 79+ (WASM SIMD support)

For threads support, SharedArrayBuffer must be enabled (requires COOP/COEP headers).

## Performance Tips

1. Use SIMD builds when available for 2-4x performance improvement
2. Set appropriate memory limits for your use case
3. Use chunked operations for large datasets
4. Enable threads for parallel processing when supported

## Examples

See `example.html` for browser usage and `example.js` for Node.js usage.
EOF

echo ""
echo "✓ WebAssembly build completed successfully!"
echo ""
echo "Build outputs:"
echo "  - Web build: ${BUILD_DIR}/web/"
echo "  - Node.js build: ${BUILD_DIR}/nodejs/"
echo "  - Bundler build: ${BUILD_DIR}/bundler/"
if [ -d "${BUILD_DIR}/web-simd" ]; then
    echo "  - SIMD build: ${BUILD_DIR}/web-simd/"
fi
if [ -d "${BUILD_DIR}/web-threads" ]; then
    echo "  - Threads build: ${BUILD_DIR}/web-threads/"
fi
echo ""
echo "To test:"
echo "  - Browser: Open ${BUILD_DIR}/example.html in a web server"
echo "  - Node.js: cd ${BUILD_DIR} && node example.js"
echo ""
echo "To publish to npm:"
echo "  - cd ${BUILD_DIR}/web && npm publish"
echo "  - cd ${BUILD_DIR}/nodejs && npm publish"
echo "  - cd ${BUILD_DIR}/bundler && npm publish"