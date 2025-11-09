# TrustformeRS WebAssembly

WebAssembly bindings for the TrustformeRS transformer library, enabling transformer models to run directly in web browsers and Node.js environments.

## Features

- 🚀 **No-std compatible**: Runs in constrained WebAssembly environments
- 🧠 **Transformer models**: BERT and other architectures in the browser
- ⚡ **SIMD support**: Hardware acceleration when available
- 🔧 **Multiple targets**: Web, Node.js, and bundler support
- 📦 **Small size**: Optimized for web deployment

## Building

### Prerequisites

- Rust (latest stable)
- wasm-pack (`curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh`)

### Build Commands

```bash
# Build for all targets
./build.sh

# Or build individually:
wasm-pack build --target web --out-dir pkg-web
wasm-pack build --target bundler --out-dir pkg-bundler
wasm-pack build --target nodejs --out-dir pkg-node
```

## Usage

### Browser (Direct)

```html
<script type="module">
import init, { TrustformersWasm, WasmTensor } from './pkg-web/trustformers_wasm.js';

async function run() {
    await init();
    
    const tf = new TrustformersWasm();
    console.log('Version:', tf.version);
    
    // Create and manipulate tensors
    const tensor = WasmTensor.new([1, 2, 3, 4], [2, 2]);
    const result = tensor.add(tensor);
    console.log('Result:', result.data);
}

run();
</script>
```

### Node.js

```javascript
const { TrustformersWasm, WasmTensor } = require('./pkg-node/trustformers_wasm.js');

const tf = new TrustformersWasm();
const tensor = WasmTensor.new([1, 2, 3, 4], [2, 2]);
console.log(tensor.toString());
```

### Webpack/Bundler

```javascript
import * as wasm from './pkg-bundler/trustformers_wasm';

async function run() {
    await wasm.default();
    
    const tf = new wasm.TrustformersWasm();
    // Use the library...
}
```

## API Overview

### Core Classes

#### `TrustformersWasm`
Main entry point for the library.

```javascript
const tf = new TrustformersWasm();
console.log(tf.version);     // "0.1.0-alpha.1"
console.log(tf.initialized); // true
```

#### `WasmTensor`
Core tensor operations.

```javascript
// Creation
const a = WasmTensor.new([1, 2, 3, 4], [2, 2]);
const b = WasmTensor.zeros([3, 3]);
const c = WasmTensor.ones([2, 4]);
const d = WasmTensor.randn([5, 5]);

// Operations
const sum = a.add(b);
const prod = a.matmul(b);
const transposed = a.transpose();

// Activations
const relu_out = a.relu();
const gelu_out = a.gelu();
const softmax_out = a.softmax(-1);
```

#### `Linear`
Fully connected layer.

```javascript
const linear = new Linear(input_size, output_size, use_bias);
const output = linear.forward(input_tensor);
```

#### `BertModelWasm`
Tiny BERT model for demonstrations.

```javascript
const config = BertConfig.tiny();
const model = new BertModelWasm(config);
const output = model.forward(input_ids, attention_mask);
```

### Utilities

```javascript
// Performance measurement
const timer = new Timer("My Operation");
// ... do work ...
console.log(`Elapsed: ${timer.elapsed()}ms`);

// Memory statistics
const stats = get_memory_stats();
console.log(`Memory used: ${stats.used_mb} MB`);

// Feature detection
console.log(`SIMD enabled: ${enable_simd()}`);
console.log(`Features: ${features()}`);
```

## Examples

See the `examples/` directory for complete examples:

- `index.html` - Interactive browser demo
- `node-example.js` - Node.js usage example

## Performance Tips

1. **Enable SIMD**: Use Chrome/Edge with experimental SIMD support
2. **Batch operations**: Process multiple inputs together
3. **Reuse tensors**: Minimize allocations in hot loops
4. **Use appropriate types**: f32 is faster than f64

## Limitations

- No file I/O (use fetch/XHR for model loading)
- Limited parallelism (WebAssembly threading is experimental)
- Memory constraints (typically 2-4GB max)
- No GPU support (WebGPU integration planned)

## Future Work

- WebGPU backend for GPU acceleration
- More model architectures
- Quantization support
- Model loading from various formats
- Tokenizer integration

## License

MIT OR Apache-2.0