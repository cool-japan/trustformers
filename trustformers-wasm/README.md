# TrustformeRS WebAssembly

WebAssembly bindings for the TrustformeRS transformer library, enabling transformer models to run directly in web browsers and Node.js environments with full WebGPU hardware acceleration.

**Version:** 0.1.1 | **Status:** Stable | **Tests:** 128 | **SLoC:** 55,504 | **Last Updated:** 2026-04-25

## Features

- **WebGPU Backend**: 50-100x speedup over CPU via GPU compute shaders (wgpu 29.0 API)
- **Web Workers Parallelism**: Multi-threaded inference via SharedArrayBuffer
- **IndexedDB Caching**: Persistent model and KV-cache storage in the browser
- **BERT WASM Model**: Complete BERT implementation running in-browser
- **React/Vue/Angular/Web Components**: First-class framework bindings
- **Streaming Inference**: Token-by-token generation with streaming API
- **SIMD Support**: Hardware-accelerated tensor ops where available
- **Mobile Optimization**: Battery-aware, network-adaptive loading
- **SciRS2 Integration**: scirs2-core tensor operations in WASM

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
    console.log('Version:', tf.version);  // "0.1.1"

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
console.log(tf.version);     // "0.1.1"
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
BERT model running entirely in WASM.

```javascript
const config = BertConfig.tiny();
const model = new BertModelWasm(config);
const output = model.forward(input_ids, attention_mask);
```

### WebGPU Backend

```javascript
import { WebGpuInference, StreamingGenerator } from './pkg-web/trustformers_wasm.js';

// Initialize WebGPU (50-100x speedup vs CPU)
const inference = await WebGpuInference.new();

// Streaming token generation
const generator = new StreamingGenerator(inference, model_id);
for await (const token of generator.stream(prompt)) {
    process.stdout.write(token);
}
```

### Framework Bindings

#### React

```jsx
import { useTrustformers, TrustformersProvider } from 'trustformers-react';

function App() {
    const { model, generate, isLoading } = useTrustformers('bert-base');
    return (
        <TrustformersProvider>
            <InferenceComponent model={model} onGenerate={generate} />
        </TrustformersProvider>
    );
}
```

#### Vue

```javascript
import { useTrustformers } from 'trustformers-vue';

export default {
    setup() {
        const { model, tokenizer, generate } = useTrustformers('bert-base');
        return { model, generate };
    }
}
```

#### Angular

```typescript
import { TrustformersService } from 'trustformers-angular';

@Injectable({ providedIn: 'root' })
export class AppComponent {
    constructor(private tf: TrustformersService) {}

    async generate(prompt: string) {
        return this.tf.generate(prompt).pipe(toArray()).toPromise();
    }
}
```

#### Web Components

```html
<trustformers-inference-engine model="bert-base"></trustformers-inference-engine>
<trustformers-model-loader src="./models/bert.bin"></trustformers-model-loader>
<trustformers-performance-monitor></trustformers-performance-monitor>
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

## Feature Flags

- `webgpu` — WebGPU compute shader backend (wgpu 29.0)
- `web-workers` — Web Workers parallelism (SharedArrayBuffer)
- `shared-memory` — Shared memory for multi-threaded WASM
- `kernel-fusion` — Fused transformer kernels (MHA, FFN, LayerNorm+Residual)
- `async-executor` — Async Rust executor for WASM
- `indexeddb` — IndexedDB model and KV-cache persistence
- `memory64` — WASM memory64 for models >4GB
- `streaming-loader` — Progressive chunked model loading
- `react-components` — React hooks and component library
- `vue-components` — Vue composables and plugin
- `angular-components` — Angular services and directives
- `web-components` — Framework-agnostic custom elements
- `playground` — Interactive browser playground
- `streaming-generation` — Token-by-token streaming inference
- `mobile-optimization` — Battery/network-adaptive loading
- `scirs2` — SciRS2-core tensor operations

## WebGPU Notes (wgpu 29.0)

This crate targets wgpu 29.0 with the following API specifics:

- `InstanceDescriptor::new_without_display_handle()` — headless instance creation
- `bind_group_layouts` accepts `&[Option<&BindGroupLayout>]` for sparse layouts
- Kernel fusion enabled for MHA (2.5x), FFN (1.8x), LayerNorm+Residual (1.5x)

## Examples

See the `examples/` directory for complete examples:

- `index.html` / `playground.html` — Interactive browser demo
- `demo/` — Full-featured playground application
- Node.js example in `examples/`

## Performance Tips

1. **Enable WebGPU**: Use Chrome 113+ / Edge 113+ for 50-100x speedup
2. **Enable SIMD**: Compile with WASM SIMD128 target feature
3. **Batch operations**: Process multiple inputs together
4. **Use IndexedDB caching**: Avoid re-downloading models between sessions
5. **Enable kernel fusion**: `webgpu` + `kernel-fusion` features
6. **Reuse tensors**: Minimize allocations in hot loops

## Testing

```bash
# Run WASM tests
wasm-pack test --headless --firefox --chrome

# Run with specific features
cargo test --target wasm32-unknown-unknown --features webgpu

# Check compilation
cargo check --target wasm32-unknown-unknown
```

128 unit tests with 100% pass rate, covering:
- Core tensor operations
- WebGPU backend (mock device)
- BERT forward pass
- Framework binding contracts
- Streaming generation
- IndexedDB model cache

## Limitations

- WebGPU requires Chrome 113+, Edge 113+, or Safari (experimental)
- SharedArrayBuffer requires cross-origin isolation headers
- SIMD requires WASM SIMD128 browser support
- Memory typically capped at 2-4GB (use `memory64` + quantization for large models)

## License

Apache-2.0
